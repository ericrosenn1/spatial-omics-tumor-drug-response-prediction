"""
Script:
    09_N_train_filtered_per_treatment_residual_models.py

Purpose:
    Train filtered per treatment residual models.

Model:
    For each eligible treatment:
        fused_residual_vs_prior = f(spatial biology features only)

Design:
    New experimental script.
    Does not overwrite canonical scripts.
    Uses no drug dummies because treatment identity is fixed within each model.
    Filters treatments by sample count, residual variance, residual range, and target uniqueness.
    Uses repeated train test splits for model screening.
    Writes final XGBoost and SHAP models for selected treatments.

Text report convention:
    Every generated .txt report starts with its own filepath.
"""

from pathlib import Path
import argparse
import json
import hashlib
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr

import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
except Exception as exc:
    raise ImportError("xgboost is required") from exc

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-treatments", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-target-std", type=float, default=0.02)
    parser.add_argument("--min-target-range", type=float, default=0.08)
    parser.add_argument("--min-unique-targets", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-features-per-model", type=int, default=40)
    parser.add_argument("--min-feature-variance", type=float, default=1e-12)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-final-models", type=int, default=80)
    parser.add_argument("--max-shap-treatments", type=int, default=80)
    parser.add_argument("--min-promising-pearson", type=float, default=0.20)
    parser.add_argument("--min-positive-fraction", type=float, default=0.60)
    parser.add_argument("--min-rmse-improvement", type=float, default=0.0)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_any(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def write_table(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def write_text_report(path, lines):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join([f"FILEPATH: {path}"] + list(lines)), encoding="utf-8")


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def savefig(path):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def safe_name(value, max_len=120):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "unnamed"
    return value[:max_len]


def short_label(value, width=55):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + "..."


def latest_dir(base, pattern):
    base = Path(base)
    hits = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def q025(x):
    return float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.025))


def q975(x):
    return float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.975))


def metric_safe(y_true, y_pred):
    out = {
        "n": int(len(y_true)),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    out["n"] = int(len(y_true))

    if len(y_true) == 0:
        return out

    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)

    if len(y_true) >= 3 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        out["r2"] = float(r2_score(y_true, y_pred))
        try:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pass
        try:
            out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            pass

    return out


def infer_theme_from_text(text):
    text = str(text).lower()

    if "tryptophan" in text or "kynurenine" in text:
        return "tryptophan kynurenine immune suppression"
    if "myeloid" in text or "macrophage" in text:
        return "myeloid macrophage tumor ecology"
    if "hypoxi" in text:
        return "hypoxia immune stress context"
    if "t_cell" in text or "interferon" in text or "immune" in text or "b_plasma" in text:
        return "immune inflammation and t cell organization"
    if "access" in text or "boundary" in text or "penetration" in text:
        return "tumor access and boundary penetration"
    if "stromal" in text or "stroma" in text or "ecm" in text or "collagen" in text or "fibroblast" in text:
        return "stromal ecm barrier architecture"
    if "tumor_proliferative" in text or "proliferation" in text or "cell_cycle" in text:
        return "tumor proliferation state"
    if "vascular" in text or "angiogenic" in text or "endothelial" in text:
        return "vascular angiogenic context"
    if "fatty_acid" in text:
        return "fatty acid metabolism"
    if "glycolysis" in text or "oxphos" in text or "oxidative" in text or "glutamine" in text or "metabolic" in text:
        return "metabolic spatial context"
    if "pair_" in text or "centroid_distance" in text or "overlap" in text:
        return "pairwise spatial relationship"
    return "other interpretable spatial signal"


def artifact_or_caution(text):
    text = str(text).lower()
    patterns = [
        "filtering__",
        "spatial_x_",
        "spatial_y_",
        "array_row",
        "array_col",
        "total_counts",
        "pct_counts_mt",
        "n_genes_by_counts",
        "genes_after",
        "barcode",
        "n_spots",
        "n_spots_scored",
        "method_structure_agreement",
        "structure_region_consensus_fraction",
        "metabolic_best_matching_state_score",
        "access_tumor_spots",
        "largest_component_spots",
    ]
    return any(p in text for p in patterns)


def normalize_manifest(manifest):
    manifest = manifest.copy()
    cols = list(manifest.columns)

    key = None
    for candidate in ["feature_name", "feature", "feature_id", "model_feature", "feature_clean"]:
        if candidate in cols:
            key = candidate
            break

    if key is None:
        raise ValueError("Could not identify feature key column in manifest. Columns: " + ", ".join(cols))

    manifest["feature_name"] = manifest[key].astype(str)

    if "feature_original" not in manifest.columns:
        if "original_feature" in manifest.columns:
            manifest["feature_original"] = manifest["original_feature"].astype(str)
        elif "feature_clean" in manifest.columns:
            manifest["feature_original"] = manifest["feature_clean"].astype(str)
        else:
            manifest["feature_original"] = manifest["feature_name"].astype(str)

    if "feature_group" not in manifest.columns:
        manifest["feature_group"] = ""

    if "feature_axis" not in manifest.columns:
        manifest["feature_axis"] = ""

    if "biological_theme" not in manifest.columns:
        manifest["biological_theme"] = manifest["feature_original"].map(infer_theme_from_text)

    manifest["feature_source"] = "manifest_fallback"
    return manifest


def load_strict_feature_table(residual_run, manifest):
    strict_path = residual_run / "09_residual_biology_interpretation" / "top_residual_biology_features_strict.csv"

    if strict_path.exists():
        strict = pd.read_csv(strict_path, low_memory=False)

        if "feature_name" in strict.columns:
            out = strict.copy()
            out["feature_name"] = out["feature_name"].astype(str)
            if "feature_original" not in out.columns:
                out["feature_original"] = out["feature_name"]
            if "feature_group" not in out.columns:
                out["feature_group"] = ""
            if "feature_axis" not in out.columns:
                out["feature_axis"] = ""
            if "biological_theme" not in out.columns:
                out["biological_theme"] = out["feature_original"].map(infer_theme_from_text)
            out["feature_source"] = "residual_biology_interpretation_strict"
            return out

    out = normalize_manifest(manifest)
    out = out[~out["feature_original"].map(artifact_or_caution)].copy()
    out["feature_source"] = "manifest_fallback_after_artifact_filter"
    return out


def build_feature_table(spatial, manifest, residual_run):
    spatial = spatial.copy()
    spatial["sample_id"] = spatial["sample_id"].astype(str)

    strict = load_strict_feature_table(residual_run, manifest)
    strict = strict.copy()
    strict["feature_name"] = strict["feature_name"].astype(str)

    available = set(spatial.columns)
    allowed = [f for f in strict["feature_name"].tolist() if f in available]

    if len(allowed) < 10:
        fallback = []
        for _, row in strict.iterrows():
            candidate = str(row.get("feature_original", ""))
            if candidate in available:
                fallback.append(candidate)
        if fallback:
            strict["feature_name"] = strict["feature_original"].astype(str)
            allowed = [f for f in strict["feature_name"].tolist() if f in available]

    if len(allowed) < 10:
        numeric_candidates = []
        for col in spatial.columns:
            if col == "sample_id":
                continue
            if artifact_or_caution(col):
                continue
            numeric_candidates.append(col)

        strict = pd.DataFrame({
            "feature_name": numeric_candidates,
            "feature_original": numeric_candidates,
            "feature_group": "",
            "feature_axis": "",
            "biological_theme": [infer_theme_from_text(x) for x in numeric_candidates],
            "feature_source": "last_resort_spatial_numeric_filter",
        })
        allowed = numeric_candidates

    if len(allowed) < 10:
        raise ValueError("Too few spatial biology features matched. Matched: " + str(len(allowed)))

    strict = strict[strict["feature_name"].isin(allowed)].drop_duplicates("feature_name").copy()

    x = spatial[["sample_id"] + allowed].copy()
    for col in allowed:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return x, strict, allowed


def treatment_eligibility(teacher, args):
    target = "fused_residual_vs_prior"

    rows = []
    for drug_key, sub in teacher.groupby("drug_key"):
        vals = pd.to_numeric(sub[target], errors="coerce").dropna()
        sample_count = sub.loc[vals.index, "sample_id"].astype(str).nunique() if len(vals) else 0

        drug_label = str(drug_key)
        for col in ["drug", "treatment", "treatment_name", "drug_name"]:
            if col in sub.columns and sub[col].notna().any():
                drug_label = str(sub[col].dropna().iloc[0])
                break

        if len(vals) > 0:
            target_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            target_range = float(vals.max() - vals.min())
            target_mean = float(vals.mean())
            target_min = float(vals.min())
            target_max = float(vals.max())
            n_unique = int(vals.nunique())
        else:
            target_std = 0.0
            target_range = 0.0
            target_mean = np.nan
            target_min = np.nan
            target_max = np.nan
            n_unique = 0

        reasons = []
        if sample_count < args.min_samples:
            reasons.append("low_sample_count")
        if target_std < args.min_target_std:
            reasons.append("low_residual_std")
        if target_range < args.min_target_range:
            reasons.append("low_residual_range")
        if n_unique < args.min_unique_targets:
            reasons.append("low_unique_targets")

        rows.append({
            "drug_key": str(drug_key),
            "drug_label": drug_label,
            "n_rows": int(len(sub)),
            "n_samples": int(sample_count),
            "target_mean": target_mean,
            "target_std": target_std,
            "target_min": target_min,
            "target_max": target_max,
            "target_range": target_range,
            "n_unique_targets": n_unique,
            "eligible": len(reasons) == 0,
            "ineligibility_reasons": ";".join(reasons),
        })

    elig = pd.DataFrame(rows)
    elig = elig.sort_values(["eligible", "n_samples", "target_std", "target_range"], ascending=[False, False, False, False])
    return elig


def prepare_xy_for_treatment(teacher, x_spatial, drug_key):
    target = "fused_residual_vs_prior"

    sub = teacher[teacher["drug_key"].astype(str) == str(drug_key)].copy()
    sub["sample_id"] = sub["sample_id"].astype(str)
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub = sub.dropna(subset=[target]).copy()

    keep_cols = ["sample_id", "drug_key", target]
    for col in ["drug", "treatment", "treatment_name", "drug_name"]:
        if col in sub.columns:
            keep_cols.append(col)

    keep_cols = list(dict.fromkeys(keep_cols))
    merged = sub[keep_cols].merge(x_spatial, on="sample_id", how="inner")
    merged = merged.dropna(subset=[target]).reset_index(drop=True)

    return merged


def select_features_training_only(X_train, y_train, feature_cols, max_features, min_var):
    values = X_train[feature_cols].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)

    variances = values.var(axis=0, ddof=1)
    candidates = variances[variances > min_var].index.tolist()

    if not candidates:
        candidates = feature_cols

    y_series = pd.Series(np.asarray(y_train, dtype=float), index=values.index)

    if y_series.std(ddof=1) <= 0:
        return variances.sort_values(ascending=False).head(max_features).index.tolist()

    x_rank = values[candidates].rank(axis=0)
    y_rank = y_series.rank()
    corr = x_rank.corrwith(y_rank).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = pd.DataFrame({
        "feature_name": corr.index,
        "abs_spearman_train": corr.values,
        "variance": variances.loc[corr.index].values,
    })

    score = score.sort_values(["abs_spearman_train", "variance"], ascending=[False, False])
    selected = score["feature_name"].head(max_features).tolist()

    if len(selected) < min(5, len(feature_cols)):
        selected = variances.sort_values(ascending=False).head(max_features).index.tolist()

    return selected


def fit_xgb(X_train, y_train, args):
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=args.random_state,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    return pipe


def validate_treatment(drug_row, teacher, x_spatial, feature_cols, args):
    drug_key = str(drug_row["drug_key"])
    drug_label = str(drug_row["drug_label"])

    data = prepare_xy_for_treatment(teacher, x_spatial, drug_key)
    target = "fused_residual_vs_prior"

    y = pd.to_numeric(data[target], errors="coerce").astype(float)
    X = data[feature_cols].copy()

    metric_rows = []
    pred_rows = []
    selected_feature_rows = []

    for repeat in range(args.n_repeats):
        seed = args.random_state + repeat

        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=args.test_size,
            random_state=seed,
        )

        selected = select_features_training_only(
            X.iloc[train_idx],
            y.iloc[train_idx],
            feature_cols,
            args.max_features_per_model,
            args.min_feature_variance,
        )

        pipe = fit_xgb(X.iloc[train_idx][selected], y.iloc[train_idx], args)

        pred_train = pipe.predict(X.iloc[train_idx][selected])
        pred_test = pipe.predict(X.iloc[test_idx][selected])

        baseline_value = float(y.iloc[train_idx].mean())
        baseline_test = np.repeat(baseline_value, len(test_idx))

        train_m = metric_safe(y.iloc[train_idx], pred_train)
        test_m = metric_safe(y.iloc[test_idx], pred_test)
        baseline_m = metric_safe(y.iloc[test_idx], baseline_test)

        metric_rows.append({
            "drug_key": drug_key,
            "drug_label": drug_label,
            "repeat": repeat,
            "random_state": seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "selected_feature_count": int(len(selected)),
            "train_mae": train_m["mae"],
            "train_rmse": train_m["rmse"],
            "train_r2": train_m["r2"],
            "train_pearson": train_m["pearson"],
            "train_spearman": train_m["spearman"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "test_r2": test_m["r2"],
            "test_pearson": test_m["pearson"],
            "test_spearman": test_m["spearman"],
            "baseline_test_mae": baseline_m["mae"],
            "baseline_test_rmse": baseline_m["rmse"],
            "baseline_test_r2": baseline_m["r2"],
            "mae_improvement_vs_baseline": baseline_m["mae"] - test_m["mae"],
            "rmse_improvement_vs_baseline": baseline_m["rmse"] - test_m["rmse"],
        })

        pred = pd.DataFrame({
            "drug_key": drug_key,
            "drug_label": drug_label,
            "repeat": repeat,
            "sample_id": data.iloc[test_idx]["sample_id"].values,
            "target": y.iloc[test_idx].values,
            "prediction": pred_test,
        })
        pred["prediction_error"] = pred["prediction"] - pred["target"]
        pred_rows.append(pred)

        selected_feature_rows.append(pd.DataFrame({
            "drug_key": drug_key,
            "drug_label": drug_label,
            "repeat": repeat,
            "feature_name": selected,
        }))

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    selected_features = pd.concat(selected_feature_rows, ignore_index=True) if selected_feature_rows else pd.DataFrame()

    return metrics, preds, selected_features


def train_final_model(drug_row, teacher, x_spatial, feature_cols, feature_meta, args, final_dir, do_shap):
    drug_key = str(drug_row["drug_key"])
    drug_label = str(drug_row["drug_label"])
    safe_drug = safe_name(drug_key)

    treatment_dir = final_dir / safe_drug
    ensure_dir(treatment_dir)

    data = prepare_xy_for_treatment(teacher, x_spatial, drug_key)
    target = "fused_residual_vs_prior"

    y = pd.to_numeric(data[target], errors="coerce").astype(float)
    X = data[feature_cols].copy()

    selected = select_features_training_only(
        X,
        y,
        feature_cols,
        args.max_features_per_model,
        args.min_feature_variance,
    )

    pipe = fit_xgb(X[selected], y, args)
    pred = pipe.predict(X[selected])

    all_m = metric_safe(y, pred)

    model_path = treatment_dir / "model.joblib"
    joblib.dump({
        "drug_key": drug_key,
        "drug_label": drug_label,
        "target": target,
        "feature_cols": selected,
        "pipeline": pipe,
        "metrics_all": all_m,
    }, model_path)

    pred_df = pd.DataFrame({
        "sample_id": data["sample_id"].astype(str).values,
        "drug_key": drug_key,
        "drug_label": drug_label,
        "target": y.values,
        "prediction": pred,
    })
    pred_df["prediction_error"] = pred_df["prediction"] - pred_df["target"]
    write_table(pred_df, treatment_dir / "final_model_predictions.tsv")

    model_step = pipe.named_steps["model"]
    gain = pd.DataFrame({
        "feature_name": selected,
        "gain_importance": model_step.feature_importances_,
    }).sort_values("gain_importance", ascending=False)

    gain = gain.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))
    write_table(gain, treatment_dir / "xgboost_feature_importance.tsv")

    shap_status = "not_requested"
    shap_importance = pd.DataFrame()

    if do_shap and HAS_SHAP:
        try:
            X_imp = pd.DataFrame(
                pipe.named_steps["imputer"].transform(X[selected]),
                columns=selected,
                index=X.index,
            )

            explainer = shap.TreeExplainer(model_step)
            values = explainer.shap_values(X_imp)

            shap_importance = pd.DataFrame({
                "feature_name": selected,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)

            shap_importance["rank"] = np.arange(1, len(shap_importance) + 1)
            shap_importance = shap_importance.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))
            shap_importance.insert(0, "drug_key", drug_key)
            shap_importance.insert(1, "drug_label", drug_label)

            write_table(shap_importance, treatment_dir / "shap_importance.tsv")

            theme = (
                shap_importance
                .groupby("biological_theme", dropna=False)
                .agg(
                    n_features=("feature_name", "count"),
                    total_mean_abs_shap=("mean_abs_shap", "sum"),
                    max_mean_abs_shap=("mean_abs_shap", "max")
                )
                .reset_index()
                .sort_values("total_mean_abs_shap", ascending=False)
            )
            write_table(theme, treatment_dir / "theme_summary.tsv")

            top = shap_importance.head(25).copy()
            labels = [
                short_label(row.get("feature_original", row.get("feature_name", "")), 65)
                for _, row in top.iterrows()
            ]
            values_plot = top["mean_abs_shap"].astype(float).values
            order = np.arange(len(top))[::-1]

            plt.figure(figsize=(10, max(6, len(top) * 0.34)))
            plt.barh(order, values_plot[::-1])
            plt.yticks(order, labels[::-1], fontsize=8)
            plt.xlabel("Mean absolute SHAP")
            plt.title("Top spatial features for " + short_label(drug_label, 50))
            savefig(treatment_dir / "top_shap_features.png")

            shap_status = "success"
        except Exception as exc:
            shap_status = "failed: " + str(exc)

    summary = {
        "drug_key": drug_key,
        "drug_label": drug_label,
        "n_rows": int(len(data)),
        "n_features": int(len(selected)),
        "model_path": str(model_path),
        "all_mae": all_m["mae"],
        "all_rmse": all_m["rmse"],
        "all_r2": all_m["r2"],
        "all_pearson": all_m["pearson"],
        "all_spearman": all_m["spearman"],
        "shap_status": shap_status,
        "treatment_dir": str(treatment_dir),
    }
    (treatment_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary, shap_importance


def make_figures(summary, feature_summary, theme_summary, figures_dir):
    ensure_dir(figures_dir)

    if len(summary) > 0:
        top = summary.sort_values("test_pearson_mean", ascending=False).head(30).copy()
        labels = [short_label(x, 35) for x in top["drug_label"]]
        values = top["test_pearson_mean"].astype(float).values
        order = np.arange(len(top))[::-1]

        plt.figure(figsize=(10, max(6, len(top) * 0.30)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Mean test Pearson across repeated splits")
        plt.title("Top per treatment residual models")
        savefig(figures_dir / "fig_01_top_treatments_by_test_pearson.png")

        plt.figure(figsize=(8, 5))
        plt.hist(summary["test_pearson_mean"].dropna().astype(float).values, bins=30)
        plt.xlabel("Mean test Pearson")
        plt.ylabel("Treatment count")
        plt.title("Per treatment model performance distribution")
        savefig(figures_dir / "fig_02_treatment_performance_distribution.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(summary["n_samples"].astype(float), summary["test_pearson_mean"].astype(float))
        plt.xlabel("Treatment sample count")
        plt.ylabel("Mean test Pearson")
        plt.title("Sample count versus per treatment performance")
        savefig(figures_dir / "fig_03_sample_count_vs_performance.png")

    if feature_summary is not None and len(feature_summary) > 0:
        topf = feature_summary.sort_values(["top10_frequency", "total_mean_abs_shap"], ascending=False).head(30).copy()
        labels = [short_label(x, 60) for x in topf["feature_original"].fillna(topf["feature_name"])]
        values = topf["top10_frequency"].astype(float).values
        order = np.arange(len(topf))[::-1]

        plt.figure(figsize=(10, max(6, len(topf) * 0.30)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Top 10 SHAP frequency across final models")
        plt.title("Recurring spatial features across treatment models")
        savefig(figures_dir / "fig_04_cross_treatment_feature_recurrence.png")

    if theme_summary is not None and len(theme_summary) > 0:
        topt = theme_summary.sort_values("total_mean_abs_shap", ascending=False).head(20).copy()
        labels = [short_label(x, 55) for x in topt["biological_theme"].fillna("unmapped")]
        values = topt["total_mean_abs_shap"].astype(float).values
        order = np.arange(len(topt))[::-1]

        plt.figure(figsize=(10, max(5, len(topt) * 0.35)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Total mean absolute SHAP across final models")
        plt.title("Cross treatment biology theme contribution")
        savefig(figures_dir / "fig_05_cross_treatment_theme_contribution.png")


def main():
    args = parse_args()

    project = Path(args.project_root)
    spm = project / "prediction_modeling_pipeline" / "spatial_prediction_model"
    output_root = Path(args.output_root)

    input_dir = output_root / "01_inputs_and_eligibility"
    validation_dir = output_root / "02_repeated_split_validation"
    final_dir = output_root / "03_final_models"
    analysis_dir = output_root / "04_analysis"
    figures_dir = analysis_dir / "figures"
    reports_dir = output_root / "05_reports"

    for folder in [input_dir, validation_dir, final_dir, analysis_dir, figures_dir, reports_dir]:
        ensure_dir(folder)

    residual_run = latest_dir(spm / "outputs", "output_run_102_governed_full102_xgboost_residual_filtered_*")
    derived = latest_dir(spm / "outputs" / "_derived_handoffs", "residual_prior_adjusted_filtered_*")

    if residual_run is None:
        raise FileNotFoundError("Could not find residual pair model run")

    if derived is None:
        raise FileNotFoundError("Could not find derived residual handoff")

    handoff = derived / "full102_handoff"

    spatial = read_any(handoff / "model_input_numeric.csv")
    teacher = read_any(handoff / "visium_fused_teacher_table.tsv")
    manifest = read_any(handoff / "feature_manifest.csv")

    teacher = teacher.copy()
    teacher["sample_id"] = teacher["sample_id"].astype(str)
    teacher["drug_key"] = teacher["drug_key"].astype(str)

    if "fused_residual_vs_prior" not in teacher.columns:
        raise ValueError("Teacher table missing fused_residual_vs_prior")

    x_spatial, feature_meta, feature_cols = build_feature_table(spatial, manifest, residual_run)

    eligibility = treatment_eligibility(teacher, args)
    write_table(eligibility, input_dir / "treatment_eligibility.tsv")
    write_table(feature_meta, input_dir / "feature_metadata.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), input_dir / "spatial_feature_list.tsv")

    eligible = eligibility[eligibility["eligible"] == True].copy()

    if len(eligible) == 0:
        raise ValueError("No eligible treatments after filtering")

    if args.max_treatments and args.max_treatments > 0:
        selected_treatments = eligible.head(args.max_treatments).copy()
    else:
        selected_treatments = eligible.copy()

    write_table(selected_treatments, input_dir / "selected_treatments_for_run.tsv")

    run_input_summary = {
        "output_root": str(output_root),
        "mode": args.mode,
        "residual_run": str(residual_run),
        "derived_handoff": str(derived),
        "n_spatial_samples": int(x_spatial["sample_id"].nunique()),
        "n_teacher_rows": int(len(teacher)),
        "n_total_treatments": int(teacher["drug_key"].nunique()),
        "n_eligible_treatments": int(len(eligible)),
        "n_selected_treatments": int(len(selected_treatments)),
        "n_features": int(len(feature_cols)),
        "min_samples": args.min_samples,
        "min_target_std": args.min_target_std,
        "min_target_range": args.min_target_range,
        "n_repeats": args.n_repeats,
    }
    (input_dir / "run_input_summary.json").write_text(json.dumps(run_input_summary, indent=2), encoding="utf-8")

    all_metrics = []
    all_predictions = []
    all_selected_features = []

    print("")
    print("=" * 95)
    print("FILTERED PER TREATMENT RESIDUAL MODEL VALIDATION")
    print("=" * 95)
    print("Mode:", args.mode)
    print("Selected treatments:", len(selected_treatments))
    print("Eligible treatments:", len(eligible))
    print("Spatial features:", len(feature_cols))
    print("Repeats per treatment:", args.n_repeats)

    for idx, row in selected_treatments.reset_index(drop=True).iterrows():
        print("")
        print(f"[{idx + 1}/{len(selected_treatments)}] {row['drug_key']} | {row['drug_label']} | n={row['n_samples']} | std={row['target_std']:.4f}")

        metrics, preds, selected_features = validate_treatment(row, teacher, x_spatial, feature_cols, args)
        all_metrics.append(metrics)
        all_predictions.append(preds)
        all_selected_features.append(selected_features)

        print("  mean test Pearson:", float(metrics["test_pearson"].mean()))
        print("  median test Pearson:", float(metrics["test_pearson"].median()))
        print("  mean RMSE improvement:", float(metrics["rmse_improvement_vs_baseline"].mean()))

    metrics_long = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    selected_feature_long = pd.concat(all_selected_features, ignore_index=True) if all_selected_features else pd.DataFrame()

    write_table(metrics_long, validation_dir / "per_treatment_repeated_split_metrics_long.tsv")
    write_table(predictions_long, validation_dir / "per_treatment_repeated_split_predictions_long.tsv")
    write_table(selected_feature_long, validation_dir / "per_treatment_selected_features_by_repeat.tsv")

    summary = (
        metrics_long
        .groupby(["drug_key", "drug_label"], dropna=False)
        .agg(
            n_repeats=("repeat", "count"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            selected_feature_count_mean=("selected_feature_count", "mean"),
            test_pearson_mean=("test_pearson", "mean"),
            test_pearson_median=("test_pearson", "median"),
            test_pearson_std=("test_pearson", "std"),
            test_pearson_q025=("test_pearson", q025),
            test_pearson_q975=("test_pearson", q975),
            test_pearson_positive_fraction=("test_pearson", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            test_r2_mean=("test_r2", "mean"),
            test_r2_median=("test_r2", "median"),
            test_mae_mean=("test_mae", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            mae_improvement_mean=("mae_improvement_vs_baseline", "mean"),
            rmse_improvement_mean=("rmse_improvement_vs_baseline", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(
        selected_treatments[["drug_key", "n_samples", "target_mean", "target_std", "target_range", "n_unique_targets"]],
        on="drug_key",
        how="left",
    )

    summary["model_selection_score"] = (
        summary["test_pearson_mean"].fillna(-1.0)
        + 0.25 * summary["test_pearson_positive_fraction"].fillna(0.0)
        + 2.0 * summary["rmse_improvement_mean"].fillna(0.0)
    )

    summary["promising"] = (
        (summary["test_pearson_mean"] >= args.min_promising_pearson)
        & (summary["test_pearson_positive_fraction"] >= args.min_positive_fraction)
        & (summary["rmse_improvement_mean"] > args.min_rmse_improvement)
    )

    summary = summary.sort_values("model_selection_score", ascending=False).reset_index(drop=True)

    promising = summary[summary["promising"] == True].copy()

    if len(promising) == 0:
        final_candidates = summary.head(args.max_final_models if args.max_final_models > 0 else len(summary)).copy()
        final_selection_note = "no treatments passed strict promising filter, selected top scoring treatments"
    else:
        final_candidates = promising.copy()
        if args.max_final_models and args.max_final_models > 0:
            final_candidates = final_candidates.head(args.max_final_models).copy()
        final_selection_note = "selected promising treatments"

    final_candidates["selected_for_final_model"] = True

    summary = summary.merge(
        final_candidates[["drug_key", "selected_for_final_model"]],
        on="drug_key",
        how="left"
    )
    summary["selected_for_final_model"] = summary["selected_for_final_model"].fillna(False)

    write_table(summary, analysis_dir / "treatment_model_summary.tsv")
    write_table(promising, analysis_dir / "promising_treatments.tsv")
    write_table(final_candidates, analysis_dir / "selected_final_model_treatments.tsv")

    final_infos = []
    all_shap_importances = []

    print("")
    print("=" * 95)
    print("TRAINING FINAL MODELS FOR SELECTED TREATMENTS")
    print("=" * 95)
    print("Final treatment count:", len(final_candidates))
    print("Selection note:", final_selection_note)

    for idx, row in final_candidates.reset_index(drop=True).iterrows():
        do_shap = idx < args.max_shap_treatments
        print("")
        print(f"Final model [{idx + 1}/{len(final_candidates)}] {row['drug_key']} | {row['drug_label']} | SHAP={do_shap}")

        final_info, shap_importance = train_final_model(row, teacher, x_spatial, feature_cols, feature_meta, args, final_dir, do_shap)
        final_infos.append(final_info)

        if shap_importance is not None and len(shap_importance) > 0:
            all_shap_importances.append(shap_importance)

        print("  all Pearson:", final_info["all_pearson"])
        print("  SHAP:", final_info["shap_status"])

    final_info_df = pd.DataFrame(final_infos)
    write_table(final_info_df, analysis_dir / "final_model_inventory.tsv")

    summary = summary.merge(
        final_info_df[["drug_key", "treatment_dir", "shap_status", "n_features", "all_pearson", "all_r2"]],
        on="drug_key",
        how="left"
    )
    write_table(summary, analysis_dir / "treatment_model_summary.tsv")

    if all_shap_importances:
        shap_all = pd.concat(all_shap_importances, ignore_index=True)
        write_table(shap_all, analysis_dir / "final_model_shap_importance_all.tsv")

        n_shap_treatments = int(shap_all["drug_key"].nunique())

        feature_summary = (
            shap_all
            .groupby(["feature_name"], dropna=False)
            .agg(
                n_treatments_present=("drug_key", "nunique"),
                total_mean_abs_shap=("mean_abs_shap", "sum"),
                mean_abs_shap_mean=("mean_abs_shap", "mean"),
                mean_rank=("rank", "mean"),
                top10_count=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") <= 10).sum())),
                top20_count=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") <= 20).sum())),
            )
            .reset_index()
        )

        meta_cols = [c for c in ["feature_name", "feature_original", "feature_group", "feature_axis", "biological_theme"] if c in feature_meta.columns]
        feature_summary = feature_summary.merge(feature_meta[meta_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

        feature_summary["n_shap_treatments"] = n_shap_treatments
        feature_summary["top10_frequency"] = feature_summary["top10_count"] / max(n_shap_treatments, 1)
        feature_summary["top20_frequency"] = feature_summary["top20_count"] / max(n_shap_treatments, 1)

        feature_summary = feature_summary.sort_values(["top10_frequency", "total_mean_abs_shap"], ascending=False)
        write_table(feature_summary, analysis_dir / "cross_treatment_feature_stability.tsv")

        theme_summary = (
            shap_all
            .groupby(["biological_theme"], dropna=False)
            .agg(
                n_features=("feature_name", "nunique"),
                n_treatments=("drug_key", "nunique"),
                total_mean_abs_shap=("mean_abs_shap", "sum"),
                mean_abs_shap=("mean_abs_shap", "mean"),
            )
            .reset_index()
            .sort_values("total_mean_abs_shap", ascending=False)
        )
        write_table(theme_summary, analysis_dir / "cross_treatment_theme_summary.tsv")
    else:
        shap_all = pd.DataFrame()
        feature_summary = pd.DataFrame()
        theme_summary = pd.DataFrame()

    make_figures(summary, feature_summary, theme_summary, figures_dir)

    script_provenance = pd.DataFrame([{
        "script": str(Path(__file__)),
        "exists": Path(__file__).exists(),
        "sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        "note": "new filtered per treatment residual script, no canonical script overwritten",
    }])
    write_table(script_provenance, output_root / "script_provenance.tsv")

    report_path = reports_dir / "filtered_per_treatment_residual_model_report.txt"

    show_cols = [
        "drug_key",
        "drug_label",
        "n_samples",
        "target_std",
        "test_pearson_mean",
        "test_pearson_median",
        "test_pearson_positive_fraction",
        "test_r2_mean",
        "rmse_improvement_mean",
        "model_selection_score",
        "promising",
        "selected_for_final_model",
        "shap_status",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]

    lines = []
    lines.append("FILTERED PER TREATMENT RESIDUAL MODEL REPORT")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"Output root: {output_root}")
    lines.append(f"Mode: {args.mode}")
    lines.append(f"Residual pair model run used: {residual_run}")
    lines.append(f"Derived handoff used: {derived}")
    lines.append("")
    lines.append("1. Design")
    lines.append("-" * 95)
    lines.append("Each model predicts fused_residual_vs_prior for one treatment using spatial biology features only.")
    lines.append("Drug dummies are not used because treatment identity is fixed within each model.")
    lines.append("Treatments are filtered by sample count, residual standard deviation, residual range, and target uniqueness.")
    lines.append("Repeated train test splits are used for screening.")
    lines.append("Final XGBoost and SHAP models are written for selected treatments.")
    lines.append("")
    lines.append("2. Input summary")
    lines.append("-" * 95)
    lines.append(json.dumps(run_input_summary, indent=2))
    lines.append("")
    lines.append("3. Treatment filtering")
    lines.append("-" * 95)
    lines.append(f"Eligible treatments: {len(eligible)}")
    lines.append(f"Selected treatments in this run: {len(selected_treatments)}")
    lines.append(f"Final model treatments: {len(final_candidates)}")
    lines.append(f"Final selection note: {final_selection_note}")
    lines.append("")
    lines.append("4. Top treatment model summary")
    lines.append("-" * 95)
    lines.append(summary[show_cols].head(40).to_string(index=False))
    lines.append("")

    if len(feature_summary) > 0:
        lines.append("5. Recurring spatial features across final treatment models")
        lines.append("-" * 95)
        feature_cols_show = [c for c in ["feature_name", "feature_original", "biological_theme", "top10_frequency", "top20_frequency", "total_mean_abs_shap", "mean_rank"] if c in feature_summary.columns]
        lines.append(feature_summary[feature_cols_show].head(40).to_string(index=False))
        lines.append("")

    if len(theme_summary) > 0:
        lines.append("6. Cross treatment biology theme summary")
        lines.append("-" * 95)
        lines.append(theme_summary.head(30).to_string(index=False))
        lines.append("")

    lines.append("7. Interpretation guide")
    lines.append("-" * 95)
    lines.append("Promising treatment models indicate treatment specific spatial biology explaining above prior or below prior residual response.")
    lines.append("Because each treatment has cohort scale sample counts, these are screening level treatment specific models.")
    lines.append("The strongest results should be interpreted by recurring SHAP themes and compared back to the residual pair model and broad residual model.")
    lines.append("Treatments with unstable or negative repeated split metrics should not be biologically interpreted as treatment specific signals.")
    lines.append("")
    lines.append("8. Output files")
    lines.append("-" * 95)
    lines.append(f"Eligibility table: {input_dir / 'treatment_eligibility.tsv'}")
    lines.append(f"Repeated split metrics: {validation_dir / 'per_treatment_repeated_split_metrics_long.tsv'}")
    lines.append(f"Treatment model summary: {analysis_dir / 'treatment_model_summary.tsv'}")
    lines.append(f"Promising treatments: {analysis_dir / 'promising_treatments.tsv'}")
    lines.append(f"Selected final models: {analysis_dir / 'selected_final_model_treatments.tsv'}")
    lines.append(f"Final model inventory: {analysis_dir / 'final_model_inventory.tsv'}")
    lines.append(f"Feature stability: {analysis_dir / 'cross_treatment_feature_stability.tsv'}")
    lines.append(f"Theme summary: {analysis_dir / 'cross_treatment_theme_summary.tsv'}")
    lines.append(f"Figures: {figures_dir}")

    write_text_report(report_path, lines)

    summary_json = {
        "output_root": str(output_root),
        "mode": args.mode,
        "residual_run": str(residual_run),
        "derived_handoff": str(derived),
        "n_eligible_treatments": int(len(eligible)),
        "n_selected_treatments": int(len(selected_treatments)),
        "n_final_models": int(len(final_candidates)),
        "n_shap_models": int(final_info_df["shap_status"].astype(str).str.contains("success", na=False).sum()) if len(final_info_df) else 0,
        "top_treatment_by_mean_test_pearson": str(summary.iloc[0]["drug_key"]) if len(summary) else "",
        "top_treatment_label": str(summary.iloc[0]["drug_label"]) if len(summary) else "",
        "top_mean_test_pearson": float(summary.iloc[0]["test_pearson_mean"]) if len(summary) else None,
        "report": str(report_path),
    }
    (output_root / "filtered_per_treatment_residual_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("")
    print("=" * 95)
    print("FILTERED PER TREATMENT RESIDUAL MODEL RUN COMPLETE")
    print("=" * 95)
    print("Output root:", output_root)
    print("Report:", report_path)
    print("Mode:", args.mode)
    print("Eligible treatments:", len(eligible))
    print("Selected treatments:", len(selected_treatments))
    print("Final model treatments:", len(final_candidates))
    print("SHAP success models:", summary_json["n_shap_models"])
    print("Top treatment:", summary_json["top_treatment_by_mean_test_pearson"], "|", summary_json["top_treatment_label"])
    print("Top mean test Pearson:", summary_json["top_mean_test_pearson"])
    print("")
    print("Top treatment model summary:")
    print(summary[show_cols].head(20).to_string(index=False))
    print("")
    print("Generated output folders:")
    for folder in [input_dir, validation_dir, final_dir, analysis_dir, figures_dir, reports_dir]:
        print(" ", folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
