"""
Script:
    08_2_train_spatial_only_broad_residual_model.py

Purpose:
    Train sample-level spatial-only models for broad residual response behavior.

Why this version exists:
    08_1 expected feature_manifest.csv to contain feature_name. Some derived
    handoff manifests may use a different schema. This version uses the strict
    residual biology feature table when available and falls back to flexible
    manifest column detection.

Design:
    New experimental script.
    Does not overwrite canonical scripts.
    Uses one row per sample.
    Uses spatial biology features only.
    Uses targets derived from fused_residual_vs_prior.
"""

from pathlib import Path
import argparse
import json
import hashlib
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=250)
    parser.add_argument("--min-feature-variance", type=float, default=1e-12)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
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


def short_label(value, width=60):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + "..."


def latest_dir(base, pattern):
    base = Path(base)
    hits = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


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

    if len(y_true) >= 3 and np.nanstd(y_true) > 0:
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


def build_targets(teacher):
    if "fused_residual_vs_prior" not in teacher.columns:
        raise ValueError("Teacher table missing fused_residual_vs_prior")

    df = teacher.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["fused_residual_vs_prior"] = pd.to_numeric(df["fused_residual_vs_prior"], errors="coerce")
    df = df.dropna(subset=["fused_residual_vs_prior"])

    rows = []

    for sample_id, sub in df.groupby("sample_id"):
        vals = sub["fused_residual_vs_prior"].astype(float).values
        vals = vals[np.isfinite(vals)]
        vals_sorted = np.sort(vals)

        n = len(vals)
        top_n = min(5, n)
        top10_n = min(10, n)

        rows.append({
            "sample_id": sample_id,
            "n_treatments": int(n),
            "mean_residual": float(np.mean(vals)),
            "median_residual": float(np.median(vals)),
            "residual_std": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
            "residual_iqr": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
            "positive_residual_fraction": float(np.mean(vals > 0)),
            "strong_positive_fraction_005": float(np.mean(vals > 0.05)),
            "strong_negative_fraction_m005": float(np.mean(vals < -0.05)),
            "top5_mean_residual": float(np.mean(vals_sorted[-top_n:])),
            "bottom5_mean_residual": float(np.mean(vals_sorted[:top_n])),
            "top10_mean_residual": float(np.mean(vals_sorted[-top10_n:])),
            "bottom10_mean_residual": float(np.mean(vals_sorted[:top10_n])),
            "broad_resistance_score": float(-np.mean(vals)),
        })

    return pd.DataFrame(rows)


def load_strict_feature_table(residual_run):
    strict_path = residual_run / "09_residual_biology_interpretation" / "top_residual_biology_features_strict.csv"

    if strict_path.exists():
        strict = pd.read_csv(strict_path, low_memory=False)

        feature_col = "feature_name" if "feature_name" in strict.columns else None
        original_col = "feature_original" if "feature_original" in strict.columns else None

        if feature_col is not None:
            out = strict.copy()
            out["feature_name"] = out[feature_col].astype(str)
            if original_col is None:
                out["feature_original"] = out["feature_name"]
            else:
                out["feature_original"] = out[original_col].astype(str)
            if "biological_theme" not in out.columns:
                out["biological_theme"] = out["feature_original"].map(infer_theme_from_text)
            if "feature_group" not in out.columns:
                out["feature_group"] = ""
            if "feature_axis" not in out.columns:
                out["feature_axis"] = ""
            out["feature_source"] = "residual_biology_interpretation_strict"
            return out

    return pd.DataFrame()


def normalize_manifest(manifest):
    manifest = manifest.copy()
    cols = list(manifest.columns)

    key = None
    for candidate in ["feature_name", "feature", "feature_id", "model_feature", "feature_clean"]:
        if candidate in cols:
            key = candidate
            break

    if key is None:
        raise ValueError("Could not identify a feature key column in manifest. Columns: " + ", ".join(cols))

    manifest["feature_name"] = manifest[key].astype(str)

    if "feature_original" not in manifest.columns:
        if "original_feature" in manifest.columns:
            manifest["feature_original"] = manifest["original_feature"].astype(str)
        elif "feature_clean" in manifest.columns:
            manifest["feature_original"] = manifest["feature_clean"].astype(str)
        else:
            manifest["feature_original"] = manifest["feature_name"]

    if "feature_group" not in manifest.columns:
        manifest["feature_group"] = ""

    if "feature_axis" not in manifest.columns:
        manifest["feature_axis"] = ""

    manifest["biological_theme"] = manifest["feature_original"].map(infer_theme_from_text)
    manifest["feature_source"] = "manifest_fallback"

    return manifest


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


def build_feature_table(spatial, manifest, residual_run):
    spatial = spatial.copy()
    spatial["sample_id"] = spatial["sample_id"].astype(str)

    strict = load_strict_feature_table(residual_run)

    if strict.empty:
        strict = normalize_manifest(manifest)
        strict = strict[~strict["feature_original"].map(artifact_or_caution)].copy()
        strict["feature_source"] = "manifest_fallback_after_artifact_filter"

    strict = strict.copy()
    strict["feature_name"] = strict["feature_name"].astype(str)

    allowed = [f for f in strict["feature_name"].tolist() if f in spatial.columns]

    if len(allowed) < 10:
        available = set(spatial.columns)
        fallback = []
        for _, row in strict.iterrows():
            candidate = str(row.get("feature_original", ""))
            if candidate in available:
                fallback.append(candidate)

        if fallback:
            strict["feature_name"] = strict["feature_original"].astype(str)
            allowed = [f for f in strict["feature_name"].tolist() if f in spatial.columns]

    if len(allowed) < 10:
        raise ValueError(
            "Too few strict biology features matched spatial table. Matched "
            + str(len(allowed))
            + ". Spatial columns example: "
            + ", ".join(list(spatial.columns[:20]))
            + ". Strict columns example: "
            + ", ".join(strict["feature_name"].head(20).astype(str).tolist())
        )

    strict = strict[strict["feature_name"].isin(allowed)].copy()

    x = spatial[["sample_id"] + allowed].copy()

    for col in allowed:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return x, strict, allowed


def select_features_by_variance(x, feature_cols, max_features, min_var):
    values = x[feature_cols].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True))

    variances = values.var(axis=0, ddof=1).sort_values(ascending=False)
    variances = variances[variances > min_var]

    selected = variances.head(max_features).index.tolist()

    report = pd.DataFrame({
        "feature_name": variances.index,
        "variance": variances.values,
        "selected": [f in selected for f in variances.index],
    })

    return selected, report


def train_one_target(target_name, sample_df, feature_cols, feature_meta, outdir, args):
    ensure_dir(outdir)

    sub = sample_df[["sample_id", target_name] + feature_cols].dropna(subset=[target_name]).copy()
    y = pd.to_numeric(sub[target_name], errors="coerce")
    mask = y.notna()
    sub = sub.loc[mask].copy()
    y = y.loc[mask].astype(float)

    xmat = sub[feature_cols].copy()

    train_idx, test_idx = train_test_split(
        np.arange(len(sub)),
        test_size=args.test_size,
        random_state=args.random_state,
    )

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

    pipe.fit(xmat.iloc[train_idx], y.iloc[train_idx])

    pred_all = pipe.predict(xmat)

    pred_df = pd.DataFrame({
        "sample_id": sub["sample_id"].values,
        "target": y.values,
        "prediction": pred_all,
        "split": "all_labeled",
    })
    pred_df.loc[train_idx, "split"] = "train"
    pred_df.loc[test_idx, "split"] = "test"
    pred_df["residual"] = pred_df["prediction"] - pred_df["target"]

    metrics = []

    for split, idx in [
        ("train", train_idx),
        ("test", test_idx),
        ("all_labeled", np.arange(len(sub))),
    ]:
        m = metric_safe(y.iloc[idx], pred_all[idx])
        m["target_name"] = target_name
        m["split"] = split
        metrics.append(m)

    metrics_df = pd.DataFrame(metrics)

    model_path = outdir / "model.joblib"
    joblib.dump({
        "target_name": target_name,
        "feature_cols": feature_cols,
        "pipeline": pipe,
        "metrics": metrics,
    }, model_path)

    write_table(pred_df, outdir / "predictions.tsv")
    write_table(metrics_df, outdir / "metrics.tsv")

    model_step = pipe.named_steps["model"]

    gain = pd.DataFrame({
        "feature_name": feature_cols,
        "gain_importance": model_step.feature_importances_,
    }).sort_values("gain_importance", ascending=False)

    gain = gain.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))
    write_table(gain, outdir / "xgboost_feature_importance.tsv")

    shap_status = "not_run"
    shap_importance = pd.DataFrame()

    if HAS_SHAP:
        try:
            x_imp = pd.DataFrame(
                pipe.named_steps["imputer"].transform(xmat),
                columns=feature_cols,
                index=xmat.index,
            )

            explainer = shap.TreeExplainer(model_step)
            values = explainer.shap_values(x_imp)

            shap_importance = pd.DataFrame({
                "feature_name": feature_cols,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)

            shap_importance = shap_importance.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))

            write_table(shap_importance, outdir / "shap_importance.tsv")

            shap_values = pd.DataFrame(values, columns=feature_cols)
            shap_values.insert(0, "sample_id", sub["sample_id"].values)
            shap_values.insert(1, "target", y.values)
            shap_values.insert(2, "prediction", pred_all)
            write_table(shap_values, outdir / "shap_values.tsv")

            top = shap_importance.head(25).copy()
            labels = [short_label(row.get("feature_original", row.get("feature_name", "")), 65) for _, row in top.iterrows()]
            scores = top["mean_abs_shap"].astype(float).values
            order = np.arange(len(top))[::-1]

            plt.figure(figsize=(10, max(6, len(top) * 0.34)))
            plt.barh(order, scores[::-1])
            plt.yticks(order, labels[::-1], fontsize=8)
            plt.xlabel("Mean absolute SHAP")
            plt.title("Top spatial features for " + target_name)
            savefig(outdir / "top_shap_features.png")

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
            write_table(theme, outdir / "theme_summary.tsv")

            shap_status = "success"
        except Exception as exc:
            shap_status = "failed: " + str(exc)

    test_row = metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict()

    summary = {
        "target_name": target_name,
        "n_rows": int(len(sub)),
        "n_features": int(len(feature_cols)),
        "model_path": str(model_path),
        "shap_status": shap_status,
        "test_pearson": float(test_row.get("pearson", np.nan)),
        "test_r2": float(test_row.get("r2", np.nan)),
        "test_mae": float(test_row.get("mae", np.nan)),
    }

    (outdir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return metrics_df, pred_df, gain, shap_importance, summary


def main():
    args = parse_args()

    project = Path(args.project_root)
    spm = project / "prediction_modeling_pipeline" / "spatial_prediction_model"
    output_root = Path(args.output_root)

    ensure_dir(output_root)

    prepared_dir = output_root / "01_prepared_sample_level_data"
    model_dir = output_root / "02_models"
    summary_dir = output_root / "03_model_summaries"
    fig_dir = output_root / "04_figures"

    for folder in [prepared_dir, model_dir, summary_dir, fig_dir]:
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

    target_df = build_targets(teacher)
    x_spatial, feature_meta, allowed_features = build_feature_table(spatial, manifest, residual_run)

    merged = target_df.merge(x_spatial, on="sample_id", how="inner")

    candidate_features = [c for c in x_spatial.columns if c != "sample_id"]
    selected_features, variance_report = select_features_by_variance(
        merged,
        candidate_features,
        args.max_features,
        args.min_feature_variance,
    )

    feature_meta = feature_meta[feature_meta["feature_name"].isin(selected_features)].copy()

    sample_level = merged[["sample_id"] + [c for c in target_df.columns if c != "sample_id"] + selected_features].copy()

    write_table(target_df, prepared_dir / "sample_level_broad_residual_targets.tsv")
    write_table(x_spatial, prepared_dir / "sample_level_spatial_features_strict_biology.tsv")
    write_table(sample_level, prepared_dir / "sample_level_modeling_table.tsv")
    write_table(feature_meta, prepared_dir / "selected_feature_metadata.tsv")
    write_table(variance_report, prepared_dir / "feature_variance_selection_report.tsv")
    write_table(pd.DataFrame({"feature_name": selected_features}), prepared_dir / "selected_spatial_features.tsv")

    target_names = [
        "mean_residual",
        "median_residual",
        "positive_residual_fraction",
        "strong_positive_fraction_005",
        "strong_negative_fraction_m005",
        "top5_mean_residual",
        "bottom5_mean_residual",
        "top10_mean_residual",
        "bottom10_mean_residual",
        "residual_std",
        "residual_iqr",
        "broad_resistance_score",
    ]

    all_metrics = []
    all_summaries = []
    all_top_features = []

    for target in target_names:
        print("")
        print("=" * 80)
        print("Training spatial-only broad residual target:", target)
        print("=" * 80)

        target_out = model_dir / target

        metrics_df, preds, gain, shap_importance, summary = train_one_target(
            target,
            sample_level,
            selected_features,
            feature_meta,
            target_out,
            args,
        )

        all_metrics.append(metrics_df)
        all_summaries.append(summary)

        if shap_importance is not None and len(shap_importance) > 0:
            top = shap_importance.head(30).copy()
            top.insert(0, "target_name", target)
            all_top_features.append(top)

        print("Rows:", summary["n_rows"])
        print("Features:", summary["n_features"])
        print("SHAP:", summary["shap_status"])
        print("Test Pearson:", summary["test_pearson"])
        print("Test R2:", summary["test_r2"])
        print("Test MAE:", summary["test_mae"])

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summaries = pd.DataFrame(all_summaries)

    write_table(metrics_all, summary_dir / "broad_residual_model_metrics.tsv")
    write_table(summaries, summary_dir / "broad_residual_model_summary.tsv")

    if all_top_features:
        top_all = pd.concat(all_top_features, ignore_index=True)
        write_table(top_all, summary_dir / "top_features_all_targets.tsv")

        theme_all = (
            top_all
            .groupby(["target_name", "biological_theme"], dropna=False)
            .agg(
                n_features=("feature_name", "count"),
                total_mean_abs_shap=("mean_abs_shap", "sum"),
                max_mean_abs_shap=("mean_abs_shap", "max")
            )
            .reset_index()
            .sort_values(["target_name", "total_mean_abs_shap"], ascending=[True, False])
        )
        write_table(theme_all, summary_dir / "theme_summary_all_targets.tsv")

        cross_target = (
            top_all
            .groupby(["feature_name", "feature_original", "biological_theme"], dropna=False)
            .agg(
                n_targets=("target_name", "nunique"),
                total_mean_abs_shap=("mean_abs_shap", "sum"),
                max_mean_abs_shap=("mean_abs_shap", "max")
            )
            .reset_index()
            .sort_values("total_mean_abs_shap", ascending=False)
        )
        write_table(cross_target, summary_dir / "cross_target_top_features.tsv")

    test_metrics = metrics_all[metrics_all["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values("pearson", ascending=False)

    plt.figure(figsize=(10, 6))
    labels = [short_label(x, 35) for x in test_metrics["target_name"]]
    plt.bar(np.arange(len(test_metrics)), test_metrics["pearson"].astype(float).values)
    plt.xticks(np.arange(len(test_metrics)), labels, rotation=35, ha="right")
    plt.ylabel("Test Pearson")
    plt.title("Spatial-only broad residual model performance")
    savefig(fig_dir / "fig_01_test_pearson_by_target.png")

    plt.figure(figsize=(10, 6))
    labels = [short_label(x, 35) for x in test_metrics["target_name"]]
    plt.bar(np.arange(len(test_metrics)), test_metrics["r2"].astype(float).values)
    plt.xticks(np.arange(len(test_metrics)), labels, rotation=35, ha="right")
    plt.ylabel("Test R2")
    plt.title("Spatial-only broad residual model R2 by target")
    savefig(fig_dir / "fig_02_test_r2_by_target.png")

    if all_top_features:
        top_all = pd.concat(all_top_features, ignore_index=True)
        cross_target = (
            top_all
            .groupby(["feature_name", "feature_original", "biological_theme"], dropna=False)
            .agg(
                n_targets=("target_name", "nunique"),
                total_mean_abs_shap=("mean_abs_shap", "sum"),
                max_mean_abs_shap=("mean_abs_shap", "max")
            )
            .reset_index()
            .sort_values("total_mean_abs_shap", ascending=False)
        )

        plot_df = cross_target.head(25).copy()
        labels = [short_label(x, 65) for x in plot_df["feature_original"]]
        vals = plot_df["total_mean_abs_shap"].astype(float).values
        order = np.arange(len(plot_df))[::-1]

        plt.figure(figsize=(10, max(6, len(plot_df) * 0.34)))
        plt.barh(order, vals[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Cross-target total mean absolute SHAP")
        plt.title("Cross-target spatial biology features")
        savefig(fig_dir / "fig_03_cross_target_top_features.png")

    script_provenance = pd.DataFrame([{
        "script": str(Path(__file__)),
        "exists": Path(__file__).exists(),
        "sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        "note": "08_2 experimental script, created because 08_1 failed on manifest schema"
    }])
    write_table(script_provenance, output_root / "script_provenance.tsv")

    report_lines = []
    report_lines.append("SPATIAL-ONLY BROAD RESIDUAL SAMPLE-LEVEL MODEL REPORT")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append(f"Output root: {output_root}")
    report_lines.append(f"Residual pair-model run used: {residual_run}")
    report_lines.append(f"Derived handoff used: {derived}")
    report_lines.append("")
    report_lines.append("1. Design")
    report_lines.append("-" * 90)
    report_lines.append("This model collapses sample-treatment residuals into one row per sample.")
    report_lines.append("It uses spatial features only. Drug dummies are not used.")
    report_lines.append("The target is derived from fused_residual_vs_prior.")
    report_lines.append("Features are restricted to strict biology features from the residual biology interpretation report.")
    report_lines.append("")
    report_lines.append("2. Data")
    report_lines.append("-" * 90)
    report_lines.append(f"Samples: {sample_level['sample_id'].nunique()}")
    report_lines.append(f"Strict biology candidate features matched to spatial table: {len(candidate_features)}")
    report_lines.append(f"Selected features by variance: {len(selected_features)}")
    report_lines.append("")
    report_lines.append("3. Target summaries")
    report_lines.append("-" * 90)
    report_lines.append(target_df.drop(columns=["sample_id"], errors="ignore").describe().T.to_string())
    report_lines.append("")
    report_lines.append("4. Model performance")
    report_lines.append("-" * 90)
    show_cols = ["target_name", "split", "n", "mae", "rmse", "r2", "pearson", "spearman"]
    report_lines.append(metrics_all[show_cols].to_string(index=False))
    report_lines.append("")
    report_lines.append("5. Best targets by test Pearson")
    report_lines.append("-" * 90)
    report_lines.append(test_metrics[show_cols].to_string(index=False))
    report_lines.append("")
    report_lines.append("6. Interpretation")
    report_lines.append("-" * 90)
    report_lines.append("High performance here means sample-level spatial architecture predicts broad above-prior or below-prior response behavior.")
    report_lines.append("This model complements the residual pair model. It does not replace the pair model.")
    report_lines.append("Because n is only 102 samples, results should be interpreted as screening-level and compared against per-treatment residual models later.")
    report_lines.append("")
    report_lines.append("7. Next recommended step")
    report_lines.append("-" * 90)
    report_lines.append("Review the strongest broad residual targets and cross-target SHAP features, then run filtered per-treatment residual models for treatments with sufficient sample count and residual variance.")

    report_path = output_root / "broad_residual_spatial_only_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "residual_run": str(residual_run),
        "derived_handoff": str(derived),
        "n_samples": int(sample_level["sample_id"].nunique()),
        "n_candidate_features": int(len(candidate_features)),
        "n_selected_features": int(len(selected_features)),
        "best_test_pearson_target": str(test_metrics.iloc[0]["target_name"]) if len(test_metrics) else "",
        "best_test_pearson": float(test_metrics.iloc[0]["pearson"]) if len(test_metrics) else None,
        "script": str(Path(__file__)),
    }
    (output_root / "broad_residual_spatial_only_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("=" * 90)
    print("SPATIAL-ONLY BROAD RESIDUAL MODEL COMPLETE")
    print("=" * 90)
    print("Output root:", output_root)
    print("Report:", report_path)
    print("Best target by test Pearson:", summary["best_test_pearson_target"])
    print("Best test Pearson:", summary["best_test_pearson"])
    print("")
    print("Generated folders:")
    for folder in [prepared_dir, model_dir, summary_dir, fig_dir]:
        print(" ", folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
