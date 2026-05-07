"""
Script: 07_train_filtered_per_treatment_residual_models.py

Purpose:
    Train filtered per-treatment residual models for
    spatial_prediction_model_V2.

Pipeline role:
    This step moves from broad residual screening to treatment-specific residual
    modeling. It filters eligible therapies, trains repeated per-treatment
    models, and writes screening and final-model evidence for curation.

Scientific role:
    Per-treatment residual models ask which spatial phenotypes are associated
    with response deviation for individual therapies or regimens. Filtering,
    repeated validation, and final feature evidence help separate plausible
    treatment-specific spatial biology from weak or under-supported screens.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP07_DOC_POLISH_V2

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    imports, constants, thresholds, hyperparameters, feature-selection
    rules, output filenames, and return codes must remain unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
# Use a non-interactive backend so per-treatment figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.io_utils import ensure_dir, read_table, write_json, write_table, write_text_report
from spm_v2.model_training import make_xgb_pipeline, select_features_training_only
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest
from spm_v2.validation import metric_safe


# =============================================================================
# Helper functions
# =============================================================================

def safe_float(value):
    """Convert a value to float while returning None for invalid or missing values."""

    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_name(text: str, max_len: int = 90) -> str:
    """Convert text into a filesystem-safe short name."""

    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len]
    return text or "unnamed"


def normalize_bool(series: pd.Series) -> pd.Series:
    """Convert a Series of common truthy values into booleans."""

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def add_treatment_stats(pair_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Summarize per-treatment support and residual target variability."""

    rows = []

    for drug_key, sub in pair_df.groupby("drug_key", dropna=False):
        y = pd.to_numeric(sub[target_col], errors="coerce")
        rows.append({
            "drug_key": str(drug_key),
            "n_rows": int(len(sub)),
            "n_samples": int(sub["sample_id"].astype(str).nunique()) if "sample_id" in sub.columns else int(len(sub)),
            "target_nonmissing": int(y.notna().sum()),
            "target_mean": safe_float(y.mean()),
            "target_std": safe_float(y.std()),
            "target_min": safe_float(y.min()),
            "target_max": safe_float(y.max()),
        })

    out = pd.DataFrame(rows)
    out["target_std"] = pd.to_numeric(out["target_std"], errors="coerce").fillna(0.0)
    return out


def load_eligible_treatments(pair_df: pd.DataFrame, eligibility_path: Path, target_col: str, min_samples: int, min_target_std: float) -> pd.DataFrame:
    """Combine data-driven and pipeline treatment eligibility filters."""

    stats = add_treatment_stats(pair_df, target_col)

    stats["eligible_by_data"] = (
        (stats["target_nonmissing"] >= min_samples)
        & (stats["n_samples"] >= min_samples)
        & (stats["target_std"] >= min_target_std)
    )

    if eligibility_path.exists():
        eligibility = read_table(eligibility_path)

        if not eligibility.empty:
            if "drug_key" not in eligibility.columns:
                for candidate in ["treatment", "treatment_name", "drug", "drug_label"]:
                    if candidate in eligibility.columns:
                        eligibility = eligibility.rename(columns={candidate: "drug_key"})
                        break

            if "drug_key" in eligibility.columns:
                keep_cols = ["drug_key"]

                for col in ["eligible", "n_samples", "target_std", "residual_std", "n_rows"]:
                    if col in eligibility.columns:
                        keep_cols.append(col)

                eligibility = eligibility[keep_cols].drop_duplicates("drug_key")
                stats = stats.merge(eligibility, on="drug_key", how="left", suffixes=("", "_eligibility"))

                if "eligible" in stats.columns:
                    stats["eligible_from_step02"] = normalize_bool(stats["eligible"])
                else:
                    stats["eligible_from_step02"] = True
            else:
                stats["eligible_from_step02"] = True
        else:
            stats["eligible_from_step02"] = True
    else:
        stats["eligible_from_step02"] = True

    stats["eligible"] = stats["eligible_by_data"] & stats["eligible_from_step02"].fillna(True)
    stats = stats.sort_values(["eligible", "n_samples", "target_std"], ascending=[False, False, False])
    return stats


def split_indices(n: int, test_size: float, random_state: int):
    """Create train/test index splits for residual-model evaluation."""

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    return train_idx, test_idx


def compute_shap_values(pipe, x_eval: pd.DataFrame, selected: list[str], max_rows: int, seed: int) -> pd.DataFrame:
    """Compute SHAP values for a fitted model, returning status metadata on failure."""

    try:
        import shap
    except Exception as exc:
        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": "not_available: " + str(exc),
        })

    try:
        if len(x_eval) > max_rows:
            rng = np.random.default_rng(seed)
            keep = rng.choice(np.arange(len(x_eval)), size=max_rows, replace=False)
            x_eval = x_eval.iloc[keep].copy()

        x_imp = pd.DataFrame(
            pipe.named_steps["imputer"].transform(x_eval[selected]),
            columns=selected,
        )

        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        values = explainer.shap_values(x_imp)

        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "shap_status": "success",
        })
    except Exception as exc:
        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": "failed: " + str(exc),
        })


def train_screen_for_treatment(
    sub: pd.DataFrame,
    drug_key: str,
    target_col: str,
    feature_cols: list[str],
    n_repeats: int,
    test_size: float,
    max_features_per_split: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated screening models for one treatment-specific residual target."""

    work = sub.copy()
    y_all = pd.to_numeric(work[target_col], errors="coerce")
    valid = y_all.notna()
    work = work.loc[valid].reset_index(drop=True)
    y_all = y_all.loc[valid].astype(float).reset_index(drop=True)

    x_all = work[feature_cols].copy()

    for col in feature_cols:
        x_all[col] = pd.to_numeric(x_all[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    metrics_rows = []
    pred_rows = []
    feature_rows = []

    for repeat in range(n_repeats):
        seed = random_state + repeat
        train_idx, test_idx = split_indices(len(work), test_size, seed)

        # Feature selection is fit on training data only to avoid test-set leakage.
        selected = select_features_training_only(
            x_train=x_all.iloc[train_idx],
            y_train=y_all.iloc[train_idx],
            feature_cols=feature_cols,
            max_features=min(max_features_per_split, len(feature_cols)),
            min_variance=1e-12,
        )

        pipe = make_xgb_pipeline(
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            n_jobs=1,
        )

        pipe.fit(x_all.iloc[train_idx][selected], y_all.iloc[train_idx])

        pred_test = pipe.predict(x_all.iloc[test_idx][selected])
        baseline_test = np.repeat(float(y_all.iloc[train_idx].mean()), len(test_idx))

        test_m = metric_safe(y_all.iloc[test_idx], pred_test)
        base_m = metric_safe(y_all.iloc[test_idx], baseline_test)

        metrics_rows.append({
            "drug_key": drug_key,
            "repeat": repeat,
            "random_state": seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_samples_total": int(len(work)),
            "target_std_total": safe_float(y_all.std()),
            "n_features_available": int(len(feature_cols)),
            "n_features_selected": int(len(selected)),
            "test_pearson": test_m["pearson"],
            "test_spearman": test_m["spearman"],
            "test_r2": test_m["r2"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "baseline_test_rmse": base_m["rmse"],
            "rmse_improvement_vs_baseline": base_m["rmse"] - test_m["rmse"],
            "mae_improvement_vs_baseline": base_m["mae"] - test_m["mae"],
        })

        pred_df = pd.DataFrame({
            "drug_key": drug_key,
            "repeat": repeat,
            "sample_id": work.iloc[test_idx]["sample_id"].astype(str).values if "sample_id" in work.columns else test_idx,
            "target": y_all.iloc[test_idx].values,
            "prediction": pred_test,
        })
        pred_df["prediction_error"] = pred_df["prediction"] - pred_df["target"]
        pred_rows.append(pred_df)

        importances = pipe.named_steps["model"].feature_importances_

        for feature, importance in zip(selected, importances):
            feature_rows.append({
                "drug_key": drug_key,
                "repeat": repeat,
                "feature_name": feature,
                "selected_in_repeat": True,
                "gain_importance": float(importance),
            })

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),
        pd.DataFrame(feature_rows),
    )


def summarize_screening(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated treatment-specific screening metrics."""

    rows = []

    for drug_key, sub in metrics.groupby("drug_key", dropna=False):
        row = {
            "drug_key": drug_key,
            "n_repeats": int(len(sub)),
            "n_samples_total": int(sub["n_samples_total"].median()),
            "target_std_total": safe_float(pd.to_numeric(sub["target_std_total"], errors="coerce").median()),
            "n_features_available": int(sub["n_features_available"].median()),
            "n_features_selected_median": int(sub["n_features_selected"].median()),
        }

        for col in [
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_mae",
            "test_rmse",
            "baseline_test_rmse",
            "rmse_improvement_vs_baseline",
            "mae_improvement_vs_baseline",
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = safe_float(vals.mean()) if len(vals.dropna()) else None
            row[f"{col}_median"] = safe_float(vals.median()) if len(vals.dropna()) else None
            row[f"{col}_std"] = safe_float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else None
            row[f"{col}_q025"] = safe_float(vals.quantile(0.025)) if len(vals.dropna()) else None
            row[f"{col}_q975"] = safe_float(vals.quantile(0.975)) if len(vals.dropna()) else None

        vals = pd.to_numeric(sub["test_pearson"], errors="coerce")
        row["test_pearson_positive_fraction"] = safe_float((vals > 0).mean()) if len(vals.dropna()) else None

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["test_pearson_mean", "test_r2_mean", "rmse_improvement_vs_baseline_mean"],
            ascending=False,
        )

    return out


def train_final_shap_models(
    pair_df: pd.DataFrame,
    selected_summary: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    registry: pd.DataFrame,
    max_features_final: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    max_shap_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit final selected treatment models and collect SHAP/gain feature evidence."""

    manifest_rows = []
    shap_rows = []
    final_top_rows = []

    keep_cols = ["feature_name"]

    for col in [
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]:
        if col in registry.columns:
            keep_cols.append(col)

    registry_meta = registry[keep_cols].drop_duplicates("feature_name")

    for i, row in selected_summary.reset_index(drop=True).iterrows():
        drug_key = str(row["drug_key"])
        sub = pair_df[pair_df["drug_key"].astype(str) == drug_key].copy()

        y = pd.to_numeric(sub[target_col], errors="coerce")
        valid = y.notna()
        sub = sub.loc[valid].reset_index(drop=True)
        y = y.loc[valid].astype(float).reset_index(drop=True)

        x = sub[feature_cols].copy()

        for col in feature_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        seed = random_state + i + 10000

        selected_features = select_features_training_only(
            x_train=x,
            y_train=y,
            feature_cols=feature_cols,
            max_features=min(max_features_final, len(feature_cols)),
            min_variance=1e-12,
        )

        pipe = make_xgb_pipeline(
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            n_jobs=1,
        )

        pipe.fit(x[selected_features], y)

        shap_df = compute_shap_values(
            pipe=pipe,
            x_eval=x[selected_features],
            selected=selected_features,
            max_rows=max_shap_rows,
            seed=seed,
        )

        shap_success = bool((shap_df["shap_status"].astype(str) == "success").any())

        gain_df = pd.DataFrame({
            "feature_name": selected_features,
            "gain_importance": pipe.named_steps["model"].feature_importances_,
        })

        final_evidence = gain_df.merge(shap_df, on="feature_name", how="left")
        final_evidence["drug_key"] = drug_key
        final_evidence["final_model_rank"] = i + 1
        final_evidence = final_evidence.merge(registry_meta, on="feature_name", how="left")
        final_evidence["mean_abs_shap"] = pd.to_numeric(final_evidence["mean_abs_shap"], errors="coerce")
        final_evidence["gain_importance"] = pd.to_numeric(final_evidence["gain_importance"], errors="coerce")
        final_evidence = final_evidence.sort_values(
            ["mean_abs_shap", "gain_importance"],
            ascending=False,
            na_position="last",
        )

        shap_rows.append(final_evidence)

        top = final_evidence.head(10).copy()
        top["top_feature_rank_within_treatment"] = np.arange(1, len(top) + 1)
        final_top_rows.append(top)

        manifest = row.to_dict()
        manifest.update({
            "final_model_rank": i + 1,
            "drug_key": drug_key,
            "n_rows_final_model": int(len(sub)),
            "n_features_final_model": int(len(selected_features)),
            "shap_status": "success" if shap_success else "failed_or_not_available",
            "uses_step05_v2_registry": "yes",
        })
        manifest_rows.append(manifest)

    manifest_df = pd.DataFrame(manifest_rows)
    shap_long = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    final_top = pd.concat(final_top_rows, ignore_index=True) if final_top_rows else pd.DataFrame()

    return manifest_df, shap_long, final_top


def summarize_recurrent_features(shap_long: pd.DataFrame) -> pd.DataFrame:
    """Summarize spatial features recurring across curated treatment models."""

    if shap_long.empty:
        return pd.DataFrame()

    score_col = "mean_abs_shap" if "mean_abs_shap" in shap_long.columns and shap_long["mean_abs_shap"].notna().sum() > 0 else "gain_importance"

    out = (
        shap_long
        .groupby("feature_name", dropna=False)
        .agg(
            treatment_count=("drug_key", lambda s: int(s.nunique())),
            total_score=(score_col, "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
            mean_gain_importance=("gain_importance", "mean"),
        )
        .reset_index()
        .sort_values(["treatment_count", "total_score"], ascending=False)
    )

    meta_cols = ["feature_name"]

    for col in [
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]:
        if col in shap_long.columns:
            meta_cols.append(col)

    meta = shap_long[meta_cols].drop_duplicates("feature_name")
    out = out.merge(meta, on="feature_name", how="left")

    return out


def summarize_recurrent_themes(shap_long: pd.DataFrame) -> pd.DataFrame:
    """Summarize biological themes recurring across curated treatment models."""

    if shap_long.empty or "biological_theme" not in shap_long.columns:
        return pd.DataFrame()

    score_col = "mean_abs_shap" if "mean_abs_shap" in shap_long.columns and shap_long["mean_abs_shap"].notna().sum() > 0 else "gain_importance"

    out = (
        shap_long
        .groupby("biological_theme", dropna=False)
        .agg(
            n_features=("feature_name", lambda s: int(s.nunique())),
            n_treatments=("drug_key", lambda s: int(s.nunique())),
            total_score=(score_col, "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
        )
        .reset_index()
        .sort_values("total_score", ascending=False)
    )

    examples = []

    for theme in out["biological_theme"].astype(str).tolist():
        sub = shap_long[shap_long["biological_theme"].astype(str) == theme].copy()
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0)
        sub = sub.sort_values(score_col, ascending=False)
        names = sub.head(4).get("feature_original", sub["feature_name"]).astype(str).tolist()
        examples.append("; ".join(names))

    out["example_features"] = examples
    return out


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, output_path: Path, title: str, xlabel: str, top_n: int = 30) -> None:
    """Save a ranked horizontal bar plot for model, feature, or theme evidence."""

    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    plot = df.copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)

    if plot.empty:
        return

    labels = plot[label_col].astype(str).tolist()
    labels = [x if len(x) <= 76 else x[:73] + "..." for x in labels]

    y = np.arange(len(plot))[::-1]
    values = plot[value_col].to_numpy()[::-1]
    labels = labels[::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(6, len(plot) * 0.34)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Run this spatial_prediction_model_V2 step and write tables, figures, reports, and provenance."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--step05-root", required=True)
    parser.add_argument("--step06-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--target-col", default="fused_residual_vs_prior")
    parser.add_argument("--max-treatments", type=int, default=60)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-target-std", type=float, default=1e-8)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-features-per-split", type=int, default=60)
    parser.add_argument("--max-features-final", type=int, default=60)
    parser.add_argument("--max-final-shap-models", type=int, default=30)
    parser.add_argument("--min-final-pearson", type=float, default=0.20)
    parser.add_argument("--min-final-positive-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-shap-rows", type=int, default=1000)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    step05_root = Path(args.step05_root)
    step06_root = Path(args.step06_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_screening_metrics")
    d03 = ensure_dir(output_root / "03_final_models")
    d04 = ensure_dir(output_root / "04_shap_feature_evidence")
    d05 = ensure_dir(output_root / "05_recurrent_features_and_themes")
    d06 = ensure_dir(output_root / "06_figures")
    d07 = ensure_dir(output_root / "07_reports")

    pair_path = dataset_root / "03_modeling_datasets" / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"
    eligibility_path = dataset_root / "03_modeling_datasets" / "v2_treatment_eligibility.tsv"
    registry_path = step05_root / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv"
    step05_summary_path = step05_root / "v2_step05_residual_biology_registry_summary.json"
    step06_summary_path = step06_root / "v2_step06_broad_residual_model_summary.json"

    pair_df = read_table(pair_path)
    registry = read_table(registry_path)

    if pair_df.empty:
        raise FileNotFoundError(f"Pair dataset missing or empty: {pair_path}")

    if registry.empty:
        raise FileNotFoundError(f"Step 05 registry missing or empty: {registry_path}")

    required_cols = ["sample_id", "drug_key", args.target_col]

    for col in required_cols:
        if col not in pair_df.columns:
            raise ValueError(f"Required pair dataset column missing: {col}")

    if "feature_name" not in registry.columns:
        raise ValueError("Step 05 registry must contain feature_name.")

    feature_cols = [str(x) for x in registry["feature_name"].astype(str).tolist() if str(x) in pair_df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    if len(feature_cols) < 10:
        raise ValueError(f"Too few Step 05 registry features are present in pair dataset: {len(feature_cols)}")

    pair_df["drug_key"] = pair_df["drug_key"].astype(str)
    pair_df["sample_id"] = pair_df["sample_id"].astype(str)

    # Treatment eligibility protects per-treatment models from under-supported residual labels.
    treatment_stats = load_eligible_treatments(
        pair_df=pair_df,
        eligibility_path=eligibility_path,
        target_col=args.target_col,
        min_samples=args.min_samples,
        min_target_std=args.min_target_std,
    )

    eligible = treatment_stats[treatment_stats["eligible"] == True].copy()
    eligible = eligible.sort_values(["n_samples", "target_std"], ascending=False)

    if args.mode == "smoke":
        screened_treatments = eligible.head(args.max_treatments)["drug_key"].astype(str).tolist()
    else:
        screened_treatments = eligible["drug_key"].astype(str).tolist()

    if len(screened_treatments) == 0:
        raise ValueError("No eligible treatments found for Step 07.")

    source_manifest = pd.DataFrame({
        "source_name": [
            "run_root",
            "pair_level_dataset",
            "treatment_eligibility",
            "step05_strict_biology_registry",
            "step05_summary",
            "step06_summary",
        ],
        "path": [
            str(run_root),
            str(pair_path),
            str(eligibility_path),
            str(registry_path),
            str(step05_summary_path),
            str(step06_summary_path),
        ],
        "exists": [
            run_root.exists(),
            pair_path.exists(),
            eligibility_path.exists(),
            registry_path.exists(),
            step05_summary_path.exists(),
            step06_summary_path.exists(),
        ],
    })

    write_table(source_manifest, d01 / "source_manifest.tsv")
    write_table(treatment_stats, d01 / "treatment_eligibility_and_data_stats.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), d01 / "step05_registry_features_used.tsv")
    write_table(pd.DataFrame({"drug_key": screened_treatments}), d01 / "screened_treatments.tsv")

    all_metrics = []
    all_predictions = []
    all_screen_feature_evidence = []

    for i, drug_key in enumerate(screened_treatments, start=1):
        sub = pair_df[pair_df["drug_key"].astype(str) == drug_key].copy()

        metrics, predictions, feature_evidence = train_screen_for_treatment(
            sub=sub,
            drug_key=drug_key,
            target_col=args.target_col,
            feature_cols=feature_cols,
            n_repeats=args.n_repeats,
            test_size=args.test_size,
            max_features_per_split=args.max_features_per_split,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state + i * 100,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_screen_feature_evidence.append(feature_evidence)

    metrics_long = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    screen_feature_evidence = pd.concat(all_screen_feature_evidence, ignore_index=True) if all_screen_feature_evidence else pd.DataFrame()

    screening_summary = summarize_screening(metrics_long)

    screening_summary["selected_for_final_shap"] = (
        (pd.to_numeric(screening_summary["test_pearson_mean"], errors="coerce") >= args.min_final_pearson)
        & (pd.to_numeric(screening_summary["test_pearson_positive_fraction"], errors="coerce") >= args.min_final_positive_fraction)
        & (pd.to_numeric(screening_summary["rmse_improvement_vs_baseline_mean"], errors="coerce") > 0)
    )

    selected_summary = screening_summary[screening_summary["selected_for_final_shap"] == True].copy()
    selected_summary = selected_summary.sort_values(
        ["test_pearson_mean", "test_r2_mean", "rmse_improvement_vs_baseline_mean"],
        ascending=False,
    ).head(args.max_final_shap_models)

    selection_fallback_used = False

    # Fallback keeps smoke runs inspectable even when strict final-model thresholds select nothing.
    if selected_summary.empty and not screening_summary.empty:
        selected_summary = screening_summary.sort_values(
            ["test_pearson_mean", "test_r2_mean", "rmse_improvement_vs_baseline_mean"],
            ascending=False,
        ).head(min(5, len(screening_summary))).copy()
        selected_summary["selected_for_final_shap"] = True
        selected_summary["selection_fallback_reason"] = "no_treatment_met_thresholds"
        selection_fallback_used = True
    else:
        selected_summary["selection_fallback_reason"] = ""

    write_table(metrics_long, d02 / "per_treatment_repeated_split_metrics_long.tsv")
    write_table(screening_summary, d02 / "per_treatment_screening_summary.tsv")
    write_table(predictions_long, d02 / "per_treatment_test_predictions_long.tsv")
    write_table(screen_feature_evidence, d02 / "per_treatment_screening_feature_evidence_long.tsv")

    # Final SHAP models are fit only for selected treatments to limit interpretation to supported screens.
    final_manifest, shap_long, final_top = train_final_shap_models(
        pair_df=pair_df,
        selected_summary=selected_summary,
        target_col=args.target_col,
        feature_cols=feature_cols,
        registry=registry,
        max_features_final=args.max_features_final,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        max_shap_rows=args.max_shap_rows,
    )

    recurrent_features = summarize_recurrent_features(shap_long)
    recurrent_themes = summarize_recurrent_themes(shap_long)

    write_table(final_manifest, d03 / "final_model_manifest.tsv")
    write_table(selected_summary, d03 / "selected_treatments_for_final_shap.tsv")
    write_table(shap_long, d04 / "per_treatment_final_shap_feature_long.tsv")
    write_table(final_top, d04 / "per_treatment_final_top10_features.tsv")
    write_table(recurrent_features, d05 / "per_treatment_recurrent_spatial_features.tsv")
    write_table(recurrent_themes, d05 / "per_treatment_recurrent_biology_themes.tsv")

    save_bar(
        screening_summary,
        "drug_key",
        "test_pearson_mean",
        d06 / "fig_01_top_screened_per_treatment_models.png",
        "V2 Step 07 screened per treatment residual models",
        "Mean test Pearson across repeated splits",
        top_n=30,
    )

    save_bar(
        recurrent_features,
        "feature_name",
        "total_score",
        d06 / "fig_02_recurrent_per_treatment_spatial_features.png",
        "V2 Step 07 recurrent per treatment spatial features",
        "Total SHAP score across final models",
        top_n=30,
    )

    save_bar(
        recurrent_themes,
        "biological_theme",
        "total_score",
        d06 / "fig_03_recurrent_per_treatment_biology_themes.png",
        "V2 Step 07 recurrent per treatment biology themes",
        "Total SHAP score across final models",
        top_n=15,
    )

    if not screening_summary.empty:
        plt.figure(figsize=(10, 6))
        vals = pd.to_numeric(screening_summary["test_pearson_mean"], errors="coerce").dropna()
        plt.hist(vals, bins=min(20, max(5, len(vals) // 2)))
        plt.xlabel("Mean test Pearson")
        plt.ylabel("Treatment count")
        plt.title("V2 Step 07 per treatment model performance distribution")
        plt.tight_layout()
        plt.savefig(d06 / "fig_04_per_treatment_performance_distribution.png", dpi=240, bbox_inches="tight")
        plt.close()

    best = screening_summary.head(1)

    best_treatment = str(best.iloc[0]["drug_key"]) if not best.empty else ""
    best_pearson = safe_float(best.iloc[0].get("test_pearson_mean")) if not best.empty else None
    best_r2 = safe_float(best.iloc[0].get("test_r2_mean")) if not best.empty else None

    shap_success_count = 0

    if not final_manifest.empty and "shap_status" in final_manifest.columns:
        shap_success_count = int((final_manifest["shap_status"].astype(str) == "success").sum())

    run_summary = {
        "status": "pass",
        "official_step": "07_train_filtered_per_treatment_residual_models",
        "mode": args.mode,
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "step05_root": str(step05_root),
        "step06_root": str(step06_root),
        "output_root": str(output_root),
        "target_col": args.target_col,
        "pair_dataset": str(pair_path),
        "step05_strict_biology_registry": str(registry_path),
        "n_pair_rows": int(len(pair_df)),
        "n_total_treatments": int(pair_df["drug_key"].astype(str).nunique()),
        "n_eligible_treatments": int(len(eligible)),
        "n_screened_treatments": int(len(screened_treatments)),
        "n_repeats": int(args.n_repeats),
        "n_step05_registry_features": int(len(registry)),
        "n_features_used": int(len(feature_cols)),
        "n_selected_final_models": int(len(final_manifest)),
        "n_shap_success_models": int(shap_success_count),
        "best_screened_treatment": best_treatment,
        "best_screened_treatment_test_pearson_mean": best_pearson,
        "best_screened_treatment_test_r2_mean": best_r2,
        "selection_fallback_used": bool(selection_fallback_used),
        "production_dependency_on_v1_outputs": "no",
        "uses_step05_v2_registry": "yes",
        "uses_treatment_identity_features": "no",
    }

    write_json(run_summary, output_root / "v2_step07_filtered_per_treatment_residual_models_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 07 FILTERED PER TREATMENT RESIDUAL MODELS REPORT")
    report_lines.append("")

    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("1. Source files")
    report_lines.append(source_manifest.to_string(index=False))

    report_lines.append("")
    report_lines.append("2. Treatment eligibility and screened treatment policy")
    report_lines.append(f"minimum samples: {args.min_samples}")
    report_lines.append(f"minimum target std: {args.min_target_std}")
    report_lines.append(f"maximum treatments in smoke mode: {args.max_treatments}")
    report_lines.append(f"screened treatments: {len(screened_treatments)}")

    report_lines.append("")
    report_lines.append("3. Screening performance summary")
    report_lines.append(screening_summary.head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("4. Final SHAP model manifest")
    if final_manifest.empty:
        report_lines.append("No final SHAP models selected.")
    else:
        report_lines.append(final_manifest.head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("5. Recurrent spatial features across final treatment models")
    if recurrent_features.empty:
        report_lines.append("No recurrent feature evidence generated.")
    else:
        report_lines.append(recurrent_features.head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("6. Recurrent biology themes across final treatment models")
    if recurrent_themes.empty:
        report_lines.append("No recurrent theme evidence generated.")
    else:
        report_lines.append(recurrent_themes.to_string(index=False))

    report_lines.append("")
    report_lines.append("7. Interpretation")
    report_lines.append("Step 07 is the treatment specific residual discovery module.")
    report_lines.append("Each model is trained within one treatment only.")
    report_lines.append("No treatment identity features are used.")
    report_lines.append("All features come from the V2 Step 05 strict biology registry.")
    report_lines.append("Step 08 will curate these screening results into Tier 1, Tier 2, and caution categories.")

    report_path = write_text_report(d07 / "v2_step07_filtered_per_treatment_residual_models_report.txt", "\n".join(report_lines))

    slide_lines = []
    slide_lines.append("V2 STEP 07 FILTERED PER TREATMENT RESIDUAL MODELS SLIDE NOTES")
    slide_lines.append("")
    slide_lines.append(f"Screened treatments: {len(screened_treatments)}")
    slide_lines.append(f"Selected final SHAP models: {len(final_manifest)}")
    slide_lines.append(f"SHAP success models: {shap_success_count}")
    slide_lines.append(f"Best screened treatment: {best_treatment}")
    slide_lines.append(f"Best screened treatment mean test Pearson: {best_pearson}")
    slide_lines.append(f"Best screened treatment mean test R2: {best_r2}")
    slide_lines.append("")
    slide_lines.append("Top recurrent biology themes:")
    if recurrent_themes.empty:
        slide_lines.append("No recurrent themes generated.")
    else:
        for theme in recurrent_themes.head(8)["biological_theme"].astype(str).tolist():
            slide_lines.append(theme)

    write_text_report(d07 / "v2_step07_filtered_per_treatment_residual_slide_notes.txt", "\n".join(slide_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Dataset root: {dataset_root}",
        f"Step 05 root: {step05_root}",
        f"Step 06 root: {step06_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Pair rows: {len(pair_df)}",
        f"Eligible treatments: {len(eligible)}",
        f"Screened treatments: {len(screened_treatments)}",
        f"Repeats per treatment: {args.n_repeats}",
        f"Step 05 registry features: {len(registry)}",
        f"Features used: {len(feature_cols)}",
        f"Selected final SHAP models: {len(final_manifest)}",
        f"SHAP success models: {shap_success_count}",
        f"Best screened treatment: {best_treatment}",
        f"Best screened treatment mean test Pearson: {best_pearson}",
        f"Best screened treatment mean test R2: {best_r2}",
        "Treatment identity features used: no",
        "Production dependency on V1 outputs: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 07 FILTERED PER TREATMENT RESIDUAL MODELS COMPLETE", terminal_lines))
    print("")

    print("Top screened per treatment models")
    show_cols = [
        "drug_key",
        "n_repeats",
        "n_samples_total",
        "target_std_total",
        "test_pearson_mean",
        "test_pearson_median",
        "test_r2_mean",
        "rmse_improvement_vs_baseline_mean",
        "test_pearson_positive_fraction",
        "selected_for_final_shap",
    ]
    show_cols = [c for c in show_cols if c in screening_summary.columns]
    print(screening_summary[show_cols].head(40).to_string(index=False))
    print("")

    if not recurrent_features.empty:
        feature_cols_show = [
            "feature_name",
            "feature_original",
            "biological_theme",
            "treatment_count",
            "total_score",
            "mean_score",
            "max_score",
        ]
        feature_cols_show = [c for c in feature_cols_show if c in recurrent_features.columns]
        print("Top recurrent spatial features")
        print(recurrent_features[feature_cols_show].head(30).to_string(index=False))
        print("")

    if not recurrent_themes.empty:
        print("Recurrent biology themes")
        print(recurrent_themes.head(15).to_string(index=False))
        print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
