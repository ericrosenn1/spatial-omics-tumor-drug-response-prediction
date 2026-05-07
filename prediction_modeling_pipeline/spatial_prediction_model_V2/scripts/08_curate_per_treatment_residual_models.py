"""
Script: 08_curate_per_treatment_residual_models.py

Purpose:
    Curate Step 08 per-treatment residual model evidence.

Pipeline role:
    This step summarizes Step 07 per-treatment residual models into
    interpretation tiers, recurrent feature summaries, recurrent biology themes,
    and a Step 09 label-shuffle handoff.

Scientific role:
    Curation is the checkpoint between treatment-specific model training and
    biological claims. Tier assignment marks which screens are ready for label-
    shuffle validation and which should remain exploratory or cautionary.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP08_DOC_POLISH_V2

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
import sys
from pathlib import Path

import matplotlib
# Use a non-interactive backend so curation figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.io_utils import ensure_dir, read_table, write_json, write_table, write_text_report
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest


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


def as_bool(value) -> bool:
    """Coerce a value into a boolean using common truthy strings."""

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Return a numeric Series for a column, or a default-valued Series when absent."""

    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def find_score_col(df: pd.DataFrame) -> str:
    """Identify the numeric evidence score column used for feature ranking."""

    for col in ["mean_abs_shap", "total_score", "mean_score", "gain_importance", "mean_gain_importance"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            return col
    return "gain_importance"


def tier_reason(row: pd.Series) -> str:
    """Build a reviewer-facing explanation for one treatment model tier assignment."""

    reasons = []

    if not as_bool(row.get("selected_for_final_shap", False)):
        reasons.append("not selected for final SHAP model")

    if str(row.get("shap_status", "")).lower() != "success":
        reasons.append("final SHAP model did not report success")

    if safe_float(row.get("test_pearson_mean")) is not None:
        reasons.append(f"mean Pearson {float(row.get('test_pearson_mean')):.3f}")

    if safe_float(row.get("test_r2_mean")) is not None:
        reasons.append(f"mean R2 {float(row.get('test_r2_mean')):.3f}")

    if safe_float(row.get("rmse_improvement_vs_baseline_mean")) is not None:
        reasons.append(f"mean RMSE improvement {float(row.get('rmse_improvement_vs_baseline_mean')):.4g}")

    if safe_float(row.get("test_pearson_positive_fraction")) is not None:
        reasons.append(f"positive split fraction {float(row.get('test_pearson_positive_fraction')):.3f}")

    return "; ".join(reasons)


def assign_tiers(
    merged: pd.DataFrame,
    tier1_min_pearson: float,
    tier1_min_positive_fraction: float,
    tier2_min_pearson: float,
    tier2_min_positive_fraction: float,
) -> pd.DataFrame:
    """Assign interpretation tiers to treatment-specific residual model screens."""

    out = merged.copy()

    for col in [
        "test_pearson_mean",
        "test_r2_mean",
        "rmse_improvement_vs_baseline_mean",
        "test_pearson_positive_fraction",
    ]:
        out[col] = numeric(out, col)

    if "selected_for_final_shap" not in out.columns:
        out["selected_for_final_shap"] = False

    if "shap_status" not in out.columns:
        out["shap_status"] = ""

    selected = out["selected_for_final_shap"].apply(as_bool)
    shap_success = out["shap_status"].astype(str).str.lower().eq("success")

    tier1 = (
        selected
        & shap_success
        & (out["test_pearson_mean"] >= tier1_min_pearson)
        & (out["test_pearson_positive_fraction"] >= tier1_min_positive_fraction)
        & (out["rmse_improvement_vs_baseline_mean"] > 0)
        & (out["test_r2_mean"] > 0)
    )

    tier2 = (
        selected
        & shap_success
        & ~tier1
        & (out["test_pearson_mean"] >= tier2_min_pearson)
        & (out["test_pearson_positive_fraction"] >= tier2_min_positive_fraction)
        & (out["rmse_improvement_vs_baseline_mean"] > 0)
    )

    tier3 = selected & shap_success & ~tier1 & ~tier2

    out["interpretation_tier"] = "not_selected_or_not_successful"
    out.loc[tier3, "interpretation_tier"] = "tier3_caution_screen"
    out.loc[tier2, "interpretation_tier"] = "tier2_screening_signal"
    out.loc[tier1, "interpretation_tier"] = "tier1_high_confidence_screen"

    out["ready_for_label_shuffle_validation"] = out["interpretation_tier"].eq("tier1_high_confidence_screen")
    out["interpretation_tier_reason"] = out.apply(tier_reason, axis=1)

    sort_cols = [
        "ready_for_label_shuffle_validation",
        "test_pearson_mean",
        "test_r2_mean",
        "rmse_improvement_vs_baseline_mean",
    ]

    out = out.sort_values(sort_cols, ascending=[False, False, False, False])
    return out


def merge_screen_and_manifest(screening: pd.DataFrame, final_manifest: pd.DataFrame) -> pd.DataFrame:
    """Merge treatment screening results with the final-model manifest."""

    screen = screening.copy()
    manifest = final_manifest.copy()

    if manifest.empty:
        screen["final_model_rank"] = ""
        screen["shap_status"] = ""
        screen["n_features_final_model"] = ""
        return screen

    keep_cols = ["drug_key"]

    for col in [
        "final_model_rank",
        "shap_status",
        "n_features_final_model",
        "n_rows_final_model",
        "uses_step05_v2_registry",
    ]:
        if col in manifest.columns:
            keep_cols.append(col)

    manifest = manifest[keep_cols].drop_duplicates("drug_key")
    merged = screen.merge(manifest, on="drug_key", how="left")
    return merged


def add_treatment_metadata(df: pd.DataFrame, treatment_stats: pd.DataFrame) -> pd.DataFrame:
    """Attach treatment support and residual target metadata to curated models."""

    if treatment_stats.empty or "drug_key" not in treatment_stats.columns:
        return df

    keep_cols = ["drug_key"]

    for col in [
        "n_rows",
        "n_samples",
        "target_nonmissing",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "eligible",
    ]:
        if col in treatment_stats.columns:
            keep_cols.append(col)

    meta = treatment_stats[keep_cols].drop_duplicates("drug_key")
    return df.merge(meta, on="drug_key", how="left", suffixes=("", "_eligibility"))


def summarize_tiers(curated: pd.DataFrame) -> pd.DataFrame:
    """Summarize treatment counts and metrics by interpretation tier."""

    if curated.empty:
        return pd.DataFrame(columns=["interpretation_tier", "treatment_count"])

    return (
        curated
        .groupby("interpretation_tier", dropna=False)
        .agg(
            treatment_count=("drug_key", "count"),
            mean_test_pearson=("test_pearson_mean", "mean"),
            median_test_pearson=("test_pearson_mean", "median"),
            mean_test_r2=("test_r2_mean", "mean"),
            mean_rmse_improvement=("rmse_improvement_vs_baseline_mean", "mean"),
        )
        .reset_index()
        .sort_values("treatment_count", ascending=False)
    )


def filter_shap_by_curated(shap_long: pd.DataFrame, curated: pd.DataFrame, tiers: set[str]) -> pd.DataFrame:
    """Restrict SHAP evidence to treatments in selected interpretation tiers."""

    if shap_long.empty or curated.empty:
        return pd.DataFrame()

    keep = curated[curated["interpretation_tier"].isin(tiers)]["drug_key"].astype(str).tolist()
    out = shap_long[shap_long["drug_key"].astype(str).isin(keep)].copy()

    tier_map = curated.set_index("drug_key")["interpretation_tier"].to_dict()
    out["interpretation_tier"] = out["drug_key"].map(tier_map)
    return out


def summarize_recurrent_features(shap_subset: pd.DataFrame) -> pd.DataFrame:
    """Summarize spatial features recurring across curated treatment models."""

    if shap_subset.empty:
        return pd.DataFrame()

    score_col = find_score_col(shap_subset)

    shap_subset = shap_subset.copy()
    shap_subset[score_col] = pd.to_numeric(shap_subset[score_col], errors="coerce").fillna(0.0)

    out = (
        shap_subset
        .groupby("feature_name", dropna=False)
        .agg(
            treatment_count=("drug_key", lambda s: int(s.nunique())),
            tier1_treatment_count=("interpretation_tier", lambda s: int((s.astype(str) == "tier1_high_confidence_screen").sum())),
            tier2_treatment_count=("interpretation_tier", lambda s: int((s.astype(str) == "tier2_screening_signal").sum())),
            total_score=(score_col, "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
        )
        .reset_index()
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
        if col in shap_subset.columns:
            meta_cols.append(col)

    meta = shap_subset[meta_cols].drop_duplicates("feature_name")
    out = out.merge(meta, on="feature_name", how="left")

    top_treatments = []

    for feature in out["feature_name"].astype(str).tolist():
        sub = shap_subset[shap_subset["feature_name"].astype(str) == feature].copy()
        sub = sub.sort_values(score_col, ascending=False)
        top_treatments.append("; ".join(sub["drug_key"].astype(str).drop_duplicates().head(5).tolist()))

    out["example_treatments"] = top_treatments
    out = out.sort_values(["treatment_count", "total_score"], ascending=False)
    return out


def summarize_recurrent_themes(shap_subset: pd.DataFrame) -> pd.DataFrame:
    """Summarize biological themes recurring across curated treatment models."""

    if shap_subset.empty or "biological_theme" not in shap_subset.columns:
        return pd.DataFrame()

    score_col = find_score_col(shap_subset)
    shap_subset = shap_subset.copy()
    shap_subset[score_col] = pd.to_numeric(shap_subset[score_col], errors="coerce").fillna(0.0)

    out = (
        shap_subset
        .groupby("biological_theme", dropna=False)
        .agg(
            n_features=("feature_name", lambda s: int(s.nunique())),
            n_treatments=("drug_key", lambda s: int(s.nunique())),
            tier1_row_count=("interpretation_tier", lambda s: int((s.astype(str) == "tier1_high_confidence_screen").sum())),
            tier2_row_count=("interpretation_tier", lambda s: int((s.astype(str) == "tier2_screening_signal").sum())),
            total_score=(score_col, "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
        )
        .reset_index()
        .sort_values("total_score", ascending=False)
    )

    examples = []

    for theme in out["biological_theme"].astype(str).tolist():
        sub = shap_subset[shap_subset["biological_theme"].astype(str) == theme].copy()
        sub = sub.sort_values(score_col, ascending=False)
        names = sub.get("feature_original", sub["feature_name"]).astype(str).drop_duplicates().head(4).tolist()
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
    labels = [x if len(x) <= 78 else x[:75] + "..." for x in labels]
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


def save_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Save a scatter plot for curated treatment-model performance."""

    if df.empty:
        return

    x = pd.to_numeric(df.get("test_pearson_mean"), errors="coerce")
    y = pd.to_numeric(df.get("test_r2_mean"), errors="coerce")

    mask = x.notna() & y.notna()

    if mask.sum() == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.scatter(x[mask], y[mask])
    plt.xlabel("Mean test Pearson")
    plt.ylabel("Mean test R2")
    plt.title("V2 Step 08 curated treatment model Pearson versus R2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_hist(df: pd.DataFrame, output_path: Path) -> None:
    """Save a histogram for treatment-model performance."""

    vals = pd.to_numeric(df.get("test_pearson_mean"), errors="coerce").dropna()

    if len(vals) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.hist(vals, bins=min(24, max(6, len(vals) // 2)))
    plt.xlabel("Mean test Pearson")
    plt.ylabel("Treatment count")
    plt.title("V2 Step 08 per treatment screening performance distribution")
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
    parser.add_argument("--step07-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tier1-min-pearson", type=float, default=0.60)
    parser.add_argument("--tier1-min-positive-fraction", type=float, default=0.875)
    parser.add_argument("--tier2-min-pearson", type=float, default=0.40)
    parser.add_argument("--tier2-min-positive-fraction", type=float, default=0.60)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    step05_root = Path(args.step05_root)
    step06_root = Path(args.step06_root)
    step07_root = Path(args.step07_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_curated_treatment_models")
    d03 = ensure_dir(output_root / "03_recurrent_spatial_features")
    d04 = ensure_dir(output_root / "04_recurrent_biology_themes")
    d05 = ensure_dir(output_root / "05_figures")
    d06 = ensure_dir(output_root / "06_reports")
    d07 = ensure_dir(output_root / "07_label_shuffle_handoff")

    screening_path = step07_root / "02_screening_metrics" / "per_treatment_screening_summary.tsv"
    final_manifest_path = step07_root / "03_final_models" / "final_model_manifest.tsv"
    selected_final_path = step07_root / "03_final_models" / "selected_treatments_for_final_shap.tsv"
    shap_long_path = step07_root / "04_shap_feature_evidence" / "per_treatment_final_shap_feature_long.tsv"
    top10_path = step07_root / "04_shap_feature_evidence" / "per_treatment_final_top10_features.tsv"
    treatment_stats_path = step07_root / "01_inputs" / "treatment_eligibility_and_data_stats.tsv"
    step05_registry_path = step05_root / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv"
    step07_summary_path = step07_root / "v2_step07_filtered_per_treatment_residual_models_summary.json"

    screening = read_table(screening_path)
    final_manifest = read_table(final_manifest_path)
    selected_final = read_table(selected_final_path)
    shap_long = read_table(shap_long_path)
    top10 = read_table(top10_path)
    treatment_stats = read_table(treatment_stats_path)
    registry = read_table(step05_registry_path)

    if screening.empty:
        raise FileNotFoundError(f"Step 07 screening summary missing or empty: {screening_path}")

    if final_manifest.empty:
        raise FileNotFoundError(f"Step 07 final model manifest missing or empty: {final_manifest_path}")

    if shap_long.empty:
        raise FileNotFoundError(f"Step 07 SHAP long table missing or empty: {shap_long_path}")

    source_manifest = pd.DataFrame({
        "source_name": [
            "run_root",
            "dataset_root",
            "step05_root",
            "step06_root",
            "step07_root",
            "step07_screening_summary",
            "step07_final_manifest",
            "step07_selected_final",
            "step07_shap_long",
            "step07_top10_features",
            "treatment_eligibility_stats",
            "step05_registry",
            "step07_summary",
        ],
        "path": [
            str(run_root),
            str(dataset_root),
            str(step05_root),
            str(step06_root),
            str(step07_root),
            str(screening_path),
            str(final_manifest_path),
            str(selected_final_path),
            str(shap_long_path),
            str(top10_path),
            str(treatment_stats_path),
            str(step05_registry_path),
            str(step07_summary_path),
        ],
        "exists": [
            run_root.exists(),
            dataset_root.exists(),
            step05_root.exists(),
            step06_root.exists(),
            step07_root.exists(),
            screening_path.exists(),
            final_manifest_path.exists(),
            selected_final_path.exists(),
            shap_long_path.exists(),
            top10_path.exists(),
            treatment_stats_path.exists(),
            step05_registry_path.exists(),
            step07_summary_path.exists(),
        ],
    })

    write_table(source_manifest, d01 / "source_manifest.tsv")

    merged = merge_screen_and_manifest(screening, final_manifest)
    merged = add_treatment_metadata(merged, treatment_stats)

    # Tier assignment separates high-confidence screens from exploratory or unsupported treatment models.
    curated = assign_tiers(
        merged,
        tier1_min_pearson=args.tier1_min_pearson,
        tier1_min_positive_fraction=args.tier1_min_positive_fraction,
        tier2_min_pearson=args.tier2_min_pearson,
        tier2_min_positive_fraction=args.tier2_min_positive_fraction,
    )

    tier_summary = summarize_tiers(curated)

    tier1 = curated[curated["interpretation_tier"].eq("tier1_high_confidence_screen")].copy()
    tier2 = curated[curated["interpretation_tier"].eq("tier2_screening_signal")].copy()
    tier3 = curated[curated["interpretation_tier"].eq("tier3_caution_screen")].copy()
    not_selected = curated[curated["interpretation_tier"].eq("not_selected_or_not_successful")].copy()

    write_table(curated, d02 / "curated_treatment_model_table.tsv")
    write_table(tier_summary, d02 / "interpretation_tier_summary.tsv")
    write_table(tier1, d02 / "tier1_high_confidence_treatment_models.tsv")
    write_table(tier2, d02 / "tier2_screening_signal_treatment_models.tsv")
    write_table(tier3, d02 / "tier3_caution_treatment_models.tsv")
    write_table(not_selected, d02 / "not_selected_or_not_successful_treatment_models.tsv")

    # Recurrent feature summaries focus on curated Tier 1 and Tier 2 treatment models.
    tier1_tier2_shap = filter_shap_by_curated(
        shap_long,
        curated,
        {"tier1_high_confidence_screen", "tier2_screening_signal"},
    )

    tier1_shap = filter_shap_by_curated(
        shap_long,
        curated,
        {"tier1_high_confidence_screen"},
    )

    recurrent_features = summarize_recurrent_features(tier1_tier2_shap)
    recurrent_themes = summarize_recurrent_themes(tier1_tier2_shap)
    tier1_recurrent_features = summarize_recurrent_features(tier1_shap)
    tier1_recurrent_themes = summarize_recurrent_themes(tier1_shap)

    write_table(tier1_tier2_shap, d03 / "curated_tier1_tier2_shap_feature_long.tsv")
    write_table(recurrent_features, d03 / "curated_recurrent_spatial_features.tsv")
    write_table(tier1_recurrent_features, d03 / "tier1_recurrent_spatial_features.tsv")

    write_table(recurrent_themes, d04 / "curated_recurrent_biology_themes.tsv")
    write_table(tier1_recurrent_themes, d04 / "tier1_recurrent_biology_themes.tsv")

    # Tier 1 models are handed to Step 09 for label-shuffle validation before claims are treated as validated.
    label_shuffle_candidates = tier1.copy()
    label_shuffle_candidates["label_shuffle_validation_status"] = "pending_step09"
    label_shuffle_candidates["step09_target_col"] = "fused_residual_vs_prior"
    label_shuffle_candidates["step09_feature_set"] = "v2_step05_strict_biology_registry"

    write_table(label_shuffle_candidates, d07 / "tier1_label_shuffle_candidates.tsv")

    # The Step 09 handoff explicitly records whether label-shuffle validation has candidates to test.
    handoff_summary = {
        "status": "ready_for_step09" if len(label_shuffle_candidates) > 0 else "no_tier1_candidates",
        "n_tier1_label_shuffle_candidates": int(len(label_shuffle_candidates)),
        "target_col": "fused_residual_vs_prior",
        "feature_set": "v2_step05_strict_biology_registry",
        "source_step": "08_curate_per_treatment_residual_models",
    }

    write_json(handoff_summary, d07 / "tier1_label_shuffle_handoff_summary.json")

    save_bar(
        tier_summary,
        "interpretation_tier",
        "treatment_count",
        d05 / "fig_01_interpretation_tier_counts.png",
        "V2 Step 08 interpretation tier counts",
        "Treatment count",
        top_n=10,
    )

    save_bar(
        tier1,
        "drug_key",
        "test_pearson_mean",
        d05 / "fig_02_tier1_treatment_models.png",
        "V2 Step 08 Tier 1 treatment models",
        "Mean test Pearson",
        top_n=30,
    )

    save_bar(
        tier2,
        "drug_key",
        "test_pearson_mean",
        d05 / "fig_03_tier2_treatment_models.png",
        "V2 Step 08 Tier 2 treatment models",
        "Mean test Pearson",
        top_n=30,
    )

    save_scatter(curated, d05 / "fig_04_curated_pearson_vs_r2.png")
    save_hist(curated, d05 / "fig_05_per_treatment_performance_distribution.png")

    save_bar(
        recurrent_features,
        "feature_name",
        "total_score",
        d05 / "fig_06_curated_recurrent_spatial_features.png",
        "V2 Step 08 recurrent spatial features across curated models",
        "Total SHAP score",
        top_n=30,
    )

    save_bar(
        recurrent_themes,
        "biological_theme",
        "total_score",
        d05 / "fig_07_curated_recurrent_biology_themes.png",
        "V2 Step 08 recurrent biology themes across curated models",
        "Total SHAP score",
        top_n=15,
    )

    best_tier1 = tier1.head(1)

    if best_tier1.empty:
        best_tier1_treatment = ""
        best_tier1_pearson = None
        best_tier1_r2 = None
    else:
        best_tier1_treatment = str(best_tier1.iloc[0]["drug_key"])
        best_tier1_pearson = safe_float(best_tier1.iloc[0].get("test_pearson_mean"))
        best_tier1_r2 = safe_float(best_tier1.iloc[0].get("test_r2_mean"))

    run_summary = {
        "status": "pass",
        "official_step": "08_curate_per_treatment_residual_models",
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "step05_root": str(step05_root),
        "step06_root": str(step06_root),
        "step07_root": str(step07_root),
        "output_root": str(output_root),
        "n_screened_treatments": int(len(screening)),
        "n_final_manifest_models": int(len(final_manifest)),
        "n_tier1_high_confidence": int(len(tier1)),
        "n_tier2_screening_signal": int(len(tier2)),
        "n_tier3_caution": int(len(tier3)),
        "n_not_selected_or_not_successful": int(len(not_selected)),
        "n_label_shuffle_candidates": int(len(label_shuffle_candidates)),
        "best_tier1_treatment": best_tier1_treatment,
        "best_tier1_test_pearson_mean": best_tier1_pearson,
        "best_tier1_test_r2_mean": best_tier1_r2,
        "tier1_min_pearson": args.tier1_min_pearson,
        "tier1_min_positive_fraction": args.tier1_min_positive_fraction,
        "tier2_min_pearson": args.tier2_min_pearson,
        "tier2_min_positive_fraction": args.tier2_min_positive_fraction,
        "production_dependency_on_v1_outputs": "no",
        "uses_step05_v2_registry": "yes",
        "uses_treatment_identity_features": "no",
        "ready_for_step09_label_shuffle": "yes" if len(label_shuffle_candidates) > 0 else "no",
    }

    write_json(run_summary, output_root / "v2_step08_curated_per_treatment_residual_models_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 08 CURATED PER TREATMENT RESIDUAL MODELS REPORT")
    report_lines.append("")

    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("1. Source files")
    report_lines.append(source_manifest.to_string(index=False))

    report_lines.append("")
    report_lines.append("2. Tier assignment thresholds")
    report_lines.append(f"Tier 1 minimum mean Pearson: {args.tier1_min_pearson}")
    report_lines.append(f"Tier 1 minimum positive split fraction: {args.tier1_min_positive_fraction}")
    report_lines.append("Tier 1 also requires selected final SHAP model, SHAP success, positive mean R2, and positive mean RMSE improvement.")
    report_lines.append(f"Tier 2 minimum mean Pearson: {args.tier2_min_pearson}")
    report_lines.append(f"Tier 2 minimum positive split fraction: {args.tier2_min_positive_fraction}")
    report_lines.append("Tier 2 also requires selected final SHAP model, SHAP success, and positive mean RMSE improvement.")

    report_lines.append("")
    report_lines.append("3. Tier summary")
    report_lines.append(tier_summary.to_string(index=False))

    report_lines.append("")
    report_lines.append("4. Tier 1 label shuffle candidates")
    if label_shuffle_candidates.empty:
        report_lines.append("No Tier 1 candidates generated.")
    else:
        cols = [
            "drug_key",
            "test_pearson_mean",
            "test_pearson_median",
            "test_r2_mean",
            "rmse_improvement_vs_baseline_mean",
            "test_pearson_positive_fraction",
            "final_model_rank",
            "shap_status",
            "interpretation_tier_reason",
        ]
        cols = [c for c in cols if c in label_shuffle_candidates.columns]
        report_lines.append(label_shuffle_candidates[cols].head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("5. Tier 2 screening signal treatments")
    if tier2.empty:
        report_lines.append("No Tier 2 treatments generated.")
    else:
        cols = [
            "drug_key",
            "test_pearson_mean",
            "test_pearson_median",
            "test_r2_mean",
            "rmse_improvement_vs_baseline_mean",
            "test_pearson_positive_fraction",
            "final_model_rank",
            "shap_status",
            "interpretation_tier_reason",
        ]
        cols = [c for c in cols if c in tier2.columns]
        report_lines.append(tier2[cols].head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("6. Recurrent spatial features across Tier 1 and Tier 2")
    if recurrent_features.empty:
        report_lines.append("No recurrent feature table generated.")
    else:
        report_lines.append(recurrent_features.head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("7. Recurrent biology themes across Tier 1 and Tier 2")
    if recurrent_themes.empty:
        report_lines.append("No recurrent theme table generated.")
    else:
        report_lines.append(recurrent_themes.to_string(index=False))

    report_lines.append("")
    report_lines.append("8. Interpretation")
    report_lines.append("Step 08 curates the Step 07 treatment specific residual models into confidence tiers.")
    report_lines.append("Tier 1 models are not final validated claims yet.")
    report_lines.append("Tier 1 models are the handoff to Step 09 label shuffle validation.")
    report_lines.append("All features remain restricted to the Step 05 V2 strict biology registry.")

    report_path = write_text_report(d06 / "v2_step08_curated_per_treatment_residual_models_report.txt", "\n".join(report_lines))

    slide_lines = []
    slide_lines.append("V2 STEP 08 CURATED PER TREATMENT RESIDUAL MODELS SLIDE NOTES")
    slide_lines.append("")
    slide_lines.append(f"Screened treatments: {len(screening)}")
    slide_lines.append(f"Final SHAP models: {len(final_manifest)}")
    slide_lines.append(f"Tier 1 high confidence screens: {len(tier1)}")
    slide_lines.append(f"Tier 2 screening signals: {len(tier2)}")
    slide_lines.append(f"Tier 3 caution screens: {len(tier3)}")
    slide_lines.append(f"Label shuffle candidates for Step 09: {len(label_shuffle_candidates)}")
    slide_lines.append(f"Best Tier 1 treatment: {best_tier1_treatment}")
    slide_lines.append(f"Best Tier 1 mean test Pearson: {best_tier1_pearson}")
    slide_lines.append(f"Best Tier 1 mean test R2: {best_tier1_r2}")
    slide_lines.append("")
    slide_lines.append("Top recurrent biology themes:")
    if recurrent_themes.empty:
        slide_lines.append("No recurrent themes.")
    else:
        for theme in recurrent_themes.head(8)["biological_theme"].astype(str).tolist():
            slide_lines.append(theme)
    slide_lines.append("")
    slide_lines.append("Caveat: Tier 1 is high confidence screening. Step 09 label shuffle validation is required for validated treatment specific claims.")

    write_text_report(d06 / "v2_step08_curated_per_treatment_residual_slide_notes.txt", "\n".join(slide_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Step 07 root: {step07_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Screened treatments: {len(screening)}",
        f"Final SHAP models: {len(final_manifest)}",
        f"Tier 1 high confidence: {len(tier1)}",
        f"Tier 2 screening signal: {len(tier2)}",
        f"Tier 3 caution: {len(tier3)}",
        f"Not selected or not successful: {len(not_selected)}",
        f"Label shuffle candidates for Step 09: {len(label_shuffle_candidates)}",
        f"Best Tier 1 treatment: {best_tier1_treatment}",
        f"Best Tier 1 mean test Pearson: {best_tier1_pearson}",
        f"Best Tier 1 mean test R2: {best_tier1_r2}",
        "Treatment identity features used: no",
        "Production dependency on V1 outputs: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 08 CURATED PER TREATMENT RESIDUAL MODELS COMPLETE", terminal_lines))
    print("")

    print("Tier summary")
    print(tier_summary.to_string(index=False))
    print("")

    if not label_shuffle_candidates.empty:
        print("Tier 1 label shuffle candidates")
        cols = [
            "drug_key",
            "test_pearson_mean",
            "test_r2_mean",
            "rmse_improvement_vs_baseline_mean",
            "test_pearson_positive_fraction",
            "final_model_rank",
            "shap_status",
        ]
        cols = [c for c in cols if c in label_shuffle_candidates.columns]
        print(label_shuffle_candidates[cols].head(40).to_string(index=False))
        print("")

    if not recurrent_features.empty:
        print("Curated recurrent spatial features")
        cols = [
            "feature_name",
            "feature_original",
            "biological_theme",
            "treatment_count",
            "total_score",
            "mean_score",
            "max_score",
            "example_treatments",
        ]
        cols = [c for c in cols if c in recurrent_features.columns]
        print(recurrent_features[cols].head(30).to_string(index=False))
        print("")

    if not recurrent_themes.empty:
        print("Curated recurrent biology themes")
        print(recurrent_themes.head(15).to_string(index=False))
        print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
