"""
Script: 10_build_model_ready_table.py

Purpose:
Build a clean model-ready numeric feature table from the full spatial feature pipeline.

This script is the handoff point between:
    1. spatial feature engineering
    2. downstream prediction / interpretation modeling

It takes the richest slide-level table from earlier scripts and prepares:
    1. model_input_numeric.csv
    2. feature_manifest.csv
    3. missingness_report.csv
    4. feature_filter_report.csv
    5. model_ready_summary.txt

Important:
This script does not train a model.
It only creates the clean feature table that model scripts can use.

Inputs:
    motif_tables/slide_features_with_motif_tables.csv
    metabolic_concordance/slide_features_with_metabolic_concordance.csv
    hotspot_metrics/slide_features_with_hotspot_metrics.csv
    accessibility_profiles/slide_features_with_accessibility.csv
    signature_scores/slide_features_with_signature_scores.csv
    scored_labels/slide_features_scored_labeled.csv
    merged_features/merged_slide_features.csv

Outputs:
    model_ready/model_input_numeric.csv
    model_ready/feature_manifest.csv
    model_ready/missingness_report.csv
    model_ready/feature_filter_report.csv
    model_ready/model_ready_summary.txt

Usage:
    python scripts/10_build_model_ready_table.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


# =========================
# Project imports
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config


# =========================
# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# Config constants
# =========================

SAMPLE_COL = "sample_id"

# Drop columns that are mostly missing.
# Keep this permissive for now because this is a heterogeneous multi-cancer cohort.
MAX_MISSING_FRACTION = 0.80

# Drop numeric columns that barely vary.
# Constant or near-constant features do not help models.
MIN_UNIQUE_VALUES = 2

# Extremely large numeric values usually come from malformed distances or bad parsing.
# We replace inf with NaN and only flag extreme finite values in the report.
EXTREME_VALUE_ABS_THRESHOLD = 1e12

# Do not impute here.
# The downstream model scripts already use median imputation.
IMPUTE_IN_THIS_SCRIPT = False

# Optional: cap feature count later in modeling, not here.
# Script 10 should preserve the full model-ready feature universe.
DO_VARIANCE_SELECTION = False


# =========================
# Argument and config helpers
# =========================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def get_output_root(config_path):
    """Load YAML config and return output_root."""
    cfg = validate_config(load_config(config_path))
    return Path(cfg["output_root"])


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    """Write dictionary data to JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


# =========================
# Input table selection
# =========================

def choose_base_table(output_root):
    """Choose the richest available slide-level table."""
    candidates = [
        output_root / "output_09_build_motif_tables" / "slide_features_with_motif_tables.csv",
        output_root / "output_08_context_alignment_and_metabolic_concordance" / "slide_features_with_metabolic_concordance.csv",
        output_root / "output_07_append_hotspot_metrics" / "slide_features_with_hotspot_metrics.csv",
        output_root / "output_06_build_accessibility_profiles" / "slide_features_with_accessibility.csv",
        output_root / "output_05_build_multi_axis_transcriptome_labels" / "slide_features_with_multi_axis_labels.csv",
        output_root / "output_04_score_and_label_slides" / "slide_features_scored_labeled.csv",
        output_root / "output_03_merge_slide_features" / "merged_slide_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find any usable slide-level feature table")


# =========================
# Column classification helpers
# =========================

def clean_name(name):
    """Normalize a column name for keyword checks."""
    return str(name).lower().replace(" ", "_")


def is_obvious_nonfeature_column(col):
    """Return True for columns that should never be numeric model features."""
    c = clean_name(col)

    # sample_id is kept separately, never used as numeric feature.
    if c == SAMPLE_COL:
        return True

    # These fields are identifiers, paths, labels, or text statuses.
    # They are useful for reports, but not direct numeric model inputs.
    blocked_terms = [
        "path",
        "file",
        "h5ad",
        "loaded_from",
        "status",
        "error",
        "notes",
        "summary",
        "rule_hits",
        "profile_label",
        "dominant_program",
        "annotation_column",
        "column_used",
        "score_column_used",
        "metadata",
        "dataset_id",
        "format",
        "timepoint",
        "cancer_type",
    ]

    return any(term in c for term in blocked_terms)


def infer_feature_group(col):
    """Assign a broad feature group based on column name."""
    c = clean_name(col)

    if c.startswith("structure_region_fraction__"):
        return "structure_region"

    if c.startswith("structure_raw_fraction__"):
        return "structure_raw"

    if c.startswith("structure_region_consensus") or c.startswith("structure_region_status_fraction__"):
        return "structure_region_qc"

    if c.startswith("filtering__") or c in ["n_spots", "n_genes"]:
        return "qc"

    if c.startswith("mean__") or c.startswith("median__") or c.startswith("spot_fraction__"):
        return "program_summary"

    if c.startswith("program__") or c.startswith("label__"):
        return "program_label"

    if "simple__" in c or "ucell__" in c or "gsva" in c:
        return "signature_score"

    if c.startswith("access_"):
        return "accessibility"

    if c.startswith("hotspot__"):
        return "hotspot"

    if c.startswith("metabolic_module__"):
        return "metabolic_module"

    if c.startswith("context_module__"):
        return "context_module"

    if c.startswith("concordance__"):
        return "context_alignment"

    if c.startswith("motif_"):
        return "motif"

    if c.startswith("pair_"):
        return "pair_relationship"

    if "slope_vs_depth" in c or "r2_vs_depth" in c:
        return "gradient"

    return "other_numeric"


def infer_feature_axis(col):
    """Assign a biological or spatial axis based on column name."""
    c = clean_name(col)

    if c.startswith("structure_region_fraction__") or c.startswith("structure_raw_fraction__"):
        return "structure_region"

    if "tumor_epithelial" in c or "tumor_mask" in c:
        return "tumor_epithelial"

    if "stromal" in c or "stroma" in c or "ecm" in c or "collagen" in c:
        return "stromal_ecm"

    if "hypoxi" in c:
        return "hypoxia"

    if "vascular" in c or "angiogenic" in c or "endothelial" in c:
        return "vascular"

    if "t_cell" in c or "interferon" in c or "immune" in c:
        return "immune"

    if "myeloid" in c or "macrophage" in c:
        return "myeloid"

    if "b_plasma" in c or "b_cell" in c:
        return "b_cell_plasma"

    if "prolifer" in c or "cell_cycle" in c:
        return "proliferation"

    if "glycolysis" in c:
        return "glycolysis"

    if "oxidative_phosphorylation" in c or "oxphos" in c:
        return "oxidative_phosphorylation"

    if "fatty_acid" in c:
        return "fatty_acid_metabolism"

    if "glutamine" in c:
        return "glutamine_metabolism"

    if "tryptophan" in c or "kynurenine" in c:
        return "immune_suppression_metabolism"

    if "accessibility" in c or "accessible" in c:
        return "accessibility"

    if "barrier" in c or "impermeable" in c:
        return "barrier"

    if "distance" in c or "nearest" in c or "centroid" in c:
        return "spatial_distance"

    if "overlap" in c or "interface" in c or "adjacent" in c:
        return "spatial_relationship"

    if "component" in c or "fragmentation" in c:
        return "spatial_topology"

    return "other"


def infer_feature_stage(col):
    """Infer which pipeline stage produced a feature."""
    c = clean_name(col)

    if c.startswith("structure_region_") or c.startswith("structure_raw_fraction__"):
        return "05_structure_region_consensus"

    if c.startswith("filtering__") or c in ["n_spots", "n_genes"]:
        return "01_processing"

    if c.startswith("mean__") or c.startswith("median__") or c.startswith("spot_fraction__"):
        return "03_merged_features"

    if c.startswith("program__") or c.startswith("label__"):
        return "04_scored_labels"

    if "simple__" in c or "ucell__" in c or "gsva" in c:
        return "05_multi_axis_transcriptome_labels"

    if c.startswith("access_"):
        return "06_accessibility"

    if c.startswith("hotspot__"):
        return "07_hotspot_metrics"

    if (
        c.startswith("metabolic_module__") or
        c.startswith("context_module__") or
        c.startswith("concordance__")
    ):
        return "08_context_alignment"

    if c.startswith("motif_") or c.startswith("pair_"):
        return "09_motif_tables"

    return "unknown"


# =========================
# Numeric feature selection
# =========================

def coerce_numeric_table(df):
    """Convert possible numeric columns to numeric while preserving sample_id."""

    if SAMPLE_COL not in df.columns:
        raise ValueError(f"Missing required column: {SAMPLE_COL}")

    out = pd.DataFrame({SAMPLE_COL: df[SAMPLE_COL].astype(str)})
    numeric_cols = {}
    conversion_rows = []

    for col in df.columns:
        if col == SAMPLE_COL:
            continue

        if is_obvious_nonfeature_column(col):
            conversion_rows.append({
                "feature": col,
                "kept_after_conversion": 0,
                "reason": "nonfeature_column",
                "numeric_fraction": np.nan,
            })
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_fraction = float(numeric.notna().mean())

        if numeric_fraction == 0:
            conversion_rows.append({
                "feature": col,
                "kept_after_conversion": 0,
                "reason": "not_numeric",
                "numeric_fraction": numeric_fraction,
            })
            continue

        numeric_cols[col] = numeric.replace([np.inf, -np.inf], np.nan)

        conversion_rows.append({
            "feature": col,
            "kept_after_conversion": 1,
            "reason": "numeric",
            "numeric_fraction": numeric_fraction,
        })

    # Build numeric columns in one concat operation to avoid DataFrame fragmentation.
    if numeric_cols:
        numeric_df = pd.concat(numeric_cols, axis=1)
        out = pd.concat([out, numeric_df], axis=1)

    return out, pd.DataFrame(conversion_rows)


def build_missingness_report(numeric_df):
    """Build per-feature missingness and variation report."""
    rows = []

    for col in numeric_df.columns:
        if col == SAMPLE_COL:
            continue

        # Re-coerce defensively in case a column was modified upstream.
        values = pd.to_numeric(numeric_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        missing_fraction = float(values.isna().mean())
        nonmissing = values.dropna()
        unique_values = int(nonmissing.nunique())
        std = float(nonmissing.std()) if len(nonmissing) > 1 else np.nan

        # Extreme values are not removed here, only flagged for review.
        extreme_count = int((nonmissing.abs() > EXTREME_VALUE_ABS_THRESHOLD).sum())

        rows.append({
            "feature": col,
            "missing_fraction": missing_fraction,
            "nonmissing_count": int(nonmissing.shape[0]),
            "unique_values": unique_values,
            "std": std,
            "extreme_abs_value_count": extreme_count,
            "feature_group": infer_feature_group(col),
            "feature_axis": infer_feature_axis(col),
            "pipeline_stage": infer_feature_stage(col),
        })

    report = pd.DataFrame(rows)

    if not report.empty:
        # Highest-missing features appear first because they are most likely to be filtered.
        report = report.sort_values(
            ["missing_fraction", "feature_group", "feature"],
            ascending=[False, True, True],
        )

    return report


def filter_numeric_features(numeric_df, missingness_report):
    """Filter numeric features for model-ready output."""
    keep = [SAMPLE_COL]
    rows = []

    # Map report rows by feature name for fast lookup.
    report_by_feature = {
        row["feature"]: row
        for _, row in missingness_report.iterrows()
    }

    for col in numeric_df.columns:
        if col == SAMPLE_COL:
            continue

        info = report_by_feature.get(col)

        # This should rarely happen, but keeps the script safe if reports get out of sync.
        if info is None:
            rows.append({
                "feature": col,
                "kept": 0,
                "reason": "missing_report_absent",
            })
            continue

        missing_fraction = float(info["missing_fraction"])
        unique_values = int(info["unique_values"])
        nonmissing_count = int(info["nonmissing_count"])

        # Completely missing columns cannot support modeling.
        if nonmissing_count == 0:
            rows.append({
                "feature": col,
                "kept": 0,
                "reason": "all_missing",
            })
            continue

        # Remove features that are mostly missing across the cohort.
        if missing_fraction > MAX_MISSING_FRACTION:
            rows.append({
                "feature": col,
                "kept": 0,
                "reason": "too_much_missing",
            })
            continue

        # Remove constant or near-constant features.
        if unique_values < MIN_UNIQUE_VALUES:
            rows.append({
                "feature": col,
                "kept": 0,
                "reason": "constant_or_near_constant",
            })
            continue

        keep.append(col)
        rows.append({
            "feature": col,
            "kept": 1,
            "reason": "kept",
        })

    filtered = numeric_df[keep].copy()
    filter_report = pd.DataFrame(rows)

    # Safety check so we do not accidentally write a featureless model table.
    if filtered.shape[1] <= 1:
        raise ValueError("No usable numeric features remained after filtering")

    return filtered, filter_report


# =========================
# Feature manifest
# =========================

def build_feature_manifest(numeric_df, missingness_report, filter_report):
    """Build feature manifest with metadata and filtering info."""
    rows = []

    # Convert reports to dictionaries so each feature can be annotated cleanly.
    missing_map = {
        row["feature"]: row
        for _, row in missingness_report.iterrows()
    }

    filter_map = {
        row["feature"]: row
        for _, row in filter_report.iterrows()
    }

    for col in numeric_df.columns:
        if col == SAMPLE_COL:
            continue

        missing_info = missing_map.get(col, {})
        filter_info = filter_map.get(col, {})

        rows.append({
            "feature": col,
            "kept": filter_info.get("kept", np.nan),
            "filter_reason": filter_info.get("reason", ""),
            "missing_fraction": missing_info.get("missing_fraction", np.nan),
            "nonmissing_count": missing_info.get("nonmissing_count", np.nan),
            "unique_values": missing_info.get("unique_values", np.nan),
            "std": missing_info.get("std", np.nan),
            "extreme_abs_value_count": missing_info.get("extreme_abs_value_count", np.nan),
            "feature_group": infer_feature_group(col),
            "feature_axis": infer_feature_axis(col),
            "pipeline_stage": infer_feature_stage(col),
        })

    manifest = pd.DataFrame(rows)

    if not manifest.empty:
        # Kept features appear first, then grouped by feature type.
        manifest = manifest.sort_values(
            ["kept", "feature_group", "feature"],
            ascending=[False, True, True],
        )

    return manifest


# =========================
# Summary builder
# =========================

def build_summary_text(
    base_path,
    df,
    numeric_df,
    filtered_df,
    conversion_report,
    missingness_report,
    filter_report
):
    """Build human-readable summary of model-ready table."""
    lines = []

    lines.append("Model-ready table summary")
    lines.append("")
    lines.append(f"Base table used: {base_path}")
    lines.append(f"Samples: {numeric_df.shape[0]}")
    lines.append(f"Original columns: {df.shape[1]}")
    lines.append(f"Total columns after numeric coercion: {numeric_df.shape[1]}")
    lines.append(f"Columns after filtering: {filtered_df.shape[1]}")
    lines.append(f"Numeric features after filtering: {filtered_df.shape[1] - 1}")
    lines.append("")

    if not conversion_report.empty:
        kept = int(conversion_report["kept_after_conversion"].sum())
        total = int(conversion_report.shape[0])

        lines.append("Conversion summary:")
        lines.append(f"  numeric columns kept: {kept}")
        lines.append(f"  total columns seen: {total}")
        lines.append("")

    if not missingness_report.empty:
        lines.append("Missingness summary:")
        lines.append(f"  max missing fraction: {missingness_report['missing_fraction'].max():.3f}")
        lines.append(f"  median missing fraction: {missingness_report['missing_fraction'].median():.3f}")
        lines.append("")

    # Surface extreme values if present (these may indicate bad parsing or scaling issues)
    if missingness_report["extreme_abs_value_count"].sum() > 0:
        lines.append("Warning: extreme values detected in some features")
        lines.append("")

    # Always show feature group distribution
    lines.append("Feature groups before filtering:")
    for group, count in missingness_report["feature_group"].value_counts().items():
        lines.append(f"  {group}: {count}")
    lines.append("")

    if not filter_report.empty:
        lines.append("Filtering summary:")
        for reason, count in filter_report["reason"].value_counts().items():
            lines.append(f"  {reason}: {count}")
        lines.append("")

    return "\n".join(lines)


# =========================
# Optional variance selection
# =========================

def select_top_variance_features(df, top_k=200):
    """Select top-K features by variance when variance selection is enabled."""
    numeric_cols = [c for c in df.columns if c != SAMPLE_COL]

    # Variance is calculated only on numeric feature columns, not sample_id.
    variances = df[numeric_cols].var(skipna=True)

    top_cols = variances.sort_values(ascending=False).head(top_k).index.tolist()

    return df[[SAMPLE_COL] + top_cols]

# =========================
# Main
# =========================

def main():
    """Run Step 10 model ready numeric table construction."""
    args = parse_args()
    output_root = get_output_root(args.config)

    out_dir = output_root / "output_10_build_model_ready_table"
    ensure_dir(out_dir)

    print("=== Build model-ready table ===")

    # Step 1: load base table
    base_path = choose_base_table(output_root)
    print("Base table:", base_path)

    df = pd.read_csv(base_path)

    if SAMPLE_COL not in df.columns:
        raise ValueError(f"Missing {SAMPLE_COL} in base table")

    # Step 2: coerce numeric features
    print("Coercing numeric features...")
    numeric_df, conversion_report = coerce_numeric_table(df)

    # Step 3: missingness report
    print("Building missingness report...")
    missingness_report = build_missingness_report(numeric_df)

    # Step 4: filter features
    print("Filtering features...")
    filtered_df, filter_report = filter_numeric_features(
        numeric_df,
        missingness_report
    )

    # Safety check: ensure we still have usable features
    if filtered_df.shape[1] <= 1:
        raise ValueError("No usable numeric features remained after filtering")

    # Step 5: optional variance selection (disabled)
    if DO_VARIANCE_SELECTION:
        print("Applying variance selection...")
        filtered_df = select_top_variance_features(filtered_df)

    # Step 6: build feature manifest
    print("Building feature manifest...")
    manifest = build_feature_manifest(
        numeric_df,
        missingness_report,
        filter_report
    )

    # Step 7: write outputs
    model_input_path = out_dir / "model_input_numeric.csv"
    manifest_path = out_dir / "feature_manifest.csv"
    missing_path = out_dir / "missingness_report.csv"
    filter_path = out_dir / "feature_filter_report.csv"
    summary_path = out_dir / "model_ready_summary.txt"

    filtered_df.to_csv(model_input_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    missingness_report.to_csv(missing_path, index=False)
    filter_report.to_csv(filter_path, index=False)

    summary_text = build_summary_text(
        base_path,
        df,
        numeric_df,
        filtered_df,
        conversion_report,
        missingness_report,
        filter_report
    )

    summary_path.write_text(summary_text, encoding="utf-8")

    print("DONE")
    print("Model input:", model_input_path)
    print("Feature manifest:", manifest_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()




