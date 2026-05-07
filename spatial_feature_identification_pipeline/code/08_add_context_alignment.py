"""
Script: 08_add_context_alignment.py

Purpose:
Add slide-level metabolic concordance features to the cohort feature table.

This script integrates:
    1. metabolic signature scores from Script 05
    2. accessibility and barrier features from Script 06
    3. hotspot structure features from Script 07

The goal is to ask whether metabolic programs agree with their expected
spatial and biological context.

Examples:
    glycolysis should often agree with hypoxia
    oxidative phosphorylation may agree with vascular access
    fatty acid metabolism may agree with stromal or ECM structure
    glutamine metabolism may agree with proliferation
    tryptophan/kynurenine may agree with immune checkpoint or suppression signals

Inputs:
    hotspot_metrics/slide_features_with_hotspot_metrics.csv
    accessibility_profiles/slide_features_with_accessibility.csv
    signature_scores/slide_features_with_signature_scores.csv
    scored_labels/slide_features_scored_labeled.csv
    merged_features/merged_slide_features.csv

Outputs:
    metabolic_concordance/metabolic_concordance_summary.csv
    metabolic_concordance/metabolic_concordance_status.csv
    metabolic_concordance/slide_features_with_metabolic_concordance.csv
    metabolic_concordance/metabolic_concordance_summary.txt

Usage:
    python scripts/08_add_metabolic_concordance.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
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
# Config constants
# =========================

# Values are converted into percentile-like ranks before combining.
# This prevents features from different methods and scales from dominating unfairly.
MIN_NONMISSING_FOR_RANK = 5

# High and low cutoffs define interpretable metabolic flags.
HIGH_CUTOFF = 0.67
LOW_CUTOFF = 0.33

# When a concordance module has fewer than this many valid input features,
# the module is reported as unavailable instead of overinterpreted.
MIN_MODULE_SUPPORT = 1


# =========================
# Biological module definitions
# =========================

# These are broad keyword groups used to find relevant columns in the merged table.
# We use keyword detection because earlier scripts may generate names from simple,
# UCell, GSVA, or hotspot metrics.
METABOLIC_PROGRAMS = {
    "glycolysis": [
        "glycolysis",
        "SLC2A1",
        "LDHA",
        "HK2",
    ],
    "oxidative_phosphorylation": [
        "oxidative_phosphorylation",
        "oxphos",
        "respiratory",
        "mitochondrial",
    ],
    "fatty_acid_metabolism": [
        "fatty_acid",
        "fatty_acid_oxidation",
        "fatty_acid_synthesis",
        "FASN",
        "CPT1A",
    ],
    "glutamine_metabolism": [
        "glutamine",
        "GLS",
        "SLC1A5",
    ],
    "nucleotide_synthesis": [
        "nucleotide",
        "RRM2",
        "TYMS",
    ],
    "tryptophan_kynurenine": [
        "tryptophan",
        "kynurenine",
        "IDO1",
        "TDO2",
    ],
    "proline_collagen_support": [
        "proline",
        "collagen_support",
        "P4HA",
        "PLOD",
    ],
}


# These contextual modules represent biological environments expected to support
# or oppose metabolic programs.
CONTEXT_PROGRAMS = {
    "hypoxia_context": [
        "hypoxia",
        "hypoxic",
        "hypoxic_stress",
    ],
    "vascular_context": [
        "vascular",
        "angiogenic",
        "endothelial",
        "oxygen",
    ],
    "stromal_ecm_context": [
        "stromal",
        "stroma",
        "ecm",
        "collagen",
        "fibroblast",
    ],
    "immune_context": [
        "immune",
        "t_cell",
        "interferon",
        "myeloid",
        "b_plasma",
    ],
    "checkpoint_context": [
        "checkpoint",
        "exhaustion",
        "PDCD1",
        "CD274",
        "CTLA4",
        "TIGIT",
    ],
    "proliferation_context": [
        "proliferation",
        "proliferative",
        "cell_cycle",
        "G2M",
        "E2F",
        "MKI67",
    ],
    "accessibility_context": [
        "accessibility",
        "accessible",
        "vascular_access",
    ],
    "barrier_context": [
        "barrier",
        "impermeable",
        "stromal_barrier",
    ],
}


# Concordance modules compare metabolic programs with biologically related context.
# sign = 1 means high metabolic score and high context score are concordant.
# sign = -1 means high metabolic score and low context score are concordant.
CONCORDANCE_RULES = {
    "glycolysis_hypoxia_concordance": {
        "metabolic": "glycolysis",
        "context": "hypoxia_context",
        "sign": 1,
        "interpretation": "glycolysis aligns with hypoxic stress",
    },
    "oxphos_vascular_concordance": {
        "metabolic": "oxidative_phosphorylation",
        "context": "vascular_context",
        "sign": 1,
        "interpretation": "oxidative metabolism aligns with vascular or oxygen access",
    },
    "fatty_acid_stromal_concordance": {
        "metabolic": "fatty_acid_metabolism",
        "context": "stromal_ecm_context",
        "sign": 1,
        "interpretation": "fatty acid metabolism aligns with stromal or ECM-rich context",
    },
    "glutamine_proliferation_concordance": {
        "metabolic": "glutamine_metabolism",
        "context": "proliferation_context",
        "sign": 1,
        "interpretation": "glutamine metabolism aligns with proliferative pressure",
    },
    "nucleotide_proliferation_concordance": {
        "metabolic": "nucleotide_synthesis",
        "context": "proliferation_context",
        "sign": 1,
        "interpretation": "nucleotide synthesis aligns with proliferative pressure",
    },
    "tryptophan_checkpoint_concordance": {
        "metabolic": "tryptophan_kynurenine",
        "context": "checkpoint_context",
        "sign": 1,
        "interpretation": "tryptophan kynurenine metabolism aligns with checkpoint or exhaustion context",
    },
    "proline_stromal_concordance": {
        "metabolic": "proline_collagen_support",
        "context": "stromal_ecm_context",
        "sign": 1,
        "interpretation": "proline collagen support aligns with stromal matrix context",
    },
    "glycolysis_accessibility_inverse": {
        "metabolic": "glycolysis",
        "context": "accessibility_context",
        "sign": -1,
        "interpretation": "glycolysis is stronger when accessibility is lower",
    },
    "barrier_metabolic_stress_concordance": {
        "metabolic": "glycolysis",
        "context": "barrier_context",
        "sign": 1,
        "interpretation": "glycolysis aligns with barrier or impermeable context",
    },
}


# =========================
# Argument and config helpers
# =========================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def get_output_root(config_path):
    """Load the YAML config and return output_root."""
    cfg = validate_config(load_config(config_path))
    return Path(cfg["output_root"])


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def choose_base_table(output_root):
    """Choose the richest existing table for metabolic concordance."""
    candidates = [
        output_root / "output_07_append_hotspot_metrics" / "slide_features_with_hotspot_metrics.csv",
        output_root / "output_06_build_accessibility_profiles" / "slide_features_with_accessibility.csv",
        output_root / "output_05_build_multi_axis_transcriptome_labels" / "slide_features_with_multi_axis_labels.csv",
        output_root / "output_04_score_and_label_slides" / "slide_features_scored_labeled.csv",
        output_root / "output_03_merge_slide_features" / "merged_slide_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find a usable slide-level feature table")


# =========================
# Safe numeric helpers
# =========================

def finite_values(values):
    """Return finite numeric values from an array-like object."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(values):
    """Return mean of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.mean(vals))


def safe_median(values):
    """Return median of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.median(vals))


def safe_std(values):
    """Return standard deviation of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.std(vals))


def safe_min(values):
    """Return minimum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.min(vals))


def safe_max(values):
    """Return maximum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.max(vals))


def safe_quantile(values, q):
    """Return quantile of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.quantile(vals, q))


def numeric_series(df, col):
    """Return one dataframe column as numeric values."""
    return pd.to_numeric(df[col], errors="coerce")


def percentile_rank(values):
    """Convert a numeric vector into percentile ranks."""
    s = pd.to_numeric(pd.Series(values), errors="coerce")

    if s.notna().sum() < MIN_NONMISSING_FOR_RANK:
        return pd.Series(np.nan, index=s.index)

    # Percentile ranks make heterogeneous feature families comparable.
    return s.rank(pct=True)


def safe_average_columns(df, columns):
    """Average selected columns row-wise while preserving missingness."""
    if len(columns) == 0:
        return pd.Series(np.nan, index=df.index)

    block = df[columns].apply(pd.to_numeric, errors="coerce")

    if block.notna().sum(axis=1).max() == 0:
        return pd.Series(np.nan, index=df.index)

    return block.mean(axis=1, skipna=True)


# =========================
# Column selection helpers
# =========================

def clean_column_name(name):
    """Normalize a column name for keyword matching."""
    return str(name).lower().replace(" ", "_")


def column_matches_any_keyword(col, keywords):
    """Return True when a column name matches any keyword."""
    col_clean = clean_column_name(col)

    return any(str(keyword).lower() in col_clean for keyword in keywords)


def is_candidate_signal_column(col):
    """Return True when a column is likely to be a useful biological signal."""
    col_clean = clean_column_name(col)

    # Exclude IDs, paths, statuses, and text labels.
    exclude_terms = [
        "sample_id",
        "path",
        "status",
        "label",
        "used",
        "summary",
        "notes",
        "rule_hits",
        "metadata",
        "h5ad",
    ]

    if any(term in col_clean for term in exclude_terms):
        return False

    # Prefer columns generated by scripts 04 to 07.
    include_terms = [
        "score",
        "simple__",
        "ucell__",
        "gsva",
        "hotspot__",
        "access_",
        "fraction",
        "mean",
        "median",
        "q75",
        "q90",
    ]

    return any(term in col_clean for term in include_terms)


def select_columns_by_keywords(df, keywords):
    """Select numeric signal columns whose names match keywords."""
    selected = []

    for col in df.columns:
        if not is_candidate_signal_column(col):
            continue

        if not column_matches_any_keyword(col, keywords):
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().sum() < MIN_NONMISSING_FOR_RANK:
            continue

        selected.append(col)

    return selected


def select_program_columns(df, program_name, program_dict):
    """Select columns matching one biological program."""
    keywords = program_dict.get(program_name, [])
    return select_columns_by_keywords(df, keywords)


# =========================
# Module score construction
# =========================

def build_module_score(df, module_name, columns):
    """Build a percentile-normalized score for one module."""
    if len(columns) == 0:
        return pd.Series(np.nan, index=df.index), 0

    ranked_cols = []

    for col in columns:
        ranked = percentile_rank(df[col])
        ranked_cols.append(ranked)

    ranked_block = pd.concat(ranked_cols, axis=1)

    # Average ranked features so the module reflects relative enrichment.
    score = ranked_block.mean(axis=1, skipna=True)

    support = ranked_block.notna().sum(axis=1)

    # Samples with no supporting features should stay missing.
    score[support < MIN_MODULE_SUPPORT] = np.nan

    return score, len(columns)


def add_program_module_scores(df):
    """Add metabolic and context module scores to a copy of the dataframe."""
    out = df.copy()
    status_rows = []

    for program_name in METABOLIC_PROGRAMS:
        columns = select_program_columns(out, program_name, METABOLIC_PROGRAMS)

        score, n_cols = build_module_score(out, program_name, columns)

        out[f"metabolic_module__{program_name}"] = score
        out[f"metabolic_module__{program_name}_n_columns"] = n_cols

        status_rows.append({
            "module_type": "metabolic",
            "module": program_name,
            "n_columns": n_cols,
            "columns": "; ".join(columns),
        })

    for context_name in CONTEXT_PROGRAMS:
        columns = select_program_columns(out, context_name, CONTEXT_PROGRAMS)

        score, n_cols = build_module_score(out, context_name, columns)

        out[f"context_module__{context_name}"] = score
        out[f"context_module__{context_name}_n_columns"] = n_cols

        status_rows.append({
            "module_type": "context",
            "module": context_name,
            "n_columns": n_cols,
            "columns": "; ".join(columns),
        })

    status_df = pd.DataFrame(status_rows)

    return out, status_df

# =========================
# Concordance computation
# =========================

def compute_concordance_score(metabolic_score, context_score, sign):
    """Compute concordance score between metabolic and context modules."""
    m = pd.to_numeric(metabolic_score, errors="coerce")
    c = pd.to_numeric(context_score, errors="coerce")

    # Concordance is directional:
    # sign = 1 â†’ high-high agreement
    # sign = -1 â†’ high-low agreement
    if sign == 1:
        concordance = (m + c) / 2.0
    else:
        concordance = (m + (1.0 - c)) / 2.0

    return concordance


def compute_concordance_flag(score):
    """Convert continuous concordance score into high/low categorical flag."""
    if not np.isfinite(score):
        return "unknown"

    if score >= HIGH_CUTOFF:
        return "high"

    if score <= LOW_CUTOFF:
        return "low"

    return "intermediate"


def add_concordance_features(df):
    """Add metabolic-context concordance features to dataframe."""
    out = df.copy()
    status_rows = []

    for rule_name, rule in CONCORDANCE_RULES.items():

        metabolic_col = f"metabolic_module__{rule['metabolic']}"
        context_col = f"context_module__{rule['context']}"

        if metabolic_col not in out.columns or context_col not in out.columns:
            out[f"concordance__{rule_name}"] = np.nan
            out[f"concordance__{rule_name}_flag"] = "unknown"

            status_rows.append({
                "rule": rule_name,
                "status": "missing_columns",
                "metabolic_col": metabolic_col,
                "context_col": context_col,
            })
            continue

        concordance = compute_concordance_score(
            out[metabolic_col],
            out[context_col],
            rule["sign"],
        )

        out[f"concordance__{rule_name}"] = concordance

        flags = concordance.apply(compute_concordance_flag)
        out[f"concordance__{rule_name}_flag"] = flags

        status_rows.append({
            "rule": rule_name,
            "status": "ok",
            "metabolic_col": metabolic_col,
            "context_col": context_col,
        })

    status_df = pd.DataFrame(status_rows)

    return out, status_df


# =========================
# Summary features
# =========================

def summarize_concordance(df):
    """Generate overall summary metrics for concordance."""
    summary = {}

    concordance_cols = [c for c in df.columns if c.startswith("concordance__") and not c.endswith("_flag")]

    for col in concordance_cols:
        values = pd.to_numeric(df[col], errors="coerce")

        summary[f"{col}_mean"] = safe_mean(values)
        summary[f"{col}_median"] = safe_median(values)
        summary[f"{col}_std"] = safe_std(values)

    return summary


def add_global_concordance_scores(df):
    """Add global aggregate concordance scores across all modules."""
    out = df.copy()

    concordance_cols = [c for c in df.columns if c.startswith("concordance__") and not c.endswith("_flag")]

    if len(concordance_cols) == 0:
        out["concordance__overall_mean"] = np.nan
        out["concordance__overall_std"] = np.nan
        return out

    block = df[concordance_cols].apply(pd.to_numeric, errors="coerce")

    out["concordance__overall_mean"] = block.mean(axis=1, skipna=True)
    out["concordance__overall_std"] = block.std(axis=1, skipna=True)

    return out


# =========================
# Merge + summary helpers
# =========================

def merge_concordance_summary(base_path, concordance_df):
    """Merge concordance features into existing slide-level table."""
    base_df = pd.read_csv(base_path)

    if "sample_id" not in base_df.columns:
        raise ValueError("Base table missing sample_id")

    if concordance_df.empty:
        return base_df

    keep_cols = [
        col for col in concordance_df.columns
        if col == "sample_id" or col not in base_df.columns
    ]

    return base_df.merge(concordance_df[keep_cols], on="sample_id", how="left")


def build_summary_text(status_df, module_status_df, concordance_status_df):
    """Build human-readable summary of concordance results."""
    lines = []

    lines.append("Metabolic concordance summary")
    lines.append("")

    lines.append(f"Samples processed: {len(status_df)}")

    if not status_df.empty:
        lines.append("")
        lines.append("Sample status counts:")
        for k, v in status_df["status"].value_counts().items():
            lines.append(f"  {k}: {v}")

    if not module_status_df.empty:
        lines.append("")
        lines.append("Module construction:")
        for _, row in module_status_df.iterrows():
            lines.append(f"  {row['module_type']}::{row['module']}: {row['n_columns']} columns")

    if not concordance_status_df.empty:
        lines.append("")
        lines.append("Concordance rules:")
        for _, row in concordance_status_df.iterrows():
            lines.append(f"  {row['rule']} â†’ {row['status']}")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main():
    """Run metabolic concordance integration across cohort."""
    args = parse_args()
    output_root = get_output_root(args.config)

    out_dir = output_root / "output_08_context_alignment_and_metabolic_concordance"
    ensure_dir(out_dir)

    base_path = choose_base_table(output_root)

    print("=== Add metabolic concordance ===")
    print("Base table:", base_path)

    df = pd.read_csv(base_path)

    # Track sample-level status
    status_rows = [{"sample_id": sid, "status": "ok"} for sid in df["sample_id"]]

    # Step 1: build module scores
    df_modules, module_status_df = add_program_module_scores(df)

    # Step 2: compute concordance features
    df_concordance, concordance_status_df = add_concordance_features(df_modules)

    # Step 3: add global summaries
    df_final = add_global_concordance_scores(df_concordance)

    # Step 4: merge with base table
    merged = merge_concordance_summary(base_path, df_final)

    # Save outputs
    concordance_summary_path = out_dir / "metabolic_concordance_summary.csv"
    concordance_status_path = out_dir / "metabolic_concordance_status.csv"
    merged_path = out_dir / "slide_features_with_metabolic_concordance.csv"
    summary_txt_path = out_dir / "metabolic_concordance_summary.txt"

    df_final.to_csv(concordance_summary_path, index=False)
    pd.DataFrame(status_rows).to_csv(concordance_status_path, index=False)
    merged.to_csv(merged_path, index=False)

    summary_text = build_summary_text(
        pd.DataFrame(status_rows),
        module_status_df,
        concordance_status_df,
    )

    summary_txt_path.write_text(summary_text, encoding="utf-8")

    print("DONE")

if __name__ == "__main__":
    main()



