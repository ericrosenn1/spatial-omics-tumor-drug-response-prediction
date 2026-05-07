"""
Script: 04_score_and_label_slides.py

Purpose:
Score and label merged slide level features from the spatial feature pipeline.

Inputs:
    merged_features/merged_slide_features.csv

Outputs:
    scored_labels/slide_features_scored_labeled.csv
    scored_labels/program_score_summary.csv
    scored_labels/label_counts.csv
    scored_labels/label_summary_by_cancer_type.csv
    scored_labels/score_and_label_summary.txt

Usage:
    python scripts/04_score_and_label_slides.py --config configs/visium_cohort_clean.yaml
"""


# =========================
# Imports
# =========================

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd



# =========================
# Project path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# Pipeline helper imports
# =========================

from lib.config import load_config, validate_config



# =========================
# Scoring thresholds and constants
# =========================

HIGH_CUTOFF = 0.67
LOW_CUTOFF = 0.33
MIN_SUPPORT_COLUMNS = 1
BLEND_SHRINKAGE_K = 10


# =========================
# Biological program feature definitions
# =========================

PROGRAM_FEATURES = {
    "tumor_epithelial": [
        "mean__tumor_epithelial_score",
        "median__tumor_epithelial_score",
        "spot_fraction__tumor_epithelial",
    ],
    "immune_general": [
        "mean__immune_general_score",
        "median__immune_general_score",
        "spot_fraction__immune_general",
    ],
    "myeloid": [
        "mean__myeloid_score",
        "median__myeloid_score",
        "spot_fraction__myeloid",
    ],
    "fibroblast_stroma": [
        "mean__fibroblast_stroma_score",
        "median__fibroblast_stroma_score",
        "spot_fraction__fibroblast_stroma",
    ],
    "endothelial": [
        "mean__endothelial_score",
        "median__endothelial_score",
        "spot_fraction__endothelial",
    ],
    "hypoxia": [
        "mean__hypoxia_score",
        "median__hypoxia_score",
        "spot_fraction__hypoxia",
    ],
    "proliferation": [
        "mean__proliferation_score",
        "median__proliferation_score",
        "spot_fraction__proliferation",
    ],
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
    """Load the config file and return the output root path."""
    cfg = validate_config(load_config(config_path))
    return Path(cfg["output_root"])



# =========================
# Input loading helpers
# =========================

def read_merged_table(input_path):
    """Read the merged slide feature table."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing merged feature table: {input_path}")

    df = pd.read_csv(input_path)

    if "sample_id" not in df.columns:
        raise ValueError("merged_slide_features.csv must contain sample_id")

    return df


# =========================
# Percentile ranking helpers
# =========================

def percentile_rank_within_groups(block, groups):
    """Convert numeric columns into percentile ranks within dataset groups."""
    ranked = pd.DataFrame(index=block.index)

    for col in block.columns:
        values = pd.to_numeric(block[col], errors="coerce")

        if values.notna().sum() <= 1:
            ranked[col] = np.nan
            continue

        group_ranked = values.groupby(groups).rank(pct=True)

        too_small = values.groupby(groups).transform(lambda x: x.notna().sum()) <= 1
        group_ranked[too_small] = values.rank(pct=True)[too_small]

        ranked[col] = group_ranked

    return ranked

def blended_percentile_rank(block, groups, k=BLEND_SHRINKAGE_K):
    """Compute percentile ranks by blending global and dataset specific ranks."""
    ranked = pd.DataFrame(index=block.index)

    for col in block.columns:
        values = pd.to_numeric(block[col], errors="coerce")

        if values.notna().sum() <= 1:
            ranked[col] = np.nan
            continue

        global_rank = values.rank(pct=True)
        dataset_rank = values.groupby(groups).rank(pct=True)
        group_size = values.groupby(groups).transform(lambda x: x.notna().sum())

        weight = group_size / (group_size + k)

        ranked[col] = (weight * dataset_rank) + ((1.0 - weight) * global_rank)

    return ranked



# =========================
# Program score construction
# =========================

def safe_numeric_block(df, columns):
    """Return available numeric columns from a DataFrame."""
    existing = [col for col in columns if col in df.columns]

    if not existing:
        return pd.DataFrame(index=df.index), existing

    block = df[existing].apply(pd.to_numeric, errors="coerce")
    return block, existing


def score_one_program(df, program_name, columns):
    """Compute a missing aware percentile based score for one biological program."""
    block, existing = safe_numeric_block(df, columns)

    if block.empty:
        score = pd.Series(np.nan, index=df.index)
        support = pd.Series(0, index=df.index)
        return score, support, existing

    support = block.notna().sum(axis=1)

    if "dataset_id" in df.columns:
        ranked = blended_percentile_rank(block, df["dataset_id"].astype(str))
    else:
        ranked = block.apply(percentile_rank, axis=0)

    score = ranked.mean(axis=1, skipna=True)
    score[support < MIN_SUPPORT_COLUMNS] = np.nan

    return score, support, existing


def add_program_scores(df):
    """Add program scores, support counts, and high or low flags."""
    out = df.copy()

    for program_name, columns in PROGRAM_FEATURES.items():
        score, support, existing = score_one_program(out, program_name, columns)

        out[f"score__{program_name}"] = score
        out[f"support_n__{program_name}"] = support
        out[f"support_cols__{program_name}"] = "; ".join(existing)
        out[f"label__{program_name}_high"] = score >= HIGH_CUTOFF
        out[f"label__{program_name}_low"] = score <= LOW_CUTOFF

    return out



# =========================
# Composite label construction
# =========================

def get_score(df, name):
    """Return a program score column or a missing series if absent."""
    col = f"score__{name}"

    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")

    return pd.Series(np.nan, index=df.index)


def add_composite_labels(df):
    """Add interpretable composite biological labels."""
    out = df.copy()

    tumor = get_score(out, "tumor_epithelial")
    immune = get_score(out, "immune_general")
    myeloid = get_score(out, "myeloid")
    stroma = get_score(out, "fibroblast_stroma")
    endothelial = get_score(out, "endothelial")
    hypoxia = get_score(out, "hypoxia")
    proliferation = get_score(out, "proliferation")

    out["label__tumor_epithelial_enriched"] = tumor >= HIGH_CUTOFF
    out["label__immune_inflamed"] = immune >= HIGH_CUTOFF
    out["label__immune_desert"] = immune <= LOW_CUTOFF
    out["label__myeloid_enriched"] = myeloid >= HIGH_CUTOFF
    out["label__stromal_barrier"] = stroma >= HIGH_CUTOFF
    out["label__vascular_endothelial"] = endothelial >= HIGH_CUTOFF
    out["label__hypoxic"] = hypoxia >= HIGH_CUTOFF
    out["label__proliferative"] = proliferation >= HIGH_CUTOFF

    out["label__immune_excluded_proxy"] = (
        (immune >= LOW_CUTOFF)
        & (
            (stroma >= HIGH_CUTOFF)
            | (myeloid >= HIGH_CUTOFF)
            | (hypoxia >= HIGH_CUTOFF)
        )
    )

    out["label__cold_barriered_proxy"] = (
        (immune <= LOW_CUTOFF)
        & (
            (stroma >= HIGH_CUTOFF)
            | (myeloid >= HIGH_CUTOFF)
            | (hypoxia >= HIGH_CUTOFF)
        )
    )

    return out



# =========================
# Dominant program and label summary helpers
# =========================

def add_dominant_program(df):
    """Add the strongest available program per slide."""
    out = df.copy()

    score_cols = [f"score__{name}" for name in PROGRAM_FEATURES]
    score_cols = [col for col in score_cols if col in out.columns]

    if not score_cols:
        out["dominant_program"] = "none"
        out["dominant_program_score"] = np.nan
        out["n_supported_programs"] = 0
        return out

    score_block = out[score_cols].apply(pd.to_numeric, errors="coerce")

    all_missing = score_block.isna().all(axis=1)

    out["dominant_program"] = "none"
    out.loc[~all_missing, "dominant_program"] = (
        score_block.loc[~all_missing].idxmax(axis=1)
        .str.replace("score__", "", regex=False)
    )

    out["dominant_program_score"] = np.nan
    out.loc[~all_missing, "dominant_program_score"] = score_block.loc[~all_missing].max(axis=1)

    support_cols = [f"support_n__{name}" for name in PROGRAM_FEATURES]
    support_cols = [col for col in support_cols if col in out.columns]

    if support_cols:
        out["n_supported_programs"] = out[support_cols].gt(0).sum(axis=1)
    else:
        out["n_supported_programs"] = 0

    out.loc[out["dominant_program_score"].isna(), "dominant_program"] = "none"

    return out


def add_label_summary(df):
    """Add a compact semicolon separated label summary column."""
    out = df.copy()
    label_cols = [col for col in out.columns if col.startswith("label__")]

    def summarize_row(row):
        """Document summarize_row within the Step 04 scoring workflow."""
        labels = []

        for col in label_cols:
            if bool(row[col]):
                labels.append(col.replace("label__", ""))

        if not labels:
            return "unlabeled"

        return "; ".join(labels)

    out["label_summary"] = out.apply(summarize_row, axis=1)
    return out



# =========================
# Summary table builders
# =========================

def build_program_score_summary(df):
    """Summarize score availability and distributions for each program."""
    rows = []

    for program_name in PROGRAM_FEATURES:
        score_col = f"score__{program_name}"
        support_col = f"support_n__{program_name}"

        score = pd.to_numeric(df[score_col], errors="coerce")
        support = pd.to_numeric(df[support_col], errors="coerce")

        rows.append({
            "program": program_name,
            "n_nonmissing_scores": int(score.notna().sum()),
            "fraction_nonmissing_scores": float(score.notna().mean()),
            "mean_score": float(score.mean()) if score.notna().sum() else np.nan,
            "median_score": float(score.median()) if score.notna().sum() else np.nan,
            "std_score": float(score.std(ddof=0)) if score.notna().sum() > 1 else np.nan,
            "min_score": float(score.min()) if score.notna().sum() else np.nan,
            "max_score": float(score.max()) if score.notna().sum() else np.nan,
            "mean_support_n": float(support.mean()) if support.notna().sum() else np.nan,
            "n_high": int((score >= HIGH_CUTOFF).sum()),
            "n_low": int((score <= LOW_CUTOFF).sum()),
        })

    return pd.DataFrame(rows)


def build_label_counts(df):
    """Count how many slides are positive for each label."""
    rows = []
    label_cols = [col for col in df.columns if col.startswith("label__")]

    for col in label_cols:
        values = df[col].fillna(False).astype(bool)

        rows.append({
            "label": col,
            "n_positive": int(values.sum()),
            "fraction_positive": float(values.mean()),
        })

    return pd.DataFrame(rows).sort_values("label")


def build_cancer_type_summary(df):
    """Summarize scores and labels by cancer type."""
    if "cancer_type" not in df.columns:
        return pd.DataFrame()

    rows = []
    label_cols = [col for col in df.columns if col.startswith("label__")]
    score_cols = [col for col in df.columns if col.startswith("score__")]

    for cancer_type, sub in df.groupby("cancer_type", dropna=False):
        row = {
            "cancer_type": cancer_type,
            "n_samples": int(len(sub)),
        }

        for col in label_cols:
            row[f"{col}_fraction"] = float(sub[col].fillna(False).astype(bool).mean())

        for col in score_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().sum() else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def build_summary_text(df, program_summary, label_counts, cancer_summary):
    """Build a human readable text report for the scoring step."""
    lines = []

    lines.append("Score and label slides summary")
    lines.append("")
    lines.append(f"Samples labeled: {len(df)}")
    lines.append(f"Columns written: {df.shape[1]}")
    lines.append(f"High cutoff: {HIGH_CUTOFF}")
    lines.append(f"Low cutoff: {LOW_CUTOFF}")
    lines.append("")

    lines.append("Dominant program counts:")
    for key, value in df["dominant_program"].value_counts(dropna=False).items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("Program score availability:")
    for _, row in program_summary.iterrows():
        lines.append(
            f"  {row['program']}: "
            f"{row['n_nonmissing_scores']} nonmissing "
            f"({row['fraction_nonmissing_scores']:.3f})"
        )

    lines.append("")
    lines.append("Label counts:")
    for _, row in label_counts.iterrows():
        lines.append(
            f"  {row['label']}: "
            f"{row['n_positive']} "
            f"({row['fraction_positive']:.3f})"
        )

    if not cancer_summary.empty:
        lines.append("")
        lines.append("Cancer type counts:")
        for _, row in cancer_summary[["cancer_type", "n_samples"]].iterrows():
            lines.append(f"  {row['cancer_type']}: {row['n_samples']}")

    return "\n".join(lines)



# =========================
# Output writers
# =========================

def write_outputs(out_dir, labeled, program_summary, label_counts, cancer_summary, summary_text):
    """Write all scored label outputs to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "slide_features_scored_labeled.csv"
    program_summary_path = out_dir / "program_score_summary.csv"
    label_counts_path = out_dir / "label_counts.csv"
    cancer_summary_path = out_dir / "label_summary_by_cancer_type.csv"
    summary_path = out_dir / "score_and_label_summary.txt"

    labeled.to_csv(labeled_path, index=False)
    program_summary.to_csv(program_summary_path, index=False)
    label_counts.to_csv(label_counts_path, index=False)

    if not cancer_summary.empty:
        cancer_summary.to_csv(cancer_summary_path, index=False)

    summary_path.write_text(summary_text, encoding="utf-8")

    return {
        "labeled": labeled_path,
        "program_summary": program_summary_path,
        "label_counts": label_counts_path,
        "cancer_summary": cancer_summary_path,
        "summary": summary_path,
    }



# =========================
# Main workflow
# =========================

def main():
    """Run slide level scoring and labeling."""
    args = parse_args()
    output_root = get_output_root(args.config)

    input_path = output_root / "output_03_merge_slide_features" / "merged_slide_features.csv"
    out_dir = output_root / "output_04_score_and_label_slides"

    df = read_merged_table(input_path)

    labeled = add_program_scores(df)
    labeled = add_composite_labels(labeled)
    labeled = add_dominant_program(labeled)
    labeled = add_label_summary(labeled)

    program_summary = build_program_score_summary(labeled)
    label_counts = build_label_counts(labeled)
    cancer_summary = build_cancer_type_summary(labeled)

    summary_text = build_summary_text(
        df=labeled,
        program_summary=program_summary,
        label_counts=label_counts,
        cancer_summary=cancer_summary,
    )

    paths = write_outputs(
        out_dir=out_dir,
        labeled=labeled,
        program_summary=program_summary,
        label_counts=label_counts,
        cancer_summary=cancer_summary,
        summary_text=summary_text,
    )

    print("DONE")
    print("Labeled table:", paths["labeled"])
    print("Program summary:", paths["program_summary"])
    print("Label counts:", paths["label_counts"])
    print("Cancer summary:", paths["cancer_summary"])
    print("Summary:", paths["summary"])
    print()
    print(summary_text)



# =========================
# Command line entry point
# =========================

if __name__ == "__main__":
    main()

