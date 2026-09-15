"""
Script: 03_merge_slide_features.py

Purpose:
Merge per sample slide level feature rows produced by 02_process_samples.py
into one cohort level feature table.

This script does not recompute features. It only collects existing
slide_level_feature_row.csv files, adds useful bookkeeping columns, and writes
a master table for downstream scoring, labeling, and modeling.

Inputs:
    A YAML config file containing:
        output_root

Expected upstream outputs:
    processed_samples/<sample_id>/tables/slide_level_feature_row.csv
    processing/processing_report.csv

Outputs:
    merged_features/merged_slide_features.csv
    merged_features/merge_report.csv
    merged_features/merge_summary.txt

Typical usage:
    python scripts/03_merge_slide_features.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
import sys

import pandas as pd


# =========================
# Import project modules
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config, require_successful_stage


# =========================
# File discovery helpers
# =========================

def find_feature_files(output_root):
    """Find all per sample slide level feature files."""
    output_root = Path(output_root)
    processed_root = output_root / "output_02_01_process_samples_data"

    if not processed_root.exists():
        return []

    feature_files = sorted(
        processed_root.glob("SAMPLE_*/tables/slide_level_feature_row.csv")
    )

    return feature_files


def read_processing_report(output_root):
    """Read processing report if it exists."""
    output_root = Path(output_root)
    report_path = output_root / "output_02_process_samples_reports" / "processing_report.csv"

    if not report_path.exists():
        return pd.DataFrame()

    return pd.read_csv(report_path)


def is_success_or_verified_cached_output(row, output_root=None):
    """Accept completed processing or verify the specific reusable cache state."""
    if row.get("status") == "OK":
        return True
    if row.get("status") != "SKIPPED" or row.get("reason") != "outputs_exist":
        return False
    if pd.notna(row.get("error")) and str(row.get("error")).strip():
        return False

    sample_id = str(row.get("sample_id", ""))
    if not sample_id or Path(sample_id).name != sample_id:
        return False
    if output_root is not None:
        sample_root = Path(output_root) / "output_02_01_process_samples_data" / sample_id
        h5ad_path = sample_root / "adata" / "02_processed.h5ad"
        feature_path = sample_root / "tables" / "slide_level_feature_row.csv"
    else:
        h5ad_path = Path(str(row.get("processed_h5ad", "")))
        feature_path = Path(str(row.get("slide_feature_row", "")))
    if not h5ad_path.is_file() or not feature_path.is_file():
        return False

    try:
        import h5py
        from anndata.io import read_elem

        features = pd.read_csv(feature_path)
        if len(features) != 1 or str(features.iloc[0]["sample_id"]) != sample_id:
            return False
        feature = features.iloc[0]
        # Read metadata only: checking a cache must not load its expression matrix.
        with h5py.File(h5ad_path, "r") as handle:
            matrix = handle["X"]
            shape = matrix.shape if isinstance(matrix, h5py.Dataset) else matrix.attrs["shape"]
            if len(shape) != 2 or min(shape) <= 0:
                return False
            if isinstance(matrix, h5py.Group):
                encoding = matrix.attrs.get("encoding-type")
                if encoding not in {"csr_matrix", "csc_matrix"}:
                    return False
                data, indices, indptr = (matrix[key] for key in ("data", "indices", "indptr"))
                if any(not isinstance(part, h5py.Dataset) or part.ndim != 1 for part in (data, indices, indptr)):
                    return False
                if data.dtype.kind not in "biufc" or indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
                    return False
                pointer_count = shape[0 if encoding == "csr_matrix" else 1] + 1
                if len(data) != len(indices) or len(indptr) != pointer_count or indptr[0] != 0 or indptr[-1] != len(data):
                    return False
            elif matrix.dtype.kind not in "biufc":
                return False
            obs = read_elem(handle["obs"])
            var = read_elem(handle["var"])
            if tuple(shape) != (len(obs), len(var)):
                return False
            if feature["n_spots"] != shape[0] or feature["n_genes"] != shape[1]:
                return False
            if not obs["sample_id"].astype(str).eq(sample_id).all():
                return False
            if obs["leiden"].isna().any() or obs["leiden"].nunique() < 1:
                return False
            return feature["n_clusters"] == obs["leiden"].nunique()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def get_successful_sample_ids(processing_report, output_root=None):
    """Return successful sample IDs, including only verified completed caches."""
    if processing_report.empty:
        return None

    if "status" not in processing_report.columns:
        return None

    if "sample_id" not in processing_report.columns:
        return None

    return {
        str(row["sample_id"])
        for row in processing_report.to_dict("records")
        if is_success_or_verified_cached_output(row, output_root)
    }


# =========================
# Merge helpers
# =========================

def read_one_feature_file(feature_path):
    """Read one slide feature file and add source path metadata."""
    feature_path = Path(feature_path)
    df = pd.read_csv(feature_path)

    if "sample_id" not in df.columns:
        sample_id = feature_path.parents[1].name
        df["sample_id"] = sample_id

    df["source_feature_file"] = str(feature_path)

    return df


def merge_feature_files(feature_files, successful_sample_ids=None):
    """Merge feature files into one DataFrame and build a merge report."""
    rows = []
    report_rows = []

    for feature_path in feature_files:
        feature_path = Path(feature_path)
        sample_id = feature_path.parents[1].name

        if successful_sample_ids is not None and sample_id not in successful_sample_ids:
            report_rows.append({
                "sample_id": sample_id,
                "status": "SKIPPED",
                "reason": "not_successful_in_processing_report",
                "feature_file": str(feature_path),
                "n_rows": 0,
                "n_columns": 0,
                "error": "",
            })
            continue

        try:
            df = read_one_feature_file(feature_path)

            rows.append(df)

            report_rows.append({
                "sample_id": sample_id,
                "status": "OK",
                "reason": "",
                "feature_file": str(feature_path),
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "error": "",
            })

        except Exception as error:
            report_rows.append({
                "sample_id": sample_id,
                "status": "ERROR",
                "reason": "",
                "feature_file": str(feature_path),
                "n_rows": 0,
                "n_columns": 0,
                "error": f"{type(error).__name__}: {error}",
            })

    if rows:
        merged = pd.concat(rows, axis=0, ignore_index=True, sort=False)
    else:
        merged = pd.DataFrame()

    merge_report = pd.DataFrame(report_rows)

    return merged, merge_report


def build_merge_summary(merged, merge_report, processing_report):
    """Build a readable text summary of the merge step."""
    lines = []

    lines.append("Merge slide features summary")
    lines.append("")
    lines.append(f"Merged rows: {len(merged)}")
    lines.append(f"Merged columns: {merged.shape[1] if not merged.empty else 0}")

    if not merge_report.empty and "status" in merge_report.columns:
        lines.append("")
        lines.append("Merge status counts:")
        for key, value in merge_report["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if not processing_report.empty and "status" in processing_report.columns:
        lines.append("")
        lines.append("Original processing status counts:")
        for key, value in processing_report["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if not merged.empty and "cancer_type" in merged.columns:
        lines.append("")
        lines.append("Cancer type counts:")
        for key, value in merged["cancer_type"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if not merged.empty and "dataset_id" in merged.columns:
        lines.append("")
        lines.append("Dataset counts:")
        for key, value in merged["dataset_id"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    errors = merge_report[merge_report["status"] == "ERROR"] if "status" in merge_report.columns else pd.DataFrame()

    lines.append("")
    lines.append(f"Merge errors: {len(errors)}")

    if len(errors) > 0:
        lines.append("")
        lines.append("Error details:")
        for _, row in errors.iterrows():
            lines.append(f"  {row['sample_id']}: {row['error']}")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main():
    """Merge all available slide level feature rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = validate_config(load_config(args.config))

    output_root = Path(cfg["output_root"])

    merge_dir = output_root / "output_03_merge_slide_features"
    merge_dir.mkdir(parents=True, exist_ok=True)

    merged_path = merge_dir / "merged_slide_features.csv"
    merge_report_path = merge_dir / "merge_report.csv"
    summary_path = merge_dir / "merge_summary.txt"

    processing_report = read_processing_report(output_root)
    successful_sample_ids = get_successful_sample_ids(processing_report, output_root)

    feature_files = find_feature_files(output_root)

    print("=== Merge slide features ===")
    print("Output root:", output_root)
    print("Feature files found:", len(feature_files))

    if successful_sample_ids is not None:
        print("Successful samples in processing report:", len(successful_sample_ids))
    else:
        print("No usable processing report found. Merging all discovered feature files.")

    print()

    merged, merge_report = merge_feature_files(
        feature_files=feature_files,
        successful_sample_ids=successful_sample_ids,
    )

    merge_report.to_csv(merge_report_path, index=False)

    summary_text = build_merge_summary(
        merged=merged,
        merge_report=merge_report,
        processing_report=processing_report,
    )

    summary_path.write_text(summary_text, encoding="utf-8")
    require_successful_stage(merge_report, "Feature merge", merge_report_path)
    merged.to_csv(merged_path, index=False)

    print("DONE")
    print("Merged table:", merged_path)
    print("Merge report:", merge_report_path)
    print("Summary:", summary_path)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()


