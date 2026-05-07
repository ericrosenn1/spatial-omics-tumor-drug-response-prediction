"""
Script: 01_validate_inputs.py

Purpose:
Validate Visium-style sample folders before full spatial pipeline run.

Role:
Check loadability, spatial files, metadata, and compressed inputs.
Auto-extract .gz files when needed.

Inputs:
    YAML config with:
        input_root
        output_root
        sample_glob

Outputs:
    input_validation_report.csv
    input_validation_summary.txt

Usage:
    python scripts/01_validate_inputs.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
import gzip
import shutil
import sys

import pandas as pd


# =========================
# Import project modules
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config
from lib.io import load_sample, detect_expression_format, check_spatial_files


# =========================
# GZ helpers
# =========================

def extract_gz_file(gz_path, overwrite=False):
    """extract one gz file"""
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")

    # skip already extracted
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return {
            "created": False,
            "skipped": True,
            "failed": False,
            "output": str(out_path),
            "error": "",
        }

    try:
        # stream copy, safer for large matrix files
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return {
            "created": True,
            "skipped": False,
            "failed": False,
            "output": str(out_path),
            "error": "",
        }

    except Exception as error:
        return {
            "created": False,
            "skipped": False,
            "failed": True,
            "output": str(out_path),
            "error": f"{type(error).__name__}: {error}",
        }


def extract_gz_inputs(sample_dir, overwrite=False):
    """extract gz files in sample folder"""
    sample_dir = Path(sample_dir)

    gz_files = sorted(sample_dir.rglob("*.gz"))

    stats = {
        "gz_files_found": len(gz_files),
        "gz_created": 0,
        "gz_skipped": 0,
        "gz_failed": 0,
        "gz_errors": [],
    }

    # extract each compressed input
    for gz_path in gz_files:
        result = extract_gz_file(gz_path, overwrite=overwrite)

        if result["created"]:
            stats["gz_created"] += 1

        if result["skipped"]:
            stats["gz_skipped"] += 1

        if result["failed"]:
            stats["gz_failed"] += 1
            stats["gz_errors"].append(f"{gz_path.name}: {result['error']}")

    return stats


# =========================
# Validation helpers
# =========================

def validate_one_sample(sample_dir, extract_gz=True, overwrite_gz=False):
    """validate one sample folder and return one report row"""
    sample_dir = Path(sample_dir)

    # decompress first, then detect/load
    if extract_gz:
        gz_stats = extract_gz_inputs(sample_dir, overwrite=overwrite_gz)
    else:
        gz_stats = {
            "gz_files_found": 0,
            "gz_created": 0,
            "gz_skipped": 0,
            "gz_failed": 0,
            "gz_errors": [],
        }

    # lightweight checks before full load
    expression_format = detect_expression_format(sample_dir)
    spatial_info = check_spatial_files(sample_dir)

    try:
        # critical full loading test
        adata, info = load_sample(sample_dir)

        row = {
            "sample_id": sample_dir.name,
            "status": "OK",
            "format": info.get("format", expression_format),
            "loaded_from": info.get("loaded_from", ""),
            "n_spots": info.get("n_spots", int(adata.n_obs)),
            "n_genes": info.get("n_genes", int(adata.n_vars)),
            "has_positions": info.get("has_positions", spatial_info["has_positions"]),
            "has_scalefactors": info.get("has_scalefactors", spatial_info["has_scalefactors"]),
            "has_hires_image": info.get("has_hires_image", spatial_info["has_hires_image"]),
            "has_lowres_image": info.get("has_lowres_image", spatial_info["has_lowres_image"]),
            "has_image": info.get("has_image", spatial_info["has_image"]),
            "dataset_id": info.get("dataset_id", ""),
            "cancer_type": info.get("cancer_type", ""),
            "timepoint": info.get("timepoint", ""),
            "sample_group": info.get("sample_group", ""),
            "original_name": info.get("original_name", ""),
            "gz_files_found": gz_stats["gz_files_found"],
            "gz_created": gz_stats["gz_created"],
            "gz_skipped": gz_stats["gz_skipped"],
            "gz_failed": gz_stats["gz_failed"],
            "gz_errors": " | ".join(gz_stats["gz_errors"]),
            "error": "",
        }

    except Exception as error:
        # capture error, keep batch running
        row = {
            "sample_id": sample_dir.name,
            "status": "ERROR",
            "format": expression_format,
            "loaded_from": "",
            "n_spots": "",
            "n_genes": "",
            "has_positions": spatial_info["has_positions"],
            "has_scalefactors": spatial_info["has_scalefactors"],
            "has_hires_image": spatial_info["has_hires_image"],
            "has_lowres_image": spatial_info["has_lowres_image"],
            "has_image": spatial_info["has_image"],
            "dataset_id": "",
            "cancer_type": "",
            "timepoint": "",
            "sample_group": "",
            "original_name": "",
            "gz_files_found": gz_stats["gz_files_found"],
            "gz_created": gz_stats["gz_created"],
            "gz_skipped": gz_stats["gz_skipped"],
            "gz_failed": gz_stats["gz_failed"],
            "gz_errors": " | ".join(gz_stats["gz_errors"]),
            "error": f"{type(error).__name__}: {error}",
        }

    return row


def build_summary_text(report):
    """build readable text summary"""
    lines = []

    lines.append("Input validation summary")
    lines.append("")
    lines.append(f"Total samples: {len(report)}")

    if "status" in report.columns:
        lines.append("")
        lines.append("Status counts:")
        for key, value in report["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if "format" in report.columns:
        lines.append("")
        lines.append("Expression format counts:")
        for key, value in report["format"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    # spatial file availability
    missing_positions = report[report["has_positions"] == False]
    missing_scalefactors = report[report["has_scalefactors"] == False]
    missing_images = report[report["has_image"] == False]

    lines.append("")
    lines.append(f"Samples missing positions: {len(missing_positions)}")
    lines.append(f"Samples missing scalefactors: {len(missing_scalefactors)}")
    lines.append(f"Samples missing images: {len(missing_images)}")

    # gzip extraction summary
    if "gz_files_found" in report.columns:
        lines.append("")
        lines.append("GZ extraction:")
        lines.append(f"  gz files found: {int(report['gz_files_found'].sum())}")
        lines.append(f"  gz files created: {int(report['gz_created'].sum())}")
        lines.append(f"  gz files skipped: {int(report['gz_skipped'].sum())}")
        lines.append(f"  gz files failed: {int(report['gz_failed'].sum())}")

    errors = report[report["status"] == "ERROR"]

    lines.append("")
    lines.append(f"Failed samples: {len(errors)}")

    if len(errors) > 0:
        lines.append("")
        lines.append("Failure details:")
        for _, row in errors.iterrows():
            lines.append(f"  {row['sample_id']}: {row['error']}")

    gz_errors = report[
        report.get("gz_failed", pd.Series([0] * len(report))).fillna(0).astype(int) > 0
    ]

    if len(gz_errors) > 0:
        lines.append("")
        lines.append("GZ failure details:")
        for _, row in gz_errors.iterrows():
            lines.append(f"  {row['sample_id']}: {row['gz_errors']}")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main():
    """run input validation across sample folders"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-extract-gz", action="store_true")
    parser.add_argument("--overwrite-gz", action="store_true")
    args = parser.parse_args()

    cfg = validate_config(load_config(args.config))

    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    sample_glob = cfg.get("sample_glob", "SAMPLE_*")

    validation_dir = output_root / "output_01_validate_inputs"
    validation_dir.mkdir(parents=True, exist_ok=True)

    report_path = validation_dir / "input_validation_report.csv"
    summary_path = validation_dir / "input_validation_summary.txt"

    sample_dirs = sorted(
        sample_dir for sample_dir in input_root.glob(sample_glob)
        if sample_dir.is_dir()
    )

    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]

    print("=== Input validation ===")
    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Sample glob:", sample_glob)
    print("Samples found:", len(sample_dirs))
    print("Extract gz:", not args.no_extract_gz)
    print()

    rows = []

    for i, sample_dir in enumerate(sample_dirs, start=1):
        print(f"[{i}/{len(sample_dirs)}] Validating {sample_dir.name}")

        row = validate_one_sample(
            sample_dir,
            extract_gz=not args.no_extract_gz,
            overwrite_gz=args.overwrite_gz,
        )

        rows.append(row)

        # save progress after every sample
        pd.DataFrame(rows).to_csv(report_path, index=False)

        if row["status"] == "OK":
            print(
                f"  OK: {row['format']} | spots={row['n_spots']} | genes={row['n_genes']}"
            )
        else:
            print(f"  ERROR: {row['error']}")

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)

    summary_text = build_summary_text(report)
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print("DONE")
    print("Report:", report_path)
    print("Summary:", summary_path)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()

