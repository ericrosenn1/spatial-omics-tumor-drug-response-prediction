#!/usr/bin/env python
"""
Script:
    05_qc_and_package_transfer_outputs.py

Purpose:
    Run final QC for spatial_transfer_inference_model and create a transfer
    inference ZIP package.

Policy:
    Research-use only.
    Not a calibrated clinical drug-response predictor.
    Not a treatment recommendation package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import traceback
from typing import List

import pandas as pd

from _stim_utils import (
    add_qc,
    build_output_manifest,
    ensure_dir,
    open_folder,
    package_zip,
    read_table,
    report_status_from_qc,
    save_output_manifest,
    sha256_file,
    write_json,
    write_text_report,
    write_tsv,
)


EXPECTED_REPORTS = [
    "01_prepared_transfer_inputs/05_reports/step01_prepare_transfer_inputs_report.txt",
    "02_aligned_features/04_reports/step02_align_single_slide_features_to_v2_report.txt",
    "03_transfer_scores/05_reports/step03_score_transfer_drug_response_alignment_report.txt",
    "04_prediction_table/02_reports/step04_make_single_slide_prediction_table_report.txt",
]

EXPECTED_CORE_FILES = [
    "01_prepared_transfer_inputs/02_single_slide_feature_input/transfer_single_slide_feature_input.tsv",
    "02_aligned_features/01_aligned_feature_vectors/single_slide_v2_scaled_feature_vector.tsv",
    "03_transfer_scores/01_treatment_alignment_scores/single_slide_treatment_alignment_scores.tsv",
    "03_transfer_scores/02_feature_contributions/single_slide_treatment_feature_contributions.tsv",
    "03_transfer_scores/03_theme_contributions/single_slide_treatment_theme_contributions.tsv",
    "04_prediction_table/01_prediction_tables/single_slide_drug_response_interpretation_table.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--pim-run-root", required=True)
    parser.add_argument("--spatial-feature-run-root", default="")
    parser.add_argument("--single-slide-feature-table", default="")
    parser.add_argument("--sample-id", default="TRANSFER_SAMPLE_001")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def json_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"json_read_error:{exc}"
    return str(data.get("status", "status_missing")).lower()


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.readline().strip()


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    output_root = Path(args.output_root)

    step_root = output_root / "05_qc_and_transfer_package"
    qc_dir = step_root / "01_final_qc"
    manifest_dir = step_root / "02_manifests"
    package_dir = step_root / "03_transfer_zip"
    report_dir = step_root / "04_reports"

    for path in [qc_dir, manifest_dir, package_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        summary_rows = []
        for step in ["01", "02", "03", "04"]:
            path = output_root / f"spatial_transfer_inference_model_step{step}_summary.json"
            summary_rows.append({
                "step": step,
                "summary_json": str(path),
                "exists": path.exists(),
                "status": json_status(path),
            })
        summary_df = pd.DataFrame(summary_rows)
        write_tsv(qc_dir / "transfer_step_summary_status.tsv", summary_df)

        bad_summaries = summary_df[~summary_df["status"].isin(["pass"])]
        add_qc(qc, "step_01_to_04_summary_statuses_pass", "pass" if bad_summaries.empty else "fail", len(bad_summaries), 0, "All Step 01-04 summary JSON files should report pass.")

        missing_reports = []
        bad_filepath = []
        for rel in EXPECTED_REPORTS:
            path = output_root / rel
            if not path.exists():
                missing_reports.append(rel)
            else:
                try:
                    if not first_line(path).startswith("FILEPATH"):
                        bad_filepath.append(rel)
                except Exception:
                    bad_filepath.append(rel)

        add_qc(qc, "expected_txt_reports_exist", "pass" if not missing_reports else "fail", len(missing_reports), 0, "; ".join(missing_reports) if missing_reports else "All expected reports exist.")
        add_qc(qc, "txt_reports_start_with_filepath", "pass" if not bad_filepath else "fail", len(bad_filepath), 0, "; ".join(bad_filepath) if bad_filepath else "All generated txt reports start with FILEPATH.")

        missing_core = [rel for rel in EXPECTED_CORE_FILES if not (output_root / rel).exists()]
        add_qc(qc, "core_transfer_files_exist", "pass" if not missing_core else "fail", len(missing_core), 0, "; ".join(missing_core) if missing_core else "All core transfer files exist.")

        prediction_table_path = output_root / "04_prediction_table/01_prediction_tables/single_slide_drug_response_interpretation_table.tsv"
        prediction = read_table(prediction_table_path) if prediction_table_path.exists() else pd.DataFrame()

        n_rows = len(prediction)
        n_samples = prediction["sample_id"].nunique() if "sample_id" in prediction.columns and not prediction.empty else 0
        n_drugs = prediction["drug_key"].nunique() if "drug_key" in prediction.columns and not prediction.empty else 0

        add_qc(qc, "prediction_table_nonempty", "pass" if n_rows > 0 else "fail", n_rows, ">0", "Prediction interpretation table should be nonempty.")
        add_qc(qc, "prediction_table_samples", "pass" if n_samples >= 1 else "fail", n_samples, ">=1", "At least one sample should be represented.")
        add_qc(qc, "prediction_table_treatments", "pass" if n_drugs == 27 else "warn", n_drugs, 27, "Expected 27 PIM validated treatments.")
        add_qc(qc, "prediction_table_explanations", "pass" if "explanation" in prediction.columns and prediction["explanation"].astype(str).str.len().gt(20).all() else "fail", "present" if "explanation" in prediction.columns else "missing", "present and nonempty", "Readable explanation text should be present.")

        if "probability_effective_research_not_calibrated" in prediction.columns:
            vals = pd.to_numeric(prediction["probability_effective_research_not_calibrated"], errors="coerce")
            in_range = bool(vals.dropna().between(0, 1).all()) if vals.notna().any() else False
            add_qc(qc, "research_score_0_1_range", "pass" if in_range else "warn", in_range, True, "Research 0-1 score should be in [0,1].")

        all_files = [p for p in sorted(output_root.rglob("*")) if p.is_file()]
        package_manifest_rows = []
        for path in all_files:
            if "05_qc_and_transfer_package/03_transfer_zip" in path.as_posix():
                continue
            package_manifest_rows.append({
                "relative_path": path.relative_to(output_root).as_posix(),
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
                "sha256": sha256_file(path),
            })
        package_manifest = pd.DataFrame(package_manifest_rows)
        package_manifest_path = manifest_dir / "transfer_package_file_manifest.tsv"
        write_tsv(package_manifest_path, package_manifest)

        zip_path = package_dir / "spatial_transfer_inference_package.zip"
        packaged_files = package_zip(zip_path, output_root)
        zip_size = zip_path.stat().st_size if zip_path.exists() else 0

        add_qc(qc, "transfer_zip_exists", "pass" if zip_path.exists() else "fail", zip_path.exists(), True, "Transfer inference ZIP package created.")
        add_qc(qc, "transfer_zip_nonempty", "pass" if zip_size > 0 else "fail", zip_size, ">0", "Transfer inference ZIP package should be nonempty.")
        add_qc(qc, "transfer_package_file_count", "pass" if len(packaged_files) >= 20 else "warn", len(packaged_files), ">=20", "Transfer package should include outputs, reports, QC, and provenance.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        zip_path = package_dir / "spatial_transfer_inference_package.zip"
        package_manifest_path = manifest_dir / "transfer_package_file_manifest.tsv"
        packaged_files = []
        prediction = pd.DataFrame()

    status = report_status_from_qc(qc, errors)
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "transfer_final_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "sample_id": args.sample_id,
        "smoke_test": bool(args.smoke_test),
        "prediction_table_rows": len(prediction),
        "final_zip": str(zip_path),
        "final_package_manifest": str(package_manifest_path),
        "package_file_count": len(packaged_files),
        "qc_fail_count": int((qc_df["status"].astype(str).str.lower() == "fail").sum()) if not qc_df.empty else 0,
        "qc_warn_count": int((qc_df["status"].astype(str).str.lower() == "warn").sum()) if not qc_df.empty else 0,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "spatial_transfer_inference_model_step05_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL FINAL QC AND PACKAGE REPORT",
        "",
        f"status: {status}",
        f"sample_id: {args.sample_id}",
        f"smoke_test: {args.smoke_test}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Final package",
        f"zip_path: {zip_path}",
        f"package_manifest: {package_manifest_path}",
        f"package_file_count: {len(packaged_files)}",
        "",
        "Prediction table",
        str(output_root / "04_prediction_table/01_prediction_tables/single_slide_drug_response_interpretation_table.tsv"),
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Interpretation caveat",
        "The transfer inference package reports research-use spatial response alignment from frozen V2/PIM signed effects. It is not a calibrated clinical efficacy prediction and not a treatment recommendation.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    report_path = report_dir / "spatial_transfer_inference_final_qc_and_package_report.txt"
    write_text_report(report_path, "\n".join(report_lines))

    try:
        packaged_files = package_zip(zip_path, output_root)
        summary["package_file_count"] = len(packaged_files)
        summary["final_zip_size_bytes"] = zip_path.stat().st_size if zip_path.exists() else 0
        summary["final_qc_report"] = str(report_path)
        write_json(output_root / "spatial_transfer_inference_model_step05_summary.json", summary)
    except Exception as exc:
        warnings.append(f"Package refresh after report failed: {exc}")

    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL STEP 05 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"prediction_table_rows: {len(prediction)}")
    print(f"final_zip: {zip_path}")
    print(f"package_file_count: {summary.get('package_file_count', len(packaged_files))}")
    print(f"qc_fail_count: {summary['qc_fail_count']}")
    print(f"qc_warn_count: {summary['qc_warn_count']}")
    print(f"report: {report_path}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
