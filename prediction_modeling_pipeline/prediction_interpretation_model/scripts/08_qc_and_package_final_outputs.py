#!/usr/bin/env python
"""
Script:
    08_qc_and_package_final_outputs.py

Description:
    Runs final QC across prediction_interpretation_model outputs and creates the
    final ZIP package containing interpretation tables, reports, figures, manifests,
    treatment cards, mechanism atlas outputs, and provenance files.

Instructions:
    Run after Step 07. The package is publishable only when final QC reports zero
    fail checks and expected final files are present.

Source-truth policy:
    This step packages completed interpretation outputs. It does not rerun V2,
    train models, select models, or make clinical treatment recommendations.
"""

# =============================================================================
# PIM_DOCS_PATCH: RUN AND MAINTENANCE INSTRUCTIONS
# =============================================================================
# Run numbered scripts through 00_run_prediction_interpretation_model.py unless
# debugging a single step. Treat the V2 full-run root as read-only source truth.
# Every generated .txt report must start with FILEPATH, and terminal summaries
# should remain concise enough for copy/paste debugging.
# =============================================================================


# =============================================================================
# PIM_DOCS_SECTION: imports and dependencies
# =============================================================================
# Keep imports explicit and standard-library-first where practical. The pipeline
# expects local scripts to run from the scripts directory or through the orchestrator.

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import traceback
from typing import List, Tuple
import zipfile

import pandas as pd

from _pim_utils import (
    add_qc,
    ensure_dir,
    load_prepared_index,
    open_folder,
    read_table,
    save_output_manifest,
    write_json,
    write_text_report,
    write_tsv,
)


# =============================================================================
# PIM_DOCS_SECTION: constants and source contracts
# =============================================================================
# Constants define expected files, output names, QC contracts, or reporting rules.

EXPECTED_REPORTS = [
    "01_prepared_inputs/05_reports/step01_prepare_interpretation_inputs_report.txt",
    "02_feature_and_treatment_dictionary/04_reports/step02_feature_and_treatment_dictionary_report.txt",
    "03_signed_spatial_effects/06_reports/step03_signed_spatial_effects_report.txt",
    "04_treatment_interpretation_cards/04_reports/step04_treatment_interpretation_cards_report.txt",
    "05_sample_level_interpretations/06_reports/step05_sample_level_interpretations_report.txt",
    "06_mechanism_atlas/07_reports/step06_mechanism_atlas_report.txt",
    "07_final_outputs/04_final_reports/prediction_interpretation_model_final_report.txt",
    "07_final_outputs/04_final_reports/final_methods_results_discussion.txt",
    "07_final_outputs/04_final_reports/final_figure_captions.txt",
]

EXPECTED_QC_TABLES = [
    "01_prepared_inputs/01_source_manifests/v2_source_qc_checks.tsv",
    "02_feature_and_treatment_dictionary/03_contracts/step02_dictionary_qc_checks.tsv",
    "03_signed_spatial_effects/05_qc/step03_signed_effect_qc_checks.tsv",
    "04_treatment_interpretation_cards/03_qc/step04_treatment_card_qc_checks.tsv",
    "05_sample_level_interpretations/05_qc/step05_sample_level_qc_checks.tsv",
    "06_mechanism_atlas/06_qc/step06_mechanism_atlas_qc_checks.tsv",
    "07_final_outputs/05_manifests/step07_final_outputs_qc_checks.tsv",
]

CORE_FINAL_FILES = [
    "07_final_outputs/01_publication_tables_tsv/Final_Treatment_Interpretation_Cards.tsv",
    "07_final_outputs/01_publication_tables_tsv/Final_Cross_Treatment_Biology_Theme_Atlas.tsv",
    "07_final_outputs/01_publication_tables_tsv/Final_Cross_Treatment_Feature_Atlas.tsv",
    "07_final_outputs/01_publication_tables_tsv/Final_Sample_Treatment_Interpretation_Scores.tsv",
    "07_final_outputs/01_publication_tables_tsv/Final_Sample_Interpretation_Summary.tsv",
    "07_final_outputs/01_publication_tables_tsv/Final_Sample_Mechanism_Summary.tsv",
    "07_final_outputs/02_publication_workbook/prediction_interpretation_model_final_publication_tables.xlsx",
    "07_final_outputs/05_manifests/final_figure_manifest.tsv",
    "07_final_outputs/05_manifests/final_publication_table_manifest.tsv",
]


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    Defaults preserve local project paths while allowing explicit overrides."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, max_bytes: int = 128 * 1024 * 1024) -> str:
    """Compute a SHA256 digest for a package file.
    Large files can be skipped to keep QC runtime bounded."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if path.stat().st_size > max_bytes:
        return "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_line(path: Path) -> str:
    """Read the first line of a text file.
    Used to verify FILEPATH-first report convention."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.readline().strip()


def read_json_status(path: Path) -> str:
    """Read a status value from a JSON summary.
    Returns explicit missing or read-error status text when needed."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"json_read_error:{exc}"
    return str(data.get("status", "status_missing")).lower()


def all_files_under(root: Path) -> List[Path]:
    """List all files under a folder.
    Used by final packaging and manifest generation."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not root.exists():
        return []
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def should_package(path: Path, output_root: Path) -> bool:
    """Decide whether a file should enter the final ZIP.
    Excludes logs, backups, caches, and intermediate packaging artifacts."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rel = path.relative_to(output_root).as_posix()

    excluded_fragments = [
        "/02_copied_v2_tables/packages/",
        "/02_copied_v2_tables/tables/per_treatment_test_predictions_long.tsv",
        "/02_copied_v2_tables/tables/broad_residual_feature_evidence_long.tsv",
        "/02_copied_v2_tables/tables/curated_tier1_tier2_shap_feature_long.tsv",
        "/02_copied_v2_tables/tables/per_treatment_final_shap_feature_long.tsv",
        "/02_copied_v2_tables/tables/v2_broad_residual_dataset.tsv",
        "/02_copied_v2_tables/tables/v2_spatial_features_broad_pool.tsv",
    ]
    if any(fragment in "/" + rel for fragment in excluded_fragments):
        return False

    include_prefixes = [
        "01_prepared_inputs/01_source_manifests/",
        "01_prepared_inputs/04_pair_level_pointer/",
        "01_prepared_inputs/05_reports/",
        "01_prepared_inputs/README_prepared_inputs.txt",
        "01_prepared_inputs/resolved_config.json",
        "01_prepared_inputs/environment_and_provenance",
        "02_feature_and_treatment_dictionary/",
        "03_signed_spatial_effects/",
        "04_treatment_interpretation_cards/",
        "05_sample_level_interpretations/",
        "06_mechanism_atlas/",
        "07_final_outputs/",
        "08_qc_and_final_package/",
        "pipeline_run_logs/",
        "prediction_interpretation_model_orchestrator_report.txt",
        "prediction_interpretation_model_orchestrator_step_manifest.tsv",
        "prediction_interpretation_model_output_manifest.tsv",
        "prediction_interpretation_model_run_summary.json",
        "prediction_interpretation_model_step",
    ]

    return any(rel.startswith(prefix) for prefix in include_prefixes)


def build_package_file_manifest(files: List[Path], output_root: Path) -> pd.DataFrame:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows = []
    for path in files:
        try:
            rows.append({
                "relative_path": path.relative_to(output_root).as_posix(),
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
                "sha256": sha256_file(path),
            })
        except Exception as exc:
            rows.append({
                "relative_path": str(path),
                "absolute_path": str(path),
                "size_bytes": "",
                "suffix": path.suffix.lower(),
                "sha256": "",
                "error": str(exc),
            })
    return pd.DataFrame(rows)


def zip_files(zip_path: Path, files: List[Path], output_root: Path) -> None:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            if path == zip_path:
                continue
            rel = path.relative_to(output_root).as_posix()
            zf.write(path, rel)


def qc_table_fail_warn_counts(path: Path) -> Tuple[int, int, int]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        return 0, 0, 0
    df = read_table(path)
    if "status" not in df.columns:
        return 0, 0, len(df)
    status = df["status"].astype(str).str.lower()
    return int((status == "fail").sum()), int((status == "warn").sum()), len(df)


# =============================================================================
# PIM_DOCS_SECTION: main entry point
# =============================================================================
# The main function wires inputs, output folders, QC checks, reports, and terminal summaries.

def main() -> int:
    """Run the script's command-line workflow.
    Writes outputs, QC checks, summaries, and terminal status messages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    args = parse_args()
    started = dt.datetime.now()
    output_root = Path(args.output_root)
    prepared_root, _ = load_prepared_index(output_root, Path(args.prepared_input_root) if args.prepared_input_root else None)

    step_root = output_root / "08_qc_and_final_package"
    qc_dir = step_root / "01_final_qc"
    manifest_dir = step_root / "02_manifests"
    package_dir = step_root / "03_final_zip"
    report_dir = step_root / "04_reports"

    for path in [qc_dir, manifest_dir, package_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        missing_reports = []
        bad_filepath_reports = []
        for rel in EXPECTED_REPORTS:
            path = output_root / rel
            if not path.exists():
                missing_reports.append(rel)
            else:
                try:
                    if not first_line(path).startswith("FILEPATH"):
                        bad_filepath_reports.append(rel)
                except Exception:
                    bad_filepath_reports.append(rel)

        add_qc(qc, "expected_txt_reports_exist", "pass" if not missing_reports else "fail", len(missing_reports), 0, "; ".join(missing_reports) if missing_reports else "All expected txt reports exist.")
        add_qc(qc, "txt_reports_start_with_filepath", "pass" if not bad_filepath_reports else "fail", len(bad_filepath_reports), 0, "; ".join(bad_filepath_reports) if bad_filepath_reports else "All generated txt reports start with FILEPATH.")

        summary_rows = []
        for step in ["01", "02", "03", "04", "05", "06", "07"]:
            path = output_root / f"prediction_interpretation_model_step{step}_summary.json"
            status = read_json_status(path)
            summary_rows.append({
                "step": step,
                "summary_json": str(path),
                "exists": path.exists(),
                "status": status,
            })
        summary_df = pd.DataFrame(summary_rows)
        write_tsv(qc_dir / "final_step_summary_status.tsv", summary_df)

        bad_summaries = summary_df[~summary_df["status"].isin(["pass"])]
        add_qc(qc, "step_summary_statuses_pass", "pass" if bad_summaries.empty else "fail", len(bad_summaries), 0, "All Step 01-07 summary JSON files should report pass.")

        qc_status_rows = []
        total_fail = 0
        total_warn = 0
        missing_qc_tables = []
        for rel in EXPECTED_QC_TABLES:
            path = output_root / rel
            if not path.exists():
                missing_qc_tables.append(rel)
                qc_status_rows.append({
                    "qc_table": rel,
                    "exists": False,
                    "rows": "",
                    "fail_count": "",
                    "warn_count": "",
                })
                continue
            fail_count, warn_count, rows = qc_table_fail_warn_counts(path)
            total_fail += fail_count
            total_warn += warn_count
            qc_status_rows.append({
                "qc_table": rel,
                "exists": True,
                "rows": rows,
                "fail_count": fail_count,
                "warn_count": warn_count,
            })

        qc_status_df = pd.DataFrame(qc_status_rows)
        write_tsv(qc_dir / "final_upstream_qc_table_status.tsv", qc_status_df)

        add_qc(qc, "expected_qc_tables_exist", "pass" if not missing_qc_tables else "fail", len(missing_qc_tables), 0, "; ".join(missing_qc_tables) if missing_qc_tables else "All expected QC tables exist.")
        add_qc(qc, "upstream_qc_fail_count", "pass" if total_fail == 0 else "fail", total_fail, 0, "No upstream QC table should contain status=fail.")
        add_qc(qc, "upstream_qc_warning_count", "pass" if total_warn <= 1 else "warn", total_warn, "<=1", "Warnings are allowed if documented; Step 05 sample coverage warning is expected.")

        missing_core = [rel for rel in CORE_FINAL_FILES if not (output_root / rel).exists()]
        add_qc(qc, "core_final_files_exist", "pass" if not missing_core else "fail", len(missing_core), 0, "; ".join(missing_core) if missing_core else "All core final files exist.")

        cards_path = output_root / "07_final_outputs/01_publication_tables_tsv/Final_Treatment_Interpretation_Cards.tsv"
        theme_path = output_root / "07_final_outputs/01_publication_tables_tsv/Final_Cross_Treatment_Biology_Theme_Atlas.tsv"
        feature_dict_path = output_root / "07_final_outputs/01_publication_tables_tsv/Final_Feature_Dictionary.tsv"
        sample_scores_path = output_root / "07_final_outputs/01_publication_tables_tsv/Final_Sample_Treatment_Interpretation_Scores.tsv"
        figure_manifest_path = output_root / "07_final_outputs/05_manifests/final_figure_manifest.tsv"

        cards = read_table(cards_path) if cards_path.exists() else pd.DataFrame()
        themes = read_table(theme_path) if theme_path.exists() else pd.DataFrame()
        feature_dict = read_table(feature_dict_path) if feature_dict_path.exists() else pd.DataFrame()
        sample_scores = read_table(sample_scores_path) if sample_scores_path.exists() else pd.DataFrame()
        figure_manifest = read_table(figure_manifest_path) if figure_manifest_path.exists() else pd.DataFrame()

        validated_treatments = int(cards["drug_key"].nunique()) if "drug_key" in cards.columns else 0
        theme_count = int(themes["biological_theme"].nunique()) if "biological_theme" in themes.columns else 0
        feature_count = int(feature_dict["feature_name"].nunique()) if "feature_name" in feature_dict.columns else 0
        sample_count = int(sample_scores["sample_id"].nunique()) if "sample_id" in sample_scores.columns else 0
        score_rows = len(sample_scores)
        figure_count = len(figure_manifest)

        add_qc(qc, "final_validated_treatment_count", "pass" if validated_treatments == 27 else "fail", validated_treatments, 27, "Final cards should represent 27 label-shuffle-validated treatments.")
        add_qc(qc, "final_biology_theme_count", "pass" if theme_count == 11 else "warn", theme_count, 11, "Final mechanism atlas should represent 11 recurrent biology themes.")
        add_qc(qc, "final_strict_feature_count", "pass" if feature_count == 139 else "warn", feature_count, 139, "Final feature dictionary should represent 139 strict biology features.")
        add_qc(qc, "final_sample_score_rows", "pass" if score_rows > 0 else "fail", score_rows, ">0", "Final sample-treatment interpretation scores should be present.")
        add_qc(qc, "final_sample_coverage_documented", "pass" if sample_count >= 90 else "warn", sample_count, ">=90", "Sample coverage reflects validated-treatment eligible rows and is documented.")
        add_qc(qc, "final_figure_count", "pass" if figure_count >= 5 else "warn", figure_count, ">=5", "Final figure manifest should include presentation-ready figures.")

        all_paths_text = "\n".join(str(p).lower() for p in all_files_under(output_root))
        deprecated_markers = ["old_deprecated_pipeline", "prediction modeling\\code_prediction modeling\\prediction interpretation model"]
        deprecated_hit = any(marker.lower() in all_paths_text for marker in deprecated_markers)
        add_qc(qc, "deprecated_outputs_not_used_as_source_truth", "pass" if not deprecated_hit else "fail", deprecated_hit, False, "Final package should not contain deprecated pipeline source-truth paths.")

        candidate_files = [p for p in all_files_under(output_root) if should_package(p, output_root)]
        package_manifest = build_package_file_manifest(candidate_files, output_root)
        package_manifest_path = manifest_dir / "final_package_file_manifest.tsv"
        write_tsv(package_manifest_path, package_manifest)

        zip_path = package_dir / "prediction_interpretation_model_final_package.zip"
        candidate_files = [p for p in all_files_under(output_root) if should_package(p, output_root)]
        zip_files(zip_path, candidate_files, output_root)

        zip_size = zip_path.stat().st_size if zip_path.exists() else 0
        add_qc(qc, "final_zip_exists", "pass" if zip_path.exists() else "fail", zip_path.exists(), True, "Final ZIP package created.")
        add_qc(qc, "final_zip_nonempty", "pass" if zip_size > 0 else "fail", zip_size, ">0", "Final ZIP package should be nonempty.")
        add_qc(qc, "final_package_file_count", "pass" if len(candidate_files) >= 50 else "warn", len(candidate_files), ">=50", "Final ZIP package should include final outputs and supporting evidence.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        zip_path = package_dir / "prediction_interpretation_model_final_package.zip"
        package_manifest_path = manifest_dir / "final_package_file_manifest.tsv"
        candidate_files = []

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "final_qc_checks.tsv", qc_df)

    final_summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "final_zip": str(zip_path),
        "final_package_manifest": str(package_manifest_path),
        "package_file_count": len(candidate_files),
        "qc_fail_count": int((qc_df["status"].astype(str).str.lower() == "fail").sum()) if not qc_df.empty and "status" in qc_df.columns else 0,
        "qc_warn_count": int((qc_df["status"].astype(str).str.lower() == "warn").sum()) if not qc_df.empty and "status" in qc_df.columns else 0,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step08_summary.json", final_summary)

    final_qc_report_lines = [
        "PREDICTION INTERPRETATION MODEL FINAL QC AND PACKAGE REPORT",
        "",
        f"status: {status}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"prepared_input_root: {prepared_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Final package",
        f"zip_path: {zip_path}",
        f"package_manifest: {package_manifest_path}",
        f"package_file_count: {len(candidate_files)}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Interpretation caveat",
        "The final ZIP contains biological interpretation outputs derived from spatial residual model associations. It does not contain causal proof and is not a clinical decision package.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    report_path = report_dir / "prediction_interpretation_model_final_qc_and_package_report.txt"
    write_text_report(report_path, "\n".join(final_qc_report_lines))

    try:
        final_summary["final_qc_report"] = str(report_path)
        write_json(output_root / "prediction_interpretation_model_step08_summary.json", final_summary)
        candidate_files = [p for p in all_files_under(output_root) if should_package(p, output_root)]
        package_manifest = build_package_file_manifest(candidate_files, output_root)
        write_tsv(package_manifest_path, package_manifest)
        candidate_files = [p for p in all_files_under(output_root) if should_package(p, output_root)]
        zip_files(zip_path, candidate_files, output_root)
        final_summary["package_file_count"] = len(candidate_files)
        final_summary["final_zip_size_bytes"] = zip_path.stat().st_size if zip_path.exists() else 0
        write_json(output_root / "prediction_interpretation_model_step08_summary.json", final_summary)
    except Exception as exc:
        errors.append(f"Final package refresh failed after report writing: {exc}")

    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 08 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"final_zip: {zip_path}")
    print(f"package_file_count: {len(candidate_files)}")
    print(f"qc_fail_count: {final_summary['qc_fail_count']}")
    print(f"qc_warn_count: {final_summary['qc_warn_count']}")
    print(f"report: {report_path}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())

