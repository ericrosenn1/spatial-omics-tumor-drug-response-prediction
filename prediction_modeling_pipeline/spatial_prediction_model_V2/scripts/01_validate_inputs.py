"""
Script: 01_validate_inputs.py

Purpose:
    Validate configured inputs for the spatial_prediction_model_V2 workflow.

Pipeline role:
    This is Step 01 of the spatial prediction model v2 pipeline. It is intended
    to run before model-table construction or model training so that missing
    paths, malformed tables, absent columns, incompatible schemas, or incomplete
    upstream teacher/spatial-feature outputs are caught early and reported in a
    reproducible way.

Scientific role:
    The spatial prediction model depends on a governed teacher-label handoff and
    sample-matched spatial feature tables. This validation step protects the
    downstream modeling analysis by making input assumptions explicit before any
    response-prediction model is trained or evaluated.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP01_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, imports, constants,
    paths, thresholds, schemas, validation rules, and outputs must remain unchanged.
"""



# =============================================================================
# Imports and dependencies
# =============================================================================
# Core imports used by the validation step.

from __future__ import annotations



# =============================================================================
# Command-line interface
# =============================================================================
# Argument parsing keeps the validation step runnable from the pipeline runner.

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.data_discovery import (
    column_presence_summary,
    load_handoff_tables,
    table_shape_summary,
    validate_handoff_files,
)
from spm_v2.io_utils import ensure_dir, write_json, write_table, write_text_report
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest




# =============================================================================
# Step 01 validation workflow
# =============================================================================
# Main workflow: load config, check inputs, validate schemas, and write reports.

def main() -> int:
    """Run spatial_prediction_model_V2 input validation and write audit outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    handoff_root = Path(args.handoff_root)
    output_root = ensure_dir(args.output_root)

    tables_dir = ensure_dir(output_root / "01_validation_tables")
    reports_dir = ensure_dir(output_root / "02_reports")

    source_manifest = validate_handoff_files(handoff_root)
    write_table(source_manifest, tables_dir / "input_source_manifest.tsv")

    teacher, spatial, manifest, paths = load_handoff_tables(handoff_root)

    shape_summary = table_shape_summary({
        "teacher": teacher,
        "spatial_numeric": spatial,
        "feature_manifest": manifest,
    })
    write_table(shape_summary, tables_dir / "input_table_shape_summary.tsv")

    required_teacher_cols = [
        "sample_id",
        "drug_key",
        "fused_residual_vs_prior",
        "fused_prob_responder",
    ]
    required_spatial_cols = ["sample_id"]

    presence = pd.concat([
        column_presence_summary(teacher, required_teacher_cols, "teacher"),
        column_presence_summary(spatial, required_spatial_cols, "spatial_numeric"),
    ], ignore_index=True)
    write_table(presence, tables_dir / "required_column_presence.tsv")

    missing_required = presence.loc[(presence["present"] == False) & (presence["column"] != "fused_prob_responder")].copy()

    teacher_summary = {
        "n_teacher_rows": int(len(teacher)),
        "n_teacher_columns": int(len(teacher.columns)),
        "n_unique_samples_teacher": int(teacher["sample_id"].astype(str).nunique()) if "sample_id" in teacher.columns else 0,
        "n_unique_treatments": int(teacher["drug_key"].astype(str).nunique()) if "drug_key" in teacher.columns else 0,
        "has_fused_residual_vs_prior": "fused_residual_vs_prior" in teacher.columns,
        "has_fused_prob_responder": "fused_prob_responder" in teacher.columns,
    }

    spatial_summary = {
        "n_spatial_rows": int(len(spatial)),
        "n_spatial_columns": int(len(spatial.columns)),
        "n_unique_samples_spatial": int(spatial["sample_id"].astype(str).nunique()) if "sample_id" in spatial.columns else 0,
    }

    residual_stats = {}
    if "fused_residual_vs_prior" in teacher.columns:
        vals = pd.to_numeric(teacher["fused_residual_vs_prior"], errors="coerce")
        residual_stats = {
            "residual_nonmissing": int(vals.notna().sum()),
            "residual_mean": float(vals.mean()),
            "residual_std": float(vals.std()),
            "residual_min": float(vals.min()),
            "residual_max": float(vals.max()),
        }

    validation_status = "pass" if len(missing_required) == 0 else "fail"

    run_summary = {
        "status": validation_status,
        "handoff_root": str(handoff_root),
        "output_root": str(output_root),
        **teacher_summary,
        **spatial_summary,
        **residual_stats,
        "missing_required_columns": int(len(missing_required)),
    }

    write_json(run_summary, output_root / "v2_input_validation_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 INPUT VALIDATION REPORT")
    report_lines.append("")
    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    report_lines.append("Input table shape summary")
    report_lines.append(shape_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("Required column presence")
    report_lines.append(presence.to_string(index=False))

    report_path = write_text_report(reports_dir / "v2_input_validation_report.txt", "\n".join(report_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        f"Status: {validation_status}",
        f"Handoff root: {handoff_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Teacher rows: {teacher_summary['n_teacher_rows']}",
        f"Spatial rows: {spatial_summary['n_spatial_rows']}",
        f"Treatments: {teacher_summary['n_unique_treatments']}",
        f"Missing required columns: {len(missing_required)}",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 INPUT VALIDATION COMPLETE", terminal_lines))
    print("")

    if validation_status != "pass":
        return 2

    return 0




# =============================================================================
# Script entry point
# =============================================================================
# The step is executable as a standalone script and through the pipeline runner.

if __name__ == "__main__":
    raise SystemExit(main())
