"""
Script: 02_build_modeling_dataset.py

Purpose:
    Build the governed modeling datasets for spatial_prediction_model_V2.

Pipeline role:
    This is Step 02 of the spatial prediction model v2 workflow. It consumes the
    validated teacher_builder handoff tables, loads the fused teacher labels,
    spatial numeric feature matrix, and feature manifest, then constructs the
    governed pair-level and broad residual modeling datasets used by downstream
    spatial response modeling steps.

Scientific role:
    The spatial model learns from residual response signal after treatment-prior
    governance, rather than from raw treatment-response probabilities alone. This
    dataset-builder step preserves the feature-governance manifest, treatment
    eligibility table, residual target columns, and run provenance so downstream
    modeling can be audited for feature inclusion, sample-treatment coverage, and
    independence from legacy V1 production outputs.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP02_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, imports, constants,
    paths, thresholds, schemas, output filenames, and return codes must remain
    unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from spm_v2.data_discovery import load_handoff_tables, validate_handoff_files
from spm_v2.dataset_builder import build_dataset_bundle
from spm_v2.io_utils import ensure_dir, write_json, write_table, write_text_report
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:

    """Build governed spatial_prediction_model_V2 modeling datasets and reports."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--residual-col", default="fused_residual_vs_prior")
    args = parser.parse_args()


    handoff_root = Path(args.handoff_root)
    output_root = ensure_dir(args.output_root)

    inputs_dir = ensure_dir(output_root / "01_input_sources")
    feature_dir = ensure_dir(output_root / "02_feature_governance")
    dataset_dir = ensure_dir(output_root / "03_modeling_datasets")
    reports_dir = ensure_dir(output_root / "04_reports")


    source_manifest = validate_handoff_files(handoff_root)
    write_table(source_manifest, inputs_dir / "input_source_manifest.tsv")


    teacher, spatial, manifest, paths = load_handoff_tables(handoff_root)


    bundle = build_dataset_bundle(
        teacher=teacher,
        spatial=spatial,
        manifest=manifest,
        residual_col=args.residual_col,
    )


    governance = bundle["all_feature_governance_manifest"]
    broad_feature_cols = bundle["broad_feature_cols"]
    spatial_features = bundle["spatial_features"]
    pair_dataset = bundle["pair_level_residual_dataset"]
    broad_dataset = bundle["broad_residual_dataset"]
    eligibility = bundle["treatment_eligibility"]
    feature_set_policy = bundle["feature_set_policy"]


    write_table(governance, feature_dir / "v2_all_feature_governance_manifest.tsv")
    write_table(
        governance.loc[governance["include_for_broad_governed_candidate_pool"] == True].copy(),
        feature_dir / "v2_broad_governed_candidate_features.tsv",
    )

    # Strict biology registry is intentionally empty at this stage; it is generated later from residual-model evidence.
    empty_strict = governance.loc[governance["include_for_v2_strict_biology_registry"] == True].copy()
    write_table(empty_strict, feature_dir / "v2_strict_biology_feature_registry_NOT_YET_GENERATED.tsv")


    write_table(spatial_features, dataset_dir / "v2_spatial_features_broad_governed_candidate_pool.tsv")
    write_table(pair_dataset, dataset_dir / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv")
    write_table(broad_dataset, dataset_dir / "v2_broad_residual_dataset_broad_governed_candidate_pool.tsv")
    write_table(eligibility, dataset_dir / "v2_treatment_eligibility.tsv")


    write_json(feature_set_policy, feature_dir / "v2_feature_set_policy.json")

    eligible_count = int((eligibility["eligible"] == True).sum()) if "eligible" in eligibility.columns else 0

    run_summary = {
        "status": "pass",
        "handoff_root": str(handoff_root),
        "output_root": str(output_root),
        "n_teacher_rows": int(len(teacher)),
        "n_spatial_samples": int(spatial["sample_id"].astype(str).nunique()),
        "n_feature_manifest_rows": int(len(manifest)),
        "n_broad_governed_candidate_features": int(len(broad_feature_cols)),
        "n_v2_strict_biology_registry_features": 0,
        "strict_biology_registry_status": "not_generated_yet",
        "strict_biology_registry_generation_step": "Step 05 residual biology registry using Step 04 residual pair model evidence",
        "n_pair_level_rows": int(len(pair_dataset)),
        "n_pair_level_columns": int(len(pair_dataset.columns)),
        "n_broad_residual_samples": int(len(broad_dataset)),
        "n_broad_residual_columns": int(len(broad_dataset.columns)),
        "n_treatments_total": int(len(eligibility)),
        "n_treatments_eligible": eligible_count,
        "production_dependency_on_v1_outputs": "no",
        "canonical_v1_scripts_modified_by_v2": "no",
    }

    write_json(run_summary, output_root / "v2_dataset_builder_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)


    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 DATASET BUILDER REPORT")
    report_lines.append("")
    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    report_lines.append("Feature set policy")
    report_lines.append(json.dumps(feature_set_policy, indent=2))
    report_lines.append("")
    report_lines.append("Feature governance class counts")
    report_lines.append(governance["governance_class"].value_counts(dropna=False).to_string())
    report_lines.append("")
    report_lines.append("Broad governed candidate pool count")
    report_lines.append(str(len(broad_feature_cols)))
    report_lines.append("")
    report_lines.append("Treatment eligibility counts")
    report_lines.append(eligibility["eligible"].value_counts(dropna=False).to_string())

    report_path = write_text_report(reports_dir / "v2_dataset_builder_report.txt", "\n".join(report_lines))


    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Handoff root: {handoff_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Broad governed candidate features: {len(broad_feature_cols)}",
        "V2 strict biology registry features: 0",
        "V2 strict biology registry status: not generated yet",
        "Production dependency on V1 outputs: no",
        f"Pair level rows: {len(pair_dataset)}",
        f"Broad residual samples: {len(broad_dataset)}",
        f"Eligible treatments: {eligible_count}",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 DATASET BUILDER COMPLETE", terminal_lines))
    print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())

