#!/usr/bin/env python
"""
Script:
    01_prepare_interpretation_inputs.py

Description:
    Validates the frozen spatial_prediction_model_V2 full-run root and prepares
    the source tables/reports needed by the final biological interpretation layer.
    Large V2 files are referenced or copied according to explicit source policies.

Instructions:
    Run this step first for every prediction_interpretation_model run. Confirm
    that the source manifest, key-count checks, and Step 12 V2 QC checks pass
    before running dictionary, signed-effect, treatment-card, or final-output steps.

Source-truth policy:
    This is an ingestion and source-contract step only. It does not rerun V2,
    train models, select models, or use deprecated outputs as final source truth.
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
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    import pandas as pd
except Exception as exc:
    print("ERROR: pandas is required for Step 01.")
    print(str(exc))
    raise


# =============================================================================
# PIM_DOCS_SECTION: constants and source contracts
# =============================================================================
# Constants define expected files, output names, QC contracts, or reporting rules.

SOURCE_FILES = [
    # Run-level V2 provenance and orchestrator outputs.
    {
        "source_id": "v2_orchestrator_run_summary",
        "relative_path": "v2_orchestrator_run_summary.json",
        "category": "run_level",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_orchestrator_report",
        "relative_path": "v2_orchestrator_report.txt",
        "category": "run_level",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_orchestrator_step_manifest",
        "relative_path": "v2_orchestrator_step_manifest.tsv",
        "category": "run_level",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 02 datasets and feature governance.
    {
        "source_id": "v2_pair_level_residual_dataset",
        "relative_path": "02_build_modeling_dataset/03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv",
        "category": "modeling_dataset",
        "required": True,
        "copy_policy": "pointer",
    },
    {
        "source_id": "v2_broad_residual_dataset",
        "relative_path": "02_build_modeling_dataset/03_modeling_datasets/v2_broad_residual_dataset_broad_governed_candidate_pool.tsv",
        "category": "modeling_dataset",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_spatial_features_broad_pool",
        "relative_path": "02_build_modeling_dataset/03_modeling_datasets/v2_spatial_features_broad_governed_candidate_pool.tsv",
        "category": "modeling_dataset",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_treatment_eligibility",
        "relative_path": "02_build_modeling_dataset/03_modeling_datasets/v2_treatment_eligibility.tsv",
        "category": "modeling_dataset",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_all_feature_governance_manifest",
        "relative_path": "02_build_modeling_dataset/02_feature_governance/v2_all_feature_governance_manifest.tsv",
        "category": "feature_governance",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_broad_governed_candidate_features",
        "relative_path": "02_build_modeling_dataset/02_feature_governance/v2_broad_governed_candidate_features.tsv",
        "category": "feature_governance",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_feature_set_policy",
        "relative_path": "02_build_modeling_dataset/02_feature_governance/v2_feature_set_policy.json",
        "category": "feature_governance",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 03 probability baseline.
    {
        "source_id": "probability_baseline_metric_summary",
        "relative_path": "03_probability_baseline/02_metrics/probability_baseline_metric_summary.tsv",
        "category": "probability_baseline",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "probability_baseline_spatial_vs_treatment",
        "relative_path": "03_probability_baseline/03_feature_evidence/probability_baseline_spatial_vs_treatment_contribution.tsv",
        "category": "probability_baseline",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 04 pair-level prior-adjusted residual model.
    {
        "source_id": "pair_level_residual_metric_summary",
        "relative_path": "04_pair_level_residual_model/02_metrics/pair_level_residual_metric_summary.tsv",
        "category": "pair_level_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pair_level_residual_feature_evidence_summary",
        "relative_path": "04_pair_level_residual_model/03_feature_evidence_for_step05/pair_level_residual_feature_evidence_summary.tsv",
        "category": "pair_level_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pair_level_residual_feature_evidence_long",
        "relative_path": "04_pair_level_residual_model/03_feature_evidence_for_step05/pair_level_residual_feature_evidence_long.tsv",
        "category": "pair_level_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "spatial_feature_evidence_for_step05",
        "relative_path": "04_pair_level_residual_model/03_feature_evidence_for_step05/spatial_feature_evidence_for_step05.tsv",
        "category": "pair_level_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pair_level_residual_spatial_vs_treatment",
        "relative_path": "04_pair_level_residual_model/03_feature_evidence_for_step05/pair_level_residual_spatial_vs_treatment_contribution.tsv",
        "category": "pair_level_residual_model",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 05 strict biology registry.
    {
        "source_id": "v2_strict_biology_feature_registry",
        "relative_path": "05_residual_biology_registry/03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv",
        "category": "strict_biology_registry",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "top_residual_biology_features_strict",
        "relative_path": "05_residual_biology_registry/03_v2_strict_biology_registry/top_residual_biology_features_strict.csv",
        "category": "strict_biology_registry",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_all_residual_spatial_features_classified",
        "relative_path": "05_residual_biology_registry/02_classified_residual_features/v2_all_residual_spatial_features_classified.tsv",
        "category": "strict_biology_registry",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_residual_biology_theme_summary",
        "relative_path": "05_residual_biology_registry/04_theme_summary/v2_residual_biology_theme_summary.tsv",
        "category": "strict_biology_registry",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 06 broad residual model.
    {
        "source_id": "broad_residual_metrics_long",
        "relative_path": "06_broad_residual_model/02_model_metrics/broad_residual_metrics_long.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "broad_residual_target_summary",
        "relative_path": "06_broad_residual_model/02_model_metrics/broad_residual_target_summary.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "broad_residual_test_predictions_long",
        "relative_path": "06_broad_residual_model/02_model_metrics/broad_residual_test_predictions_long.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "broad_residual_feature_evidence_long",
        "relative_path": "06_broad_residual_model/03_feature_evidence/broad_residual_feature_evidence_long.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "broad_residual_feature_evidence_summary",
        "relative_path": "06_broad_residual_model/03_feature_evidence/broad_residual_feature_evidence_summary.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "broad_residual_theme_evidence_summary",
        "relative_path": "06_broad_residual_model/03_feature_evidence/broad_residual_theme_evidence_summary.tsv",
        "category": "broad_residual_model",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 07 per-treatment residual model outputs.
    {
        "source_id": "per_treatment_screening_summary",
        "relative_path": "07_filtered_per_treatment_residual_models/02_screening_metrics/per_treatment_screening_summary.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "per_treatment_test_predictions_long",
        "relative_path": "07_filtered_per_treatment_residual_models/02_screening_metrics/per_treatment_test_predictions_long.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "final_model_manifest",
        "relative_path": "07_filtered_per_treatment_residual_models/03_final_models/final_model_manifest.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "selected_treatments_for_final_shap",
        "relative_path": "07_filtered_per_treatment_residual_models/03_final_models/selected_treatments_for_final_shap.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "per_treatment_final_shap_feature_long",
        "relative_path": "07_filtered_per_treatment_residual_models/04_shap_feature_evidence/per_treatment_final_shap_feature_long.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "per_treatment_final_top10_features",
        "relative_path": "07_filtered_per_treatment_residual_models/04_shap_feature_evidence/per_treatment_final_top10_features.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "per_treatment_recurrent_spatial_features",
        "relative_path": "07_filtered_per_treatment_residual_models/05_recurrent_features_and_themes/per_treatment_recurrent_spatial_features.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "per_treatment_recurrent_biology_themes",
        "relative_path": "07_filtered_per_treatment_residual_models/05_recurrent_features_and_themes/per_treatment_recurrent_biology_themes.tsv",
        "category": "per_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 08 curated treatment models.
    {
        "source_id": "curated_treatment_model_table",
        "relative_path": "08_curated_per_treatment_residual_models/02_curated_treatment_models/curated_treatment_model_table.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "tier1_high_confidence_treatment_models",
        "relative_path": "08_curated_per_treatment_residual_models/02_curated_treatment_models/tier1_high_confidence_treatment_models.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "tier2_screening_signal_treatment_models",
        "relative_path": "08_curated_per_treatment_residual_models/02_curated_treatment_models/tier2_screening_signal_treatment_models.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "curated_recurrent_spatial_features",
        "relative_path": "08_curated_per_treatment_residual_models/03_recurrent_spatial_features/curated_recurrent_spatial_features.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "curated_tier1_tier2_shap_feature_long",
        "relative_path": "08_curated_per_treatment_residual_models/03_recurrent_spatial_features/curated_tier1_tier2_shap_feature_long.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "curated_recurrent_biology_themes",
        "relative_path": "08_curated_per_treatment_residual_models/04_recurrent_biology_themes/curated_recurrent_biology_themes.tsv",
        "category": "curated_treatment_models",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 09 label-shuffle validation.
    {
        "source_id": "tier1_label_shuffle_validation_results",
        "relative_path": "09_tier1_label_shuffle_validation/04_validation_results/tier1_label_shuffle_validation_results.tsv",
        "category": "label_shuffle_validation",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "tier1_label_shuffle_validated_treatments",
        "relative_path": "09_tier1_label_shuffle_validation/04_validation_results/tier1_label_shuffle_validated_treatments.tsv",
        "category": "label_shuffle_validation",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "tier1_label_shuffle_not_validated_treatments",
        "relative_path": "09_tier1_label_shuffle_validation/04_validation_results/tier1_label_shuffle_not_validated_treatments.tsv",
        "category": "label_shuffle_validation",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "label_shuffle_validated_recurrent_spatial_features",
        "relative_path": "09_tier1_label_shuffle_validation/05_validated_features_and_themes/label_shuffle_validated_recurrent_spatial_features.tsv",
        "category": "label_shuffle_validation",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "label_shuffle_validated_recurrent_biology_themes",
        "relative_path": "09_tier1_label_shuffle_validation/05_validated_features_and_themes/label_shuffle_validated_recurrent_biology_themes.tsv",
        "category": "label_shuffle_validation",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 10 integrated interpretation package.
    {
        "source_id": "integrated_master_summary_report",
        "relative_path": "10_integrated_interpretation_package/01_master_summary/master_summary_report.txt",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_model_comparison_table",
        "relative_path": "10_integrated_interpretation_package/02_model_comparison/model_comparison_table.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_validated_treatment_table",
        "relative_path": "10_integrated_interpretation_package/03_validated_treatments/validated_treatment_table.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_recurrent_spatial_feature_table",
        "relative_path": "10_integrated_interpretation_package/04_recurrent_spatial_features/recurrent_spatial_feature_table.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_recurrent_biology_theme_table",
        "relative_path": "10_integrated_interpretation_package/05_recurrent_biology_themes/recurrent_biology_theme_table.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_figure_manifest",
        "relative_path": "10_integrated_interpretation_package/06_figures_for_presentation/figure_manifest.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_figure_captions",
        "relative_path": "10_integrated_interpretation_package/06_figures_for_presentation/figure_captions.txt",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_methods_results_discussion",
        "relative_path": "10_integrated_interpretation_package/07_methods_results_discussion/methods_results_discussion_narrative.txt",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_pipeline_recommendations",
        "relative_path": "10_integrated_interpretation_package/08_pipeline_integration_recommendations/pipeline_integration_recommendations.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_provenance_table",
        "relative_path": "10_integrated_interpretation_package/09_provenance_and_manifest/provenance_table.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "integrated_generated_output_manifest",
        "relative_path": "10_integrated_interpretation_package/09_provenance_and_manifest/generated_output_manifest.tsv",
        "category": "integrated_interpretation_package",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 11 publication tables.
    {
        "source_id": "publication_excel_workbook",
        "relative_path": "11_publication_tables/01_publication_excel/v2_integrated_publication_tables.xlsx",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "publication_zip_package",
        "relative_path": "11_publication_tables/05_package_zip/v2_publication_tables_and_supporting_files.zip",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "publication_output_manifest",
        "relative_path": "11_publication_tables/publication_output_manifest.tsv",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pub_validated_treatments",
        "relative_path": "11_publication_tables/02_publication_ready_tsv/Pub_Validated_Treatments.tsv",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pub_recurrent_features",
        "relative_path": "11_publication_tables/02_publication_ready_tsv/Pub_Recurrent_Features.tsv",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pub_biology_themes",
        "relative_path": "11_publication_tables/02_publication_ready_tsv/Pub_Biology_Themes.tsv",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "pub_model_comparison",
        "relative_path": "11_publication_tables/02_publication_ready_tsv/Pub_Model_Comparison.tsv",
        "category": "publication_tables",
        "required": True,
        "copy_policy": "copy",
    },

    # Step 12 final QC.
    {
        "source_id": "v2_step12_qc_summary_json",
        "relative_path": "12_v2_output_qc/v2_step12_output_qc_summary.json",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_step12_qc_report",
        "relative_path": "12_v2_output_qc/08_reports/v2_step12_output_qc_report.txt",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_file_contract_qc",
        "relative_path": "12_v2_output_qc/01_file_contract_qc/file_contract_report.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_run_file_manifest",
        "relative_path": "12_v2_output_qc/01_file_contract_qc/run_file_manifest.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_branch_governance_qc",
        "relative_path": "12_v2_output_qc/02_branch_governance_qc/branch_governance_qc.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_qc_by_sample",
        "relative_path": "12_v2_output_qc/03_pair_level_and_prediction_qc/qc_by_sample.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_qc_by_treatment",
        "relative_path": "12_v2_output_qc/03_pair_level_and_prediction_qc/qc_by_treatment.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_qc_value_ranges",
        "relative_path": "12_v2_output_qc/03_pair_level_and_prediction_qc/qc_value_ranges.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_qc_model_metrics",
        "relative_path": "12_v2_output_qc/04_model_metric_qc/qc_model_metrics.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_sample_treatment_contract_qc",
        "relative_path": "12_v2_output_qc/05_sample_treatment_contract_qc/qc_sample_treatment_contract.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
    {
        "source_id": "v2_publication_package_qc",
        "relative_path": "12_v2_output_qc/06_publication_package_qc/qc_publication_package.tsv",
        "category": "v2_final_qc",
        "required": True,
        "copy_policy": "copy",
    },
]


STEP_DIRS = [
    "01_prepared_inputs",
    "02_feature_and_treatment_dictionary",
    "03_signed_spatial_effects",
    "04_treatment_interpretation_cards",
    "05_sample_level_interpretations",
    "06_mechanism_atlas",
    "07_final_outputs",
    "08_qc_and_final_package",
    "pipeline_run_logs",
]


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def now_stamp() -> str:
    """Return a filesystem-safe timestamp string.
    Used for run names, patch logs, and reproducible report folders."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_text_report(path: Path, body: str) -> None:
    """Write a text report with FILEPATH on the first line.
    This convention is required for all generated text reports."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    """Write structured metadata as formatted JSON.
    Creates parent folders and preserves readable provenance output."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_tsv(path: Path, rows: List[dict]) -> None:
    """Write a pandas DataFrame as a tab-separated table.
    Creates parent folders before writing the output artifact."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataframe_to_tsv(path: Path, df: pd.DataFrame) -> None:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def open_folder(path: Path) -> None:
    """Open an output folder in the local operating system.
    Failures are intentionally nonfatal so batch runs can continue."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def sha256_file(path: Path, max_bytes: int) -> str:
    """Compute a SHA256 digest for a package file.
    Large files can be skipped to keep QC runtime bounded."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    size = path.stat().st_size
    if size > max_bytes:
        return "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_sep(path: Path) -> str:
    """Infer the delimiter for a CSV/TSV-style table path.
    TSV and TAB files use tab; other table files default to comma."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return "\t" if path.suffix.lower() in [".tsv", ".tab"] else ","


def read_header(path: Path) -> List[str]:
    """Read only the header row from a table.
    Avoids loading large V2 source tables when only the schema is needed."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    sep = choose_sep(path)
    df = pd.read_csv(path, sep=sep, nrows=0)
    return list(df.columns)


def read_table(path: Path, nrows: Optional[int] = None, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a CSV/TSV table with the project delimiter convention.
    Supports optional row and column restrictions for large V2 files."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    sep = choose_sep(path)
    return pd.read_csv(path, sep=sep, nrows=nrows, usecols=usecols, low_memory=False)


def choose_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    lower_to_original = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def safe_slug(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug.
    Used for stable filenames and compact artifact names."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def build_output_manifest(root: Path) -> List[dict]:
    """Inventory files under an output root.
    Captures relative paths, absolute paths, file sizes, and suffixes."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            rows.append({
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "absolute_path": str(path),
                "size_bytes": size,
                "suffix": path.suffix.lower(),
            })
    return rows


def source_path_by_id(source_rows: List[dict], source_id: str) -> Path:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    for row in source_rows:
        if row["source_id"] == source_id:
            return Path(row["absolute_path"])
    raise KeyError(source_id)


def qc_status_from_table(path: Path) -> Tuple[int, int]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        return 0, 0
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return 0, 0
    if "status" not in df.columns:
        return 0, 0
    statuses = df["status"].astype(str).str.lower()
    return int((statuses == "fail").sum()), int((statuses == "warn").sum())


def text_has_pass_status(path: Path) -> Optional[bool]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"status\s*:\s*pass", text, flags=re.IGNORECASE):
        return True
    if re.search(r"status\s*:\s*fail", text, flags=re.IGNORECASE):
        return False
    return None


def json_status(path: Path) -> Optional[str]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("status")
    return None if value is None else str(value).lower()


def add_check(checks: List[dict], errors: List[str], warnings: List[str], check_id: str, status: str, observed: object, expected: object, detail: str) -> None:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    checks.append({
        "check_id": check_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })
    if status == "fail":
        errors.append(f"{check_id}: {detail}")
    elif status == "warn":
        warnings.append(f"{check_id}: {detail}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    Defaults preserve local project paths while allowing explicit overrides."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-name", default=f"prediction_interpretation_model_full_{now_stamp()}")
    parser.add_argument("--large-file-mb", type=float, default=75.0)
    parser.add_argument("--hash-file-mb", type=float, default=25.0)
    parser.add_argument("--copy-large-files", action="store_true")
    parser.add_argument("--expected-pair-rows", type=int, default=34881)
    parser.add_argument("--expected-samples", type=int, default=102)
    parser.add_argument("--expected-treatments", type=int, default=374)
    parser.add_argument("--expected-strict-biology-features", type=int, default=139)
    parser.add_argument("--expected-validated-treatments", type=int, default=27)
    parser.add_argument("--expected-recurrent-biology-themes", type=int, default=11)
    parser.add_argument("--open-output", action="store_true")

    return parser.parse_args()


# =============================================================================
# PIM_DOCS_SECTION: main entry point
# =============================================================================
# The main function wires inputs, output folders, QC checks, reports, and terminal summaries.

def main() -> int:
    """Run the script's command-line workflow.
    Writes outputs, QC checks, summaries, and terminal status messages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    args = parse_args()

    project_root = Path(args.project_root)
    if args.model_root:
        model_root = Path(args.model_root)
    else:
        model_root = Path(__file__).resolve().parents[1]

    v2_run_root = Path(args.v2_run_root)

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = model_root / "outputs" / args.run_name

    output_root.mkdir(parents=True, exist_ok=True)

    for dirname in STEP_DIRS:
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    step_root = output_root / "01_prepared_inputs"
    manifest_root = step_root / "01_source_manifests"
    copied_root = step_root / "02_copied_v2_tables"
    schema_root = step_root / "03_source_table_schemas"
    pointer_root = step_root / "04_pair_level_pointer"
    reports_root = step_root / "05_reports"

    for path in [manifest_root, copied_root, schema_root, pointer_root, reports_root]:
        path.mkdir(parents=True, exist_ok=True)

    copied_tables_root = copied_root / "tables"
    copied_reports_root = copied_root / "reports"
    copied_metadata_root = copied_root / "metadata"
    copied_packages_root = copied_root / "packages"

    for path in [copied_tables_root, copied_reports_root, copied_metadata_root, copied_packages_root]:
        path.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now()
    errors: List[str] = []
    warnings: List[str] = []
    checks: List[dict] = []
    source_manifest_rows: List[dict] = []
    prepared_index_rows: List[dict] = []

    large_threshold_bytes = int(args.large_file_mb * 1024 * 1024)
    hash_threshold_bytes = int(args.hash_file_mb * 1024 * 1024)

    add_check(
        checks,
        errors,
        warnings,
        "v2_run_root_exists",
        "pass" if v2_run_root.exists() else "fail",
        v2_run_root.exists(),
        True,
        f"V2 source root: {v2_run_root}",
    )

    deprecated_markers = [
        "old_deprecated_pipeline",
        "\\prediction modeling\\code_prediction modeling\\prediction interpretation model",
        "\\prediction_modeling_pipeline\\spatial_prediction_model\\outputs\\final_integrated",
    ]

    lower_root = str(v2_run_root).lower()
    bad_deprecated = any(marker.lower() in lower_root for marker in deprecated_markers)
    add_check(
        checks,
        errors,
        warnings,
        "deprecated_source_root_not_used",
        "fail" if bad_deprecated else "pass",
        v2_run_root,
        "spatial_prediction_model_V2 full-run root",
        "Step 01 must use the V2 full-run root, not old deprecated or V1 source roots.",
    )

    if not v2_run_root.exists():
        raise SystemExit(1)

    for item in SOURCE_FILES:
        source_id = item["source_id"]
        relative_path = item["relative_path"]
        category = item["category"]
        required = bool(item["required"])
        copy_policy = str(item["copy_policy"])

        source_path = v2_run_root / relative_path
        exists = source_path.exists()
        size_bytes = source_path.stat().st_size if exists else None
        suffix = source_path.suffix.lower() if exists else Path(relative_path).suffix.lower()

        copied_path = ""
        copy_status = "not_copied"
        sha256 = ""

        if exists:
            try:
                sha256 = sha256_file(source_path, hash_threshold_bytes)
            except Exception as exc:
                sha256 = f"hash_failed: {exc}"

            should_copy = copy_policy == "copy"
            if copy_policy == "pointer":
                should_copy = bool(args.copy_large_files)

            if should_copy and size_bytes is not None and size_bytes > large_threshold_bytes and not args.copy_large_files:
                should_copy = False
                copy_status = "pointer_large_file"

            if should_copy:
                if suffix in [".tsv", ".tab", ".csv"]:
                    dest_dir = copied_tables_root
                elif suffix in [".txt", ".md"]:
                    dest_dir = copied_reports_root
                elif suffix in [".xlsx", ".zip"]:
                    dest_dir = copied_packages_root
                else:
                    dest_dir = copied_metadata_root

                dest = dest_dir / f"{safe_slug(source_id)}{suffix}"
                shutil.copy2(source_path, dest)
                copied_path = str(dest)
                copy_status = "copied"
            elif copy_status == "not_copied":
                copy_status = "pointer_only"

            if suffix in [".tsv", ".tab", ".csv"]:
                try:
                    cols = read_header(source_path)
                    schema_rows = [
                        {
                            "source_id": source_id,
                            "column_index": i,
                            "column_name": col,
                        }
                        for i, col in enumerate(cols)
                    ]
                    write_tsv(schema_root / f"{safe_slug(source_id)}_columns.tsv", schema_rows)
                except Exception as exc:
                    warnings.append(f"schema read failed for {source_id}: {exc}")

        if required and not exists:
            errors.append(f"Required source file missing: {source_path}")

        source_manifest_rows.append({
            "source_id": source_id,
            "category": category,
            "relative_path": relative_path,
            "absolute_path": str(source_path),
            "required": required,
            "exists": exists,
            "size_bytes": size_bytes,
            "suffix": suffix,
            "copy_policy": copy_policy,
            "copy_status": copy_status,
            "copied_path": copied_path,
            "sha256": sha256,
        })

        prepared_index_rows.append({
            "source_id": source_id,
            "category": category,
            "preferred_read_path": copied_path if copied_path else str(source_path),
            "copied_path": copied_path,
            "source_path": str(source_path),
            "is_pointer_only": copied_path == "",
            "required": required,
        })

    missing_required = [row for row in source_manifest_rows if row["required"] and not row["exists"]]
    add_check(
        checks,
        errors,
        warnings,
        "required_v2_files_exist",
        "pass" if not missing_required else "fail",
        len(missing_required),
        0,
        "All required V2 handoff files should exist.",
    )

    source_paths_text = "\n".join(str(row["absolute_path"]).lower() for row in source_manifest_rows)
    bad_source_path = any(marker.lower() in source_paths_text for marker in deprecated_markers)
    add_check(
        checks,
        errors,
        warnings,
        "deprecated_source_files_not_used",
        "fail" if bad_source_path else "pass",
        bad_source_path,
        False,
        "Prepared source files should all come from the V2 full-run root.",
    )

    dataframe_to_tsv(manifest_root / "v2_source_file_manifest.tsv", pd.DataFrame(source_manifest_rows))
    dataframe_to_tsv(manifest_root / "prepared_interpretation_source_index.tsv", pd.DataFrame(prepared_index_rows))

    # Key scientific/contract counts.
    key_counts: List[dict] = []

    try:
        pair_path = source_path_by_id(source_manifest_rows, "v2_pair_level_residual_dataset")
        pair_columns = read_header(pair_path)
        sample_col = choose_column(pair_columns, ["sample_id", "slide_id", "sample"])
        treatment_col = choose_column(pair_columns, ["drug_key", "treatment_key", "drug", "treatment"])
        required_target_cols = [
            col for col in ["fused_prob_responder", "fused_residual_vs_prior", "treatment_prior", "prior_prob_responder"]
            if col in pair_columns
        ]

        if sample_col is None or treatment_col is None:
            raise ValueError("Could not identify sample and treatment columns in pair-level dataset.")

        pair_usecols = [sample_col, treatment_col] + required_target_cols
        pair_key = read_table(pair_path, usecols=pair_usecols)

        pair_rows = int(pair_key.shape[0])
        sample_count = int(pair_key[sample_col].astype(str).nunique())
        treatment_count = int(pair_key[treatment_col].astype(str).nunique())
        duplicate_pairs = int(pair_key.duplicated([sample_col, treatment_col]).sum())

        key_counts.extend([
            {
                "metric": "pair_level_rows",
                "observed": pair_rows,
                "expected": args.expected_pair_rows,
                "status": "pass" if pair_rows == args.expected_pair_rows else "fail",
                "source_id": "v2_pair_level_residual_dataset",
            },
            {
                "metric": "sample_count",
                "observed": sample_count,
                "expected": args.expected_samples,
                "status": "pass" if sample_count == args.expected_samples else "fail",
                "source_id": "v2_pair_level_residual_dataset",
            },
            {
                "metric": "treatment_count",
                "observed": treatment_count,
                "expected": args.expected_treatments,
                "status": "pass" if treatment_count == args.expected_treatments else "fail",
                "source_id": "v2_pair_level_residual_dataset",
            },
            {
                "metric": "duplicate_sample_treatment_pairs",
                "observed": duplicate_pairs,
                "expected": 0,
                "status": "pass" if duplicate_pairs == 0 else "fail",
                "source_id": "v2_pair_level_residual_dataset",
            },
        ])

        pair_preview = pair_key.head(25).copy()
        dataframe_to_tsv(pointer_root / "pair_level_dataset_selected_column_preview.tsv", pair_preview)

        pointer_rows = [{
            "source_id": "v2_pair_level_residual_dataset",
            "source_path": str(pair_path),
            "copy_policy": "pointer_only_by_default",
            "reason": "The pair-level dataset is large and is treated as frozen V2 source truth. Step 01 records the path, hash policy, schema, selected-column preview, and key counts.",
            "sample_col": sample_col,
            "treatment_col": treatment_col,
            "selected_columns_read": ",".join(pair_usecols),
            "pair_level_rows": pair_rows,
            "sample_count": sample_count,
            "treatment_count": treatment_count,
            "duplicate_sample_treatment_pairs": duplicate_pairs,
        }]
        write_tsv(pointer_root / "pair_level_dataset_pointer.tsv", pointer_rows)

    except Exception as exc:
        add_check(
            checks,
            errors,
            warnings,
            "pair_level_dataset_contract",
            "fail",
            "exception",
            "readable pair-level dataset with expected columns",
            str(exc),
        )

    try:
        strict_path = source_path_by_id(source_manifest_rows, "v2_strict_biology_feature_registry")
        strict_df = read_table(strict_path)
        strict_rows = int(strict_df.shape[0])
        feature_col = choose_column(strict_df.columns, ["feature", "feature_name", "spatial_feature", "model_feature", "variable"])
        if feature_col is None:
            feature_col = strict_df.columns[0]
        feature_values = strict_df[feature_col].astype(str).str.lower()
        treatment_identity_hits = int(
            feature_values.str.contains(r"(^drug__|treatment|drug_key|prior_prob|treatment_prior)", regex=True).sum()
        )

        key_counts.extend([
            {
                "metric": "strict_biology_feature_rows",
                "observed": strict_rows,
                "expected": args.expected_strict_biology_features,
                "status": "pass" if strict_rows == args.expected_strict_biology_features else "fail",
                "source_id": "v2_strict_biology_feature_registry",
            },
            {
                "metric": "strict_biology_treatment_identity_hits",
                "observed": treatment_identity_hits,
                "expected": 0,
                "status": "pass" if treatment_identity_hits == 0 else "fail",
                "source_id": "v2_strict_biology_feature_registry",
            },
        ])

    except Exception as exc:
        add_check(
            checks,
            errors,
            warnings,
            "strict_biology_registry_contract",
            "fail",
            "exception",
            "readable V2 strict biology registry",
            str(exc),
        )

    try:
        validated_path = source_path_by_id(source_manifest_rows, "tier1_label_shuffle_validated_treatments")
        validated_df = read_table(validated_path)
        validated_rows = int(validated_df.shape[0])
        key_counts.append({
            "metric": "label_shuffle_validated_treatments",
            "observed": validated_rows,
            "expected": args.expected_validated_treatments,
            "status": "pass" if validated_rows == args.expected_validated_treatments else "fail",
            "source_id": "tier1_label_shuffle_validated_treatments",
        })
    except Exception as exc:
        add_check(
            checks,
            errors,
            warnings,
            "validated_treatment_contract",
            "fail",
            "exception",
            "readable label-shuffle validated treatment table",
            str(exc),
        )

    try:
        theme_path = source_path_by_id(source_manifest_rows, "label_shuffle_validated_recurrent_biology_themes")
        theme_df = read_table(theme_path)
        theme_rows = int(theme_df.shape[0])
        key_counts.append({
            "metric": "label_shuffle_validated_recurrent_biology_themes",
            "observed": theme_rows,
            "expected": args.expected_recurrent_biology_themes,
            "status": "pass" if theme_rows == args.expected_recurrent_biology_themes else "fail",
            "source_id": "label_shuffle_validated_recurrent_biology_themes",
        })
    except Exception as exc:
        add_check(
            checks,
            errors,
            warnings,
            "recurrent_biology_theme_contract",
            "fail",
            "exception",
            "readable validated recurrent biology theme table",
            str(exc),
        )

    for count_row in key_counts:
        add_check(
            checks,
            errors,
            warnings,
            count_row["metric"],
            count_row["status"],
            count_row["observed"],
            count_row["expected"],
            f"Source: {count_row['source_id']}",
        )

    dataframe_to_tsv(manifest_root / "v2_key_count_summary.tsv", pd.DataFrame(key_counts))

    # V2 Step 12 QC status checks.
    try:
        qc_report = source_path_by_id(source_manifest_rows, "v2_step12_qc_report")
        qc_report_status = text_has_pass_status(qc_report)
        add_check(
            checks,
            errors,
            warnings,
            "v2_step12_qc_report_status",
            "pass" if qc_report_status is True else ("fail" if qc_report_status is False else "warn"),
            qc_report_status,
            True,
            "Step 12 QC report should contain status: pass.",
        )
    except Exception as exc:
        add_check(checks, errors, warnings, "v2_step12_qc_report_status", "fail", "exception", "pass", str(exc))

    try:
        qc_summary = source_path_by_id(source_manifest_rows, "v2_step12_qc_summary_json")
        status_value = json_status(qc_summary)
        add_check(
            checks,
            errors,
            warnings,
            "v2_step12_qc_summary_status",
            "pass" if status_value in ["pass", "passed"] else ("warn" if status_value is None else "fail"),
            status_value,
            "pass",
            "Step 12 QC summary JSON should report pass if status is present.",
        )
    except Exception as exc:
        add_check(checks, errors, warnings, "v2_step12_qc_summary_status", "fail", "exception", "pass", str(exc))

    for source_id, check_id in [
        ("v2_file_contract_qc", "v2_file_contract_qc_no_fails"),
        ("v2_branch_governance_qc", "v2_branch_governance_qc_no_fails"),
        ("v2_sample_treatment_contract_qc", "v2_sample_treatment_contract_qc_no_fails"),
        ("v2_publication_package_qc", "v2_publication_package_qc_no_fails"),
    ]:
        try:
            qc_path = source_path_by_id(source_manifest_rows, source_id)
            fail_count, warn_count = qc_status_from_table(qc_path)
            add_check(
                checks,
                errors,
                warnings,
                check_id,
                "pass" if fail_count == 0 else "fail",
                f"fail={fail_count}; warn={warn_count}",
                "fail=0",
                f"QC table: {qc_path}",
            )
        except Exception as exc:
            add_check(checks, errors, warnings, check_id, "fail", "exception", "fail=0", str(exc))

    checks_df = pd.DataFrame(checks)
    dataframe_to_tsv(manifest_root / "v2_source_qc_checks.tsv", checks_df)

    # Run config/provenance.
    resolved_config = {
        "project_root": str(project_root),
        "model_root": str(model_root),
        "v2_run_root": str(v2_run_root),
        "output_root": str(output_root),
        "step_root": str(step_root),
        "large_file_mb": args.large_file_mb,
        "hash_file_mb": args.hash_file_mb,
        "copy_large_files": args.copy_large_files,
        "expected_counts": {
            "pair_rows": args.expected_pair_rows,
            "samples": args.expected_samples,
            "treatments": args.expected_treatments,
            "strict_biology_features": args.expected_strict_biology_features,
            "validated_treatments": args.expected_validated_treatments,
            "recurrent_biology_themes": args.expected_recurrent_biology_themes,
        },
        "started": started.isoformat(timespec="seconds"),
    }
    write_json(step_root / "resolved_config.json", resolved_config)

    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": os.getcwd(),
        "pandas_version": pd.__version__,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(step_root / "environment_and_provenance.json", environment)

    provenance_body = "\n".join([
        "PREDICTION INTERPRETATION MODEL STEP 01 ENVIRONMENT AND PROVENANCE",
        "",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version}",
        f"platform: {sys.platform}",
        f"working_directory: {os.getcwd()}",
        f"pandas_version: {pd.__version__}",
        f"timestamp: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Policy",
        "This step records and prepares V2 source outputs.",
        "It does not rerun spatial_prediction_model_V2.",
        "It does not train or select models.",
        "It does not use deprecated outputs as source truth.",
    ])
    write_text_report(step_root / "environment_and_provenance.txt", provenance_body)

    status = "pass" if not errors else "fail"
    finished = dt.datetime.now()

    # Step report.
    source_count = len(source_manifest_rows)
    copied_count = sum(1 for row in source_manifest_rows if row["copy_status"] == "copied")
    pointer_count = sum(1 for row in source_manifest_rows if row["copy_status"] != "copied")
    fail_checks = sum(1 for row in checks if row["status"] == "fail")
    warn_checks = sum(1 for row in checks if row["status"] == "warn")

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 01 REPORT",
        "",
        f"status: {status}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Source roots",
        f"project_root: {project_root}",
        f"model_root: {model_root}",
        f"v2_run_root: {v2_run_root}",
        f"output_root: {output_root}",
        "",
        "Source preparation",
        f"source_file_rows: {source_count}",
        f"copied_source_files: {copied_count}",
        f"pointer_only_source_files: {pointer_count}",
        "",
        "Key V2 contract counts",
    ]

    for row in key_counts:
        report_lines.append(
            f"{row['metric']}: observed={row['observed']} | expected={row['expected']} | status={row['status']}"
        )

    report_lines.extend([
        "",
        "QC checks",
        f"fail_checks: {fail_checks}",
        f"warn_checks: {warn_checks}",
        "",
        "Generated handoff files",
        str(manifest_root / "v2_source_file_manifest.tsv"),
        str(manifest_root / "prepared_interpretation_source_index.tsv"),
        str(manifest_root / "v2_key_count_summary.tsv"),
        str(manifest_root / "v2_source_qc_checks.tsv"),
        str(pointer_root / "pair_level_dataset_pointer.tsv"),
        str(pointer_root / "pair_level_dataset_selected_column_preview.tsv"),
        "",
        "Policy",
        "This step did not rerun V2.",
        "This step did not perform open model selection.",
        "This step did not use deprecated outputs as source truth.",
        "",
        "Errors",
    ])

    report_lines.extend(errors if errors else ["none"])
    report_lines.append("")
    report_lines.append("Warnings")
    report_lines.extend(warnings if warnings else ["none"])

    write_text_report(reports_root / "step01_prepare_interpretation_inputs_report.txt", "\n".join(report_lines))

    readme_lines = [
        "PREDICTION INTERPRETATION MODEL PREPARED INPUTS README",
        "",
        f"status: {status}",
        f"v2_run_root: {v2_run_root}",
        f"prepared_input_root: {step_root}",
        "",
        "What this folder contains",
        "01_source_manifests: source file manifest, prepared source index, key counts, and QC checks.",
        "02_copied_v2_tables: copied V2 tables/reports/packages small enough to safely duplicate.",
        "03_source_table_schemas: column schemas for CSV/TSV sources.",
        "04_pair_level_pointer: pointer and selected-column preview for the large pair-level dataset.",
        "05_reports: Step 01 report.",
        "",
        "Important source-truth rule",
        "The V2 full run remains the authoritative source. Large files are not duplicated by default.",
        "Deprecated prediction interpretation outputs are not used as final source truth.",
        "",
        "Next intended steps",
        "02_build_feature_and_treatment_dictionary.py",
        "03_compute_signed_spatial_effects.py",
    ]
    write_text_report(step_root / "README_prepared_inputs.txt", "\n".join(readme_lines))

    step_summary = {
        "status": status,
        "project_root": str(project_root),
        "model_root": str(model_root),
        "v2_run_root": str(v2_run_root),
        "output_root": str(output_root),
        "step_root": str(step_root),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "source_file_rows": source_count,
        "copied_source_files": copied_count,
        "pointer_only_source_files": pointer_count,
        "fail_checks": fail_checks,
        "warn_checks": warn_checks,
        "errors": errors,
        "warnings": warnings,
        "key_counts": key_counts,
        "source_manifest": str(manifest_root / "v2_source_file_manifest.tsv"),
        "prepared_source_index": str(manifest_root / "prepared_interpretation_source_index.tsv"),
        "key_count_summary": str(manifest_root / "v2_key_count_summary.tsv"),
        "qc_checks": str(manifest_root / "v2_source_qc_checks.tsv"),
        "step_report": str(reports_root / "step01_prepare_interpretation_inputs_report.txt"),
    }
    write_json(output_root / "prediction_interpretation_model_step01_summary.json", step_summary)

    output_manifest_rows = build_output_manifest(output_root)
    write_tsv(output_root / "prediction_interpretation_model_output_manifest.tsv", output_manifest_rows)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 01 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"v2_run_root: {v2_run_root}")
    print(f"output_root: {output_root}")
    print(f"step_root: {step_root}")
    print("")
    print("Key counts")
    for row in key_counts:
        print(f"  {row['metric']}: observed={row['observed']} expected={row['expected']} status={row['status']}")
    print("")
    print(f"source_file_rows: {source_count}")
    print(f"copied_source_files: {copied_count}")
    print(f"pointer_only_source_files: {pointer_count}")
    print(f"fail_checks: {fail_checks}")
    print(f"warn_checks: {warn_checks}")
    print("")
    print("Main report:")
    print(reports_root / "step01_prepare_interpretation_inputs_report.txt")
    print("Prepared source index:")
    print(manifest_root / "prepared_interpretation_source_index.tsv")

    if errors:
        print("")
        print("ERRORS")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("")
        print("WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("UNHANDLED ERROR IN STEP 01")
        print("".join(traceback.format_exception(exc)))
        raise SystemExit(1)

