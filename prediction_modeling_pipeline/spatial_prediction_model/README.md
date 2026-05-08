# spatial_prediction_model

This folder contains the earlier spatial prediction modeling workflow that sits downstream of `teacher_builder`. It is retained in the repository for review and provenance because it contains the original model-selection, residual-modeling, treatment-specific modeling, and interpretation-package development code that informed the governed `spatial_prediction_model_V2` workflow.

This module is **not the current production workflow**. For the current governed spatial prediction implementation, use:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/
```

Do not use this folder in place of V2 for new full analyses unless you are intentionally reviewing, reproducing, or comparing the earlier workflow.

## Purpose

The original `spatial_prediction_model` workflow was developed to connect teacher-builder outputs to spatial response modeling. It includes code for validating prediction inputs, building spatial modeling datasets, training global and treatment-specific response models, explaining spatial-response associations, predicting sample-treatment pairs, residual modeling, label-shuffle validation, and building interpretation-support outputs.

Although V2 is the current workflow, this folder remains useful because it documents and preserves the modeling ideas that led to V2, including:

- early spatial response model construction;
- model-selection and model-comparison logic;
- residual-response modeling concepts;
- treatment-specific modeling experiments;
- sample-treatment prediction table generation;
- QC and validation logic;
- interpretation-package and publication-support table prototypes.

## Relationship to spatial_prediction_model_V2

`spatial_prediction_model_V2` is the current governed implementation and should be used for the active project workflow.

This earlier folder is retained because it provides provenance for the development path into V2. In particular, it contains earlier model-choosing and residual-modeling code that helped define the V2 design. Keeping it in GitHub allows a teacher or reviewer to inspect how the final governed implementation evolved from the earlier modeling workflow.

Recommended use:

```text
Use spatial_prediction_model_V2/ for current analysis.
Use spatial_prediction_model/ for historical review, comparison, and provenance.
```

## Upstream handoff

The original workflow expected compact teacher-builder handoff files such as:

```text
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/model_input_numeric.csv
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/visium_fused_teacher_table.tsv
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/feature_manifest.csv
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/prediction_ready_training_table.tsv
```

These generated files are not included in GitHub. To rerun or inspect the workflow locally, provide equivalent teacher-builder outputs and update the configuration paths.

## Configuration

The main configuration file is:

```text
configs/spatial_prediction_model.yaml
```

Configuration files may contain local path assumptions from earlier development. Before running on a new machine, review and update all input paths, output paths, run names, and sample-selection settings.

For GitHub use, avoid committing machine-specific or timestamped local run configs. Keep clean templates or clearly labeled example configs when possible.

## Original run modes

The early scaffold supported a small test mode and a later full-cohort mode.

Example test-mode settings:

```yaml
run_name: "output_run_10"
test_mode: true
test_n_samples: 10
limit_training_to_test_samples: true
prediction_sample_mode: "test_labeled_samples"
```

Example full-run settings from the original workflow concept:

```yaml
run_name: "output_run_102"
run_scope: "full_102"
test_mode: false
limit_training_to_test_samples: false
prediction_sample_mode: "all_spatial_samples"
run_per_treatment_models: true
```

These examples describe the earlier workflow design. For current production analyses, use the V2 configuration and runner instead.

## Main scripts

The original core workflow scripts include:

```text
scripts/01_validate_prediction_inputs.py
scripts/02_build_spatial_modeling_dataset.py
scripts/03_train_global_spatial_response_model.py
scripts/04_train_per_treatment_models.py
scripts/05_explain_spatial_response_model.py
scripts/06_predict_all_sample_treatment_pairs.py
scripts/07_qc_spatial_prediction_outputs.py
```

Additional later scripts extend the workflow into residual modeling, treatment-specific residual curation, label-shuffle validation, interpretation packaging, and publication-support table generation:

```text
scripts/08_1_train_spatial_only_broad_residual_model.py
scripts/08_2_train_spatial_only_broad_residual_model.py
scripts/08_3_validate_spatial_only_broad_residual_model.py
scripts/09_1_train_filtered_per_treatment_residual_models.py
scripts/10_1_curate_per_treatment_residual_models.py
scripts/11_1_label_shuffle_validate_tier1_per_treatment_residual_models.py
scripts/12_build_final_integrated_interpretation_package.py
scripts/12_2_build_final_integrated_interpretation_package.py
scripts/12_3_make_publication_tables_and_supporting_files.py
scripts/12_4_repair_publication_tables_and_supporting_files.py
```

## Running the earlier workflow

A local runner is provided:

```powershell
cd "YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_prediction_model"

.
un_spatial_prediction_model.ps1
```

or, if using the Python runner directly:

```powershell
python run_spatial_prediction_model.py --config configs/spatial_prediction_model.yaml
```

Before running, confirm that the configured teacher-builder handoff files exist and that output paths point to a local writable folder. Generated outputs are local-only and are not tracked in GitHub.

## Output policy

Generated outputs from this earlier workflow should remain local. Do not commit:

```text
outputs/
logs/
model artifacts
large CSV/TSV result tables
SHAP or explanation artifacts
publication figure bundles
backup files
run-specific timestamped configs
```

Commit only source scripts, durable documentation, small reusable configs, and lightweight examples.

## Notes for reviewers

This folder is included so the earlier modeling and model-selection work remains inspectable. It should be read as a development and provenance module rather than as the current final prediction system.

For the active workflow, begin with:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/README.md
```

Use this folder only if you want to inspect the earlier modeling design that informed V2.

