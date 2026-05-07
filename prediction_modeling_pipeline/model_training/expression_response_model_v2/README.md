# expression_response_model_v2

`expression_response_model_v2` trains, audits, and optionally applies deployable expression-response teacher models for the Adv_Omics_Fenyo spatial treatment-response project.

The workflow uses GDC expression profiles, named-treatment labels, and binary response labels to train treatment-specific calibrated models. Approved models can then be used as upstream teacher candidates for `teacher_builder`. This module is not the final spatial prediction model.

## Quick start

From a new PowerShell session on the workstation where the project lives:

    cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\model_training\expression_response_model_v2"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4

This runs the main deployable expression-response workflow:

    Step 00  treatment ontology
    Step 01  input validation
    Step 02  canonical training table
    Step 03  deployable model training
    Step 04  deployable model audit

Optional Visium expression-teacher scoring is Step 05:

    .\run_expression_response_model_v2.ps1 -StartAt 5 -StopAt 5

Run Step 05 only after deployable models have been trained and audited and after the configured Visium pseudobulk expression table exists.

To run one step at a time:

    .\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2

The runner defaults to:

    D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe

If that virtual environment is unavailable, the runner falls back to `python` on the active `PATH`.

## Reproducibility and configuration rules

For normal use, change paths, thresholds, model settings, teacher-scoring options, and output locations in:

    configs/expression_response_model_v2.yaml

Do not edit files in `scripts/` unless you are intentionally changing pipeline logic. The Python scripts are the active source implementation; the YAML config is the normal user-editable control surface.

Important rules:

- edit YAML values when changing input paths, output locations, support thresholds, model parameters, calibration settings, or teacher-scoring paths;
- do not edit numbered scripts for routine reruns;
- do not overwrite canonical outputs unless the rerun is intentional;
- run smoke tests or scratch reruns before changing active outputs;
- keep very large training tables and trained model artifacts local or in external archival storage rather than committing them directly to GitHub.

## Scientific role

The larger project asks whether spatial tumor architecture can help explain treatment-response biology. This module contributes the expression-teacher stream:

    GDC expression + treatment labels + response labels
            ↓
    canonical case-treatment training table
            ↓
    calibrated treatment-specific expression models
            ↓
    approved expression teacher scores or deployable model artifacts
            ↓
    teacher_builder fusion with histology teachers and treatment priors
            ↓
    spatial_prediction_model

Because downstream spatial models learn from teacher labels, this module is intentionally conservative. A trained expression model is not automatically trusted as a teacher. It must satisfy support, grouped cross-validation, calibration, probability-extremeness, and reliability checks before it is promoted to `model_index_approved.tsv`.

## What changed from expression_response_model v1

This folder supersedes the older `expression_response_model` workflow. The older workflow is retained only as provenance. Version 2 is organized as a deployable, auditable, teacher-ready expression-response model workflow.

Main improvements in v2:

- separates treatment ontology construction, input validation, canonical training table construction, model training, model audit, and optional Visium teacher scoring into numbered steps;
- creates a canonical treatment-key contract before model fitting;
- writes a canonical expression-response training table and metadata sidecars;
- trains treatment-specific deployable models rather than only exploratory fold reports;
- serializes trained model artifacts as `.joblib` files;
- writes `model_index.tsv` and `model_index_approved.tsv` as explicit downstream contracts;
- applies calibration and teacher reliability scoring;
- records skipped drugs and audit checks rather than silently dropping under-supported treatments;
- optionally scores Visium pseudobulk expression for downstream teacher use;
- separates model fitting from `teacher_builder` so downstream teacher construction does not retrain expression models internally.

## Repository layout

The active workflow is organized around these files:

| Type | File | Purpose |
| --- | --- | --- |
| Config | `configs/expression_response_model_v2.yaml` | Main user-editable configuration file. |
| Runner | `run_expression_response_model_v2.ps1` | Executes the numbered workflow steps. |
| Shared library | `scripts/expression_model_v2_lib.py` | Shared config, treatment, response, calibration, reliability, I/O, and probability-shrinkage helpers. |
| Step 00 | `scripts/00_build_treatment_ontology.py` | Builds treatment ontology and alias tables from raw treatment labels. |
| Step 01 | `scripts/01_validate_inputs.py` | Checks required columns, gene features, response labels, and treatment support. |
| Step 02 | `scripts/02_build_canonical_training_table.py` | Builds canonical expression-response modeling tables and metadata. |
| Step 03 | `scripts/03_train_deployable_models.py` | Trains treatment-specific calibrated logistic-response models. |
| Step 04 | `scripts/04_audit_deployable_models.py` | Audits trained model indexes, calibration fields, reliability weights, and CV predictions. |
| Step 05 | `scripts/05_score_visium_expression_teacher.py` | Optionally scores Visium pseudobulk expression samples with approved deployable models. |

Development backups, temporary dry-run helpers, and local provenance are not part of the active workflow and should live outside the active source path, usually under `local_archive/` or `docs/`.

## Pipeline steps and outputs

### Step 00: treatment ontology

Builds treatment ontology and alias tables from raw treatment labels.

Primary outputs:

    outputs/deployable_CH1/treatment_ontology.tsv
    outputs/deployable_CH1/treatment_aliases.tsv

### Step 01: input validation

Checks source-table columns, binary response availability, treatment support, and expression gene columns.

Primary outputs:

    outputs/deployable_CH1/validation/input_validation_summary.txt
    outputs/deployable_CH1/validation/drug_count_summary.tsv
    outputs/deployable_CH1/validation/eligible_drugs_preview.tsv

### Step 02: canonical training table

Builds the canonical expression-response modeling table, metadata sidecar, conflict table, deduplication report, and gene-column list.

Primary outputs:

    outputs/deployable_CH1/data/training_table_canonical.tsv
    outputs/deployable_CH1/data/training_metadata_canonical.tsv
    outputs/deployable_CH1/data/conflicting_case_drug_labels.tsv
    outputs/deployable_CH1/data/deduplication_report.tsv
    outputs/deployable_CH1/data/gene_columns.txt

### Step 03: deployable model training

Trains treatment-specific calibrated logistic-response models with grouped cross-validation and reliability scoring.

Primary outputs:

    outputs/deployable_CH1/models/*.joblib
    outputs/deployable_CH1/model_index.tsv
    outputs/deployable_CH1/model_index_approved.tsv
    outputs/deployable_CH1/skipped_drugs.tsv
    outputs/deployable_CH1/cv/cv_fold_summary.tsv
    outputs/deployable_CH1/cv/cv_fold_predictions.tsv

### Step 04: model audit

Audits trained model indexes, approved teacher models, calibration fields, reliability weights, and cross-validation prediction outputs.

Primary outputs:

    outputs/deployable_CH1/audit/deployable_model_audit_summary.txt
    outputs/deployable_CH1/audit/deployable_model_audit_checks.tsv
    outputs/deployable_CH1/audit/model_index_for_audit.tsv
    outputs/deployable_CH1/audit/training_run_summary.json

### Step 05: optional Visium teacher scoring

Applies approved deployable models to the configured Visium pseudobulk expression table.

Primary outputs are written to the `teacher_scoring.output_dir` path in the YAML config and usually include:

    expression_teacher_scores.tsv
    expression_teacher_summary.tsv
    expression_teacher_scoring_summary.txt

## Model design

Each treatment-specific deployable model uses a transparent classical machine-learning structure:

    median imputation
        → variance filtering
        → standard scaling
        → PCA
        → logistic regression
        → post-hoc probability calibration

Training is grouped by case identifier using grouped cross-validation so that the same case is not split across train and test folds. Out-of-fold predictions are used for performance and calibration metrics. After audit, the final deployable model is refit on all eligible rows for that treatment and serialized with its metadata.

## Teacher approval logic

A trained model is considered for teacher use only after these checks:

- enough labeled rows;
- enough unique cases;
- enough responder and non-responder examples;
- enough successful grouped CV folds;
- adequate cross-validated AUC;
- Brier improvement relative to treatment prior;
- acceptable probability-extremeness fraction;
- sufficient reliability weight.

Models that fail these checks are still visible in audit tables, but they are not promoted as approved teacher models.

## Teacher-builder handoff

`teacher_builder` should consume approved expression model artifacts or expression teacher score tables. It should not retrain expression models internally. This separation is important for reproducibility: model fitting, calibration, reliability assessment, and teacher use are distinct stages.

The safest downstream handoff is:

    outputs/deployable_CH1/model_index_approved.tsv
    outputs/deployable_CH1/models/*.joblib

or, after optional Step 05:

    expression_teacher_scores.tsv
    expression_teacher_summary.tsv

Recommended downstream interpretation:

- use approved models only unless intentionally performing exploratory analysis;
- preserve reliability weights, calibration metadata, and treatment priors;
- treat skipped drugs as informative audit outputs, not silent failures;
- do not retrain expression models inside `teacher_builder`.

## Environment and preflight checks

Before running, confirm that the expected config and Python environment exist:

    Test-Path ".\configs\expression_response_model_v2.yaml"
    Test-Path "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

Check that the configured training table exists:

    $ProjectRoot = "D:\Adv_Omics_Fenyo\project"
    $TrainingTable = "prediction_modeling_pipeline\data_manifests\gdc_expression_training_data\treatment_filtered_outputs\training_table_CH1_tpm_unstranded.tsv"
    Test-Path (Join-Path $ProjectRoot $TrainingTable)

Check that the YAML parses:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -c "import yaml, pathlib; yaml.safe_load(pathlib.Path(r'configs/expression_response_model_v2.yaml').read_text(encoding='utf-8')); print('expression config OK')"

Check package imports:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -c "import numpy, pandas, yaml, joblib, sklearn; print('expression_response_model_v2 environment OK')"

Compile the active Python scripts without running the pipeline:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -m py_compile scripts\expression_model_v2_lib.py scripts\00_build_treatment_ontology.py scripts\01_validate_inputs.py scripts\02_build_canonical_training_table.py scripts\03_train_deployable_models.py scripts\04_audit_deployable_models.py scripts\05_score_visium_expression_teacher.py

From the parent `model_training/` folder, the repository smoke test is:

    .\tests\smoke_test_model_training.ps1

## Verify outputs after Steps 00-04

After running the main workflow, check the required outputs:

    $Out = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\model_training\expression_response_model_v2\outputs\deployable_CH1"
    Test-Path "$Out\treatment_ontology.tsv"
    Test-Path "$Out\data\training_table_canonical.tsv"
    Test-Path "$Out\data\gene_columns.txt"
    Test-Path "$Out\model_index.tsv"
    Test-Path "$Out\model_index_approved.tsv"
    Test-Path "$Out\audit\deployable_model_audit_summary.txt"
    Test-Path "$Out\audit\deployable_model_audit_checks.tsv"

A compact post-run model summary can be printed with:

    $Index = "$Out\model_index.tsv"
    $Approved = "$Out\model_index_approved.tsv"
    python -c "import pandas as pd; idx=pd.read_csv(r'$Index', sep='\t'); app=pd.read_csv(r'$Approved', sep='\t'); print('models:', len(idx)); print('approved:', len(app)); print(idx[[ 'drug', 'approved_for_teacher', 'reliability_weight' ]].head(20).to_string(index=False))"

## Step 05 Visium scoring requirements

Step 05 is optional and requires all of the following:

- `outputs/deployable_CH1/model_index.tsv`;
- trained model artifacts under `outputs/deployable_CH1/models/`;
- approved models if `teacher_scoring.approved_only` is true;
- the configured Visium pseudobulk table;
- the configured Visium sample column, usually `slide_id`.

The Visium pseudobulk path is controlled by `teacher_scoring.visium_pseudobulk_table` in the YAML config.

Before Step 05, confirm the pseudobulk table exists:

    $Pseudo = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder\outputs\output_run_102_reactome75_20260503\02_expression_teacher\visium_pseudobulk_expression.tsv"
    Test-Path $Pseudo

If this returns `False`, do not run Step 05 until the upstream teacher-builder expression pseudobulk output exists or the YAML path is updated.

## Common rerun patterns

Rerun only validation:

    .\run_expression_response_model_v2.ps1 -StartAt 1 -StopAt 1

Rebuild the canonical table after changing treatment or response labels:

    .\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2

Retrain and audit models after canonical data changes:

    .\run_expression_response_model_v2.ps1 -StartAt 3 -StopAt 4

Run the full main workflow again:

    .\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4

Run optional Visium scoring only:

    .\run_expression_response_model_v2.ps1 -StartAt 5 -StopAt 5

## GitHub and publication notes

For GitHub, commit active source scripts, configs, READMEs, runbooks, tests, small manifests, and documentation. Do not commit large local outputs unless using external archival storage or Git LFS.

Large local artifacts include:

- canonical expression training tables;
- fold prediction tables;
- trained `.joblib` model artifacts;
- generated teacher-score tables;
- local archives and development backups.

The output README under `outputs/` describes which artifacts are canonical and which are too large for ordinary Git commits.

## Troubleshooting

If PowerShell blocks script execution, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If the runner cannot find Python, either restore the project virtual environment or pass Python explicitly:

    .\run_expression_response_model_v2.ps1 -Python "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -StartAt 0 -StopAt 4

If Step 03 says the canonical table is missing, run Step 02 first:

    .\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2

If Step 04 reports no trained models in a tiny scratch run, check whether the sample size is too small to train any deployable model. A 5-row micro rerun can test plumbing but is not enough to validate expression model training.

If Step 05 says the Visium pseudobulk table is missing, confirm the upstream teacher-builder output exists or update `teacher_scoring.visium_pseudobulk_table` in the YAML config.

## Review and validation notes

This module has been reviewed with non-behavioral documentation edits. Python files were checked with `py_compile` and AST-equivalence checks after ignoring docstrings. YAML comments were added without changing parsed values. PowerShell comments were added without changing executable command lines.

The repository-level smoke test and micro rerun checks are intended to verify file contracts and runtime plumbing without changing canonical outputs. A scientifically meaningful full rerun should use the full configured data and should be performed intentionally.
