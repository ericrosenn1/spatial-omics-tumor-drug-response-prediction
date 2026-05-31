# expression_response_model_v2

`expression_response_model_v2` trains, audits, and optionally applies deployable expression-response teacher models for the spatial omics tumor drug response project.

The workflow uses expression profiles, treatment labels, and binary response labels to train treatment-specific calibrated models. Approved models can then be used as upstream teacher candidates for `teacher_builder`. This module is not the final spatial prediction model.

This folder is source-focused for GitHub. Generated outputs, trained model artifacts, large training tables, local archives, and development backups are intentionally excluded from version control.

## Scientific role

The larger project asks whether spatial tumor architecture can help explain treatment-response biology. This module contributes the expression-teacher stream:

```text
expression profiles + treatment labels + response labels
        ↓
canonical case-treatment training table
        ↓
calibrated treatment-specific expression models
        ↓
approved expression teacher scores or deployable model artifacts
        ↓
teacher_builder fusion with histology teachers and treatment priors
        ↓
spatial prediction and interpretation workflows
```

Because downstream spatial models learn from teacher labels, this module is intentionally conservative. A trained expression model is not automatically trusted as a teacher. It must satisfy support, grouped cross-validation, calibration, probability-extremeness, and reliability checks before it is promoted to the approved model index.

## Relationship to the full project

This module is one component of:

```text
prediction_modeling_pipeline/model_training/
```

The expected downstream flow is:

```text
expression_response_model_v2
        ↓
teacher_builder
        ↓
spatial_prediction_model_V2
        ↓
prediction_interpretation_model
        ↓
spatial_transfer_inference_model
```

`teacher_builder` should consume approved expression model artifacts or expression teacher-score tables. It should not retrain expression models internally. This separation keeps model fitting, calibration, reliability assessment, teacher fusion, and final spatial modeling as distinct auditable stages.

## What changed from expression_response_model v1

This folder supersedes the older `expression_response_model` workflow. The older workflow is retained only as provenance outside the active source path. Version 2 is organized as a deployable, auditable, teacher-ready expression-response model workflow.

Main improvements in v2:

- separates treatment ontology construction, input validation, canonical training-table construction, model training, model audit, and optional Visium teacher scoring into numbered steps;
- creates a canonical treatment-key contract before model fitting;
- writes canonical expression-response training tables and metadata sidecars;
- trains treatment-specific deployable models rather than only exploratory fold reports;
- serializes trained model artifacts locally as `.joblib` files;
- writes explicit model indexes and approved-model indexes as downstream contracts;
- applies calibration and teacher reliability scoring;
- records skipped drugs and audit checks rather than silently dropping under-supported treatments;
- optionally scores Visium pseudobulk expression for downstream teacher use;
- separates expression model fitting from `teacher_builder`.

## Repository layout

```text
expression_response_model_v2/
├── README.md
├── run_expression_response_model_v2.ps1
├── configs/
├── scripts/
└── docs/
```

Important source files:

| Type | File | Purpose |
| --- | --- | --- |
| Config | `configs/expression_response_model_v2.example.yaml` | Tracked template; copy to `configs/expression_response_model_v2.yaml` for local runs. |
| Runner | `run_expression_response_model_v2.ps1` | Executes the numbered workflow steps. |
| Shared library | `scripts/expression_model_v2_lib.py` | Shared config, treatment, response, calibration, reliability, I/O, and probability-shrinkage helpers. |
| Step 00 | `scripts/00_build_treatment_ontology.py` | Builds treatment ontology and alias tables from raw treatment labels. |
| Step 01 | `scripts/01_validate_inputs.py` | Checks required columns, gene features, response labels, and treatment support. |
| Step 02 | `scripts/02_build_canonical_training_table.py` | Builds canonical expression-response modeling tables and metadata. |
| Step 03 | `scripts/03_train_deployable_models.py` | Trains treatment-specific calibrated logistic-response models. |
| Step 04 | `scripts/04_audit_deployable_models.py` | Audits trained model indexes, calibration fields, reliability weights, and CV predictions. |
| Step 05 | `scripts/05_score_visium_expression_teacher.py` | Optionally scores Visium pseudobulk expression samples with approved deployable models. |

Development backups, temporary dry-run helpers, generated outputs, and local provenance are not part of the active GitHub source package.

## Configuration

For normal use, copy the tracked template to the local runtime config and change paths, thresholds, model settings, teacher-scoring options, and output locations there:

```powershell
Copy-Item .\configs\expression_response_model_v2.example.yaml .\configs\expression_response_model_v2.yaml
```
Do not edit files in `scripts/` unless intentionally changing pipeline logic. The Python scripts are the active source implementation; the YAML config is the normal user-editable control surface.

Configuration rules:

- use YAML values for input paths, output locations, support thresholds, model parameters, calibration settings, and teacher-scoring paths;
- do not edit numbered scripts for routine reruns;
- do not overwrite canonical local outputs unless the rerun is intentional;
- run smoke tests or scratch reruns before changing active output locations;
- keep large training tables and trained model artifacts local or in external archival storage rather than committing them directly to GitHub.

GitHub-facing configs should use relative paths or placeholders when possible. Machine-specific configs with absolute local paths should be treated as local run files unless deliberately redacted as examples.

## Requirements

Use the project-level environment or create an environment for this module. From the module folder:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\model_training\expression_response_model_v2"

python -m pip install --upgrade pip
```

The module requires common Python scientific and machine-learning libraries such as `numpy`, `pandas`, `pyyaml`, `joblib`, and `scikit-learn`. If a parent `model_training` requirements file is provided, install that from the parent folder:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\model_training"
python -m pip install -r requirements-model-training.txt
```

## Quick start

From a PowerShell session:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\model_training\expression_response_model_v2"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4
```

This runs the main deployable expression-response workflow:

```text
Step 00  treatment ontology
Step 01  input validation
Step 02  canonical training table
Step 03  deployable model training
Step 04  deployable model audit
```

Optional Visium expression-teacher scoring is Step 05:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 5 -StopAt 5
```

Run Step 05 only after deployable models have been trained and audited and after the configured Visium pseudobulk expression table exists.

To run one step at a time:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2
```

## Pipeline steps and expected local outputs

The output paths below describe the default local output structure. These generated files are not included in GitHub.

### Step 00: treatment ontology

Builds treatment ontology and alias tables from raw treatment labels.

Typical outputs:

```text
outputs/deployable_CH1/treatment_ontology.tsv
outputs/deployable_CH1/treatment_aliases.tsv
```

### Step 01: input validation

Checks source-table columns, binary response availability, treatment support, and expression gene columns.

Typical outputs:

```text
outputs/deployable_CH1/validation/input_validation_summary.txt
outputs/deployable_CH1/validation/drug_count_summary.tsv
outputs/deployable_CH1/validation/eligible_drugs_preview.tsv
```

### Step 02: canonical training table

Builds the canonical expression-response modeling table, metadata sidecar, conflict table, deduplication report, and gene-column list.

Typical outputs:

```text
outputs/deployable_CH1/data/training_table_canonical.tsv
outputs/deployable_CH1/data/training_metadata_canonical.tsv
outputs/deployable_CH1/data/conflicting_case_drug_labels.tsv
outputs/deployable_CH1/data/deduplication_report.tsv
outputs/deployable_CH1/data/gene_columns.txt
```

### Step 03: deployable model training

Trains treatment-specific calibrated logistic-response models with grouped cross-validation and reliability scoring.

Typical outputs:

```text
outputs/deployable_CH1/models/*.joblib
outputs/deployable_CH1/model_index.tsv
outputs/deployable_CH1/model_index_approved.tsv
outputs/deployable_CH1/skipped_drugs.tsv
outputs/deployable_CH1/cv/cv_fold_summary.tsv
outputs/deployable_CH1/cv/cv_fold_predictions.tsv
```

### Step 04: model audit

Audits trained model indexes, approved teacher models, calibration fields, reliability weights, and cross-validation prediction outputs.

Typical outputs:

```text
outputs/deployable_CH1/audit/deployable_model_audit_summary.txt
outputs/deployable_CH1/audit/deployable_model_audit_checks.tsv
outputs/deployable_CH1/audit/model_index_for_audit.tsv
outputs/deployable_CH1/audit/training_run_summary.json
```

### Step 05: optional Visium teacher scoring

Applies approved deployable models to the configured Visium pseudobulk expression table.

Typical outputs are written to the `teacher_scoring.output_dir` path in the YAML config and may include:

```text
expression_teacher_scores.tsv
expression_teacher_summary.tsv
expression_teacher_scoring_summary.txt
```

## Model design

Each treatment-specific deployable model uses a transparent classical machine-learning structure:

```text
median imputation
    → variance filtering
    → standard scaling
    → PCA
    → logistic regression
    → post-hoc probability calibration
```

Training is grouped by case identifier using grouped cross-validation so that the same case is not split across train and test folds. Out-of-fold predictions are used for performance and calibration metrics. After audit, the final deployable model is refit on all eligible rows for that treatment and serialized locally with metadata.

## Teacher approval logic

A trained model is considered for teacher use only after checks such as:

- enough labeled rows;
- enough unique cases;
- enough responder and non-responder examples;
- enough successful grouped CV folds;
- adequate cross-validated AUC;
- Brier improvement relative to the treatment prior;
- acceptable probability-extremeness fraction;
- sufficient reliability weight.

Models that fail these checks can remain visible in local audit tables, but they are not promoted as approved teacher models.

## Teacher-builder handoff

The safest downstream handoff is:

```text
outputs/deployable_CH1/model_index_approved.tsv
outputs/deployable_CH1/models/*.joblib
```

or, after optional Step 05:

```text
expression_teacher_scores.tsv
expression_teacher_summary.tsv
```

Recommended downstream interpretation:

- use approved models only unless intentionally performing exploratory analysis;
- preserve reliability weights, calibration metadata, and treatment priors;
- treat skipped drugs as informative audit outputs, not silent failures;
- do not retrain expression models inside `teacher_builder`.

## Preflight checks

Before running, confirm that the expected config exists:

```powershell
Test-Path ".\configs\expression_response_model_v2.yaml"
```

Check that the YAML parses:

```powershell
python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path(r'configs/expression_response_model_v2.yaml').read_text(encoding='utf-8')); print('expression config OK')"
```

Check package imports:

```powershell
python -c "import numpy, pandas, yaml, joblib, sklearn; print('expression_response_model_v2 environment OK')"
```

Compile the active Python scripts without running the pipeline:

```powershell
python -m py_compile `
    scripts\expression_model_v2_lib.py `
    scripts\00_build_treatment_ontology.py `
    scripts\01_validate_inputs.py `
    scripts\02_build_canonical_training_table.py `
    scripts\03_train_deployable_models.py `
    scripts\04_audit_deployable_models.py `
    scripts\05_score_visium_expression_teacher.py
```

From the parent `model_training/` folder, the repository smoke test is:

```powershell
.\tests\smoke_test_model_training.ps1
```

## Verify outputs after Steps 00-04

After running the main workflow, check the required local outputs:

```powershell
$Out = "outputs\deployable_CH1"
Test-Path "$Out\treatment_ontology.tsv"
Test-Path "$Out\data\training_table_canonical.tsv"
Test-Path "$Out\data\gene_columns.txt"
Test-Path "$Out\model_index.tsv"
Test-Path "$Out\model_index_approved.tsv"
Test-Path "$Out\audit\deployable_model_audit_summary.txt"
Test-Path "$Out\audit\deployable_model_audit_checks.tsv"
```

A compact post-run model summary can be printed with:

```powershell
$Index = "$Out\model_index.tsv"
$Approved = "$Out\model_index_approved.tsv"
python -c "import pandas as pd; idx=pd.read_csv(r'$Index', sep='\t'); app=pd.read_csv(r'$Approved', sep='\t'); print('models:', len(idx)); print('approved:', len(app)); print(idx[[ 'drug', 'approved_for_teacher', 'reliability_weight' ]].head(20).to_string(index=False))"
```

## Step 05 Visium scoring requirements

Step 05 is optional and requires all of the following:

- local `outputs/deployable_CH1/model_index.tsv`;
- trained model artifacts under local `outputs/deployable_CH1/models/`;
- approved models if `teacher_scoring.approved_only` is true;
- the configured Visium pseudobulk table;
- the configured Visium sample column, usually `slide_id`.

The Visium pseudobulk path is controlled by `teacher_scoring.visium_pseudobulk_table` in the YAML config.

Before Step 05, confirm the configured pseudobulk table exists:

```powershell
$Pseudo = "<path-to-visium-pseudobulk-expression-table>"
Test-Path $Pseudo
```

If this returns `False`, do not run Step 05 until the upstream pseudobulk output exists or the YAML path is updated.

## Common rerun patterns

Rerun only validation:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 1 -StopAt 1
```

Rebuild the canonical table after changing treatment or response labels:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2
```

Retrain and audit models after canonical data changes:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 3 -StopAt 4
```

Run the full main workflow again:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4
```

Run optional Visium scoring only:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 5 -StopAt 5
```

## GitHub and publication notes

For GitHub, commit active source scripts, config templates or small reusable configs, READMEs, runbooks, lightweight tests, and concise durable documentation. Do not commit large local outputs unless using external archival storage or Git LFS.

Keep local or archive externally:

- canonical expression training tables;
- fold prediction tables;
- trained `.joblib`, `.pkl`, `.pt`, or similar model artifacts;
- generated teacher-score tables;
- local archives and development backups;
- one-off diagnostics, scratch-run reports, and patch backups.

The `.gitignore` files are intended to help prevent accidental commits of local provenance, large generated outputs, model artifacts, and temporary files.

## Troubleshooting

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If the runner cannot find Python, either restore/create the project environment or pass Python explicitly:

```powershell
.\run_expression_response_model_v2.ps1 -Python "<path-to-python.exe>" -StartAt 0 -StopAt 4
```

If Step 03 says the canonical table is missing, run Step 02 first:

```powershell
.\run_expression_response_model_v2.ps1 -StartAt 2 -StopAt 2
```

If Step 04 reports no trained models in a tiny scratch run, check whether the sample size is too small to train any deployable model. A five-row micro rerun can test plumbing but is not enough to validate expression model training.

If Step 05 says the Visium pseudobulk table is missing, confirm the upstream teacher-builder or spatial feature output exists, or update `teacher_scoring.visium_pseudobulk_table` in the YAML config.

## Review and validation notes

This module is intended to be reviewed as source code plus configuration. A scientifically meaningful full rerun requires the full configured expression dataset and should be performed intentionally. Smoke tests and micro reruns are useful for checking file contracts and runtime plumbing, but they are not substitutes for full model validation.

