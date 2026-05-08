# Spatial Prediction Model V2

`spatial_prediction_model_V2` is the current governed spatial prediction model used in the downstream prediction-modeling layer of the spatial omics tumor drug response project.

The pipeline starts from a completed `teacher_builder` handoff and asks whether spatial biology features explain treatment-response residuals beyond treatment-prior expectations. It is designed for interpretable residual modeling, biological mechanism discovery, treatment-specific validation, and downstream reporting. It is not intended to make clinical treatment recommendations.

## Overview

The V2 pipeline uses spatial feature tables, fused teacher labels, treatment priors, and governed sample-treatment training data to model prior-adjusted response residuals. Instead of treating treatment identity alone as the final explanation, the workflow tests whether spatial tumor architecture, immune context, stromal organization, metabolic state, accessibility, motifs, hotspots, gradients, and other spatial biology features are associated with above-prior or below-prior response evidence.

The primary outputs are residual spatial biology models, treatment-level model validation results, recurrent biology themes, interpretation tables, publication-support tables, and final QC reports. Generated outputs are local artifacts and are not tracked in GitHub.

## Relationship to other modules

`spatial_prediction_model_V2` is downstream of:

```text
prediction_modeling_pipeline/teacher_builder/
spatial_feature_identification_pipeline/
```

It is upstream of:

```text
prediction_modeling_pipeline/prediction_interpretation_model/
prediction_modeling_pipeline/spatial_transfer_inference_model/
```

The older `spatial_prediction_model/` folder is retained for review and provenance because it contains earlier model-selection and modeling logic that informed the governed V2 design. The current production workflow is `spatial_prediction_model_V2/`.

## Expected input

The main input is a completed teacher-builder handoff folder containing prediction-ready tables. A typical handoff includes files such as:

```text
model_input_numeric.csv
visium_fused_teacher_table.tsv
feature_manifest.csv
prediction_ready_training_table.tsv
```

The exact source paths should be controlled by command-line arguments or configuration files. Machine-specific local paths should not be committed to GitHub.

## Repository layout

```text
spatial_prediction_model_V2/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
├── examples/
├── scripts/
├── src/
├── tests/
└── outputs/        # generated locally; not committed
```

Expected source folders:

```text
scripts/        numbered workflow scripts and the main orchestrator
src/spm_v2/     reusable package code for V2 logic
configs/        reusable or example configuration files
examples/       example launch scripts for smoke and full runs
tests/          lightweight tests or smoke checks
```

Generated folders such as `outputs/`, `logs/`, `local/`, `archive/`, `backup/`, and temporary run folders should remain local and should not be committed.

## Pipeline steps

The V2 workflow is organized as numbered stages run by the main orchestrator:

```text
00_run_spatial_prediction_model_v2.py
```

Major pipeline functions include:

1. Input validation
2. Modeling dataset construction
3. Probability baseline modeling
4. Pair-level residual modeling
5. Residual biology registry construction
6. Broad residual model training
7. Filtered per-treatment residual models
8. Tiered residual model curation
9. Label-shuffle validation
10. Integrated interpretation package generation
11. Publication table generation
12. Output QC
```

These steps validate the input handoff, build model-ready sample-treatment tables, model residual response signals, curate treatment-specific models, validate selected models against shuffled-label controls, and package final source-of-truth outputs for interpretation.

## Smoke run

A smoke run is a lightweight end-to-end check. It is intended to confirm that the code can execute, required inputs are visible, and output structure is created. It is not a full biological validation.

From the `spatial_prediction_model_V2` folder:

```powershell
python .\scripts\00_run_spatial_prediction_model_v2.py `
    --mode smoke `
    --handoff-root "<path-to-teacher-builder-handoff>" `
    --max-workers 0 `
    --open-output
```

Use the smoke run before running the full cohort workflow.

## Full run

A full run performs the governed production workflow, including label-shuffle validation.

```powershell
python .\scripts\00_run_spatial_prediction_model_v2.py `
    --mode full `
    --handoff-root "<path-to-teacher-builder-handoff>" `
    --max-workers 0 `
    --full-step09-n-shuffles 100 `
    --full-step09-n-repeats 5 `
    --open-output
```

Adjust shuffle and repeat settings only when intentionally changing the validation burden. Full runs may take longer and may create large output folders.

## Main output concepts

A successful full run produces local outputs such as:

```text
validated input reports
modeling dataset tables
probability baseline outputs
pair-level residual tables
residual biology registry tables
broad residual model outputs
per-treatment residual model outputs
tiered model curation tables
label-shuffle validation reports
integrated interpretation package
publication-support tables
final QC reports
```

These outputs are designed to support downstream biological interpretation and reporting. They are not committed to GitHub by default.

## Source-of-truth role

A completed V2 full run is the source-of-truth input for the downstream `prediction_interpretation_model`. The interpretation layer should consume completed V2 outputs; it should not rerun V2, perform open model selection, or use deprecated prediction-interpretation outputs as source truth.

## GitHub policy

Commit:

```text
README.md
.gitignore
.gitattributes
scripts/
src/
configs/ example or reusable configs
examples/
tests/
small durable documentation
```

Do not commit:

```text
outputs/
logs/
local/
archive/
backup/
private/
temp/
raw data
large CSV/TSV result tables
Excel workbooks
figures
PDFs
ZIP packages
model artifacts
patch backups
combined code/data bundles
```

Generated outputs should be regenerated locally from the code and required input data, or archived externally through an intentional data-release strategy.

## Interpretation caveats

The V2 pipeline models associations between spatial biology features and prior-adjusted treatment-response residuals. These models support biological interpretation and hypothesis generation. They do not establish causality and do not provide clinical treatment recommendations.

## Recommended reviewer path

For review, start with:

```text
README.md
examples/run_smoke.ps1
examples/run_full.ps1
scripts/00_run_spatial_prediction_model_v2.py
src/spm_v2/
```

Then inspect the numbered scripts and final QC outputs from a local run if outputs are available.

