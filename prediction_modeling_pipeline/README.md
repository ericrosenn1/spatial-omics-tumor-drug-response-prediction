# Prediction Modeling Pipeline

This folder contains the downstream modeling, teacher-label construction, spatial response prediction, biological interpretation, and transfer-inference components of the spatial omics tumor drug response project.

The code is organized as a source-focused GitHub package. Generated outputs, trained model artifacts, large training tables, logs, local archives, and machine-specific run products are excluded from version control.

## Overview

The prediction modeling pipeline connects spatial feature tables from the Visium feature-identification workflow to treatment-response modeling and interpretation. It includes:

1. Training expression- and histology-based response models
2. Building governed teacher labels for Visium samples
3. Training spatial response models from spatial features and teacher signals
4. Interpreting spatial response associations biologically
5. Applying the frozen interpretation atlas to new Visium samples

The project is designed for scientific interpretation and auditability. Outputs should be interpreted as model-derived spatial response-alignment evidence, not as clinical treatment recommendations.

## Repository structure

```text
prediction_modeling_pipeline/
├── README.md
├── model_training/
├── teacher_builder/
├── spatial_prediction_model/
├── spatial_prediction_model_V2/
├── prediction_interpretation_model/
└── spatial_transfer_inference_model/
```

## Major components

### 1. Model training

Path:

```text
model_training/
```

This module contains source code for expression- and histology-based response modeling.

Main submodules:

```text
model_training/expression_response_model_v2/
model_training/histology_response_model_v2/
```

The expression model workflow supports treatment ontology construction, expression training-table construction, deployable model training, model auditing, and scoring Visium samples.

The histology model workflow supports case label construction, slide manifest generation, tiling, tile-level training table construction, patient splitting, model training, control inference, and model auditing.

Generated model files, training tables, tile outputs, prediction tables, and artifact metrics are local outputs and are not tracked in GitHub.

### 2. Teacher builder

Path:

```text
teacher_builder/
```

The teacher builder combines expression and histology response information into governed, prediction-ready teacher tables for spatial modeling.

It supports input validation, expression teacher construction, histology teacher construction, teacher fusion, final teacher-table construction, and QC.

The main local handoff products include fused teacher tables and prediction-ready training tables. These are generated locally and excluded from GitHub.

### 3. Spatial prediction model

Path:

```text
spatial_prediction_model/
```

This folder contains the earlier spatial prediction workflow retained for review, provenance, and comparison with the governed V2 implementation.

It includes scripts for validating inputs, building spatial modeling datasets, training spatial response models, generating sample-treatment predictions, residual modeling, label-shuffle validation, interpretation packaging, and publication-support table generation.

### 4. Spatial prediction model V2

Path:

```text
spatial_prediction_model_V2/
```

This is the current governed spatial prediction model implementation.

Major functions include:

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

This model uses spatial feature tables and teacher outputs to model treatment response residuals and identify recurrent spatial biology themes associated with predicted sensitivity or resistance.

### 5. Prediction interpretation model

Path:

```text
prediction_interpretation_model/
```

This module converts spatial prediction model outputs into structured biological interpretation products.

It prepares interpretation inputs, builds feature and treatment dictionaries, computes signed spatial effects, creates treatment interpretation cards, creates sample-level interpretations, builds a mechanism atlas, and packages final outputs.

This module consumes completed `spatial_prediction_model_V2` outputs as its source data. It should not rerun V2, redo model selection, or use deprecated prediction-interpretation outputs as source truth.

### 6. Spatial transfer inference model

Path:

```text
spatial_transfer_inference_model/
```

This module applies the completed prediction interpretation atlas to one or more new Visium samples.

It takes transfer-ready spatial feature tables, aligns each sample to the frozen strict-feature registry, and scores sample-by-treatment spatial response alignment. It supports both single-sample transfer and small multi-sample transfer batches.

Generated transfer packages, QC reports, feature contribution tables, theme contribution tables, and sample-treatment interpretation tables are local outputs and are not tracked in GitHub.

## Conceptual workflow

```text
Spatial feature identification pipeline
        ↓
model_training/
        ↓
teacher_builder/
        ↓
spatial_prediction_model_V2/
        ↓
prediction_interpretation_model/
        ↓
spatial_transfer_inference_model/
```

The earlier `spatial_prediction_model/` folder is retained for review and provenance. The current primary governed implementation is `spatial_prediction_model_V2/`.

## Expected inputs

The modeling pipeline expects locally generated or externally prepared inputs, including:

```text
Spatial feature tables from spatial_feature_identification_pipeline
Expression response model outputs
Histology response model outputs
Teacher builder outputs
Spatial prediction model V2 outputs
Prediction interpretation model outputs
Transfer-ready Visium feature tables
```

Large input data, model artifacts, generated outputs, and local result folders are not included in GitHub.

## Configuration

Paths should be controlled through configuration files rather than hardcoded in scripts.

Before running on a new machine, update YAML or JSON config files to point to local data locations, output folders, model roots, and completed upstream run folders.

GitHub-facing configs should use relative paths or placeholders when possible. Machine-specific configs with absolute local paths should be treated as local run files and should not be committed unless intentionally redacted as examples.

## Running workflows

Each major module contains its own README, runbook, scripts, or examples. Start with the module-specific documentation before running a workflow.

Typical order for a full local run:

1. Train or load expression and histology response models.
2. Build governed teacher tables.
3. Run spatial prediction model V2.
4. Run prediction interpretation model.
5. Run spatial transfer inference on new processed Visium sample(s), if needed.
6. Review QC, validation, interpretation, and transfer reports.

## Output policy

Generated folders such as the following should remain local:

```text
outputs/
logs/
local/
archive/
backup/
deprecated/
private/
temp/
model artifacts
large CSV/TSV tables
Excel workbooks
figures
PDFs
ZIP packages
H5/H5AD files
```

Only source code, small configuration examples, durable documentation, and lightweight test or example files should be committed to GitHub.

## Notes for reviewers

This repository snapshot is source-focused. It does not include the raw expression datasets, whole-slide images, Visium raw data, generated model outputs, trained models, or transfer packages needed to reproduce every result directly after cloning.

To reproduce results, provide equivalent local input data, update configuration paths, run the relevant modules in order, and regenerate outputs locally.
