# Spatial Omics Tumor Drug Response Prediction

A modular computational workflow for extracting tumor microenvironment features from 10x Genomics Visium spatial transcriptomics data and relating those features to treatment-response estimates derived from expression and histology models.

The workflow supports:

1. Spatial feature identification from Visium samples
2. Construction of treatment-specific response targets from expression and histology models
3. Treatment-specific spatial response modeling and biological interpretation
4. Prediction and interpretation for additional Visium samples

The repository contains source code, configuration files, documentation, tests, and small reproducibility files. Large raw datasets, generated outputs, trained model files, figures, spreadsheets, and local archives are intentionally excluded from GitHub.

## How to use the workflow

For most users, the workflow can be run end to end from Visium input without rebuilding the expression- and histology-response models.

With the included precomputed response-target package, the default path is:

```text
Visium input
    |
    v
Spatial preprocessing and feature extraction
    |
    +---- included precomputed treatment-response targets
    |
    v
Treatment-specific spatial response modeling
    |
    v
Feature and biological interpretation
    |
    v
Treatment-specific predictions and weighted feature scores
```

The included response-target package supplies the development targets and matching spatial feature references needed to fit the downstream spatial models. A user can therefore provide Visium data, extract spatial features, run the spatial-response workflow, and generate treatment-specific outputs without supplying the original expression-training data or histology whole-slide images.

Users who want to regenerate the response targets can instead run the expression- and histology-response training modules and rebuild the response-target package before continuing with the same spatial workflow.

The code and filenames use the term **teacher** for these model-derived treatment-response targets. These are training targets for the spatial models, not observed treatment outcomes from the Visium samples.

## Configuration

The repository includes a root-level configuration template:

```text
project_profile.example.yaml
```

This file records common data and output locations, points to module-specific configuration files, and controls whether the included precomputed response-target package is used. Detailed analysis parameters remain in the YAML or JSON configuration files for each module.

For a new machine:

```powershell
Copy-Item project_profile.example.yaml project_profile.local.yaml
notepad project_profile.local.yaml
```

The local profile is ignored by Git and can contain machine-specific paths.

For the standard end-to-end Visium workflow using the included response-target package:

```yaml
workflow_mode:
  use_precomputed_teacher_handoff: true
```

With this setting, Visium data can proceed through spatial feature extraction, spatial-response modeling, interpretation, and prediction without retraining the upstream expression and histology response models.

To rebuild the complete workflow, set `use_precomputed_teacher_handoff` to `false`, configure the upstream expression and histology inputs, run `model_training`, and regenerate the response-target files with `teacher_builder`.

### Included precomputed response-target package

The repository includes one compact analysis-derived input package:

```text
prediction_modeling_pipeline/teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz
prediction_modeling_pipeline/teacher_builder/model_input_numeric.csv
prediction_modeling_pipeline/teacher_builder/feature_manifest.csv
prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json
```

The fused teacher table contains the treatment-specific response targets used to fit the downstream spatial models. The matching development spatial feature table, ordered feature list, and manifest are included so the package can be validated before use.

These are the only study-derived analysis files intentionally distributed with the source repository. Raw expression data, raw histology data, whole-slide images, H5AD files, complete generated outputs, and trained upstream model files are not included.

Users who want full upstream reproduction can regenerate these files by running `model_training/` followed by `teacher_builder/`.

## Repository structure

```text
.
├── spatial_feature_identification_pipeline/
│   ├── README.md
│   ├── run_pipeline.py
│   ├── code/
│   ├── configs/
│   └── tools/
│
├── prediction_modeling_pipeline/
│   ├── README.md
│   ├── model_training/
│   │   ├── expression_response_model_v2/
│   │   └── histology_response_model_v2/
│   ├── teacher_builder/
│   ├── spatial_prediction_model_V2/
│   ├── prediction_interpretation_model/
│   └── spatial_transfer_inference_model/
│
├── data_manifest/
├── docs/
├── scripts/
├── configs/
├── tests/
├── project_profile.example.yaml
├── requirements-reviewer.txt
├── requirements-notebooks.txt
├── .gitignore
└── .gitattributes
```

## Major components

### 1. Spatial feature identification

Path:

```text
spatial_feature_identification_pipeline/
```

This pipeline processes Visium spatial transcriptomics samples and generates section-level feature tables for downstream modeling. It includes input validation, sample processing, structural/functional/metabolic program scoring, accessibility measurements, hotspot and fragmentation metrics, pairwise spatial relationships, tumor-boundary measurements, model-ready feature construction, histology overlays, visual summaries, and comparison with published spatial annotations.

Main files:

```text
spatial_feature_identification_pipeline/run_pipeline.py
spatial_feature_identification_pipeline/code/
spatial_feature_identification_pipeline/configs/
spatial_feature_identification_pipeline/tools/
```

The principal downstream output is a model-ready section-level feature table.

### 2. Expression and histology response models

Path:

```text
prediction_modeling_pipeline/model_training/
```

Subcomponents:

```text
prediction_modeling_pipeline/model_training/expression_response_model_v2/
prediction_modeling_pipeline/model_training/histology_response_model_v2/
```

The expression workflow supports treatment-name harmonization, input validation, training-table construction, grouped cross-validation, model fitting and calibration, reliability estimation, and scoring Visium sections.

The histology workflow supports response-label construction, slide manifests, image tiling, patient-level data splitting, model training, blank/noise control evaluation, model performance checks, and scoring Visium histology images.

Predictions from these models are used to construct the treatment-specific response targets for the spatial models.

### 3. Response-target construction

Path:

```text
prediction_modeling_pipeline/teacher_builder/
```

The teacher builder combines expression- and histology-derived response estimates into treatment-specific response-target tables. A treatment prior is the baseline responder fraction for a treatment profile in the source response data. Available expression and histology estimates are combined relative to that prior using model-reliability weights, and the resulting prior-adjusted residuals are used as targets for spatial modeling.

The module also performs input validation and quality control.

Main files:

```text
prediction_modeling_pipeline/teacher_builder/scripts/
prediction_modeling_pipeline/teacher_builder/configs/
prediction_modeling_pipeline/teacher_builder/README.md
```

### 4. Spatial prediction model

Path:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/
```

This module uses section-level spatial feature tables and treatment-specific response targets to train and interpret treatment-specific spatial response models.

Major functions include:

1. Input validation
2. Modeling dataset construction
3. Pooled response modeling
4. Treatment-prior residual construction
5. Spatial feature selection for residual modeling
6. Pooled residual modeling
7. Per-treatment residual modeling
8. Treatment-model screening
9. Within-treatment permutation validation
10. Feature and biological interpretation outputs
11. Summary table generation
12. Output quality control

Main files:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/
prediction_modeling_pipeline/spatial_prediction_model_V2/src/spm_v2/
prediction_modeling_pipeline/spatial_prediction_model_V2/configs/
prediction_modeling_pipeline/spatial_prediction_model_V2/tests/
```

### 5. Prediction interpretation

Path:

```text
prediction_modeling_pipeline/prediction_interpretation_model/
```

This component converts spatial-model outputs into treatment-level and sample-level interpretation tables. It maps model features to readable biological labels, calculates signed feature effects, summarizes feature contributions by biological theme, and prepares outputs for downstream analysis and reporting.

Main outputs include:

```text
per-treatment interpretation summaries
sample-by-treatment weighted feature scores
feature contribution tables
biological-theme summaries
figures and reports
quality-control tables
```

### 6. Prediction on new Visium samples

Path:

```text
prediction_modeling_pipeline/spatial_transfer_inference_model/
```

This module generates outputs for Visium samples that were not used to fit the spatial models. After spatial features are extracted for a new sample, the module aligns them to the saved model feature order and applies the fitted treatment-specific models and feature weights.

Outputs include:

```text
treatment-specific residual predictions
weighted feature scores
feature contribution tables
biological-theme contribution tables
feature coverage and quality-control summaries
```

The fitted treatment models are not retrained for these samples.

## Public Visium data staging

Large public Visium source files are not stored directly in the repository. A staging manifest and reconstruction utility are provided to create the local input layout used by the spatial feature pipeline.

Manifest:

```text
data_manifest/public_visium_cohort_staging_manifest.tsv
```

Script:

```text
scripts/download_and_reconstruct_public_visium_sources.py
```

Documentation:

```text
docs/PUBLIC_SOURCE_RECONSTRUCTION.md
```

Expected local layout:

```text
Visium_samples/
  raw_visium_new/
  visium_cohort_clean/
  public_visium_staging_inventory.tsv
  public_visium_staging_summary.txt
```

Dry run:

```powershell
python scripts\download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage `
    --dry-run
```

Staging run:

```powershell
python scripts\download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage
```

The staging manifest preserves stable internal sample IDs `SAMPLE_0000` through `SAMPLE_0102`. The TLS_VISIUM_USZ samples are staged from Zenodo record `14620362` / DOI `10.5281/zenodo.14620362` into `SAMPLE_0095` through `SAMPLE_0102`. Use `--skip-zenodo` to skip the large TLS archive while staging the other public files.

Generated `metadata.json` files contain source and staging information derived from the manifest.

## Expected local data

A complete upstream rebuild uses Visium spatial transcriptomics data together with the expression and histology resources required by the response models. Large inputs and generated outputs are expected to exist outside GitHub.

Typical local locations are:

```text
YOUR_PROJECT_ROOT/spatial_feature_identification_pipeline/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/teacher_builder/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_prediction_model_V2/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/prediction_interpretation_model/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_transfer_inference_model/outputs
YOUR_PROJECT_ROOT/Visium_samples
```

## Installation

Python 3.12 is used for the maintained workflow.

From the repository root:

```powershell
cd "YOUR_PROJECT_ROOT"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install -r requirements-reviewer.txt -r requirements-notebooks.txt
if ($LASTEXITCODE -ne 0) { throw "Installation failed: $LASTEXITCODE" }

python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed: $LASTEXITCODE" }
```

Upstream expression- and histology-model modules may require additional component-specific dependencies described in their README files.

## Default end-to-end workflow

When using the included response-target package, the recommended order is:

1. Stage or provide Visium input data.
2. Run spatial feature identification.
3. Validate and prepare the included response-target package.
4. Run the spatial prediction model.
5. Run prediction interpretation.
6. Generate treatment-specific predictions and weighted feature scores for the Visium sample or batch.
7. Review quality-control, validation, interpretation, and prediction outputs.

### 1. Run spatial feature identification

```powershell
cd "YOUR_PROJECT_ROOT\spatial_feature_identification_pipeline"

Copy-Item .\configs\visium_cohort_clean.example.yaml .\configs\visium_cohort_clean.local.yaml
python run_pipeline.py --config configs\visium_cohort_clean.local.yaml
```

Edit the local configuration file so that `input_root` points to the staged or locally available Visium data.

### 2. Prepare the included response-target package

From the repository root:

```powershell
python prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\16_prepare_precomputed_teacher.py `
    --manifest prediction_modeling_pipeline\teacher_builder\precomputed_handoff_manifest.json `
    --output local\precomputed_handoff

if ($LASTEXITCODE -ne 0) { throw "Precomputed input validation failed: $LASTEXITCODE" }
```

### 3. Run the spatial prediction model

Smoke run:

```powershell
python prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\00_run_spatial_prediction_model_v2.py `
    --mode smoke `
    --handoff-root local\precomputed_handoff `
    --max-workers 2 `
    --open-output
```

Full run:

```powershell
python prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\00_run_spatial_prediction_model_v2.py `
    --mode full `
    --handoff-root local\precomputed_handoff `
    --max-workers 2 `
    --full-step09-n-shuffles 1000 `
    --full-step09-n-repeats 5 `
    --open-output

if ($LASTEXITCODE -ne 0) { throw "Spatial prediction workflow failed: $LASTEXITCODE" }
```

A smoke run is a lightweight execution check. Use the full run for the complete spatial analysis.

### 4. Run prediction interpretation

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\prediction_interpretation_model"

python scripts\00_run_prediction_interpretation_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --v2-run-root "<path-to-completed-spatial-prediction-model-run>" `
    --run-name "prediction_interpretation_model_full_local" `
    --output-root "outputs\prediction_interpretation_model_full_local" `
    --steps all `
    --open-output
```

### 5. Generate predictions for new Visium samples

First process the new Visium sample or batch with the spatial feature identification pipeline so that a compatible model-ready feature table is available.

Then:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_transfer_inference_model"
```

If using the provided file-map template:

```powershell
Copy-Item .\configs\resolved_pim_transfer_file_map.example.json .\configs\resolved_pim_transfer_file_map.json
notepad .\configs\resolved_pim_transfer_file_map.json
```

Run:

```powershell
python scripts\00_run_spatial_transfer_inference_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --pim-run-root "<path-to-completed-prediction-interpretation-model-run>" `
    --single-slide-feature-table "<path-to-model-ready-feature-table>" `
    --run-name "spatial_transfer_inference_example" `
    --output-root "outputs\spatial_transfer_inference_example" `
    --sample-id "TRANSFER_BATCH" `
    --steps all
```

The feature table should contain one row per sample and a `sample_id` column. A single sample or a small batch can be processed.

## Rebuilding the upstream response targets

The included response-target package is intended to make the Visium workflow usable without rerunning upstream response-model training.

To rebuild those targets from source data, follow the component documentation in:

```text
prediction_modeling_pipeline/model_training/README.md
prediction_modeling_pipeline/model_training/expression_response_model_v2/README.md
prediction_modeling_pipeline/model_training/histology_response_model_v2/README.md
prediction_modeling_pipeline/teacher_builder/README.md
prediction_modeling_pipeline/teacher_builder/configs/README.md
```

After regenerating the response-target package, continue with the spatial prediction workflow above.

## Data included with the repository

Large raw datasets, whole-slide images, processed spatial data, trained model files, generated validation figures, transfer outputs, and large result tables are not distributed in the Git repository.

The included precomputed response-target package allows users to run the downstream Visium spatial workflow without rebuilding the upstream expression and histology response models.

Outputs are research-use model estimates and interpretation products.

## Reproducibility

The repository includes configuration templates, public-source staging utilities, tests, model and feature manifests, deterministic seed handling, saved feature ordering, handoff validation, and quality-control checks at major pipeline stages.

Generated output folders are not tracked in Git. Full upstream reconstruction requires the relevant source data, local configuration paths, and execution of the corresponding pipeline components.

Each major module contains its own README or runbook with additional input, configuration, and execution details.

## Citation

Associated manuscript:

> Rosenn E. *A Visium Transcriptomics Workflow Linking Tumor Spatial Features to Treatment Response*. Manuscript in preparation.

A formal publication citation will be added when available.

## License

No open-source license has yet been assigned to this repository.

## Contact

Eric Rosenn  
GitHub: [@ericrosenn1](https://github.com/ericrosenn1)
