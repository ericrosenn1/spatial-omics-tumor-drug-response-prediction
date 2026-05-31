# Spatial Omics Tumor Drug Response Prediction

This project explores a research question:

**Can spatial patterns in tumor tissue help explain why a tumor may be more or less sensitive to a treatment?**

The code in this repository builds a multi-stage workflow for Visium spatial transcriptomics data. It starts with spatial feature extraction, connects those features to expression and histology-based teacher signals, trains spatial response prediction models, and then turns model outputs into interpretable biological summaries.

This is a student research project. The goal is to make the workflow understandable, reproducible, and easy to inspect. It is not a clinical tool and should not be used to make treatment decisions.

## What This Repository Contains

This GitHub repository is source-focused. It contains:

- Python and PowerShell pipeline code
- README files and run instructions
- Example configuration files
- Small manifests and reproducibility scaffolding
- Tests and validation utilities

It does not contain the large files used or produced during full local runs, such as:

- Raw Visium data
- Whole-slide images
- Large processed output folders
- Trained model files
- Full figures, spreadsheets, and local result archives

Those files are expected to live on a local machine and are excluded from GitHub to keep the repository manageable.

## Project Map

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
│   ├── spatial_prediction_model/
│   ├── spatial_prediction_model_V2/
│   ├── prediction_interpretation_model/
│   └── spatial_transfer_inference_model/
│
├── data_manifest/
├── docs/
├── scripts/
├── project_profile.example.yaml
└── README.md
```

## The Workflow In Plain Language

The project has seven main parts.

### 1. Spatial Feature Identification

Folder:

```text
spatial_feature_identification_pipeline/
```

This pipeline starts with Visium spatial transcriptomics samples and builds spatial feature tables. These features describe the tumor and its surrounding tissue environment, including patterns related to expression, tissue context, accessibility, hotspots, motifs, and visual summaries.

The most important downstream handoff is a model-ready feature table, usually called:

```text
model_input_numeric.csv
```

That table is used later by the prediction models.

### 2. Expression Response Model

Folder:

```text
prediction_modeling_pipeline/model_training/expression_response_model_v2/
```

This component uses expression-based information to build treatment response teacher signals. In simpler terms, it estimates response-related patterns from expression data so those signals can later guide the spatial prediction model.

### 3. Histology Response Model

Folder:

```text
prediction_modeling_pipeline/model_training/histology_response_model_v2/
```

This component uses histology image information. It includes steps for slide manifests, tile tables, patient splits, model training, control inference, and auditing.

### 4. Teacher Builder

Folder:

```text
prediction_modeling_pipeline/teacher_builder/
```

The teacher builder combines expression and histology information into treatment-aware teacher tables. These tables provide target signals for the spatial prediction model.

For convenience, this repository includes a compact precomputed teacher handoff:

```text
prediction_modeling_pipeline/teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz
```

This file lets the downstream spatial prediction workflow start without rerunning the full expression and histology training process.

### 5. Spatial Prediction Model V2

Folder:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/
```

This is the main current prediction workflow. It uses spatial features and teacher labels to model sample-treatment response patterns.

Major steps include:

- Input validation
- Modeling dataset construction
- Probability baseline modeling
- Pair-level residual modeling
- Residual biology registry construction
- Broad residual model training
- Per-treatment residual modeling
- Label-shuffle validation
- Interpretation package generation
- Publication table generation
- Output QC

The older folder `spatial_prediction_model/` is still kept for provenance and comparison.

### 6. Prediction Interpretation Model

Folder:

```text
prediction_modeling_pipeline/prediction_interpretation_model/
```

This component turns prediction model outputs into biological interpretation products. It builds signed feature effects, treatment cards, sample-level interpretation tables, mechanism summaries, and final output packages.

### 7. Spatial Transfer Inference

Folder:

```text
prediction_modeling_pipeline/spatial_transfer_inference_model/
```

This component applies the completed interpretation atlas to new Visium samples. It aligns a new sample's spatial features to the trained feature registry and scores sample-by-treatment spatial response alignment.

## Project Profile

The root file:

```text
project_profile.example.yaml
```

is a safe template for local setup. It records common paths, expected handoff files, module config locations, and workflow options.

To use it locally:

```powershell
Copy-Item project_profile.example.yaml project_profile.local.yaml
notepad project_profile.local.yaml
```

The `.local.yaml` file should stay on your machine and should not be committed.

For a teacher or reviewer who wants to inspect the downstream workflow without rebuilding every upstream model, the most practical setting is:

```yaml
workflow_mode:
  use_precomputed_teacher_handoff: true
```

## Public Visium Data Staging

The repository includes a helper for reconstructing the local public Visium input layout from a tracked staging manifest.

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

Dry-run example:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage `
    --dry-run
```

Real staging example:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage
```

The script stages public files into a local layout such as:

```text
Visium_samples/
  raw_visium_new/
  visium_cohort_clean/
  public_visium_staging_inventory.tsv
  public_visium_staging_summary.txt
```

The staging workflow preserves the project's stable internal sample IDs from `SAMPLE_0000` through `SAMPLE_0102`. The TLS Visium samples are large and come from Zenodo; use `--skip-zenodo` if you want to stage other public files without that archive.

## Installation

Create a Python environment from the project root:

```powershell
cd "YOUR_PROJECT_ROOT"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Install requirements for the component you want to run. For example:

```powershell
pip install -r spatial_feature_identification_pipeline\requirements.txt
pip install -r prediction_modeling_pipeline\spatial_prediction_model_V2\requirements.txt
```

Some modules have their own README files and configuration notes. Those should be checked before running a full pipeline.

## Basic Run Order

A full local workflow generally follows this order:

1. Prepare or stage the Visium input data.
2. Run the spatial feature identification pipeline.
3. Train or load expression and histology teacher sources.
4. Build the teacher table.
5. Run spatial prediction model V2.
6. Run the prediction interpretation model.
7. Run spatial transfer inference on new Visium samples, if needed.
8. Review QC reports, validation summaries, and interpretation outputs.

## Running Key Components

### Spatial Feature Identification

```powershell
cd "YOUR_PROJECT_ROOT\spatial_feature_identification_pipeline"

Copy-Item .\configs\visium_cohort_clean.example.yaml .\configs\visium_cohort_clean.local.yaml
python run_pipeline.py --config configs\visium_cohort_clean.local.yaml
```

### Teacher Builder

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\teacher_builder"
```

Use the templates in:

```text
prediction_modeling_pipeline/teacher_builder/configs/
```

See:

```text
prediction_modeling_pipeline/teacher_builder/README.md
prediction_modeling_pipeline/teacher_builder/configs/README.md
```

### Spatial Prediction Model V2

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_prediction_model_V2"
```

Smoke run:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py `
    --mode smoke `
    --handoff-root "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_prediction_model\outputs\_derived_handoffs\residual_prior_adjusted_filtered_20260506_223625\full102_handoff" `
    --max-workers 0 `
    --open-output
```

Full run:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py `
    --mode full `
    --handoff-root "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_prediction_model\outputs\_derived_handoffs\residual_prior_adjusted_filtered_20260506_223625\full102_handoff" `
    --max-workers 0 `
    --full-step09-n-shuffles 100 `
    --full-step09-n-repeats 5 `
    --open-output
```

Run the smoke test before a full cohort run. The smoke test checks that the workflow can execute and create the expected output structure; it is not a biological validation.

### Prediction Interpretation Model

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\prediction_interpretation_model"

python scripts\00_run_prediction_interpretation_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --v2-run-root "<path-to-completed-spatial-prediction-model-V2-run>" `
    --run-name "prediction_interpretation_model_full_local" `
    --output-root "outputs\prediction_interpretation_model_full_local" `
    --steps all `
    --open-output
```

### Spatial Transfer Inference

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_transfer_inference_model"
```

If using the resolved transfer file map, first copy the example config and edit local paths:

```powershell
Copy-Item .\configs\resolved_pim_transfer_file_map.example.json .\configs\resolved_pim_transfer_file_map.json
notepad .\configs\resolved_pim_transfer_file_map.json
```

Then run:

```powershell
python scripts\00_run_spatial_transfer_inference_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --pim-run-root "<path-to-completed-prediction-interpretation-model-run>" `
    --single-slide-feature-table "<path-to-transfer-ready-model_input_numeric.csv>" `
    --run-name "spatial_transfer_inference_example" `
    --output-root "outputs\spatial_transfer_inference_example" `
    --sample-id "TRANSFER_BATCH" `
    --steps all
```

The transfer feature table should contain one row per sample and a `sample_id` column.

## How To Read This Project

If you want the basic idea, start here:

1. This root README
2. `spatial_feature_identification_pipeline/README.md`
3. `prediction_modeling_pipeline/README.md`

If you want the modeling details, read:

1. `prediction_modeling_pipeline/model_training/README.md`
2. `prediction_modeling_pipeline/teacher_builder/README.md`
3. `prediction_modeling_pipeline/spatial_prediction_model_V2/README.md`

If you want the final interpretation and transfer steps, read:

1. `prediction_modeling_pipeline/prediction_interpretation_model/README.md`
2. `prediction_modeling_pipeline/spatial_transfer_inference_model/README.md`

## Current Limitations

This repository is still a research project. A fresh clone contains the source code and configuration templates, but a full scientific run also requires local data, local configuration paths, and generated handoff files.

Important limitations:

- Large raw and processed data files are not stored in GitHub.
- Some local paths must be edited before running pipelines.
- Some modules require upstream outputs from earlier modules.
- Model outputs are research evidence, not clinical recommendations.
- Results should be reviewed through the QC and validation reports produced by each component.

## License

No license has been selected yet. Reuse permissions should be clarified with the repository owner.

## Repository Owner

```text
ericrosenn1
```
