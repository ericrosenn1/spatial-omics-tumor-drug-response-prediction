# Spatial Omics Tumor Drug Response Prediction

This repository contains a source-only version of a spatial omics and machine learning project for identifying tumor microenvironment features from Visium spatial transcriptomics data and using those features to support tumor drug response prediction.

The project integrates three major analysis layers:

1. Spatial feature identification from Visium samples
2. Teacher model construction from expression and histology response models
3. Spatial response prediction and interpretation of sample treatment effects

The repository is organized as code, configuration files, documentation, and small examples only. Large raw data, generated outputs, model artifacts, figures, spreadsheets, and local archives are intentionally excluded from GitHub.

## Project overview

The main goal is to build a computational workflow that connects spatial tumor biology to drug response prediction. The workflow begins with spatial transcriptomics data, derives interpretable tissue-level features, links those features to teacher model outputs, and then evaluates whether spatial biology explains sample-treatment sensitivity or resistance.

The project is designed around interpretability and scientific auditability rather than only maximizing prediction accuracy. The intended output is a set of pipeline components that can describe which spatial, histologic, expression, immune, stromal, metabolic, and tumor-boundary features may contribute to treatment response.

## Repository structure

```text
.
├── prediction_modeling_pipeline/
│   ├── README.md
│   ├── model_training/
│   ├── teacher_builder/
│   ├── spatial_prediction_model_V2/
│   └── prediction_interpretation_model/
│
├── spatial_feature_identification_pipeline/
│   ├── README.md
│   ├── run_pipeline.py
│   ├── code/
│   ├── configs/
│   ├── docs/
│   └── tools/
│
├── sort_training_dataset_intake.py
├── .gitignore
└── .gitattributes
```

## Major components

### 1. Spatial feature identification pipeline

Path:

```text
spatial_feature_identification_pipeline/
```

This pipeline processes Visium spatial transcriptomics samples and generates spatial feature tables used by downstream modeling. It includes scripts for input validation, sample processing, slide feature merging, multi-axis transcriptome labeling, accessibility profiles, hotspot metrics, context alignment, motif tables, model-ready feature table construction, overlays, visual summaries, and external study validation.

Main files:

```text
spatial_feature_identification_pipeline/run_pipeline.py
spatial_feature_identification_pipeline/code/
spatial_feature_identification_pipeline/configs/
spatial_feature_identification_pipeline/docs/
```

Important conceptual outputs generated locally, but not committed:

```text
outputs/output_05_build_multi_axis_transcriptome_labels/
outputs/output_06_build_accessibility_profiles/
outputs/output_07_append_hotspot_metrics/
outputs/output_08_01_context_alignment_and_metabolic_concordance/
outputs/output_09_build_motif_tables/
outputs/output_10_build_model_ready_table/
outputs/output_11_overlay/
outputs/output_12_data_analysis_and_visuals/
outputs/output_13_external_study_validation/
```

The model-ready table produced locally by this pipeline is used as a core input to downstream prediction modeling.

### 2. Model training

Path:

```text
prediction_modeling_pipeline/model_training/
```

This section contains source code and documentation for training expression and histology-based response models. These models provide teacher signals that are later fused with spatial biology features.

Subcomponents include:

```text
prediction_modeling_pipeline/model_training/expression_response_model_v2/
prediction_modeling_pipeline/model_training/histology_response_model_v2/
```

Expression model scripts support treatment ontology construction, input validation, canonical training table construction, deployable model training, model auditing, and scoring Visium samples.

Histology model scripts support case label construction, slide manifest generation, tiling, tile training table construction, patient splitting, model training, control inference, and model auditing.

### 3. Teacher builder

Path:

```text
prediction_modeling_pipeline/teacher_builder/
```

The teacher builder combines expression and histology response information into prediction-ready teacher tables. It includes governance logic, input validation, expression teacher construction, histology teacher construction, fusion, final teacher table generation, and QC.

Main files:

```text
prediction_modeling_pipeline/teacher_builder/scripts/
prediction_modeling_pipeline/teacher_builder/configs/
prediction_modeling_pipeline/teacher_builder/docs/
```

The teacher builder is intended to produce calibrated, audited, treatment-aware teacher outputs for the spatial prediction model.

### 4. Spatial prediction model V2

Path:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/
```

This is the main current spatial prediction model implementation. It uses spatial feature tables and teacher outputs to train and interpret treatment response models.

Major functions include:

1. Input validation
2. Modeling dataset construction
3. Probability baseline modeling
4. Pair-level residual modeling
5. Residual biology registry construction
6. Broad residual model training
7. Filtered per-treatment residual models
8. Tiered residual model curation
9. Label shuffle validation
10. Integrated interpretation package generation
11. Publication table generation
12. Output QC

Main files:

```text
prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/
prediction_modeling_pipeline/spatial_prediction_model_V2/src/spm_v2/
prediction_modeling_pipeline/spatial_prediction_model_V2/configs/
prediction_modeling_pipeline/spatial_prediction_model_V2/tests/
```

### 5. Prediction interpretation model

Path:

```text
prediction_modeling_pipeline/prediction_interpretation_model/
```

This component converts model outputs into structured interpretation products. It prepares interpretation inputs, builds feature and treatment dictionaries, computes signed spatial effects, creates treatment interpretation cards, creates sample-level interpretations, builds a mechanism atlas, and packages final outputs.

Main files:

```text
prediction_modeling_pipeline/prediction_interpretation_model/scripts/
prediction_modeling_pipeline/prediction_interpretation_model/configs/
prediction_modeling_pipeline/prediction_interpretation_model/docs/
```

## Data availability and GitHub exclusions

This repository does not include raw data or generated outputs.

The following are intentionally excluded by `.gitignore`:

```text
.venv/
outputs/
logs/
local/
archive/
backup/
deprecated/
private/
temp/
Visium_samples/
raw data folders
processed data folders
large CSV and TSV outputs
Excel files
figures
PDFs
ZIP archives
H5 and H5AD files
model artifacts
combined transcript bundles
temporary pasted text files
```

This keeps the GitHub repository focused on source code, configuration, documentation, and small reproducible scaffolding files.

## Expected local data

The full local project used Visium spatial transcriptomics data and additional expression and histology resources. Large inputs and outputs are expected to exist outside GitHub in the user's local project directory.

Typical local paths used during development included:

```text
D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline\outputs
D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder\outputs
D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\spatial_prediction_model_V2\outputs
D:\Adv_Omics_Fenyo\project\Visium_samples
```

These folders are not included in the repository.

## Installation

Create and activate a Python environment from the project root.

```powershell
cd "D:\Adv_Omics_Fenyo\project"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Install requirements for the specific component you want to run. For example:

```powershell
pip install -r spatial_feature_identification_pipeline\requirements.txt
pip install -r prediction_modeling_pipeline\spatial_prediction_model_V2\requirements.txt
```

Some components may require additional packages depending on whether expression, histology, spatial transcriptomics, visualization, or model interpretation steps are being run.

## Running the spatial feature identification pipeline

From the spatial feature identification pipeline folder:

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

python run_pipeline.py --config configs\visium_cohort_clean.yaml
```

Configuration files are stored in:

```text
spatial_feature_identification_pipeline/configs/
```

Documentation and run instructions are stored in:

```text
spatial_feature_identification_pipeline/docs/
```

## Running the teacher builder

From the teacher builder folder:

```powershell
cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder"
```

Use the configuration files in:

```text
prediction_modeling_pipeline/teacher_builder/configs/
```

Scripts are located in:

```text
prediction_modeling_pipeline/teacher_builder/scripts/
```

See:

```text
prediction_modeling_pipeline/teacher_builder/docs/RUNBOOK_teacher_builder.md
```

## Running spatial prediction model V2

From the V2 spatial prediction model folder:

```powershell
cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\spatial_prediction_model_V2"
```

Run the V2 entry point with a config file:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py --config configs\smoke_test.yaml
```

For a full run, use:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py --config configs\full_run.yaml
```

The smoke test should be run before a full cohort run.

## Repository status

This repository is a source-focused project snapshot. It includes code, configuration, documentation, and tests, but excludes local outputs and large files.

At the time of repository preparation, the codebase included:

1. Spatial feature identification pipeline source code
2. External validation scripts
3. Teacher builder source code
4. Expression response model V2 source code
5. Histology response model V2 source code
6. Spatial prediction model V2 source code
7. Prediction interpretation model source code
8. Documentation and runbooks
9. GitHub-safe `.gitignore` and `.gitattributes`

## Reproducibility notes

Generated output folders are not tracked in Git. To reproduce results, users need to provide local input data, update configuration paths, run the relevant pipeline components, and regenerate outputs locally.

Each major submodule contains its own README or runbook. Begin with the component README before running scripts.

Recommended order for a full local workflow:

1. Prepare Visium data and manifests
2. Run the spatial feature identification pipeline
3. Train or load expression and histology response models
4. Build teacher tables
5. Run spatial prediction model V2
6. Run prediction interpretation model
7. Review QC and validation reports

## Current limitations

This repository is under active development. Some paths in configuration files may need to be updated for a new local machine. Generated outputs, trained model files, raw spatial data, WSI files, and validation figures are not included. Some scripts may require project-specific inputs not distributed with the repository.

## License

See the included license file if present. If no license file is provided, reuse permissions should be clarified before redistribution.

## Contact

Repository owner:

```text
ericrosenn1
```
