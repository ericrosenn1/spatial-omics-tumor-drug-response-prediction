# Spatial Omics Tumor Drug Response Prediction

This repository contains a source-only version of a spatial omics and machine learning project for identifying tumor microenvironment features from Visium spatial transcriptomics data and using those features to support tumor drug response prediction and interpretation.

The project integrates four major analysis layers:

1. Spatial feature identification from Visium samples
2. Teacher model construction from expression and histology response models
3. Spatial response prediction and biological interpretation of sample-treatment effects
4. Spatial transfer inference for applying a frozen interpretation atlas to new Visium samples

The repository is organized as code, configuration files, documentation, and small examples only. Large raw data, generated outputs, model artifacts, figures, spreadsheets, and local archives are intentionally excluded from GitHub.

## Project overview

## Project profile and configuration

This repository includes a root-level project profile template:

    project_profile.example.yaml

This file is a GitHub-safe starting point for configuring the multi-stage workflow. It does not replace the module-specific YAML or JSON configs. Instead, it records common project roots, external data locations, expected handoff files, module config locations, QC behavior, and whether to use the precomputed teacher-builder handoff.

For a new machine, copy the example profile to a local profile and edit local paths:

    Copy-Item project_profile.example.yaml project_profile.local.yaml
    notepad project_profile.local.yaml

The local profile should not be committed to GitHub. It is ignored by .gitignore and may contain machine-specific paths.

The most reviewer-friendly setting is:

    workflow_mode:
      use_precomputed_teacher_handoff: true

With this setting, users can use the included compressed fused teacher table and do not need to provide expression-training data, histology whole-slide image data, or rerun the upstream model_training workflows just to start the downstream Visium-facing spatial prediction workflow.

Users who want full upstream reproducibility can set use_precomputed_teacher_handoff to false, configure the expression and histology training inputs, run model_training, and regenerate the teacher-builder outputs locally.


<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_START -->
### Optional precomputed teacher handoff

For reviewer convenience, the repository includes one curated compressed teacher-builder handoff:

```text
prediction_modeling_pipeline/teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz
```

This file is a compact derived fused teacher-label table for the 102-sample analysis. It allows downstream Visium-facing workflows to start from the governed teacher labels without rerunning expression-response model training, histology-response model training, or the upstream teacher-builder fusion steps.

The file is not raw expression data, raw histology data, whole-slide image data, h5ad data, or a trained model artifact. Users who want full upstream reproducibility can regenerate it by running `model_training/` followed by `teacher_builder/`; users who want to focus on downstream spatial prediction can use this precomputed handoff.
<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_END -->

The main goal is to build a computational workflow that connects spatial tumor biology to drug response prediction. The workflow begins with spatial transcriptomics data, derives interpretable tissue-level features, links those features to teacher model outputs, evaluates whether spatial biology explains sample-treatment sensitivity or resistance, and applies the resulting interpretation atlas to new Visium samples.

The project is designed around interpretability and scientific auditability rather than only maximizing prediction accuracy. The intended output is a set of pipeline components that can describe which spatial, histologic, expression, immune, stromal, metabolic, tumor-boundary, accessibility, and hotspot features may contribute to treatment response.

## Repository structure

```text
.
├── prediction_modeling_pipeline/
│   ├── README.md
│   ├── model_training/
│   ├── teacher_builder/
│   ├── spatial_prediction_model/
│   ├── spatial_prediction_model_V2/
│   ├── prediction_interpretation_model/
│   └── spatial_transfer_inference_model/
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

The model-ready table produced locally by this pipeline is used as a core input to downstream prediction modeling and transfer inference.

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

### 4. Spatial prediction model

Path:

```text
prediction_modeling_pipeline/spatial_prediction_model/
```

This folder contains the earlier spatial prediction workflow retained for review, provenance, and comparison with the governed V2 implementation. It includes source scripts for validating prediction inputs, constructing spatial modeling datasets, training response models, predicting sample-treatment pairs, running QC, training residual models, generating interpretation outputs, and making supporting publication tables.

The current primary implementation is `spatial_prediction_model_V2/`, but this earlier workflow is retained because it documents the project development path and remains useful for teacher review.

Main files:

```text
prediction_modeling_pipeline/spatial_prediction_model/run_spatial_prediction_model.py
prediction_modeling_pipeline/spatial_prediction_model/run_spatial_prediction_model.ps1
prediction_modeling_pipeline/spatial_prediction_model/scripts/
prediction_modeling_pipeline/spatial_prediction_model/configs/
prediction_modeling_pipeline/spatial_prediction_model/docs/
```

### 5. Spatial prediction model V2

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

### 6. Prediction interpretation model

Path:

```text
prediction_modeling_pipeline/prediction_interpretation_model/
```

This component converts spatial prediction model outputs into structured biological interpretation products. It prepares interpretation inputs, builds feature and treatment dictionaries, computes signed spatial effects, creates treatment interpretation cards, creates sample-level interpretations, builds a mechanism atlas, and packages final outputs.

Main files:

```text
prediction_modeling_pipeline/prediction_interpretation_model/scripts/
prediction_modeling_pipeline/prediction_interpretation_model/configs/
prediction_modeling_pipeline/prediction_interpretation_model/docs/
```

Key outputs generated locally include treatment interpretation cards, sample-treatment signed interpretation scores, feature contribution tables, biology theme atlases, mechanism summaries, final publication tables, figures, reports, and QC packages.

### 7. Spatial transfer inference model

Path:

```text
prediction_modeling_pipeline/spatial_transfer_inference_model/
```

The spatial transfer inference model applies the completed prediction interpretation atlas to one or more new Visium samples. It takes a transfer-ready spatial feature table, aligns each sample to the frozen strict-feature registry, and scores sample-by-treatment spatial response alignment.

This module is used after a Visium sample has been processed by the spatial feature identification pipeline. It supports both single-sample transfer and small multi-sample batches.

Main files:

```text
prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/
prediction_modeling_pipeline/spatial_transfer_inference_model/configs/
prediction_modeling_pipeline/spatial_transfer_inference_model/docs/
```

Main outputs generated locally include:

```text
sample-by-treatment interpretation table
feature contribution table
theme contribution table
confidence / evidence-support labels
QC reports
final transfer package
```

For a single sample with 27 validated treatment signatures, the expected output is 27 sample-treatment rows. For a four-sample batch with the same treatment atlas, the expected output is 108 sample-treatment rows.

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
```

This keeps the GitHub repository focused on source code, configuration, documentation, and small reproducible scaffolding files.

## Expected local data

The full local project used Visium spatial transcriptomics data and additional expression and histology resources. Large inputs and outputs are expected to exist outside GitHub in the user's local project directory.

Typical local folders used during development included:

```text
YOUR_PROJECT_ROOT/spatial_feature_identification_pipeline/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/teacher_builder/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_prediction_model/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_prediction_model_V2/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/prediction_interpretation_model/outputs
YOUR_PROJECT_ROOT/prediction_modeling_pipeline/spatial_transfer_inference_model/outputs
YOUR_PROJECT_ROOT/Visium_samples
```

These folders are not included in the repository.

## Installation

Create and activate a Python environment from the project root.

```powershell
cd "YOUR_PROJECT_ROOT"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Install requirements for the specific component you want to run. For example:

```powershell
pip install -r spatial_feature_identification_pipeline\requirements.txt
pip install -r prediction_modeling_pipeline\spatial_prediction_model_V2\requirements.txt
```

If a component does not include a standalone requirements file, install the requirements listed in that component's README or runbook. Some components may require additional packages depending on whether expression, histology, spatial transcriptomics, visualization, interpretation, or transfer-inference steps are being run.

## Running the spatial feature identification pipeline

From the spatial feature identification pipeline folder:

```powershell
cd "YOUR_PROJECT_ROOT\spatial_feature_identification_pipeline"

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
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\teacher_builder"
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
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_prediction_model_V2"
```

Run the V2 entry point with a config file:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py --config configs\smoke_test.yaml
```

For a full run, use:

```powershell
python scripts\00_run_spatial_prediction_model_v2.py --config configs\full_run.yaml
```

The smoke test should be run before a full cohort run. A smoke test is a lightweight end-to-end check that confirms the pipeline can execute and produce the expected output structure; it is not a full biological validation.

## Running the prediction interpretation model

From the prediction interpretation model folder:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\prediction_interpretation_model"
```

Run the interpretation model entry point with a completed spatial prediction model V2 run:

```powershell
python scripts\00_run_prediction_interpretation_model.py `
    --config configs\example_prediction_interpretation_model_full_run.json
```

This produces signed spatial effects, treatment interpretation cards, sample-level interpretation tables, mechanism atlases, final publication tables, figures, reports, and QC packages.

## Running spatial transfer inference

From the spatial transfer inference model folder:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_transfer_inference_model"
```

Run the transfer entry point with a completed prediction interpretation model run and a transfer-ready feature table:

```powershell
python scripts\00_run_spatial_transfer_inference_model.py `
    --model-root . `
    --pim-run-root "<path-to-completed-prediction-interpretation-model-run>" `
    --single-slide-feature-table "<path-to-transfer-ready-model_input_numeric.csv>" `
    --run-name "spatial_transfer_inference_example" `
    --output-root "outputs\spatial_transfer_inference_example" `
    --sample-id "TRANSFER_BATCH" `
    --steps all
```

The transfer feature table should contain one row per sample and a `sample_id` column. The transfer model can be run on a single sample or on a small batch of samples.

A smoke test can be run before a real transfer run to confirm that the pipeline can execute and produce the expected output structure.

## Repository status

This repository is a source-focused project snapshot. It includes code, configuration, documentation, and tests, but excludes local outputs and large files.

At the time of repository preparation, the codebase included:

1. Spatial feature identification pipeline source code
2. External validation scripts
3. Teacher builder source code
4. Expression response model V2 source code
5. Histology response model V2 source code
6. Earlier spatial prediction model source code retained for review and provenance
7. Spatial prediction model V2 source code
8. Prediction interpretation model source code
9. Spatial transfer inference model source code
10. Documentation and runbooks
11. GitHub-safe `.gitignore` and `.gitattributes`

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
7. Run spatial transfer inference on new Visium sample(s), if applying the frozen atlas to external or newly processed samples
8. Review QC, validation, interpretation, and transfer reports

## Current limitations

This repository is under active development. Some paths in configuration files may need to be updated for a new local machine. Generated outputs, trained model files, raw spatial data, WSI files, validation figures, transfer outputs, and large result tables are not included. Some scripts may require project-specific inputs not distributed with the repository.

The model outputs are intended for research interpretation. They should be interpreted as spatial response-alignment evidence rather than clinical treatment recommendations.

## License

No license has been selected yet. Reuse permissions should be clarified with the repository owner.

## Contact

Repository owner:

```text
ericrosenn1
```


