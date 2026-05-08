# Spatial Feature Identification Pipeline

This folder contains the spatial transcriptomics feature-engineering pipeline used to convert 10x Visium samples into biologically interpretable spatial feature tables for downstream teacher building, spatial response prediction, prediction interpretation, and transfer inference.

The pipeline is source-focused in GitHub. Raw Visium data, processed h5ad files, generated outputs, validation figures, local archives, and run logs are intentionally excluded from version control.

## Overview

The pipeline builds a multi-layer description of each Visium sample. It performs input validation, per-sample processing, slide-level feature merging, program scoring, multi-axis transcriptome labeling, accessibility profiling, hotspot extraction, metabolic/context alignment, spatial motif table construction, model-ready feature-table construction, overlay generation, cohort-level visualization, and external study validation.

The pipeline is intended for scientific feature engineering and interpretation. It does not make clinical treatment recommendations. Its compact model-ready output is used by downstream modeling components in `prediction_modeling_pipeline/`.

## Repository layout

```text
spatial_feature_identification_pipeline/
├── README.md
├── run_pipeline.py
├── requirements.txt
├── configs/
├── code/
├── tools/
└── docs/
```

`run_pipeline.py` is the main runner for Steps 01 through 12.

`configs/` contains YAML configuration files. Users should update local paths before running the pipeline on a new machine.

`code/` contains active numbered pipeline scripts and reusable internal helper code.

`tools/` contains audit, maintenance, and repository-check utilities.

`docs/` contains runbooks and source-facing documentation.

Generated folders such as `outputs/`, `logs/`, `_repo_local_archive/`, `_LIVE_.txt`, and local provenance/audit reports are not part of the GitHub-tracked source package.

## Current analysis reference

The local analysis used 103 input sample folders and produced 102 usable downstream samples. `SAMPLE_0049` was skipped because it had `01_loaded.h5ad` but no `02_processed.h5ad`.

Reference local analysis state:

```text
Input sample folders: 103
Usable downstream samples: 102
Skipped sample: SAMPLE_0049

Step 02 loaded h5ad files: 103
Step 02 processed h5ad files: 102
Step 02 slide feature rows: 102
Step 02 cluster summaries: 102

Step 03 through Step 11 main tables: 102 sample rows
Step 10 model table: 102 sample rows
Step 10 feature manifest: 720 features

Reactome max terms: 75
Reactome terms loaded: 75
Hallmark terms loaded: 11
External library source: msigdb_2023.1.Hs
```

These generated outputs are not included in GitHub. They must be regenerated locally from the appropriate input data and configuration paths.

The main downstream handoff files produced by a local run are:

```text
outputs/output_10_build_model_ready_table/model_input_numeric.csv
outputs/output_10_build_model_ready_table/feature_manifest.csv
```

These files provide the compact numeric feature matrix and feature manifest used by downstream teacher-building, spatial-prediction, interpretation, and transfer-inference workflows.

## Requirements

Create and activate a Python environment from the project root or from this pipeline folder.

```powershell
cd "<path-to-project>/spatial_feature_identification_pipeline"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If this repository is used as part of the full project, a shared project-level environment can also be used, provided it contains the scientific Python packages required by the scripts. Typical dependencies include `pandas`, `numpy`, `scanpy`, `anndata`, `scipy`, `scikit-learn`, `matplotlib`, YAML parsing utilities, and image/plotting libraries used by selected steps.

## Configuration

The active cohort configuration used during development was:

```text
configs/visium_cohort_clean.yaml
```

A full reference configuration may also be present:

```text
configs/spatial_feature_pipeline_full_config.yaml
```

Before running on a new machine, update local paths in the configuration file to point to the available Visium sample folder, processed-sample location, output root, and any external reference resources.

Typical local data locations are expected to be outside GitHub, for example:

```text
<path-to-project>/Visium_samples/visium_cohort_clean
<path-to-project>/Visium_samples/processed_samples
<path-to-project>/spatial_feature_identification_pipeline/outputs
```

## Quick check

From the pipeline folder, confirm that the runner and configuration are visible:

```powershell
cd "<path-to-project>/spatial_feature_identification_pipeline"

python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml --dry-run
```

The dry run prints the commands that would execute Steps 01 through 12 without regenerating outputs.

Expected step sequence:

```text
01_validate_inputs.py
02_process_samples.py
03_merge_slide_features.py
04_score_and_label_slides.py
05_build_multi_axis_transcriptome_labels.py
06_build_accessibility_profiles.py
07_append_hotspot_metrics.py
08_add_context_alignment.py
09_build_motif_tables.py
10_build_model_ready_table.py
11_overlay.py
12_data_analysis_and_visuals.py
```

## Running the full pipeline

From the pipeline folder:

```powershell
cd "<path-to-project>/spatial_feature_identification_pipeline"

python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml
```

A full run can overwrite or regenerate output folders. Archive important local outputs before major reruns.

After a run, the audit utility can be used to inspect generated outputs:

```powershell
python .\tools\audit_pipeline_outputs.py --pipeline-root . --open
```

Audit reports are generated locally and should not be committed unless intentionally curated as small documentation artifacts.

## Running individual steps

Individual scripts can be run directly for debugging, partial regeneration, or controlled reruns.

```powershell
cd "<path-to-project>/spatial_feature_identification_pipeline"
```

Step 01 validates the input cohort:

```powershell
python .\code\01_validate_inputs.py --config .\configs\visium_cohort_clean.yaml
```

Step 02 processes individual samples:

```powershell
python .\code\02_process_samples.py --config .\configs\visium_cohort_clean.yaml
```

Step 03 merges per-sample slide features:

```powershell
python .\code\03_merge_slide_features.py --config .\configs\visium_cohort_clean.yaml
```

Step 04 scores and labels slide-level programs:

```powershell
python .\code\04_score_and_label_slides.py --config .\configs\visium_cohort_clean.yaml
```

Step 05 builds multi-axis transcriptome labels:

```powershell
python .\code\05_build_multi_axis_transcriptome_labels.py --config .\configs\visium_cohort_clean.yaml
```

Step 06 builds tumor accessibility profiles:

```powershell
python .\code\06_build_accessibility_profiles.py --config .\configs\visium_cohort_clean.yaml
```

Step 07 appends hotspot metrics:

```powershell
python .\code\07_append_hotspot_metrics.py --config .\configs\visium_cohort_clean.yaml
```

Step 08 adds context alignment and metabolic concordance:

```powershell
python .\code\08_add_context_alignment.py --config .\configs\visium_cohort_clean.yaml
```

Step 09 builds spatial motif tables:

```powershell
python .\code\09_build_motif_tables.py --config .\configs\visium_cohort_clean.yaml
```

Step 10 builds the model-ready numeric table:

```powershell
python .\code\10_build_model_ready_table.py --config .\configs\visium_cohort_clean.yaml
```

Step 11 builds overlay review outputs:

```powershell
python .\code\11_overlay.py --config .\configs\visium_cohort_clean.yaml
```

Step 12 builds cohort analysis and visualization outputs:

```powershell
python .\code\12_data_analysis_and_visuals.py --config .\configs\visium_cohort_clean.yaml
```

Step 13 scripts are external-validation scripts and are run separately after the required reference assets are available:

```powershell
python .\code\13a_external_study_validation.py --help
python .\code\13b_pdac_curated_non_image_audit.py --help
python .\code\13c_pdac_collect_and_pair_validation_figures.py --help
```

## Step summary

### Step 01: Input validation

Script:

```text
code/01_validate_inputs.py
```

Validates the raw or cleaned Visium cohort folder before feature generation. It writes an input validation report and summary.

### Step 02: Sample processing

Script:

```text
code/02_process_samples.py
```

Loads each Visium sample, performs per-sample processing, writes processed sample files to the configured processed-sample location, and generates slide-level feature rows and cluster summaries. Large h5ad outputs are local-only and not tracked in Git.

### Step 03: Slide feature merge

Script:

```text
code/03_merge_slide_features.py
```

Merges per-sample slide feature rows into a cohort-level table.

### Step 04: Slide program scoring

Script:

```text
code/04_score_and_label_slides.py
```

Scores slide-level programs and assigns initial slide labels.

### Step 05: Multi-axis transcriptome labels

Script:

```text
code/05_build_multi_axis_transcriptome_labels.py
```

Builds the canonical structural, functional, metabolic, and transcriptomic label layer. This step can be computationally heavy because it produces per-sample spot-level labels and scores.

### Step 06: Accessibility profiles

Script:

```text
code/06_build_accessibility_profiles.py
```

Quantifies tumor boundary, tumor core, accessibility, and barrier-related spatial profiles.

### Step 07: Hotspot metrics

Script:

```text
code/07_append_hotspot_metrics.py
```

Builds hotspot masks and appends hotspot fraction, fragmentation, and connected-component metrics.

### Step 08: Context alignment and metabolic concordance

Script:

```text
code/08_add_context_alignment.py
```

Adds metabolic module, context module, and concordance features.

### Step 09: Motif tables

Script:

```text
code/09_build_motif_tables.py
```

Generates motif-level tables, pairwise spatial relationship tables, and gradient tables.

### Step 10: Model-ready table

Script:

```text
code/10_build_model_ready_table.py
```

Converts engineered spatial features into a numeric model-ready matrix and feature manifest. This is the main compact handoff for downstream machine learning.

### Step 11: Overlays

Script:

```text
code/11_overlay.py
```

Creates interactive overlay review outputs for spatial interpretation.

### Step 12: Data analysis and visuals

Script:

```text
code/12_data_analysis_and_visuals.py
```

Generates cohort-level analysis figures, feature summaries, heatmaps, motif plots, accessibility plots, hotspot plots, metabolism plots, and selected review packets.

### Step 13: External validation

Scripts:

```text
code/13a_external_study_validation.py
code/13b_pdac_curated_non_image_audit.py
code/13c_pdac_collect_and_pair_validation_figures.py
```

Step 13 is an external validation layer used to compare pipeline outputs with independent spatial transcriptomics studies. It is not part of the main Steps 01 through 12 runner and does not change the model-ready feature table.

## Output policy

Canonical outputs generated locally include:

```text
outputs/output_01_validate_inputs
outputs/output_02_process_samples_reports
outputs/output_03_merge_slide_features
outputs/output_04_score_and_label_slides
outputs/output_05_build_multi_axis_transcriptome_labels
outputs/output_06_build_accessibility_profiles
outputs/output_07_append_hotspot_metrics
outputs/output_08_context_alignment_and_metabolic_concordance
outputs/output_09_build_motif_tables
outputs/output_10_build_model_ready_table
outputs/output_11_overlay
outputs/output_12_data_analysis_and_visuals
outputs/output_13_external_study_validation
outputs/_external_study_validation
```

These folders are generated locally and excluded from GitHub. The important compact handoff files are:

```text
outputs/output_10_build_model_ready_table/model_input_numeric.csv
outputs/output_10_build_model_ready_table/feature_manifest.csv
```

Large processed h5ad files, raw sample folders, generated figures, and external validation assets should be handled through a separate data-release or local-storage strategy rather than Git.

## Reproducibility checklist

Before using a run downstream, verify:

```text
1. Input validation completed successfully.
2. Step 02 processed the expected number of samples.
3. Steps 03 through 11 have the expected sample count in the main tables.
4. Step 10 model_input_numeric.csv exists.
5. Step 10 feature_manifest.csv exists.
6. Step 12 visualization outputs exist if figures are needed.
7. Step 13 source assets and validation outputs exist if external validation is discussed.
8. The final output audit has no unexpected failures.
```

For the frozen local 102-sample analysis, the expected model handoff state was:

```text
model_input_numeric.csv: 102 rows
feature_manifest.csv: 720 rows
```

## Data availability

This repository does not include raw Visium data, processed h5ad intermediates, generated output folders, validation figures, or external publication assets. To reproduce the analysis, provide equivalent local data, update the configuration paths, run the pipeline, and regenerate outputs locally.

## Troubleshooting

### Step 02 internal data folder is missing

Large Step 02 processed h5ad files may be configured to live outside the pipeline folder. This is intentional to avoid storing large h5ad intermediates in the source repository.

### Old output names appear in local audit files

Local audit reports or file-tree snapshots can contain historical paths from earlier development. The active runner, active config, and current README should be treated as the source-facing reference.

### Python creates `__pycache__`

That is normal. These folders are ignored by Git and can be removed safely.

```powershell
Get-ChildItem ".\code", ".\tools" -Recurse -Directory -Force |
    Where-Object { $_.Name -eq "__pycache__" } |
    Remove-Item -Recurse -Force
```

## Notes for reviewers

Start with:

```powershell
cd "<path-to-project>/spatial_feature_identification_pipeline"
python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml --dry-run
```

Then inspect the numbered scripts in `code/` and the configuration file in `configs/`. Generated outputs are not included in the repository and should be regenerated locally when needed.
