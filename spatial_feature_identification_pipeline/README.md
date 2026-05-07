# Spatial Feature Identification Pipeline

This repository contains a documented spatial transcriptomics feature engineering pipeline for 10x Visium cohorts. The pipeline converts per sample Visium data into biologically interpretable spatial feature tables, region labels, spatial motif summaries, overlay review outputs, and external validation figures.

The current active analysis is the 102 usable sample Visium cohort. The input cohort contains 103 sample folders. `SAMPLE_0049` is the one skipped sample because it has `01_loaded.h5ad` but no `02_processed.h5ad`.

## 1. Current frozen status

Current verified state after the repository cleanup and final audit:

```text
Input sample folders: 103
Usable downstream samples: 102
Skipped sample: SAMPLE_0049

Step 02 external processed sample folders: 103
Step 02 loaded h5ad files: 103
Step 02 processed h5ad files: 102
Step 02 slide feature rows: 102
Step 02 cluster summaries: 102

Step 03 through Step 11 main tables: 102 sample rows
Step 10 model table: 102 sample rows
Step 10 feature manifest: 720 features
Step 12 visualization and report files: 1225
Step 13 generated validation files: 358
Step 13 source asset files: 243

Reactome max terms: 75
Reactome terms loaded: 75
Hallmark terms loaded: 11
External library status: ok
External library source: msigdb_2023.1.Hs
```

The main downstream handoff files are:

```text
outputs/output_10_build_model_ready_table/model_input_numeric.csv
outputs/output_10_build_model_ready_table/feature_manifest.csv
```

These two files are the main compact feature handoff for downstream teacher building, modeling, and cross pipeline reuse.

## 2. What this pipeline does

The pipeline builds a multi layer description of spatial transcriptomics samples.

It performs input validation, per sample Visium processing, slide level feature merging, program scoring, multi axis transcriptome labeling, accessibility profiling, hotspot extraction, metabolic and context alignment, spatial motif table construction, model ready feature table construction, overlay visualization, cohort level figure generation, and external study validation.

The pipeline is designed for scientific feature engineering rather than direct clinical prediction. It produces interpretable spatial descriptors that can be reused by downstream modeling pipelines.

## 3. Repository layout

```text
spatial_feature_identification_pipeline/
    README.md
    run_pipeline.py
    configs/
    code/
    tools/
    docs/
    outputs/
    _repo_local_archive/
    _LIVE_.txt
```

`run_pipeline.py` is the main runner for Steps 01 through 12.

`configs/` contains active and reference YAML configuration files.

`code/` contains active numbered pipeline scripts and the reusable internal library module.

`tools/` contains maintenance and audit utilities.

`docs/` contains runbooks, provenance reports, migration logs, and audit reports.

`outputs/` contains canonical generated pipeline outputs and external validation material.

`_repo_local_archive/` contains local backups, deprecated runs, old dry runs, documentation polish artifacts, and archived exploratory material. It is not part of the active scientific result.

`_LIVE_.txt` is an auto updating live file tree snapshot. It is intentionally left in place even when it contains older historical strings.

## 4. Important data locations

The current active pipeline root is:

```text
D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline
```

The current input cohort is:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\visium_cohort_clean
```

The current external Step 02 processed sample store is:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\processed_samples
```

The canonical output root is:

```text
D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline\outputs
```

The active configuration is:

```text
configs/visium_cohort_clean.yaml
```

The full reference configuration is:

```text
configs/spatial_feature_pipeline_full_config.yaml
```

The full reference configuration is useful for documentation, but may contain historical path names from earlier cleanup stages. Use the active configuration and the current README for actual run instructions.

## 5. Environment setup

Use the project virtual environment that already exists:

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"
```

Check that Python runs:

```powershell
& $Python --version
```

The active environment should include the scientific Python packages used by the scripts. The scripts rely on common spatial transcriptomics and data science libraries such as pandas, numpy, scanpy, anndata, scipy, sklearn, matplotlib, yaml handling, and image or plotting utilities depending on the step.

For reproducibility, use the existing project environment unless a new requirements file is created and validated.

### 5.1 Optional environment export

To document the current environment for GitHub or review:

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python -m pip freeze > .\docs\provenance\environment_freeze_current.txt
```

A future public release should include either `requirements.txt` or `environment.yml` after the environment has been tested from a clean installation.

## 6. Quick verification before running anything

From the pipeline root, run:

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\tools\audit_pipeline_outputs.py --pipeline-root . --open
```

Expected high level result:

```text
01 exists True, 103 rows
02_reports exists True, 103 rows, OK 102, SKIPPED 1
02_data_external exists True, 103 samples, processed_h5ad 102
03 through 11 exist True, 102 sample rows
10_manifest exists True, 720 rows
12 exists True, 1225 files
13_generated exists True, nonzero file count
13_source_assets exists True, nonzero file count
```

This audit writes reports under:

```text
docs/provenance/output_audits
```

Do not place new audit reports directly in the root of `outputs/`.

## 7. Dry run the pipeline

The dry run prints the commands that would execute Steps 01 through 12.

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml --dry-run
```

Expected dry run selected steps:

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

## 8. Run the full active pipeline

To run the full active pipeline:

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml
```

After the run finishes, immediately run:

```powershell
& $Python .\tools\audit_pipeline_outputs.py --pipeline-root . --open
```

A full rerun can overwrite or regenerate output folders. Commit, copy, or archive important outputs before major reruns.

## 9. Run individual steps manually

Individual steps can be run directly. This is useful for debugging, partial regeneration, or rerunning one downstream layer after a controlled change.

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"
```

Step 01 validates the input cohort:

```powershell
& $Python .\code\01_validate_inputs.py --config .\configs\visium_cohort_clean.yaml
```

Step 02 processes individual samples and writes reports:

```powershell
& $Python .\code\02_process_samples.py --config .\configs\visium_cohort_clean.yaml
```

Step 03 merges per sample slide features:

```powershell
& $Python .\code\03_merge_slide_features.py --config .\configs\visium_cohort_clean.yaml
```

Step 04 scores and labels slide level programs:

```powershell
& $Python .\code\04_score_and_label_slides.py --config .\configs\visium_cohort_clean.yaml
```

Step 05 builds multi axis transcriptome labels:

```powershell
& $Python .\code\05_build_multi_axis_transcriptome_labels.py --config .\configs\visium_cohort_clean.yaml
```

Step 06 builds tumor accessibility profiles:

```powershell
& $Python .\code\06_build_accessibility_profiles.py --config .\configs\visium_cohort_clean.yaml
```

Step 07 appends hotspot metrics:

```powershell
& $Python .\code\07_append_hotspot_metrics.py --config .\configs\visium_cohort_clean.yaml
```

Step 08 adds context alignment and metabolic concordance:

```powershell
& $Python .\code\08_add_context_alignment.py --config .\configs\visium_cohort_clean.yaml
```

Step 09 builds spatial motif tables:

```powershell
& $Python .\code\09_build_motif_tables.py --config .\configs\visium_cohort_clean.yaml
```

Step 10 builds the model ready numeric table:

```powershell
& $Python .\code\10_build_model_ready_table.py --config .\configs\visium_cohort_clean.yaml
```

Step 11 builds overlay review outputs:

```powershell
& $Python .\code\11_overlay.py --config .\configs\visium_cohort_clean.yaml
```

Step 12 builds data analysis and visualization outputs:

```powershell
& $Python .\code\12_data_analysis_and_visuals.py --config .\configs\visium_cohort_clean.yaml
```

Step 13 scripts are external validation scripts and are run separately:

```powershell
& $Python .\code\13a_external_study_validation.py --help
& $Python .\code\13b_pdac_curated_non_image_audit.py --help
& $Python .\code\13c_pdac_collect_and_pair_validation_figures.py --help
```

Run Step 13 scripts only after confirming the required source assets exist under `outputs/_external_study_validation`.

## 10. Step by step explanation

### 10.1 Step 01 input validation

Script:

```text
code/01_validate_inputs.py
```

Purpose:

```text
Validate the raw or cleaned Visium cohort folder before feature generation.
```

Input:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\visium_cohort_clean
```

Main output folder:

```text
outputs/output_01_validate_inputs
```

Important outputs:

```text
input_validation_report.csv
input_validation_summary.txt
```

Current expected result:

```text
103 input sample folders
103 OK in input validation
```

### 10.2 Step 02 sample processing

Script:

```text
code/02_process_samples.py
```

Purpose:

```text
Load each Visium sample, perform per sample processing, generate processed h5ad files, cluster summaries, and slide level feature rows.
```

Main report output folder:

```text
outputs/output_02_process_samples_reports
```

Large processed data location:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\processed_samples
```

Important report output:

```text
processing_report.csv
```

Important per sample external outputs:

```text
SAMPLE_<id>/adata/01_loaded.h5ad
SAMPLE_<id>/adata/02_processed.h5ad
SAMPLE_<id>/tables/slide_level_feature_row.csv
SAMPLE_<id>/tables/cluster_summary.csv
```

Current expected result:

```text
103 sample folders
103 loaded h5ad files
102 processed h5ad files
102 slide level feature rows
102 cluster summaries
SAMPLE_0049 skipped
```

### 10.3 Step 03 slide feature merge

Script:

```text
code/03_merge_slide_features.py
```

Purpose:

```text
Merge per sample slide feature rows into a cohort level table.
```

Main output folder:

```text
outputs/output_03_merge_slide_features
```

Important outputs:

```text
merged_slide_features.csv
merge_report.csv
```

Current expected result:

```text
102 sample rows
```

### 10.4 Step 04 slide program scoring

Script:

```text
code/04_score_and_label_slides.py
```

Purpose:

```text
Score slide level feature programs and assign initial slide labels.
```

Main output folder:

```text
outputs/output_04_score_and_label_slides
```

Important outputs:

```text
slide_features_scored_labeled.csv
label_counts.csv
```

Current expected result:

```text
102 sample rows
```

### 10.5 Step 05 multi axis transcriptome labels

Script:

```text
code/05_build_multi_axis_transcriptome_labels.py
```

Purpose:

```text
Build the canonical structure, function, metabolism, and transcriptome label layer.
```

Main output folder:

```text
outputs/output_05_build_multi_axis_transcriptome_labels
```

Important outputs:

```text
multi_axis_label_status.csv
multi_axis_slide_summary.csv
slide_features_with_multi_axis_labels.csv
multi_axis_label_metadata.json
per_sample/<sample_id>_spot_labels.csv
per_sample/<sample_id>_spot_scores.csv
per_sample_h5ad/<sample_id>_with_multi_axis_transcriptome_labels.h5ad
per_sample_status/<sample_id>_status.csv
```

Current expected result:

```text
102 samples
Reactome max terms 75
Reactome terms loaded 75
Hallmark terms loaded 11
```

Notes:

Step 05 is computationally heavy because it generates per sample spot level labels and scores. It is the main biological labeling layer used by later spatial metrics.

### 10.6 Step 06 accessibility profiles

Script:

```text
code/06_build_accessibility_profiles.py
```

Purpose:

```text
Quantify tumor boundary, tumor core, accessibility, and barrier related spatial profiles.
```

Main output folder:

```text
outputs/output_06_build_accessibility_profiles
```

Important outputs:

```text
slide_accessibility_profiles.csv
slide_features_with_accessibility.csv
per_sample/<sample_id>_accessibility_spot_profile.csv
```

Current expected result:

```text
102 sample rows
```

### 10.7 Step 07 hotspot metrics

Script:

```text
code/07_append_hotspot_metrics.py
```

Purpose:

```text
Build hotspot masks and append hotspot fraction, fragmentation, and connected component metrics.
```

Main output folder:

```text
outputs/output_07_append_hotspot_metrics
```

Important outputs:

```text
hotspot_slide_summary.csv
slide_features_with_hotspot_metrics.csv
per_sample/<sample_id>_hotspot_spot_masks.csv
```

Current expected result:

```text
102 sample rows
```

### 10.8 Step 08 context alignment and metabolic concordance

Script:

```text
code/08_add_context_alignment.py
```

Purpose:

```text
Add biologically interpretable metabolic module, context module, and concordance features.
```

Main output folder:

```text
outputs/output_08_context_alignment_and_metabolic_concordance
```

Important outputs:

```text
slide_features_with_metabolic_concordance.csv
metabolic_concordance_summary.csv
metabolic_concordance_summary.txt
metabolic_concordance_status.csv
```

Current expected result:

```text
102 sample rows
```

### 10.9 Step 09 motif tables

Script:

```text
code/09_build_motif_tables.py
```

Purpose:

```text
Generate motif level tables, pairwise spatial relationship tables, and gradient tables.
```

Main output folder:

```text
outputs/output_09_build_motif_tables
```

Important outputs:

```text
all_motif_table.csv
all_pair_table.csv
all_gradient_table.csv
slide_features_with_motif_tables.csv
slide_motif_summary.csv
per_sample/<sample_id>_motif_table.csv
per_sample/<sample_id>_pair_table.csv
per_sample/<sample_id>_gradient_table.csv
```

Current expected result:

```text
102 sample rows in the merged slide feature table
```

### 10.10 Step 10 model ready table

Script:

```text
code/10_build_model_ready_table.py
```

Purpose:

```text
Convert the engineered spatial feature tables into a numeric model ready matrix and feature manifest.
```

Main output folder:

```text
outputs/output_10_build_model_ready_table
```

Important outputs:

```text
model_input_numeric.csv
feature_manifest.csv
feature_filter_report.csv
missingness_report.csv
model_ready_summary.txt
```

Current expected result:

```text
model_input_numeric.csv has 102 rows
feature_manifest.csv has 720 rows
```

This is the main compact handoff for downstream machine learning and teacher building.

### 10.11 Step 11 overlays

Script:

```text
code/11_overlay.py
```

Purpose:

```text
Create interactive region overlay review outputs for spatial interpretation.
```

Main output folder:

```text
outputs/output_11_overlay
```

Important outputs:

```text
overlay_status.csv
SAMPLE_<id>/combined_region_overlay.html
SAMPLE_<id>/fine_structure_region_overlay.html
SAMPLE_<id>/functional_region_overlay.html
SAMPLE_<id>/metabolic_region_overlay.html
```

Current expected result:

```text
102 samples
overlay status ok 102
```

### 10.12 Step 12 data analysis and visuals

Script:

```text
code/12_data_analysis_and_visuals.py
```

Purpose:

```text
Generate cohort level analysis figures, contact sheets, feature summaries, heatmaps, motif plots, accessibility plots, hotspot plots, metabolism plots, feature manifest plots, and selected top figure packets.
```

Main output folder:

```text
outputs/output_12_data_analysis_and_visuals
```

Important output groups:

```text
00_summary
01_cohort_overview
02_feature_space
03_feature_heatmaps
04_programs
05_motifs
06_pairwise_relationships
07_gradients
08_accessibility
09_hotspots
10_metabolism
11_feature_manifest
12_visium_region_overlays
```

Current expected result:

```text
1225 visualization and report files
```

### 10.13 Step 13 external validation

Step 13 is not part of the main Steps 01 through 12 runner. It is a validation layer used to compare this pipeline's outputs to external spatial transcriptomics studies.

Active Step 13 scripts:

```text
code/13a_external_study_validation.py
code/13b_pdac_curated_non_image_audit.py
code/13c_pdac_collect_and_pair_validation_figures.py
```

Source and reference assets:

```text
outputs/_external_study_validation
```

Generated validation outputs:

```text
outputs/output_13_external_study_validation
```

Current expected result:

```text
243 source asset files
358 generated validation files
```

Step 13a performs general external study validation and creates validation maps, study PNG use folders, side by side pairs, marker gene summaries, manual review manifests, metadata, and study summaries.

Step 13b performs curated PDAC non image validation using a curated validation marker dictionary. It writes marker dictionaries, marker coverage tables, enrichment summaries, pass rate summaries, model input subsets, feature manifest copies, relevant feature subsets, and curated audit summaries.

Step 13c collects and pairs PDAC validation figures. It writes pipeline candidate figures, review packets, sample name checks, and meaningful validation pairs.

Step 13 validation outputs do not change the main model input table.

## 11. Output folder policy

Canonical active outputs are:

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

Do not restore these historical folders as active outputs:

```text
outputs/output_02_01_process_samples_data
outputs/output_02_02_process_samples_reports
outputs/output_08_01_context_alignment_and_metabolic_concordance
outputs/output_13a_external_study_validation
```

The historical Step 02 internal data folder is not active because Step 02 processed sample data are stored externally under `Visium_samples/processed_samples`.

## 12. Source asset policy for Step 13

Keep source assets here:

```text
outputs/_external_study_validation
```

This folder should contain study source data, supplementary tables, extracted publication figure panels, publication grade study PNGs, reference files, and sample mapping.

Keep generated outputs here:

```text
outputs/output_13_external_study_validation
```

This folder should contain validation maps, audits, paired figures, review packets, manual review manifests, and validation reports.

Old exploratory hepatoblastoma manual comparison packets were archived under `_repo_local_archive` during the cleanup pass. Do not move them back into active source assets unless a new validation run explicitly needs them.

## 13. Provenance and logs

The canonical human readable migration log is:

```text
docs/provenance/MIGRATION_LOG_20260507_repository_cleanup.txt
```

Final snapshot audits are stored under:

```text
docs/provenance/final_snapshot_audits
```

Output audits are stored under:

```text
docs/provenance/output_audits
```

Deletion candidate audits are stored under:

```text
docs/provenance/deletion_candidate_audits
```

External validation asset sorting reports are stored under:

```text
docs/provenance/external_validation_asset_sorting
```

Repository cleanup and polish reports are stored under:

```text
docs/provenance/final_repo_polish_reports
```

Do not delete provenance reports until the project has been committed, reviewed, and backed up.

## 14. GitHub guidance

The repository should track source code, configuration files, documentation, provenance summaries, and small essential outputs as appropriate.

The repository should not track large local archives, deprecated run folders, pycache folders, or large h5ad data unless they are intentionally handled through a data release or large file system.

Likely not for GitHub commit:

```text
_repo_local_archive
__pycache__
*.pyc
large h5ad files
large local deprecated runs
```

Potentially suitable for GitHub commit:

```text
README.md
run_pipeline.py
code/
configs/
tools/
docs/
selected small provenance reports
selected small output summaries
```

Use `.gitignore` to prevent accidental commits of large generated or archived data.

### 14.1 Data availability and large file policy

This repository does not bundle the large Visium h5ad intermediates or raw sample folders. The active processed sample store is external:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\processed_samples
```

For reuse on another machine, update the config paths and provide equivalent input data and processed sample folders. The compact downstream feature handoff files are much smaller and are located under:

```text
outputs/output_10_build_model_ready_table
```

Large local archives and deprecated runs should remain outside version control unless a separate data release strategy is used.

## 15. Common workflows

### 15.1 Verify current outputs

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\tools\audit_pipeline_outputs.py --pipeline-root . --open
```

### 15.2 Dry run the active Steps 01 through 12

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml --dry-run
```

### 15.3 Run the active Steps 01 through 12

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml
```

### 15.4 Run with a permanent terminal log

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force ".\outputs\_run_logs" | Out-Null

$Log = ".\outputs\_run_logs\spatial_feature_pipeline_$(Get-Date -Format yyyyMMdd_HHmmss).log"

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml 2>&1 | Tee-Object -FilePath $Log
```

### 15.5 Run audit after a full run

```powershell
& $Python .\tools\audit_pipeline_outputs.py --pipeline-root . --open
```

### 15.6 Check the model handoff files

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python -c "import pandas as pd; from pathlib import Path; paths=[Path('outputs/output_10_build_model_ready_table/model_input_numeric.csv'), Path('outputs/output_10_build_model_ready_table/feature_manifest.csv')]; [print(str(p), pd.read_csv(p).shape) for p in paths]"
```

Expected current shapes:

```text
model_input_numeric.csv: 102 rows
feature_manifest.csv: 720 rows
```

## 16. Known caveats

`SAMPLE_0049` is the one skipped sample. It has loaded data but no processed h5ad output.

Step 02 processed h5ad data are external to this repository tree. This is intentional to avoid placing large per sample h5ad files in the source repository.

The full reference config may retain historical names for earlier output folders. The canonical current names are listed in this README and verified by the final output audits.

`_LIVE_.txt` auto updates and may contain historical strings because it records a file tree snapshot. Do not treat its old references as active code references.

`_repo_local_archive` contains old backups and deprecated runs. It is preserved for local recovery and provenance, not for active execution.

## 17. Troubleshooting

### 17.1 Audit says Step 02 internal data folder is missing

This is expected if the audit refers to the historical internal folder:

```text
outputs/output_02_01_process_samples_data
```

Use the external Step 02 processed sample location instead:

```text
D:\Adv_Omics_Fenyo\project\Visium_samples\processed_samples
```

### 17.2 Audit says old names appear in `_LIVE_.txt`

Ignore old path references in `_LIVE_.txt`. It is an auto updating file tree snapshot and may include historical strings.

### 17.3 Dry run shows old output names

Check the active `run_pipeline.py`, active step scripts, and `configs/visium_cohort_clean.yaml`. Do not rely on archived copies under `_repo_local_archive`.

### 17.4 A command points to `C:\Users\Owner\OneDrive`

That is a historical path from earlier development. Replace it with the current root:

```text
D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline
```

### 17.5 Python creates `__pycache__`

That is normal. It can be deleted safely.

```powershell
Get-ChildItem ".\code", ".\tools" -Recurse -Directory -Force |
    Where-Object { $_.Name -eq "__pycache__" } |
    Remove-Item -Recurse -Force
```

## 18. Minimal reproducibility checklist

Before claiming a run is valid, verify:

```text
1. Input validation reports 103 input samples.
2. Step 02 reports 102 OK and 1 skipped.
3. External Step 02 processed samples contain 102 processed h5ad files.
4. Steps 03 through 11 have 102 sample rows in the main tables.
5. Step 10 model_input_numeric.csv exists and has 102 rows.
6. Step 10 feature_manifest.csv exists and has 720 rows.
7. Step 12 visualization output exists if figures are needed.
8. Step 13 source assets and generated outputs exist if external validation is discussed.
9. Reactome max terms are 75 for the current frozen analysis.
10. The final output audit has no unexpected failures.
```

## 19. Main scientific outputs

The most important compact outputs for downstream use are:

```text
outputs/output_10_build_model_ready_table/model_input_numeric.csv
outputs/output_10_build_model_ready_table/feature_manifest.csv
```

The most important interpretability outputs are:

```text
outputs/output_05_build_multi_axis_transcriptome_labels
outputs/output_06_build_accessibility_profiles
outputs/output_07_append_hotspot_metrics
outputs/output_08_context_alignment_and_metabolic_concordance
outputs/output_09_build_motif_tables
outputs/output_11_overlay
outputs/output_12_data_analysis_and_visuals
```

The most important validation outputs are:

```text
outputs/output_13_external_study_validation
outputs/_external_study_validation
```

## 20. Recommended first commands for a new reviewer

```powershell
cd "D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline"

$Python = "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

& $Python .\tools\audit_pipeline_outputs.py --pipeline-root . --open

& $Python .\run_pipeline.py --config .\configs\visium_cohort_clean.yaml --dry-run

notepad .\docs\provenance\MIGRATION_LOG_20260507_repository_cleanup.txt
```

These commands let a reviewer see the current output state, inspect the runnable command sequence, and read the repository migration history.