# histology_response_model_v2

`histology_response_model_v2` trains, audits, and exports a treatment-conditioned H&E whole-slide image response teacher for the Adv_Omics_Fenyo spatial treatment-response project.

This module links named-treatment clinical response labels to local SVS whole-slide images, tiles slides, filters artifacts, creates leakage-safe patient splits, trains histology response baselines, runs blank/noise controls, and writes an audited model index for downstream `teacher_builder` use.

## Quick start

From a new PowerShell session on the workstation where the project lives:

    cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\model_training\histology_response_model_v2"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\run_histology_response_model_v2.ps1 -StartAt 0 -StopAt 9

To resume from model training after upstream tables already exist:

    .\run_histology_response_model_v2.ps1 -StartAt 7 -StopAt 9

To run one step at a time:

    .\run_histology_response_model_v2.ps1 -StartAt 4 -StopAt 4

The runner defaults to:

    D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe

If that virtual environment is unavailable, the runner falls back to `python` on the active `PATH`.

## Reproducibility and configuration rules

For normal use, change paths, thresholds, tiling settings, split settings, training options, and audit thresholds in:

    configs/histology_response_model_v2.yaml

Do not edit files in `scripts/` unless you are intentionally changing pipeline logic. The Python scripts are the active source implementation; the YAML config is the normal user-editable control surface.

Important rules:

- edit YAML values when changing locations, thresholds, tiling settings, or training settings;
- do not edit numbered scripts for routine reruns;
- do not overwrite canonical outputs unless the rerun is intentional;
- run smoke tests or scratch reruns before changing active outputs;
- keep large outputs local or in external archival storage rather than committing them directly to GitHub.

## Scientific role

This module is an upstream base-teacher workflow. It is not the final spatial prediction model. Its job is to test whether H&E image morphology adds treatment-response information beyond treatment identity alone and to export a conservative, auditable teacher candidate for downstream fusion.

The central control is the `treatment_only` baseline. The `image_treatment` model is only useful as a teacher if it improves on treatment identity under patient-level validation. Blank/noise controls are included because non-informative images can reveal treatment-driven or poorly calibrated output behavior.

Conceptually:

    clinical treatment and response labels
            +
    local H&E SVS whole-slide images
            ↓
    slide manifest and tissue tiles
            ↓
    patient-level split and artifact-filtered tile table
            ↓
    treatment_only, image_only, and image_treatment models
            ↓
    blank/noise controls and audit gates
            ↓
    histology_model_index.tsv for teacher_builder

## What changed from histology_response_model v1

This folder supersedes the older `histology_response_model` workflow. The older folder is retained only as provenance. Version 2 is organized as a clearer, auditable, treatment-conditioned teacher workflow.

Main improvements in v2:

- separates clinical labeling, slide manifest construction, tiling, training-table construction, patient splitting, model training, control inference, and audit into numbered steps;
- uses named-treatment labels and canonical treatment keys rather than ambiguous generic therapy labels;
- links local SVS files to labeled cases through an explicit slide manifest;
- applies artifact QC before training and keeps excluded artifact rows as audit outputs;
- splits at the patient level to prevent leakage across train, validation, and test sets;
- trains `treatment_only`, `image_only`, and `image_treatment` model families so image signal can be compared against treatment identity;
- runs blank/noise image controls after training;
- writes an audited model index for `teacher_builder` instead of leaving teacher use implicit;
- documents outputs, assumptions, and validation checks for GitHub, publication review, and reuse.

## Repository layout

The active workflow is organized around these files:

| Type | File | Purpose |
| --- | --- | --- |
| Config | `configs/histology_response_model_v2.yaml` | Main user-editable configuration file. |
| Runner | `run_histology_response_model_v2.ps1` | Executes the numbered workflow steps. |
| Shared library | `scripts/histology_model_v2_lib.py` | Shared config, path, treatment, response, I/O, and metric helpers. |
| Step 00 | `scripts/00_build_treatment_ontology.py` | Builds named-treatment ontology outputs. |
| Step 01 | `scripts/01_validate_inputs.py` | Checks configured clinical, slide, and tile inputs. |
| Step 02 | `scripts/02_build_case_label_table.py` | Builds strict case-level treatment-response labels. |
| Step 03 | `scripts/03_build_slide_manifest.py` | Links labeled cases to local SVS slides. |
| Step 04 | `scripts/04_tile_slides.py` | Tiles whole-slide images and writes tile manifests. |
| Step 05 | `scripts/05_build_tile_training_table.py` | Builds artifact-filtered tile and patient training tables. |
| Step 06 | `scripts/06_build_patient_split.py` | Creates leakage-safe patient-level train/val/test splits. |
| Step 07 | `scripts/07_train_baselines_and_conditioned_model.py` | Trains treatment_only, image_only, and image_treatment models. |
| Step 08 | `scripts/08_run_control_inference.py` | Runs blank/noise image controls. |
| Step 09 | `scripts/09_audit_histology_model.py` | Audits model suitability and writes the teacher handoff index. |

Development backups, temporary dry-run helpers, and local provenance are not part of the active workflow and should live outside the active source path, usually under `local_archive/` or `docs/`.

## Pipeline steps and outputs

### Step 00: treatment ontology

Scans clinical treatment columns, removes generic therapy categories, harmonizes named agents and regimens, and writes the treatment ontology.

Primary outputs:

    outputs/histology_v2/00_treatment_ontology/treatment_ontology.tsv
    outputs/histology_v2/00_treatment_ontology/treatment_ontology_summary.txt

### Step 01: input validation

Checks configured clinical, slide, and tile paths. This is a preflight audit step, not a modeling step.

Primary outputs:

    outputs/histology_v2/01_input_audit/input_path_report.tsv
    outputs/histology_v2/01_input_audit/input_audit_summary.txt

### Step 02: case label table

Collapses clinical data to case-level records, maps raw responses to `RESPONDER` or `NON_RESPONDER`, canonicalizes treatment keys, and writes full and strict label tables.

Primary outputs:

    outputs/histology_v2/02_case_labels/case_label_table.tsv
    outputs/histology_v2/02_case_labels/case_label_table_strict.tsv
    outputs/histology_v2/02_case_labels/case_label_summary.txt

### Step 03: slide manifest

Links labeled patients/cases to local SVS files and writes the manifest used by tiling. This is the bridge between clinical labels and image data.

Primary outputs:

    outputs/histology_v2/03_slide_manifest/slide_manifest.tsv
    outputs/histology_v2/03_slide_manifest/slide_manifest_with_labels.tsv
    outputs/histology_v2/03_slide_manifest/slide_manifest_summary.txt

### Step 04: tile slides

Tiles SVS images using configured tile size, pyramid level, stride, tissue fraction threshold, and optional parallel workers.

Primary outputs:

    outputs/histology_v2/04_tiles/tile_manifest.tsv
    outputs/histology_v2/04_tiles/tile_status.tsv

### Step 05: tile training table and artifact QC

Joins tile rows to strict labels, confirms tile files exist, computes artifact metrics, removes flagged tiles, and writes model-ready tile and patient tables.

Primary outputs:

    outputs/histology_v2/05_training_table/tile_training_table.tsv
    outputs/histology_v2/05_training_table/patient_training_table.tsv
    outputs/histology_v2/05_training_table/artifact_qc/

### Step 06: patient split

Creates train/validation/test assignments at patient level, then merges those assignments back to tile rows. Patient-level splitting prevents multiple tiles or slides from one patient leaking across evaluation splits.

Primary outputs:

    outputs/histology_v2/06_patient_split/patient_split.tsv
    outputs/histology_v2/06_patient_split/tile_training_table_split.tsv
    outputs/histology_v2/06_patient_split/patient_split_summary.txt

### Step 07: model training

Trains three model families:

    treatment_only
    image_only
    image_treatment

The comparison among these models determines whether image morphology contributes beyond treatment identity.

Primary outputs:

    outputs/histology_v2/07_models/model_comparison.tsv
    outputs/histology_v2/07_models/training_run_summary.json
    outputs/histology_v2/07_models/*/best_model.pt
    outputs/histology_v2/07_models/*/metrics_by_split.tsv

### Step 08: blank/noise controls

Runs the trained `image_treatment` model on blank and random-noise control images across treatment embeddings. These controls are post-training sanity checks and are not used for model fitting.

Primary outputs:

    outputs/histology_v2/08_controls/blank_noise_control_predictions.tsv
    outputs/histology_v2/08_controls/control_summary.tsv

### Step 09: audit

Applies teacher-export criteria, computes the image-treatment AUC delta over the treatment-only baseline, bounds the reliability weight, and writes the final model index consumed by `teacher_builder`.

Primary outputs:

    outputs/histology_v2/09_audit/histology_model_index.tsv
    outputs/histology_v2/09_audit/histology_model_audit_summary.txt

## Quality-control principles

The workflow is designed around several safeguards:

1. Named treatments are harmonized and generic therapy categories are excluded.
2. Response labels are collapsed to case-level records before slide linkage.
3. Whole-slide images are split at the patient level, not tile level.
4. Artifact tiles are retained in audit files rather than silently discarded.
5. Treatment-only and image-only baselines are trained beside the conditioned model.
6. Blank/noise controls are run after training to probe non-informative image behavior.
7. Teacher approval depends on held-out performance and improvement over treatment-only baseline.

## Teacher-builder handoff

The downstream handoff file is:

    outputs/histology_v2/09_audit/histology_model_index.tsv

`teacher_builder` should use this model conservatively. Histology probabilities should be governed with reliability weighting and shrinkage toward treatment priors, especially when blank/noise controls show broad or extreme outputs.

Recommended downstream interpretation:

- use the audited model index rather than raw training outputs;
- preserve the reliability weight and approval status fields;
- treat blank/noise control behavior as evidence for conservative teacher weighting;
- do not retrain the histology model inside `teacher_builder`.

## Environment and preflight checks

Before running, confirm that the expected config and Python environment exist:

    Test-Path ".\configs\histology_response_model_v2.yaml"
    Test-Path "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe"

Check that the YAML parses:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -c "import yaml, pathlib; yaml.safe_load(pathlib.Path(r'configs/histology_response_model_v2.yaml').read_text(encoding='utf-8')); print('histology config OK')"

Compile the active Python scripts without running the pipeline:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -m py_compile scripts\histology_model_v2_lib.py scripts\00_build_treatment_ontology.py scripts\01_validate_inputs.py scripts\02_build_case_label_table.py scripts\03_build_slide_manifest.py scripts\04_tile_slides.py scripts\05_build_tile_training_table.py scripts\06_build_patient_split.py scripts\07_train_baselines_and_conditioned_model.py scripts\08_run_control_inference.py scripts\09_audit_histology_model.py

From the parent `model_training/` folder, the repository smoke test is:

    .\tests\smoke_test_model_training.ps1

## Common rerun patterns

Run only input validation:

    .\run_histology_response_model_v2.ps1 -StartAt 1 -StopAt 1

Rebuild case labels after changing clinical label rules:

    .\run_histology_response_model_v2.ps1 -StartAt 2 -StopAt 2

Rebuild slide and tile manifests after changing slide inputs or tiling settings:

    .\run_histology_response_model_v2.ps1 -StartAt 3 -StopAt 5

Retrain and audit models after model-ready tables already exist:

    .\run_histology_response_model_v2.ps1 -StartAt 7 -StopAt 9

Run the full workflow:

    .\run_histology_response_model_v2.ps1 -StartAt 0 -StopAt 9

## GitHub and publication notes

For GitHub, commit active source scripts, configs, READMEs, runbooks, tests, small manifests, and documentation. Do not commit large local outputs unless using external archival storage or Git LFS.

Large local artifacts include:

- tile manifests and tile-level training tables;
- artifact metric tables;
- trained `.pt` model checkpoints;
- tile-level prediction tables;
- local archives and development backups.

The output README under `outputs/` describes which artifacts are canonical and which are too large for ordinary Git commits.

## Review and validation notes

This module has been reviewed with non-behavioral documentation edits. Python files were checked with `py_compile` and AST-equivalence checks after ignoring docstrings. YAML comments were added without changing parsed values. PowerShell comments were added without changing executable command lines.

The repository-level smoke test and micro rerun checks are intended to verify file contracts and runtime plumbing without changing canonical outputs. A scientifically meaningful full rerun should use the full configured data and should be performed intentionally.
