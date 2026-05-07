# model_training

`model_training` contains the active upstream model-training workflows used to build response-teacher candidates for the Adv_Omics_Fenyo spatial treatment-response project.

These workflows are not the final spatial prediction model. They train and audit base teachers that can be consumed by `teacher_builder`, which then fuses expression, histology, treatment-prior, and spatial information for downstream modeling.

## Active modules

The active production modules are:

| Module | Role | Main handoff |
| --- | --- | --- |
| `expression_response_model_v2` | Trains calibrated treatment-specific expression-response models from GDC expression, treatment, and response labels. | Approved expression model artifacts and optional Visium expression-teacher scores. |
| `histology_response_model_v2` | Trains treatment-conditioned H&E whole-slide image response models and audits image contribution beyond treatment identity. | `outputs/histology_v2/09_audit/histology_model_index.tsv`. |

Deprecated earlier workflows are retained only as local provenance under `local_archive/` and are not part of the active workflow.

## Quick start

From a new PowerShell session:

    cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\model_training"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Run the repository-level smoke test without rerunning model training:

    .\tests\smoke_test_model_training.ps1

The smoke test checks active file presence, Python compilation, YAML parsing, and selected output readability. It does not retrain models and does not rewrite canonical outputs.

To run an individual module, enter that module folder and use its runner:

    cd ".\expression_response_model_v2"
    .\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4

    cd "..\histology_response_model_v2"
    .\run_histology_response_model_v2.ps1 -StartAt 0 -StopAt 9

Run the module README first before intentionally rerunning either workflow.

## Reproducibility and configuration rules

For normal use, change paths, thresholds, training options, tiling settings, and output locations in each module YAML file:

    expression_response_model_v2\configs\expression_response_model_v2.yaml
    histology_response_model_v2\configs\histology_response_model_v2.yaml

Do not edit files in `scripts/` unless you are intentionally changing pipeline logic. The `scripts/` folders are the active source-code implementation and should remain named `scripts/`.

Important rules:

- use YAML for routine configuration changes;
- do not edit active Python scripts for ordinary reruns;
- do not overwrite canonical outputs unless the rerun is intentional;
- run smoke tests or scratch rerun probes before changing active outputs;
- keep large local outputs out of ordinary Git commits unless using external archival storage or Git LFS;
- keep local provenance, old backups, and deprecated workflows under `local_archive/`.

## Repository layout

The active `model_training` layout is:

    model_training/
    ├── README.md
    ├── RUNBOOK.md
    ├── requirements-model-training.txt
    ├── .gitignore
    ├── expression_response_model_v2/
    │   ├── README.md
    │   ├── run_expression_response_model_v2.ps1
    │   ├── configs/
    │   ├── scripts/
    │   ├── outputs/
    │   └── docs/
    ├── histology_response_model_v2/
    │   ├── README.md
    │   ├── run_histology_response_model_v2.ps1
    │   ├── configs/
    │   ├── scripts/
    │   ├── outputs/
    │   └── docs/
    ├── docs/
    ├── tests/
    ├── tools/
    └── local_archive/

Folder meanings:

| Folder | Purpose | GitHub handling |
| --- | --- | --- |
| `expression_response_model_v2/` | Active expression-response teacher workflow. | Commit source, config, README, runner, and small docs. Keep large outputs local/external. |
| `histology_response_model_v2/` | Active histology-response teacher workflow. | Commit source, config, README, runner, and small docs. Keep large outputs local/external. |
| `docs/` | Repository layout notes, cleanup manifests, smoke-test reports, diagnostics, and provenance summaries. | Commit concise docs and manifests when useful. |
| `tests/` | Lightweight non-rerun smoke tests. | Commit. |
| `tools/` | Reserved for future maintenance/audit utilities. | Commit useful reusable tools only. |
| `local_archive/` | Deprecated workflows, installers, backups, dry-run artifacts, generated bundles, and local provenance. | Do not commit. |

## Module summaries

### expression_response_model_v2

This module trains deployable expression-response teacher models. It uses GDC expression profiles, named-treatment labels, and binary response labels to train treatment-specific calibrated models.

Main workflow:

    treatment ontology
        → input validation
        → canonical expression-response training table
        → deployable calibrated model training
        → model audit
        → optional Visium pseudobulk teacher scoring

Main outputs:

    expression_response_model_v2\outputs\deployable_CH1\model_index.tsv
    expression_response_model_v2\outputs\deployable_CH1\model_index_approved.tsv
    expression_response_model_v2\outputs\deployable_CH1\models\*.joblib

Read the module guide:

    expression_response_model_v2\README.md

### histology_response_model_v2

This module trains and audits treatment-conditioned H&E whole-slide image response models. It explicitly compares `treatment_only`, `image_only`, and `image_treatment` model families to test whether image morphology adds signal beyond treatment identity.

Main workflow:

    treatment ontology
        → input validation
        → case-level treatment-response labels
        → slide manifest
        → tiling
        → artifact-filtered training table
        → patient split
        → model training
        → blank/noise controls
        → audit and teacher handoff

Main output:

    histology_response_model_v2\outputs\histology_v2\09_audit\histology_model_index.tsv

Read the module guide:

    histology_response_model_v2\README.md

## Validation status

After documentation and folder reorganization, the repository-level non-rerun smoke test passed with active scripts/configs/runners and existing model artifacts still usable. A 5-sample scratch micro rerun also confirmed histology Step 07-09 plumbing and showed that expression Steps 00-03 run on scratch input, while Step 04 can warn on tiny samples because five rows are not enough to train deployable expression models.

Use these checks for review:

    .\tests\smoke_test_model_training.ps1

For a stricter end-to-end validation, run each module intentionally with full configured data and a scratch output root. Do not use a full rerun as a casual smoke test because model training can be slow and may overwrite outputs if the active YAML output paths are unchanged.

## GitHub and publication guidance

Recommended to commit:

- active module scripts;
- active YAML configs;
- module READMEs;
- top-level `README.md` and `RUNBOOK.md`;
- `requirements-model-training.txt`;
- `.gitignore`;
- `tests/` smoke tests;
- concise docs and small manifests;
- output README files.

Recommended to ignore or archive externally:

- `local_archive/`;
- Python cache files;
- installer bundles;
- trained `.joblib` and `.pt` artifacts unless using Git LFS or another deliberate model-artifact policy;
- very large TSV outputs such as canonical expression training tables, tile manifests, tile training tables, tile prediction tables, and artifact metric tables;
- generated all-code/all-data bundles.

The `.gitignore` in this folder is designed to help prevent accidental commits of local provenance and large generated outputs.

## Relationship to teacher_builder

The model-training layer should produce audited teacher candidates. It should not be collapsed into `teacher_builder` and should not be rerun implicitly inside downstream teacher fusion.

Expected handoff pattern:

1. train and audit expression and histology teachers here;
2. preserve model indexes, approval flags, calibration fields, reliability weights, and audit summaries;
3. let `teacher_builder` consume approved artifacts or scored teacher tables;
4. keep treatment-prior governance and spatial fusion downstream.

This separation makes the project easier to audit: model fitting, model approval, teacher fusion, and final spatial modeling remain distinct stages.

## Local provenance

`local_archive/` stores material that is useful for provenance but should not clutter the active repository:

- deprecated v1 workflows;
- old installers;
- patch backups;
- documentation backups;
- temporary helper scripts;
- dry-run outputs;
- generated bundles;
- output backups.

Do not delete `local_archive/` until the project is fully committed or externally archived. Do not include it in a normal GitHub commit.

## Practical review path

For a teacher, reviewer, or collaborator, the recommended reading order is:

1. this `README.md`;
2. `RUNBOOK.md`;
3. `expression_response_model_v2\README.md`;
4. `histology_response_model_v2\README.md`;
5. each module YAML config;
6. each module shared library;
7. model-training and audit scripts;
8. output READMEs and audit summaries.

## Current status

The `model_training` folder has been reorganized for GitHub readability and reproducible review. Active source remains under `scripts/`; no `scripts/` folder was renamed to `code/`. Documentation updates and folder cleanup were intended to be non-behavioral. Existing active scripts, configs, runners, and canonical output folders were preserved.
