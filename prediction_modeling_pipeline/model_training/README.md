# model_training

`model_training` contains the upstream model-training workflows used to build response-teacher candidates for the spatial omics tumor drug response project.

These workflows are not the final spatial prediction model. They train and audit base expression- and histology-derived response models that can be consumed by `teacher_builder`, which then prepares governed teacher signals for downstream spatial prediction and interpretation.

The repository is source-focused. Large training tables, trained model artifacts, generated outputs, logs, local archives, and machine-specific run products are intentionally excluded from GitHub.

## Active modules

| Module | Role | Main local handoff |
| --- | --- | --- |
| `expression_response_model_v2/` | Trains calibrated treatment-specific expression-response models from expression, treatment, and response-label data. | Approved expression model index and optional Visium expression-teacher scores. |
| `histology_response_model_v2/` | Trains treatment-conditioned H&E whole-slide image response models and audits whether image features add signal beyond treatment identity. | Histology model index and audit outputs. |

Deprecated earlier workflows should remain outside the GitHub-tracked source package, for example under a local archive folder.

## Quick start

<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_START -->
## Reviewer shortcut: precomputed teacher handoff

Running the expression-response and histology-response model-training workflows can require large external datasets and substantial runtime. For reviewer convenience, the repository includes a compact precomputed fused teacher handoff in `teacher_builder`:

```text
../teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz
```

If this file is used, a reviewer does not need to provide expression-training data, histology whole-slide image data, or rerun anything in `model_training/` just to start the downstream Visium-facing spatial prediction workflow.

The file is a derived governed teacher-label table generated from the expression-response and histology-response teacher workflows. It is not raw expression data, raw histology data, whole-slide image data, h5ad data, or a trained model artifact.

Users who need full upstream reproducibility should still run `expression_response_model_v2/`, `histology_response_model_v2/`, and `teacher_builder/` from their own configured data.
<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_END -->

From a new PowerShell session:

```powershell
cd "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\model_training"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Run the repository-level smoke test without rerunning model training:

```powershell
.\tests\smoke_test_model_training.ps1
```

The smoke test checks active file presence, Python compilation, YAML parsing, and selected output readability when local outputs are available. It does not retrain models and does not rewrite canonical outputs.

To run an individual module, enter that module folder and use its runner:

```powershell
cd ".\expression_response_model_v2"
.\run_expression_response_model_v2.ps1 -StartAt 0 -StopAt 4

cd "..\histology_response_model_v2"
.\run_histology_response_model_v2.ps1 -StartAt 0 -StopAt 9
```

Read the module README before intentionally rerunning either workflow.

## Reproducibility and configuration rules

For routine use, copy each tracked `.example.yaml` template to a local YAML config, then change paths, thresholds, training options, tiling settings, and output locations in the local file:

```text
expression_response_model_v2/configs/expression_response_model_v2.example.yaml -> expression_response_model_v2/configs/expression_response_model_v2.yaml
histology_response_model_v2/configs/histology_response_model_v2.example.yaml -> histology_response_model_v2/configs/histology_response_model_v2.yaml
```
Do not edit files in `scripts/` unless you are intentionally changing pipeline logic. The `scripts/` folders contain the active source-code implementation and should remain named `scripts/`.

Important rules:

- use YAML files for routine configuration changes;
- avoid hardcoded local paths in source scripts;
- do not overwrite canonical local outputs unless the rerun is intentional;
- run smoke tests or scratch rerun probes before changing active outputs;
- keep large local outputs out of ordinary Git commits unless using Git LFS or external archival storage;
- keep local provenance, old backups, and deprecated workflows outside the GitHub-tracked source package.

## Repository layout

The active `model_training` layout is:

```text
model_training/
├── README.md
├── RUNBOOK.md
├── requirements-model-training.txt
├── .gitignore
├── expression_response_model_v2/
│   ├── README.md
│   ├── run_expression_response_model_v2.ps1
│   ├── configs/
│   └── scripts/
├── histology_response_model_v2/
│   ├── README.md
│   ├── run_histology_response_model_v2.ps1
│   ├── configs/
│   └── scripts/
├── tests/
└── tools/
```

Generated folders such as `outputs/`, `logs/`, cache folders, and local archive folders are expected to remain local and are not part of the GitHub source package.

| Folder | Purpose | GitHub handling |
| --- | --- | --- |
| `expression_response_model_v2/` | Active expression-response teacher workflow. | Commit source, config templates/examples, README, runner, and small durable docs. Keep large outputs local/external. |
| `histology_response_model_v2/` | Active histology-response teacher workflow. | Commit source, config templates/examples, README, runner, and small durable docs. Keep large outputs local/external. |
| `tests/` | Lightweight non-rerun smoke tests. | Commit. |
| `tools/` | Optional reusable maintenance/audit utilities. | Commit only useful reusable tools. |

## Module summaries

### expression_response_model_v2

This module trains deployable expression-response teacher models. It uses expression profiles, named-treatment labels, and binary response labels to train treatment-specific calibrated models.

Main workflow:

```text
treatment ontology
    → input validation
    → canonical expression-response training table
    → deployable calibrated model training
    → model audit
    → optional Visium pseudobulk teacher scoring
```

Typical local outputs include:

```text
expression_response_model_v2/outputs/deployable_CH1/model_index.tsv
expression_response_model_v2/outputs/deployable_CH1/model_index_approved.tsv
expression_response_model_v2/outputs/deployable_CH1/models/*.joblib
```

These outputs are scientifically meaningful but are generated locally and are not included in GitHub.

Read the module guide:

```text
expression_response_model_v2/README.md
```

### histology_response_model_v2

This module trains and audits treatment-conditioned H&E whole-slide image response models. It compares `treatment_only`, `image_only`, and `image_treatment` model families to assess whether image morphology adds signal beyond treatment identity.

Main workflow:

```text
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
```

Typical local output:

```text
histology_response_model_v2/outputs/histology_v2/09_audit/histology_model_index.tsv
```

This output is generated locally and is not included in GitHub.

Read the module guide:

```text
histology_response_model_v2/README.md
```

## Validation status

After documentation and folder cleanup, the repository-level non-rerun smoke test was used to check active scripts, configs, runners, and selected readable outputs. Scratch micro-reruns were also used during development to test plumbing without overwriting canonical outputs.

Use this command for lightweight review:

```powershell
.\tests\smoke_test_model_training.ps1
```

For stricter end-to-end validation, run each module intentionally with full configured data and a scratch output root. Do not use a full rerun as a casual smoke test because model training can be slow and may overwrite outputs if active YAML output paths are unchanged.

## GitHub and publication guidance

Recommended to commit:

- active module scripts;
- active config templates or small reusable configs;
- module READMEs;
- top-level `README.md` and `RUNBOOK.md`;
- `requirements-model-training.txt`;
- `.gitignore`;
- lightweight tests;
- concise durable documentation.

Recommended to keep local or archive externally:

- local archive folders;
- Python cache files;
- installer bundles;
- trained `.joblib`, `.pkl`, `.pt`, or similar model artifacts unless using Git LFS or another deliberate model-artifact policy;
- large canonical expression training tables;
- tile manifests, tile training tables, tile prediction tables, and artifact metric tables;
- generated all-code/all-data bundles;
- one-off diagnostic reports and patch backups.

The `.gitignore` in this folder is intended to help prevent accidental commits of local provenance and large generated outputs.

## Relationship to teacher_builder

The model-training layer produces audited teacher candidates. It should not be collapsed into `teacher_builder` and should not be rerun implicitly inside downstream teacher fusion.

Expected handoff pattern:

1. train and audit expression and histology teachers in `model_training/`;
2. preserve model indexes, approval flags, calibration fields, reliability weights, and audit summaries locally;
3. let `teacher_builder` consume approved artifacts or scored teacher tables;
4. keep treatment-prior governance and spatial fusion downstream.

This separation keeps model fitting, model approval, teacher fusion, and final spatial modeling as distinct auditable stages.

## Notes for reviewers

Recommended reading order:

1. this `README.md`;
2. `RUNBOOK.md`;
3. `expression_response_model_v2/README.md`;
4. `histology_response_model_v2/README.md`;
5. each module YAML config;
6. shared library/helper scripts;
7. model-training and audit scripts;
8. available local output READMEs or audit summaries, if provided separately.

This folder has been organized for GitHub readability and reproducible review. Active source remains under `scripts/`; documentation and cleanup changes are intended to be non-behavioral unless explicitly noted in a module changelog or script header.


