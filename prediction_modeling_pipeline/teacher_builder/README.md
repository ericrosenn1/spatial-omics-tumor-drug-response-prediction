# Visium Teacher Builder

## Overview

`teacher_builder` builds governed sample-by-treatment teacher labels for Visium spatial prediction modeling.

The workflow combines upstream response-teacher signals from:

- `model_training/expression_response_model_v2/`
- `model_training/histology_response_model_v2/`
- treatment-specific response priors

It anchors teacher signals to treatment priors, applies reliability-aware shrinkage, preserves modality provenance, assigns label-quality fields, and writes prediction-ready tables for downstream spatial response modeling.

This module does not train the final spatial prediction model. It prepares the governed teacher labels and spatial-feature handoff consumed by downstream modeling modules.

## Scientific role

The larger project asks whether spatial tumor architecture can help explain treatment-response biology. `teacher_builder` sits between upstream base-teacher models and downstream spatial modeling:

```text
expression_response_model_v2
         +
histology_response_model_v2
         +
treatment response priors
         ↓
governed teacher_builder fusion
         ↓
sample-by-treatment fused teacher labels
         ↓
spatial prediction modeling
```

The governed design is intentionally conservative. Raw teacher probabilities are not used as final labels without context. Each label records treatment prior, modality availability, reliability weights, confidence terms, residual-vs-prior targets, and label-quality flags.

The resulting labels are intended for model development and scientific analysis. They are not clinical treatment recommendations.

## Repository layout

```text
teacher_builder/
├── README.md
├── run_teacher_builder_governed.ps1
├── precomputed_governed_fused_teacher_table_102samples.tsv.gz
├── configs/
├── scripts/
└── docs/
```

Generated folders such as `outputs/`, `logs/`, local archives, diagnostic figures, and run-specific reports are local-only and are excluded from GitHub.

## Active governed workflow

The active governed workflow is run by:

```text
run_teacher_builder_governed.ps1
```

Core scripts:

```text
scripts/teacher_governance_lib.py
scripts/01_validate_teacher_inputs.py
scripts/02_build_expression_teacher.py
scripts/03_build_histology_teacher.py
scripts/04_fuse_teacher_tables.py
scripts/05_build_prediction_ready_teacher.py
scripts/06_qc_teacher_outputs.py
```

Typical configuration files:

```text
configs/visium_teacher_builder_governed_full.local.yaml
configs/visium_teacher_builder_governed_smoke_test.local.yaml
```

Use the full configuration for the full governed run and the sample configuration for lightweight smoke testing. Before running on a new machine, update paths in the YAML config to point to local spatial features, expression teacher artifacts, histology teacher artifacts, processed Visium files, and output locations.

## Pipeline steps

| Step | Script | Role |
|---:|---|---|
| 01 | `01_validate_teacher_inputs.py` | Validates spatial, metadata, expression, histology, processed Visium, and model-index inputs; writes sample availability, treatment priors, teacher registry, and governance config. |
| 02 | `02_build_expression_teacher.py` | Builds Visium pseudobulk expression profiles and scores approved expression-response models. |
| 03 | `03_build_histology_teacher.py` | Builds or imports governed histology teacher scores while preserving modality provenance. |
| 04 | `04_fuse_teacher_tables.py` | Standardizes expression and histology teacher tables, anchors labels to treatment priors, applies shrinkage, fuses modalities, and assigns label-quality fields. |
| 05 | `05_build_prediction_ready_teacher.py` | Joins fused teacher labels to numeric spatial features and writes model-ready handoff tables. |
| 06 | `06_qc_teacher_outputs.py` | Runs final QC summaries, checks, audit tables, diagnostic figures, and QC decision output. |

## Main local outputs

A successful governed run creates output folders similar to:

```text
outputs/01_input_validation/
outputs/02_expression_teacher/
outputs/03_histology_teacher/
outputs/04_fused_teacher/
outputs/05_prediction_ready_teacher/
outputs/06_teacher_qc/
```

Primary handoff files include:

```text
outputs/04_fused_teacher/fused_teacher_table.tsv
outputs/04_fused_teacher/teacher_fusion_audit.tsv
outputs/05_prediction_ready_teacher/model_input_numeric.csv
outputs/05_prediction_ready_teacher/visium_fused_teacher_table.tsv
outputs/05_prediction_ready_teacher/prediction_ready_training_table.tsv
outputs/05_prediction_ready_teacher/feature_manifest.csv
outputs/06_teacher_qc/qc_summary.txt
outputs/06_teacher_qc/qc_checks.tsv
outputs/06_teacher_qc/teacher_qc_decision.txt
```

These outputs are generated locally and are not committed to GitHub, except for the curated precomputed fused teacher handoff described below.

## Precomputed fused teacher handoff

For reviewer convenience, this folder includes a compressed precomputed governed fused teacher table:

```text
precomputed_governed_fused_teacher_table_102samples.tsv.gz
```

This file is a compact derived teacher-label handoff generated from the expression-response and histology-response teacher workflows. It is included so downstream Visium-facing workflows can be run without retraining the upstream expression and histology teacher models.

The table contains governed sample-treatment teacher labels for the full configured cohort:

```text
Rows: 34,881 sample-treatment pairs
Samples: 102
Treatments: 374
Columns: 52
Compressed size: approximately 3 MB
```

This file is not raw expression data, raw histology data, whole-slide image data, h5ad data, or a trained model artifact. It contains the fused teacher labels used as input to downstream spatial prediction.

Users who want to reproduce the full upstream workflow can regenerate this table by running `model_training/` followed by `teacher_builder`. Users who want to start from the downstream spatial prediction workflow can use this precomputed handoff.

## Quick start

From a new PowerShell session:

```powershell
cd "<path-to-project>\prediction_modeling_pipeline\teacher_builder"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Run the full governed workflow:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Config .\configs\visium_teacher_builder_governed_full.local.yaml `
  -StartAt 1 `
  -StopAt 6
```

Run a smoke test or small sample run:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Config .\configs\visium_teacher_builder_governed_smoke_test.local.yaml `
  -StartAt 1 `
  -StopAt 6
```

Run one step at a time:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Config .\configs\visium_teacher_builder_governed_full.local.yaml `
  -StartAt 4 `
  -StopAt 4
```

## Environment and package prerequisites

The runner accepts a `-Python` argument. If no Python executable is provided, it uses the configured/default Python environment.

The Python environment should include packages used across the governed scripts, including:

```text
numpy
pandas
PyYAML
scipy
scanpy
joblib
scikit-learn
matplotlib
```

Example import check:

```powershell
python -c "import numpy, pandas, yaml, scipy, scanpy, joblib, sklearn, matplotlib; print('teacher_builder environment OK')"
```

If this command fails, fix the Python environment before running the governed workflow.

## Preflight checks before running

From the `teacher_builder` folder, check the runner, config, and core scripts:

```powershell
Test-Path ".\run_teacher_builder_governed.ps1"
Test-Path ".\configs\visium_teacher_builder_governed_full.local.yaml"
Test-Path ".\scripts\teacher_governance_lib.py"
Test-Path ".\scripts\01_validate_teacher_inputs.py"
Test-Path ".\scripts\06_qc_teacher_outputs.py"
```

Each required file should return `True`.

The governed configs contain upstream paths for spatial features, metadata, expression model artifacts, histology model artifacts, processed Visium files, raw or derived image resources, and output roots. Confirm these config paths before running the full workflow.

## Step details

### Step 01: input validation

Writes sample availability reports, treatment priors, teacher reliability registry, governance config, and input validation summaries. This step records missing paths and sample availability and builds the treatment-prior table used by governed fusion.

### Step 02: expression teacher

Builds Visium pseudobulk expression profiles and expression teacher scores. Expression teacher rows include treatment keys, responder probabilities, model reliability, treatment-prior metadata, and teacher-mode provenance.

### Step 03: histology teacher

Builds or imports histology teacher scores and slide-level histology score summaries. Histology outputs should be interpreted with reliability weighting and shrinkage, especially when control warnings are present.

### Step 04: governed fusion

Starts from the treatment prior and adds only reliability-supported expression and histology deltas. This prevents raw teacher probabilities from becoming unqualified labels.

### Step 05: prediction-ready teacher handoff

Writes the numeric spatial feature matrix, fused teacher table, prediction-ready training table, feature manifest, run config, and summary outputs used by downstream spatial prediction models.

### Step 06: teacher QC

Writes QC summaries, QC checks, sample/treatment/feature QC tables, teacher fusion audit outputs, QC run config, and QC decision files. It may also write diagnostic figures for fused probabilities, residuals, modality composition, priors, label-quality flags, heatmaps, and shrinkage behavior.

## Verify outputs after a full run

After running Steps 01-06, check the key outputs under the configured output folder:

```powershell
$Out = "<path-to-teacher-builder-output-root>"
Test-Path "$Out\01_input_validation\treatment_priors.tsv"
Test-Path "$Out\02_expression_teacher\expression_teacher_scores.tsv"
Test-Path "$Out\03_histology_teacher\histology_teacher_scores.tsv"
Test-Path "$Out\04_fused_teacher\fused_teacher_table.tsv"
Test-Path "$Out\05_prediction_ready_teacher\model_input_numeric.csv"
Test-Path "$Out\05_prediction_ready_teacher\visium_fused_teacher_table.tsv"
Test-Path "$Out\06_teacher_qc\qc_summary.txt"
Test-Path "$Out\06_teacher_qc\teacher_qc_decision.txt"
```

A compact row-count check can be run after setting `$Out`:

```powershell
$Teacher = "$Out\05_prediction_ready_teacher\visium_fused_teacher_table.tsv"
$Features = "$Out\05_prediction_ready_teacher\model_input_numeric.csv"
python -c "import pandas as pd; t=pd.read_csv(r'$Teacher', sep='\t'); x=pd.read_csv(r'$Features'); print('teacher rows:', len(t)); print('samples:', t['sample_id'].nunique()); print('treatments:', t['drug_key'].nunique()); print('feature matrix:', x.shape)"
```

## Common rerun patterns

Rerun only input validation:

```powershell
.\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full.local.yaml -StartAt 1 -StopAt 1
```

Rerun expression teacher scoring only:

```powershell
.\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full.local.yaml -StartAt 2 -StopAt 2
```

Rerun histology teacher scoring only:

```powershell
.\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full.local.yaml -StartAt 3 -StopAt 3
```

Rerun fusion through final QC after teacher scores already exist:

```powershell
.\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full.local.yaml -StartAt 4 -StopAt 6
```

Rerun final QC only:

```powershell
.\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full.local.yaml -StartAt 6 -StopAt 6
```

## Governance fields to preserve downstream

Downstream spatial prediction should preserve these fields when possible:

```text
sample_id
slide_id
drug
drug_key
fused_prob_responder
fused_residual_vs_prior
treatment_prior
prior_source
expression_available
histology_available
expression_effective_weight
histology_effective_weight
modality_used
label_quality_flag
label_quality_reason
```

These fields make it possible to audit whether downstream performance is driven by spatial features, treatment priors, expression teachers, histology teachers, or weak/saturated labels.

## Interpretation notes

A high fused probability is not a clinical recommendation. It is a governed teacher label intended for model development and scientific analysis.

Treatment priors are central to interpretation. A fused residual near zero means the sample-specific teacher signal did not move far away from the treatment prior. A large positive or negative residual means expression and/or histology teachers contributed sample-specific evidence after reliability-aware shrinkage.

Histology outputs should be interpreted cautiously when control warnings are present. The governed fusion step records histology control factors and warning fields so sensitivity analyses can exclude or down-weight affected labels.

## Troubleshooting

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If the runner cannot find Python, pass it explicitly:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Python "<path-to-python-executable>" `
  -Config .\configs\visium_teacher_builder_governed_full.local.yaml `
  -StartAt 1 `
  -StopAt 6
```

If Step 02 fails because processed h5ad files are missing, check `processed_samples_dir` in the governed YAML config.

If Step 03 fails because histology source artifacts are missing, check the histology model path and histology scoring settings in the governed YAML config.

If Step 04 fails because both modalities are empty, rerun Steps 02 and 03 or inspect the expression and histology teacher output folders.

If Step 05 fails because spatial features are missing, check `spatial_feature_table` and `spatial_feature_manifest` in the governed YAML config.

If Step 06 returns `WARN` or `FAIL`, read `outputs/06_teacher_qc/qc_summary.txt`, `qc_checks.tsv`, and the diagnostic figures before using the teacher labels downstream.

## GitHub and publication guidance

Recommended to commit:

- active source scripts;
- active config templates or small reusable configs;
- module README and durable documentation;
- runner scripts;
- lightweight examples or smoke-test configs;
- the curated precomputed fused teacher handoff.

Recommended to keep local or archive externally:

- generated `outputs/` folders except the curated precomputed handoff;
- run logs;
- local archive folders;
- processed h5ad files;
- raw Visium images;
- raw expression data;
- whole-slide images and tile outputs;
- trained model artifacts;
- diagnostic figures;
- large TSV/CSV tables not explicitly curated for review;
- Excel workbooks;
- ZIP packages;
- one-off diagnostic reports and patch backups.

The `.gitignore` in this folder is intended to help prevent accidental commits of local provenance, large generated outputs, model artifacts, and temporary files.

## Review path

Recommended review order:

1. `README.md` for workflow overview and run instructions.
2. `configs/visium_teacher_builder_governed_full.local.yaml` for paths and governance controls.
3. `scripts/teacher_governance_lib.py` for treatment priors, key normalization, shrinkage, and label quality.
4. `scripts/04_fuse_teacher_tables.py` for the core governed fusion logic.
5. `scripts/05_build_prediction_ready_teacher.py` for downstream handoff tables.
6. `scripts/06_qc_teacher_outputs.py` for final QC checks and plots.

## Status

The governed `teacher_builder` workflow has been documented for source-code readability and reproducible review. Generated outputs are excluded from GitHub and should be regenerated locally or archived separately, with the exception of the curated compressed fused teacher handoff included for reviewer convenience.

