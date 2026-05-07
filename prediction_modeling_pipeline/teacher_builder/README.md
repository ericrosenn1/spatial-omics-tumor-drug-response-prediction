# Visium Teacher Builder

<!-- TEACHER_BUILDER_README_DOC_POLISH_V1 -->
<!-- TEACHER_BUILDER_README_DOC_POLISH_V2_CLEAN -->

## Purpose

`teacher_builder` builds governed sample-by-treatment teacher labels for Visium spatial prediction modeling.

The workflow combines two upstream response-teacher modalities:

- expression-response teachers from `expression_response_model_v2`;
- histology-response teachers from `histology_response_model_v2`.

It then anchors those teacher signals to treatment-specific response priors, applies reliability-aware shrinkage, preserves modality provenance, assigns label-quality fields, and writes prediction-ready tables for downstream spatial response modeling.

This module does not train the final spatial prediction model. It prepares the governed teacher labels and feature handoff that downstream modeling consumes.

## Scientific context

The project asks whether spatial tumor architecture can help explain treatment-response biology. The `teacher_builder` workflow sits between upstream base teachers and downstream spatial modeling:

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
    spatial prediction model

The governed design is intentionally conservative. Raw model probabilities are not used as final labels without context. Each label records treatment prior, modality availability, reliability weights, confidence terms, residual-vs-prior targets, and label-quality flags.

## Active governed workflow

The active governed workflow is run by:

    run_teacher_builder_governed.ps1

The active scripts are:

    scripts\teacher_governance_lib.py
    scripts\01_validate_teacher_inputs.py
    scripts\02_build_expression_teacher.py
    scripts\03_build_histology_teacher.py
    scripts\04_fuse_teacher_tables.py
    scripts\05_build_prediction_ready_teacher.py
    scripts\06_qc_teacher_outputs.py

The active governed configs are:

    configs\visium_teacher_builder_governed_full102.yaml
    configs\visium_teacher_builder_governed_sample5.yaml

Use `visium_teacher_builder_governed_full102.yaml` for the full governed run. Use `visium_teacher_builder_governed_sample5.yaml` for a small smoke test.

## Folder layout

    teacher_builder/
    ├── configs/
    │   ├── visium_teacher_builder_governed_full102.yaml
    │   └── visium_teacher_builder_governed_sample5.yaml
    ├── scripts/
    │   ├── teacher_governance_lib.py
    │   ├── 01_validate_teacher_inputs.py
    │   ├── 02_build_expression_teacher.py
    │   ├── 03_build_histology_teacher.py
    │   ├── 04_fuse_teacher_tables.py
    │   ├── 05_build_prediction_ready_teacher.py
    │   └── 06_qc_teacher_outputs.py
    ├── outputs/
    ├── run_teacher_builder_governed.ps1
    └── README.md

Archived, deprecated, backup, and historical run folders are retained for provenance but are not the canonical governed source workflow.

## Pipeline steps

| Step | Script | Role |
|---:|---|---|
| 01 | `01_validate_teacher_inputs.py` | Validates spatial, metadata, expression, histology, processed Visium, and model-index inputs; writes sample availability, treatment priors, teacher registry, and governance config. |
| 02 | `02_build_expression_teacher.py` | Builds Visium pseudobulk expression profiles and scores approved expression-response models. |
| 03 | `03_build_histology_teacher.py` | Runs the governed histology wrapper, delegates to the archived original histology scorer, and preserves compatible histology teacher outputs. |
| 04 | `04_fuse_teacher_tables.py` | Standardizes expression and histology teacher tables, anchors labels to treatment priors, applies shrinkage, fuses modalities, and assigns label-quality fields. |
| 05 | `05_build_prediction_ready_teacher.py` | Joins fused teacher labels to numeric spatial features and writes model-ready handoff tables. |
| 06 | `06_qc_teacher_outputs.py` | Runs final QC summaries, checks, figures, audit tables, and QC decision output. |

## Main outputs

The governed full run writes to the configured `output_root`. In the full102 config this is usually:

    D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder\outputs

Important output folders are:

    outputs\01_input_validation
    outputs\02_expression_teacher
    outputs\03_histology_teacher
    outputs\04_fused_teacher
    outputs\05_prediction_ready_teacher
    outputs\06_teacher_qc

Primary handoff files include:

    outputs\04_fused_teacher\fused_teacher_table.tsv
    outputs\04_fused_teacher\teacher_fusion_audit.tsv
    outputs\05_prediction_ready_teacher\model_input_numeric.csv
    outputs\05_prediction_ready_teacher\visium_fused_teacher_table.tsv
    outputs\05_prediction_ready_teacher\prediction_ready_training_table.tsv
    outputs\05_prediction_ready_teacher\feature_manifest.csv
    outputs\06_teacher_qc\qc_summary.txt
    outputs\06_teacher_qc\qc_checks.tsv
    outputs\06_teacher_qc\teacher_qc_decision.txt

## Fresh PowerShell quick start

Open a new PowerShell session and run:

    cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 1 -StopAt 6

For a smoke test:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_sample5.yaml -StartAt 1 -StopAt 6

To run one step at a time:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 4 -StopAt 4

## Environment and package prerequisites

The runner accepts a `-Python` argument and otherwise uses its configured/default Python. The project virtual environment is usually:

    D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe

The Python environment must include the packages used across the governed scripts:

    numpy
    pandas
    PyYAML
    scipy
    scanpy
    joblib
    scikit-learn
    matplotlib

Run this quick import check:

    & "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -c "import numpy, pandas, yaml, scipy, scanpy, joblib, sklearn, matplotlib; print(""teacher_builder environment OK"")"

If this command fails, fix the Python environment before running the governed workflow.

## Preflight checks before running

From the `teacher_builder` folder, check the full102 config and runner:

    Test-Path ".\run_teacher_builder_governed.ps1"
    Test-Path ".\configs\visium_teacher_builder_governed_full102.yaml"
    Test-Path ".\scripts\teacher_governance_lib.py"
    Test-Path ".\scripts\01_validate_teacher_inputs.py"
    Test-Path ".\scripts\06_qc_teacher_outputs.py"

Expected result for each required file is:

    True

The governed configs contain the exact upstream paths for spatial features, metadata, expression model artifacts, histology model artifacts, processed Visium h5ad files, raw Visium image folders, and output root. Confirm those config paths are correct before running the full workflow.

## Step 01 input validation

Step 01 writes:

    outputs\01_input_validation\sample_availability_report.csv
    outputs\01_input_validation\treatment_priors.tsv
    outputs\01_input_validation\teacher_reliability_registry.tsv
    outputs\01_input_validation\teacher_governance_config.json
    outputs\01_input_validation\teacher_input_validation_summary.txt

This step records missing paths and sample availability. It also builds the treatment-prior table used by governed fusion.

## Step 02 expression teacher

Step 02 writes:

    outputs\02_expression_teacher\visium_pseudobulk_expression.tsv
    outputs\02_expression_teacher\expression_teacher_scores.tsv
    outputs\02_expression_teacher\expression_teacher_summary.tsv
    outputs\02_expression_teacher\expression_teacher_summary.txt

Expression teacher rows include treatment keys, responder probabilities, model reliability, treatment prior metadata, and teacher-mode provenance.

## Step 03 histology teacher

Step 03 writes:

    outputs\03_histology_teacher\histology_teacher_scores.tsv
    outputs\03_histology_teacher\visium_histology_slide_scores.tsv
    outputs\03_histology_teacher\histology_teacher_treatment_summary.tsv
    outputs\03_histology_teacher\histology_teacher_summary.txt

The active Step 03 script is a governed wrapper. It calls the archived original histology scorer and refuses recursive wrapper calls.

## Step 04 governed fusion

Step 04 writes:

    outputs\04_fused_teacher\fused_teacher_table.tsv
    outputs\04_fused_teacher\visium_fused_teacher_table.tsv
    outputs\04_fused_teacher\teacher_fusion_audit.tsv
    outputs\04_fused_teacher\fused_teacher_by_sample.tsv
    outputs\04_fused_teacher\fused_teacher_by_drug.tsv
    outputs\04_fused_teacher\fused_teacher_missingness.tsv
    outputs\04_fused_teacher\fused_teacher_summary.txt

Fusion starts at the treatment prior and adds only reliability-supported expression and histology deltas. This prevents raw teacher probabilities from becoming unqualified labels.

## Step 05 prediction-ready teacher handoff

Step 05 writes:

    outputs\05_prediction_ready_teacher\model_input_numeric.csv
    outputs\05_prediction_ready_teacher\visium_fused_teacher_table.tsv
    outputs\05_prediction_ready_teacher\prediction_ready_training_table.tsv
    outputs\05_prediction_ready_teacher\feature_manifest.csv
    outputs\05_prediction_ready_teacher\prediction_ready_summary.txt
    outputs\05_prediction_ready_teacher\run_config.json

`model_input_numeric.csv` is the sample-by-feature spatial matrix. `visium_fused_teacher_table.tsv` and `prediction_ready_training_table.tsv` carry the governed teacher labels and joined feature rows.

## Step 06 teacher QC

Step 06 writes:

    outputs\06_teacher_qc\qc_summary.tsv
    outputs\06_teacher_qc\qc_summary.txt
    outputs\06_teacher_qc\qc_checks.tsv
    outputs\06_teacher_qc\qc_by_sample.tsv
    outputs\06_teacher_qc\qc_by_treatment.tsv
    outputs\06_teacher_qc\qc_by_feature.tsv
    outputs\06_teacher_qc\teacher_fusion_audit.tsv
    outputs\06_teacher_qc\teacher_qc_decision.txt
    outputs\06_teacher_qc\qc_run_config.json

Step 06 also writes diagnostic PNG figures for fused probabilities, residuals, modality composition, label-quality flags, priors, heatmaps, and modality shrinkage.

## Verify outputs after a full run

After running Steps 01-06, check:

    $Out = "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder\outputs"
    Test-Path "$Out\01_input_validation\treatment_priors.tsv"
    Test-Path "$Out\02_expression_teacher\expression_teacher_scores.tsv"
    Test-Path "$Out\03_histology_teacher\histology_teacher_scores.tsv"
    Test-Path "$Out\04_fused_teacher\fused_teacher_table.tsv"
    Test-Path "$Out\05_prediction_ready_teacher\model_input_numeric.csv"
    Test-Path "$Out\05_prediction_ready_teacher\visium_fused_teacher_table.tsv"
    Test-Path "$Out\06_teacher_qc\qc_summary.txt"
    Test-Path "$Out\06_teacher_qc\teacher_qc_decision.txt"

Expected result for each required output is:

    True

Print a compact row-count summary with:

    $Teacher = "$Out\05_prediction_ready_teacher\visium_fused_teacher_table.tsv"
    $Features = "$Out\05_prediction_ready_teacher\model_input_numeric.csv"
    python -c "import pandas as pd; t=pd.read_csv(r""$Teacher"", sep=""\t""); x=pd.read_csv(r""$Features""); print(""teacher rows:"", len(t)); print(""samples:"", t[""sample_id""].nunique()); print(""treatments:"", t[""drug_key""].nunique()); print(""feature matrix:"", x.shape)"

## Common rerun patterns

Rerun only input validation:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 1 -StopAt 1

Rerun expression teacher scoring only:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 2 -StopAt 2

Rerun histology teacher scoring only:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 3 -StopAt 3

Rerun fusion through final QC after teacher scores already exist:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 4 -StopAt 6

Rerun final QC only:

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 6 -StopAt 6

## Governance fields to preserve downstream

Downstream spatial prediction should preserve these fields when possible:

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

These fields make it possible to audit whether model performance is driven by spatial features, treatment priors, expression teachers, histology teachers, or weak/saturated labels.

## Interpretation notes

A high fused probability is not a clinical recommendation. It is a governed teacher label intended for model development and scientific analysis.

Treatment priors are central to interpretation. A fused residual near zero means the sample-specific teacher signal did not move far away from the treatment prior. A large positive or negative residual means expression and/or histology teachers contributed sample-specific evidence after reliability-aware shrinkage.

Histology outputs should be interpreted cautiously when control warnings are present. The governed fusion step records histology control factors and warning fields so sensitivity analyses can exclude or down-weight affected labels.

## Troubleshooting

If PowerShell blocks script execution, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If the runner cannot find Python, pass it explicitly:

    .\run_teacher_builder_governed.ps1 -Python "D:\Adv_Omics_Fenyo\project\.venv\Scripts\python.exe" -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 1 -StopAt 6

If Step 02 fails because processed h5ad files are missing, check `processed_samples_dir` in the governed YAML config.

If Step 03 fails because the original histology scorer is missing, check the archived scorer/backup location used by the Step 03 wrapper.

If Step 04 fails because both modalities are empty, rerun Steps 02 and 03 or inspect the expression and histology teacher output folders.

If Step 05 fails because spatial features are missing, check `spatial_feature_table` and `spatial_feature_manifest` in the governed YAML config.

If Step 06 returns WARN or FAIL, read `outputs\06_teacher_qc\qc_summary.txt`, `qc_checks.tsv`, and the diagnostic figures before using the teacher labels downstream.

## Documentation and safety policy

This documentation polish pass added module headers, section headers, inline comments, docstrings, YAML comments, runner comments, and README guidance only.

For Python scripts, safety checks used:

- `py_compile`;
- executable AST unchanged after docstrings were removed;
- backup creation before write.

For YAML configs, safety checks used:

- original YAML parse;
- candidate YAML parse;
- parsed YAML values unchanged after comment insertion;
- backup creation before write.

For PowerShell runner documentation, safety checks used:

- non-comment PowerShell lines unchanged before and after write;
- backup creation before write.

## GitHub/publication review guide

Recommended review order:

1. `README.md` for workflow overview and run instructions.
2. `configs\visium_teacher_builder_governed_full102.yaml` for paths and governance controls.
3. `scripts\teacher_governance_lib.py` for treatment priors, key normalization, shrinkage, and label quality.
4. `scripts\04_fuse_teacher_tables.py` for the core governed fusion logic.
5. `scripts\05_build_prediction_ready_teacher.py` for downstream handoff tables.
6. `scripts\06_qc_teacher_outputs.py` for final QC checks and plots.

## Current status

The governed teacher_builder workflow has been documented for public-code readability. Existing executable behavior, thresholds, paths, schemas, model artifact loading, fusion logic, QC logic, and output filenames were not intentionally changed by the documentation pass.
