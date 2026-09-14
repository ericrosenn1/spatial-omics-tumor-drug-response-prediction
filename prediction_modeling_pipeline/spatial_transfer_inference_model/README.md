# Spatial Transfer Inference Model

## Overview

`spatial_transfer_inference_model` applies a completed spatial treatment-response interpretation atlas to new Visium spatial transcriptomics samples.

The module takes a transfer-ready spatial feature table, aligns each sample to a frozen strict-feature registry, and scores how strongly each sample's spatial architecture aligns with signed feature effects associated with higher or lower teacher residuals. It can run on a single Visium sample or on a small batch of samples.

The main output is a sample-by-treatment interpretation table with spatial alignment scores, rescaled interpretation scores, confidence/evidence support labels, feature drivers, biological theme drivers, QC reports, and a final local output package.

The outputs are intended for biological interpretation and hypothesis generation. They are not clinical treatment recommendations.

## Use case

This module is useful for questions such as:

- Which treatment signatures show the highest signed spatial alignment for a given Visium sample?
- Which samples show stronger alignment with lower teacher residuals?
- Which spatial features drive each treatment-specific score?
- Which biological themes recur across treatment signatures?
- How do multiple Visium samples compare across the same frozen treatment atlas?

## Workflow position

The transfer model is the final downstream inference layer.

```text
Visium sample or batch
    -> spatial feature identification
    -> transfer-ready feature handoff
    -> spatial transfer inference
    -> treatment-alignment tables and interpretation outputs
```

It uses two completed upstream resources:

1. Spatial feature outputs from the spatial feature identification pipeline.
2. Frozen treatment-feature and treatment-theme effects from the prediction interpretation model.

## Repository structure

```text
spatial_transfer_inference_model/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
│   └── resolved_pim_transfer_file_map.example.json
└── scripts/
    ├── 00_run_spatial_transfer_inference_model.py
    ├── 01_prepare_transfer_inputs.py
    ├── 02_align_single_slide_features_to_v2.py
    ├── 03_score_transfer_drug_response_alignment.py
    ├── 04_make_single_slide_prediction_table.py
    ├── 05_qc_and_package_transfer_outputs.py
    ├── 00b_build_improved_transfer_handoff.py
    ├── 00c_audit_transfer_strict_feature_missingness.py
    ├── 00d_apply_reviewed_zero_fill.py
    └── _stim_utils.py
```

Generated folders such as `outputs/`, `logs/`, local archives, ZIP packages, workbooks, figures, and temporary reports are local-only and are not tracked in GitHub.

## Main pipeline scripts

### `00_run_spatial_transfer_inference_model.py`

Main orchestrator. Runs selected steps, manages output folders, captures logs, writes run summaries, and packages final outputs.

### `01_prepare_transfer_inputs.py`

Loads the transfer feature table, standardizes sample IDs, loads the frozen strict-feature registry, and prepares input rows for alignment.

### `02_align_single_slide_features_to_v2.py`

Aligns each input sample to the frozen strict spatial feature set. It records observed features, missing features, and neutral/imputed values used for transfer scoring.

### `03_score_transfer_drug_response_alignment.py`

Combines sample feature values with signed treatment-feature effects. It produces treatment-specific spatial alignment scores, higher-residual-supporting contributions, lower-residual-supporting contributions, feature contributions, and theme contributions.

### `04_make_single_slide_prediction_table.py`

Creates the readable sample-by-treatment interpretation table. This is the main table for reviewing transfer inference results.

### `05_qc_and_package_transfer_outputs.py`

Runs final QC checks and writes a portable local output package.

## Transfer handoff helper scripts

### `00b_build_improved_transfer_handoff.py`

Builds a transfer-ready feature table from spatial feature outputs and maps available features to the frozen strict feature registry.

### `00c_audit_transfer_strict_feature_missingness.py`

Audits strict feature coverage and classifies features as observed, recovered, absence-like, missing, or unavailable.

### `00d_apply_reviewed_zero_fill.py`

Applies reviewed zero-fill only for features where biological absence can be represented as zero. This improves transfer feature coverage while preserving true missingness where zero is not justified.

## Inputs

### Transfer feature table

The primary input is a transfer-ready `model_input_numeric.csv`-style table with one row per sample.

Required structure:

```text
sample_id
<strict spatial feature columns>
```

Example:

```text
sample_id,feature_1,feature_2,feature_3
SAMPLE_A,0.12,0.00,1.45
SAMPLE_B,0.08,0.20,1.10
```

### Frozen interpretation atlas

The module requires an explicitly selected completed prediction interpretation model run. The `--pim-run-root` argument binds every atlas table to that run. A module-global resolved file map is never consulted. Missing or duplicate atlas keys fail; another run is not substituted.


## Basic usage

Run all transfer steps from a PowerShell session after updating paths for your local machine:

```powershell
$TransferRoot = "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_transfer_inference_model"
$PimRunRoot = "<path-to-completed-prediction-interpretation-model-run>"
$FeatureTable = "<path-to-transfer-ready-model_input_numeric.csv>"
$Python = "<path-to-python-executable>"

& $Python (Join-Path $TransferRoot "scripts\00_run_spatial_transfer_inference_model.py") `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root $TransferRoot `
    --pim-run-root $PimRunRoot `
    --run-name "spatial_transfer_inference_example" `
    --output-root (Join-Path $TransferRoot "outputs\spatial_transfer_inference_example") `
    --sample-id "TRANSFER_BATCH" `
    --single-slide-feature-table $FeatureTable `
    --steps all `
    --python $Python `
    --open-output
```

## Single-sample run

For a single Visium sample, use a feature table with one row.

Expected output with a 27-treatment atlas:

```text
1 sample × 27 treatment signatures = 27 sample-treatment rows
```

## Multi-sample run

For multiple Visium samples, use one row per sample and keep `sample_id` as the first column.

Expected output with four samples and a 27-treatment atlas:

```text
4 samples × 27 treatment signatures = 108 sample-treatment rows
```

The `--sample-id` argument can be used as a run or batch label unless it exactly matches a sample ID in the input table.

## Output structure

A successful run creates a local output folder such as:

```text
outputs/<run_name>/
├── 01_prepared_transfer_inputs/
├── 02_aligned_features/
├── 03_transfer_scores/
├── 04_prediction_table/
├── 05_qc_and_transfer_package/
├── pipeline_run_logs/
├── spatial_transfer_inference_model_run_summary.json
├── spatial_transfer_inference_model_orchestrator_report.txt
└── spatial_transfer_inference_model_orchestrator_step_manifest.tsv
```

These generated outputs are intentionally excluded from GitHub.

## Key output files

### Main prediction table

```text
04_prediction_table/01_prediction_tables/single_slide_drug_response_interpretation_table.tsv
```

Contains one row per sample-treatment pair.

### Feature contribution table

```text
03_transfer_scores/02_feature_contributions/single_slide_treatment_feature_contributions.tsv
```

Lists feature-level drivers for each treatment score.

### Theme contribution table

```text
03_transfer_scores/03_theme_contributions/single_slide_treatment_theme_contributions.tsv
```

Summarizes biological theme-level drivers.

### Final QC report

```text
05_qc_and_transfer_package/04_reports/spatial_transfer_inference_final_qc_and_package_report.txt
```

Summarizes run status, row counts, package status, and QC checks.

### Final package

```text
05_qc_and_transfer_package/03_transfer_zip/spatial_transfer_inference_package.zip
```

Portable ZIP containing the main transfer outputs. ZIP packages are generated locally and should not be committed to GitHub.

## Important output columns

### `sample_id`

Input sample identifier.

### `drug_key`

Treatment signature from the frozen interpretation atlas.

### `rescaled_alignment_score_0_1`

Sigmoid(2 × signed alignment), an uncalibrated interpretation score on a 0–1 scale. It is not a measured or calibrated drug-response probability. Fitted V2 residual predictions are supplied by separate predictor bundles.

### `spatial_alignment_score`

Signed spatial alignment score. Positive values indicate alignment with higher teacher residuals; negative values indicate alignment with lower teacher residuals. These directions do not measure treatment sensitivity or resistance.

### `spatial_alignment_summary`

Qualitative label describing alignment with higher teacher residuals, lower teacher residuals, or an indeterminate/balanced profile.

### `confidence_level`

Evidence-support label based on feature coverage, score magnitude, and contribution support.

### `top_higher_teacher_residual_features`

Spatial features that most strongly support the higher-teacher-residual direction.

### `top_lower_teacher_residual_features`

Spatial features that most strongly support the lower-teacher-residual direction.

### `explanation`

Plain-language interpretation of the sample-treatment spatial alignment result.

## Recommended transfer workflow

### 1. Generate spatial features

Run the Visium sample or batch through the spatial feature identification pipeline.

### 2. Build transfer handoff

Use `00b_build_improved_transfer_handoff.py` to create a transfer-ready feature table.

### 3. Audit feature coverage

Use `00c_audit_transfer_strict_feature_missingness.py` to classify missing and observed strict features.

### 4. Apply reviewed zero-fill

`00d_apply_reviewed_zero_fill.py` requires explicit sample-feature absence evidence before any fill. Names alone do not establish absence. Without `--absence-evidence`, missing values remain NA; the alignment layer records neutral training-mean z=0 imputation separately from biological absence.

### 5. Run transfer inference

Use `00_run_spatial_transfer_inference_model.py` to generate treatment-alignment scores and interpretation outputs.

### 6. Review results

Start with:

```text
single_slide_drug_response_interpretation_table.tsv
spatial_transfer_inference_final_qc_and_package_report.txt
```

Then inspect feature and theme contribution tables for biological interpretation.

## Smoke test

A smoke run can be used to confirm that the pipeline executes and writes the expected output structure. Update the paths before running.

```powershell
$TransferRoot = "YOUR_PROJECT_ROOT\prediction_modeling_pipeline\spatial_transfer_inference_model"
$PimRunRoot = "<path-to-completed-prediction-interpretation-model-run>"
$Python = "<path-to-python-executable>"

& $Python (Join-Path $TransferRoot "scripts\00_run_spatial_transfer_inference_model.py") `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root $TransferRoot `
    --pim-run-root $PimRunRoot `
    --run-name "spatial_transfer_inference_smoke" `
    --output-root (Join-Path $TransferRoot "outputs\spatial_transfer_inference_smoke") `
    --sample-id "TRANSFER_SMOKE_SAMPLE_001" `
    --steps all `
    --python $Python `
    --smoke-test `
    --open-output
```

A passing smoke run writes prediction tables, contribution tables, QC files, and a final local package.

## Interpretation guidance

The model provides spatial response-alignment scores. It is best used to compare relative spatial treatment-alignment patterns across samples or treatment signatures.

A higher positive alignment indicates that the sample spatial feature profile aligns with features associated with higher teacher residuals for the full recorded-treatment profile.

A stronger negative alignment indicates that the sample spatial feature profile aligns with features associated with lower teacher residuals.

Treatment keys are inherited from the frozen model atlas. When needed, downstream reports can use simplified treatment-card labels while retaining the raw `drug_key` for provenance.

## GitHub policy

Commit source code, small reusable config examples, and durable documentation.

Do not commit:

```text
outputs/
logs/
local/
archive/
*.zip
*.xlsx
*.png
*.pdf
*.before_*
*_combined_*.txt
Pasted text*.txt
_LIVE_*.txt
```

Generated outputs should be regenerated from scripts or stored separately from the GitHub code repository.

## Development status

This module has been tested on smoke-test transfer input, one real Visium sample, and a four-sample Visium batch. The multi-sample run preserved all sample rows and produced the expected sample-by-treatment output table.

These test outputs are local artifacts and are not included in GitHub.


## Frozen deployment atlas and reproducibility

The reusable atlas contains the ordered reference features, training means and population standard deviations, full treatment keys, signed effects, dictionaries and source hashes. It uses the selected training reference for a single sample or a batch. Missing feature columns and duplicate sample identities fail. Explicit NA measurements remain missing in source tables and are scored at the training mean with observed-weight coverage reported.

```powershell
.venv\Scripts\python.exe prediction_modeling_pipeline\spatial_transfer_inference_model\scripts\alignment_bundle.py export --pim-run-root "PIM_RUN" --bundle "local\alignment_atlas.json"
.venv\Scripts\python.exe prediction_modeling_pipeline\spatial_transfer_inference_model\scripts\alignment_bundle.py score --bundle "local\alignment_atlas.json" --feature-table "TRANSFER_FEATURES.csv" --output-root "local\alignment_results"
.venv\Scripts\python.exe -m pytest prediction_modeling_pipeline\spatial_transfer_inference_model\tests -q
```

Expected rows are the actual sample count times the bundle's treatment-profile count. Historical examples using 27 profiles are not acceptance criteria. This atlas has no automatic claim to the held-out performance of the V2 fitted estimator; its own exact scoring rule requires separate evaluation before making a prediction-accuracy claim.

## Verified spatial feature export

Use `scripts/export_verified_spatial_feature_handoff.py` to export identified cumulative Step09 measurements before frozen-reference prediction or alignment. It verifies source/map/matrix hashes and retained-barcode identity evidence, preserves exact measurement values and NA, and rejects duplicate identities or raw headers. It does not extract features or retrain teachers. Input schemas and the maintained command are in [verified feature export](docs/verified_feature_export.md).
