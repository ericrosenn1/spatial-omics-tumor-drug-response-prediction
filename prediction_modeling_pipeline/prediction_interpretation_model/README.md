# Prediction Interpretation Model

## Overview

`prediction_interpretation_model` is the final interpretation and reporting layer for the spatial treatment-response modeling workflow. It consumes completed outputs from `spatial_prediction_model_V2` and converts them into biologically interpretable summaries of how spatial tissue architecture relates to treatment-response residuals.

This module does not train a new treatment-response predictor from scratch. It starts from the validated spatial residual modeling results produced by `spatial_prediction_model_V2` and builds a structured interpretation package.

The main interpretation products include:

- signed spatial feature effects;
- treatment-level interpretation cards;
- sample-level spatial sensitivity/resistance alignment scores;
- cross-treatment mechanism atlases;
- treatment similarity summaries;
- final publication tables;
- final figures;
- final reports;
- final QC and package manifests.

The goal is to make spatial prediction results interpretable at three levels:

1. **Feature level:** Which spatial architecture features are associated with higher or lower treatment-response residuals?
2. **Treatment level:** Which spatial mechanisms are associated with sensitivity or resistance for each validated treatment?
3. **Sample level:** Which samples have spatial profiles aligned with sensitivity-associated or resistance-associated signatures?

## Scientific motivation

Treatment response can be influenced not only by which biological programs are present, but also by where and how they are organized in tissue. Spatial features may capture biological barriers and microenvironmental states that are not fully represented by expression or histology alone.

Examples of spatial biology summarized by this module include:

- tumor-core and tumor-boundary organization;
- immune exclusion or immune penetration;
- stromal and extracellular-matrix barrier structure;
- hypoxic or stress-associated tumor regions;
- myeloid/macrophage spatial enrichment;
- angiogenic and vascular context;
- metabolic spatial programs;
- hotspot fragmentation and spatial continuity;
- tumor accessibility and partial-penetration patterns.

The output is intended for biological interpretation and hypothesis generation. It is not intended to provide clinical treatment recommendations.

## Input data

The required input is a completed `spatial_prediction_model_V2` run folder. The upstream V2 model produces residual modeling outputs, the strict spatial biology feature registry, label-shuffle-validated treatment models, recurrent spatial features, recurrent biology themes, and final QC files used by this interpretation layer.

The development run used for this project had the following source contract:

```text
34,881 sample-treatment pair rows
102 spatial samples
374 treatment keys
139 strict biology spatial features
27 label-shuffle-validated treatment models
11 recurrent biology themes
final V2 QC pass
no production dependency on deprecated V1 outputs
```

The main input argument is:

```text
--v2-run-root <path_to_completed_spatial_prediction_model_V2_run>
```

## What the model does

The pipeline converts V2 residual modeling outputs into a structured interpretation package.

First, it validates and prepares the V2 source files. It checks the expected V2 run structure, confirms QC status, records file provenance, and prepares a stable source manifest for downstream steps.

Second, it builds dictionaries for features, treatments, treatment components, and biology themes. These dictionaries convert model feature names into readable biological annotations and organize treatments into interpretable component classes.

Third, it computes signed spatial effects. V2 feature importance scores are useful, but they do not by themselves indicate whether a feature is associated with sensitivity or resistance. This module assigns directionality by correlating spatial feature values with the V2 residual response target, `fused_residual_vs_prior`, and weighting those associations by V2 model evidence.

A positive signed effect means that higher values of a spatial feature are associated with above-prior response residuals for a treatment. A negative signed effect means that higher values of a spatial feature are associated with below-prior response residuals.

Fourth, it summarizes treatment-level mechanisms. For each label-shuffle-validated treatment, the model generates a treatment interpretation card containing the strongest sensitivity-associated features, resistance-associated features, sensitivity-associated biology themes, resistance-associated biology themes, and relevant validation metrics.

Fifth, it generates sample-level interpretation scores. For each sample-treatment pair with validated treatment coverage, the model scores whether the sample's spatial profile aligns more strongly with sensitivity-associated or resistance-associated spatial signatures.

Finally, it builds final reporting outputs, including mechanism atlases, final publication tables, final figures, final narrative reports, QC checks, and a final package.

## Pipeline steps

The workflow is organized into numbered scripts.

| Step | Script | Purpose |
|---|---|---|
| 00 | `00_run_prediction_interpretation_model.py` | Runs one or more numbered pipeline steps. |
| 01 | `01_prepare_interpretation_inputs.py` | Validates the V2 run root and prepares source manifests and input tables. |
| 02 | `02_build_feature_and_treatment_dictionary.py` | Builds feature, biology-theme, treatment, component, and source-column dictionaries. |
| 03 | `03_compute_signed_spatial_effects.py` | Computes signed treatment-feature and treatment-theme effects. |
| 04 | `04_build_treatment_interpretation_cards.py` | Creates one interpretation card per validated treatment. |
| 05 | `05_build_sample_level_interpretations.py` | Computes sample-treatment spatial interpretation scores and feature contributions. |
| 06 | `06_build_mechanism_atlas.py` | Builds cross-treatment mechanism, component, similarity, and sample mechanism atlases. |
| 07 | `07_make_final_outputs.py` | Creates final publication tables, workbook, figures, captions, and reports. |
| 08 | `08_qc_and_package_final_outputs.py` | Runs final QC and creates the final package. |

Shared helper functions are stored in:

```text
scripts/_pim_utils.py
```

## Repository layout

```text
prediction_interpretation_model/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
│   └── example_prediction_interpretation_model_full_run.json
├── docs/
│   └── SOURCE_OF_TRUTH_POLICY.md
└── scripts/
    ├── 00_run_prediction_interpretation_model.py
    ├── 01_prepare_interpretation_inputs.py
    ├── 02_build_feature_and_treatment_dictionary.py
    ├── 03_compute_signed_spatial_effects.py
    ├── 04_build_treatment_interpretation_cards.py
    ├── 05_build_sample_level_interpretations.py
    ├── 06_build_mechanism_atlas.py
    ├── 07_make_final_outputs.py
    ├── 08_qc_and_package_final_outputs.py
    └── _pim_utils.py
```

Generated local folders such as `outputs/`, `logs/`, and `local/` are excluded from GitHub.

## Installation

Create or activate a Python environment with the required dependencies. The scripts use standard Python libraries plus common scientific and reporting packages.

```powershell
python -m pip install pandas numpy matplotlib openpyxl
```

## Configuration

Use the GitHub-safe example config as a template:

```text
configs/example_prediction_interpretation_model_full_run.json
```

Copy it to a local config file and update paths for your machine. Local configs with absolute machine-specific paths should remain untracked.

A typical local config must identify:

```text
project_root
model_root
v2_run_root
output_root
steps
policy settings
```

The repository should not track configs that contain private local paths, timestamped local output folders, or machine-specific source roots.

## Basic usage

Run the full interpretation workflow through the orchestrator.

```powershell
cd "YOUR_PROJECT_ROOT/prediction_modeling_pipeline/prediction_interpretation_model"

python .\scripts\00_run_prediction_interpretation_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --v2-run-root "<path-to-completed-spatial-prediction-model-V2-run>" `
    --run-name "prediction_interpretation_model_full_local" `
    --output-root "outputs\prediction_interpretation_model_full_local" `
    --steps all `
    --open-output
```

To run only selected steps, pass a comma-separated list:

```powershell
--steps 03,04,05
```

For example, after Step 01 and Step 02 have already passed, Steps 03 through 05 can be rerun with:

```powershell
python .\scripts\00_run_prediction_interpretation_model.py `
    --project-root "YOUR_PROJECT_ROOT" `
    --model-root "." `
    --v2-run-root "<path-to-completed-spatial-prediction-model-V2-run>" `
    --run-name "prediction_interpretation_model_full_local" `
    --output-root "outputs\prediction_interpretation_model_full_local" `
    --steps 03,04,05 `
    --open-output
```

## Main outputs

A successful full run creates local output folders such as:

```text
01_prepared_inputs/
02_feature_and_treatment_dictionary/
03_signed_spatial_effects/
04_treatment_interpretation_cards/
05_sample_level_interpretations/
06_mechanism_atlas/
07_final_outputs/
08_qc_and_final_package/
```

Important final output groups include:

```text
07_final_outputs/01_publication_tables_tsv/
07_final_outputs/02_publication_workbook/
07_final_outputs/03_final_figures/
07_final_outputs/04_final_reports/
08_qc_and_final_package/01_final_qc/
08_qc_and_final_package/03_final_package/
08_qc_and_final_package/04_reports/
```

These outputs are generated locally and are not committed to GitHub.

## Key output tables

The final output layer includes publication-oriented tables such as:

- `Final_Treatment_Interpretation_Cards.tsv`
- `Final_Signed_Treatment_Feature_Effects.tsv`
- `Final_Signed_Treatment_Theme_Effects.tsv`
- `Final_Cross_Treatment_Biology_Theme_Atlas.tsv`
- `Final_Cross_Treatment_Feature_Atlas.tsv`
- `Final_Component_Class_Mechanism_Atlas.tsv`
- `Final_Sample_Treatment_Interpretation_Scores.tsv`
- `Final_Sample_Interpretation_Summary.tsv`
- `Final_Sample_Mechanism_Summary.tsv`
- `Final_Treatment_Theme_Similarity_Edges.tsv`

The workbook combines final publication tables into one local review file. Workbooks are generated outputs and should not be committed unless intentionally released through a separate artifact strategy.

## Treatment interpretation cards

Step 04 writes one text card per label-shuffle-validated treatment. Each card summarizes:

- treatment key and treatment components;
- validation evidence;
- sensitivity-associated spatial features;
- resistance-associated spatial features;
- sensitivity-associated biology themes;
- resistance-associated biology themes;
- interpretation caveats.

These cards make the model output readable without requiring the reader to inspect raw SHAP or feature-effect tables.

## Sample-level interpretation scores

Step 05 computes sample-treatment spatial interpretation scores for validated treatments. A positive net score means the sample's spatial profile is aligned with sensitivity-associated spatial biology for that treatment. A negative net score means the sample's spatial profile is aligned with resistance-associated spatial biology for that treatment.

These scores are not response predictions and are not treatment recommendations. They are interpretation-layer summaries of how a sample's spatial architecture aligns with signed spatial mechanisms learned from V2 residual models.

## Coverage note: 93 of 102 samples in sample-level treatment scoring

The upstream V2 full run contains 102 spatial samples. However, sample-level treatment scoring in this interpretation layer is restricted to the 27 label-shuffle-validated treatment keys. Those validated treatment keys have pair-level residual rows for 93 samples, not all 102.

Therefore:

```text
27 validated treatments x 93 samples = 2,511 sample-treatment interpretation rows
```

The nine samples not scored in Step 05 are:

```text
SAMPLE_0029
SAMPLE_0030
SAMPLE_0031
SAMPLE_0032
SAMPLE_0033
SAMPLE_0077
SAMPLE_0085
SAMPLE_0086
SAMPLE_0095
```

A coverage audit showed that these samples are present in the full V2 sample universe but have zero pair-level overlap with the 27 validated treatment keys. They have rows only for a restricted set of 11 low-coverage single-agent treatment keys. This is an upstream sample-treatment coverage issue and should be treated as a coverage-qualified pass, not as a downstream scoring failure.

## Quality control

Each step writes reports, summary JSON files, manifests, and QC tables where relevant.

The final QC step checks:

- required reports exist;
- expected QC tables exist;
- upstream step summaries are present;
- final tables and figures are present;
- final package files are present;
- final package is created.

## GitHub policy

Commit source code, GitHub-safe config examples, and durable documentation.

Recommended to commit:

```text
README.md
.gitignore
.gitattributes
scripts/*.py
configs/example_prediction_interpretation_model_full_run.json
docs/*.md
```

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
local configs with absolute machine-specific paths
```

Generated outputs should be regenerated from the scripts or stored separately from the GitHub code repository.

## Interpretation caveats

This module summarizes associations between spatial features and treatment-response residuals. Signed effects are based on feature-target correlations weighted by model evidence. These associations do not establish causality.

Treatment cards, mechanism atlases, and sample-level scores should be interpreted as biological hypotheses and reporting summaries. They are not clinical treatment recommendations.

