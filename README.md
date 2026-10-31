# Spatial Omics Tumor Drug Response Prediction

[![Workflow tests](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/actions/workflows/reviewer-smoke.yml/badge.svg?branch=main)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/actions/workflows/reviewer-smoke.yml) [![Codecov](https://codecov.io/gh/ericrosenn1/spatial-omics-tumor-drug-response-prediction/graph/badge.svg)](https://app.codecov.io/gh/ericrosenn1/spatial-omics-tumor-drug-response-prediction) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22775462.svg)](https://doi.org/10.5281/zenodo.22775462) [![Build Jupyter Book](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/actions/workflows/build-jupyter-book.yml/badge.svg?branch=main)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/actions/workflows/build-jupyter-book.yml) [![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fericrosenn1.github.io%2Fspatial-omics-tumor-drug-response-prediction%2F&label=docs)](https://ericrosenn1.github.io/spatial-omics-tumor-drug-response-prediction/) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/badge)](https://scorecard.dev/viewer/?uri=github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction) [![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/blob/main/constraints-reviewer.txt) [![License: MIT](https://img.shields.io/github/license/ericrosenn1/spatial-omics-tumor-drug-response-prediction)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/blob/main/LICENSE) [![Citation](https://img.shields.io/badge/citation-CITATION.cff-blue)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/blob/main/CITATION.cff) [![Latest release](https://img.shields.io/github/v/release/ericrosenn1/spatial-omics-tumor-drug-response-prediction)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/releases/latest) [![Release date](https://img.shields.io/github/release-date/ericrosenn1/spatial-omics-tumor-drug-response-prediction)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/releases/latest) [![Release downloads](https://img.shields.io/github/downloads/ericrosenn1/spatial-omics-tumor-drug-response-prediction/total)](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/releases/latest)

A research workflow for extracting tumor microenvironment features from Visium spatial transcriptomics, modeling treatment-specific response estimates, and applying saved models and feature-weight summaries to new sections.

**Start with the software smoke test below.** It runs without raw study data. Reproducing the full analysis requires additional inputs and substantially more computation. Model outputs are research estimates and spatial response-alignment evidence, not clinical treatment recommendations.

## Workflow and guides

| Stage | Guide | Inputs and outputs |
|---|---|---|
| 1. Stage public data | [Public Visium staging](docs/PUBLIC_SOURCE_RECONSTRUCTION.md) | Source manifest and public downloads → provenance-preserving sample folders |
| 2. Extract spatial features | [Spatial extraction](spatial_feature_identification_pipeline/README.md) | Counts, registered coordinates and image metadata → section-level features |
| 3. Prepare response targets | [Included development package](prediction_modeling_pipeline/teacher_builder/README.md) | Verified precomputed features and fused expression/histology targets, or a full upstream rebuild |
| 4. Fit spatial models | [Spatial prediction V2](prediction_modeling_pipeline/spatial_prediction_model_V2/README.md) | Matching features and targets → evaluated models and saved predictors |
| 5. Interpret models | [Prediction interpretation](prediction_modeling_pipeline/prediction_interpretation_model/README.md) | Completed model run → signed feature effects and biological-theme summaries |
| 6. Apply to new sections | [New-sample scoring](prediction_modeling_pipeline/spatial_transfer_inference_model/README.md) | New Step 09 features and saved development references → fitted predictions and separate weighted scores |

The [execution guide](docs/REVIEWER_EXECUTION.md) describes resource requirements and validation boundaries. The [Jupyter Book](https://ericrosenn1.github.io/spatial-omics-tumor-drug-response-prediction/) provides a browsable source and project-history reference.

## Installation

Use **Python 3.12**. Spatial extraction and reproducible V2 modeling use separate environments because their validated numerical dependencies differ. The commands below are for PowerShell; run them from the repository root. Each native command is checked before proceeding.

```powershell
git clone https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction.git
if ($LASTEXITCODE -ne 0) { throw "Repository clone failed" }
Set-Location spatial-omics-tumor-drug-response-prediction

# Spatial extraction, teacher construction, and software smoke tests.
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Spatial environment creation failed" }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Spatial installation failed" }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "Spatial dependency check failed" }

# Reproducible V2 modeling, interpretation, and saved-model inference.
py -3.12 -m venv .venv_v2
if ($LASTEXITCODE -ne 0) { throw "V2 environment creation failed" }
& .\.venv_v2\Scripts\python.exe -m pip install -r prediction_modeling_pipeline/spatial_prediction_model_V2/requirements-v2-reproducible.txt
if ($LASTEXITCODE -ne 0) { throw "V2 installation failed" }
& .\.venv_v2\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "V2 dependency check failed" }
```

On other platforms, use `python3.12 -m venv` and each environment's `bin/python` in place of `Scripts/python.exe`. Optional expression/histology training has [additional dependencies](prediction_modeling_pipeline/model_training/README.md), including native image libraries for histology.

The component guides use `python` for brevity. Activate the appropriate environment first (`.\.venv\Scripts\Activate.ps1` for spatial extraction or `.\.venv_v2\Scripts\Activate.ps1` for V2 and downstream modeling), or substitute its full Python path as in the examples below.

## First check: run without study data

Use a new or empty output directory:

```powershell
& .\.venv\Scripts\python.exe scripts/run_smoke.py --output local/smoke
if ($LASTEXITCODE -ne 0) { throw "Software smoke test failed" }
& .\.venv\Scripts\python.exe scripts/audit_public_repository.py --output local/source_audit.json
if ($LASTEXITCODE -ne 0) { throw "Repository audit failed" }
```

The deterministic smoke fixture exercises retained-spot processing, clustering, program scoring, teacher fusion, fitted prediction, conditional-test resume, interpretation and weighted scoring. Inspect `local/smoke/smoke_report.json` for the actual checks, source hashes, versions and exclusions. A passing fixture demonstrates software execution; it does not validate biological findings or clinical performance.

## Prepare the included development inputs

<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_START -->
The repository includes a compact, manifest-bound development package:

- `prediction_modeling_pipeline/teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz`: 34,881 section–treatment rows covering 102 sections and 374 treatment profiles.
- `prediction_modeling_pipeline/teacher_builder/model_input_numeric.csv`: the matching 102 sections and 661 spatial features.
- `prediction_modeling_pipeline/teacher_builder/feature_manifest.csv`: the ordered feature list.

These are derived development inputs. The response targets are transferred expression/histology model estimates, not observed Visium treatment outcomes. They cannot label arbitrary new samples.
<!-- PRECOMPUTED_TEACHER_HANDOFF_NOTE_END -->

Prepare and verify the three-file handoff:

```powershell
& .\.venv_v2\Scripts\python.exe prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py --manifest prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json --output local/precomputed_handoff
if ($LASTEXITCODE -ne 0) { throw "Development input preparation failed" }
```

This validates source hashes, sample identity, feature ordering, missingness and residual arithmetic. It preserves the source package and refuses to overwrite a populated destination.

## Model, interpret, and apply

After input preparation, start with the smaller V2 execution run:

```powershell
& .\.venv_v2\Scripts\python.exe prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py --mode smoke --handoff-root local/precomputed_handoff --output-root local/spatial_model_smoke --max-workers 2
if ($LASTEXITCODE -ne 0) { throw "V2 smoke run failed" }
```

This uses the real development package but reduced modeling settings. The complete analysis uses a new output directory and the established validation settings:

```powershell
& .\.venv_v2\Scripts\python.exe prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py --mode full --handoff-root local/precomputed_handoff --output-root local/spatial_model --max-workers 2 --full-step09-n-shuffles 1000 --full-step09-n-repeats 5
if ($LASTEXITCODE -ne 0) { throw "Full spatial modeling failed" }
& .\.venv_v2\Scripts\python.exe prediction_modeling_pipeline/prediction_interpretation_model/scripts/00_run_prediction_interpretation_model.py --v2-run-root local/spatial_model --output-root local/interpretation --steps all
if ($LASTEXITCODE -ne 0) { throw "Interpretation failed" }
```

Full mode is substantially more expensive than the software fixture. Inspect QC and validation results before interpretation; a successful process exit alone does not establish model acceptance.

For new Visium sections, follow the [spatial extraction guide](spatial_feature_identification_pipeline/README.md) through **Step 09**, then the [saved prediction](prediction_modeling_pipeline/spatial_prediction_model_V2/README.md#save-models-and-predict-new-samples) and [weighted scoring](prediction_modeling_pipeline/spatial_transfer_inference_model/README.md) guides. Both require saved development references. Do not fit Step 10 transformations on the external batch. Fitted response predictions and signed weighted feature scores are separate outputs.

## Configuration

The optional project profile records shared paths and expected handoffs:

```powershell
Copy-Item project_profile.example.yaml project_profile.local.yaml
notepad project_profile.local.yaml
```

It is a planning template, not an executable pipeline configuration. Edit the relevant module's local config or CLI arguments too. The V2 and interpretation runners accept CLI arguments; their example configuration files do not imply a `--config` option. Keep commands at the repository root unless a guide explicitly says otherwise.

Local profiles, raw inputs, environments, models, generated outputs, figures and archives are ignored by Git. Preserve validated outputs and use a fresh output folder for a new analysis. The source archive does not include raw Visium data, whole-slide images, trained model bundles or complete numerical result archives; [artifact requirements](docs/artifact_requirements.json) describe the external resources.

## Validation and limits

- Keep source identifiers, sample/feature order, source hashes and missing measurements attached to handoffs. Missing biological measurements must not be fabricated or silently filled with zero.
- Conditional permutation testing uses a fixed upstream-selected feature and treatment family. It does not measure independent discovery performance. See the [evaluation contract](prediction_modeling_pipeline/spatial_prediction_model_V2/docs/corrected_evaluation_and_prediction.md).
- The terms “teacher,” “mechanism atlas,” and “alignment” refer to transferred response estimates, feature/theme summaries, and weighted spatial scores. They do not establish measured treatment efficacy or causal mechanisms.
- Software tests cover reproducible fixtures and specific contracts. Full raw-data extraction, upstream neural training and external biological validation require the corresponding data, runtime resources and scientific review.
- Load serialized estimators only from trusted sources. Cross-platform execution can require native-library setup; the numerical workflow CI runs on Windows with Python 3.12.

The maintained implementation is `spatial_prediction_model_V2/`. Earlier spatial-model development remains available through Git history rather than as a runnable `spatial_prediction_model/` directory in this checkout.

## Citation and license

Use [CITATION.cff](CITATION.cff) and the [archived release DOI](https://doi.org/10.5281/zenodo.22775462) when citing the software. Released under the [MIT License](LICENSE).
