# Spatial Omics Tumor Drug Response Prediction

A Python workflow that converts Visium tumor measurements into spatial features, fits treatment-specific response models, and applies saved models to new Visium samples. An included development response-target package lets users run downstream spatial modeling without retraining the expression and histology models.

Research use only. Outputs are model-derived estimates, not clinical treatment recommendations.

## What the tool does

- Extracts structural, functional and metabolic programs and their spatial organization.
- Builds section-level features describing tissue composition, accessibility, hotspots, regions and pairwise relationships.
- Fits pooled and treatment-specific models of transferred response estimates and their deviations from treatment priors.
- Produces feature/theme interpretations and predicts new samples using saved preprocessing and models.

Existing filenames use “teacher” for a treatment-specific response estimate transferred from expression and/or histology models. Public instructions refer to these as response targets.

## Workflow

```text
Development Visium input -> spatial feature extraction
                                      +
                    included development response targets
                                      |
                       spatial-response model fitting
                                      |
                  interpretation + saved models/reference
                                      |
New Visium input -> spatial extraction -> predictions + weighted feature scores
```

The included package contains targets and matching spatial features for the same 102 development sections. It does not provide response labels for arbitrary new samples. New samples are scored with saved models, without refitting.

A separate [upstream rebuild](#rebuilding-expressionhistology-response-targets) trains expression and histology models from response-linked source data before constructing replacement development targets.

## Repository structure

| Location | Purpose |
|---|---|
| [spatial_feature_identification_pipeline](spatial_feature_identification_pipeline/README.md) | Visium preprocessing and spatial feature extraction |
| [model_training](prediction_modeling_pipeline/model_training/README.md) | Optional expression and histology model training |
| [teacher_builder](prediction_modeling_pipeline/teacher_builder/README.md) | Response-target construction and included development package |
| [spatial_prediction_model_V2](prediction_modeling_pipeline/spatial_prediction_model_V2/README.md) | Maintained spatial modeling, validation and fitted predictors |
| [prediction_interpretation_model](prediction_modeling_pipeline/prediction_interpretation_model/README.md) | Feature weights and biological-theme summaries |
| [spatial_transfer_inference_model](prediction_modeling_pipeline/spatial_transfer_inference_model/README.md) | Saved-reference feature alignment and weighted scoring |
| [scripts](scripts) / [data_manifest](data_manifest) | Smoke test, source audit and public Visium staging |
| [notebooks](notebooks/README.md) / [docs](docs) | Code reference and execution documentation |

## Installation

Use Python 3.12. From a fresh checkout, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installation failed" }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
```

On Linux/macOS, create the environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`, then run the same pip commands.

The main requirements include spatial extraction, modeling and tests. The compatibility constraints retain NumPy 1.26 for PyUCell 0.6. Original neural training has separate [optional dependencies](prediction_modeling_pipeline/model_training/requirements-model-training.txt), including PyTorch/torchvision and an OpenSlide native runtime. See [environment and resource notes](docs/REVIEWER_EXECUTION.md).

## Quick start

Run the deterministic software smoke test and prepare the included development inputs:

```powershell
python scripts/run_smoke.py --output local/smoke
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py --manifest prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json --output local/precomputed_handoff
if ($LASTEXITCODE -ne 0) { throw "Response-target preparation failed" }
```

Use new or empty output directories. Inspect `local/smoke/smoke_report.json` and `local/precomputed_handoff/precomputed_handoff_manifest.json`.

The smoke test creates small synthetic data and fitted test objects. It exercises preprocessing, clustering, fusion, regression, interpretation, saved-model prediction and weighted scoring. It is not biological reproduction or a substitute for the full 1,000-permutation analysis.

## Full Visium workflow

Run commands from the repository root.

### 1. Provide Visium input and extract features

Stage public data as described below, or supply compatible sample folders. Copy the configuration and set its `input_root` and `output_root`:

```powershell
Copy-Item spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.example.yaml spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.local.yaml
python spatial_feature_identification_pipeline/run_pipeline.py --config spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.local.yaml --start 01 --end 10
if ($LASTEXITCODE -ne 0) { throw "Spatial extraction failed" }
```

The example uses `Visium_samples/visium_cohort_clean` and writes under `local/spatial_features`. Scientific parameters are defined by stage CLIs/constants, not unused configuration fields. Required clustering/scoring failures stop processing. Preserve the recorded gene-set source; an explicitly chosen curated library is not interchangeable with the development MSigDB library.

Step 09 produces the cumulative raw spatial measurement table. Step 10 constructs development modeling features. The included package supplies a matching validated development table if feature extraction is not being repeated.

### 2. Fit spatial-response models

After the quick-start preparation:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py --mode full --handoff-root local/precomputed_handoff --output-root local/spatial_model --max-workers 2 --full-step09-n-shuffles 1000 --full-step09-n-repeats 5
if ($LASTEXITCODE -ne 0) { throw "Spatial modeling failed" }
```

Full mode requires 1,000 within-treatment permutations and five repeated 80:20 splits. Reduced settings are restricted to smoke mode. Conditional testing uses the selected development feature and treatment families; see the [evaluation contract](prediction_modeling_pipeline/spatial_prediction_model_V2/docs/corrected_evaluation_and_prediction.md).

The command uses the included matching feature/target package. To substitute a newly extracted development feature table, use Script 16's explicit input/hash arguments after verifying sample identities and ordered features; do not silently replace the distributed package or assign its targets to another cohort.

### 3. Interpret and export fitted models

```powershell
python prediction_modeling_pipeline/prediction_interpretation_model/scripts/00_run_prediction_interpretation_model.py --v2-run-root local/spatial_model --output-root local/interpretation --steps all
if ($LASTEXITCODE -ne 0) { throw "Interpretation failed" }
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/14_fit_predictor_bundles.py --run-root local/spatial_model --teacher-root local/precomputed_handoff --raw-feature-table local/spatial_features/output_09_build_motif_tables/slide_features_with_motif_tables.csv --output local/fitted_models
if ($LASTEXITCODE -ne 0) { throw "Model export failed" }
```

Model export requires the corresponding raw development Step 09 table to save reference transformations. That raw table is not part of the compact package. Supply it from extraction or a verified matching run. Inspect output QC before applying models.

### 4. Predict new Visium samples

Extract new samples through Step 09 using a separate input/output configuration. Do not fit Step 10 transformations on an external batch.

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/15_predict_spatial_features.py --bundles local/fitted_models --features YOUR_NEW_STEP09_TABLE --representation raw_reference --mode external --output local/new_predictions.tsv --contributions local/new_contributions.tsv
if ($LASTEXITCODE -ne 0) { throw "Prediction failed" }
```

Replace `YOUR_NEW_STEP09_TABLE` with the extracted table path. Models reuse their saved feature order, preprocessing and reference distributions. Missing required columns fail; explicit missing measurements use fitted imputation with coverage reporting. Unsupported omitted fields require a source-bound missingness manifest.

For separate signed feature-weight scores, follow the [weighted-scoring commands](prediction_modeling_pipeline/spatial_transfer_inference_model/README.md). These scores are not fitted regression predictions.

## Rebuilding expression/histology response targets

The [upstream training guide](prediction_modeling_pipeline/model_training/README.md) documents configuration and runner commands. Supply response-linked expression tables and whole-slide images/clinical labels, train the expression and histology models, then run [response-target construction](prediction_modeling_pipeline/teacher_builder/README.md).

The fusion model anchors modality estimates to treatment-specific priors using reliability, confidence and histology-control attenuation. The spatial regression target is `fused_prob_responder - treatment_prior`. Treatment keys preserve recorded profile components, not assumed simultaneous regimens.

## Inputs

Visium extraction requires count matrices, gene identifiers, barcodes, spatial coordinates and the required image/registration metadata. Modeling requires matching section IDs, numeric features, an ordered feature manifest and section–treatment response targets. Prediction requires trusted saved model/reference artifacts and new spatial features.

Raw data, trained models and full numerical result archives are not distributed. Rebuild them or supply verified copies; [artifact requirements](docs/artifact_requirements.json) describe the prerequisites without claiming a public model download.

## Outputs

| Stage | Main outputs |
|---|---|
| Spatial extraction | Processed sections, annotation/QC tables, cumulative raw measurements and development feature table |
| Response-target preparation | Matching response table, numeric features and ordered/hash-verified manifest |
| Spatial modeling | Screening metrics, selected predictors, permutation results/checkpoints and final QC |
| Interpretation | Signed feature weights, treatment summaries and biological-theme tables |
| New-sample prediction | Fitted residuals, prior-anchored estimates where defined, contributions and coverage |
| Weighted scoring | Signed feature scores, display rescaling, feature/theme contributions and coverage |

Generated data, logs and models remain local and are ignored by Git.

## Public Visium data staging

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py --visium-root Visium_samples --dry-run
python scripts/download_and_reconstruct_public_visium_sources.py --visium-root Visium_samples --download --stage
if ($LASTEXITCODE -ne 0) { throw "Public data staging failed" }
```

Downloads can be large. Use `--sample-id SAMPLE_0000` for a selected sample, `--download` alone to cache, `--stage` alone to reconstruct from cached files, or `--skip-zenodo` to omit the large TLS archive explicitly. The manifest retains 103 candidate sample IDs; input availability and QC determine the usable cohort.

See [staging documentation](docs/PUBLIC_SOURCE_RECONSTRUCTION.md) for source URLs, checksum/resume behavior, inventories and errors.

## Reproducibility and testing

```powershell
python -m pytest tests spatial_feature_identification_pipeline/tests prediction_modeling_pipeline/teacher_builder/tests prediction_modeling_pipeline/spatial_prediction_model_V2/tests prediction_modeling_pipeline/prediction_interpretation_model/tests prediction_modeling_pipeline/spatial_transfer_inference_model/tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
python scripts/audit_public_repository.py
if ($LASTEXITCODE -ne 0) { throw "Source audit failed" }
python -m pip check
```

Tests cover fusion arithmetic, hashes and identities, feature order/missingness, deterministic permutation behavior, saved-model reload, batch/reordering invariance, failure propagation and safe staging. CI runs unit tests, the synthetic smoke test, package verification and the source/link audit without downloading scientific datasets.

## Data included with the repository

The only analysis-derived input package contains 34,881 section–treatment pairs, 374 recorded treatment profiles, the matching 102-section/661-feature table, an ordered feature manifest and an integrity manifest. See [package details](prediction_modeling_pipeline/teacher_builder/README.md). Small synthetic test fixtures and the public-source staging manifest are also included.

## Citation

Associated working manuscript (unpublished): Eric Rosenn, *A Visium Transcriptomics Workflow Linking Tumor Spatial Features to Treatment Response*. The manuscript remains a draft. Software citation metadata are provided in [CITATION.cff](CITATION.cff); no DOI or journal publication is asserted.

## License

Released under the [MIT License](LICENSE).

## Contact

For software questions or reproducibility issues, open a [GitHub issue](https://github.com/ericrosenn1/spatial-omics-tumor-drug-response-prediction/issues).
