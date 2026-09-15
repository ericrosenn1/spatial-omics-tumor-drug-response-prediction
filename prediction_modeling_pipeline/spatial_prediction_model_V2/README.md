# Spatial treatment-response models

Fit pooled and treatment-specific regressors to spatial features and prior-adjusted response targets. This is the maintained, self-contained spatial modeling implementation.

## Inputs and execution

Prepare the [included development inputs](../teacher_builder/README.md) first. The input directory contains `model_input_numeric.csv`, `visium_fused_teacher_table.tsv`, and the ordered `feature_manifest.csv`.

Run from the repository root:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py --mode full --handoff-root local/precomputed_handoff --output-root local/spatial_model --max-workers 2 --full-step09-n-shuffles 1000 --full-step09-n-repeats 5
if ($LASTEXITCODE -ne 0) { throw "Spatial modeling failed" }
```

Use a new output folder. Full mode requires 1,000 permutations and five repeated 80:20 splits; reduced settings are rejected. `--mode smoke` uses smaller screening/validation settings for execution checks, not scientific reproduction. For a short data-independent test, run `python scripts/run_smoke.py --output local/smoke` instead.

The runner keeps the active Python interpreter; `--python` is an explicit override. Two validation workers are the default. Each worker uses one XGBoost thread.

## Stages and model definitions

| Steps | Purpose |
|---|---|
| 01–02 | Validate matching inputs, build pooled residual data and treatment eligibility |
| 03–04 | Pooled response-estimate and residual models, grouped by section |
| 05 | Select biological spatial predictors and exclude technical fields |
| 06 | Model section-level residual summaries |
| 07–08 | Screen treatment-specific models and select the permutation-test family |
| 09 | Within-treatment conditional permutation test with resumable numerical blocks |
| 10–12 | Feature interpretation tables, reporting and output QC |

“Registry” in filenames means the selected spatial feature list. “Tier 1” identifies the candidates meeting the Step 08 permutation-test criteria; it is not a clinical evidence grade.

Full pooled models use five section-grouped 80:20 splits and 150/200 trees for the response-estimate/residual targets. Treatment-specific screening uses ten 80:20 splits, up to 60 training-selected predictors and 120 trees. Validation uses 80 trees. Common XGBoost settings are depth 2, learning rate 0.03, subsample 0.85, feature subsample 0.80, squared-error loss and histogram tree construction.

The development eligibility table requires at least 60 sections, residual SD 0.02, range 0.08 and ten distinct targets. Reported feature/model counts are outcomes, not hardcoded memberships.

Step 09 fixes the upstream-selected feature and treatment families, while repeating feature ranking, imputation and model fitting inside each split. It uses (1 + exceedances)/(1 + permutations), counts undefined null statistics as exceedances, and applies Benjamini–Hochberg correction over the complete selected family. Acceptance requires q <= 0.10, observed mean correlation above the null 95th percentile, and positive mean RMSE improvement. This is conditional validation, not an independent test of upstream discovery.

## Save models and predict new samples

Export fitted models with their selected features, priors and preprocessing:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/14_fit_predictor_bundles.py --run-root local/spatial_model --teacher-root local/precomputed_handoff --raw-feature-table YOUR_DEVELOPMENT_STEP09_TABLE --output local/fitted_models
if ($LASTEXITCODE -ne 0) { throw "Model export failed" }
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/15_predict_spatial_features.py --bundles local/fitted_models --features YOUR_NEW_STEP09_TABLE --representation raw_reference --mode external --output local/new_predictions.tsv --contributions local/new_contributions.tsv
if ($LASTEXITCODE -ne 0) { throw "Prediction failed" }
```

The raw development Step 09 table supplies the saved reference transformations. It is produced by spatial extraction and is not the normalized 661-feature modeling table. `--existing-bundles` attaches that reference to saved fitted estimators without refitting them.

Missing required columns fail. Explicit missing measurements use the saved imputer and retain coverage information. Unsupported omitted extraction fields require a source-hash-bound `--raw-missingness-manifest`; do not fill them arbitrarily. External data never fit a new normalization or feature-selection rule.

Fitted residual predictions are separate from [weighted feature scores](../spatial_transfer_inference_model/README.md). See the [evaluation contract](docs/corrected_evaluation_and_prediction.md) for the separately implemented grouped evaluation and independently accepted-model exporter. Serialized models should only be loaded from trusted sources.
