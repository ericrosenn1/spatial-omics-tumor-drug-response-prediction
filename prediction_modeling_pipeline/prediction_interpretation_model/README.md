# Spatial model interpretation

Summarize a completed spatial model run using signed feature effects, treatment profiles and biological themes. This module reads the completed run; it does not repeat model selection or change fitted estimators.

## Run

From the repository root:

```powershell
python prediction_modeling_pipeline/prediction_interpretation_model/scripts/00_run_prediction_interpretation_model.py --v2-run-root local/spatial_model --output-root local/interpretation --steps all
if ($LASTEXITCODE -ne 0) { throw "Interpretation failed" }
```

The project root defaults to this checkout and can be overridden with `--project-root`. Use `--steps 03,04` only when the preceding outputs already exist. The JSON example documents argument values; the runner accepts CLI arguments, not a `--config` option.

Steps 01–08 validate source tables, build feature/treatment dictionaries, calculate signed effects, generate treatment cards, summarize development samples, aggregate biological themes, produce final tables, and check output completeness.

## Quantities

Normalized feature importance is based on mean absolute SHAP values, with gain retained as a supported alternative. Directional feature weight equals normalized importance multiplied by the feature–residual Pearson correlation over finite paired development observations. Undefined correlations receive no directional weight.

Theme summaries sum signed and absolute feature weights. Development sample summaries use the 20 leading effects per profile. External weighted scoring uses the complete exported effect set, so these are distinct summaries.

The files named “mechanism atlas” contain treatment-specific feature-weight and theme summaries, not demonstrated causal mechanisms. The terms “frozen” and “reference” mean that development statistics are saved and are not refitted on new samples.

## New Visium samples

Export the completed interpretation with the same saved feature-reference models:

```powershell
python prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/alignment_bundle.py export --pim-run-root local/interpretation --feature-reference-bundles local/fitted_models --bundle local/feature_weights.json
if ($LASTEXITCODE -ne 0) { throw "Feature-weight export failed" }
```

Continue with [new-sample feature scoring](../spatial_transfer_inference_model/README.md). The exported weighted score is not an XGBoost prediction and must not inherit that estimator's performance claims. Fitted models are exported separately by spatial modeling Script 14.

Required source fields fail explicitly when missing. Preserve sample/treatment keys and inspect coverage; some development sections have no response-target overlap with the selected treatment family.
