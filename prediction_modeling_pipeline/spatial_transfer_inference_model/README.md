# Prediction and feature scoring on new Visium samples

This module applies saved development transformations and treatment-specific feature weights to new spatial measurements. Fitted regression predictions are produced separately by spatial modeling Script 15.

## Prerequisites

- Extract each new section through spatial feature Step 09.
- Use fitted spatial-model bundles with the saved raw-development reference.
- Export the completed interpretation to a feature-weight JSON bundle.

Raw data, trained models and the complete feature-weight bundle are not distributed. Reproduce them using the [spatial workflow](../spatial_prediction_model_V2/README.md), or obtain a verified trusted copy. Do not fit Step 10 normalization on the external batch.

## Saved-reference weighted scoring

Run from the repository root after model and interpretation export:

```powershell
python prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/alignment_bundle.py prepare-raw --bundle local/feature_weights.json --feature-reference-bundles local/fitted_models --raw-feature-table YOUR_NEW_STEP09_TABLE --output-feature-table local/new_reference_features.tsv
if ($LASTEXITCODE -ne 0) { throw "Feature alignment failed" }
python prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/alignment_bundle.py score --bundle local/feature_weights.json --feature-table local/new_reference_features.tsv --output-root local/new_feature_scores
if ($LASTEXITCODE -ne 0) { throw "Weighted scoring failed" }
```

Missing required raw fields fail. For unsupported measurements, supply an explicit, source-hash-bound `--unavailable-feature-manifest`; absent biological evidence is not zero-filled. The [verified feature exporter](docs/verified_feature_export.md) checks source identities, retained barcodes and mappings when combining existing extraction outputs.

## Score definition and outputs

“Alignment” in filenames means the signed weighted feature score:

```text
z = (feature - development_mean) / development_population_SD
A = sum(signed_weight * z) / sum(abs(signed_weight))
display_score = 1 / (1 + exp(-2*A))
```

Missing measurements contribute a neutral standardized value of zero only in this defined weighted calculation; their weights remain in the denominator. Feature and absolute-weight coverage are reported. A pair with no observed effect weight remains unscored.

The output directory contains `sample_treatment_alignment.tsv`, `feature_contributions.tsv`, `theme_contributions.tsv` when themes exist, and `alignment_inference_manifest.json`. These scores are not fitted response probabilities or clinical efficacy estimates.

For fitted residual predictions, use [Script 15](../spatial_prediction_model_V2/scripts/15_predict_spatial_features.py). Saved preprocessing, feature order, priors and estimators are reused without refitting. Tests cover row/column reordering, single-sample versus batch application, missingness, duplicate identifiers and reload consistency.

Older numbered transfer utilities are retained for compatible saved-table workflows. The bundle commands above are the maintained raw-feature entry point; do not use the historical zero-fill helper to infer unavailable biology.
