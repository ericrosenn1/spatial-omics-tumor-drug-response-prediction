# Treatment-response modeling

This layer constructs treatment-specific response targets, fits spatial regressors, and applies saved models and feature-weight summaries to new Visium samples. Outputs are research estimates, not treatment recommendations.

## Choose an entry point

- [Included response-target package](teacher_builder/README.md): develop spatial models without retraining expression or histology models. Targets cover the same 102 development sections as the included feature table, not arbitrary new sections.
- [Spatial modeling](spatial_prediction_model_V2/README.md): pooled models, spatial feature selection, treatment-specific regression, conditional permutation testing, and saved predictors.
- [Interpretation](prediction_interpretation_model/README.md): signed feature weights and biological-theme summaries from a completed spatial run.
- [New-sample scoring](spatial_transfer_inference_model/README.md): saved-reference transformations and weighted scores, kept separate from fitted regression predictions.
- [Upstream model training](model_training/README.md): optional full rebuild from response-linked expression and histology data.

Run commands from the repository root. Installation and the deterministic software smoke test are described in the [root README](../README.md).

“Teacher” in existing filenames means a treatment-specific response estimate transferred from the expression and/or histology model. The modeled residual equals the fused estimate minus its treatment prior.

Full spatial modeling uses the maintained Step 09 with 1,000 within-treatment permutations and five repeated splits. Historical schema fields remain compatible; no earlier spatial-model outputs are required.
