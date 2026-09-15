# Treatment-specific response targets

This module combines expression and histology response estimates for each section–treatment pair. Existing filenames use “teacher” for these transferred targets. Targets are not measured Visium treatment outcomes.

## Included development package

The following resources are distributed together and bound by [precomputed_handoff_manifest.json](precomputed_handoff_manifest.json):

| Resource | Content |
|---|---|
| `precomputed_governed_fused_teacher_table_102samples.tsv.gz` | 34,881 unique section–treatment pairs; 374 recorded treatment profiles |
| `model_input_numeric.csv` | 102 development sections and 661 numeric spatial features |
| `feature_manifest.csv` | The exact ordered list of 661 features |

There are 33,759 histology-only, 471 expression-only and 651 both-modality pairs. Missing feature measurements remain missing. Treatment keys retain their recorded components; they do not establish simultaneous administration.

From the repository root:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py --manifest prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json --output local/precomputed_handoff
if ($LASTEXITCODE -ne 0) { throw "Response-target preparation failed" }
```

Use a new or empty output folder. Preparation verifies source hashes, duplicate identifiers, sample matching, ordered features, numerical values and residual arithmetic. It does not generate targets for new Visium samples. Use [saved spatial models](../spatial_prediction_model_V2/README.md) for those samples.

## Rebuild from expression and histology inputs

Copy [the full example configuration](configs/visium_teacher_builder_governed_full.example.yaml) to a `*.local.yaml` file and configure the response-linked source tables, fitted expression models, fitted histology model and audit, raw/processed Visium inputs, and matching spatial feature table/manifest. These upstream resources are not included.

```powershell
& prediction_modeling_pipeline/teacher_builder/run_teacher_builder_governed.ps1 -Config YOUR_CONFIG_PATH -StartAt 1 -StopAt 6 -Python YOUR_PYTHON_EXECUTABLE
```

Steps 01–06 validate inputs and construct priors, score expression, score histology, fuse estimates, build matching modeling inputs, and run QC. The final three-file modeling input is under `05_prediction_ready_teacher/`. Inspect `06_teacher_qc/` before fitting spatial models.

## Fusion contract

For treatment prior pi and modality probabilities pE and pH:

```text
wE = clip(reliabilityE * 2*abs(pE - 0.5) * controlE, 0, 1)
wH = clip(reliabilityH * 2*abs(pH - 0.5) * controlH, 0, 1)
fused = clip(pi + wE*(pE - pi) + wH*(pH - pi), 0.01, 0.99)
residual = fused - pi
```

An unavailable modality has zero weight. Histology control warnings apply a factor of 0.50; reliability comes from the model audit, not a hardcoded manuscript result. A histology-only row therefore remains prior-anchored and is not simply its raw model probability.

Exact treatment priors take precedence over support-weighted component priors and the overall response-rate fallback. Prior sources, availability, reliability and control flags remain attached to each pair. No missing biological feature is silently replaced with zero during this handoff.

