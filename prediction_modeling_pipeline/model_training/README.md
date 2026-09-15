# Upstream expression and histology response models

Rebuild treatment-response models from response-linked source data. This route is optional: the [included development response-target package](../teacher_builder/README.md) allows spatial model development without training either upstream model.

## Inputs and environment

The source data are not bundled. Supply expression/response tables and ordered Ensembl genes for the expression model; patient/treatment labels, clinical tables and whole-slide images for histology. Access and redistribution conditions are determined by each data provider.

Install the root workflow environment first, then the optional [training dependencies](requirements-model-training.txt). Histology additionally needs PyTorch/torchvision matched to the intended CPU/CUDA environment, OpenSlide and its native runtime. Run `python -m pip check` after installation.

## Expression workflow

Copy [the example configuration](expression_response_model_v2/configs/expression_response_model_v2.example.yaml) to a local YAML file and set the project root and source paths. From the repository root:

```powershell
& prediction_modeling_pipeline/model_training/expression_response_model_v2/run_expression_response_model_v2.ps1 -Config YOUR_CONFIG_PATH -StartAt 0 -StopAt 5 -Python YOUR_PYTHON_EXECUTABLE
```

Steps harmonize treatment names, validate inputs, resolve case–treatment records, fit calibrated classifiers, assess reliability and score Visium expression. Each classifier retains its imputer, variance filter, scaler, PCA, gene order and calibrator.

Source expression is log(1+TPM); Visium transfer is log(1+CPM) pseudobulk from retained raw counts. These are distinct normalizations, preserved by design. Cross-validation is grouped by case and all preprocessing is fitted within training folds. The pooled out-of-fold calibrator's reported calibration metrics are descriptive, not an independent calibration evaluation.

## Histology workflow

Copy [the histology configuration](histology_response_model_v2/configs/histology_response_model_v2.example.yaml) and supply source locations:

```powershell
& prediction_modeling_pipeline/model_training/histology_response_model_v2/run_histology_response_model_v2.ps1 -Config YOUR_CONFIG_PATH -StartAt 0 -StopAt 9 -Python YOUR_PYTHON_EXECUTABLE
```

The workflow builds case labels, inventories slides, extracts/filter tiles, builds patient-disjoint partitions, trains treatment-only/image-only/image-plus-treatment models, runs image controls and records reliability. The image-plus-treatment model supplies histology response estimates; controls determine warning attenuation during response-target construction.

Preserve patient grouping across all slides/tiles. Do not substitute random tile-level splits. The example uses ResNet18, 224-pixel model inputs, a 32-dimensional treatment embedding and 256-unit hidden layer. Tiling uses 256-pixel patches and source-specific tissue thresholds.

After both pipelines, configure and run [response-target construction](../teacher_builder/README.md). The source-inspection [runbook](RUNBOOK.md) checks configuration and compilation but is not evidence that neural training completed.


