# Configs

This folder contains tracked YAML templates used to run or document the spatial feature identification pipeline.

Tracked templates:

- visium_cohort_clean.example.yaml
- spatial_feature_pipeline_full_config.example.yaml
- visium_5_sample_test.example.yaml

For a new machine, copy a tracked `.example.yaml` file to a local `.local.yaml` file and edit paths there:

```powershell
Copy-Item .\visium_cohort_clean.example.yaml .\visium_cohort_clean.local.yaml
```

Before rerunning, confirm `input_root`, processed-sample paths, reference resources, and `output_root` point to the current computer. Local `.local.yaml` files are machine-specific and should not be committed.
