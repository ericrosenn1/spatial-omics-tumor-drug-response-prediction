# Teacher builder configs

This folder contains YAML configuration files for the governed `teacher_builder` workflow.

## Files

```text
visium_teacher_builder_governed_full.example.yaml
visium_teacher_builder_governed_smoke_test.example.yaml
```

`visium_teacher_builder_governed_full.example.yaml` is the full governed run configuration template. It should be copied to a local config and edited for the user's cohort, input paths, model artifacts, and output location.

`visium_teacher_builder_governed_smoke_test.example.yaml` is a smoke-test configuration template for checking that the workflow can execute on a limited sample subset.

## Local setup

For a new machine, copy an example config to a local config and edit paths there:

```powershell
Copy-Item .\visium_teacher_builder_governed_smoke_test.example.yaml .\visium_teacher_builder_governed_smoke_test.local.yaml
Copy-Item .\visium_teacher_builder_governed_full.example.yaml .\visium_teacher_builder_governed_full.local.yaml
```

Then edit the `.local.yaml` file to point to local data, model artifacts, and output folders.

Local config copies should not be committed to GitHub.

## Paths to review

Before running, review and update paths for:

```text
project_dir
pipeline_dir
spatial feature table
spatial feature manifest
metadata table
processed Visium sample location
expression teacher/model outputs
histology teacher/model outputs
output_root
```

Large input data and generated outputs are expected to live outside GitHub or under ignored local output folders.

## Recommended run commands

From the `teacher_builder` folder:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Config .\configs\visium_teacher_builder_governed_smoke_test.local.yaml `
  -StartAt 1 `
  -StopAt 6
```

For the full run:

```powershell
.\run_teacher_builder_governed.ps1 `
  -Config .\configs\visium_teacher_builder_governed_full.local.yaml `
  -StartAt 1 `
  -StopAt 6
```

## GitHub policy

Commit example configs and durable documentation. Do not commit machine-specific `.local.yaml` files, generated outputs, logs, archives, model artifacts, whole-slide images, h5ad files, or large derived tables unless they are explicitly curated as reviewer handoffs.
