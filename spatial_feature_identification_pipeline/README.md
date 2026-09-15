# Spatial feature extraction

Convert Visium counts, barcodes, registered coordinates and histology metadata into section-level biological and spatial measurements.

## Inputs and configuration

Each sample must have a stable `SAMPLE_*` folder with supported 10x-style matrix, barcode and gene files (or the supported processed AnnData input), spatial positions, and required image/scale metadata. Preserve accession-to-sample mappings. Never identify a specimen by a reused staging folder name alone.

The [public staging guide](../docs/PUBLIC_SOURCE_RECONSTRUCTION.md) describes supported source layouts.

From the repository root:

```powershell
Copy-Item spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.example.yaml spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.local.yaml
# Edit input_root and output_root in the local file.
python spatial_feature_identification_pipeline/run_pipeline.py --config spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.local.yaml --start 01 --end 10
if ($LASTEXITCODE -ne 0) { throw "Spatial extraction failed" }
```

Relative paths resolve against the calling directory. The runner's `--start`/`--end` select stages; unused YAML section flags do not select them. Use `--dry-run` to inspect commands. Individual stage CLIs/constants define scientific parameters.

## Stages

| Steps | Outputs and behavior |
|---|---|
| 01–02 | Validate inputs; barcode matching, spot/gene QC, normalization, full normalized expression, HVGs, PCA, neighbors, Leiden and UMAP |
| 03–04 | Merge sample summaries; section-level program transformations |
| 05 | Structural, functional and metabolic scores; spatial/community refinement and overlapping labels |
| 06–09 | Accessibility, hotspots, context summaries, regions, pairwise spatial relationships and gradients |
| 10 | Filter and construct the development model-ready feature table |
| 11–12 | Optional overlays and figures |

QC retains spots with >=500 counts, >=200 genes and <=25% mitochondrial counts, then removes genes detected in fewer than three retained spots. Normalization is 10,000 counts per spot followed by log1p. The count layer and full normalized matrix are preserved. Dimensionality reduction uses up to 3,000 HVGs, 30 PCs, 15 neighbors and Leiden resolution 0.6. Required clustering failure or insufficient PCA dimensions terminates the stage; failed samples remain recorded in error reports.

Program scoring combines mean expression, gene-wise percentile ranks, UCell and signature/Hallmark/Reactome variation scores. Component weights and unavailable-component handling are defined in Step 05. Development gene sets use MSigDB 2023.1.Hs. Reproducing a run requires its recorded library and pathway mapping. The explicit `fallback_curated_lite` option is a different curated library used by the external extraction; it must not be silently substituted for development libraries.

Step 10 excludes features with >80% missing measurements or fewer than two distinct finite values and retains explicit missingness. The included development table has 661 features; counts from new cohorts are not forced to match it.

## New samples

Extract new samples through Step 09 only. Its cumulative `output_09_build_motif_tables/slide_features_with_motif_tables.csv` is the raw-feature input to [saved-reference prediction](../prediction_modeling_pipeline/spatial_prediction_model_V2/README.md). Do not derive new cohort ranks or a new selected feature set by fitting Step 10 on an external batch.

Coordinate distances retain the supplied image-coordinate units; graph depth is measured in neighbor steps. Cross-source physical calibration is not established merely by combining feature tables.

Full raw data and extraction outputs are not distributed. The [included modeling package](../prediction_modeling_pipeline/teacher_builder/README.md) contains only the matching development numeric feature table, ordered feature manifest and response targets.
