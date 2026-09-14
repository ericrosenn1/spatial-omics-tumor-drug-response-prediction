# Verified feature-table export

After spatial extraction Steps01–09, the cumulative `slide_features_with_motif_tables.csv` contains one row per processed section. Processing IDs such as `SAMPLE_0000` are local to that extraction run. Supply an explicit one-to-one map to verified external section identities before inference.

The JSON source manifest is a list of extraction groups. Each group requires `mapping_path`, `mapping_sha256`, `source07_path`, `source07_sha256`, `source09_path`, `source09_sha256`, and `sample_ids` (the exact external identity list). The mapping TSV requires `internal_sample_id` and `transfer_sample_id`. Relative paths resolve beside the JSON manifest. Obtain hashes from reviewed source authorities; a newly computed hash alone does not establish sample identity.

The identity-audit TSV requires `sample_id`, `source_matrix_path`, `source_matrix_sha256`, `retained_barcode_subset`, and `duplicate_retained_barcodes`. Matrix paths can be absolute or relative to this TSV. Record actual retained-barcode checks against the original matrix, preserve accession/library metadata separately, and never infer identity from a reused processing ID.

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_transfer_inference_model\scripts\export_verified_spatial_feature_handoff.py --source-manifest '<verified extraction sources.json>' --identity-audit '<verified section identity audit.tsv>' --output '<new external raw feature table.csv>'
if ($LASTEXITCODE -ne 0) { throw "Verified feature export failed: $LASTEXITCODE" }
```

The output preserves the original measurement columns and adds the original internal ID and source provenance. It does not attach teacher labels, zero-fill missing measurements, or convert coordinate units. Euclidean distances retain the source's full-resolution image pixels unless the recorded extraction explicitly used another unit.

Use V2 `15_predict_spatial_features.py --representation raw_reference` with an externally supplied, hash-verified fitted bundle for residual predictions. Use `alignment_bundle.py prepare-raw` followed by `score` with the separately saved atlas for descriptive alignment. Missing extraction fields require reviewed per-feature reasons; the alignment adapter additionally binds those reasons to the exact raw input hash. Neither finite imputed outputs nor a successful export establishes prediction accuracy. The real fitted objects and raw Visium inputs are separate artifact prerequisites; this source command does not imply a public model download.
