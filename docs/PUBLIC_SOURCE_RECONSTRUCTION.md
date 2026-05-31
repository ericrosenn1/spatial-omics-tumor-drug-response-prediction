# Public Visium Data Staging

This repository does not store the full public Visium data folders in Git. Instead, it provides a curated manifest and a root-level staging script that help users create a visible local input layout before running the spatial feature identification pipeline.

In this document, staging means downloading or reusing public source files, caching them under a local Visium root, and copying the files into stable `SAMPLE_####` cohort folders. It is not an exact recreation of every historical local helper file or audit artifact.

## Tracked Staging Manifest

```text
data_manifest/public_visium_cohort_staging_manifest.tsv
```

The manifest contains one row per public Visium input file to stage. It preserves the stable internal sample IDs `SAMPLE_0000` through `SAMPLE_0102`, including `SAMPLE_0049` as part of the candidate input cohort. Historical helper files such as local inventories, old build scripts, test reports, and tree snapshots are not active public input rows.

Manifest columns include source accessions, source acquisition URLs (`source_download_url`), source archive type, file roles, raw-cache paths, final cohort paths, archive member paths for compressed or archive-backed rows, and factual notes.

## Staging Script

```text
scripts/download_and_reconstruct_public_visium_sources.py
```

The filename is retained for compatibility, but the current behavior is public Visium data staging.

Typical local output layout:

```text
Visium_samples/
  raw_visium_new/
  visium_cohort_clean/
  public_visium_staging_inventory.tsv
  public_visium_staging_summary.txt
```

Use `Visium_samples/visium_cohort_clean` as the `input_root` for `spatial_feature_identification_pipeline/` after staging, unless you already have a compatible Visium input layout.

## Commands

Dry run:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage `
    --dry-run
```

Real staging run:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage
```

Skip the large TLS Zenodo archive while staging the other public files:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --download `
    --stage `
    --skip-zenodo
```

If neither `--download` nor `--stage` is supplied, the script defaults to both actions.
Validate source URLs without downloading full files:

```powershell
python scripts/download_and_reconstruct_public_visium_sources.py `
    --repo-root "YOUR_PROJECT_ROOT" `
    --visium-root "YOUR_PROJECT_ROOT\Visium_samples" `
    --manifest data_manifest\public_visium_cohort_staging_manifest.tsv `
    --validate-urls `
    --max-url-checks 25 `
    --skip-zenodo
```

## GEO And Zenodo Sources

GEO rows are downloaded from each row's `source_download_url` into `raw_visium_new/`. Direct files are copied as-is; gzip-compressed files are decompressed during staging; GEO tar archive rows are extracted to a deterministic raw-cache folder before staging. Files are then staged into `visium_cohort_clean/SAMPLE_####/`.

TLS_VISIUM_USZ rows use Zenodo record `14620362` / DOI `10.5281/zenodo.14620362`. The script uses the Zenodo REST API to identify `TLS_VISIUM_USZ.zip`, verifies the MD5 checksum when Zenodo provides one, extracts the ZIP under the raw cache, and stages KC1, KC2, KC3, LC1, LC2, LC3, LC4, and LC5 into `SAMPLE_0095` through `SAMPLE_0102` according to the manifest.

The script does not scrape the Zenodo HTML page.

## Generated Metadata And Inventory

The staging script may generate minimal `metadata.json` files under staged `SAMPLE_####` folders. These files contain only factual provenance derived from the manifest, such as sample ID, public source dataset, accessions, source sample label, source type, script name, and notes.

The script does not invent or infer biological metadata, clinical labels, cancer types, quality scores, response labels, or biological interpretations.

The generated inventory file contains one row per staged file with:

```text
sample_id, file_role, source_type, source_url, raw_cache_path, cohort_path, status, bytes_written_or_existing, checksum_status, notes
```

The generated summary file reports counts by sample, source dataset, file role, and status.

## Git Policy

Large raw data, staged cohort folders, generated inventories, generated summaries, extracted archives, and ZIP files remain local-only and are excluded from Git. The tracked source package contains the script, this documentation, and the curated staging manifest.