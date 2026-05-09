# Public-source Visium reconstruction

This repository does not store the full Visium data folders in Git.

Instead, it provides manifests and scripts for reconstructing the expected Visium data layout from public source files.

## Expected local structure

Visium_samples/
  raw_visium_new/
  visium_cohort_clean/
  processed_samples/

## Final manifests

data_manifest/visium_public_source_reconstruction_manifest_final.tsv
data_manifest/visium_expected_cohort_files_final.tsv
data_manifest/tls_visium_usz_sample_mapping.tsv
data_manifest/tls_visium_usz_zenodo_source_rule.tsv

## Download and reconstruction script

scripts/download_and_reconstruct_public_visium_sources.py

## Notes

Most public source files are reconstructed from inferred GEO supplementary file URLs.
TLS_VISIUM_USZ is reconstructed as a Zenodo dataset-level source using DOI 10.5281/zenodo.14620362.
SAMPLE_0095 through SAMPLE_0102 map to KC1, KC2, KC3, LC1, LC2, LC3, LC4, and LC5.
Remaining local documentation screenshots and notes are not required for public reconstruction.

## Current status

Final source rows: 3244
Final expected cohort rows: 788
TLS expected rows: 64
Direct GEO expected rows: 1
Remaining needs_manual_url rows: 0
Remaining needs_manual_mapping rows: 103
Critical required Visium input files still unmapped: 0

## Dry run

python scripts/download_and_reconstruct_public_visium_sources.py --repo-root "D:\Adv_Omics_Fenyo\project" --visium-root "D:\Adv_Omics_Fenyo\project\Visium_samples" --source-manifest data_manifest/visium_public_source_reconstruction_manifest_final.tsv --expected-manifest data_manifest/visium_expected_cohort_files_final.tsv --download-raw --reconstruct-cohort --dry-run

## Real run

python scripts/download_and_reconstruct_public_visium_sources.py --repo-root "D:\Adv_Omics_Fenyo\project" --visium-root "D:\Adv_Omics_Fenyo\project\Visium_samples" --source-manifest data_manifest/visium_public_source_reconstruction_manifest_final.tsv --expected-manifest data_manifest/visium_expected_cohort_files_final.tsv --download-raw --reconstruct-cohort
