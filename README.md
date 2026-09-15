# A Visium Transcriptomics Workflow Linking Tumor Spatial Features to Treatment Response

This repository accompanies a computational workflow that converts 10x Genomics Visium expression, tissue-coordinate, and histology inputs into quantitative section-level descriptions of tumor spatial architecture, then relates those features to treatment-specific response estimates transferred from independently trained expression and histology models. The workflow was developed because public Visium tumor cohorts generally do not include matched longitudinal treatment-response labels.

For each treatment or recorded treatment profile, a baseline response prior is combined with reliability-weighted estimates from the modalities available for each section. Subtracting the corresponding prior yields a response residual, allowing downstream treatment-specific models to focus on section-level variation relative to that treatment-specific baseline. The models are evaluated with held-out splits and permutation-based null models and summarized through signed feature and biological-theme interpretations. These targets and outputs are model-derived research quantities, not observed outcomes in the Visium sections or clinical predictors.

## Study overview

| Study element | Count |
|---|---:|
| Development Visium sections | 102 |
| Public data collections | 12 |
| Tumor-type labels | 11, including one unresolved/unknown label |
| Retained spatial features | 661 |
| Section-treatment pairs | 34,881 |
| Treatments or recorded treatment profiles represented | 374 |
| Final fitted treatment models | 30 |
| Models advanced to conditional permutation testing | 27 |
| Within-profile permutations per tested model | 1,000 |
| Additional Visium sections | 5 |

Treatment keys preserve the full recorded treatment profile. A key containing several components does not by itself establish simultaneous administration or a named clinical regimen.

## Workflow

```text
Visium expression + tissue coordinates + histology
                         |
                         v
              Spatial feature identification
                         |
                         v
        Section-level tumor architecture representation

Independent expression-response model ----\
                                            +--> Reliability-weighted fused teacher estimate
Independent histology-response model ------/

Fused teacher estimate and treatment-specific response prior
                         |
                         v
       Section-treatment residual target (estimate - prior)

Section-level architecture representation + response residual target
                         |
                         v
             Treatment-specific spatial modeling
                         |
                         v
                Conditional permutation testing
                         |
                         v
          Feature and biological-theme interpretation
                         |
                         v
            Application to additional Visium sections
```

## Spatial representation

The spatial feature pipeline converts spot-level measurements into one architecture vector per Visium section. Its 661 retained numerical features describe structural, functional, and metabolic programs; the abundance and organization of connected tissue regions; pairwise distance, overlap, and adjacency; hotspot abundance and fragmentation; tumor-border and depth relationships; accessibility; and spatial gradients. The representation preserves complementary biological axes instead of reducing each spot or section to a single label.

## Response-teacher construction

Expression-response and histology-response models are trained separately on independent response-linked source data. They provide treatment-specific estimates for the Visium sections, while treatment-specific response priors calculated from the response-model source data provide the baseline expected response. Available modality estimates are weighted by model reliability and section-level confidence before fusion. The spatial-modeling target is then calculated as:

```text
response residual = fused teacher response estimate - treatment prior
```

This adjustment reduces the dominance of treatment identity and directs the spatial analysis toward section-level variation relative to the treatment-specific baseline. The teacher table contains 33,759 histology-only pairs, 471 expression-only pairs, and 651 pairs supported by both modalities. These transferred estimates are not observed treatment outcomes from the Visium sections.

## Spatial-response modeling and validation

The analysis first compared pooled response-probability and pooled residual models, then used the prior-adjusted target for treatment-specific regression. The treatment-specific analysis retained 135 candidate spatial predictors. Of 374 represented treatment profiles, 297 met the development analysis requirements for sample size and target variation; 30 final treatment-specific models were fitted, and 27 advanced to conditional permutation testing.

Each of the 27 selected development models was compared with label-shuffled null evaluations using 1,000 within-profile permutations and five repeated 80:20 train/test splits. All 27 satisfied the predefined conditional permutation criteria. Because predictor identification and candidate selection used the development cohort, these results are conditional development evidence rather than independent validation of the complete discovery procedure.

A separate evaluation kept learned preprocessing and feature selection inside grouped training partitions. In that analysis, adding spatial relationships did not improve median section-level prediction error, and a separate discovery partition retained no treatment model under the predefined selection criteria. The repository therefore distinguishes conditional development models, independently supported models, fitted residual predictions, and signed alignment scores.

## Biological interpretation

The interpretation layer combines SHAP-based feature importance with signed feature-residual associations to identify features associated with higher or lower teacher response residuals for each treatment profile. Feature effects are aggregated into biological themes and reported through treatment-specific feature tables, theme summaries, and interpretation profiles. These associations are descriptive and hypothesis-generating; they do not establish causal mechanisms or clinical treatment effects.

## Application to additional Visium sections

The fitted workflow was applied without rebuilding the treatment models to five additional sections:

- `NYU_UCEC3_Vis` (endometrial tumor)
- `NYU_OVCA1_Vis` (ovarian tumor)
- `NYU_OVCA3_Vis` (ovarian tumor)
- `GSM7019835_OCCR2v` (ovarian tumor)
- `GSM7019836_OCCS3v` (ovarian tumor)

All 30 fitted regressors were applied successfully to all five sections, producing 150 fitted residual predictions. The 27 interpreted profiles were also applied to all five sections, producing 135 signed weighted-feature alignment scores. Feature coverage and missingness were retained explicitly. Public accessions for the two NYU ovarian sections remain unresolved, so none are assigned here. These outputs demonstrate application to additional Visium sections; they do not constitute prospective or observed treatment-response validation.

## Repository structure

```text
.
├── spatial_feature_identification_pipeline/
├── prediction_modeling_pipeline/
│   ├── model_training/
│   ├── teacher_builder/
│   ├── spatial_prediction_model_V2/
│   ├── prediction_interpretation_model/
│   └── spatial_transfer_inference_model/
├── data_manifest/
├── docs/
├── scripts/
├── project_profile.example.yaml
└── README.md
```

- [`spatial_feature_identification_pipeline/`](spatial_feature_identification_pipeline/) processes Visium inputs and builds section-level spatial feature tables.
- [`prediction_modeling_pipeline/model_training/`](prediction_modeling_pipeline/model_training/) contains the independent expression-response and histology-response workflows.
- [`prediction_modeling_pipeline/teacher_builder/`](prediction_modeling_pipeline/teacher_builder/) combines response priors and available modality estimates into prediction-ready teacher targets.
- [`prediction_modeling_pipeline/spatial_prediction_model_V2/`](prediction_modeling_pipeline/spatial_prediction_model_V2/) implements pooled and treatment-specific residual modeling, permutation testing, grouped evaluation, and fitted-model export.
- [`prediction_modeling_pipeline/prediction_interpretation_model/`](prediction_modeling_pipeline/prediction_interpretation_model/) produces signed feature and biological-theme interpretations.
- [`prediction_modeling_pipeline/spatial_transfer_inference_model/`](prediction_modeling_pipeline/spatial_transfer_inference_model/) applies fitted references and interpretation profiles to additional Visium sections.
- [`data_manifest/`](data_manifest/) records public Visium source and staging information.
- [`docs/`](docs/) contains execution, reconstruction, artifact, and validation-scope documentation.
- [`scripts/`](scripts/) contains repository-level public-data staging and seeded execution utilities.

## Data

The development cohort was assembled from public source datasets; access and redistribution remain subject to each source's terms. Large raw Visium files, whole-slide images, processed H5AD files, complete expression and histology training resources, and most trained model artifacts are not stored in GitHub. The [public Visium staging manifest](data_manifest/public_visium_cohort_staging_manifest.tsv) records accessions, source files, and stable internal identifiers, and the [public source reconstruction guide](docs/PUBLIC_SOURCE_RECONSTRUCTION.md) documents the staging utility.

103 candidate sample folders were represented in staging; 102 sections were retained in the development analysis.

A compact three-file downstream handoff is tracked in [`prediction_modeling_pipeline/teacher_builder/`](prediction_modeling_pipeline/teacher_builder/): [`precomputed_governed_fused_teacher_table_102samples.tsv.gz`](prediction_modeling_pipeline/teacher_builder/precomputed_governed_fused_teacher_table_102samples.tsv.gz), [`model_input_numeric.csv`](prediction_modeling_pipeline/teacher_builder/model_input_numeric.csv), and [`feature_manifest.csv`](prediction_modeling_pipeline/teacher_builder/feature_manifest.csv). Their dimensions and SHA-256 hashes are recorded in [`precomputed_handoff_manifest.json`](prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json). Full reconstruction still requires the public/source data and the relevant upstream modules.

## Installation

Python 3.12 is tested for the maintained execution path. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-reviewer.txt
if ($LASTEXITCODE -ne 0) { throw "Installation failed: $LASTEXITCODE" }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed: $LASTEXITCODE" }
```

Install [`requirements-notebooks.txt`](requirements-notebooks.txt) only when building the Jupyter Book documentation. Upstream expression and whole-slide training modules have additional requirements described in their module documentation.

## Reproducing the analysis

### Reproduce downstream analysis from the precomputed handoff

The tracked handoff permits downstream spatial modeling without rebuilding the expression-response and histology-response source models. First validate its manifest and extract the three matching files into a new output directory:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py `
    --manifest prediction_modeling_pipeline/teacher_builder/precomputed_handoff_manifest.json `
    --output local/precomputed_handoff
if ($LASTEXITCODE -ne 0) { throw "Handoff verification failed: $LASTEXITCODE" }
```

Then run the spatial modeling workflow with the manuscript validation settings:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py `
    --mode full `
    --handoff-root local/precomputed_handoff `
    --max-workers 2 `
    --full-step09-n-shuffles 1000 `
    --full-step09-n-repeats 5
if ($LASTEXITCODE -ne 0) { throw "Spatial modeling failed: $LASTEXITCODE" }
```

Use a new or empty output directory when rerunning the handoff preparation. See [execution and resource requirements](docs/REVIEWER_EXECUTION.md) for the seeded smoke path, resource expectations, downstream predictor export, and artifacts that are not distributed with the repository.

### Rebuild from public/source data

Reconstruct or stage the public Visium cohort using [`docs/PUBLIC_SOURCE_RECONSTRUCTION.md`](docs/PUBLIC_SOURCE_RECONSTRUCTION.md), then follow the module documentation in analysis order:

1. [Spatial feature identification](spatial_feature_identification_pipeline/README.md)
2. [Expression and histology response modeling](prediction_modeling_pipeline/model_training/README.md)
3. [Response-teacher construction](prediction_modeling_pipeline/teacher_builder/README.md)
4. [Spatial-response modeling](prediction_modeling_pipeline/spatial_prediction_model_V2/README.md)
5. [Biological interpretation](prediction_modeling_pipeline/prediction_interpretation_model/README.md)
6. [Application to additional sections](prediction_modeling_pipeline/spatial_transfer_inference_model/README.md)

The [project profile template](project_profile.example.yaml) lists the principal local paths and expected handoffs. It is a configuration guide rather than an executable workflow definition.

## Validation and reproducibility

- Expression-response cross-validation is grouped by source case, and histology-response splitting is performed at the patient level where applicable.
- The conditional treatment-specific analysis uses five repeated 80:20 splits and 1,000 within-profile permutations; random seeds and run settings are recorded with outputs.
- The separate evaluation keeps preprocessing, selection, and related learned steps inside grouped training partitions.
- The precomputed handoff preparation verifies file hashes, dimensions, key uniqueness, ordered features, numerical values, missingness, and the residual identity against its tracked manifest.
- The seeded smoke workflow exercises software execution, restart behavior, teacher fusion branches, model fitting, and inference interfaces with small fixtures.

A passing smoke test shows that the maintained software path executes and produces the expected file structure. It does not reproduce the full scientific analysis, validate biological claims, or substitute for outcome-linked evaluation.

## Manuscript

**A Visium Transcriptomics Workflow Linking Tumor Spatial Features to Treatment Response**

Eric Rosenn<br>
Department of Biomedical Engineering, New York University Tandon School of Engineering

Status: Manuscript in preparation / working manuscript draft

## Citation

If you use this workflow before a formal publication record is available, please cite the repository and manuscript title above. A formal citation will be added upon publication.

## License

Repository licensing is being finalized.

## Contact

Eric Rosenn<br>
GitHub: [@ericrosenn1](https://github.com/ericrosenn1)
