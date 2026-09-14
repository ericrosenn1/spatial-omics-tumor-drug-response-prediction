# Reviewer execution and resource requirements

Run the root README installation and seeded smoke commands from a fresh checkout using Python 3.12. The compatible constraints apply to the isolated reviewer environment; they do not change a research environment. `python -m pip check` must pass. The smoke output records the interpreter, imported source paths and hashes, dependency versions, command exit codes, numerical checks and explicit exclusions in `smoke_report.json`.

The CPU-only fixture uses one model/numerical thread, 16 training and two external sections, with 48 spots and 256 genes per section. Its measured execution was about 11 seconds in the tested Windows environment; installation is separate. Real spatial extraction, complete cohort training and 1,000-permutation validation require larger, separately planned compute budgets.

## Included reproduction inputs

The maintained teacher-builder directory includes the corrected compressed teacher, numeric spatial table and ordered feature manifest. `precomputed_handoff_manifest.json` binds all three hashes. Script16 verifies identities, duplicate keys, exact feature order, numeric/NA values and `residual = probability - prior` before extracting a new handoff. It retains all 651 both-modality, 471 expression-only and 33,759 histology-only rows. The 102-section handoff has 661 numeric features; the completed downstream supervised registry has 135 features. These are different stages.

The seeded smoke exercises raw 10x-style matrix/coordinate loading, retained-spot QC, normalization, log transformation, marker-program and neighbor refinement, saved expression base-model/calibrator scoring, all fusion modality branches, the precomputed interface, training-only feature selection, fitted XGBoost fixtures, interruption/resume, and interpretation/model-inference interfaces. Its histology stream comes from a small fitted probability fixture. It excludes image tiling/CNN training, full spatial extraction, full registry discovery, external gene-set retrieval and scientific validation. It uses three permutations and two splits; production conditional validation uses 1,000 and five.

Tests can be run after installation:

```powershell
python -m pytest tests spatial_feature_identification_pipeline/tests prediction_modeling_pipeline/teacher_builder/tests prediction_modeling_pipeline/spatial_prediction_model_V2/tests prediction_modeling_pipeline/prediction_interpretation_model/tests prediction_modeling_pipeline/spatial_transfer_inference_model/tests --ignore=prediction_modeling_pipeline/prediction_interpretation_model/tests/test_manuscript_docx.py -q
if ($LASTEXITCODE -ne 0) { throw "Regression tests failed: $LASTEXITCODE" }
```

The document-generation helper tests remain in the repository but are excluded from this publication check; no manuscript documents are generated.

## Full execution and external artifacts

The three-file handoff permits downstream training without rebuilding the expression and histology cohorts. Full original teacher training additionally requires expression matrices and response metadata, histology slides/tiles and saved or newly trained objects described by those modules. A raw Visium matrix/image is not a drug-response label. Inference on an unseen section uses fitted spatial models or the frozen interpretation atlas, without retraining the upstream response cohorts.

Trained bundles, the complete interpretation atlas, raw Visium data and full numerical archives are not distributed in this source-focused repository. The [artifact requirements](artifact_requirements.json) list checksums for the completed development bundles and reference manifests. No public download is claimed for these excluded artifacts. Supply a verified copy from its owner or reproduce training/export locally. This prerequisite is separate from the self-contained seeded smoke.

From a completed V2 run and its matching teacher handoff, the maintained exporter fits only the documented final development estimators, checks their selected features and saved gain evidence, and saves preprocessing, priors, ordered features, estimator and provenance together:

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/14_fit_predictor_bundles.py --run-root '<completed V2 run>' --teacher-root '<matching three-file handoff>' --raw-feature-table '<training Step09 slide_features_with_motif_tables.csv>' --output '<new development bundle directory>'
if ($LASTEXITCODE -ne 0) { throw "Bundle export failed: $LASTEXITCODE" }
```

`--existing-bundles` attaches the verified training reference to saved estimators without refitting. Script17 exports the actual accepted independent family from a separately audited evaluation; an empty accepted family remains empty. Full-cohort deployment fitting is separate from held-out performance. See the [evaluation and quantity contract](../prediction_modeling_pipeline/spatial_prediction_model_V2/docs/corrected_evaluation_and_prediction.md).

## Real Visium to inference

Use the spatial feature pipeline's safe example configuration with externally supplied raw matrices, gene identifiers, barcodes, spatial coordinates and required image/scale metadata. Its Steps01–09 produce the cumulative raw measurement table. Required clustering dependencies are included in the reviewer requirements; a clustering failure now terminates processing. The explicit curated gene-set fallback is available when reproducing a run that recorded that source. Do not silently switch a run's gene-set library.

For saved extraction outputs, use the [verified feature exporter](../prediction_modeling_pipeline/spatial_transfer_inference_model/docs/verified_feature_export.md). Its hash-bound mapping and retained-barcode audit distinguish repeated internal `SAMPLE_*` IDs by actual accession/source. It does not recompute measurements or attach teacher labels. Distances retain the extraction's full-resolution image-pixel units; a cross-source physical calibration is not established.

```powershell
python prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/15_predict_spatial_features.py --bundles '<verified bundle directory>' --features '<verified raw spatial features.csv>' --representation raw_reference --mode external --output '<new predictions.tsv>' --contributions '<new model contributions.tsv>'
if ($LASTEXITCODE -ne 0) { throw "Model inference failed: $LASTEXITCODE" }
```

The saved training reference supplies normalization and imputation, so adding unrelated samples does not change a sample's prediction. Required missing columns fail; explicit NA retains coverage information and uses the fitted imputer. Omitted unsupported extraction fields require a reviewed reason/evidence manifest through `--raw-missingness-manifest`; they are not blanket-filled with zero. Inspect coverage and exact `drug_key`, including long recorded treatment histories.

The prediction columns contain a fitted residual and a prior-anchored estimate where the bundle defines a unique prior. Unclipped estimates and any bounded displays remain separate. They are model-derived teacher estimates, not measured or clinically calibrated drug efficacy.

The separate `alignment_bundle.py` interface exports a completed PIM atlas, prepares raw features using that same saved training reference, and writes signed alignment, sigmoid-rescaled interpretation scores, feature/theme contributions and coverage. The all-effect transfer rule and PIM top-effect summary are distinct; estimator performance is not assigned to either scorer automatically.

## Validation scope and publication regression

The completed corrected analysis contains 30 development predictors and 27 profiles accepted by its fixed-registry conditional permutation test. The separate leakage-controlled discovery family was empty, so independently supported deployment bundles are empty. These counts are verified result memberships, not smoke expectations or clinical validation claims.

Publication regression reuses saved numerical outputs, checks the actual five external samples with reloaded models/atlas, and regenerates reporting/QC into new directories. It does not retrain the cohort or rerun all 1,000 permutations. A separate externally supplied real-Visium test checks raw extraction. Exact comparisons distinguish new extraction/QC failure handling from fixed-result reproduction; see [output and QC changes](OUTPUT_AND_QC_CHANGES.md).

The historical 100-permutation production result, old manuscript-specific Step09B, corrected 1,000-permutation conditional result and grouped evaluation retain separate provenance. Step09 now has one maintained durable execution route. Hash-verified original execution snapshots permit audits of archived results without pretending the current code bytes produced them. New runs save those source snapshots automatically; incompatible tasks cannot be resumed.

No manuscript, presentation or manuscript-derived figure is changed by this publication. A later editorial task starts from the pre-Codex manuscript. No license has been selected; the existing authorship and license notice remain unchanged.
