# Corrected evaluation and prediction contract

The versioned result manifest identifies the actual teacher, feature table, candidate family, numerical audit and model bundles. A historical run, a conditional permutation result and an independently evaluated procedure are separate sources of evidence. Treatment keys describe the recorded treatment profile; the delimiter does not establish concurrent administration.

| Output | Definition | Interpretation |
| --- | --- | --- |
| Teacher probability | Reliability- and confidence-weighted expression/histology estimates anchored to the recorded prior | Model-derived training label; not observed Visium response |
| Teacher residual | Fused teacher probability minus its recorded treatment prior | Regression target |
| Spatial prediction | Reloaded estimator prediction of that residual with saved preprocessing and ordered features | Fitted regression output |
| Reconstructed teacher estimate | Predicted residual plus the treatment's recorded prior, where a unique prior is defined | Unclipped estimate retained; bounded display is separate and is not calibrated efficacy |
| Signed alignment | Signed feature/theme effects combined with reference-normalized feature values | Separate derived interpretation rule |
| Rescaled alignment | The documented sigmoid transform of signed alignment | Display score; not an efficacy probability |

## Evaluation sources

Original production Step09, historical manuscript Step09B, corrected conditional Step09 and the separate grouped evaluation have distinct manifests. The corrected Step09 retains the fixed corrected registry and selected candidate set. Its 1,000 permutations and five repeated splits do not remove upstream discovery/selection leakage. It saves predictions, selected features, split membership, observed/null metrics and stable task seeds in atomic numerical blocks. Only compatible, hash-verified blocks are resumed. Undefined split metrics remain explicit; nonfinite null statistics count as exceedances rather than shrinking the permutation denominator. R-squared is 1-SSE/SST, including negative values.

Script13 starts from the full reviewed measurement pool. Grouped outer partitions contain every section from a verified patient in one partition; unresolved patient maps use conservative source blocks. Training-only preprocessing, pooled feature discovery, eligibility and treatment screening are repeated within each outer training set. A separate source-held-out discovery/evaluation split supplies the independently tested family; its joint outcome-vector permutations preserve source and availability strata. BH applies to that discovery-selected family. The exact split allocation, eligibility rules, null design and accepted set are saved with the results. No accepted count is hard-coded.

For the matched comparison, composition/program-state features must be independent of physical spatial information throughout their upstream derivation. Marginal summaries of spatially smoothed scores belong to the spatial arm. The required reviewed class manifest must cover the exact raw feature pool. Both arms use identical samples, targets, partitions and model settings. A feature-class correction can reuse a complete unchanged spatial arm only through the explicit dependency-verification path; successful numerical results are never inferred from status files alone.

## Explicit precomputed-teacher entry point

The three handoff files and their recorded authority hashes are required together. This command verifies all three source hashes, identities, exact feature-manifest order, numeric-or-missing feature values, feature/sample correspondence and residual arithmetic before creating a new derivative. Obtain hashes from the reviewed artifact manifest; recomputing hashes of an unverified same-ID table does not establish its provenance. Source bytes and explicit NA are preserved. The command does not generate labels for unseen sections.

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\16_prepare_precomputed_teacher.py --teacher '<corrected teacher.tsv.gz>' --teacher-sha256 '<recorded teacher SHA256>' --spatial-features '<model_input_numeric.csv>' --spatial-features-sha256 '<recorded spatial-table SHA256>' --feature-manifest '<feature_manifest.csv>' --feature-manifest-sha256 '<recorded feature-manifest SHA256>' --output '<new handoff directory>'
if ($LASTEXITCODE -ne 0) { throw "Precomputed handoff validation failed: $LASTEXITCODE" }
```

## Corrected conditional recovery

Use an explicitly versioned corrected V2 run with verified Steps01–08. Supply the original full candidate family. The production Step09 entry point now persists numerical results; the historical implementation remains in source for provenance. These parameters reproduce the corrected conditional design rather than the smaller smoke defaults.

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\09_label_shuffle_validate_tier1.py --run-root '<corrected V2 run>' --dataset-root '<that run\02_build_modeling_dataset>' --step05-root '<that run\05_residual_biology_registry>' --step08-root '<that run\08_curated_per_treatment_residual_models>' --output-root '<that run\09_tier1_label_shuffle_validation>' --n-shuffles 1000 --n-repeats 5 --test-size 0.2 --max-features-per-split 60 --n-estimators 80 --max-depth 2 --learning-rate 0.03 --random-state 42 --fdr-threshold 0.10 --max-workers 2
if ($LASTEXITCODE -ne 0) { throw "Conditional validation or numerical audit failed: $LASTEXITCODE" }
```

Rerunning this exact command resumes compatible completed blocks. Do not change the candidate order, source hashes or settings to bypass a failed compatibility check. A source change requires a sibling output. The command performs the numerical audit before it reports completion.

## Grouped evaluation and independent predictor export

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\13_evaluate_grouped_procedure.py --teacher '<corrected teacher.tsv>' --raw-features '<full slide_features_with_motif_tables.csv>' --metadata '<verified sample and dependency groups.tsv>' --feature-classes '<reviewed feature classes.tsv>' --output '<new grouped evaluation>'
if ($LASTEXITCODE -ne 0) { throw "Grouped evaluation failed: $LASTEXITCODE" }
$env:PYTHONPATH = (Resolve-Path '.\prediction_modeling_pipeline\spatial_prediction_model_V2\src').Path
& .\.venv\Scripts\python.exe -m spm_v2.leakage_audit --root '<new grouped evaluation>' --metadata '<verified sample and dependency groups.tsv>' --teacher '<corrected teacher.tsv>'
if ($LASTEXITCODE -ne 0) { throw "Independent numerical audit failed: $LASTEXITCODE" }
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\17_export_accepted_predictors.py --evaluation-root '<audited grouped evaluation>' --output '<new supported bundle directory>' --fit-all-data-deployment
if ($LASTEXITCODE -ne 0) { throw "Supported predictor export failed: $LASTEXITCODE" }
```

The exporter checks the complete numerical result and exports only the actual accepted set. An empty set produces an explicit empty manifest. Discovery-fitted bundles reproduce the independent held-out predictions exactly. The optional deployment refit uses all cohort sections, retains the frozen discovery-selected feature order and settings, and is stored separately; its fitted training values are not held-out predictions. Historical corrected Step07 bundles remain development models, even when a treatment key also appears in the independent family.

## Score a verified external section or batch

Run the existing spatial feature pipeline on the new section, or reuse extraction outputs whose accession, source matrix, retained barcodes and coordinate units have been verified. The raw-reference route applies the frozen training reference; adding another section to the batch cannot change normalization. No expression/histology teacher retraining is needed for this prediction route.

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe .\prediction_modeling_pipeline\spatial_prediction_model_V2\scripts\15_predict_spatial_features.py --bundles '<selected bundle directory>' --features '<verified spatial feature table.csv>' --representation raw_reference --mode external --output '<new predictions.tsv>' --contributions '<new feature contributions.tsv>'
if ($LASTEXITCODE -ne 0) { throw "Spatial prediction failed: $LASTEXITCODE" }
```

Missing required columns fail. Explicit measured-data NA uses the fitted training imputer. If an extraction stage omits a column because its computation is unsupported, `--raw-missingness-manifest` requires per-feature reason and source evidence to preserve that omission as NA. This is not blanket zero filling. Inspect observed-feature fractions and contribution/coverage tables; a finite imputed prediction does not establish external accuracy. Duplicate identities/headers and training/external identifier collisions are rejected. `heldout_evaluation_reproduction` is restricted to the recorded held-out identities.

The interpretation and transfer modules use their own explicitly selected atlas root and fitted alignment reference. Their exact scoring rule has no predictive-performance claim unless that rule itself was evaluated with training-only effects and scaling. Base-estimator metrics cannot be assigned to the alignment scorer.

## Local release boundary

Run the teacher, V2, interpretation and transfer regression suites with the project environment. Keep the versioned corrected teacher artifact and fitted models local. The source release candidate includes tests, example configuration, quantity definitions and exact result provenance; it does not replace the public teacher automatically, push a branch, or upload data/models. The closeout manifest records the tested local commands and pending publication action.
