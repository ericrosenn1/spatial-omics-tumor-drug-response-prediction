# Execution and resource requirements

Use Python 3.12 and the root `requirements.txt` in an isolated environment. The compatibility constraints retain NumPy 1.26 because PyUCell 0.6 requires NumPy <2. `python -m pip check` must pass. Historical installation/script filenames remain supported.

## Software smoke test

```powershell
python scripts/run_smoke.py --output local/smoke
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
```

Use a new or empty directory. The deterministic CPU fixture has 16 training and two external sections, each with 48 spots and 256 genes. It runs retained-spot processing, PCA/neighbors/UMAP/Leiden, program scoring/refinement, saved expression-model/calibrator scoring, all fusion modality branches, matching-input preparation, training-only feature selection, fitted XGBoost models, conditional-test interruption/resume, interpretation, and saved-model/weighted-score inference.

Three permutations and two splits test software execution only. No clinical labels, pretrained images or gene-set downloads are needed. CNN training, complete architecture extraction, full feature discovery and the 1,000-permutation scientific run are outside this smoke test. `smoke_report.json` records actual checks, source hashes, versions, outputs and exclusions.

## Full development and new-sample workflow

1. Supply/stage the development Visium sections and extract spatial features.
2. Prepare the matching included 102-section response-target package, or rebuild upstream response targets.
3. Run full spatial modeling and inspect output QC.
4. Generate interpretation tables; export fitted estimators and their raw-development feature reference.
5. Extract new Visium samples through Step 09 and apply the saved models and separate feature-weight scorer without refitting.

The [spatial modeling guide](../prediction_modeling_pipeline/spatial_prediction_model_V2/README.md) and [new-sample guide](../prediction_modeling_pipeline/spatial_transfer_inference_model/README.md) contain executable CLI templates.

The included package does not label arbitrary unseen samples. Replacing its development features with a new extraction requires explicit sample/feature validation and source hashes; do not relabel a new cohort as the original development set.

## Resources not distributed

Raw Visium inputs, response-linked expression/histology cohorts, trained models, raw-development reference tables and complete numerical archives are not bundled. Supply a verified trusted copy or rebuild them. [Artifact requirements](artifact_requirements.json) identify existing reference hashes; they do not claim a public download.

Full extraction, upstream training and conditional validation require substantially more time/storage than the fixture. Preserve successful stages and compatible Step 09 checkpoints. Two validation workers are a conservative default.

## Validation boundaries

Conditional permutation testing uses a fixed upstream-selected feature and treatment family. It does not measure independent discovery performance. The [evaluation contract](../prediction_modeling_pipeline/spatial_prediction_model_V2/docs/corrected_evaluation_and_prediction.md) describes the separate grouped procedure.

Fitted residual predictions, prior-anchored estimates and signed weighted feature scores are separate quantities. Neither synthetic execution tests nor external application establish clinical efficacy. Missing measurements, treatment-profile identity and feature coverage remain explicit.
