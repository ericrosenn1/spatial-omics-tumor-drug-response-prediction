# Output and QC changes

This record separates changes already present in the completed corrected analysis (A) from repository regression checks (B). Treatment keys are preserved in full. Conditional development results, independent evaluation, fitted residual predictions, and signed-alignment displays are separate quantities.

## A. Former baseline versus completed corrected analysis

The teacher keeps **34,881 sample–profile rows, 102 sections and 374 profile keys**. Expression-only rows changed from 1,122 to 471, with 651 restored expression/histology overlaps; 33,759 histology-only rows remain. Fused probability changed in 1,122 rows (maximum absolute change 0.384716496); recorded priors did not change. Residuals have 1,125 exact serialized differences, of which 1,122 exceed 1e-12. The extra three are floating-point representation differences. Display-name capitalization also changed; exact treatment keys did not.

| Stage | Former baseline | Corrected | Shared keys |
|---|---:|---:|---:|
| Eligible profiles |291|297|291|
| Strict registry features |139|135|135|
| Final development predictors |30|30|19|
| Conditional candidates/accepted profiles |27|27|15|

Feature counts refer to different stages: the original full training extraction table has 102 rows and 823 columns including metadata; its feature-filter audit has 720 candidate rows, 661 retained. The supplied numeric handoff has 661 features, the broad V2 pool 610, the corrected registry 135, and each final model 60 selected features. The five-sample external raw export has 633 columns including metadata; it is not the training feature count.

Historical production validation used 100 permutations. Historical 09B and corrected conditional validation each used 1,000 permutations with five splits; they remain distinct runs. Old 09B has stored metrics but no held-out vectors, so its aggregation can be checked but prediction-level metrics cannot be reconstructed. Corrected validation preserves 27 conditional acceptances; separate grouped discovery selected no independently supported model. Changes in metrics/P/q are not isolated expression effects: teacher repairs, feature and candidate selection, original task-index seeds, permutation depth, and reporting definitions also differ.

The signed-feature tables each contain 1,620 rows, with 888 shared profile–feature keys and 111 sign changes among them. Theme rows changed from 252 to 243, with 138 shared keys and seven sign changes. External alignment, contributions and coverage consequently differ. The corrected gain comparison accounts for selection frequency and uses matched normalized gain; it does not equate gain with SHAP or with explained variance. Higher/lower teacher-residual association labels replace sensitivity/resistance language. Missing evidence, including old fitted-model bundles, is unavailable rather than a scientific zero.

Three historical QC decisions changed. The sample-scoring check now expects the 93-section union supported by the retained scoring profiles, replacing the old expectation of 102; the observed count remains 93 (warn to pass). Minimum feature-effect coverage changes from 1.0 to 0.183 for the endometrial example and 0.2 for the ovarian batch (pass to warn), with the same preferred threshold of 0.80. These reflect explicit missingness handling and the supported-cohort contract; they are separate from expression scoring.

## B. Completed corrected analysis versus candidate execution

Candidate-source execution using the recorded research environment reproduced all 150 fitted predictions, 9,150 estimator contributions, 135 alignment rows and 8,100 alignment feature contributions exactly, including parsed values, NA masks, identities and schemas. PIM steps 01–06 and transfer steps 03–05 regenerated numeric tables, readable explanations and QC from the same corrected sources. Saved numerical audits rechecked 135,135 conditional split records and 2,567,565 held-out predictions, plus 57,598 grouped outer predictions, without refitting models or repeating permutation training.

The same five external samples were also scored in the isolated, dependency-consistent environment using NumPy 1.26.4 and SHAP 0.49.1, versus research versions 2.4.4 and 0.51.0. The saved predictions, estimator contributions, fitted reference transforms, alignment scores, and feature/theme contributions match both protected results and the research-environment candidate exactly at zero numerical tolerance. scikit-learn 1.8.0 and XGBoost 3.2.0 were unchanged. This verifies saved inference, not equivalence of new training under different environments.

**No fixed-run scientific value, model membership, validation decision or QC pass/warn/fail decision changed.** Reporting differences are explicit:

- Public teacher path metadata became repository-relative: 1,122 expression-model paths and 34,410 image paths. Every other cell string, key, NA mask, row order and column is identical to the corrected authority. The three matching handoff resources are supplied together.
- Regeneration adds four existing `matched_gain_*` report fields to a stale source-schema inventory. Its QC row-count observation changes from 3,224 to 3,228; the `>0` criterion and pass decision are unchanged.
- Output/card locations and absolute versus relative report paths differ. ZIP sizes and package-file counts differ because fresh selective regression packages omit historical orchestrator logs and previous package reports; numerical core-file checks remain unchanged. Both external packages retain two coverage warnings and zero failed checks.
- Execution-source snapshots permit audits of historical checkpoints after Git normalizes line endings. All historical execution SHA256 values remain strict; the current auditor is identified separately. This does not authorize resuming a run with changed model code.

Real bundles and raw inputs used here are externally supplied resources, not implied public downloads. Raw Visium extraction and clustering/QC repairs are a separate execution review; the fixed feature-table comparisons above do not assert raw-input equivalence. Manuscripts and manuscript-derived figures were outside this publication task.

### Separate real-input execution and new failure checks

A fresh extraction of the externally supplied GSM7019836 sample ran spatial steps 01–09 and the saved prediction/alignment commands. It retained the same 795 spots, 17,572 genes, counts, processed expression values and coordinates as the historical extraction. Recorded gene-set source, gene support, scoring settings and all six successful scoring-method statuses agree. Required clustering now succeeds with five clusters; the historical artifact recorded a missing-clustering-library error and substituted one cluster. All 106 underlying method-score columns agree; 23 refined axis-score columns differ. The existing refinement formula includes cluster smoothing, which provides a specific route from corrected clustering to changed spatial measurements. The runner now returns failures, and required clustering errors terminate processing instead of producing successful-looking outputs. These are new execution/QC checks.

This single-sample extraction differs from the historical four-sample batch and uses a dependency-consistent runtime. Historical source/package provenance is incomplete, so individual causes of every feature difference are not isolated. Of 445 common numeric fields, 310 differ beyond relative tolerance 1e-10/absolute tolerance 1e-12; 127 numeric missingness changes are recorded separately. All 30 predicted residuals changed (maximum absolute difference 0.0247167572), as did all 27 signed alignments (maximum 0.3945774316). Raw-input numerical equivalence is **not established**; no frozen expected values or fitted models were replaced.

The new route passed 102 endpoint checks, including reload, ordering, batch invariance at the fitted-reference/inference interface, missing-column rejection and identifier collisions. Its 99 reviewed absent fields remain NA with no zero fill. Observed model-feature coverage is 48.33–75.00%; observed alignment weight is 26.29–69.57%. Missingness reasons are bound to the new source and checked against hotspot support, observed labels and motif tables. These are tested software outputs with explicit limitations, not new predictive-accuracy evidence.
