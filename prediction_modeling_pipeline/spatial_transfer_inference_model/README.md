\# Spatial Transfer Inference Model



\## Overview



`spatial\_transfer\_inference\_model` applies a completed spatial treatment-response interpretation atlas to new Visium spatial transcriptomics samples.



The module takes a transfer-ready spatial feature table, aligns each sample to a frozen strict-feature registry, and scores how strongly each sample's spatial architecture aligns with treatment-associated sensitivity or resistance/barrier biology. It can run on a single Visium sample or on a small batch of samples.



The main output is a sample-by-treatment interpretation table with spatial alignment scores, probability-like research scores, confidence/evidence support, feature drivers, biological theme drivers, QC reports, and a final package of output files.



\## Use case



This module is useful for questions such as:



\- Which treatment signatures show the strongest favorable spatial alignment for a given Visium sample?

\- Which samples show stronger resistance/barrier-associated spatial programs?

\- Which spatial features drive each treatment-specific score?

\- Which biological themes recur across treatment signatures?

\- How do multiple Visium samples compare across the same treatment atlas?



\## Workflow position



The transfer model is the final downstream inference layer.



```text

Visium sample

→ spatial feature identification

→ transfer-ready feature handoff

→ spatial transfer inference

→ treatment-alignment tables and interpretation outputs

```



It uses two completed upstream resources:



1\. \*\*Spatial feature outputs\*\* from the spatial feature identification pipeline.

2\. \*\*Frozen treatment-feature and treatment-theme effects\*\* from the prediction interpretation model.



\## Repository structure



```text

spatial\_transfer\_inference\_model/

├── README.md

├── README.txt

├── .gitignore

├── .gitattributes

├── configs/

│   └── example\_spatial\_transfer\_inference\_model.json

├── docs/

│   ├── STEP\_ORDER.md

│   └── SOURCE\_OF\_TRUTH\_POLICY.md

└── scripts/

&#x20;   ├── 00\_run\_spatial\_transfer\_inference\_model.py

&#x20;   ├── 01\_prepare\_transfer\_inputs.py

&#x20;   ├── 02\_align\_single\_slide\_features\_to\_v2.py

&#x20;   ├── 03\_score\_transfer\_drug\_response\_alignment.py

&#x20;   ├── 04\_make\_single\_slide\_prediction\_table.py

&#x20;   ├── 05\_qc\_and\_package\_transfer\_outputs.py

&#x20;   ├── 00b\_build\_improved\_transfer\_handoff.py

&#x20;   ├── 00c\_audit\_transfer\_strict\_feature\_missingness.py

&#x20;   └── 00d\_apply\_reviewed\_zero\_fill.py

```



\## Main pipeline scripts



\### `00\_run\_spatial\_transfer\_inference\_model.py`



Main orchestrator. Runs selected steps, manages output folders, captures logs, writes run summaries, and packages final outputs.



\### `01\_prepare\_transfer\_inputs.py`



Loads the transfer feature table, standardizes sample IDs, loads the frozen strict-feature registry, and prepares input rows for alignment.



\### `02\_align\_single\_slide\_features\_to\_v2.py`



Aligns each input sample to the frozen strict spatial feature set. It records observed features, missing features, and neutral/imputed values used for transfer scoring.



\### `03\_score\_transfer\_drug\_response\_alignment.py`



Combines sample feature values with signed treatment-feature effects. It produces treatment-specific spatial alignment scores, sensitivity-supporting scores, resistance/barrier-supporting scores, feature contributions, and theme contributions.



\### `04\_make\_single\_slide\_prediction\_table.py`



Creates the readable sample-by-treatment interpretation table. This is the main table for reviewing transfer predictions.



\### `05\_qc\_and\_package\_transfer\_outputs.py`



Runs final QC checks and writes a portable output package.



\## Transfer handoff helper scripts



\### `00b\_build\_improved\_transfer\_handoff.py`



Builds a transfer-ready feature table from spatial feature outputs and maps available features to the frozen strict feature registry.



\### `00c\_audit\_transfer\_strict\_feature\_missingness.py`



Audits strict feature coverage and classifies features as observed, recovered, absence-like, missing, or unavailable.



\### `00d\_apply\_reviewed\_zero\_fill.py`



Applies reviewed zero-fill only for features where biological absence can be represented as zero. This improves transfer feature coverage while preserving true missingness where zero is not justified.



\## Inputs



\### Transfer feature table



The primary input is:



```text

model\_input\_numeric.csv

```



Required structure:



```text

sample\_id

<strict spatial feature columns>

```



The table can contain one row or multiple rows.



Example:



```text

sample\_id,feature\_1,feature\_2,feature\_3

SAMPLE\_A,0.12,0.00,1.45

SAMPLE\_B,0.08,0.20,1.10

```



\### Frozen interpretation atlas



The module also requires a completed prediction interpretation model output folder containing:



\- strict feature registry

\- signed treatment-feature effects

\- signed treatment-theme effects

\- treatment dictionaries

\- treatment interpretation cards



\## Basic usage



Run all transfer steps:



```powershell

Set-StrictMode -Version Latest

$ErrorActionPreference = "Stop"



$TransferRoot = "<path-to-spatial\_transfer\_inference\_model>"

$PimRunRoot = "<path-to-completed-prediction-interpretation-model-run>"

$FeatureTable = "<path-to-transfer-ready-model\_input\_numeric.csv>"

$Python = "<path-to-python-executable>"



\& $Python (Join-Path $TransferRoot "scripts/00\_run\_spatial\_transfer\_inference\_model.py") `

&#x20;   --model-root $TransferRoot `

&#x20;   --pim-run-root $PimRunRoot `

&#x20;   --run-name "spatial\_transfer\_inference\_example" `

&#x20;   --output-root (Join-Path $TransferRoot "outputs/spatial\_transfer\_inference\_example") `

&#x20;   --sample-id "TRANSFER\_BATCH" `

&#x20;   --single-slide-feature-table $FeatureTable `

&#x20;   --steps all `

&#x20;   --python $Python `

&#x20;   --open-output

```



\## Single-sample run



For a single Visium sample, use a feature table with one row.



Expected output with a 27-treatment atlas:



```text

1 sample × 27 treatment signatures = 27 prediction rows

```



\## Multi-sample run



For multiple Visium samples, use one row per sample and keep `sample\_id` as the first column.



Expected output with four samples and a 27-treatment atlas:



```text

4 samples × 27 treatment signatures = 108 prediction rows

```



The `--sample-id` argument can be used as a run or batch label unless it exactly matches a sample ID in the input table.



\## Output structure



A successful run creates:



```text

outputs/<run\_name>/

├── 01\_prepared\_transfer\_inputs/

├── 02\_aligned\_features/

├── 03\_transfer\_scores/

├── 04\_prediction\_table/

├── 05\_qc\_and\_transfer\_package/

├── pipeline\_run\_logs/

├── spatial\_transfer\_inference\_model\_run\_summary.json

├── spatial\_transfer\_inference\_model\_orchestrator\_report.txt

└── spatial\_transfer\_inference\_model\_orchestrator\_step\_manifest.tsv

```



\## Key output files



\### Main prediction table



```text

04\_prediction\_table/01\_prediction\_tables/single\_slide\_drug\_response\_interpretation\_table.tsv

```



Contains one row per sample-treatment pair.



\### Feature contribution table



```text

03\_transfer\_scores/02\_feature\_contributions/single\_slide\_treatment\_feature\_contributions.tsv

```



Lists feature-level drivers for each treatment score.



\### Theme contribution table



```text

03\_transfer\_scores/03\_theme\_contributions/single\_slide\_treatment\_theme\_contributions.tsv

```



Summarizes biological theme-level drivers.



\### Final QC report



```text

05\_qc\_and\_transfer\_package/04\_reports/spatial\_transfer\_inference\_final\_qc\_and\_package\_report.txt

```



Summarizes run status, row counts, package status, and QC checks.



\### Final package



```text

05\_qc\_and\_transfer\_package/03\_transfer\_zip/spatial\_transfer\_inference\_package.zip

```



Portable ZIP containing the main transfer outputs.



\## Important output columns



\### `sample\_id`



Input sample identifier.



\### `drug\_key`



Treatment signature from the frozen interpretation atlas.



\### `probability\_effective\_research\_not\_calibrated`



Research-use probability-like score derived from spatial alignment. Higher values indicate stronger favorable spatial alignment relative to the treatment-associated spatial response pattern.



\### `spatial\_alignment\_score`



Signed spatial alignment score. Positive values indicate sensitivity-associated spatial alignment. Negative values indicate resistance/barrier-associated spatial alignment.



\### `research\_prediction\_label`



Qualitative spatial profile label, such as favorable, indeterminate, or unfavorable/barrier-aligned.



\### `confidence\_level`



Evidence-support label based on feature coverage, score magnitude, and contribution support.



\### `top\_sensitivity\_features`



Spatial features that most strongly support the sensitivity-associated direction.



\### `top\_resistance\_or\_barrier\_features`



Spatial features that most strongly support the resistance/barrier-associated direction.



\### `explanation`



Plain-language interpretation of the sample-treatment spatial alignment result.



\## Recommended transfer workflow



\### 1. Generate spatial features



Run the Visium sample or batch through the spatial feature identification pipeline.



\### 2. Build transfer handoff



Use:



```text

00b\_build\_improved\_transfer\_handoff.py

```



to create a transfer-ready feature table.



\### 3. Audit feature coverage



Use:



```text

00c\_audit\_transfer\_strict\_feature\_missingness.py

```



to classify missing and observed strict features.



\### 4. Apply reviewed zero-fill



Use:



```text

00d\_apply\_reviewed\_zero\_fill.py

```



to fill biologically absence-like features when appropriate.



\### 5. Run transfer inference



Use:



```text

00\_run\_spatial\_transfer\_inference\_model.py

```



to generate treatment-alignment scores and interpretation outputs.



\### 6. Review results



Start with:



```text

single\_slide\_drug\_response\_interpretation\_table.tsv

spatial\_transfer\_inference\_final\_qc\_and\_package\_report.txt

```



Then inspect feature and theme contribution tables for biological interpretation.



\## Smoke test



```powershell

$TransferRoot = "<path-to-spatial\_transfer\_inference\_model>"

$PimRunRoot = "<path-to-completed-prediction-interpretation-model-run>"

$Python = "<path-to-python-executable>"



\& $Python (Join-Path $TransferRoot "scripts/00\_run\_spatial\_transfer\_inference\_model.py") `

&#x20;   --model-root $TransferRoot `

&#x20;   --pim-run-root $PimRunRoot `

&#x20;   --run-name "spatial\_transfer\_inference\_smoke" `

&#x20;   --output-root (Join-Path $TransferRoot "outputs/spatial\_transfer\_inference\_smoke") `

&#x20;   --sample-id "TRANSFER\_SMOKE\_SAMPLE\_001" `

&#x20;   --steps all `

&#x20;   --python $Python `

&#x20;   --smoke-test `

&#x20;   --open-output

```



A passing smoke run writes prediction tables, contribution tables, QC files, and a final ZIP package.



\## Interpretation guidance



The model provides spatial response-alignment scores. It is best used to compare relative spatial treatment-alignment patterns across samples or treatment signatures.



A higher positive alignment indicates that the sample spatial feature profile resembles sensitivity-associated spatial biology for a treatment signature.



A stronger negative alignment indicates that the sample spatial feature profile resembles resistance-associated or barrier-associated spatial biology.



Treatment keys are inherited from the frozen model atlas. When needed, downstream reports can use simplified treatment-card labels while retaining the raw `drug\_key` for provenance.





\## Development status



This module has been tested on:



\- smoke-test transfer input

\- one real Visium sample

\- a four-sample Visium batch



The multi-sample run preserved all sample rows and produced the expected sample-by-treatment output table.



