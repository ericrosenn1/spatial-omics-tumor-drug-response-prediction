\# Prediction Interpretation Model



\## Overview



`prediction\_interpretation\_model` is the final interpretation and reporting layer for the spatial treatment-response modeling workflow. It takes the completed outputs from `spatial\_prediction\_model\_V2` and converts them into biologically interpretable summaries of how spatial tissue architecture relates to treatment-response residuals.



The model does not train a new treatment-response predictor from scratch. Instead, it starts from the validated spatial residual modeling results produced by `spatial\_prediction\_model\_V2` and builds a final interpretation package:



\- signed spatial feature effects

\- treatment-level interpretation cards

\- sample-level spatial response/resistance interpretation scores

\- cross-treatment mechanism atlases

\- treatment similarity summaries

\- final publication tables

\- final figures

\- final reports

\- final QC and package manifests



The goal is to make the spatial prediction results interpretable at three levels:



1\. \*\*Feature level:\*\* Which spatial architecture features are associated with higher or lower treatment-response residuals?

2\. \*\*Treatment level:\*\* Which spatial mechanisms are associated with sensitivity or resistance for each validated treatment?

3\. \*\*Sample level:\*\* Which samples have spatial profiles aligned with sensitivity-associated or resistance-associated signatures?



\## Scientific motivation



The central hypothesis is that treatment response is influenced not only by which biological programs are present, but also by where and how they are organized in tissue. Spatial features may capture biological barriers and microenvironmental states that are not fully represented by expression or histology alone.



Examples of spatial biology summarized by the model include:



\- tumor-core and tumor-boundary organization

\- immune exclusion or immune penetration

\- stromal and extracellular-matrix barrier structure

\- hypoxic or stress-associated tumor regions

\- myeloid/macrophage spatial enrichment

\- angiogenic and vascular context

\- metabolic spatial programs

\- hotspot fragmentation and spatial continuity

\- tumor accessibility and partial-penetration patterns



The output is intended for biological interpretation and hypothesis generation. It is not intended to provide clinical treatment recommendations.



\## Input data



The required input is a completed `spatial\_prediction\_model\_V2` run folder. The upstream V2 model produces the residual modeling outputs, strict spatial biology feature registry, label-shuffle-validated treatment models, recurrent spatial features, recurrent biology themes, and final QC files used by this interpretation layer.



The development run used for this project had the following source contract:



\- 34,881 sample-treatment pair rows

\- 102 spatial samples

\- 374 treatment keys

\- 139 strict biology spatial features

\- 27 label-shuffle-validated treatment models

\- 11 recurrent biology themes

\- final V2 QC pass

\- no production dependency on deprecated V1 outputs



The main input argument is:



```text

\--v2-run-root <path\_to\_completed\_spatial\_prediction\_model\_V2\_run>

```



\## What the model does



The pipeline converts V2 residual modeling outputs into a structured interpretation package.



First, it validates and prepares the V2 source files. It checks the expected V2 run structure, confirms QC status, records file provenance, and prepares a stable source manifest for downstream steps.



Second, it builds dictionaries for features, treatments, treatment components, and biology themes. These dictionaries convert model feature names into readable biological annotations and organize treatments into interpretable component classes.



Third, it computes signed spatial effects. V2 feature importance scores are useful but do not by themselves indicate whether a feature is associated with sensitivity or resistance. This model assigns directionality by correlating spatial feature values with the V2 residual response target, `fused\_residual\_vs\_prior`, and weighting those associations by V2 model evidence.



A positive signed effect means that higher values of a spatial feature are associated with above-prior response residuals for a treatment. A negative signed effect means that higher values of a spatial feature are associated with below-prior response residuals.



Fourth, it summarizes treatment-level mechanisms. For each label-shuffle-validated treatment, the model generates a treatment interpretation card containing the strongest sensitivity-associated features, resistance-associated features, sensitivity-associated biology themes, resistance-associated biology themes, and relevant validation metrics.



Fifth, it generates sample-level interpretation scores. For each sample-treatment pair with validated treatment coverage, the model scores whether the sample’s spatial profile aligns more strongly with sensitivity-associated or resistance-associated spatial signatures.



Finally, it builds final reporting outputs, including mechanism atlases, final publication tables, final figures, final narrative reports, QC checks, and a package ZIP.



\## Pipeline steps



The workflow is organized into numbered scripts.



| Step | Script | Purpose |

|---|---|---|

| 00 | `00\_run\_prediction\_interpretation\_model.py` | Runs one or more numbered pipeline steps. |

| 01 | `01\_prepare\_interpretation\_inputs.py` | Validates the V2 run root and prepares source manifests and input tables. |

| 02 | `02\_build\_feature\_and\_treatment\_dictionary.py` | Builds feature, biology-theme, treatment, component, and source-column dictionaries. |

| 03 | `03\_compute\_signed\_spatial\_effects.py` | Computes signed treatment-feature and treatment-theme effects. |

| 04 | `04\_build\_treatment\_interpretation\_cards.py` | Creates one interpretation card per validated treatment. |

| 05 | `05\_build\_sample\_level\_interpretations.py` | Computes sample-treatment spatial interpretation scores and feature contributions. |

| 06 | `06\_build\_mechanism\_atlas.py` | Builds cross-treatment mechanism, component, similarity, and sample mechanism atlases. |

| 07 | `07\_make\_final\_outputs.py` | Creates final publication tables, workbook, figures, captions, and reports. |

| 08 | `08\_qc\_and\_package\_final\_outputs.py` | Runs final QC and creates the final package ZIP. |



Shared helper functions are stored in:



```text

scripts/\_pim\_utils.py

```



\## Repository layout



```text

prediction\_interpretation\_model/

&#x20; README.md

&#x20; .gitignore

&#x20; .gitattributes



&#x20; configs/

&#x20;   example\_prediction\_interpretation\_model\_full\_run.json



&#x20; docs/

&#x20;   SOURCE\_OF\_TRUTH\_POLICY.md

&#x20;   STEP\_ORDER.md

&#x20;   GITHUB\_REPOSITORY\_NOTES.md



&#x20; scripts/

&#x20;   00\_run\_prediction\_interpretation\_model.py

&#x20;   01\_prepare\_interpretation\_inputs.py

&#x20;   02\_build\_feature\_and\_treatment\_dictionary.py

&#x20;   03\_compute\_signed\_spatial\_effects.py

&#x20;   04\_build\_treatment\_interpretation\_cards.py

&#x20;   05\_build\_sample\_level\_interpretations.py

&#x20;   06\_build\_mechanism\_atlas.py

&#x20;   07\_make\_final\_outputs.py

&#x20;   08\_qc\_and\_package\_final\_outputs.py

&#x20;   \_pim\_utils.py



&#x20; outputs/

&#x20;   generated local outputs; not committed



&#x20; logs/

&#x20;   generated local logs; not committed



&#x20; local/

&#x20;   local archives, cleanup reports, and temporary checks; not committed

```



\## Installation



Create or activate a Python environment with the required dependencies.



```bash

pip install pandas numpy matplotlib openpyxl

```



The scripts use only standard Python libraries plus these packages.



\## Basic usage



Run the full interpretation workflow through the orchestrator.



```powershell

$ProjectRoot = "D:\\Adv\_Omics\_Fenyo\\project"

$PimRoot = Join-Path $ProjectRoot "prediction\_modeling\_pipeline\\prediction\_interpretation\_model"

$Python = Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"



$V2RunRoot = "D:\\path\\to\\spatial\_prediction\_model\_V2\\outputs\\v2\_full\_run\_TIMESTAMP"

$RunOutputRoot = Join-Path $PimRoot "outputs\\prediction\_interpretation\_model\_full\_local"



\& $Python (Join-Path $PimRoot "scripts\\00\_run\_prediction\_interpretation\_model.py") `

&#x20;   --project-root $ProjectRoot `

&#x20;   --model-root $PimRoot `

&#x20;   --v2-run-root $V2RunRoot `

&#x20;   --run-name "prediction\_interpretation\_model\_full\_local" `

&#x20;   --output-root $RunOutputRoot `

&#x20;   --steps all `

&#x20;   --python $Python `

&#x20;   --open-output

```



To run only selected steps, pass a comma-separated list.



```powershell

\--steps 03,04,05

```



For example, after Step 01 and Step 02 have already passed, Steps 03 through 05 can be rerun with:



```powershell

\& $Python (Join-Path $PimRoot "scripts\\00\_run\_prediction\_interpretation\_model.py") `

&#x20;   --project-root $ProjectRoot `

&#x20;   --model-root $PimRoot `

&#x20;   --v2-run-root $V2RunRoot `

&#x20;   --run-name "prediction\_interpretation\_model\_full\_local" `

&#x20;   --output-root $RunOutputRoot `

&#x20;   --steps 03,04,05 `

&#x20;   --python $Python `

&#x20;   --open-output

```



\## Main outputs



A successful full run creates the following output folders:



```text

01\_prepared\_inputs/

02\_feature\_and\_treatment\_dictionary/

03\_signed\_spatial\_effects/

04\_treatment\_interpretation\_cards/

05\_sample\_level\_interpretations/

06\_mechanism\_atlas/

07\_final\_outputs/

08\_qc\_and\_final\_package/

```



Important final outputs include:



```text

07\_final\_outputs/01\_publication\_tables\_tsv/

07\_final\_outputs/02\_publication\_workbook/

07\_final\_outputs/03\_final\_figures/

07\_final\_outputs/04\_final\_reports/

08\_qc\_and\_final\_package/01\_final\_qc/

08\_qc\_and\_final\_package/03\_final\_zip/

08\_qc\_and\_final\_package/04\_reports/

```



The final package ZIP is written under:



```text

08\_qc\_and\_final\_package/03\_final\_zip/

```



\## Key output tables



The final output layer includes publication-ready tables such as:



\- `Final\_Treatment\_Interpretation\_Cards.tsv`

\- `Final\_Signed\_Treatment\_Feature\_Effects.tsv`

\- `Final\_Signed\_Treatment\_Theme\_Effects.tsv`

\- `Final\_Cross\_Treatment\_Biology\_Theme\_Atlas.tsv`

\- `Final\_Cross\_Treatment\_Feature\_Atlas.tsv`

\- `Final\_Component\_Class\_Mechanism\_Atlas.tsv`

\- `Final\_Sample\_Treatment\_Interpretation\_Scores.tsv`

\- `Final\_Sample\_Interpretation\_Summary.tsv`

\- `Final\_Sample\_Mechanism\_Summary.tsv`

\- `Final\_Treatment\_Theme\_Similarity\_Edges.tsv`



The Excel workbook combines the final publication tables into one file for review.



\## Treatment interpretation cards



Step 04 writes one text card per label-shuffle-validated treatment. Each card summarizes:



\- treatment key and treatment components

\- validation evidence

\- sensitivity-associated spatial features

\- resistance-associated spatial features

\- sensitivity-associated biology themes

\- resistance-associated biology themes

\- interpretation caveats



These cards are intended to make the model output readable without requiring the reader to inspect raw SHAP or feature-effect tables.



\## Sample-level interpretation scores



Step 05 computes sample-treatment spatial interpretation scores for validated treatments. A positive net score means the sample’s spatial profile is aligned with sensitivity-associated spatial biology for that treatment. A negative net score means the sample’s spatial profile is aligned with resistance-associated spatial biology for that treatment.



These scores are not response predictions and are not treatment recommendations. They are interpretation-layer summaries of how a sample’s spatial architecture aligns with the signed spatial mechanisms learned from the V2 residual models.



\## Coverage note: 93 of 102 samples in sample-level treatment scoring



The upstream V2 full run contains 102 spatial samples. However, sample-level treatment scoring in this interpretation layer is restricted to the 27 label-shuffle-validated treatment keys. Those validated treatment keys have pair-level residual rows for 93 samples, not all 102.



Therefore:



```text

27 validated treatments x 93 samples = 2,511 sample-treatment interpretation rows

```



The nine samples not scored in Step 05 are:



```text

SAMPLE\_0029

SAMPLE\_0030

SAMPLE\_0031

SAMPLE\_0032

SAMPLE\_0033

SAMPLE\_0077

SAMPLE\_0085

SAMPLE\_0086

SAMPLE\_0095

```



A coverage audit showed that these samples are present in the full V2 sample universe but have zero pair-level overlap with the 27 validated treatment keys. They have rows only for a restricted set of 11 low-coverage single-agent treatment keys. This is an upstream sample-treatment coverage issue and should be treated as a coverage-qualified pass, not as a downstream scoring failure.



\## Quality control



Each step writes reports, summary JSON files, manifests, and QC tables where relevant. Text reports begin with `FILEPATH:` to support traceability.



The final QC step checks:



\- required reports exist

\- expected QC tables exist

\- upstream step summaries are present

\- final tables and figures are present

\- final package files are present

\- final ZIP package is created



\## GitHub policy



Commit code, configuration examples, and durable documentation.



Commit:



```text

README.md

.gitignore

.gitattributes

scripts/\*.py

configs/example\_prediction\_interpretation\_model\_full\_run.json

docs/\*.md

```



Do not commit:



```text

outputs/

logs/

local/

archive/

\*.zip

\*.xlsx

\*.png

\*.pdf

\*.before\_\*

\*\_combined\_\*.txt

Pasted text\*.txt

\_LIVE\_\*.txt

```



Generated outputs should be regenerated from the scripts or stored separately from the GitHub code repository.



\## Interpretation caveats



This model summarizes associations between spatial features and treatment-response residuals. Signed effects are based on feature-target correlations weighted by model evidence. These associations do not establish causality.



Treatment cards, mechanism atlases, and sample-level scores should be interpreted as biological hypotheses and reporting summaries. They are not clinical treatment recommendations.

