---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Pipeline Architecture

This Jupyter Book page summarizes the architecture from tracked README, docs, configs, and script inventories only. It does not execute production scripts.

```{code-cell} ipython3
:tags: [remove-input]

from pathlib import Path
import collections
import re
import subprocess
from IPython.display import Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


def git_files():
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        display(Markdown(f"**Tracked file inventory not available:** `{result.stderr.strip()}`"))
        return []
    files = []
    for line in result.stdout.splitlines():
        item = line.strip().replace("\\", "/")
        if item and (ROOT / item).exists():
            files.append(item)
    return files


def read_tracked(path, limit=6000):
    path = path.replace("\\", "/")
    files = set(git_files())
    if path not in files:
        return ""
    full = ROOT / path
    try:
        return full.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception as exc:
        return f"Could not read {path}: {exc}"


def tracked_exists(path):
    return path.replace("\\", "/") in set(git_files())


def tracked_matching(*patterns):
    files = git_files()
    out = []
    for path in files:
        low = path.lower()
        if all(pattern.lower() in low for pattern in patterns):
            out.append(path)
    return out


def md(lines):
    if isinstance(lines, str):
        display(Markdown(lines))
    else:
        display(Markdown("\n".join(lines)))


def bullet(paths, limit=40):
    paths = list(paths)
    if not paths:
        return ["- Not available in tracked repository."]
    shown = paths[:limit]
    lines = [f"- `{p}`" for p in shown]
    if len(paths) > limit:
        lines.append(f"- ... {len(paths) - limit} more tracked files omitted from this display")
    return lines
```

```{code-cell} ipython3
:tags: [remove-input]

components = [
    ("Public Visium Data Staging Tool", ["scripts/download_and_reconstruct_public_visium_sources.py", "data_manifest/public_visium_cohort_staging_manifest.tsv", "docs/PUBLIC_SOURCE_RECONSTRUCTION.md"]),
    ("Spatial Feature Identification Pipeline", ["spatial_feature_identification_pipeline/README.md"]),
    ("Expression Response Model V2", ["prediction_modeling_pipeline/model_training/expression_response_model_v2/README.md", "prediction_modeling_pipeline/model_training/expression_response_model_v2/run_expression_response_model_v2.ps1"]),
    ("Histology Response Model V2", ["prediction_modeling_pipeline/model_training/histology_response_model_v2/README.md", "prediction_modeling_pipeline/model_training/histology_response_model_v2/run_histology_response_model_v2.ps1"]),
    ("Teacher Builder", ["prediction_modeling_pipeline/teacher_builder/README.md", "prediction_modeling_pipeline/teacher_builder/run_teacher_builder_governed.ps1"]),
    ("Spatial Prediction Model V2", ["prediction_modeling_pipeline/spatial_prediction_model_V2/README.md"]),
    ("Prediction Interpretation Model", ["prediction_modeling_pipeline/prediction_interpretation_model/README.md"]),
    ("Spatial Transfer Inference Model", ["prediction_modeling_pipeline/spatial_transfer_inference_model/README.md"]),
]
lines = ["## Architecture Components", ""]
for name, paths in components:
    lines.append(f"### {name}")
    for path in paths:
        status = "available" if tracked_exists(path) else "not available in tracked repository"
        lines.append(f"- `{path}`: {status}")
    lines.append("")
md(lines)
```

````{code-cell} ipython3
:tags: [remove-input]

md("""
## High-Level Flow

```text
Public Visium staging
        ↓
Spatial feature identification
        ↓
Teacher-builder inputs + upstream response teachers
        ↓
Expression Response Model V2 + Histology Response Model V2
        ↓
Teacher Builder governed fusion
        ↓
Spatial Prediction Model V2
        ↓
Prediction Interpretation Model
        ↓
Spatial Transfer Inference Model
```

The notebooks are intentionally outside this production flow. They are a reporting and navigation surface for tracked repository state.
""")
````
