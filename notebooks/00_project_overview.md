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

# Project Overview

This Jupyter Book page introduces the repository-level goal and the major components visible in the tracked GitHub checkout. It is a reporting layer, not the production pipeline.

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

files = git_files()
major_components = {
    "Public Visium data staging": "scripts/download_and_reconstruct_public_visium_sources.py",
    "Spatial feature identification": "spatial_feature_identification_pipeline/README.md",
    "Prediction modeling pipeline": "prediction_modeling_pipeline/README.md",
    "Expression response model V2": "prediction_modeling_pipeline/model_training/expression_response_model_v2/README.md",
    "Histology response model V2": "prediction_modeling_pipeline/model_training/histology_response_model_v2/README.md",
    "Teacher builder": "prediction_modeling_pipeline/teacher_builder/README.md",
    "Spatial prediction model V2": "prediction_modeling_pipeline/spatial_prediction_model_V2/README.md",
    "Prediction interpretation model": "prediction_modeling_pipeline/prediction_interpretation_model/README.md",
    "Spatial transfer inference model": "prediction_modeling_pipeline/spatial_transfer_inference_model/README.md",
}
lines = [
    "## Repository Goal",
    "",
    "The project studies whether spatial tumor architecture and spatial omics features can support treatment-response prediction and interpretation. The tracked repository contains data-staging, spatial feature engineering, teacher-label construction, model training, prediction, interpretation, and transfer-inference components.",
    "",
    "## Reporting Boundary",
    "",
    "These notebooks summarize tracked GitHub files only. They do not execute production workflows, train models, run inference, download datasets, inspect ignored local outputs, or copy private local data.",
    "",
    "## Major Components Found In This Checkout",
]
for name, path in major_components.items():
    status = "available" if path in files else "not available in tracked repository"
    lines.append(f"- **{name}**: `{path}` ({status})")
md(lines)
```
