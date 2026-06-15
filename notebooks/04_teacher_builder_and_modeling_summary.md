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

# Teacher Builder And Modeling Summary

This Jupyter Book page summarizes teacher-label construction and modeling components using tracked files only.

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

sections = {
    "Teacher Builder": "prediction_modeling_pipeline/teacher_builder/",
    "Model Training": "prediction_modeling_pipeline/model_training/",
    "Spatial Prediction Model V2": "prediction_modeling_pipeline/spatial_prediction_model_V2/",
}
lines = []
for name, prefix in sections.items():
    paths = [p for p in git_files() if p.startswith(prefix)]
    lines.extend([f"## {name}", ""])
    lines.extend(bullet(paths, limit=60))
    lines.append("")
md(lines)
```

```{code-cell} ipython3
:tags: [remove-input]

tracked_text = []
for p in git_files():
    if p.startswith("prediction_modeling_pipeline/") and p.lower().endswith((".md", ".txt", ".yaml", ".yml", ".json")):
        tracked_text.append(read_tracked(p, limit=10000))
combined = "\n".join(tracked_text)
expected = {
    "34,881 sample-treatment rows": ["34,881"],
    "374 treatments": ["374"],
    "291 screened treatments": ["291"],
    "27 high-confidence residual models": ["27", "high-confidence"],
}
lines = ["## Tracked Documentation Numbers", ""]
for label, terms in expected.items():
    found = all(term.lower() in combined.lower() for term in terms)
    lines.append(f"- {label}: {'found in tracked documentation' if found else 'not available in tracked repository text'}")
lines.extend([
    "", "## Workflow Summary", "",
    "The teacher-builder combines treatment priors with expression and histology teacher signals, applies governed shrinkage and label-quality fields, and writes teacher-label handoffs for spatial prediction. The modeling folders contain response-teacher training workflows and spatial prediction workflows. This Jupyter Book page does not run those workflows."
])
md(lines)
```
