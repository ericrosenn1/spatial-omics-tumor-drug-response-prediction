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

# Prediction Interpretation And Transfer Summary

This Jupyter Book page summarizes the tracked interpretation and transfer-inference modules. It does not run inference.

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
    "Prediction Interpretation Model": "prediction_modeling_pipeline/prediction_interpretation_model/",
    "Spatial Transfer Inference Model": "prediction_modeling_pipeline/spatial_transfer_inference_model/",
}
lines = []
for name, prefix in sections.items():
    paths = [p for p in git_files() if p.startswith(prefix)]
    lines.extend([f"## {name}", ""])
    lines.extend(bullet(paths, limit=80))
    lines.append("")
md(lines)
```

```{code-cell} ipython3
:tags: [remove-input]

text = "\n".join(read_tracked(p, limit=12000) for p in git_files() if p.startswith("prediction_modeling_pipeline/") and p.lower().endswith((".md", ".json", ".yaml", ".yml")))
checks = {
    "Frozen or strict feature registry references": ["strict", "feature"],
    "Research-use transfer score language": ["research", "transfer"],
    "Source-of-truth policy references": ["source", "truth"],
}
lines = ["## Tracked Policy/Boundary Signals", ""]
for label, terms in checks.items():
    found = all(term.lower() in text.lower() for term in terms)
    lines.append(f"- {label}: {'found in tracked text' if found else 'not available in tracked repository text'}")
lines.extend([
    "", "## Interpretation Boundary", "",
    "The interpretation and transfer modules are tracked as command-line workflows. The notebook layer links and summarizes them, but does not compute feature effects, apply the frozen registry, or run transfer inference."
])
md(lines)
```
