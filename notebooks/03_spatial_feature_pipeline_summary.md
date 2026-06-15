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

# Spatial Feature Pipeline Summary

This Jupyter Book page summarizes the spatial feature identification pipeline from tracked files only.

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
paths = [p for p in files if p.startswith("spatial_feature_identification_pipeline/")]
lines = ["## Tracked Spatial Feature Pipeline Files", ""]
lines.extend(bullet(paths, limit=80))
md(lines)
```

```{code-cell} ipython3
:tags: [remove-input]

text_sources = [p for p in git_files() if p.startswith("spatial_feature_identification_pipeline/") and p.lower().endswith((".md", ".txt", ".yaml", ".yml", ".json"))]
combined = "\n".join(read_tracked(p, limit=12000) for p in text_sources)
expected = {
    "102 retained Visium samples": ["102", "retained"],
    "661 numeric spatial architecture features": ["661", "feature"],
    "12 source datasets": ["12", "dataset"],
}
lines = ["## Tracked Documentation Numbers", ""]
for label, terms in expected.items():
    found = all(term.lower() in combined.lower() for term in terms)
    lines.append(f"- {label}: {'found in tracked documentation' if found else 'not available in tracked repository text'}")
lines.extend(["", "## Role", "", "The spatial feature pipeline converts staged spatial omics/Visium inputs into numeric sample-level architecture features and manifests used by downstream teacher-building and spatial prediction modules."])
md(lines)
```
