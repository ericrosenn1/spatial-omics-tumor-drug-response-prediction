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

# Current Reports And Outputs Summary

This Jupyter Book page lists tracked reports, docs, figures, tables, and manifests. Ignored local outputs are intentionally not inspected.

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
tracked_docs = [p for p in files if p.lower().endswith((".md", ".txt"))]
tracked_tables = [p for p in files if p.lower().endswith((".tsv", ".csv", ".json"))]
tracked_figures = [p for p in files if p.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf"))]
tracked_manifests = [p for p in files if "manifest" in p.lower()]
sections = {
    "Tracked documentation and reports": tracked_docs,
    "Tracked tables and JSON files": tracked_tables,
    "Tracked manifests": tracked_manifests,
    "Tracked figures/media": tracked_figures,
}
lines = []
for title, paths in sections.items():
    lines.extend([f"## {title}", ""])
    lines.extend(bullet(paths, limit=70))
    lines.append("")
if not tracked_figures:
    lines.append("Tracked figure artifacts are not available in the GitHub-only notebook build.")
md(lines)
```
