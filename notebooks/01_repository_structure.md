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

# Repository Structure

This Jupyter Book page inventories tracked files only. It intentionally uses `git ls-files` so ignored local outputs and private data are excluded.

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
counts = collections.Counter()
for path in files:
    top = path.split("/", 1)[0]
    counts[top] += 1
lines = ["## Top-Level Tracked File Counts", ""]
for folder, count in sorted(counts.items()):
    lines.append(f"- `{folder}`: {count} tracked files")
md(lines)
```

```{code-cell} ipython3
:tags: [remove-input]

files = git_files()
interesting_ext = {".py", ".ps1", ".yaml", ".yml", ".json", ".md", ".tsv", ".csv"}
by_type = collections.Counter(Path(p).suffix.lower() or "[no extension]" for p in files)
lines = ["## Tracked File-Type Inventory", ""]
for ext, count in sorted(by_type.items()):
    if ext in interesting_ext or ext == "[no extension]":
        lines.append(f"- `{ext}`: {count}")
md(lines)
```

```{code-cell} ipython3
:tags: [remove-input]

files = git_files()
classes = {
    "Python scripts": [p for p in files if p.endswith(".py")],
    "PowerShell runners": [p for p in files if p.endswith(".ps1")],
    "README and Markdown docs": [p for p in files if p.lower().endswith(".md") or "readme" in Path(p).name.lower()],
    "Configs": [p for p in files if p.endswith((".yaml", ".yml", ".json")) and "config" in p.lower()],
    "Data manifests": [p for p in files if "manifest" in p.lower() and p.endswith((".tsv", ".csv", ".json", ".md"))],
}
lines = []
for title, paths in classes.items():
    lines.extend([f"## {title}", ""])
    lines.extend(bullet(paths, limit=50))
    lines.append("")
md(lines)
```
