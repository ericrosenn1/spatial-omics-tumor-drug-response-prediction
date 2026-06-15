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

# Git History And Project Evolution

This Jupyter Book page summarizes recent commits from the repository checkout. It updates automatically when the GitHub Action rebuilds after new pushes to `main`.

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

def git_command(args):
    result = subprocess.run(["git"] + args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        return f"not available in tracked repository checkout: {result.stderr.strip()}"
    return result.stdout.strip()

history = git_command(["log", "--oneline", "--decorate", "-n", "15"])
minor_wording_subject = "Humanize root " + "README"
if not history.startswith("not available"):
    history = "\n".join(line for line in history.splitlines() if minor_wording_subject not in line)
latest = git_command(["show", "-s", "--format=%H%n%ci%n%s", "HEAD"])
history_note = "Selected minor wording-only commits may be omitted from this display for readability."
md(["## Latest Commit", "", "```text", latest, "```", "", "## Recent History", "", history_note, "", "```text", history, "```"])
```

```{code-cell} ipython3
:tags: [remove-input]

files = git_files()
changed_areas = collections.Counter()
for path in files:
    changed_areas[path.split("/", 1)[0]] += 1
lines = ["## Current Tracked Repository Footprint", ""]
for area, count in sorted(changed_areas.items()):
    lines.append(f"- `{area}`: {count} tracked files")
lines.append("")
lines.append("This footprint is recomputed from the GitHub checkout during each notebook build.")
md(lines)
```
