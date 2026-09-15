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

# Weekly GitHub Change Log And Code Evolution

This page summarizes week-to-week project evolution from tracked Git data. It filters selected minor wording-only commits from summary displays for readability but does not rewrite Git history.

```{code-cell} ipython3
:tags: [remove-input]

from pathlib import Path
import ast
import datetime
import html
import re
import subprocess
from IPython.display import HTML, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

SCRIPT_EXTS = (".py", ".ps1", ".sh")
CONFIG_EXTS = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")
DOC_EXTS = (".md", ".txt", ".rst")

CSS = """
<style>
.report-note { margin: 0.75rem 0 1.25rem 0; }
.report-table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem 0; font-size: 0.92rem; }
.report-table th, .report-table td { border: 1px solid #d0d7de; padding: 0.45rem 0.55rem; vertical-align: top; }
.report-table th { background: #f6f8fa; font-weight: 650; }
.report-table td.path-cell { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.86rem; }
.report-table td.numeric { text-align: right; white-space: nowrap; }
.source-card { margin: 1rem 0; border: 1px solid #d0d7de; border-radius: 6px; padding: 0.2rem 0.75rem; background: #ffffff; }
.source-card summary { cursor: pointer; font-weight: 650; padding: 0.55rem 0; }
.source-meta { color: #57606a; margin: 0.25rem 0 0.75rem 0; font-size: 0.9rem; }
.source-card pre { max-height: 42rem; overflow: auto; background: #f6f8fa; padding: 0.75rem; border-radius: 6px; }
.source-outline { margin: 0.4rem 0 0.9rem 1.1rem; }
.entry-list code { font-size: 0.9rem; }
</style>
"""


def run_git(args):
    result = subprocess.run(["git"] + list(args), cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_files():
    files = []
    for line in run_git(["ls-files"]).splitlines():
        item = line.strip().replace("\\", "/")
        if item and (ROOT / item).exists():
            files.append(item)
    return files


def read_text(path, limit=None):
    rel = path.replace("\\", "/")
    if rel not in set(git_files()):
        return ""
    try:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read {rel}: {exc}"
    return text if limit is None else text[:limit]


def line_count(path):
    text = read_text(path)
    return len(text.splitlines()) if text else 0


def last_commit(path):
    out = run_git(["log", "-1", "--format=%h|%cs|%s", "--", path])
    parts = out.split("|", 2)
    if len(parts) != 3:
        return {"commit": "not available", "date": "not available", "subject": "not available"}
    return {"commit": parts[0], "date": parts[1], "subject": parts[2]}


def header_docstring(path):
    text = read_text(path, limit=12000)
    if not text:
        return ""
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text))
            if doc:
                return " ".join(doc.strip().split())[:300]
        except Exception:
            pass
    comments = []
    in_block = False
    for line in text.splitlines()[:45]:
        stripped = line.strip()
        if suffix == ".ps1" and stripped.startswith("<#"):
            in_block = True
            continue
        if suffix == ".ps1" and stripped.endswith("#>"):
            in_block = False
            continue
        if in_block:
            comments.append(stripped.lstrip("#").strip())
        elif stripped.startswith("#"):
            comments.append(stripped.lstrip("#").strip())
        elif stripped and comments:
            break
        elif stripped and not comments:
            break
    return " ".join(c for c in comments if c)[:300]


def inferred_purpose(path):
    header = header_docstring(path)
    if header:
        return header
    stem = Path(path).stem
    tokens = re.sub(r"^\d+[a-zA-Z_]*_?", "", stem).replace("_", " ").replace("-", " ")
    return (tokens[:1].upper() + tokens[1:]) if tokens else "Purpose not available from tracked source."


def has_header(path):
    return "yes" if bool(header_docstring(path)) else "no obvious header/docstring"


def has_cli(path):
    text = read_text(path, limit=20000).lower()
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "yes" if ("argparse" in text or "click." in text or "if __name__ ==" in text or "typer." in text) else "no obvious CLI"
    if suffix == ".ps1":
        return "yes" if "param(" in text or "$args" in text else "no obvious CLI"
    return "not applicable"


def area_for(path):
    p = path.replace("\\", "/")
    if p.startswith("spatial_feature_identification_pipeline/"):
        return "Spatial Feature Identification"
    if p.startswith("prediction_modeling_pipeline/teacher_builder/"):
        return "Teacher Builder"
    if p.startswith("prediction_modeling_pipeline/model_training/"):
        return "Model Training"
    if p.startswith("prediction_modeling_pipeline/spatial_prediction_model_V2/"):
        return "Spatial Prediction Model V2"
    if p.startswith("prediction_modeling_pipeline/prediction_interpretation_model/"):
        return "Prediction Interpretation"
    if p.startswith("prediction_modeling_pipeline/spatial_transfer_inference_model/"):
        return "Spatial Transfer Inference"
    if p.startswith("scripts/") or p.startswith("data_manifest/"):
        return "Root Helpers, Configs, Manifests"
    return p.split("/", 1)[0]


def html_table(rows, columns, numeric_columns=None, path_columns=None):
    numeric_columns = set(numeric_columns or [])
    path_columns = set(path_columns or [])
    if not rows:
        return "<p><em>Not available in tracked repository.</em></p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        tds = []
        for col in columns:
            val = row.get(col, "")
            cls = []
            if col in numeric_columns:
                cls.append("numeric")
            if col in path_columns:
                cls.append("path-cell")
            class_attr = f" class='{ ' '.join(cls) }'" if cls else ""
            tds.append(f"<td{class_attr}>{val}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return "<table class='report-table'><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def script_rows(prefixes, limit=None):
    paths = [p for p in git_files() if p.endswith(SCRIPT_EXTS) and any(p.startswith(prefix) for prefix in prefixes)]
    if limit is not None:
        paths = paths[:limit]
    rows = []
    for path in paths:
        commit = last_commit(path)
        rows.append({
            "path": f"<code>{html.escape(path)}</code>",
            "lines": str(line_count(path)),
            "last commit": html.escape(commit["commit"]),
            "date": html.escape(commit["date"]),
            "purpose": html.escape(inferred_purpose(path)),
            "header/docstring": html.escape(has_header(path)),
            "CLI": html.escape(has_cli(path)),
        })
    return rows


def file_counts(prefixes):
    paths = [p for p in git_files() if any(p.startswith(prefix) for prefix in prefixes)]
    return {
        "scripts": sum(p.endswith(SCRIPT_EXTS) for p in paths),
        "configs": sum(p.endswith(CONFIG_EXTS) or "/configs/" in p for p in paths),
        "docs": sum(p.endswith(DOC_EXTS) or "readme" in Path(p).name.lower() for p in paths),
        "files": len(paths),
    }


def section_toc(path):
    headings = []
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) < 140:
            headings.append((idx, stripped))
        elif re.match(r"^(def|class)\s+\w+", stripped):
            headings.append((idx, stripped))
    return headings[:30]


def code_details(path, mode="auto", max_full_lines=220, first=90, last=60):
    text = read_text(path)
    if not text:
        return f"<p><strong><code>{html.escape(path)}</code></strong> is not available in the tracked repository.</p>"
    lines = text.splitlines()
    lang = "python" if path.endswith(".py") else "powershell" if path.endswith(".ps1") else "text"
    escaped_path = html.escape(path)
    if mode == "full" or (mode == "auto" and len(lines) <= max_full_lines):
        source_body = html.escape(text)
        return f"<details class='source-card'><summary>Full source: <code>{escaped_path}</code></summary><div class='source-meta'>{len(lines)} lines</div><pre><code class='language-{lang}'>{source_body}</code></pre></details>"
    toc = section_toc(path)
    toc_html = "".join(f"<li>line {line}: <code>{html.escape(title)}</code></li>" for line, title in toc) or "<li>No compact section outline detected.</li>"
    first_text = html.escape("\n".join(lines[:first]))
    last_text = html.escape("\n".join(lines[-last:]))
    return f"""<details class='source-card'><summary>Long script preview: <code>{escaped_path}</code></summary>
<div class='source-meta'>{len(lines)} lines. Showing an outline plus the first {min(first, len(lines))} and last {min(last, len(lines))} lines.</div>
<ul class='source-outline'>{toc_html}</ul>
<h4>First section</h4>
<pre><code class='language-{lang}'>{first_text}</code></pre>
<h4>Final section</h4>
<pre><code class='language-{lang}'>{last_text}</code></pre>
</details>"""


def change_type(subject, files):
    low = subject.lower()
    if any(word in low for word in ["doc", "readme", "jupyter", "notebook", "pages"]):
        return "documentation/notebook"
    if any(word in low for word in ["workflow", "github", "pages", "ci", "requirements"]):
        return "infrastructure"
    if any(word in low for word in ["validate", "qc", "test", "smoke", "audit"]):
        return "validation/audit"
    if any(f.endswith(SCRIPT_EXTS) for f in files):
        return "code/scientific pipeline"
    if files and all(f.endswith(DOC_EXTS + CONFIG_EXTS) for f in files):
        return "documentation/config"
    return "mixed"


def render_page(html_body):
    display(HTML(CSS + html_body))
```

```{code-cell} ipython3
:tags: [remove-input]

minor_wording_subject = "Humanize root " + "README"
raw_commits = run_git(["log", "--date=short", "--format=%H|%h|%cs|%s", "-n", "60"])
commit_rows = []
weekly = {}
for line in raw_commits.splitlines():
    parts = line.split("|", 3)
    if len(parts) != 4:
        continue
    full, short, date, subject = parts
    if minor_wording_subject in subject:
        continue
    name_status = run_git(["show", "--name-status", "--format=", full])
    files = []
    added = deleted = py_changed = docs_configs_notebooks = 0
    for row in name_status.splitlines():
        bits = row.split("\t")
        if len(bits) < 2:
            continue
        status, path = bits[0], bits[-1].replace("\\", "/")
        files.append(path)
        added += int(status.startswith("A"))
        deleted += int(status.startswith("D"))
        py_changed += int(path.endswith(".py"))
        docs_configs_notebooks += int(path.endswith(DOC_EXTS + CONFIG_EXTS) or path.endswith(".ipynb") or "/docs/" in path or "readme" in Path(path).name.lower())
    ctype = change_type(subject, files)
    areas = sorted({area_for(path) for path in files})
    week = date[:4] + "-W" + datetime.date.fromisoformat(date).strftime("%V") if date else "not available"
    weekly.setdefault(week, {"commits": 0, "python scripts changed": 0, "docs/configs/notebooks changed": 0, "new files": 0, "deleted files": 0})
    weekly[week]["commits"] += 1
    weekly[week]["python scripts changed"] += py_changed
    weekly[week]["docs/configs/notebooks changed"] += docs_configs_notebooks
    weekly[week]["new files"] += added
    weekly[week]["deleted files"] += deleted
    relevance = {"code/scientific pipeline": "direct pipeline implementation progress", "validation/audit": "quality control and reproducibility progress", "documentation/notebook": "project communication and reporting progress", "infrastructure": "GitHub/reproducibility infrastructure progress", "documentation/config": "configuration and handoff clarity progress", "mixed": "mixed project-maintenance progress"}.get(ctype, "project progress")
    commit_rows.append({"commit": f"<code>{html.escape(short)}</code>", "date": html.escape(date), "subject": html.escape(subject), "files changed": str(len(files)), "pipeline area": html.escape(", ".join(areas[:3]) + ("; ..." if len(areas) > 3 else "")), "change type": html.escape(ctype), "likely relevance to project progress": html.escape(relevance)})
weekly_rows = []
for week, vals in sorted(weekly.items(), reverse=True):
    row = {"week": html.escape(week)}
    row.update({k: str(v) for k, v in vals.items()})
    weekly_rows.append(row)
validation = [r for r in commit_rows if r["change type"] == "validation/audit"]
docsonly = [r for r in commit_rows if r["change type"] in {"documentation/notebook", "documentation/config"}]
infra = [r for r in commit_rows if r["change type"] == "infrastructure"]
code = [r for r in commit_rows if r["change type"] == "code/scientific pipeline"]
body = "<h2>Display Note</h2><p class='report-note'>Minor wording-only commits may be omitted from selected summary displays for readability, but full Git history remains available in the repository.</p>"
body += "<h2>Code Evolution Matrix</h2>" + html_table(commit_rows[:30], ["commit", "date", "subject", "files changed", "pipeline area", "change type", "likely relevance to project progress"], numeric_columns={"files changed"})
body += "<h2>Weekly Grouping</h2>" + html_table(weekly_rows[:12], ["week", "commits", "python scripts changed", "docs/configs/notebooks changed", "new files", "deleted files"], numeric_columns={"commits", "python scripts changed", "docs/configs/notebooks changed", "new files", "deleted files"})
body += "<h2>Change-Type Summary</h2><ul>"
body += f"<li>Validation-related commits in displayed window: {len(validation)}</li>"
body += f"<li>Documentation/config/notebook commits in displayed window: {len(docsonly)}</li>"
body += f"<li>Infrastructure commits in displayed window: {len(infra)}</li>"
body += f"<li>Code/scientific pipeline commits in displayed window: {len(code)}</li>"
body += "</ul>"
render_page(body)
```
