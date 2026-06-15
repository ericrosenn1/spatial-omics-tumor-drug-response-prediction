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

# Configs, Manifests, And Root Helpers

Tracked configs, manifests, and root helper scripts that make the repository reproducible from GitHub-visible files. This page does not download data or validate URLs.

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
    if p.startswith("scripts/") or p.startswith("data_manifest/") or p.endswith("project_profile.example.yaml"):
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

files = git_files()
config_paths = [p for p in files if p.endswith(CONFIG_EXTS) or "/configs/" in p or p.endswith("requirements.txt") or p.endswith("requirements-notebooks.txt")]
manifest_paths = [p for p in files if "manifest" in p.lower()]
root_scripts = [p for p in files if p.startswith("scripts/") and p.endswith(SCRIPT_EXTS)]
rows = []
for path in config_paths + manifest_paths + root_scripts:
    commit = last_commit(path)
    rows.append({"path": f"<code>{html.escape(path)}</code>", "area": html.escape(area_for(path)), "lines": str(line_count(path)), "last commit": html.escape(commit["commit"]), "date": html.escape(commit["date"]), "role": html.escape(inferred_purpose(path))})
body = "<h2>Config, Manifest, And Root Helper Inventory</h2>" + html_table(rows, ["path", "area", "lines", "last commit", "date", "role"], numeric_columns={"lines"}, path_columns={"path"})
body += "<p class='report-note'>These tracked files define fresh-clone configuration examples, public staging manifests, and lightweight repository helper scripts. They are displayed for review only; this Jupyter Book page does not download data.</p>"
body += "<h2>Selected Tracked Source And Config Snippets</h2>"
for path in ["scripts/download_and_reconstruct_public_visium_sources.py", "data_manifest/public_visium_cohort_staging_manifest.tsv", "spatial_feature_identification_pipeline/configs/spatial_feature_pipeline_full_config.example.yaml", "prediction_modeling_pipeline/teacher_builder/configs/visium_teacher_builder_governed_full.example.yaml", "prediction_modeling_pipeline/spatial_transfer_inference_model/configs/resolved_pim_transfer_file_map.example.json"]:
    body += code_details(path, first=80, last=40)
render_page(body)
```
