"""Check tracked source syntax, portable inputs, artifacts and local Markdown links.

Runs without raw data. Terminology matches are reported for contextual review,
not treated as defects merely because they occur in an API or scientific term.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ps1", ".yaml", ".yml", ".json", ".toml", ".txt", ".md", ".ipynb", ".tsv", ".csv", ".cff"}
TERMS = re.compile(r"\b(Codex|ChatGPT|OpenAI|agents?|reviewers?|students?|forensic|authority|canonical|deprecated|temporary|TODO|FIXME|HACK|backup|archive)\b|DOC_POLISH|PATCH_V|FIX_V", re.I)


def audit(root=ROOT):
    root = Path(root).resolve()
    # Untracked source belongs to users too: only audit tracked files by default.
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    errors, terminology, inventory = [], [], []
    for relative in sorted(tracked - {""}):
        path = root / relative
        if not path.exists():
            continue  # Intentional pending deletions are absent from the next release.
        inventory.append({"path": relative, "bytes": path.stat().st_size})
        if any(part in {"__pycache__", ".venv", ".ipynb_checkpoints", "outputs", "local"} for part in path.relative_to(root).parts) or path.suffix in {".pyc", ".zip", ".joblib", ".pdf", ".docx"}:
            errors.append({"path": relative, "check": "generated_artifact"})
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        body = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".py":
            try:
                ast.parse(body, filename=relative)
            except SyntaxError as exc:
                errors.append({"path": relative, "check": "python_syntax", "detail": str(exc)})
        if path.suffix in {".json", ".ipynb"}:
            try:
                json.loads(body)
            except ValueError as exc:
                errors.append({"path": relative, "check": "json_syntax", "detail": str(exc)})
        for number, line in enumerate(body.splitlines(), 1):
            if re.search(r"[A-Za-z]:[/\\]+Users[/\\]|[A-Za-z]:[/\\]+Adv_Omics|/[h]ome/[^/\s]+/|/mnt/[a-z]/Users/", line, re.I):
                errors.append({"path": relative, "line": number, "check": "personal_path"})
            if TERMS.search(line):
                terminology.append({"path": relative, "line": number, "text": line.strip()[:240]})
            if re.search(r"full-step09-n-shuffles\s+100(?:\s|$)", line):
                errors.append({"path": relative, "line": number, "check": "obsolete_full_permutation_setting"})
        if path.suffix == ".md":
            prose = re.sub(r"(?ms)^```.*?^```[^\n]*", "", body)
            for destination in re.findall(r"\]\(([^)]+)\)", prose):
                target = destination.split(' "', 1)[0].strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("#") or not parsed.path:
                    continue
                if not (path.parent / unquote(parsed.path)).exists():
                    errors.append({"path": relative, "check": "broken_relative_link", "target": target})
    return {"status": "PASS" if not errors else "FAIL", "files_scanned": len(inventory),
            "inventory": inventory, "errors": errors, "terminology_for_review": terminology}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"inventory", "terminology_for_review"}}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
