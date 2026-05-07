from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json, write_text_report


def package_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return ""


def environment_summary() -> dict:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "xgboost",
        "shap",
        "matplotlib",
        "openpyxl",
    ]
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {name: package_version(name) for name in packages},
    }


def git_status(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    out = {"repo_root": str(repo_root), "available": False, "status": "", "commit": ""}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True, stderr=subprocess.STDOUT)
        out.update({"available": True, "commit": commit, "status": status})
    except Exception:
        pass
    return out


def write_run_provenance(output_root: str | Path, repo_root: str | Path, extra: dict | None = None) -> dict:
    output_root = ensure_dir(output_root)
    payload = {
        "environment": environment_summary(),
        "git": git_status(repo_root),
        "extra": extra or {},
    }
    write_json(payload, output_root / "environment_and_provenance.json")
    write_text_report(
        output_root / "environment_and_provenance.txt",
        json.dumps(payload, indent=2),
    )
    return payload


def script_inventory(script_root: str | Path) -> pd.DataFrame:
    root = Path(script_root)
    rows = []
    if root.exists():
        for path in sorted(root.glob("*.py")):
            rows.append({
                "script": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            })
    return pd.DataFrame(rows)
