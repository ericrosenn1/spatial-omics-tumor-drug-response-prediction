from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".json":
        obj = read_json(path)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        return pd.DataFrame()
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)
    return path


def write_text_report(path: str | Path, body: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")
    return path


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_dir(root: str | Path, pattern: str) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def build_source_manifest(paths: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for key, value in paths.items():
        p = Path(value) if value else Path("")
        rows.append({
            "source_key": key,
            "path": str(value) if value else "",
            "exists": bool(value and p.exists()),
            "is_file": bool(value and p.exists() and p.is_file()),
            "is_directory": bool(value and p.exists() and p.is_dir()),
            "sha256": file_sha256(p) if value and p.exists() and p.is_file() else "",
        })
    return pd.DataFrame(rows)
