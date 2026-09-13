#!/usr/bin/env python
"""
Internal utilities for spatial_transfer_inference_model.

This helper is not a numbered pipeline step.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_sep(path: Path) -> str:
    path = Path(path)
    return "\t" if path.suffix.lower() in [".tsv", ".tab"] else ","


def read_table(path: Path, nrows: Optional[int] = None, usecols: Optional[Sequence[str]] = None, keep_default_na: bool = True) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=choose_sep(path)))
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate raw table headers: {path}")
    return pd.read_csv(path, sep=choose_sep(path), nrows=nrows, usecols=usecols, low_memory=False, keep_default_na=keep_default_na)


def read_header(path: Path) -> List[str]:
    return list(read_table(path, nrows=0).columns)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_text_report(path: Path, body: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")


def open_folder(path: Path) -> None:
    path = Path(path)
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def safe_slug(value: object, max_len: int = 120) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "blank"
    return text[:max_len]


def sigmoid(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def choose_col(columns: Sequence[str], candidates: Sequence[str], required: bool = False, label: str = "column") -> Optional[str]:
    original_by_lower = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in original_by_lower:
            return original_by_lower[candidate.lower()]
    if required:
        raise ValueError(f"Could not find {label}. Tried {candidates}. Available columns include {list(columns)[:25]}")
    return None


def first_existing(paths: Sequence[Path], required: bool = True, label: str = "file") -> Optional[Path]:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"Could not find required {label}. Tried: " + "; ".join(str(p) for p in paths))
    return None


def add_qc(rows: List[dict], check_id: str, status: str, observed: object, expected: object, detail: str) -> None:
    rows.append({
        "check_id": check_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })


def build_output_manifest(root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    root = Path(root)
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            rows.append({
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "absolute_path": str(path),
                "size_bytes": size,
                "suffix": path.suffix.lower(),
            })
    return pd.DataFrame(rows)


def save_output_manifest(output_root: Path) -> Path:
    path = Path(output_root) / "spatial_transfer_inference_model_output_manifest.tsv"
    write_tsv(path, build_output_manifest(Path(output_root)))
    return path


def pim_file(pim_run_root: Path, relative_candidates: Sequence[str], required: bool = True, label: str = "PIM file") -> Optional[Path]:
    candidates = [Path(pim_run_root) / rel for rel in relative_candidates]
    return first_existing(candidates, required=required, label=label)


def load_pim_feature_dictionary(pim_run_root: Path) -> pd.DataFrame:
    path = pim_file(
        pim_run_root,
        [
            "07_final_outputs/01_publication_tables_tsv/Final_Feature_Dictionary.tsv",
            "02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv",
        ],
        required=True,
        label="PIM strict feature dictionary",
    )
    df = read_table(path)
    feature_col = choose_col(df.columns, ["feature_name", "feature", "spatial_feature"], required=True, label="feature dictionary feature column")
    if feature_col != "feature_name":
        df = df.rename(columns={feature_col: "feature_name"})
    df["feature_name"] = df["feature_name"].astype(str)
    return df


def strict_feature_names(pim_run_root: Path) -> List[str]:
    df = load_pim_feature_dictionary(pim_run_root)
    return sorted(df["feature_name"].dropna().astype(str).unique())


def load_pim_treatment_cards(pim_run_root: Path) -> pd.DataFrame:
    path = pim_file(
        pim_run_root,
        [
            "07_final_outputs/01_publication_tables_tsv/Final_Treatment_Interpretation_Cards.tsv",
            "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv",
        ],
        required=True,
        label="PIM treatment interpretation cards",
    )
    return read_table(path)


def load_pim_signed_feature_effects(pim_run_root: Path) -> pd.DataFrame:
    path = pim_file(
        pim_run_root,
        [
            "07_final_outputs/01_publication_tables_tsv/Final_Signed_Treatment_Feature_Effects.tsv",
            "03_signed_spatial_effects/01_treatment_feature_effects/signed_treatment_feature_effects.tsv",
        ],
        required=True,
        label="PIM signed treatment feature effects",
    )
    return read_table(path)


def load_pim_signed_theme_effects(pim_run_root: Path) -> pd.DataFrame:
    path = pim_file(
        pim_run_root,
        [
            "07_final_outputs/01_publication_tables_tsv/Final_Signed_Treatment_Theme_Effects.tsv",
            "03_signed_spatial_effects/02_treatment_theme_effects/signed_treatment_theme_effects.tsv",
        ],
        required=True,
        label="PIM signed treatment theme effects",
    )
    return read_table(path)


def load_pim_spatial_feature_pool(pim_run_root: Path, usecols: Optional[Sequence[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    path = pim_file(
        pim_run_root,
        [
            "01_prepared_inputs/02_copied_v2_tables/tables/v2_spatial_features_broad_pool.tsv",
        ],
        required=True,
        label="PIM copied V2 spatial feature pool",
    )
    return read_table(path, usecols=usecols, nrows=nrows)


def summarize_examples(values: Iterable[object], max_items: int = 5) -> str:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value)
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            out.append(text)
            seen.add(text)
        if len(out) >= max_items:
            break
    return "; ".join(out)


def report_status_from_qc(qc: List[dict], errors: List[str]) -> str:
    if errors:
        return "fail"
    if any(str(row.get("status", "")).lower() == "fail" for row in qc):
        return "fail"
    return "pass"


def package_zip(zip_path: Path, root: Path, include_prefixes: Optional[Sequence[str]] = None) -> List[Path]:
    import zipfile

    zip_path = Path(zip_path)
    root = Path(root)
    ensure_dir(zip_path.parent)

    files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path == zip_path:
            continue
        rel = path.relative_to(root).as_posix()
        if include_prefixes and not any(rel.startswith(prefix) for prefix in include_prefixes):
            continue
        files.append(path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, path.relative_to(root).as_posix())

    return files


def sha256_file(path: Path, max_bytes: int = 128 * 1024 * 1024) -> str:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        return "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
# ============================================================
# Robust PIM loader overrides appended by transfer smoke-test patch.
# These definitions intentionally override earlier helper functions.
# ============================================================

def _stim_norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _stim_resolve_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    norm_to_original = {_stim_norm_col(c): c for c in columns}
    for alias in aliases:
        key = _stim_norm_col(alias)
        if key in norm_to_original:
            return norm_to_original[key]
    return None


def _stim_existing_candidate_paths(root: Path, relative_candidates: Sequence[str], filenames: Sequence[str]) -> List[Path]:
    root = Path(root)
    paths: List[Path] = []

    for rel in relative_candidates:
        p = root / rel
        if p.exists() and p.is_file():
            paths.append(p)

    seen = {str(p).lower() for p in paths}

    for filename in filenames:
        try:
            for p in root.rglob(filename):
                if p.is_file() and str(p).lower() not in seen:
                    paths.append(p)
                    seen.add(str(p).lower())
        except Exception:
            pass

    return paths


def _stim_read_first_valid_table(
    pim_run_root: Path,
    relative_candidates: Sequence[str],
    filenames: Sequence[str],
    required_columns: Sequence[str] | None = None,
    label: str = "PIM table",
) -> pd.DataFrame:
    candidates = _stim_existing_candidate_paths(Path(pim_run_root), relative_candidates, filenames)
    errors = []

    if required_columns is None:
        required_columns = []

    for path in candidates:
        try:
            df = read_table(path)
            if df.empty and required_columns:
                errors.append(f"{path}: empty table")
                continue

            missing = []
            for col in required_columns:
                if _stim_resolve_col(df.columns, [col]) is None:
                    missing.append(col)

            if missing:
                errors.append(f"{path}: missing normalized required columns {missing}; columns={list(df.columns)[:20]}")
                continue

            return df

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise FileNotFoundError(
        f"Could not load valid {label}. Tried {len(candidates)} candidates. "
        + " | ".join(errors[:8])
    )


def load_pim_feature_dictionary(pim_run_root: Path) -> pd.DataFrame:
    # Prefer raw Step 02 dictionary because it preserves machine-readable feature_name.
    df = _stim_read_first_valid_table(
        pim_run_root,
        [
            "02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv",
            "01_prepared_inputs/02_copied_v2_tables/tables/v2_strict_biology_feature_registry.tsv",
            "07_final_outputs/01_publication_tables_tsv/Final_Feature_Dictionary.tsv",
        ],
        [
            "strict_spatial_feature_dictionary.tsv",
            "v2_strict_biology_feature_registry.tsv",
            "Final_Feature_Dictionary.tsv",
        ],
        required_columns=[],
        label="PIM strict feature dictionary",
    )

    feature_col = _stim_resolve_col(
        df.columns,
        [
            "feature_name",
            "feature",
            "spatial_feature",
            "spatial_feature_name",
            "model_feature",
            "variable",
            "feature_id",
            "Feature Name",
            "Spatial Feature",
        ],
    )

    if feature_col is None:
        raise ValueError(
            "Loaded PIM feature dictionary, but could not identify feature column. "
            f"Columns: {list(df.columns)}"
        )

    if feature_col != "feature_name":
        df = df.rename(columns={feature_col: "feature_name"})

    df["feature_name"] = df["feature_name"].astype(str)
    df = df[df["feature_name"].notna()].copy()
    df = df[df["feature_name"].astype(str).str.lower() != "nan"].copy()
    df = df.drop_duplicates("feature_name").reset_index(drop=True)
    return df


def load_pim_treatment_cards(pim_run_root: Path) -> pd.DataFrame:
    # Prefer raw Step 04 card table because it preserves drug_key.
    df = _stim_read_first_valid_table(
        pim_run_root,
        [
            "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv",
            "07_final_outputs/01_publication_tables_tsv/Final_Treatment_Interpretation_Cards.tsv",
        ],
        [
            "treatment_interpretation_cards.tsv",
            "Final_Treatment_Interpretation_Cards.tsv",
        ],
        required_columns=[],
        label="PIM treatment interpretation cards",
    )

    drug_col = _stim_resolve_col(df.columns, ["drug_key", "treatment_key", "drug", "treatment", "Treatment Key"])
    if drug_col is None:
        raise ValueError(f"Treatment cards loaded but no drug_key-like column found. Columns: {list(df.columns)}")
    if drug_col != "drug_key":
        df = df.rename(columns={drug_col: "drug_key"})
    df["drug_key"] = df["drug_key"].astype(str)
    return df


def load_pim_signed_feature_effects(pim_run_root: Path) -> pd.DataFrame:
    # Prefer raw Step 03 effects because it preserves drug_key, feature_name, signed_effect.
    df = _stim_read_first_valid_table(
        pim_run_root,
        [
            "03_signed_spatial_effects/01_treatment_feature_effects/signed_treatment_feature_effects.tsv",
            "07_final_outputs/01_publication_tables_tsv/Final_Signed_Treatment_Feature_Effects.tsv",
        ],
        [
            "signed_treatment_feature_effects.tsv",
            "Final_Signed_Treatment_Feature_Effects.tsv",
        ],
        required_columns=[],
        label="PIM signed treatment feature effects",
    )

    rename = {}
    drug_col = _stim_resolve_col(df.columns, ["drug_key", "treatment_key", "drug", "treatment", "Treatment Key"])
    feature_col = _stim_resolve_col(df.columns, ["feature_name", "feature", "spatial_feature", "spatial_feature_name", "Feature Name"])
    signed_col = _stim_resolve_col(df.columns, ["signed_effect", "signed feature effect", "signed_feature_effect", "effect", "Signed Effect"])

    if drug_col is None or feature_col is None or signed_col is None:
        raise ValueError(
            "Signed feature effects loaded but required columns were not found. "
            f"drug_col={drug_col}; feature_col={feature_col}; signed_col={signed_col}; columns={list(df.columns)}"
        )

    if drug_col != "drug_key":
        rename[drug_col] = "drug_key"
    if feature_col != "feature_name":
        rename[feature_col] = "feature_name"
    if signed_col != "signed_effect":
        rename[signed_col] = "signed_effect"

    if rename:
        df = df.rename(columns=rename)

    df["drug_key"] = df["drug_key"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    return df


def load_pim_signed_theme_effects(pim_run_root: Path) -> pd.DataFrame:
    df = _stim_read_first_valid_table(
        pim_run_root,
        [
            "03_signed_spatial_effects/02_treatment_theme_effects/signed_treatment_theme_effects.tsv",
            "07_final_outputs/01_publication_tables_tsv/Final_Signed_Treatment_Theme_Effects.tsv",
        ],
        [
            "signed_treatment_theme_effects.tsv",
            "Final_Signed_Treatment_Theme_Effects.tsv",
        ],
        required_columns=[],
        label="PIM signed treatment theme effects",
    )

    rename = {}
    drug_col = _stim_resolve_col(df.columns, ["drug_key", "treatment_key", "drug", "treatment", "Treatment Key"])
    theme_col = _stim_resolve_col(df.columns, ["biological_theme", "biology_theme", "theme", "Biology Theme"])
    signed_col = _stim_resolve_col(df.columns, ["signed_theme_effect", "signed_effect", "effect", "Signed Theme Effect"])

    if drug_col is not None and drug_col != "drug_key":
        rename[drug_col] = "drug_key"
    if theme_col is not None and theme_col != "biological_theme":
        rename[theme_col] = "biological_theme"
    if signed_col is not None and signed_col != "signed_theme_effect":
        rename[signed_col] = "signed_theme_effect"

    if rename:
        df = df.rename(columns=rename)

    if "drug_key" in df.columns:
        df["drug_key"] = df["drug_key"].astype(str)
    return df


def load_pim_spatial_feature_pool(pim_run_root: Path, usecols: Optional[Sequence[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    candidates = _stim_existing_candidate_paths(
        Path(pim_run_root),
        [
            "01_prepared_inputs/02_copied_v2_tables/tables/v2_spatial_features_broad_pool.tsv",
            "01_prepared_inputs/02_copied_v2_tables/tables/v2_spatial_features_broad_governed_candidate_pool.tsv",
            "02_build_modeling_dataset/03_modeling_datasets/v2_spatial_features_broad_governed_candidate_pool.tsv",
        ],
        [
            "v2_spatial_features_broad_pool.tsv",
            "v2_spatial_features_broad_governed_candidate_pool.tsv",
        ],
    )

    errors = []
    for path in candidates:
        try:
            if usecols is not None:
                header = read_header(path)
                available = [c for c in usecols if c in header]
                missing = [c for c in usecols if c not in header]
                if not available:
                    errors.append(f"{path}: none of requested usecols are present")
                    continue
                # Keep pandas from failing if a future feature is absent.
                return read_table(path, usecols=available, nrows=nrows)
            return read_table(path, usecols=usecols, nrows=nrows)
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise FileNotFoundError(
        "Could not load PIM spatial feature pool. "
        + " | ".join(errors[:8])
    )

# Source-bound endpoint loading. Never use a module-global map to override an explicit run.
PIM_ENDPOINT_PATHS = {
    "feature_dictionary": "02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv",
    "signed_feature_effects": "03_signed_spatial_effects/01_treatment_feature_effects/signed_treatment_feature_effects.tsv",
    "signed_theme_effects": "03_signed_spatial_effects/02_treatment_theme_effects/signed_treatment_theme_effects.tsv",
    "treatment_cards": "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv",
    "spatial_feature_pool": "01_prepared_inputs/02_copied_v2_tables/tables/v2_spatial_features_broad_pool.tsv",
}


def _source_bound_table(pim_run_root, key, usecols=None, nrows=None):
    root = Path(pim_run_root).resolve()
    path = root / PIM_ENDPOINT_PATHS[key]
    if not path.is_file() and key == "spatial_feature_pool":
        # Large reference tables can be manifest-referenced instead of copied.
        index_path = root / "01_prepared_inputs/01_source_manifests/prepared_interpretation_source_index.tsv"
        index = read_table(index_path)
        hit = index[index.source_id == "v2_spatial_features_broad_pool"]
        if len(hit) != 1:
            raise ValueError("Ambiguous or missing spatial reference in selected PIM source index")
        value = hit.iloc[0]["source_path"]
        path = Path(value)
        summary = json.loads((root / "prediction_interpretation_model_step01_summary.json").read_text())
        v2_root = Path(summary["v2_run_root"]).resolve()
        if not path.resolve().is_relative_to(v2_root):
            raise ValueError("Reference source lies outside selected PIM's V2 run")
    if not path.is_file():
        raise FileNotFoundError(f"Selected PIM run lacks {key}: {path}")
    return read_table(path, nrows=nrows, usecols=usecols)


def _unique_endpoint_table(root, key, columns):
    from _alignment_contract import require_unique
    result = _source_bound_table(root, key)
    require_unique(result, columns, key)
    return result


def load_pim_feature_dictionary(pim_run_root):
    return _unique_endpoint_table(pim_run_root, "feature_dictionary", ["feature_name"])


def strict_feature_names(pim_run_root):
    return load_pim_feature_dictionary(pim_run_root).feature_name.astype(str).tolist()


def load_pim_spatial_feature_pool(pim_run_root, usecols=None, nrows=None):
    return _source_bound_table(pim_run_root, "spatial_feature_pool", usecols, nrows)


def load_pim_signed_feature_effects(pim_run_root):
    return _unique_endpoint_table(pim_run_root, "signed_feature_effects", ["drug_key", "feature_name"])


def load_pim_signed_theme_effects(pim_run_root):
    return _unique_endpoint_table(pim_run_root, "signed_theme_effects", ["drug_key", "biological_theme"])


def load_pim_treatment_cards(pim_run_root):
    return _unique_endpoint_table(pim_run_root, "treatment_cards", ["drug_key"])
