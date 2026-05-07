#!/usr/bin/env python
"""
Module:
    _pim_utils.py

Description:
    Shared utility functions used by prediction_interpretation_model scripts.
    Helpers centralize file I/O, QC rows, source-index lookup, naming, statistics,
    component parsing, and report-writing conventions.

Instructions:
    Import this module from numbered step scripts only. Keep helper behavior small,
    explicit, deterministic, and compatible with the pipeline's FILEPATH-first
    report convention.

Source-truth policy:
    Utilities support interpretation-layer data movement and reporting only. They
    should not rerun V2, perform model selection, or create clinical recommendations.
"""

# =============================================================================
# PIM_DOCS_PATCH: RUN AND MAINTENANCE INSTRUCTIONS
# =============================================================================
# Run numbered scripts through 00_run_prediction_interpretation_model.py unless
# debugging a single step. Treat the V2 full-run root as read-only source truth.
# Every generated .txt report must start with FILEPATH, and terminal summaries
# should remain concise enough for copy/paste debugging.
# =============================================================================


# =============================================================================
# PIM_DOCS_SECTION: imports and dependencies
# =============================================================================
# Keep imports explicit and standard-library-first where practical. The pipeline
# expects local scripts to run from the scripts directory or through the orchestrator.

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PIM_DOCS_SECTION: constants and source contracts
# =============================================================================
# Constants define expected files, output names, QC contracts, or reporting rules.

TEXT_EXTS = {".txt", ".md"}
TABLE_EXTS = {".tsv", ".tab", ".csv"}


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def now_stamp() -> str:
    """Return a filesystem-safe timestamp string.
    Used for run names, patch logs, and reproducible report folders."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it as a Path.
    Keeps output-folder creation explicit and reusable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_sep(path: Path) -> str:
    """Infer the delimiter for a CSV/TSV-style table path.
    TSV and TAB files use tab; other table files default to comma."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def read_table(path: Path, nrows: Optional[int] = None, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a CSV/TSV table with the project delimiter convention.
    Supports optional row and column restrictions for large V2 files."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    return pd.read_csv(path, sep=choose_sep(path), nrows=nrows, usecols=usecols, low_memory=False)


def read_header(path: Path) -> List[str]:
    """Read only the header row from a table.
    Avoids loading large V2 source tables when only the schema is needed."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return list(read_table(path, nrows=0).columns)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    """Write a pandas DataFrame as a tab-separated table.
    Creates parent folders before writing the output artifact."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def write_json(path: Path, data: object) -> None:
    """Write structured metadata as formatted JSON.
    Creates parent folders and preserves readable provenance output."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_text_report(path: Path, body: str) -> None:
    """Write a text report with FILEPATH on the first line.
    This convention is required for all generated text reports."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")


def open_folder(path: Path) -> None:
    """Open an output folder in the local operating system.
    Failures are intentionally nonfatal so batch runs can continue."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
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
    """Convert arbitrary text into a filesystem-safe slug.
    Used for stable filenames and compact artifact names."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "blank"
    return text[:max_len]


def safe_filename(value: object, max_len: int = 96) -> str:
    """Create a safe filename with a short hash suffix.
    Prevents collisions for long treatment keys and card names."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = safe_slug(value, max_len=max_len)
    digest = hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{text}_{digest}"


def build_output_manifest(root: Path) -> pd.DataFrame:
    """Inventory files under an output root.
    Captures relative paths, absolute paths, file sizes, and suffixes."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows: List[dict] = []
    root = Path(root)
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
    """Write or refresh the run-level output manifest.
    Called after steps create new tables, reports, or packages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = Path(output_root) / "prediction_interpretation_model_output_manifest.tsv"
    write_tsv(path, build_output_manifest(Path(output_root)))
    return path


def add_qc(rows: List[dict], check_id: str, status: str, observed: object, expected: object, detail: str) -> None:
    """Append one structured QC check row.
    Keeps status, observed value, expected value, and detail together."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows.append({
        "check_id": check_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })


def load_prepared_index(output_root: Path, prepared_input_root: Optional[Path] = None) -> Tuple[Path, pd.DataFrame]:
    """Load the Step 01 prepared source index.
    Returns both the prepared input root and index DataFrame."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if prepared_input_root is None:
        prepared_input_root = Path(output_root) / "01_prepared_inputs"
    prepared_input_root = Path(prepared_input_root)
    index_path = prepared_input_root / "01_source_manifests" / "prepared_interpretation_source_index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(f"Prepared source index not found: {index_path}")
    return prepared_input_root, read_table(index_path)


def source_path(index_df: pd.DataFrame, source_id: str, prefer_copied: bool = True, required: bool = True) -> Optional[Path]:
    """Resolve a source_id to an existing file path.
    Uses copied, preferred, or source paths according to read policy."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    hit = index_df[index_df["source_id"].astype(str) == str(source_id)].copy()
    if hit.empty:
        if required:
            raise KeyError(f"source_id not found in prepared source index: {source_id}")
        return None

    row = hit.iloc[0].to_dict()
    candidates: List[str] = []
    if prefer_copied:
        candidates.extend([str(row.get("copied_path", "")), str(row.get("preferred_read_path", "")), str(row.get("source_path", ""))])
    else:
        candidates.extend([str(row.get("source_path", "")), str(row.get("preferred_read_path", "")), str(row.get("copied_path", ""))])

    for value in candidates:
        if value and value.lower() != "nan":
            path = Path(value)
            if path.exists():
                return path

    if required:
        raise FileNotFoundError(f"No existing path found for source_id {source_id}: {candidates}")
    return None


def read_source_table(index_df: pd.DataFrame, source_id: str, prefer_copied: bool = True, required: bool = True) -> pd.DataFrame:
    """Read a prepared source table by source_id.
    Centralizes lookup through the Step 01 prepared source index."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path = source_path(index_df, source_id, prefer_copied=prefer_copied, required=required)
    if path is None:
        return pd.DataFrame()
    return read_table(path)


def choose_col(columns: Sequence[str], candidates: Sequence[str], required: bool = False, label: str = "column") -> Optional[str]:
    """Select the first available column from candidate names.
    Raises a clear error when a required column is absent."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    original_by_lower = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in original_by_lower:
            return original_by_lower[candidate.lower()]
    if required:
        raise ValueError(f"Could not find {label}; tried {candidates}; available columns include {list(columns)[:25]}")
    return None


def numeric_series(values: pd.Series) -> pd.Series:
    """Convert a Series to numeric float values.
    Invalid entries become NaN so downstream statistics remain explicit."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return pd.to_numeric(values, errors="coerce").astype(float)


def finite_pair(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return paired finite numeric values from two Series.
    Used before correlation calculations to remove missing values."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    xx = numeric_series(x)
    yy = numeric_series(y)
    mask = np.isfinite(xx.values) & np.isfinite(yy.values)
    return xx.loc[mask], yy.loc[mask]


def corr_pair(x: pd.Series, y: pd.Series, method: str = "pearson") -> Tuple[float, int]:
    """Compute a correlation and the number of usable pairs.
    Returns NaN when there are too few values or no variance."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    xx, yy = finite_pair(x, y)
    n = int(len(xx))
    if n < 6:
        return float("nan"), n
    if float(xx.std(ddof=0)) == 0.0 or float(yy.std(ddof=0)) == 0.0:
        return float("nan"), n
    try:
        value = float(xx.corr(yy, method=method))
    except Exception:
        value = float("nan")
    return value, n


def humanize_feature(feature_name: object) -> str:
    """Convert a model feature name into a readable label.
    Used in dictionaries, cards, reports, and final tables."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(feature_name)
    text = re.sub(r"^feature__", "", text)
    text = text.replace("__", " / ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_feature_group(feature_name: object) -> str:
    """Infer a broad feature group from a feature name.
    Provides a fallback grouping when upstream metadata is incomplete."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(feature_name)
    if "__" in text:
        return text.split("__", 1)[0]
    if "_" in text:
        return text.split("_", 1)[0]
    return "other"


def clean_component(value: object) -> str:
    """Normalize a treatment-component string.
    Lowercases and compresses whitespace for rule-based classification."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


COMPONENT_CLASS_RULES: List[Tuple[str, List[str]]] = [
    ("platinum_or_DNA_crosslinking", ["cisplatin", "carboplatin", "oxaliplatin"]),
    ("taxane_or_microtubule", ["paclitaxel", "docetaxel", "cabazitaxel", "vinblastine", "vincristine", "vinorelbine", "ixabepilone", "eribulin"]),
    ("antimetabolite_or_nucleotide", ["fluorouracil", "capecitabine", "gemcitabine", "pemetrexed", "methotrexate", "cytarabine", "doxifluridine", "floxuridine", "leucovorin"]),
    ("topoisomerase_or_anthracycline", ["doxorubicin", "epirubicin", "etoposide", "irinotecan", "topotecan", "mitoxantrone"]),
    ("alkylating_or_other_cytotoxic", ["cyclophosphamide", "ifosfamide", "dacarbazine", "temozolomide", "lomustine", "carmustine", "mitomycin", "bleomycin", "hydroxyurea"]),
    ("anti_angiogenic", ["bevacizumab", "aflibercept", "cediranib", "brivanib", "tivozanib"]),
    ("targeted_kinase_mtor_egfr", ["erlotinib", "dasatinib", "imatinib", "sorafenib", "everolimus", "temsirolimus", "lapatinib", "cetuximab", "panitumumab", "regorafenib", "teprotumumab"]),
    ("endocrine_or_androgen_axis", ["anastrozole", "leuprolide", "goserelin", "degarelix", "abiraterone", "bicalutamide", "tamoxifen", "letrozole", "exemestane", "fulvestrant", "megestrol"]),
    ("immune_or_vaccine", ["pembrolizumab", "ipilimumab", "nivolumab", "bcg", "vaccine"]),
    ("local_radiation_or_ablation", ["radiation", "ablation", "radiofrequency", "ethanol injection"]),
    ("steroid_or_supportive_context", ["hydrocortisone", "dexamethasone", "prednisone", "cyanocobalamin"]),
]


def classify_component(value: object) -> str:
    """Assign a descriptive class to a treatment component.
    Classes are reporting categories, not clinical recommendations."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = clean_component(value)
    for label, needles in COMPONENT_CLASS_RULES:
        if any(needle in text for needle in needles):
            return label
    if not text or text == "nan":
        return "unknown_or_blank"
    return "other_or_unclassified"


def parse_treatment_components(drug_key: object) -> List[str]:
    """Split a treatment key into component names.
    Treatment keys use pipe-separated components from upstream harmonization."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(drug_key)
    parts = [clean_component(x) for x in text.split("|")]
    parts = [p for p in parts if p and p != "nan"]
    return parts


def summarize_examples(values: Iterable[object], max_items: int = 5) -> str:
    """Create a compact semicolon-separated example list.
    Used to keep atlas and report fields readable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
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


def effect_label(corr_value: object, positive_label: str = "sensitivity_associated", negative_label: str = "resistance_associated") -> str:
    """Convert a signed numeric effect into a direction label.
    Positive and negative labels are supplied by the caller."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        value = float(corr_value)
    except Exception:
        return "ambiguous_no_numeric_direction"
    if not math.isfinite(value):
        return "ambiguous_no_numeric_direction"
    if value > 0:
        return positive_label
    if value < 0:
        return negative_label
    return "ambiguous_zero_direction"


def evidence_grade(abs_corr: object, n: object) -> str:
    """Assign a qualitative evidence grade from correlation and N.
    Grades summarize association strength without implying causality."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        c = abs(float(abs_corr))
        nn = int(n)
    except Exception:
        return "insufficient"
    if nn < 20:
        return "low_n_screening"
    if c >= 0.45:
        return "strong_directional_association"
    if c >= 0.30:
        return "moderate_directional_association"
    if c >= 0.15:
        return "weak_directional_association"
    return "very_weak_or_ambiguous"


def zscore_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Z-score selected numeric columns in a DataFrame.
    Zero-variance columns are set to zero to avoid divide-by-zero errors."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    out = df.copy()
    for col in columns:
        vals = numeric_series(out[col]) if col in out.columns else pd.Series(dtype=float)
        mean = float(vals.mean()) if vals.notna().any() else 0.0
        std = float(vals.std(ddof=0)) if vals.notna().any() else 0.0
        if not math.isfinite(std) or std == 0.0:
            out[col] = 0.0
        else:
            out[col] = (vals - mean) / std
    return out


def selected_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy containing only available requested columns.
    Missing columns are skipped intentionally for flexible schemas."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    keep = [c for c in columns if c in df.columns]
    return df[keep].copy()
