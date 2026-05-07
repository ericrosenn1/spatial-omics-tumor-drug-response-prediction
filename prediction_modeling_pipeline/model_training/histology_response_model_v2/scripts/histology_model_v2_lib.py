"""
Script: histology_model_v2_lib.py

Purpose:
    Shared utility layer for histology_response_model_v2.

Pipeline role:
    Provides common configuration, path, text normalization, treatment
    harmonization, response-label encoding, table I/O, and metric helpers used
    by the numbered histology response modeling scripts.

Scientific context:
    This library keeps treatment naming, response mapping, and output handling
    consistent across the GDC named-treatment histology response teacher
    workflow. Those shared conventions are important because downstream
    teacher_builder uses the audited histology model as a conservative base
    teacher for spatial prediction.

Documentation safety:
    This file has been documented for readability and auditability. Documentation
    edits should not change executable behavior, thresholds, paths, schemas,
    model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Shared constants
# =============================================================================

# Clinical exports use many missing-value sentinels; centralizing them keeps all steps consistent.
MISSING_STRINGS = {
    "", "na", "n/a", "nan", "none", "null", "not reported", "not applicable",
    "unknown", "not available", "not allowed to collect", "not submitted", "missing", "--", "---", "."
}

# Generic therapy categories are excluded so teacher labels are tied to named agents/regimens.
GENERIC_TREATMENTS = {
    "chemotherapy", "radiation therapy", "radiation therapy, nos", "radiotherapy",
    "immunotherapy", "immunotherapy (including vaccines)", "pharmaceutical therapy",
    "pharmaceutical therapy, nos", "hormone therapy", "targeted therapy",
    "targeted molecular therapy", "therapy, nos", "surgery", "surgery, nos",
}

# Alias harmonization prevents spelling and salt-form variants from becoming separate treatment keys.
ALIASES = {
    "gemcitabine hydrochloride": "gemcitabine",
    "leucovorin calcium": "leucovorin",
    "leuprolide acetate": "leuprolide",
    "doxorubicin hydrochloride": "doxorubicin",
    "irinotecan hydrochloride": "irinotecan",
    "vinorelbine tartrate": "vinorelbine",
    "sorafenib tosylate": "sorafenib",
    "pazopanib hydrochloride": "pazopanib",
    "erlotinib hydrochloride": "erlotinib",
    "capecitabine hydrochloride": "capecitabine",
}



# =============================================================================
# Configuration and path helpers
# =============================================================================

def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file into a Python dictionary."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(cfg: dict, value: str | Path | None, base: str | Path | None = None) -> Path | None:
    """Resolve a configured relative or absolute path against the project root."""
    if value in [None, ""]:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    if base is not None:
        return Path(base) / p
    return Path(cfg.get("project_root", ".")) / p


def output_root(cfg: dict) -> Path:
    """Resolve and create the configured histology output root."""
    return ensure_dir(resolve_path(cfg, cfg["output_root"]))



# =============================================================================
# Text, treatment, and response normalization
# =============================================================================

def clean_text(x: Any) -> str:
    """Normalize missing clinical text values to a clean string."""
    if pd.isna(x):
        return ""
    text = str(x).strip().replace("\u00a0", " ").strip("'\". ").strip()
    if text.lower() in MISSING_STRINGS:
        return ""
    if text and set(text) == {"-"}:
        return ""
    return text


def normalize_text(x: Any) -> str:
    """Return lowercase whitespace-normalized text for matching."""
    text = clean_text(x).lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_multi(x: Any) -> list[str]:
    """Split multi-agent treatment strings into cleaned components."""
    text = clean_text(x)
    if not text:
        return []
    text = re.sub(r"\s+and\s+", " | ", text, flags=re.I)
    text = re.sub(r"\s*[+/;,]\s*", " | ", text)
    return [clean_text(v) for v in text.split("|") if clean_text(v)]


def canonical_component(x: Any) -> str:
    """Map one treatment component to its canonical drug key."""
    text = normalize_text(x)
    text = re.sub(r"^drug\s+", "", text)
    text = re.sub(r"^drug__", "", text)
    text = re.sub(r"^drug_", "", text)
    return ALIASES.get(text, text)


def treatment_components(*values: Any) -> list[str]:
    """Extract named treatment components while excluding generic therapy labels."""
    out = []
    for value in values:
        for part in split_multi(value):
            comp = canonical_component(part)
            if comp and comp not in GENERIC_TREATMENTS:
                out.append(comp)
    return sorted(set(out))


def canonical_regimen(*values: Any) -> str:
    """Build a stable regimen key from one or more treatment text fields."""
    comps = treatment_components(*values)
    return " | ".join(comps)


def is_specific_treatment(*values: Any) -> bool:
    """Return whether at least one named treatment component is present."""
    return len(treatment_components(*values)) > 0


def response_label(raw: Any, responder_labels: Iterable[str], non_responder_labels: Iterable[str]) -> str:
    """Map a raw clinical response value to RESPONDER, NON_RESPONDER, or blank."""
    x = normalize_text(raw)
    resp = {normalize_text(v) for v in responder_labels}
    non = {normalize_text(v) for v in non_responder_labels}
    if x in resp:
        return "RESPONDER"
    if x in non:
        return "NON_RESPONDER"
    return ""


def response_id(label: Any) -> float:
    """Convert a binary response label to numeric model target encoding."""
    x = normalize_text(label)
    if x == "responder":
        return 1.0
    if x == "non responder" or x == "non_responder":
        return 0.0
    return np.nan



# =============================================================================
# Table I/O and metrics
# =============================================================================

def read_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Read a delimited table using extension-aware delimiter defaults."""
    p = Path(path)
    if sep is None:
        sep = "\t" if p.suffix.lower() in [".tsv", ".txt"] else ","
    return pd.read_csv(p, sep=sep, low_memory=False)


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON object with stable indentation."""
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(lines: List[str], path: str | Path) -> None:
    """Write a list of text lines to disk."""
    Path(path).write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def safe_auc(y_true, y_prob):
    """Compute ROC AUC when both response classes are present; otherwise return NaN."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(pd.Series(y_true).dropna().astype(int))) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def brier(y_true, y_prob):
    """Compute Brier score after removing non-finite labels and probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((y[mask] - p[mask]) ** 2))
