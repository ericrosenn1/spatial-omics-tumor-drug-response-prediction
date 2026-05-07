"""
Script: expression_model_v2_lib.py

Purpose:
    Shared utility library for expression_response_model_v2.

Project context:
    This module centralizes reusable helpers for the deployable expression
    response teacher workflow. The numbered scripts use these utilities to
    load YAML configuration, resolve project-relative paths, harmonize
    treatment names, encode binary response labels, discover expression gene
    columns, write/read JSON and tabular outputs, calibrate probabilities, and
    compute teacher reliability weights.

Scientific role:
    The expression_response_model_v2 pipeline trains calibrated, deployable
    transcriptomic response models that can be audited before teacher_builder
    consumes them. Keeping these helpers small and explicit makes it easier to
    verify that treatment keys, labels, calibration metrics, and shrinkage logic
    are used consistently across training, audit, and Visium teacher scoring.

Documentation polish marker:
    EXPRESSION_MODEL_V2_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""

# =========================
# Imports
# =========================

from __future__ import annotations

from pathlib import Path
import re
import json
import math
import hashlib
from typing import Any

import numpy as np
import pandas as pd

import yaml




# =========================
# Missing-value and treatment-alias constants
# =========================
# Centralized constants keep text cleaning and treatment harmonization
# consistent across validation, training, audit, and teacher scoring.

MISSING_STRINGS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not reported",
    "not applicable",
    "not available",
    "not submitted",
    "unknown",
    "missing",
}



# =========================
# Canonical treatment alias map
# =========================
# These aliases collapse common salt forms, brand names, and formatting
# variants onto the canonical treatment keys used by downstream joins.

DEFAULT_ALIASES = {
    "gemcitabine hydrochloride": "gemcitabine",
    "leucovorin calcium": "leucovorin",
    "leuprolide acetate": "leuprolide",
    "doxorubicin hydrochloride": "doxorubicin",
    "irinotecan hydrochloride": "irinotecan",
    "vinorelbine tartrate": "vinorelbine",
    "sorafenib tosylate": "sorafenib",
    "pazopanib hydrochloride": "pazopanib",
    "pegylated liposomal doxorubicin hydrochloride": "doxorubicin",
    "erlotinib hydrochloride": "erlotinib",
    "trastuzumab deruxtecan": "trastuzumab deruxtecan",
    "nab paclitaxel": "paclitaxel",
    "nab-paclitaxel": "paclitaxel",
    "xeloda": "capecitabine",
}




# =========================
# Configuration and path helpers
# =========================
# All numbered scripts use these helpers to keep path resolution
# project-relative and reproducible across machines.

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file for the expression response model workflow."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root_from_config(cfg: dict[str, Any]) -> Path:
    """Resolve the configured project root as an absolute Path."""

    return Path(cfg["project_root"]).resolve()


def resolve_path(cfg: dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve an absolute or project-relative path from the pipeline configuration."""

    if value in [None, ""]:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root_from_config(cfg) / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p




# =========================
# Text normalization helpers
# =========================
# Clinical treatment and response tables contain mixed missing-value
# tokens, punctuation, capitalization, and whitespace conventions.

def clean_text(value: Any) -> str:
    """Return a stripped string representation while normalizing common whitespace artifacts."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.strip("'\". ").strip()
    return text


def is_present(value: Any) -> bool:
    """Return True when a value is not blank, missing, or a placeholder token."""

    text = clean_text(value).lower()
    if text == "":
        return False
    if set(text) == {"-"}:
        return False
    if text in MISSING_STRINGS:
        return False
    return True


def normalize_key(value: Any) -> str:
    """Normalize free-text labels into lowercase keys used for joins and comparisons."""

    text = clean_text(value).lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_name(value: Any, max_len: int = 140) -> str:
    """Convert a free-text label into a filesystem-safe lowercase name."""

    text = normalize_key(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    return text[:max_len]




# =========================
# Treatment harmonization helpers
# =========================
# Treatment keys are the bridge between GDC expression labels, model
# artifacts, teacher_builder, and spatial_prediction_model.

def canonical_drug_name(value: Any, aliases: dict[str, str] | None = None) -> str:
    """Map a raw treatment label to a canonical single-drug key."""

    aliases = aliases or DEFAULT_ALIASES
    key = normalize_key(value)
    # Remove historical "drug" prefixes before alias lookup.
    key = re.sub(r"^drug\s+", "", key)
    key = re.sub(r"^drug__", "", key)
    key = re.sub(r"^drug_", "", key)
    key = aliases.get(key, key)
    return key


def treatment_components(value: Any, aliases: dict[str, str] | None = None) -> list[str]:
    """Split a single-agent or combination treatment label into canonical components."""

    text = normalize_key(value)
    # Treat common regimen separators as equivalent component delimiters.
    text = re.sub(r"\s+and\s+", " | ", text)
    text = re.sub(r"\s*[+/;,]\s*", " | ", text)
    parts = [canonical_drug_name(part, aliases=aliases) for part in text.split("|")]
    parts = [p for p in parts if p]
    return sorted(set(parts))


def canonical_regimen_key(value: Any, aliases: dict[str, str] | None = None) -> str:
    """Return a stable pipe-delimited canonical regimen key."""

    return " | ".join(treatment_components(value, aliases=aliases))




# =========================
# Response labels and expression feature discovery
# =========================
# Binary response labels and expression-gene detection are intentionally
# kept simple and auditable before model fitting.

def encode_binary_response(value: Any) -> float:
    """Encode responder/non-responder labels as 1.0, 0.0, or NaN."""

    text = normalize_key(value)
    if text == "responder":
        return 1.0
    if text in {"non_responder", "non responder", "non-responder"}:
        return 0.0
    return np.nan


def find_gene_columns(df: pd.DataFrame, gene_prefix: str = "ENSG") -> list[str]:
    """Return expression feature columns matching the configured gene prefix."""

    return [c for c in df.columns if str(c).startswith(gene_prefix)]




# =========================
# Table and JSON I/O helpers
# =========================
# Small wrappers keep delimiter handling and UTF-8 JSON output consistent
# without hiding the underlying pandas/json behavior.

def read_table(path: str | Path, sep: str | None = None, **kwargs) -> pd.DataFrame:
    """Read a CSV or TSV table with delimiter inferred from the file suffix."""

    p = Path(path)
    if sep is None:
        sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(p, sep=sep, low_memory=False, **kwargs)


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary-like object to a UTF-8 JSON file."""

    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file into a Python dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))




# =========================
# Calibration diagnostics and probability transforms
# =========================
# These helpers support calibrated deployable model artifacts rather
# than raw classifier scores.

def sigmoid_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to logits after clipping away from 0 and 1."""

    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    """Compute equal-width-bin expected calibration error for binary probabilities."""

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return float("nan")
    # Use equal-width probability bins for a compact calibration summary.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if int(m.sum()) == 0:
            continue
        ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)




# =========================
# Teacher reliability weighting
# =========================
# Reliability weights summarize discrimination, calibration, fold success,
# and sample support so weak models can be down-weighted downstream.

def reliability_weight(
    auc: float,
    brier_improvement: float,
    prior_brier: float,
    ece: float,
    n_rows: int,
    n_folds_ok: int,
    n_folds_requested: int,
) -> float:
    """Compute a bounded teacher reliability weight from performance and calibration diagnostics."""

    if not np.isfinite(auc):
        auc_component = 0.0
    else:
        auc_component = np.clip((auc - 0.5) / 0.30, 0.0, 1.0)

    if not np.isfinite(brier_improvement) or not np.isfinite(prior_brier) or prior_brier <= 0:
        brier_component = 0.0
    else:
        brier_component = np.clip(brier_improvement / prior_brier, 0.0, 1.0)

    if not np.isfinite(ece):
        cal_component = 0.0
    else:
        cal_component = np.clip(1.0 - ece / 0.25, 0.0, 1.0)

    n_component = np.clip(n_rows / 120.0, 0.0, 1.0)
    fold_component = np.clip(n_folds_ok / max(1, n_folds_requested), 0.0, 1.0)

    # Combine bounded quality components into a single teacher-use weight.
    weight = (
        0.35 * auc_component
        + 0.30 * brier_component
        + 0.20 * cal_component
        + 0.15 * fold_component
    )

    weight *= n_component
    return float(np.clip(weight, 0.0, 1.0))




# =========================
# Calibration application and probability shrinkage
# =========================
# Teacher scoring applies the saved calibrator, then can shrink predictions
# toward treatment priors before handing them to teacher_builder.

def apply_calibrator(raw_prob: np.ndarray, method: str, calibrator: Any) -> np.ndarray:
    """Apply an identity, sigmoid, or isotonic calibrator to raw probabilities."""

    raw_prob = np.asarray(raw_prob, dtype=float)
    raw_prob = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)

    if calibrator is None or method == "identity":
        return raw_prob

    # Sigmoid calibrators are trained on the logit probability scale.
    if method == "sigmoid":
        x = sigmoid_logit(raw_prob).reshape(-1, 1)
        return calibrator.predict_proba(x)[:, 1]

    if method == "isotonic":
        return calibrator.transform(raw_prob)

    return raw_prob


def shrink_probability(prob: np.ndarray, prior: float, reliability: float) -> np.ndarray:
    """Shrink model probabilities toward a treatment prior using a reliability weight."""

    prob = np.asarray(prob, dtype=float)
    # Only the reliability-supported delta is allowed to move away from the prior.
    return prior + float(reliability) * (prob - prior)
