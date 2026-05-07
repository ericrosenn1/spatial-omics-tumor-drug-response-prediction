"""
Script: teacher_governance_lib.py

Purpose:
    Shared governance utility library for the Visium teacher_builder workflow.

Project context:
    teacher_builder fuses expression-response and histology-response teacher
    streams into governed sample-by-treatment labels for downstream spatial
    prediction modeling. This library centralizes reusable helpers for YAML
    configuration, path resolution, treatment-key harmonization, treatment
    prior construction, model-index parsing, probability shrinkage, label
    quality flags, and compact numeric summaries.

Scientific role:
    The governed teacher_builder workflow is intentionally conservative. It
    anchors response labels to treatment priors, uses reliability-weighted and
    confidence-weighted teacher deltas, preserves modality/provenance fields,
    and emits warning/exclusion metadata rather than silently trusting raw
    model probabilities.

Documentation polish marker:
    TEACHER_BUILDER_GOVERNANCE_LIB_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    constants, paths, thresholds, schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# The library intentionally uses lightweight dependencies shared by all
# teacher_builder steps: pathlib, JSON/YAML, NumPy, and pandas.

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np
import pandas as pd
import yaml




# =========================
# Configuration, path, and JSON helpers
# =========================
# These helpers keep every governed teacher_builder step on the same
# configuration and output-writing conventions.

def load_config(path: str | Path) -> dict:
    """Load a YAML teacher_builder configuration file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Write a JSON object to disk with parent-directory creation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path




# =========================
# Text and treatment-key normalization
# =========================
# Treatment names appear in several upstream model indexes and clinical
# tables, so canonical keys must be stable before fusion.

def clean_text(x: Any) -> str:
    """Return a stripped string while treating missing-like values as blank."""

    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_key(x: Any) -> str:
    """Normalize treatment or regimen text into a canonical pipe-delimited key."""

    x = clean_text(x).lower()
    x = x.replace("||", "|")
    x = x.replace(";", "|")
    x = x.replace(",", "|")
    x = x.replace("+", "|")
    x = re.sub(r"\s+", " ", x)
    parts = [p.strip() for p in x.split("|") if p.strip()]
    parts = [normalize_drug_component(p) for p in parts]
    parts = [p for p in parts if p]
    return " | ".join(parts)


def display_drug_name(key: str) -> str:
    """Convert a canonical treatment key into a display-friendly drug name."""

    key = normalize_key(key)
    return " | ".join([p[:1].upper() + p[1:] for p in key.split(" | ")])


def normalize_drug_component(x: Any) -> str:
    """Normalize one drug-name component and apply known synonym rewrites."""

    x = clean_text(x).lower()
    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = re.sub(r"\s+", " ", x).strip()

    # Collapse spelling, salt-form, and shorthand variants before treatment-key joins.
    synonyms = {
        "5 fu": "fluorouracil",
        "5 fluorouracil": "fluorouracil",
        "5-fluorouracil": "fluorouracil",
        "leucovorin calcium": "leucovorin",
        "folinic acid": "leucovorin",
        "gemcitabine hydrochloride": "gemcitabine",
        "vinorelbine tartrate": "vinorelbine",
        "doxorubicin hydrochloride": "doxorubicin",
        "pegylated liposomal doxorubicin hydrochloride": "doxorubicin",
        "cisplatinum": "cisplatin",
        "carboplatinum": "carboplatin",
        "xeloda": "capecitabine",
    }

    return synonyms.get(x, x)


def split_components(key: str) -> list[str]:
    """Split a normalized treatment/regimen key into component drug names."""

    key = normalize_key(key)
    if not key:
        return []
    return [p.strip() for p in key.split(" | ") if p.strip()]




# =========================
# Path and tabular I/O helpers
# =========================
# Config paths may be absolute or project-relative; table readers/writers
# standardize delimiter handling across teacher_builder outputs.

def resolve_path(cfg: dict, value: str | Path | None, project_key: str = "project_dir") -> Path | None:
    """Resolve an absolute or project-relative path from the YAML config."""

    if value in [None, ""]:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    project = Path(cfg.get(project_key, "."))
    return project / p


def cfg_path(cfg: dict, key: str, default: str | None = None) -> Path | None:
    """Resolve one named config path, with an optional default."""

    value = cfg.get(key, default)
    return resolve_path(cfg, value)


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV/TXT table with delimiter inferred from the suffix."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    low_memory = kwargs.pop("low_memory", False)

    if path.suffix.lower() in [".tsv", ".txt", ".tab"]:
        return pd.read_csv(path, sep="\t", low_memory=low_memory, **kwargs)
    return pd.read_csv(path, low_memory=low_memory, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, sep: str = "\t") -> None:
    """Write a DataFrame as a tabular output with parent-directory creation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from a case-insensitive candidate list."""

    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None




# =========================
# Scalar coercion and probability helpers
# =========================
# Governed fusion reads model indexes produced by different upstream
# pipelines, so numeric and boolean fields are coerced defensively.

def boolish(x: Any) -> bool:
    """Coerce common text and numeric approval flags to boolean values."""

    s = clean_text(x).lower()
    if s in ["true", "1", "yes", "y", "approved", "pass"]:
        return True
    if s in ["false", "0", "no", "n", "fail", ""]:
        return False
    return bool(x)


def safe_float(x: Any, default: float = np.nan) -> float:
    """Convert a value to float, returning a default for blank, NaN, or invalid input."""

    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        out = float(x)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def clip_prob(x: Any, low: float = 0.01, high: float = 0.99) -> float:
    """Clip a probability-like value to the configured bounded probability interval."""

    try:
        return float(np.clip(float(x), low, high))
    except Exception:
        return float("nan")


def confidence_from_probability(p: Any) -> float:
    """Compute confidence as distance from an uncertain 0.5 probability."""

    p = safe_float(p, np.nan)
    if not np.isfinite(p):
        return 0.0
    return float(np.clip(abs(p - 0.5) * 2.0, 0.0, 1.0))


def encode_response(x: Any) -> float:
    """Map response labels into numeric responder/non-responder encoding."""

    s = clean_text(x).lower()
    if s in ["1", "true", "responder", "response", "complete response", "partial response", "cr", "pr", "sensitive", "effective"]:
        return 1.0
    if s in ["0", "false", "non_responder", "non-responder", "non responder", "stable disease", "progressive disease", "sd", "pd", "resistant", "invalid"]:
        return 0.0
    return np.nan




# =========================
# Training-table schema detection
# =========================
# Expression teachers and priors may come from tables with slightly
# different treatment and response column names.

def find_gene_columns(df: pd.DataFrame) -> list[str]:
    """Return expression gene columns using the ENSG prefix convention."""

    return [c for c in df.columns if str(c).startswith("ENSG")]


def detect_treatment_col(df: pd.DataFrame) -> str | None:
    """Detect the treatment-key column in a model or training table."""

    return first_existing_column(
        df,
        [
            "canonical_treatment_key",
            "canonical_drug_key",
            "drug_key",
            "resolved_drug",
            "drug",
            "treatment",
            "treatment_key",
            "canonical_treatment",
        ],
    )


def detect_response_col(df: pd.DataFrame) -> str | None:
    """Detect the binary response column in a training table."""

    return first_existing_column(
        df,
        [
            "binary_response_label",
            "resolved_episode_binary_response",
            "response",
            "response_label",
            "is_responder",
            "y",
            "label",
        ],
    )




# =========================
# Treatment prior construction and lookup
# =========================
# Treatment priors are the anchor for governed teacher labels and residual
# targets used downstream by spatial prediction modeling.

def build_treatment_priors(training: pd.DataFrame, min_exact_n: int = 5) -> pd.DataFrame:
    """Estimate exact and global treatment response priors from training labels."""

    treatment_col = detect_treatment_col(training)
    response_col = detect_response_col(training)

    if treatment_col is None or response_col is None:
        raise ValueError("Could not detect treatment and response columns for treatment prior construction.")

    tab = training[[treatment_col, response_col]].copy()
    # Canonical treatment keys define the unit for treatment-prior lookup.
    tab["canonical_treatment_key"] = tab[treatment_col].map(normalize_key)
    tab["y"] = tab[response_col].map(encode_response)

    if tab["y"].isna().all():
        tab["y"] = pd.to_numeric(tab[response_col], errors="coerce")

    tab = tab.dropna(subset=["canonical_treatment_key", "y"]).copy()
    tab = tab[tab["canonical_treatment_key"].astype(str).str.len() > 0].copy()
    tab["y"] = tab["y"].astype(float)

    # Global prior is the fallback when exact treatment support is weak or absent.
    global_prior = float(tab["y"].mean()) if len(tab) else 0.5
    rows = []

    if len(tab):
        exact = (
            tab.groupby("canonical_treatment_key", as_index=False)
            .agg(prior_n=("y", "size"), prior_responders=("y", "sum"))
        )
        exact["prior_nonresponders"] = exact["prior_n"] - exact["prior_responders"]
        exact["prior_prob_responder"] = exact["prior_responders"] / exact["prior_n"]
        exact["prior_source"] = np.where(exact["prior_n"] >= min_exact_n, "exact", "exact_weak")
        exact["global_prior"] = global_prior
        rows.append(exact)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(
            columns=[
                "canonical_treatment_key",
                "prior_n",
                "prior_responders",
                "prior_nonresponders",
                "prior_prob_responder",
                "prior_source",
                "global_prior",
            ]
        )

    return out.sort_values(["prior_source", "canonical_treatment_key"]).reset_index(drop=True)


def lookup_prior(drug_key: str, priors: pd.DataFrame, global_prior: float | None = None) -> dict:
    """Look up a treatment response prior, falling back to global prior when needed."""

    key = normalize_key(drug_key)

    if priors.empty:
        gp = 0.5 if global_prior is None else float(global_prior)
        return {
            "treatment_prior": gp,
            "prior_source": "global_no_prior_table",
            "prior_n": 0,
            "prior_responders": np.nan,
            "prior_nonresponders": np.nan,
        }

    ptab = priors.copy()
    ptab["canonical_treatment_key"] = ptab["canonical_treatment_key"].map(normalize_key)

    if global_prior is None:
        if "global_prior" in ptab.columns and ptab["global_prior"].notna().any():
            global_prior = float(pd.to_numeric(ptab["global_prior"], errors="coerce").dropna().iloc[0])
        else:
            global_prior = float(pd.to_numeric(ptab["prior_prob_responder"], errors="coerce").mean())

    hit = ptab[ptab["canonical_treatment_key"] == key]

    if len(hit):
        r = hit.iloc[0]
        return {
            "treatment_prior": safe_float(r.get("prior_prob_responder"), global_prior),
            "prior_source": clean_text(r.get("prior_source", "exact")),
            "prior_n": int(safe_float(r.get("prior_n"), 0)),
            "prior_responders": safe_float(r.get("prior_responders"), np.nan),
            "prior_nonresponders": safe_float(r.get("prior_nonresponders"), np.nan),
        }

    comps = split_components(key)
    comp_hits = ptab[ptab["canonical_treatment_key"].isin(comps)]

    if len(comp_hits):
        ns = pd.to_numeric(comp_hits["prior_n"], errors="coerce").fillna(0).to_numpy()
        ps = pd.to_numeric(comp_hits["prior_prob_responder"], errors="coerce").fillna(global_prior).to_numpy()
        if ns.sum() > 0:
            prior = float(np.average(ps, weights=np.maximum(ns, 1)))
        else:
            prior = float(np.mean(ps))
        return {
            "treatment_prior": prior,
            "prior_source": "component_drug_prior",
            "prior_n": int(ns.sum()),
            "prior_responders": np.nan,
            "prior_nonresponders": np.nan,
        }

    return {
        "treatment_prior": float(global_prior),
        "prior_source": "global",
        "prior_n": 0,
        "prior_responders": np.nan,
        "prior_nonresponders": np.nan,
    }




# =========================
# Model artifact and index normalization
# =========================
# Model indexes are normalized into a common teacher_builder contract before
# expression or histology scores are consumed.

def find_artifact_path(row: pd.Series, base_dirs: list[Path]) -> Path | None:
    """Resolve a deployable model artifact path from index rows and candidate base folders."""

    preferred = [
        "model_path",
        "artifact_path",
        "model_file",
        "model_artifact",
        "calibrated_model_path",
        "pipeline_path",
        "joblib_path",
        "pickle_path",
    ]

    candidate_values = []

    for c in preferred:
        if c in row.index:
            candidate_values.append(row[c])

    for c in row.index:
        cl = str(c).lower()
        if ("path" in cl or "file" in cl) and any(tok in cl for tok in ["model", "artifact", "joblib", "pkl", "pickle"]):
            candidate_values.append(row[c])

    for value in candidate_values:
        s = clean_text(value)
        if not s:
            continue

        p = Path(s)

        if p.is_absolute() and p.exists():
            return p

        for base in base_dirs:
            q = base / p
            if q.exists():
                return q

    drug_key = normalize_key(row.get("canonical_treatment_key", row.get("drug_key", row.get("drug", ""))))
    safe = drug_key.replace(" | ", "__").replace(" ", "_")

    patterns = [
        f"*{safe}*.joblib",
        f"*{safe}*.pkl",
        f"*{safe}*.pickle",
        f"*{drug_key.replace(' | ', '_').replace(' ', '_')}*.joblib",
        f"*{drug_key.replace(' | ', '_').replace(' ', '_')}*.pkl",
    ]

    for base in base_dirs:
        if not base.exists():
            continue
        for pat in patterns:
            hits = sorted(base.rglob(pat))
            if hits:
                return hits[0]

    return None


def parse_model_index(index: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a model index into teacher_builder treatment, approval, and reliability fields.
    """

    df = index.copy()

    key_col = first_existing_column(
        df,
        [
            "canonical_treatment_key",
            "drug_key",
            "drug",
            "treatment_key",
            "canonical_drug",
            "treatment",
        ],
    )

    if key_col is None:
        raise ValueError("Could not detect treatment key column in model index.")

    df["canonical_treatment_key"] = df[key_col].map(normalize_key)
    df["drug_key"] = df["canonical_treatment_key"]
    df["drug"] = df["drug_key"].map(display_drug_name)

    rel_col = first_existing_column(
        df,
        [
            "reliability_weight",
            "model_reliability_weight",
            "teacher_weight",
            "approved_reliability_weight",
            "weight",
        ],
    )

    if rel_col is not None:
        df["reliability_weight"] = pd.to_numeric(df[rel_col], errors="coerce").fillna(0.0)
    else:
        df["reliability_weight"] = 0.0

    app_col = first_existing_column(df, ["approved_for_teacher", "approved", "teacher_approved", "use_for_teacher"])

    if app_col is not None:
        df["approved_for_teacher"] = df[app_col].map(boolish)
    else:
        df["approved_for_teacher"] = True

    return df




# =========================
# Governed probability shrinkage and label quality
# =========================
# Fusion uses reliability, sample confidence, and control factors to shrink
# teacher deltas toward treatment priors rather than trusting raw scores.

def shrink_probability(prob: Any, prior: Any, reliability_weight: Any, sample_confidence: Any, control_factor: Any = 1.0) -> dict:
    """
    Shrink a raw teacher probability toward the treatment prior using bounded effective weight.
    """

    p = safe_float(prob, np.nan)
    pr = safe_float(prior, 0.5)
    rw = safe_float(reliability_weight, 0.0)
    sc = safe_float(sample_confidence, 0.0)
    cf = safe_float(control_factor, 1.0)

    if not np.isfinite(p):
        return {
            "shrunk_prob": np.nan,
            "delta": 0.0,
            "effective_weight": 0.0,
        }

    # Effective weight combines model reliability, sample confidence, and control penalties.
    eff = float(np.clip(rw * sc * cf, 0.0, 1.0))
    # Only the reliability-supported delta is allowed to move away from the prior.
    delta = eff * (p - pr)
    shrunk = float(np.clip(pr + delta, 0.01, 0.99))

    return {
        "shrunk_prob": shrunk,
        "delta": float(delta),
        "effective_weight": eff,
    }


def label_quality_for_row(row: pd.Series) -> tuple[str, str]:
    """Assign a label-quality flag and reason string for one fused teacher row."""

    reasons = []

    if clean_text(row.get("modality_used")) == "none":
        reasons.append("no_teacher_modality")

    prior_n = safe_float(row.get("prior_n"), 0)
    if prior_n < 5:
        reasons.append("weak_or_global_prior")

    if clean_text(row.get("modality_used")) == "histology_only":
        if safe_float(row.get("histology_reliability_weight"), 0) < 0.10:
            reasons.append("histology_only_low_reliability")

    # Histology control warnings are carried forward so downstream users can stratify sensitivity analyses.
    if clean_text(row.get("histology_control_warning")):
        reasons.append("histology_control_warning")

    fp = safe_float(row.get("fused_prob_responder"), np.nan)
    if np.isfinite(fp) and (fp <= 0.011 or fp >= 0.989):
        reasons.append("near_clip_boundary")

    if reasons:
        if "no_teacher_modality" in reasons:
            return "exclude", ";".join(reasons)
        return "warn", ";".join(reasons)

    return "ok", "ok"


def basic_numeric_feature_manifest(model_input: pd.DataFrame, sample_col: str, min_nonmissing: float, drop_constant: bool) -> pd.DataFrame:
    """Helper function used by the governed teacher_builder workflow."""

    rows = []

    for c in model_input.columns:
        if c == sample_col:
            continue

        s = pd.to_numeric(model_input[c], errors="coerce")
        nonmissing = float(s.notna().mean())
        n_unique = int(s.dropna().nunique())

        included = nonmissing >= min_nonmissing

        reason = "included"

        if nonmissing < min_nonmissing:
            included = False
            reason = "low_nonmissing"

        if drop_constant and n_unique <= 1:
            included = False
            reason = "constant_or_empty"

        if not np.issubdtype(s.dropna().dtype if len(s.dropna()) else np.dtype(float), np.number):
            pass

        rows.append(
            {
                "feature": c,
                "included": bool(included),
                "reason": reason,
                "nonmissing_fraction": nonmissing,
                "n_unique": n_unique,
                "feature_group": str(c).split("__")[0].split("_")[0],
                "n_samples": int(model_input[sample_col].nunique()) if sample_col in model_input.columns else len(model_input),
            }
        )

    return pd.DataFrame(rows)




# =========================
# Numeric summary helper
# =========================
# QC scripts use compact numeric summaries to describe teacher output
# distributions without duplicating summary logic.

def summarize_series(x: pd.Series) -> dict:
    """Return mean, standard deviation, minimum, and maximum for a numeric Series."""

    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "min": float(y.min()),
        "max": float(y.max()),
    }
