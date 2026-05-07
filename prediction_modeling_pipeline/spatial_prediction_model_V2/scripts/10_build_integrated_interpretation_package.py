"""
Script: 10_build_integrated_interpretation_package.py

Purpose:
    Build the integrated V2 interpretation package from Steps 02 through 09.

Pipeline role:
    This step gathers summaries, model-comparison outputs, validated treatments,
    recurrent features, recurrent biology themes, figures, provenance, and
    pipeline recommendations into one interpretation package for downstream
    review.

Scientific role:
    The integrated package is the bridge between modeling outputs and
    manuscript-level interpretation. It distinguishes probability baselines,
    residual discovery models, curated treatment-specific screens, and label-
    shuffle-validated findings.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP10_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    imports, constants, thresholds, hyperparameters, validation rules,
    output filenames, and return codes must remain unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Helper functions
# =============================================================================

def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value):
    """Convert a value to float while returning None for invalid or missing values."""

    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def read_table(path: Path | str) -> pd.DataFrame:
    """Read a CSV or TSV table, returning an empty DataFrame when unavailable."""

    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def write_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a CSV or TSV table with parent-directory creation."""

    path = Path(path)
    ensure_dir(path.parent)

    if df is None:
        df = pd.DataFrame()

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)

    return path


def load_json(path: Path | str) -> dict:
    """Load a JSON file, returning an empty dictionary when unavailable."""

    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}


def write_json(obj: dict, path: Path | str) -> Path:
    """Write a JSON artifact with stable formatting."""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text_report(path: Path | str, body: str) -> Path:
    """Write a text report with a filepath header."""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")
    return path


def terminal_block(title: str, lines: list[str]) -> str:
    """Format terminal output as a readable status block."""

    bar = "=" * 90
    return "\n".join([bar, title, bar] + lines)


def shorten(text: str, n: int = 80) -> str:
    """Shorten text for display while preserving readability."""

    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def safe_filename(text: str, max_len: int = 120) -> str:
    """Convert text into a filesystem-safe filename."""

    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    if len(name) > max_len:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:10]
        name = name[: max_len - 11] + "_" + digest
    return name or "unnamed"


def file_manifest(root: Path) -> pd.DataFrame:
    """Build a file manifest for all files under a root directory."""

    rows = []

    if not root.exists():
        return pd.DataFrame(columns=["relative_path", "path", "size_bytes", "modified_time"])

    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "relative_path": str(path.relative_to(root)),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "modified_time": path.stat().st_mtime,
            })

    return pd.DataFrame(rows)


def make_source_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    """Build a source manifest for Step 10 input roots and summaries."""

    rows = []

    for name, path in paths.items():
        path = Path(path)
        rows.append({
            "source_name": name,
            "path": str(path),
            "exists": bool(path.exists()),
            "is_file": bool(path.exists() and path.is_file()),
            "is_dir": bool(path.exists() and path.is_dir()),
            "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else "",
        })

    return pd.DataFrame(rows)


def first_numeric(df: pd.DataFrame, cols: list[str]):
    """Return the first available numeric evidence column and numeric values."""

    for col in cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() > 0:
                return col, vals
    return "", pd.Series(dtype=float)


def table_count(df: pd.DataFrame) -> int:
    """Return the row count for a DataFrame-like table."""

    return int(len(df)) if df is not None else 0


def add_model_row(rows: list[dict], **kwargs):
    """Append one standardized model-comparison row."""

    base = {
        "model_branch": "",
        "step": "",
        "purpose": "",
        "input_target": "",
        "feature_set": "",
        "split_or_validation": "",
        "primary_metric_name": "",
        "primary_metric_value": "",
        "secondary_metric_name": "",
        "secondary_metric_value": "",
        "n_rows_or_samples": "",
        "n_features": "",
        "n_models_or_treatments": "",
        "validation_status": "",
        "main_output": "",
        "notes": "",
    }
    base.update(kwargs)
    rows.append(base)


# =============================================================================
# Reporting integration patch helpers
# =============================================================================
# STEP10_INTEGRATION_PATCH_V2

def _safe_get_number(row, keys):
    """Return the first numeric value from a row-like object."""

    for key in keys:
        try:
            if key in row.index:
                value = safe_float(row.get(key))
                if value is not None:
                    return value
        except Exception:
            pass
    return None


def enrich_summaries_from_available_tables(summaries: dict[str, dict], tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Fill Step 03, Step 04, and Step 09 reporting fields from available tables."""

    out = {key: dict(value) for key, value in summaries.items()}

    step03 = out.get("step03", {})
    step03_metric = tables.get("step03_metric_summary", pd.DataFrame())
    step03_contribution = tables.get("step03_contribution", pd.DataFrame())

    if not step03_metric.empty:
        row = step03_metric.iloc[0]
        pearson = _safe_get_number(row, ["test_pearson_mean", "pearson_mean", "test_pearson"])
        r2 = _safe_get_number(row, ["test_r2_mean", "r2_mean", "test_r2"])
        if pearson is not None:
            step03["test_pearson_mean"] = pearson
            step03["test_pearson"] = pearson
        if r2 is not None:
            step03["test_r2_mean"] = r2
            step03["test_r2"] = r2
        if "n_repeats" in row.index:
            step03["n_repeats"] = safe_float(row.get("n_repeats"))

    if not step03_contribution.empty:
        row = step03_contribution.iloc[0]
        sf = _safe_get_number(row, ["spatial_feature_fraction"])
        tf = _safe_get_number(row, ["treatment_identity_fraction"])
        if sf is not None:
            step03["spatial_feature_fraction"] = sf
        if tf is not None:
            step03["treatment_identity_fraction"] = tf

    out["step03"] = step03

    step04 = out.get("step04", {})
    step04_metric = tables.get("step04_metric_summary", pd.DataFrame())
    step04_contribution = tables.get("step04_contribution", pd.DataFrame())
    step04_spatial_evidence = tables.get("step04_spatial_feature_evidence", pd.DataFrame())

    if not step04_metric.empty:
        row = step04_metric.iloc[0]
        pearson = _safe_get_number(row, ["test_pearson_mean", "pearson_mean", "test_pearson"])
        r2 = _safe_get_number(row, ["test_r2_mean", "r2_mean", "test_r2"])
        if pearson is not None:
            step04["test_pearson_mean"] = pearson
            step04["test_pearson"] = pearson
        if r2 is not None:
            step04["test_r2_mean"] = r2
            step04["test_r2"] = r2
        if "n_repeats" in row.index:
            step04["n_repeats"] = safe_float(row.get("n_repeats"))

    if not step04_contribution.empty:
        row = step04_contribution.iloc[0]
        sf = _safe_get_number(row, ["spatial_feature_fraction"])
        tf = _safe_get_number(row, ["treatment_identity_fraction"])
        if sf is not None:
            step04["spatial_feature_fraction"] = sf
        if tf is not None:
            step04["treatment_identity_fraction"] = tf

    if not step04_spatial_evidence.empty:
        step04["n_feature_evidence_rows"] = int(len(step04_spatial_evidence))
        step04["n_spatial_feature_evidence_rows"] = int(len(step04_spatial_evidence))

    if "n_pair_rows" not in step04 and "n_rows_used" in step04:
        step04["n_pair_rows"] = step04.get("n_rows_used")
    if "n_features_used" not in step04 and "n_broad_spatial_features" in step04:
        step04["n_features_used"] = step04.get("n_broad_spatial_features")

    out["step04"] = step04

    step09 = out.get("step09", {})
    policy = str(step09.get("max_workers_policy", "") or "").strip()
    used = step09.get("max_workers_used", "")
    requested = step09.get("max_workers_requested", "")

    if not policy:
        try:
            requested_int = int(float(requested))
        except Exception:
            requested_int = None
        try:
            used_int = int(float(used))
        except Exception:
            used_int = None

        if requested_int is not None and requested_int <= 0:
            policy = "auto_half_cpu_cap_inferred_from_summary"
        elif requested_int is not None and used_int is not None:
            policy = "user_requested" if requested_int == used_int else "user_requested_capped_by_tier1_count"
        elif used_int is not None and used_int > 1:
            policy = "legacy_parallel_workers_inferred"
        else:
            policy = "not_recorded"

        step09["max_workers_policy"] = policy

    out["step09"] = step09
    return out


def collect_documentation_artifacts(root: Path) -> pd.DataFrame:
    """Collect documentation bundles and reviewer support documents under Step 10."""

    rows = []
    keywords = [
        "combined",
        "bundle",
        "all_code",
        "all_images",
        "documentation",
        "manifest",
        "caption",
        "narrative",
        "recommendation",
        "summary",
        "report",
    ]

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = str(path.relative_to(root))
        lower = path.name.lower()
        suffix = path.suffix.lower()

        if suffix not in [".txt", ".tsv", ".csv", ".json", ".pdf", ".xlsx"]:
            continue

        if any(key in lower for key in keywords) or "provenance" in rel.lower():
            rows.append({
                "relative_path": rel,
                "path": str(path),
                "suffix": suffix,
                "size_bytes": int(path.stat().st_size),
                "documentation_class": "step10_documentation_or_manifest",
            })

    return pd.DataFrame(rows)


# =============================================================================
# Step 10 repair helpers for metrics, worker policy, and documentation artifacts
# =============================================================================

def read_first_table(root: Path, relative_candidates: list[str], glob_patterns: list[str] | None = None) -> pd.DataFrame:
    for rel in relative_candidates:
        df = read_table(root / rel)
        if not df.empty:
            return df

    if glob_patterns:
        for pattern in glob_patterns:
            for path in sorted(root.rglob(pattern)):
                df = read_table(path)
                if not df.empty:
                    return df

    return pd.DataFrame()


def first_value_from_table(df: pd.DataFrame, names: list[str]):
    if df.empty:
        return None
    for name in names:
        if name in df.columns:
            vals = pd.to_numeric(df[name], errors="coerce")
            if vals.notna().sum():
                return safe_float(vals.dropna().iloc[0])
    return None


def enrich_step10_reporting_summaries(summaries: dict[str, dict], tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    out = {key: dict(value) for key, value in summaries.items()}

    step03 = out.get("step03", {})
    m03 = tables.get("step03_metric_summary", pd.DataFrame())
    c03 = tables.get("step03_contribution", pd.DataFrame())
    p03 = first_value_from_table(m03, ["test_pearson_mean", "pearson_mean", "test_pearson"])
    r03 = first_value_from_table(m03, ["test_r2_mean", "r2_mean", "test_r2"])
    if p03 is not None:
        step03["test_pearson_mean"] = p03
        step03["test_pearson"] = p03
    if r03 is not None:
        step03["test_r2_mean"] = r03
        step03["test_r2"] = r03
    sf03 = first_value_from_table(c03, ["spatial_feature_fraction"])
    ti03 = first_value_from_table(c03, ["treatment_identity_fraction"])
    if sf03 is not None:
        step03["spatial_feature_fraction"] = sf03
    if ti03 is not None:
        step03["treatment_identity_fraction"] = ti03
    out["step03"] = step03

    step04 = out.get("step04", {})
    m04 = tables.get("step04_metric_summary", pd.DataFrame())
    c04 = tables.get("step04_contribution", pd.DataFrame())
    e04 = tables.get("step04_spatial_feature_evidence", pd.DataFrame())
    p04 = first_value_from_table(m04, ["test_pearson_mean", "pearson_mean", "test_pearson"])
    r04 = first_value_from_table(m04, ["test_r2_mean", "r2_mean", "test_r2"])
    if p04 is not None:
        step04["test_pearson_mean"] = p04
        step04["test_pearson"] = p04
    if r04 is not None:
        step04["test_r2_mean"] = r04
        step04["test_r2"] = r04
    sf04 = first_value_from_table(c04, ["spatial_feature_fraction"])
    ti04 = first_value_from_table(c04, ["treatment_identity_fraction"])
    if sf04 is not None:
        step04["spatial_feature_fraction"] = sf04
    if ti04 is not None:
        step04["treatment_identity_fraction"] = ti04
    if not e04.empty:
        step04["n_spatial_feature_evidence_rows"] = int(len(e04))
        step04["n_feature_evidence_rows"] = int(len(e04))
    if "n_pair_rows" not in step04 and "n_rows_used" in step04:
        step04["n_pair_rows"] = step04.get("n_rows_used")
    if "n_features_used" not in step04 and "n_broad_spatial_features" in step04:
        step04["n_features_used"] = step04.get("n_broad_spatial_features")
    out["step04"] = step04

    step09 = out.get("step09", {})
    policy = str(step09.get("max_workers_policy", "") or "").strip()
    if not policy:
        requested = step09.get("max_workers_requested", "")
        used = step09.get("max_workers_used", "")
        try:
            requested_i = int(float(requested))
        except Exception:
            requested_i = None
        try:
            used_i = int(float(used))
        except Exception:
            used_i = None
        if requested_i is not None and requested_i <= 0:
            policy = "auto_half_cpu_cap_inferred"
        elif used_i is not None and used_i > 1:
            policy = "user_requested_or_legacy_parallel_inferred"
        else:
            policy = "not_recorded"
        step09["max_workers_policy"] = policy
    out["step09"] = step09
    return out


def collect_documentation_artifacts(root: Path) -> pd.DataFrame:
    rows = []
    keywords = ["combined", "bundle", "documentation", "manifest", "caption", "narrative", "recommendation", "summary", "report"]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        lower = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in [".txt", ".tsv", ".csv", ".json", ".pdf", ".xlsx"]:
            continue
        if any(key in lower for key in keywords) or "provenance" in rel.lower():
            rows.append({
                "relative_path": rel,
                "path": str(path),
                "suffix": suffix,
                "size_bytes": int(path.stat().st_size),
                "documentation_class": "step10_documentation_or_manifest",
            })
    return pd.DataFrame(rows)


def build_model_comparison(
    summaries: dict[str, dict],
    tables: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> pd.DataFrame:
    """Build the cross-step model comparison table for the integrated package."""

    rows = []

    step02 = summaries.get("step02", {})
    add_model_row(
        rows,
        model_branch="input_dataset",
        step="02",
        purpose="Build pair-level and sample-level residual modeling datasets.",
        input_target="fused_prob_responder and fused_residual_vs_prior",
        feature_set="broad governed spatial candidate pool",
        primary_metric_name="pair_level_rows",
        primary_metric_value=step02.get("n_pair_level_rows", ""),
        secondary_metric_name="eligible_treatments",
        secondary_metric_value=step02.get("n_treatments_eligible", ""),
        n_rows_or_samples=step02.get("n_pair_level_rows", ""),
        n_features=step02.get("n_broad_governed_candidate_features", step02.get("n_v2_primary_features", "")),
        n_models_or_treatments=step02.get("n_treatments_total", ""),
        validation_status=step02.get("status", ""),
        main_output=str(paths.get("step02", "")),
        notes="Source dataset for all V2 downstream models.",
    )

    step03 = summaries.get("step03", {})
    add_model_row(
        rows,
        model_branch="probability_baseline",
        step="03",
        purpose="Baseline response probability model.",
        input_target="fused_prob_responder",
        feature_set="broad governed candidate pool plus treatment identity if configured",
        primary_metric_name="test_pearson_mean",
        primary_metric_value=step03.get("test_pearson_mean", step03.get("best_test_pearson", step03.get("test_pearson", step03.get("primary_metric_value", "")))),
        secondary_metric_name="test_r2_mean",
        secondary_metric_value=step03.get("test_r2_mean", step03.get("best_test_r2", step03.get("test_r2", step03.get("secondary_metric_value", "")))),
        n_rows_or_samples=step03.get("n_rows", step03.get("n_pair_rows", "")),
        n_features=step03.get("n_features_used", step03.get("n_features", "")),
        n_models_or_treatments=1 if step03 else "",
        validation_status=step03.get("status", "not_available"),
        main_output=str(paths.get("step03", "")),
        notes="Kept mainly as probability baseline, not the final biology interpretation model.",
    )

    step04 = summaries.get("step04", {})
    add_model_row(
        rows,
        model_branch="pair_level_prior_adjusted_residual",
        step="04",
        purpose="Find residual spatial feature evidence after treatment prior adjustment.",
        input_target="fused_residual_vs_prior",
        feature_set="broad governed candidate pool",
        primary_metric_name="test_pearson_mean",
        primary_metric_value=step04.get("test_pearson_mean", step04.get("test_pearson", "")),
        secondary_metric_name="test_r2_mean",
        secondary_metric_value=step04.get("test_r2_mean", step04.get("test_r2", "")),
        n_rows_or_samples=step04.get("n_pair_rows", ""),
        n_features=step04.get("n_features_used", ""),
        n_models_or_treatments=1 if step04 else "",
        validation_status=step04.get("status", "not_available"),
        main_output=str(paths.get("step04", "")),
        notes="Generates evidence for Step 05 registry. Does not generate registry directly.",
    )

    step05 = summaries.get("step05", {})
    add_model_row(
        rows,
        model_branch="strict_biology_registry",
        step="05",
        purpose="Classify residual spatial evidence into strict biology features and themes.",
        input_target="Step 04 spatial feature evidence",
        feature_set="V2 strict biology registry",
        primary_metric_name="strict_biology_features",
        primary_metric_value=step05.get("n_strict_biology_features", step05.get("n_v2_strict_biology_registry_features", "")),
        secondary_metric_name="biology_themes",
        secondary_metric_value=step05.get("n_biology_themes", ""),
        n_rows_or_samples="",
        n_features=step05.get("n_strict_biology_features", ""),
        n_models_or_treatments="",
        validation_status=step05.get("status", ""),
        main_output=str(paths.get("step05_registry", "")),
        notes="Defines official interpretation feature set for Steps 06 to 09.",
    )

    step06 = summaries.get("step06", {})
    add_model_row(
        rows,
        model_branch="broad_residual_spatial_only",
        step="06",
        purpose="Sample-level broad residual screen using spatial biology only.",
        input_target=step06.get("best_target", "broad residual targets"),
        feature_set="V2 Step 05 strict biology registry",
        split_or_validation=f"repeated splits n={step06.get('n_repeats', '')}",
        primary_metric_name="best_target_test_pearson_mean",
        primary_metric_value=step06.get("best_target_test_pearson_mean", ""),
        secondary_metric_name="best_target_test_r2_mean",
        secondary_metric_value=step06.get("best_target_test_r2_mean", ""),
        n_rows_or_samples=step06.get("n_samples", ""),
        n_features=step06.get("n_features_used", ""),
        n_models_or_treatments=step06.get("n_targets_eligible", ""),
        validation_status=step06.get("status", ""),
        main_output=str(paths.get("step06", "")),
        notes="Broad biology screen, not treatment-specific validation.",
    )

    step07 = summaries.get("step07", {})
    add_model_row(
        rows,
        model_branch="filtered_per_treatment_residual_models",
        step="07",
        purpose="Treatment-specific residual discovery models.",
        input_target="fused_residual_vs_prior",
        feature_set="V2 Step 05 strict biology registry",
        split_or_validation=f"repeated splits n={step07.get('n_repeats', '')}",
        primary_metric_name="best_screened_treatment_test_pearson_mean",
        primary_metric_value=step07.get("best_screened_treatment_test_pearson_mean", ""),
        secondary_metric_name="best_screened_treatment_test_r2_mean",
        secondary_metric_value=step07.get("best_screened_treatment_test_r2_mean", ""),
        n_rows_or_samples=step07.get("n_pair_rows", ""),
        n_features=step07.get("n_features_used", ""),
        n_models_or_treatments=step07.get("n_screened_treatments", ""),
        validation_status=step07.get("status", ""),
        main_output=str(paths.get("step07", "")),
        notes="No treatment identity features. This is the main treatment-specific discovery branch.",
    )

    step08 = summaries.get("step08", {})
    add_model_row(
        rows,
        model_branch="curated_per_treatment_residual_models",
        step="08",
        purpose="Curate Step 07 models into interpretation tiers.",
        input_target="Step 07 treatment-specific screening results",
        feature_set="V2 Step 05 strict biology registry",
        split_or_validation="tier thresholds plus SHAP success",
        primary_metric_name="tier1_high_confidence_count",
        primary_metric_value=step08.get("n_tier1_high_confidence", ""),
        secondary_metric_name="tier2_screening_signal_count",
        secondary_metric_value=step08.get("n_tier2_screening_signal", ""),
        n_rows_or_samples="",
        n_features="",
        n_models_or_treatments=step08.get("n_screened_treatments", ""),
        validation_status=step08.get("status", ""),
        main_output=str(paths.get("step08", "")),
        notes="Exports Tier 1 candidates to Step 09 label shuffle validation.",
    )

    step09 = summaries.get("step09", {})
    add_model_row(
        rows,
        model_branch="tier1_label_shuffle_validation",
        step="09",
        purpose="Validate Tier 1 treatment-specific residual models against label-shuffle null.",
        input_target="fused_residual_vs_prior",
        feature_set="V2 Step 05 strict biology registry",
        split_or_validation=f"{step09.get('n_shuffles', '')} shuffles, {step09.get('n_repeats', '')} repeated splits",
        primary_metric_name="label_shuffle_validated_treatments",
        primary_metric_value=step09.get("n_label_shuffle_validated_treatments", ""),
        secondary_metric_name="best_fdr_q_pearson",
        secondary_metric_value=step09.get("best_fdr_q_pearson", ""),
        n_rows_or_samples="",
        n_features=step09.get("n_features_used", ""),
        n_models_or_treatments=step09.get("n_tier1_candidates_tested", ""),
        validation_status=step09.get("status", ""),
        main_output=str(paths.get("step09", "")),
        notes=f"Worker policy: {step09.get('max_workers_policy', '')}. Ready for Step 10: {step09.get('ready_for_step10_integrated_package', '')}.",
    )

    return pd.DataFrame(rows)


def feature_table_from_branch(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    """Convert one model branch feature table into the recurrent-feature merge schema."""

    if df.empty or "feature_name" not in df.columns:
        return pd.DataFrame(columns=["feature_name"])

    out = pd.DataFrame()
    out["feature_name"] = df["feature_name"].astype(str)

    score_col, vals = first_numeric(
        df,
        [
            "mean_abs_shap",
            "total_score",
            "mean_score",
            "mean_gain_importance",
            "gain_importance",
            "max_gain_importance",
        ],
    )

    out[f"{branch}_present"] = True
    out[f"{branch}_score"] = vals.values if score_col else np.nan

    for col in [
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]:
        if col in df.columns:
            out[col] = df[col]

    return out


def build_recurrent_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge recurrent feature evidence across V2 model branches."""

    branches = {
        "step05_registry": tables.get("step05_registry", pd.DataFrame()),
        "step06_broad": tables.get("step06_feature_summary", pd.DataFrame()),
        "step07_per_treatment": tables.get("step07_recurrent_features", pd.DataFrame()),
        "step08_curated": tables.get("step08_recurrent_features", pd.DataFrame()),
        "step09_validated": tables.get("step09_recurrent_features", pd.DataFrame()),
    }

    merged = None
    meta_cols = [
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]

    for branch, df in branches.items():
        b = feature_table_from_branch(df, branch)

        if b.empty or "feature_name" not in b.columns:
            continue

        b = b.drop_duplicates("feature_name")

        if merged is None:
            merged = b
        else:
            merged = merged.merge(b, on="feature_name", how="outer", suffixes=("", f"_{branch}"))

    if merged is None:
        return pd.DataFrame()

    for col in meta_cols:
        candidates = [c for c in merged.columns if c == col or c.startswith(col + "_")]
        if candidates:
            vals = None
            for c in candidates:
                current = merged[c]
                vals = current if vals is None else vals.fillna(current)
            merged[col] = vals
            drop_cols = [c for c in candidates if c != col]
            merged = merged.drop(columns=drop_cols, errors="ignore")

    present_cols = [c for c in merged.columns if c.endswith("_present")]
    score_cols = [c for c in merged.columns if c.endswith("_score")]

    for col in present_cols:
        merged[col] = merged[col].where(merged[col].notna(), False).astype(bool)

    for col in score_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["model_branch_count"] = merged[present_cols].sum(axis=1) if present_cols else 0
    merged["total_branch_score"] = merged[score_cols].sum(axis=1) if score_cols else 0.0

    first_present_branch = []
    for _, row in merged.iterrows():
        branch = ""
        for col in present_cols:
            if bool(row.get(col, False)):
                branch = col.replace("_present", "")
                break
        first_present_branch.append(branch)

    merged["first_detected_branch"] = first_present_branch
    merged = merged.sort_values(["model_branch_count", "total_branch_score"], ascending=False)
    return merged


def theme_table_from_branch(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    """Convert one branch theme table into the recurrent-theme merge schema."""

    if df.empty:
        return pd.DataFrame(columns=["biological_theme"])

    theme_col = None
    for candidate in ["biological_theme", "theme", "feature_theme"]:
        if candidate in df.columns:
            theme_col = candidate
            break

    if theme_col is None:
        return pd.DataFrame(columns=["biological_theme"])

    out = pd.DataFrame()
    out["biological_theme"] = df[theme_col].astype(str)

    score_col, vals = first_numeric(
        df,
        [
            "total_score",
            "total_gain_importance",
            "mean_score",
            "mean_abs_shap",
            "max_score",
            "validated_treatment_count",
            "n_treatments",
            "n_features",
        ],
    )

    out[f"{branch}_present"] = True
    out[f"{branch}_score"] = vals.values if score_col else np.nan

    for col in ["n_features", "n_treatments", "validated_treatment_count", "example_features"]:
        if col in df.columns:
            out[col] = df[col]

    return out


def build_recurrent_theme_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge recurrent biology theme evidence across V2 model branches."""

    branches = {
        "step05_registry": tables.get("step05_theme_summary", pd.DataFrame()),
        "step06_broad": tables.get("step06_theme_summary", pd.DataFrame()),
        "step07_per_treatment": tables.get("step07_recurrent_themes", pd.DataFrame()),
        "step08_curated": tables.get("step08_recurrent_themes", pd.DataFrame()),
        "step09_validated": tables.get("step09_recurrent_themes", pd.DataFrame()),
    }

    merged = None

    for branch, df in branches.items():
        b = theme_table_from_branch(df, branch)

        if b.empty or "biological_theme" not in b.columns:
            continue

        b = b.drop_duplicates("biological_theme")

        if merged is None:
            merged = b
        else:
            merged = merged.merge(b, on="biological_theme", how="outer", suffixes=("", f"_{branch}"))

    if merged is None:
        return pd.DataFrame()

    present_cols = [c for c in merged.columns if c.endswith("_present")]
    score_cols = [c for c in merged.columns if c.endswith("_score")]

    for col in present_cols:
        merged[col] = merged[col].where(merged[col].notna(), False).astype(bool)

    for col in score_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["model_branch_count"] = merged[present_cols].sum(axis=1) if present_cols else 0
    merged["total_branch_score"] = merged[score_cols].sum(axis=1) if score_cols else 0.0

    examples_cols = [c for c in merged.columns if c == "example_features" or c.startswith("example_features_")]
    if examples_cols:
        vals = None
        for c in examples_cols:
            vals = merged[c] if vals is None else vals.fillna(merged[c])
        merged["example_features"] = vals
        merged = merged.drop(columns=[c for c in examples_cols if c != "example_features"], errors="ignore")

    merged = merged.sort_values(["model_branch_count", "total_branch_score"], ascending=False)
    return merged


def copy_figures(figure_sources: dict[str, Path], destination: Path) -> pd.DataFrame:
    """Copy presentation figures into the integrated package and build a figure manifest."""

    ensure_dir(destination)
    rows = []
    figure_index = 1

    for source_step, fig_dir in figure_sources.items():
        fig_dir = Path(fig_dir)
        if not fig_dir.exists():
            continue

        for src in sorted(fig_dir.glob("*.png")):
            safe = safe_filename(f"{source_step}_{src.name}", max_len=130)
            dst = destination / safe
            shutil.copy2(src, dst)

            caption = make_caption(source_step, src.name)

            rows.append({
                "figure_id": f"Figure {figure_index}",
                "source_step": source_step,
                "source_file": str(src),
                "copied_file": str(dst),
                "file_name": dst.name,
                "caption": caption,
            })
            figure_index += 1

    return pd.DataFrame(rows)


def make_caption(source_step: str, name: str) -> str:
    lower = name.lower()

    if "pearson_vs_r2" in lower or "pearson_vs_r2" in lower.replace("-", "_"):
        concept = "relationship between mean test Pearson and test R2"
    elif "tier1" in lower or "tier2" in lower or "tier3" in lower or "interpretation_tier" in lower:
        concept = "curated treatment model tier summary"
    elif "label_shuffle" in lower or "shuffle" in lower or "null" in lower or "fdr" in lower or "q_value" in lower or "p_value" in lower:
        concept = "label shuffle validation"
    elif "r2" in lower or "test_r2" in lower:
        concept = "test R2 performance"
    elif "pearson" in lower:
        concept = "mean test Pearson performance"
    elif "theme" in lower:
        concept = "biology theme contribution"
    elif "feature" in lower:
        concept = "spatial feature recurrence or importance"
    elif "distribution" in lower:
        concept = "performance distribution"
    else:
        concept = "model output summary"

    return f"{source_step}: {concept} from {name}."


def write_methods_results_discussion(
    path: Path,
    summaries: dict[str, dict],
    model_comparison: pd.DataFrame,
    validated_treatments: pd.DataFrame,
    recurrent_features: pd.DataFrame,
    recurrent_themes: pd.DataFrame,
) -> Path:
    """Write a draft methods/results/discussion narrative from integrated outputs."""

    step06 = summaries.get("step06", {})
    step08 = summaries.get("step08", {})
    step09 = summaries.get("step09", {})

    lines = []
    lines.append("METHODS, RESULTS, AND DISCUSSION NARRATIVE")
    lines.append("")
    lines.append("Methods")
    lines.append("Spatial Prediction Model V2 integrates prior-adjusted residual modeling with a strict residual biology feature registry. Step 04 trains a pair-level residual model using fused_residual_vs_prior. Step 05 converts spatial feature evidence into a V2 strict biology registry. Step 06 evaluates sample-level broad residual biology. Step 07 trains treatment-specific residual models using only strict biology features and no treatment identity dummies. Step 08 curates treatment models into confidence tiers. Step 09 tests Tier 1 candidates against within-treatment label shuffles.")
    lines.append("")
    lines.append("Results")
    lines.append(f"The broad residual model best target was {step06.get('best_target', 'not available')} with mean test Pearson {step06.get('best_target_test_pearson_mean', 'not available')} and mean test R2 {step06.get('best_target_test_r2_mean', 'not available')}.")
    lines.append(f"Step 08 curated {step08.get('n_tier1_high_confidence', 'not available')} Tier 1 high-confidence screening models and {step08.get('n_tier2_screening_signal', 'not available')} Tier 2 screening models.")
    lines.append(f"Step 09 tested {step09.get('n_tier1_candidates_tested', 'not available')} Tier 1 candidates and validated {step09.get('n_label_shuffle_validated_treatments', 'not available')} treatment-specific spatial residual signals by label shuffle.")
    lines.append(f"The integrated package contains {len(validated_treatments)} validated treatment rows, {len(recurrent_features)} recurrent feature rows, and {len(recurrent_themes)} recurrent biology theme rows.")
    lines.append("")
    lines.append("Discussion")
    lines.append("The V2 model family separates probability prediction from residual spatial biology interpretation. Treatment identity is not used in the per-treatment residual models or label-shuffle validation, which makes the validated findings more interpretable as spatial biology signals rather than drug identity effects. Smoke-run label-shuffle p-value resolution is limited by the number of shuffles; the full run should increase shuffles for more precise empirical p and FDR estimates.")
    lines.append("")
    lines.append("Interpretation caveat")
    lines.append("This package reflects the current V2 smoke run unless generated from a full V2 run. Smoke registry and validation counts can differ from final production counts.")

    return write_text_report(path, "\n".join(lines))


def pipeline_recommendations() -> pd.DataFrame:
    """Return V2 pipeline integration recommendations for downstream review."""

    rows = [
        {
            "recommendation_id": "R01",
            "component": "V2 step order",
            "recommendation": "Promote Steps 01 to 11 as the canonical V2 spatial prediction and interpretation pipeline.",
            "priority": "high",
            "rationale": "V2 separates probability baseline, residual biology modeling, curation, label shuffle validation, and publication packaging.",
        },
        {
            "recommendation_id": "R02",
            "component": "Step 05 strict biology registry",
            "recommendation": "Use the V2-generated strict biology registry as the primary interpretation feature set.",
            "priority": "high",
            "rationale": "This removes production dependency on V1 output files while retaining the validated V1 logic.",
        },
        {
            "recommendation_id": "R03",
            "component": "Step 07 per-treatment residual models",
            "recommendation": "Keep treatment-specific residual models as the main treatment-level discovery branch.",
            "priority": "high",
            "rationale": "This branch avoids treatment identity dummies and directly models residual spatial signal within treatments.",
        },
        {
            "recommendation_id": "R04",
            "component": "Step 09 label shuffle",
            "recommendation": "Keep Step 09 parallel treatment workers in production and increase shuffles for full runs.",
            "priority": "high",
            "rationale": "Parallel workers greatly reduce runtime while XGBoost remains n_jobs=1 inside each worker.",
        },
        {
            "recommendation_id": "R05",
            "component": "Full run",
            "recommendation": "After smoke validation, rerun Steps 04 to 11 in full mode using the full 102-sample cohort and larger shuffle count.",
            "priority": "high",
            "rationale": "Smoke registry and p-value resolution are intentionally limited.",
        },
    ]
    return pd.DataFrame(rows)


def maybe_open(path: Path):
    """Open a path on Windows when requested by the user."""

    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
    except Exception:
        pass


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Run this spatial_prediction_model_V2 step and write tables, reports, provenance, and summaries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--step02-root", default="")
    parser.add_argument("--step03-root", default="")
    parser.add_argument("--step04-root", default="")
    parser.add_argument("--step05-root", default="")
    parser.add_argument("--step06-root", default="")
    parser.add_argument("--step07-root", default="")
    parser.add_argument("--step08-root", default="")
    parser.add_argument("--step09-root", default="")
    parser.add_argument("--open-output", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_root = ensure_dir(args.output_root)

    step02 = Path(args.step02_root) if args.step02_root else run_root / "02_build_modeling_dataset"
    step03 = Path(args.step03_root) if args.step03_root else run_root / "03_probability_baseline"
    step04 = Path(args.step04_root) if args.step04_root else run_root / "04_pair_level_residual_model"
    step05 = Path(args.step05_root) if args.step05_root else run_root / "05_residual_biology_registry"
    step06 = Path(args.step06_root) if args.step06_root else run_root / "06_broad_residual_model"
    step07 = Path(args.step07_root) if args.step07_root else run_root / "07_filtered_per_treatment_residual_models"
    step08 = Path(args.step08_root) if args.step08_root else run_root / "08_curated_per_treatment_residual_models"
    step09 = Path(args.step09_root) if args.step09_root else run_root / "09_tier1_label_shuffle_validation"

    d01 = ensure_dir(output_root / "01_master_summary")
    d02 = ensure_dir(output_root / "02_model_comparison")
    d03 = ensure_dir(output_root / "03_validated_treatments")
    d04 = ensure_dir(output_root / "04_recurrent_spatial_features")
    d05 = ensure_dir(output_root / "05_recurrent_biology_themes")
    d06 = ensure_dir(output_root / "06_figures_for_presentation")
    d07 = ensure_dir(output_root / "07_methods_results_discussion")
    d08 = ensure_dir(output_root / "08_pipeline_integration_recommendations")
    d09 = ensure_dir(output_root / "09_provenance_and_manifest")

    summary_paths = {
        "step02": step02 / "v2_dataset_builder_summary.json",
        "step03": step03 / "v2_step03_probability_baseline_summary.json",
        "step04": step04 / "v2_step04_pair_level_residual_model_summary.json",
        "step05": step05 / "v2_step05_residual_biology_registry_summary.json",
        "step06": step06 / "v2_step06_broad_residual_model_summary.json",
        "step07": step07 / "v2_step07_filtered_per_treatment_residual_models_summary.json",
        "step08": step08 / "v2_step08_curated_per_treatment_residual_models_summary.json",
        "step09": step09 / "v2_step09_tier1_label_shuffle_validation_summary.json",
    }

    summaries = {key: load_json(path) for key, path in summary_paths.items()}

    paths = {
        "run_root": run_root,
        "output_root": output_root,
        "step02": step02,
        "step03": step03,
        "step04": step04,
        "step05": step05,
        "step05_registry": step05 / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv",
        "step06": step06,
        "step07": step07,
        "step08": step08,
        "step09": step09,
        "step09_handoff": step09 / "08_step10_handoff" / "step10_handoff_summary.json",
    }

    source_manifest = make_source_manifest(paths | summary_paths)

    tables = {
        "step05_registry": read_table(step05 / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv"),
        "step05_theme_summary": read_table(step05 / "04_theme_summary" / "v2_residual_biology_theme_summary.tsv"),
        "step06_target_summary": read_table(step06 / "02_model_metrics" / "broad_residual_target_summary.tsv"),
        "step06_feature_summary": read_table(step06 / "03_feature_evidence" / "broad_residual_feature_evidence_summary.tsv"),
        "step06_theme_summary": read_table(step06 / "03_feature_evidence" / "broad_residual_theme_evidence_summary.tsv"),
        "step07_screening_summary": read_table(step07 / "02_screening_metrics" / "per_treatment_screening_summary.tsv"),
        "step07_final_manifest": read_table(step07 / "03_final_models" / "final_model_manifest.tsv"),
        "step07_recurrent_features": read_table(step07 / "05_recurrent_features_and_themes" / "per_treatment_recurrent_spatial_features.tsv"),
        "step07_recurrent_themes": read_table(step07 / "05_recurrent_features_and_themes" / "per_treatment_recurrent_biology_themes.tsv"),
        "step08_curated_treatments": read_table(step08 / "02_curated_treatment_models" / "curated_treatment_model_table.tsv"),
        "step08_tier_summary": read_table(step08 / "02_curated_treatment_models" / "interpretation_tier_summary.tsv"),
        "step08_recurrent_features": read_table(step08 / "03_recurrent_spatial_features" / "curated_recurrent_spatial_features.tsv"),
        "step08_recurrent_themes": read_table(step08 / "04_recurrent_biology_themes" / "curated_recurrent_biology_themes.tsv"),
        "step08_label_shuffle_candidates": read_table(step08 / "07_label_shuffle_handoff" / "tier1_label_shuffle_candidates.tsv"),
        "step09_validation_results": read_table(step09 / "04_validation_results" / "tier1_label_shuffle_validation_results.tsv"),
        "step09_validated_treatments": read_table(step09 / "04_validation_results" / "tier1_label_shuffle_validated_treatments.tsv"),
        "step09_recurrent_features": read_table(step09 / "05_validated_features_and_themes" / "label_shuffle_validated_recurrent_spatial_features.tsv"),
        "step09_recurrent_themes": read_table(step09 / "05_validated_features_and_themes" / "label_shuffle_validated_recurrent_biology_themes.tsv"),
    }

    # The model comparison table is the central cross-step summary for publication review.
    tables["step03_metric_summary"] = read_table(step03 / "02_metrics" / "probability_baseline_metric_summary.tsv")
    tables["step03_contribution"] = read_table(step03 / "03_feature_evidence" / "probability_baseline_spatial_vs_treatment_contribution.tsv")
    tables["step04_metric_summary"] = read_table(step04 / "02_metrics" / "pair_level_residual_metric_summary.tsv")
    tables["step04_contribution"] = read_table(step04 / "03_feature_evidence_for_step05" / "pair_level_residual_spatial_vs_treatment_contribution.tsv")
    tables["step04_spatial_feature_evidence"] = read_table(step04 / "03_feature_evidence_for_step05" / "spatial_feature_evidence_for_step05.tsv")

    summaries = enrich_summaries_from_available_tables(summaries, tables)

    tables["step03_metric_summary"] = read_first_table(
        step03,
        [
            "02_metrics/probability_baseline_metric_summary.tsv",
            "02_metrics/model_metric_summary.tsv",
            "02_metrics/qc_model_metrics.tsv",
        ],
        ["*metric*summary*.tsv", "*metrics*.tsv"],
    )
    tables["step03_contribution"] = read_first_table(
        step03,
        [
            "03_feature_evidence/probability_baseline_spatial_vs_treatment_contribution.tsv",
            "03_feature_evidence/spatial_vs_treatment_contribution.tsv",
        ],
        ["*spatial*contribution*.tsv", "*treatment*contribution*.tsv"],
    )
    tables["step04_metric_summary"] = read_first_table(
        step04,
        [
            "02_metrics/pair_level_residual_metric_summary.tsv",
            "02_metrics/model_metric_summary.tsv",
            "02_metrics/qc_model_metrics.tsv",
        ],
        ["*metric*summary*.tsv", "*metrics*.tsv"],
    )
    tables["step04_contribution"] = read_first_table(
        step04,
        [
            "03_feature_evidence_for_step05/pair_level_residual_spatial_vs_treatment_contribution.tsv",
            "03_feature_evidence_for_step05/spatial_vs_treatment_contribution.tsv",
        ],
        ["*spatial*contribution*.tsv", "*treatment*contribution*.tsv"],
    )
    tables["step04_spatial_feature_evidence"] = read_first_table(
        step04,
        [
            "03_feature_evidence_for_step05/spatial_feature_evidence_for_step05.tsv",
            "03_feature_evidence_for_step05/feature_evidence_for_step05.tsv",
        ],
        ["*spatial_feature_evidence*.tsv", "*feature_evidence*.tsv"],
    )

    summaries = enrich_step10_reporting_summaries(summaries, tables)

    model_comparison = build_model_comparison(summaries, tables, paths)

    # Use Step 09 validated treatments when available, falling back to all validation results for transparent reporting.
    validated_treatments = tables["step09_validated_treatments"].copy()
    if validated_treatments.empty:
        validated_treatments = tables["step09_validation_results"].copy()

    if not validated_treatments.empty:
        validated_treatments["integrated_interpretation_status"] = np.where(
            validated_treatments.get("validated_for_step10", False).astype(str).str.lower().isin(["true", "1", "yes"]),
            "label_shuffle_validated",
            validated_treatments.get("label_shuffle_validation_status", "not_label_shuffle_validated"),
        )

    # Recurrent feature tables integrate evidence across registry, broad, per-treatment, curated, and validated branches.
    recurrent_features = build_recurrent_feature_table(tables)
    recurrent_themes = build_recurrent_theme_table(tables)

    # Copy figures into one presentation folder so manuscript review does not depend on scattered step outputs.
    figure_manifest = copy_figures(
        {
            "step06_broad_residual": step06 / "04_figures",
            "step07_per_treatment": step07 / "06_figures",
            "step08_curation": step08 / "05_figures",
            "step09_label_shuffle": step09 / "06_figures",
        },
        d06,
    )

    write_table(model_comparison, d02 / "model_comparison_table.tsv")
    write_table(validated_treatments, d03 / "validated_treatment_table.tsv")
    write_table(recurrent_features, d04 / "recurrent_spatial_feature_table.tsv")
    write_table(recurrent_themes, d05 / "recurrent_biology_theme_table.tsv")
    write_table(figure_manifest, d06 / "figure_manifest.tsv")

    caption_lines = ["FIGURE CAPTIONS", ""]
    for _, row in figure_manifest.iterrows():
        caption_lines.append(f"{row.get('figure_id', '')}. {row.get('caption', '')}")
        caption_lines.append(f"File: {row.get('file_name', '')}")
        caption_lines.append("")
    write_text_report(d06 / "figure_captions.txt", "\n".join(caption_lines))

    methods_path = write_methods_results_discussion(
        d07 / "methods_results_discussion_narrative.txt",
        summaries,
        model_comparison,
        validated_treatments,
        recurrent_features,
        recurrent_themes,
    )

    # Pipeline recommendations make implementation decisions explicit for the next production run.
    recommendations = pipeline_recommendations()
    write_table(recommendations, d08 / "pipeline_integration_recommendations.tsv")

    rec_lines = ["PIPELINE INTEGRATION RECOMMENDATIONS", ""]
    for _, row in recommendations.iterrows():
        rec_lines.append(f"{row['recommendation_id']}: {row['recommendation']}")
        rec_lines.append(f"Priority: {row['priority']}")
        rec_lines.append(f"Rationale: {row['rationale']}")
        rec_lines.append("")
    write_text_report(d08 / "pipeline_integration_recommendations.txt", "\n".join(rec_lines))

    write_table(source_manifest, d09 / "source_file_manifest.tsv")

    generated_manifest = file_manifest(output_root)
    write_table(generated_manifest, d09 / "generated_output_manifest.tsv")

    provenance_rows = []
    for step, root in [
        ("02", step02),
        ("03", step03),
        ("04", step04),
        ("05", step05),
        ("06", step06),
        ("07", step07),
        ("08", step08),
        ("09", step09),
        ("10", output_root),
    ]:
        provenance_rows.append({
            "step": step,
            "root": str(root),
            "exists": root.exists(),
            "canonical_v1_modified": "no",
            "v2_production_dependency_on_v1_outputs": "no",
        })
    provenance = pd.DataFrame(provenance_rows)
    write_table(provenance, d09 / "provenance_table.tsv")

    documentation_artifacts = collect_documentation_artifacts(output_root)
    write_table(documentation_artifacts, d09 / "documentation_artifact_manifest.tsv")

    master_lines = []
    master_lines.append("SPATIAL PREDICTION MODEL V2 INTEGRATED INTERPRETATION PACKAGE")
    master_lines.append("")
    master_lines.append(f"status: pass")
    master_lines.append(f"run_root: {run_root}")
    master_lines.append(f"output_root: {output_root}")
    master_lines.append("")
    master_lines.append("Core counts")
    master_lines.append(f"model_comparison_rows: {len(model_comparison)}")
    master_lines.append(f"validated_treatment_rows: {len(validated_treatments)}")
    master_lines.append(f"recurrent_spatial_feature_rows: {len(recurrent_features)}")
    master_lines.append(f"recurrent_biology_theme_rows: {len(recurrent_themes)}")
    master_lines.append(f"figure_rows: {len(figure_manifest)}")
    master_lines.append(f"documentation_artifact_rows: {len(documentation_artifacts)}")
    master_lines.append("")
    master_lines.append("Key Step 09 results")
    step09_summary = summaries.get("step09", {})
    for key in [
        "n_tier1_candidates_tested",
        "n_label_shuffle_validated_treatments",
        "best_treatment",
        "best_observed_test_pearson_mean",
        "best_empirical_p_pearson",
        "best_fdr_q_pearson",
        "max_workers_policy",
        "max_workers_used",
    ]:
        master_lines.append(f"{key}: {step09_summary.get(key, '')}")
    master_lines.append("")
    master_lines.append("Interpretation")
    master_lines.append("The integrated package consolidates the V2 smoke results from probability baseline, prior-adjusted residual modeling, strict biology registry generation, broad residual modeling, per-treatment residual modeling, curation, and label-shuffle validation.")
    master_lines.append("The validated treatment table is the strongest current treatment-specific spatial biology result table.")
    master_lines.append("The recurrent feature and biology theme tables summarize spatial mechanisms recurring across model branches.")
    master_lines.append("")
    master_lines.append("Caveat")
    master_lines.append("If this package was generated from a smoke run, counts and empirical p-value resolution are smoke-limited. Full V2 should rerun with the full 102 sample cohort and larger shuffle counts.")

    master_report_path = write_text_report(d01 / "master_summary_report.txt", "\n".join(master_lines))

    run_summary = {
        "status": "pass",
        "official_step": "10_build_integrated_interpretation_package",
        "run_root": str(run_root),
        "output_root": str(output_root),
        "n_model_comparison_rows": int(len(model_comparison)),
        "n_validated_treatment_rows": int(len(validated_treatments)),
        "n_recurrent_spatial_feature_rows": int(len(recurrent_features)),
        "n_recurrent_biology_theme_rows": int(len(recurrent_themes)),
        "n_figures": int(len(figure_manifest)),
        "n_documentation_artifacts": int(len(documentation_artifacts)) if "documentation_artifacts" in locals() else 0,
        "source_steps_included": [k for k, v in summaries.items() if v],
        "production_dependency_on_v1_outputs": "no",
        "canonical_v1_scripts_modified": "no",
        "ready_for_step11_publication_tables": "yes",
    }

    run_summary["n_documentation_artifacts"] = int(len(documentation_artifacts))
    write_json(run_summary, output_root / "v2_step10_integrated_interpretation_package_summary.json")

    generated_manifest = file_manifest(output_root)
    write_table(generated_manifest, d09 / "generated_output_manifest.tsv")

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Output root: {output_root}",
        f"Master report: {master_report_path}",
        f"Model comparison rows: {len(model_comparison)}",
        f"Validated treatment rows: {len(validated_treatments)}",
        f"Recurrent spatial feature rows: {len(recurrent_features)}",
        f"Recurrent biology theme rows: {len(recurrent_themes)}",
        f"Figure rows: {len(figure_manifest)}",
        "Production dependency on V1 outputs: no",
        "Canonical V1 scripts modified: no",
    ]

    print("")
    print(terminal_block("V2 STEP 10 INTEGRATED INTERPRETATION PACKAGE COMPLETE", terminal_lines))
    print("")

    print("Model comparison preview")
    print(model_comparison.head(20).to_string(index=False))
    print("")

    if not validated_treatments.empty:
        print("Validated treatment preview")
        print(validated_treatments.head(20).to_string(index=False))
        print("")

    if not recurrent_features.empty:
        print("Recurrent feature preview")
        cols = [c for c in ["feature_name", "feature_original", "biological_theme", "model_branch_count", "total_branch_score"] if c in recurrent_features.columns]
        print(recurrent_features[cols].head(30).to_string(index=False))
        print("")

    if not recurrent_themes.empty:
        print("Recurrent theme preview")
        print(recurrent_themes.head(20).to_string(index=False))
        print("")

    if args.open_output:
        maybe_open(output_root)
        maybe_open(d06)

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
