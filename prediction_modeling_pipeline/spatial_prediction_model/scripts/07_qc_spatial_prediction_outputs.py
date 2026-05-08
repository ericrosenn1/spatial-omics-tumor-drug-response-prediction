"""
Script:
    07_qc_spatial_prediction_outputs.py

Purpose:
    Build final quality control reports, statistical summaries, and
    presentation-ready figures for the spatial prediction model pipeline.

Role:
    Seventh step in spatial_prediction_model.
    Consumes outputs from steps 02 through 06.
    Does not train, explain, or predict new values.
    Audits file contracts, dataset shape, splits, model metrics, prediction
    distributions, teacher overlap, treatment ranking, and feature explanations.

Pipeline position:
    01_validate_prediction_inputs.py
        validates the teacher_builder handoff.

    02_build_spatial_modeling_dataset.py
        builds the leakage-safe model table, feature matrix, target vector,
        sample split, and feature manifest.

    03_train_global_spatial_response_model.py
        trains the pooled global model and writes metrics, predictions,
        feature importance, and model artifacts.

    04_train_per_treatment_models.py
        optionally trains one model per treatment.
        For the 10-sample run, this step normally writes empty contract outputs.

    05_explain_spatial_response_model.py
        explains the global model with SHAP, permutation, or native importance.

    06_predict_all_sample_treatment_pairs.py
        applies the trained global model to selected sample-treatment pairs.

    07_qc_spatial_prediction_outputs.py
        produces final QC tables and figures for reporting and presentation.

Expected outputs:
    outputs/<run>/07_prediction_qc/
        qc_summary.tsv
        qc_summary.txt
        qc_file_contract_report.tsv
        qc_dataset_summary.tsv
        qc_split_summary.tsv
        qc_model_metrics.tsv
        qc_teacher_overlap_metrics.tsv
        qc_teacher_overlap_by_treatment.tsv
        qc_teacher_overlap_by_sample.tsv
        qc_prediction_distribution.tsv
        qc_by_sample.tsv
        qc_by_treatment.tsv
        qc_top_treatment_per_sample.tsv
        qc_feature_explanation_summary.tsv
        qc_spatial_vs_drug_summary.tsv
        qc_per_treatment_model_status.tsv
        presentation_figure_manifest.tsv
        run_config.json
        figures/*.png

Design contract:
    YAML controls output_root and output_subdirs.
    The same code supports the 10-sample smoke run and future 102-sample run.
    Missing optional steps are reported rather than treated as fatal.
    Figures use matplotlib only, one plot per image, and presentation friendly names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import argparse
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import yaml


SCRIPT_NAME = "07_qc_spatial_prediction_outputs.py"
STEP_NAME = "07_qc_spatial_prediction_outputs"

DEFAULT_SUBDIRS = {
    "input_validation": "01_input_validation",
    "modeling_dataset": "02_modeling_dataset",
    "global_model": "03_global_model",
    "per_treatment_models": "04_per_treatment_models",
    "model_explanation": "05_model_explanation",
    "model_explanations": "05_model_explanation",
    "all_sample_predictions": "06_all_sample_predictions",
    "prediction_qc": "07_prediction_qc",
}


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    Config is the only required input."""
    parser = argparse.ArgumentParser(description="QC spatial prediction outputs")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config.
    UTF8 BOM tolerant for Windows-edited files."""
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read config value.
    Small wrapper for consistent fallback behavior."""
    return cfg[key] if key in cfg else default


def clean_text(value: Any) -> str:
    """Convert scalar to clean text.
    Missing values become empty strings."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def bool_like(value: Any) -> bool:
    """Parse flexible boolean values.
    Handles YAML booleans and text exports."""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on", "included"}


def is_windows_absolute_path(value: str | Path) -> bool:
    """Detect Windows drive-letter paths.
    Useful for cross-platform path checks."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(value)))


def path_is_absolute_like(value: str | Path) -> bool:
    """Detect native or Windows absolute paths.
    Keeps path resolution stable on Windows and Linux."""
    text = str(value)
    return Path(text).is_absolute() or is_windows_absolute_path(text)


def infer_model_root(config_path: Path, cfg: dict[str, Any]) -> Path:
    """Infer spatial_prediction_model root.
    Uses model_root when present, otherwise config folder."""
    configured = get_cfg(cfg, "model_root", None) or get_cfg(cfg, "spatial_prediction_model_dir", None)

    if configured:
        return Path(str(configured))

    if config_path.parent.name.lower() == "configs":
        return config_path.parent.parent.resolve()

    return Path.cwd().resolve()


def get_output_root(cfg: dict[str, Any], model_root: Path) -> Path:
    """Resolve output root.
    Relative paths are resolved under model_root."""
    value = get_cfg(cfg, "output_root", "outputs")
    path = Path(str(value))

    if path_is_absolute_like(path):
        return path

    return model_root / path


def active_run_root(cfg: dict[str, Any], model_root: Path) -> Path:
    """Resolve active run output folder.
    Supports output_root as either outputs/ or outputs/output_run_10."""
    output_root = get_output_root(cfg, model_root)
    run_name = clean_text(get_cfg(cfg, "run_name", ""))

    # Config already points at current run folder
    if run_name and output_root.name == run_name:
        return output_root

    # Step folders already live directly under output_root
    if (output_root / DEFAULT_SUBDIRS["modeling_dataset"]).exists():
        return output_root

    # Standard scaffold layout: outputs/run_name/step
    if run_name:
        return output_root / run_name

    return output_root


def get_output_subdir(cfg: dict[str, Any], key: str) -> str:
    """Resolve a step subfolder name.
    output_subdirs in YAML overrides defaults."""
    subdirs = get_cfg(cfg, "output_subdirs", {}) or {}

    if isinstance(subdirs, dict) and key in subdirs:
        return str(subdirs[key])

    return DEFAULT_SUBDIRS[key]


def step_dir(cfg: dict[str, Any], model_root: Path, key: str) -> Path:
    """Resolve a standard step directory.
    Uses active run root plus configured subdir."""
    return active_run_root(cfg, model_root) / get_output_subdir(cfg, key)


def find_step05_dir(cfg: dict[str, Any], model_root: Path) -> Path:
    """Find explanation output folder.
    Accepts singular and plural historical names."""
    run_root = active_run_root(cfg, model_root)

    candidates = [
        run_root / get_output_subdir(cfg, "model_explanation"),
        run_root / get_output_subdir(cfg, "model_explanations"),
        run_root / "05_model_explanation",
        run_root / "05_model_explanations",
    ]

    seen: set[str] = set()
    unique: list[Path] = []

    # Avoid duplicate candidates
    for path in candidates:
        text = str(path)
        if text not in seen:
            unique.append(path)
            seen.add(text)

    for path in unique:
        if path.exists():
            return path

    return unique[0]


def qc_output_dir(cfg: dict[str, Any], model_root: Path) -> Path:
    """Resolve QC output folder.
    Creates the final step 07 directory."""
    return step_dir(cfg, model_root, "prediction_qc")


def ensure_dir(path: Path) -> None:
    """Create a directory.
    Parent folders are included."""
    path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """Make objects JSON serializable.
    Handles Path, numpy, and pandas-like values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return None

    return str(value)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON file.
    Uses readable indentation."""
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def write_text(path: Path, lines: Sequence[str]) -> None:
    """Write text report.
    Adds a final newline."""
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# TABLE AND NUMERIC HELPERS
# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or TSV.
    Parser is chosen by suffix."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size <= 2:
        return pd.DataFrame()

    suffix = path.suffix.lower()

    try:
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)

        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)

        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_optional_table(path: Path | None) -> pd.DataFrame:
    """Load optional table.
    Missing or empty files return an empty frame."""
    if path is None or not path.exists() or not path.is_file():
        return pd.DataFrame()

    return load_table(path)


def write_table(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    """Write table.
    Parent folder is created first."""
    ensure_dir(path.parent)
    df.to_csv(path, sep=sep, index=False)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce series to numeric.
    Invalid and infinite values become missing."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_float(value: Any) -> float:
    """Convert value to float.
    Missing and invalid values become NaN."""
    try:
        out = float(value)
        if math.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely.
    Zero denominator returns NaN."""
    if denominator == 0 or pd.isna(denominator):
        return float("nan")

    return float(numerator) / float(denominator)


def compact_list(values: Sequence[Any], max_items: int = 12) -> str:
    """Create compact semicolon list.
    Long lists are truncated."""
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    shown = cleaned[:max_items]

    if len(cleaned) > max_items:
        shown.append(f"plus {len(cleaned) - max_items} more")

    return "; ".join(shown)


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return first matching column.
    Candidate order is preserved."""
    for col in candidates:
        if col in df.columns:
            return col

    return None


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    """Compute Pearson correlation.
    Undefined cases return NaN."""
    x = safe_numeric_series(a)
    y = safe_numeric_series(b)
    mask = x.notna() & y.notna()

    if int(mask.sum()) < 2:
        return float("nan")

    if float(x[mask].std()) == 0 or float(y[mask].std()) == 0:
        return float("nan")

    return float(x[mask].corr(y[mask]))


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Compute Spearman correlation.
    Uses pandas rank fallback."""
    x = safe_numeric_series(a)
    y = safe_numeric_series(b)
    mask = x.notna() & y.notna()

    if int(mask.sum()) < 2:
        return float("nan")

    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")

    if float(xr.std()) == 0 or float(yr.std()) == 0:
        return float("nan")

    return float(xr.corr(yr))


def rmse(a: pd.Series, b: pd.Series) -> float:
    """Compute root mean squared error.
    Missing pairs are removed."""
    x = safe_numeric_series(a)
    y = safe_numeric_series(b)
    mask = x.notna() & y.notna()

    if int(mask.sum()) == 0:
        return float("nan")

    return float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))


def mae(a: pd.Series, b: pd.Series) -> float:
    """Compute mean absolute error.
    Missing pairs are removed."""
    x = safe_numeric_series(a)
    y = safe_numeric_series(b)
    mask = x.notna() & y.notna()

    if int(mask.sum()) == 0:
        return float("nan")

    return float(np.mean(np.abs(x[mask] - y[mask])))


# ============================================================
# INPUT FILE CONTRACT
# ============================================================


def expected_files(cfg: dict[str, Any], model_root: Path) -> list[dict[str, Any]]:
    """Build expected file contract.
    Includes required and optional pipeline artifacts."""
    d01 = step_dir(cfg, model_root, "input_validation")
    d02 = step_dir(cfg, model_root, "modeling_dataset")
    d03 = step_dir(cfg, model_root, "global_model")
    d04 = step_dir(cfg, model_root, "per_treatment_models")
    d05 = find_step05_dir(cfg, model_root)
    d06 = step_dir(cfg, model_root, "all_sample_predictions")

    rows = [
        ("01", "input_validation_summary", d01 / "input_validation_summary.txt", True),
        ("01", "validation_issues", d01 / "validation_issues.tsv", False),
        ("02", "modeling_table", d02 / "modeling_table.tsv", True),
        ("02", "X_features", d02 / "X_features.csv", True),
        ("02", "y_target", d02 / "y_target.csv", True),
        ("02", "model_feature_manifest", d02 / "model_feature_manifest.csv", True),
        ("02", "sample_split", d02 / "sample_split.tsv", True),
        ("02", "feature_quality_report", d02 / "feature_quality_report.tsv", False),
        ("03", "metrics", d03 / "metrics.tsv", True),
        ("03", "predictions_train", d03 / "predictions_train.tsv", True),
        ("03", "predictions_test", d03 / "predictions_test.tsv", True),
        ("03", "predictions_all_labeled", d03 / "predictions_all_labeled.tsv", True),
        ("03", "feature_importance", d03 / "feature_importance.tsv", True),
        ("03", "split_summary", d03 / "split_summary.tsv", False),
        ("04", "per_treatment_model_summary", d04 / "per_treatment_model_summary.tsv", False),
        ("04", "skipped_treatments", d04 / "skipped_treatments.tsv", False),
        ("04", "per_treatment_predictions_all", d04 / "per_treatment_predictions_all.tsv", False),
        ("04", "per_treatment_feature_importance_top", d04 / "per_treatment_feature_importance_top.tsv", False),
        ("05", "global_feature_explanations", d05 / "global_feature_explanations.tsv", False),
        ("05", "global_feature_group_summary", d05 / "global_feature_group_summary.tsv", False),
        ("05", "global_feature_axis_summary", d05 / "global_feature_axis_summary.tsv", False),
        ("05", "global_spatial_vs_drug_summary", d05 / "global_spatial_vs_drug_summary.tsv", False),
        ("05", "global_treatment_explanation_summary", d05 / "global_treatment_explanation_summary.tsv", False),
        ("05", "model_explanation_summary", d05 / "model_explanation_summary.txt", False),
        ("06", "all_sample_treatment_predictions", d06 / "all_sample_treatment_predictions.tsv", True),
        ("06", "prediction_summary_by_sample", d06 / "prediction_summary_by_sample.tsv", False),
        ("06", "prediction_summary_by_treatment", d06 / "prediction_summary_by_treatment.tsv", False),
        ("06", "top_treatment_per_sample", d06 / "top_treatment_per_sample.tsv", False),
        ("06", "teacher_labeled_prediction_comparison", d06 / "teacher_labeled_prediction_comparison.tsv", False),
        ("06", "prediction_matrix_sample_by_treatment", d06 / "prediction_matrix_sample_by_treatment.tsv", False),
    ]

    return [
        {"step": step, "file_key": key, "path": path, "required": required}
        for step, key, path, required in rows
    ]


def table_shape(path: Path) -> tuple[int | float, int | float]:
    """Return table shape for existing files.
    Non-table and empty files return NaN dimensions."""
    if not path.exists() or not path.is_file() or path.stat().st_size <= 2:
        return np.nan, np.nan

    if path.suffix.lower() not in {".tsv", ".csv"}:
        return np.nan, np.nan

    try:
        df = load_table(path)
        return int(df.shape[0]), int(df.shape[1])
    except Exception:
        return np.nan, np.nan


def build_file_contract_report(cfg: dict[str, Any], model_root: Path) -> pd.DataFrame:
    """Build file contract QC table.
    Records file existence, size, and table shape."""
    rows: list[dict[str, Any]] = []

    for spec in expected_files(cfg, model_root):
        path = spec["path"]
        exists = bool(path.exists())
        is_file = bool(exists and path.is_file())
        n_rows, n_cols = table_shape(path)

        if spec["required"] and not is_file:
            status = "missing_required"
        elif is_file:
            status = "ok"
        else:
            status = "missing_optional"

        rows.append(
            {
                "step": spec["step"],
                "file_key": spec["file_key"],
                "path": str(path),
                "required": bool(spec["required"]),
                "exists": exists,
                "is_file": is_file,
                "size_bytes": int(path.stat().st_size) if is_file else np.nan,
                "n_rows": n_rows,
                "n_columns": n_cols,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD PIPELINE OUTPUTS
# ============================================================


def load_pipeline_tables(cfg: dict[str, Any], model_root: Path) -> dict[str, pd.DataFrame]:
    """Load all QC source tables.
    Missing optional tables become empty frames."""
    d02 = step_dir(cfg, model_root, "modeling_dataset")
    d03 = step_dir(cfg, model_root, "global_model")
    d04 = step_dir(cfg, model_root, "per_treatment_models")
    d05 = find_step05_dir(cfg, model_root)
    d06 = step_dir(cfg, model_root, "all_sample_predictions")

    paths = {
        "modeling_table": d02 / "modeling_table.tsv",
        "x_features": d02 / "X_features.csv",
        "y_target": d02 / "y_target.csv",
        "feature_manifest": d02 / "model_feature_manifest.csv",
        "sample_split": d02 / "sample_split.tsv",
        "feature_quality": d02 / "feature_quality_report.tsv",
        "model_metrics": d03 / "metrics.tsv",
        "pred_train": d03 / "predictions_train.tsv",
        "pred_test": d03 / "predictions_test.tsv",
        "pred_all": d03 / "predictions_all_labeled.tsv",
        "feature_importance": d03 / "feature_importance.tsv",
        "split_summary": d03 / "split_summary.tsv",
        "per_treatment_summary": d04 / "per_treatment_model_summary.tsv",
        "skipped_treatments": d04 / "skipped_treatments.tsv",
        "per_treatment_importance": d04 / "per_treatment_feature_importance_top.tsv",
        "explain_features": d05 / "global_feature_explanations.tsv",
        "explain_groups": d05 / "global_feature_group_summary.tsv",
        "explain_axes": d05 / "global_feature_axis_summary.tsv",
        "explain_spatial_vs_drug": d05 / "global_spatial_vs_drug_summary.tsv",
        "explain_treatment": d05 / "global_treatment_explanation_summary.tsv",
        "all_predictions": d06 / "all_sample_treatment_predictions.tsv",
        "pred_by_sample": d06 / "prediction_summary_by_sample.tsv",
        "pred_by_treatment": d06 / "prediction_summary_by_treatment.tsv",
        "top_treatment": d06 / "top_treatment_per_sample.tsv",
        "teacher_comparison": d06 / "teacher_labeled_prediction_comparison.tsv",
        "prediction_matrix": d06 / "prediction_matrix_sample_by_treatment.tsv",
    }

    tables = {key: load_optional_table(path) for key, path in paths.items()}
    tables["_paths"] = pd.DataFrame({"table": list(paths.keys()), "path": [str(p) for p in paths.values()]})

    return tables


# ============================================================
# QC SUMMARIES
# ============================================================


def identify_drug_dummy_cols(columns: Sequence[str], cfg: dict[str, Any]) -> list[str]:
    """Identify drug dummy columns.
    Uses configured prefixes plus common one-hot prefixes."""
    prefixes = get_cfg(
        cfg,
        "drug_dummy_prefixes",
        ["drug__", "drug_dummy__", "drug_key__", "drug_", "drug_key_", "treatment__"],
    )
    prefixes = [str(p).lower() for p in prefixes]

    return [col for col in columns if any(str(col).lower().startswith(prefix) for prefix in prefixes)]


def build_dataset_summary(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Build dataset shape summary.
    Combines modeling table, X matrix, manifest, and predictions."""
    modeling = tables["modeling_table"]
    X = tables["x_features"]
    manifest = tables["feature_manifest"]
    all_pred = tables["all_predictions"]

    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    drug_dummy_cols = identify_drug_dummy_cols(list(X.columns), cfg) if not X.empty else []

    rows = [
        {
            "run_name": get_cfg(cfg, "run_name", ""),
            "run_scope": get_cfg(cfg, "run_scope", ""),
            "n_labeled_rows": int(len(modeling)),
            "n_labeled_samples": int(modeling[sample_col].nunique()) if sample_col in modeling.columns else np.nan,
            "n_labeled_treatments": int(modeling[drug_key_col].nunique()) if drug_key_col in modeling.columns else np.nan,
            "n_x_rows": int(len(X)),
            "n_x_features": int(X.shape[1]) if not X.empty else 0,
            "n_drug_dummy_features": int(len(drug_dummy_cols)),
            "n_spatial_features_in_x": int(max(0, X.shape[1] - len(drug_dummy_cols))) if not X.empty else 0,
            "n_manifest_rows": int(len(manifest)),
            "n_prediction_rows": int(len(all_pred)),
            "n_prediction_samples": int(all_pred[sample_col].nunique()) if sample_col in all_pred.columns else np.nan,
            "n_prediction_treatments": int(all_pred[drug_key_col].nunique()) if drug_key_col in all_pred.columns else np.nan,
            "n_prediction_drug_names": int(all_pred[drug_col].nunique()) if drug_col in all_pred.columns else np.nan,
        }
    ]

    return pd.DataFrame(rows)


def build_split_summary(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Build split QC table.
    Uses step 03 split summary when available."""
    if not tables["split_summary"].empty:
        return tables["split_summary"].copy()

    modeling = tables["modeling_table"]
    sample_split = tables["sample_split"]

    if modeling.empty or sample_split.empty:
        return pd.DataFrame()

    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    if sample_col not in modeling.columns or sample_col not in sample_split.columns or "split" not in sample_split.columns:
        return pd.DataFrame()

    work = modeling.merge(sample_split[[sample_col, "split"]].drop_duplicates(sample_col), on=sample_col, how="left")

    rows = []
    for split_name, df in work.groupby("split", dropna=False):
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(df)),
                "n_samples": int(df[sample_col].nunique()),
                "n_treatments": int(df[drug_key_col].nunique()) if drug_key_col in df.columns else np.nan,
                "target_mean": float(safe_numeric_series(df[target_col]).mean()) if target_col in df.columns else np.nan,
                "target_std": float(safe_numeric_series(df[target_col]).std()) if target_col in df.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_model_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return global model metrics.
    Adds a source label for QC exports."""
    metrics = tables["model_metrics"].copy()

    if metrics.empty:
        return pd.DataFrame()

    metrics.insert(0, "metric_source", "03_global_model")
    return metrics


def prediction_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction distribution.
    Includes quantiles and teacher overlap count."""
    if predictions.empty or "predicted_fused_prob_responder" not in predictions.columns:
        return pd.DataFrame()

    y = safe_numeric_series(predictions["predicted_fused_prob_responder"])

    rows = [
        {
            "n_rows": int(len(predictions)),
            "n_nonmissing_predictions": int(y.notna().sum()),
            "mean_prediction": float(y.mean()),
            "std_prediction": float(y.std()),
            "min_prediction": float(y.min()),
            "q10_prediction": float(y.quantile(0.10)),
            "q25_prediction": float(y.quantile(0.25)),
            "median_prediction": float(y.median()),
            "q75_prediction": float(y.quantile(0.75)),
            "q90_prediction": float(y.quantile(0.90)),
            "max_prediction": float(y.max()),
            "n_teacher_labeled": int(predictions["is_teacher_labeled"].sum()) if "is_teacher_labeled" in predictions.columns else 0,
        }
    ]

    return pd.DataFrame(rows)


def teacher_overlap_metrics(comp: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Compute prediction versus teacher metrics.
    Optional grouping supports treatment and sample summaries."""
    if comp.empty:
        return pd.DataFrame()

    pred_col = first_existing_column(comp, ["predicted_fused_prob_responder", "predicted_prob_responder"])
    teacher_col = first_existing_column(comp, ["teacher_fused_prob_responder", "fused_prob_responder", "y_true"])

    if pred_col is None or teacher_col is None:
        return pd.DataFrame()

    group_cols = list(group_cols or [])
    rows: list[dict[str, Any]] = []

    if group_cols:
        iterator = comp.groupby(group_cols, dropna=False)
    else:
        iterator = [("overall", comp)]

    for key, df in iterator:
        pred = safe_numeric_series(df[pred_col])
        teacher = safe_numeric_series(df[teacher_col])
        mask = pred.notna() & teacher.notna()

        if isinstance(key, tuple):
            key_values = dict(zip(group_cols, key))
        elif group_cols:
            key_values = {group_cols[0]: key}
        else:
            key_values = {"group": "overall"}

        rows.append(
            {
                **key_values,
                "n_rows": int(len(df)),
                "n_valid_pairs": int(mask.sum()),
                "teacher_mean": float(teacher[mask].mean()) if mask.any() else np.nan,
                "prediction_mean": float(pred[mask].mean()) if mask.any() else np.nan,
                "mean_error_prediction_minus_teacher": float((pred[mask] - teacher[mask]).mean()) if mask.any() else np.nan,
                "mae": mae(pred, teacher),
                "rmse": rmse(pred, teacher),
                "pearson": pearson_corr(pred, teacher),
                "spearman": spearman_corr(pred, teacher),
            }
        )

    return pd.DataFrame(rows)


def build_prediction_by_sample(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Build sample-level prediction summary.
    Uses step 06 summary when available."""
    existing = tables["pred_by_sample"]

    if not existing.empty:
        return existing.copy()

    predictions = tables["all_predictions"]
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if predictions.empty or sample_col not in predictions.columns:
        return pd.DataFrame()

    return (
        predictions.groupby(sample_col, as_index=False)
        .agg(
            n_treatments_predicted=("predicted_fused_prob_responder", "size"),
            mean_predicted_response=("predicted_fused_prob_responder", "mean"),
            median_predicted_response=("predicted_fused_prob_responder", "median"),
            min_predicted_response=("predicted_fused_prob_responder", "min"),
            max_predicted_response=("predicted_fused_prob_responder", "max"),
            std_predicted_response=("predicted_fused_prob_responder", "std"),
        )
        .reset_index(drop=True)
    )


def build_prediction_by_treatment(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Build treatment-level prediction summary.
    Uses step 06 summary when available."""
    existing = tables["pred_by_treatment"]

    if not existing.empty:
        return existing.copy()

    predictions = tables["all_predictions"]
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    if predictions.empty or drug_col not in predictions.columns:
        return pd.DataFrame()

    return (
        predictions.groupby([drug_col, drug_key_col], as_index=False)
        .agg(
            n_samples_predicted=("predicted_fused_prob_responder", "size"),
            mean_predicted_response=("predicted_fused_prob_responder", "mean"),
            median_predicted_response=("predicted_fused_prob_responder", "median"),
            min_predicted_response=("predicted_fused_prob_responder", "min"),
            max_predicted_response=("predicted_fused_prob_responder", "max"),
            std_predicted_response=("predicted_fused_prob_responder", "std"),
        )
        .sort_values("mean_predicted_response", ascending=False)
        .reset_index(drop=True)
    )


def build_feature_explanation_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return feature explanation summary.
    Falls back to step 03 feature importance."""
    explain = tables["explain_features"]

    if not explain.empty:
        return explain.copy()

    imp = tables["feature_importance"].copy()

    if imp.empty:
        return pd.DataFrame()

    if "primary_explanation_score" not in imp.columns and "importance" in imp.columns:
        imp["primary_explanation_score"] = safe_numeric_series(imp["importance"])
        imp["primary_explanation_method"] = "model_native_importance"

    if "feature_class" not in imp.columns:
        imp["feature_class"] = np.where(imp.get("is_drug_dummy", False).astype(str).str.lower().eq("true"), "drug_identity", "spatial_feature") if "is_drug_dummy" in imp.columns else "unannotated"

    return imp


def build_spatial_vs_drug_summary(tables: dict[str, pd.DataFrame], feature_summary: pd.DataFrame) -> pd.DataFrame:
    """Build spatial versus drug contribution table.
    Uses step 05 output if present."""
    existing = tables["explain_spatial_vs_drug"]

    if not existing.empty:
        return existing.copy()

    if feature_summary.empty or "feature_class" not in feature_summary.columns:
        return pd.DataFrame()

    score_col = "primary_explanation_score" if "primary_explanation_score" in feature_summary.columns else "importance"
    if score_col not in feature_summary.columns:
        return pd.DataFrame()

    work = feature_summary.copy()
    work[score_col] = safe_numeric_series(work[score_col]).fillna(0.0)

    summary = (
        work.groupby("feature_class", as_index=False)
        .agg(
            n_features=("feature_class", "size"),
            total_explanation_score=(score_col, "sum"),
            mean_explanation_score=(score_col, "mean"),
        )
        .sort_values("total_explanation_score", ascending=False)
        .reset_index(drop=True)
    )

    total = float(summary["total_explanation_score"].sum())
    summary["fraction_of_total_score"] = summary["total_explanation_score"].map(lambda x: safe_divide(x, total))

    return summary


def build_per_treatment_status(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build per-treatment model status table.
    Handles disabled step 04 outputs."""
    summary = tables["per_treatment_summary"].copy()

    if summary.empty:
        return pd.DataFrame(
            [
                {
                    "status_summary": "missing_or_empty",
                    "n_rows": 0,
                    "n_trained": 0,
                    "n_skipped": 0,
                    "n_failed": 0,
                }
            ]
        )

    status_col = "status" if "status" in summary.columns else None

    if status_col is None:
        return pd.DataFrame(
            [
                {
                    "status_summary": "present_no_status_column",
                    "n_rows": int(len(summary)),
                    "n_trained": np.nan,
                    "n_skipped": np.nan,
                    "n_failed": np.nan,
                }
            ]
        )

    counts = summary[status_col].astype(str).value_counts().to_dict()

    return pd.DataFrame(
        [
            {
                "status_summary": "present",
                "n_rows": int(len(summary)),
                "n_trained": int(counts.get("trained", 0)),
                "n_skipped": int(counts.get("skipped", 0)),
                "n_failed": int(counts.get("failed", 0)),
                "status_counts": json.dumps(counts),
            }
        ]
    )


def build_qc_summary(
    file_report: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_dist: pd.DataFrame,
    teacher_metrics: pd.DataFrame,
    per_treatment_status: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact one-row QC summary.
    Designed for quick dashboard reading."""
    required_missing = int(((file_report["required"] == True) & (file_report["is_file"] != True)).sum())

    row: dict[str, Any] = {
        "required_files_missing": required_missing,
        "n_files_ok": int((file_report["status"] == "ok").sum()),
        "n_files_missing_optional": int((file_report["status"] == "missing_optional").sum()),
    }

    if not dataset_summary.empty:
        row.update(dataset_summary.iloc[0].to_dict())

    if not split_summary.empty and "split" in split_summary.columns:
        row["split_labels"] = compact_list(split_summary["split"].astype(str).tolist())
        if "n_rows" in split_summary.columns:
            row["total_split_rows"] = int(safe_numeric_series(split_summary["n_rows"]).sum())

    if not metrics.empty:
        test_metrics = metrics[metrics.get("split", "").astype(str).str.lower().eq("test")] if "split" in metrics.columns else pd.DataFrame()
        if not test_metrics.empty:
            for col in ["mae", "rmse", "r2", "pearson", "spearman", "auc_at_binary_threshold"]:
                if col in test_metrics.columns:
                    row[f"test_{col}"] = safe_float(test_metrics.iloc[0][col])

    if not prediction_dist.empty:
        row.update({f"prediction_{k}": v for k, v in prediction_dist.iloc[0].to_dict().items()})

    if not teacher_metrics.empty:
        overall = teacher_metrics.iloc[0]
        for col in ["n_valid_pairs", "mae", "rmse", "pearson", "spearman"]:
            if col in overall:
                row[f"teacher_overlap_{col}"] = overall[col]

    if not per_treatment_status.empty:
        row.update({f"per_treatment_{k}": v for k, v in per_treatment_status.iloc[0].to_dict().items()})

    return pd.DataFrame([row])


# ============================================================
# FIGURE HELPERS
# ============================================================


def configure_matplotlib() -> Any:
    """Import matplotlib for file rendering.
    Uses Agg backend for non-interactive runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_current_figure(plt: Any, path: Path) -> None:
    """Save current matplotlib figure.
    Applies tight layout and closes figure."""
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def short_label(value: Any, max_len: int = 55) -> str:
    """Shorten long labels for plots.
    Keeps presentation text readable."""
    text = clean_text(value)

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def figure_manifest_row(path: Path, title: str, source: str, note: str, slide_use: str) -> dict[str, Any]:
    """Create one figure manifest row.
    Centralizes figure metadata."""
    return {
        "figure_file": str(path),
        "figure_name": path.name,
        "title": title,
        "source": source,
        "note": note,
        "presentation_use": slide_use,
        "exists": path.exists(),
    }


def make_pipeline_file_contract_figure(file_report: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot file contract status by step.
    Useful as an end-to-end pipeline dashboard."""
    if file_report.empty:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_01_pipeline_file_contract.png"

    work = file_report.copy()
    work["is_ok"] = work["status"].eq("ok")
    summary = work.groupby("step", as_index=False).agg(
        n_ok=("is_ok", "sum"),
        n_files=("file_key", "size"),
    )
    summary["fraction_ok"] = summary["n_ok"] / summary["n_files"]

    plt.figure(figsize=(9, 5))
    plt.bar(summary["step"].astype(str), summary["fraction_ok"])
    plt.ylim(0, 1.05)
    plt.xlabel("Pipeline step")
    plt.ylabel("Fraction of expected files present")
    plt.title("Pipeline file contract completion")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Pipeline file contract completion",
        "qc_file_contract_report.tsv",
        "Checks whether expected files exist for each pipeline step",
        "Pipeline completion dashboard",
    )


def make_observed_vs_predicted_figure(pred_test: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot observed versus predicted test response.
    Main global model performance figure."""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    pred_col = first_existing_column(pred_test, ["predicted_fused_prob_responder", "predicted_prob_responder", "prediction"])

    if pred_test.empty or target_col not in pred_test.columns or pred_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_02_observed_vs_predicted_test.png"

    x = safe_numeric_series(pred_test[target_col])
    y = safe_numeric_series(pred_test[pred_col])
    mask = x.notna() & y.notna()

    plt.figure(figsize=(6.5, 6))
    plt.scatter(x[mask], y[mask], alpha=0.75)
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Teacher fused response")
    plt.ylabel("Predicted fused response")
    plt.title("Observed versus predicted response, test split")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Observed versus predicted response",
        "03_global_model/predictions_test.tsv",
        "Diagonal line marks perfect prediction",
        "Main model performance slide",
    )


def make_residual_distribution_figure(pred_test: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot residual distribution.
    Shows prediction error spread and bias."""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    pred_col = first_existing_column(pred_test, ["predicted_fused_prob_responder", "predicted_prob_responder", "prediction"])

    if pred_test.empty or target_col not in pred_test.columns or pred_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_03_prediction_residual_distribution.png"

    residual = safe_numeric_series(pred_test[pred_col]) - safe_numeric_series(pred_test[target_col])
    residual = residual.dropna()

    if residual.empty:
        return None

    plt.figure(figsize=(7, 5))
    plt.hist(residual, bins=20)
    plt.axvline(0, linestyle=":")
    plt.xlabel("Predicted minus teacher response")
    plt.ylabel("Number of rows")
    plt.title("Prediction residual distribution, test split")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Prediction residual distribution",
        "03_global_model/predictions_test.tsv",
        "Residual is predicted minus teacher response",
        "Model error and bias slide",
    )


def make_prediction_heatmap_figure(predictions: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot sample by treatment prediction heatmap.
    Strong visual for treatment ranking patterns."""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))

    if predictions.empty or sample_col not in predictions.columns or drug_col not in predictions.columns:
        return None

    if "predicted_fused_prob_responder" not in predictions.columns:
        return None

    matrix = predictions.pivot_table(
        index=sample_col,
        columns=drug_col,
        values="predicted_fused_prob_responder",
        aggfunc="mean",
    )

    if matrix.empty:
        return None

    # Limit very large heatmaps for presentation readability
    max_cols = int(get_cfg(cfg, "qc_heatmap_max_treatments", 40))
    if matrix.shape[1] > max_cols:
        means = matrix.mean(axis=0).sort_values(ascending=False)
        matrix = matrix[means.head(max_cols).index]

    plt = configure_matplotlib()
    path = fig_dir / "fig_04_prediction_heatmap_sample_by_treatment.png"

    height = max(5, min(14, 0.35 * matrix.shape[0] + 3))
    width = max(8, min(18, 0.28 * matrix.shape[1] + 4))

    plt.figure(figsize=(width, height))
    plt.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="Predicted response")
    plt.yticks(range(matrix.shape[0]), [short_label(x, 30) for x in matrix.index])

    if matrix.shape[1] <= 45:
        plt.xticks(range(matrix.shape[1]), [short_label(x, 24) for x in matrix.columns], rotation=90)
    else:
        plt.xticks([])

    plt.xlabel("Treatment")
    plt.ylabel("Sample")
    plt.title("Predicted response heatmap")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Predicted response heatmap",
        "06_all_sample_predictions/all_sample_treatment_predictions.tsv",
        "Rows are samples and columns are treatments",
        "Core spatial prediction output slide",
    )


def make_treatment_mean_figure(by_treatment: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot top treatments by mean prediction.
    Cohort-level ranking figure."""
    label_col = first_existing_column(by_treatment, ["drug", "drug_key"])
    value_col = first_existing_column(by_treatment, ["mean_predicted_response", "mean_prediction"])

    if by_treatment.empty or label_col is None or value_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_05_treatment_mean_predicted_response_top25.png"

    work = by_treatment[[label_col, value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=False).head(25).iloc[::-1]

    if work.empty:
        return None

    plt.figure(figsize=(10, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work[label_col]], work[value_col])
    plt.xlabel("Mean predicted response")
    plt.title("Top treatments by mean predicted response")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Top treatments by mean predicted response",
        "06_all_sample_predictions/prediction_summary_by_treatment.tsv",
        "Ranks treatments by cohort mean predicted response",
        "Treatment ranking overview slide",
    )


def make_top_treatment_per_sample_figure(top_treatment: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot top treatment per sample.
    Shows best ranked treatment for each sample."""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))

    if top_treatment.empty or sample_col not in top_treatment.columns or drug_col not in top_treatment.columns:
        return None

    if "predicted_fused_prob_responder" not in top_treatment.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_06_top_treatment_per_sample.png"

    work = top_treatment.copy()
    work["label"] = work[sample_col].astype(str) + " | " + work[drug_col].astype(str).map(lambda x: short_label(x, 28))
    work = work.sort_values("predicted_fused_prob_responder", ascending=True)

    max_rows = int(get_cfg(cfg, "qc_top_treatment_max_samples", 35))
    if len(work) > max_rows:
        work = work.tail(max_rows)

    plt.figure(figsize=(11, max(5, 0.36 * len(work) + 2)))
    plt.barh(work["label"], safe_numeric_series(work["predicted_fused_prob_responder"]))
    plt.xlabel("Predicted response")
    plt.title("Top predicted treatment per sample")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Top predicted treatment per sample",
        "06_all_sample_predictions/top_treatment_per_sample.tsv",
        "Each bar is the highest predicted treatment for one sample",
        "Personalized treatment ranking slide",
    )


def make_sample_distribution_figure(predictions: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot prediction distribution by sample.
    Shows within-sample treatment heterogeneity."""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if predictions.empty or sample_col not in predictions.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_07_sample_prediction_distribution.png"

    work = predictions[[sample_col, "predicted_fused_prob_responder"]].dropna().copy()
    if work.empty:
        return None

    order = work.groupby(sample_col)["predicted_fused_prob_responder"].mean().sort_values(ascending=False).index.tolist()
    max_samples = int(get_cfg(cfg, "qc_sample_boxplot_max_samples", 35))
    order = order[:max_samples]

    data = [work.loc[work[sample_col] == sample, "predicted_fused_prob_responder"].to_numpy() for sample in order]

    plt.figure(figsize=(max(9, 0.35 * len(order) + 4), 6))
    plt.boxplot(data, labels=[short_label(s, 20) for s in order], showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Predicted response")
    plt.title("Prediction distribution across treatments by sample")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Sample-level prediction heterogeneity",
        "06_all_sample_predictions/all_sample_treatment_predictions.tsv",
        "Boxplots show treatment response spread within each sample",
        "Sample heterogeneity slide",
    )


def make_treatment_distribution_figure(predictions: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot prediction distribution by treatment.
    Shows across-sample treatment heterogeneity."""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))

    if predictions.empty or drug_col not in predictions.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_08_treatment_prediction_distribution_top25.png"

    work = predictions[[drug_col, "predicted_fused_prob_responder"]].dropna().copy()
    if work.empty:
        return None

    order = work.groupby(drug_col)["predicted_fused_prob_responder"].mean().sort_values(ascending=False).head(25).index.tolist()
    data = [work.loc[work[drug_col] == drug, "predicted_fused_prob_responder"].to_numpy() for drug in order]

    plt.figure(figsize=(max(10, 0.45 * len(order) + 4), 6))
    plt.boxplot(data, labels=[short_label(s, 20) for s in order], showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Predicted response")
    plt.title("Prediction distribution across samples by treatment")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Treatment-level prediction heterogeneity",
        "06_all_sample_predictions/all_sample_treatment_predictions.tsv",
        "Top 25 treatments by mean predicted response",
        "Treatment heterogeneity slide",
    )


def make_spatial_vs_drug_figure(spatial_vs_drug: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot spatial versus drug contribution.
    Explains drug dummy dominance."""
    if spatial_vs_drug.empty or "feature_class" not in spatial_vs_drug.columns:
        return None

    value_col = first_existing_column(spatial_vs_drug, ["fraction_of_total_score", "total_explanation_score"])
    if value_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_09_spatial_vs_drug_identity_contribution.png"

    work = spatial_vs_drug[["feature_class", value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=True)

    plt.figure(figsize=(7, 4.5))
    plt.barh(work["feature_class"], work[value_col])
    plt.xlabel(value_col)
    plt.title("Drug identity versus spatial biology contribution")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Drug identity versus spatial biology contribution",
        "05_model_explanation/global_spatial_vs_drug_summary.tsv",
        "Separates treatment identity features from spatial features",
        "Model interpretation context slide",
    )


def feature_label_column(feature_summary: pd.DataFrame) -> str:
    """Pick readable feature label column.
    Original feature names are preferred."""
    return first_existing_column(feature_summary, ["feature_original", "feature_label", "feature_name"]) or "feature_name"


def make_top_feature_figure(feature_summary: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot top full-model features.
    Includes drug identity and spatial features."""
    if feature_summary.empty:
        return None

    label_col = feature_label_column(feature_summary)
    value_col = first_existing_column(feature_summary, ["primary_explanation_score", "mean_abs_shap", "importance"])

    if value_col is None or label_col not in feature_summary.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_10_top_global_features_full_model.png"

    work = feature_summary[[label_col, value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=False).head(25).iloc[::-1]

    if work.empty:
        return None

    plt.figure(figsize=(11, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work[label_col]], work[value_col])
    plt.xlabel("Explanation score")
    plt.title("Top global model features")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Top global model features",
        "05_model_explanation/global_feature_explanations.tsv",
        "Full model includes treatment identity and spatial biology",
        "Global model interpretation slide",
    )


def make_top_spatial_feature_figure(feature_summary: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot top spatial features only.
    Main biological interpretation figure."""
    if feature_summary.empty:
        return None

    work = feature_summary.copy()

    if "feature_class" in work.columns:
        work = work[work["feature_class"].astype(str).eq("spatial_feature")].copy()
    elif "is_drug_dummy" in work.columns:
        work = work[~work["is_drug_dummy"].astype(str).str.lower().eq("true")].copy()
    else:
        work = work[~work["feature_name"].astype(str).str.startswith("drug__")].copy()

    label_col = feature_label_column(work)
    value_col = first_existing_column(work, ["primary_explanation_score", "mean_abs_shap", "importance"])

    if work.empty or value_col is None or label_col not in work.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_11_top_spatial_features_only.png"

    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=False).head(25).iloc[::-1]

    if work.empty:
        return None

    plt.figure(figsize=(11, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work[label_col]], work[value_col])
    plt.xlabel("Explanation score")
    plt.title("Top spatial features only")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Top spatial features only",
        "05_model_explanation/global_feature_explanations.tsv",
        "Drug dummy features removed for biological interpretation",
        "Biological interpretation slide",
    )


def make_group_contribution_figure(group_summary: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot feature group contribution.
    Converts many features into biology groups."""
    if group_summary.empty or "feature_group" not in group_summary.columns:
        return None

    value_col = first_existing_column(group_summary, ["total_explanation_score", "fraction_of_total_score"])
    if value_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_12_feature_group_contribution.png"

    work = group_summary[["feature_group", value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=False).head(25).iloc[::-1]

    if work.empty:
        return None

    plt.figure(figsize=(10, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work["feature_group"]], work[value_col])
    plt.xlabel(value_col)
    plt.title("Feature group contribution")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Feature group contribution",
        "05_model_explanation/global_feature_group_summary.tsv",
        "Aggregates model explanation score by feature group",
        "Biology group interpretation slide",
    )


def make_axis_contribution_figure(axis_summary: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot feature axis contribution.
    Summarizes biology and spatial axes."""
    if axis_summary.empty or "feature_axis" not in axis_summary.columns:
        return None

    value_col = first_existing_column(axis_summary, ["total_explanation_score", "fraction_of_total_score"])
    if value_col is None:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_13_feature_axis_contribution.png"

    work = axis_summary[["feature_axis", value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna().sort_values(value_col, ascending=False).head(25).iloc[::-1]

    if work.empty:
        return None

    plt.figure(figsize=(10, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work["feature_axis"]], work[value_col])
    plt.xlabel(value_col)
    plt.title("Feature axis contribution")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Feature axis contribution",
        "05_model_explanation/global_feature_axis_summary.tsv",
        "Aggregates model explanation score by biological axis",
        "High-level biology theme slide",
    )


def make_treatment_explanation_heatmap(treatment_exp: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot treatment by feature group explanation heatmap.
    Shows treatment-specific explanation patterns."""
    if treatment_exp.empty:
        return None

    drug_col = first_existing_column(treatment_exp, ["drug", "drug_key"])
    group_col = first_existing_column(treatment_exp, ["feature_group", "feature_name"])
    value_col = first_existing_column(treatment_exp, ["mean_abs_shap", "primary_explanation_score", "importance"])

    if drug_col is None or group_col is None or value_col is None:
        return None

    work = treatment_exp[[drug_col, group_col, value_col]].copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.dropna()

    if work.empty:
        return None

    # Collapse to feature groups and keep top groups for readability
    group_scores = work.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(15)
    drug_scores = work.groupby(drug_col)[value_col].sum().sort_values(ascending=False).head(30)

    work = work[work[group_col].isin(group_scores.index) & work[drug_col].isin(drug_scores.index)].copy()
    matrix = work.pivot_table(index=drug_col, columns=group_col, values=value_col, aggfunc="sum", fill_value=0.0)

    if matrix.empty:
        return None

    matrix = matrix.loc[drug_scores.index.intersection(matrix.index), group_scores.index.intersection(matrix.columns)]

    plt = configure_matplotlib()
    path = fig_dir / "fig_14_treatment_explanation_heatmap.png"

    plt.figure(figsize=(max(8, 0.45 * matrix.shape[1] + 5), max(6, 0.30 * matrix.shape[0] + 3)))
    plt.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="Explanation score")
    plt.yticks(range(matrix.shape[0]), [short_label(x, 30) for x in matrix.index])
    plt.xticks(range(matrix.shape[1]), [short_label(x, 24) for x in matrix.columns], rotation=90)
    plt.xlabel("Feature group")
    plt.ylabel("Treatment")
    plt.title("Treatment explanation heatmap")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Treatment explanation heatmap",
        "05_model_explanation/global_treatment_explanation_summary.tsv",
        "Rows are treatments and columns are top explanation groups",
        "Treatment-specific mechanism slide",
    )


def make_feature_missingness_figure(feature_quality: pd.DataFrame, fig_dir: Path) -> dict[str, Any] | None:
    """Plot spatial feature missingness.
    QC figure for feature usability."""
    if feature_quality.empty or "missing_fraction" not in feature_quality.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_15_spatial_feature_missingness.png"

    miss = safe_numeric_series(feature_quality["missing_fraction"]).dropna()

    if miss.empty:
        return None

    plt.figure(figsize=(7, 5))
    plt.hist(miss, bins=25)
    plt.xlabel("Missing fraction")
    plt.ylabel("Number of features")
    plt.title("Spatial feature missingness")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Spatial feature missingness",
        "02_modeling_dataset/feature_quality_report.tsv",
        "Shows missingness distribution for candidate spatial features",
        "Feature quality QC slide",
    )


def make_teacher_error_by_treatment_figure(comp: pd.DataFrame, cfg: dict[str, Any], fig_dir: Path) -> dict[str, Any] | None:
    """Plot teacher overlap MAE by treatment.
    Shows where global model matches teacher labels."""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))

    if comp.empty or drug_col not in comp.columns:
        return None

    if "absolute_error_vs_teacher" not in comp.columns:
        return None

    plt = configure_matplotlib()
    path = fig_dir / "fig_16_teacher_overlap_error_by_treatment.png"

    work = (
        comp.groupby(drug_col, as_index=False)
        .agg(mean_absolute_error=("absolute_error_vs_teacher", "mean"), n_labeled=("absolute_error_vs_teacher", "count"))
        .sort_values("mean_absolute_error", ascending=False)
        .head(25)
        .iloc[::-1]
    )

    if work.empty:
        return None

    plt.figure(figsize=(10, max(5, 0.35 * len(work) + 2)))
    plt.barh([short_label(x) for x in work[drug_col]], work["mean_absolute_error"])
    plt.xlabel("Mean absolute error")
    plt.title("Teacher overlap error by treatment")

    save_current_figure(plt, path)

    return figure_manifest_row(
        path,
        "Teacher overlap error by treatment",
        "06_all_sample_predictions/teacher_labeled_prediction_comparison.tsv",
        "Ranks treatments by prediction error on teacher-labeled rows",
        "Model limitation and QC slide",
    )


def write_all_figures(
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Generate all presentation-ready figures.
    Missing inputs are skipped gracefully."""
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    figure_rows: list[dict[str, Any]] = []

    makers = [
        lambda: make_pipeline_file_contract_figure(summaries["file_contract"], fig_dir),
        lambda: make_observed_vs_predicted_figure(tables["pred_test"], cfg, fig_dir),
        lambda: make_residual_distribution_figure(tables["pred_test"], cfg, fig_dir),
        lambda: make_prediction_heatmap_figure(tables["all_predictions"], cfg, fig_dir),
        lambda: make_treatment_mean_figure(summaries["by_treatment"], fig_dir),
        lambda: make_top_treatment_per_sample_figure(summaries["top_treatment"], cfg, fig_dir),
        lambda: make_sample_distribution_figure(tables["all_predictions"], cfg, fig_dir),
        lambda: make_treatment_distribution_figure(tables["all_predictions"], cfg, fig_dir),
        lambda: make_spatial_vs_drug_figure(summaries["spatial_vs_drug"], fig_dir),
        lambda: make_top_feature_figure(summaries["feature_explanation"], fig_dir),
        lambda: make_top_spatial_feature_figure(summaries["feature_explanation"], fig_dir),
        lambda: make_group_contribution_figure(tables["explain_groups"], fig_dir),
        lambda: make_axis_contribution_figure(tables["explain_axes"], fig_dir),
        lambda: make_treatment_explanation_heatmap(tables["explain_treatment"], fig_dir),
        lambda: make_feature_missingness_figure(tables["feature_quality"], fig_dir),
        lambda: make_teacher_error_by_treatment_figure(summaries["teacher_comparison"], cfg, fig_dir),
    ]

    for make in makers:
        try:
            row = make()
            if row is not None:
                figure_rows.append(row)
        except Exception as exc:
            # Figure failures should not stop QC table generation
            figure_rows.append(
                {
                    "figure_file": "",
                    "figure_name": "",
                    "title": "figure generation failed",
                    "source": "",
                    "note": f"{type(exc).__name__}: {exc}",
                    "presentation_use": "",
                    "exists": False,
                }
            )

    return pd.DataFrame(figure_rows)


# ============================================================
# SUMMARY TEXT
# ============================================================


def write_qc_summary_text(
    path: Path,
    qc_summary: pd.DataFrame,
    file_report: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_dist: pd.DataFrame,
    teacher_metrics: pd.DataFrame,
    spatial_vs_drug: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """Write human-readable QC summary.
    Focuses on presentation-relevant results."""
    lines: list[str] = []

    lines.append("Spatial prediction QC summary")
    lines.append("")
    lines.append("Run settings")
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  output_root: {get_cfg(cfg, 'output_root', '')}")
    lines.append("")

    if not qc_summary.empty:
        row = qc_summary.iloc[0]
        lines.append("Pipeline contract")
        lines.append(f"  required files missing: {row.get('required_files_missing', np.nan)}")
        lines.append(f"  files ok: {row.get('n_files_ok', np.nan)}")
        lines.append(f"  optional files missing: {row.get('n_files_missing_optional', np.nan)}")
        lines.append("")

    if not dataset_summary.empty:
        row = dataset_summary.iloc[0]
        lines.append("Dataset")
        lines.append(f"  labeled rows: {row.get('n_labeled_rows', np.nan)}")
        lines.append(f"  labeled samples: {row.get('n_labeled_samples', np.nan)}")
        lines.append(f"  labeled treatments: {row.get('n_labeled_treatments', np.nan)}")
        lines.append(f"  X features: {row.get('n_x_features', np.nan)}")
        lines.append(f"  drug dummy features: {row.get('n_drug_dummy_features', np.nan)}")
        lines.append(f"  prediction rows: {row.get('n_prediction_rows', np.nan)}")
        lines.append("")

    if not split_summary.empty:
        lines.append("Split")
        for _, row in split_summary.iterrows():
            split = row.get("split", "")
            n_rows = row.get("n_rows", "")
            n_samples = row.get("n_samples", "")
            n_treatments = row.get("n_treatments", "")
            lines.append(f"  {split}: rows={n_rows}; samples={n_samples}; treatments={n_treatments}")
        lines.append("")

    if not metrics.empty and "split" in metrics.columns:
        lines.append("Global model metrics")
        for _, row in metrics.iterrows():
            split = row.get("split", "")
            mae_value = row.get("mae", np.nan)
            rmse_value = row.get("rmse", np.nan)
            r2_value = row.get("r2", np.nan)
            pearson_value = row.get("pearson", np.nan)
            lines.append(f"  {split}: mae={safe_float(mae_value):.4f}; rmse={safe_float(rmse_value):.4f}; r2={safe_float(r2_value):.4f}; pearson={safe_float(pearson_value):.4f}")
        lines.append("")

    if not prediction_dist.empty:
        row = prediction_dist.iloc[0]
        lines.append("Prediction distribution")
        lines.append(f"  rows: {row.get('n_rows', np.nan)}")
        lines.append(f"  mean: {safe_float(row.get('mean_prediction', np.nan)):.4f}")
        lines.append(f"  median: {safe_float(row.get('median_prediction', np.nan)):.4f}")
        lines.append(f"  min: {safe_float(row.get('min_prediction', np.nan)):.4f}")
        lines.append(f"  max: {safe_float(row.get('max_prediction', np.nan)):.4f}")
        lines.append("")

    if not teacher_metrics.empty:
        row = teacher_metrics.iloc[0]
        lines.append("Teacher overlap")
        lines.append(f"  valid pairs: {row.get('n_valid_pairs', np.nan)}")
        lines.append(f"  mae: {safe_float(row.get('mae', np.nan)):.4f}")
        lines.append(f"  rmse: {safe_float(row.get('rmse', np.nan)):.4f}")
        lines.append(f"  pearson: {safe_float(row.get('pearson', np.nan)):.4f}")
        lines.append("")

    if not spatial_vs_drug.empty and "feature_class" in spatial_vs_drug.columns:
        lines.append("Spatial versus drug identity explanation")
        for _, row in spatial_vs_drug.iterrows():
            lines.append(f"  {row.get('feature_class', '')}: fraction={safe_float(row.get('fraction_of_total_score', np.nan)):.3f}; n_features={row.get('n_features', np.nan)}")
        lines.append("")

    if not figure_manifest.empty:
        lines.append("Presentation figures")
        lines.append(f"  generated figures: {int(figure_manifest['exists'].sum()) if 'exists' in figure_manifest.columns else len(figure_manifest)}")
        for _, row in figure_manifest.head(16).iterrows():
            lines.append(f"  {row.get('figure_name', '')}: {row.get('title', '')}")
        lines.append("")

    lines.append("Notes")
    lines.append("  10 sample mode is an integration and presentation smoke test")
    lines.append("  Per treatment models may be disabled for 10 samples by design")
    lines.append("  Drug dummy dominance is expected in the global pooled model")
    lines.append("  Spatial-only explanation figures should be used for biological interpretation")

    write_text(path, lines)


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    """Run final QC step.
    Writes all reports and figures."""
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    model_root = infer_model_root(config_path, cfg)
    out_dir = qc_output_dir(cfg, model_root)

    ensure_dir(out_dir)

    print("Building spatial prediction QC outputs")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")

    tables = load_pipeline_tables(cfg, model_root)

    file_contract = build_file_contract_report(cfg, model_root)
    dataset_summary = build_dataset_summary(tables, cfg)
    split_summary = build_split_summary(tables, cfg)
    model_metrics = build_model_metrics(tables)
    prediction_dist = prediction_distribution(tables["all_predictions"])
    by_sample = build_prediction_by_sample(tables, cfg)
    by_treatment = build_prediction_by_treatment(tables, cfg)
    top_treatment = tables["top_treatment"].copy()

    teacher_comparison = tables["teacher_comparison"].copy()
    teacher_overall = teacher_overlap_metrics(teacher_comparison)
    teacher_by_treatment = teacher_overlap_metrics(teacher_comparison, [str(get_cfg(cfg, "drug_col", "drug"))])
    teacher_by_sample = teacher_overlap_metrics(teacher_comparison, [str(get_cfg(cfg, "sample_col", "sample_id"))])

    modality_col = first_existing_column(teacher_comparison, ["teacher_modality_used", "modality_used"])
    teacher_by_modality = teacher_overlap_metrics(teacher_comparison, [modality_col]) if modality_col else pd.DataFrame()

    feature_explanation = build_feature_explanation_summary(tables)
    spatial_vs_drug = build_spatial_vs_drug_summary(tables, feature_explanation)
    per_treatment_status = build_per_treatment_status(tables)

    qc_summary = build_qc_summary(
        file_report=file_contract,
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        metrics=model_metrics,
        prediction_dist=prediction_dist,
        teacher_metrics=teacher_overall,
        per_treatment_status=per_treatment_status,
    )

    summaries = {
        "file_contract": file_contract,
        "dataset": dataset_summary,
        "split": split_summary,
        "metrics": model_metrics,
        "prediction_dist": prediction_dist,
        "by_sample": by_sample,
        "by_treatment": by_treatment,
        "top_treatment": top_treatment,
        "teacher_comparison": teacher_comparison,
        "teacher_overall": teacher_overall,
        "feature_explanation": feature_explanation,
        "spatial_vs_drug": spatial_vs_drug,
        "per_treatment_status": per_treatment_status,
    }

    figure_manifest = write_all_figures(out_dir, tables, summaries, cfg)

    # Machine-readable reports
    write_table(qc_summary, out_dir / "qc_summary.tsv")
    write_table(file_contract, out_dir / "qc_file_contract_report.tsv")
    write_table(dataset_summary, out_dir / "qc_dataset_summary.tsv")
    write_table(split_summary, out_dir / "qc_split_summary.tsv")
    write_table(model_metrics, out_dir / "qc_model_metrics.tsv")
    write_table(teacher_overall, out_dir / "qc_teacher_overlap_metrics.tsv")
    write_table(teacher_by_treatment, out_dir / "qc_teacher_overlap_by_treatment.tsv")
    write_table(teacher_by_sample, out_dir / "qc_teacher_overlap_by_sample.tsv")
    write_table(teacher_by_modality, out_dir / "qc_teacher_overlap_by_modality.tsv")
    write_table(prediction_dist, out_dir / "qc_prediction_distribution.tsv")
    write_table(by_sample, out_dir / "qc_by_sample.tsv")
    write_table(by_treatment, out_dir / "qc_by_treatment.tsv")
    write_table(top_treatment, out_dir / "qc_top_treatment_per_sample.tsv")
    write_table(feature_explanation, out_dir / "qc_feature_explanation_summary.tsv")
    write_table(spatial_vs_drug, out_dir / "qc_spatial_vs_drug_summary.tsv")
    write_table(per_treatment_status, out_dir / "qc_per_treatment_model_status.tsv")
    write_table(figure_manifest, out_dir / "presentation_figure_manifest.tsv")

    write_qc_summary_text(
        path=out_dir / "qc_summary.txt",
        qc_summary=qc_summary,
        file_report=file_contract,
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        metrics=model_metrics,
        prediction_dist=prediction_dist,
        teacher_metrics=teacher_overall,
        spatial_vs_drug=spatial_vs_drug,
        figure_manifest=figure_manifest,
        cfg=cfg,
    )

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "n_required_files_missing": int(((file_contract["required"] == True) & (file_contract["is_file"] != True)).sum()),
        "n_figures_generated": int(figure_manifest["exists"].sum()) if "exists" in figure_manifest.columns else int(len(figure_manifest)),
        "input_tables": tables["_paths"].to_dict(orient="records") if "_paths" in tables else [],
        "outputs": {
            "qc_summary": str(out_dir / "qc_summary.tsv"),
            "qc_summary_text": str(out_dir / "qc_summary.txt"),
            "presentation_figure_manifest": str(out_dir / "presentation_figure_manifest.tsv"),
            "figures_dir": str(out_dir / "figures"),
        },
    }
    save_json(run_info, out_dir / "run_config.json")

    print("\nDONE")
    print(f"Required files missing: {run_info['n_required_files_missing']}")
    print(f"Figures generated: {run_info['n_figures_generated']}")
    print(f"Wrote: {out_dir / 'qc_summary.tsv'}")
    print(f"Wrote: {out_dir / 'qc_summary.txt'}")
    print(f"Wrote: {out_dir / 'presentation_figure_manifest.tsv'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
