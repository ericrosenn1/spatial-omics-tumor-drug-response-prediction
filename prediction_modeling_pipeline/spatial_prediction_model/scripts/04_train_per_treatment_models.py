"""
Script:
    04_train_per_treatment_models.py

Purpose:
    Train one spatial response model per treatment.

Role:
    Optional treatment specific step in spatial_prediction_model.
    Consumes the leakage safe modeling data from step 02.
    Uses the same sample level split created before global modeling.
    Trains independent models for treatments with enough labeled samples.
    Writes per treatment predictions, metrics, feature importance, and model files.

Pipeline position:
    01_validate_prediction_inputs.py
        validates teacher_builder handoff files.

    02_build_spatial_modeling_dataset.py
        builds the leakage safe model matrix and grouped sample split.

    03_train_global_spatial_response_model.py
        trains the pooled model across all sample treatment rows.

    04_train_per_treatment_models.py
        trains one model per treatment using only spatial features.
        Drug dummy features are removed because treatment is fixed within each model.

Design contract:
    YAML controls all paths, thresholds, task choice, and model parameters.
    The script can be left disabled for the 10 sample smoke test.
    The script becomes useful for the full 102 sample run.
    Treatment models are skipped rather than forced when sample size or target variation is weak.
    Sample splits are respected exactly to prevent same sample leakage.
    Every output is written even when no treatment is eligible.

Expected step 02 inputs:
    outputs/<run_name>/02_modeling_dataset/modeling_table.tsv
        Metadata and target table.
        One row per labeled sample treatment pair.

    outputs/<run_name>/02_modeling_dataset/X_features.csv
        Feature matrix aligned row for row with modeling_table.tsv.

    outputs/<run_name>/02_modeling_dataset/y_target.csv
        Target vector aligned row for row with modeling_table.tsv.

    outputs/<run_name>/02_modeling_dataset/model_feature_manifest.csv
        Feature list from step 02, preferred source for allowed feature columns.

    outputs/<run_name>/02_modeling_dataset/sample_split.tsv
        Sample level train and test split.

Primary outputs:
    outputs/<run_name>/04_per_treatment_models/per_treatment_model_summary.tsv
    outputs/<run_name>/04_per_treatment_models/skipped_treatments.tsv
    outputs/<run_name>/04_per_treatment_models/per_treatment_predictions_all.tsv
    outputs/<run_name>/04_per_treatment_models/per_treatment_feature_importance_top.tsv
    outputs/<run_name>/04_per_treatment_models/per_treatment_model_summary.txt
    outputs/<run_name>/04_per_treatment_models/run_config.json

Model file outputs:
    outputs/<run_name>/04_per_treatment_models/models/<drug_key>.joblib

Prediction outputs:
    outputs/<run_name>/04_per_treatment_models/predictions/<drug_key>_predictions.tsv

Feature importance outputs:
    outputs/<run_name>/04_per_treatment_models/feature_importance/<drug_key>_feature_importance.tsv

Notes:
    For the current 10 sample test run, run_per_treatment_models should usually be false.
    For the full 102 sample run, enable this step in YAML and keep thresholds conservative.
    This script uses spatial features only by default, because drug identity is constant within each treatment subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


SCRIPT_NAME = "04_train_per_treatment_models.py"
STEP_NAME = "04_train_per_treatment_models"
DEFAULT_MODELING_DATASET_SUBDIR = "02_modeling_dataset"
DEFAULT_OUTPUT_SUBDIR = "04_per_treatment_models"


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI arguments
    config path plus optional force flag"""
    parser = argparse.ArgumentParser(description="Train per treatment spatial response models")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when run_per_treatment_models is false",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config
    UTF8 with BOM tolerant"""
    if not path.exists():
        raise FileNotFoundError(path)

    # utf 8 sig handles occasional Windows BOM
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """read config value
    fallback when absent"""
    return cfg[key] if key in cfg else default


def bool_like(value: Any) -> bool:
    """parse truth like values
    tolerant string handling"""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on", "included"}


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve absolute or project relative path
    None stays None"""
    if value in [None, ""]:
        return None

    path = Path(str(value))

    if path.is_absolute():
        return path

    return project_dir / path


def get_output_subdir(cfg: dict[str, Any], key: str, default: str) -> str:
    """resolve named output subdir
    supports output_subdirs mapping"""
    subdirs = get_cfg(cfg, "output_subdirs", {}) or {}

    if isinstance(subdirs, dict) and key in subdirs:
        return str(subdirs[key])

    return default


def get_step02_dir(cfg: dict[str, Any]) -> Path:
    """resolve modeling dataset directory
    output_root plus step 02 subdir"""
    output_root = Path(str(cfg["output_root"]))
    subdir = get_output_subdir(cfg, "modeling_dataset", DEFAULT_MODELING_DATASET_SUBDIR)

    return output_root / subdir


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """resolve step output directory
    output_root plus step 04 subdir"""
    output_root = Path(str(cfg["output_root"]))
    subdir = get_output_subdir(cfg, "per_treatment_models", DEFAULT_OUTPUT_SUBDIR)

    return output_root / subdir


def ensure_dir(path: Path) -> None:
    """create directory
    parents included"""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write text report
    newline joined"""
    path.write_text("\n".join(lines), encoding="utf-8")


def save_json(data: dict[str, Any], path: Path) -> None:
    """write JSON file
    readable indent"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


# ============================================================
# BASIC TEXT AND NUMERIC HELPERS
# ============================================================


def clean_text(value: Any) -> str:
    """clean scalar value
    empty string for missing"""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_key(value: Any) -> str:
    """normalize treatment key
    lowercase compact spaces"""
    text = clean_text(value).lower()
    return " ".join(text.split())


def safe_filename(value: Any, max_len: int = 160) -> str:
    """safe file name token
    keeps readable treatment names"""
    text = normalize_key(value)

    # replace Windows unsafe characters
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)

    # collapse whitespace and punctuation runs
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        text = "unknown_treatment"

    return text[:max_len]


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """coerce series to numeric
    invalid values become missing"""
    return pd.to_numeric(series, errors="coerce")


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """coerce frame to numeric
    invalid cells become missing"""
    return df.apply(pd.to_numeric, errors="coerce")


def fraction(numerator: int | float, denominator: int | float) -> float:
    """safe fraction
    zero denominator gives NaN"""
    if denominator == 0:
        return float("nan")

    return float(numerator) / float(denominator)


def compact_list(values: list[Any], max_items: int = 12) -> str:
    """compact list text
    useful for reports"""
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    shown = cleaned[:max_items]

    if len(cleaned) > max_items:
        shown.append(f"... {len(cleaned) - max_items} more")

    return "; ".join(shown)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """find first column present
    preserves candidate order"""
    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# INPUT LOADING
# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """load CSV or TSV table
    suffix based parsing"""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    # fallback for unusual extension
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def expected_step02_paths(step02_dir: Path) -> dict[str, Path]:
    """expected step 02 files
    fixed handoff names"""
    return {
        "modeling_table": step02_dir / "modeling_table.tsv",
        "x_features": step02_dir / "X_features.csv",
        "y_target": step02_dir / "y_target.csv",
        "model_feature_manifest": step02_dir / "model_feature_manifest.csv",
        "sample_split": step02_dir / "sample_split.tsv",
    }


def load_step02_inputs(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """load step 02 handoff files
    returns tables and file report"""
    step02_dir = get_step02_dir(cfg)
    path_map = expected_step02_paths(step02_dir)

    rows: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}

    for key, path in path_map.items():
        exists = bool(path.exists())
        is_file = bool(exists and path.is_file())
        size_bytes = int(path.stat().st_size) if is_file else np.nan

        rows.append(
            {
                "input_name": key,
                "path": str(path),
                "exists": exists,
                "is_file": is_file,
                "size_bytes": size_bytes,
            }
        )

        if not is_file:
            continue

        tables[key] = load_table(path)

    report = pd.DataFrame(rows)

    missing = report.loc[report["is_file"] != True, "input_name"].tolist()

    if missing:
        raise FileNotFoundError(
            "Missing required step 02 files: " + compact_list(missing)
        )

    return tables, report


def resolve_target_vector(
    modeling_table: pd.DataFrame,
    y_target: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """resolve target vector
    modeling table preferred"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    if target_col in modeling_table.columns:
        return safe_numeric_series(modeling_table[target_col])

    # y_target may be one column or may preserve the target name
    if target_col in y_target.columns:
        return safe_numeric_series(y_target[target_col])

    if "target" in y_target.columns:
        return safe_numeric_series(y_target["target"])

    if "y" in y_target.columns:
        return safe_numeric_series(y_target["y"])

    if y_target.shape[1] == 1:
        return safe_numeric_series(y_target.iloc[:, 0])

    raise ValueError(f"Could not resolve target vector: {target_col}")


def resolve_manifest_feature_col(manifest: pd.DataFrame) -> str | None:
    """find feature name column
    known aliases only"""
    for col in ["feature_name", "feature", "column", "column_name"]:
        if col in manifest.columns:
            return col

    return None


def get_manifest_features(manifest: pd.DataFrame) -> list[str]:
    """extract included manifest features
    empty if unresolved"""
    feature_col = resolve_manifest_feature_col(manifest)

    if feature_col is None:
        return []

    out = manifest.copy()

    # included column is optional
    if "included" in out.columns:
        out = out[out["included"].map(bool_like)].copy()

    features = out[feature_col].dropna().astype(str).map(clean_text).tolist()
    return [feature for feature in features if feature]


def identify_drug_dummy_columns(columns: list[str], cfg: dict[str, Any]) -> list[str]:
    """find drug identity columns
    removed for per treatment models"""
    prefixes = get_cfg(
        cfg,
        "drug_dummy_prefixes",
        [
            "drug_dummy__",
            "drug_key__",
            "drug__",
            "treatment__",
            "treatment_key__",
        ],
    )

    out: list[str] = []

    for col in columns:
        low = col.lower()

        # explicit configured prefixes
        if any(low.startswith(str(prefix).lower()) for prefix in prefixes):
            out.append(col)
            continue

        # common one hot style names
        if low.startswith("drug_key_") or low.startswith("drug_"):
            out.append(col)
            continue

    return out


def get_non_feature_columns(cfg: dict[str, Any]) -> set[str]:
    """known non feature columns
    identifiers target and teacher fields"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    slide_col = str(get_cfg(cfg, "slide_col", "slide_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    leakage = set(str(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or [])
    metadata = set(str(x) for x in get_cfg(cfg, "metadata_passthrough_columns", []) or [])

    base = {
        sample_col,
        slide_col,
        drug_col,
        drug_key_col,
        target_col,
        "split",
        "row_id",
        "index",
    }

    return base | leakage | metadata


def select_feature_columns(
    x_features: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """select spatial feature columns
    manifest preferred then numeric fallback"""
    non_features = get_non_feature_columns(cfg)
    drug_dummy_cols = set(identify_drug_dummy_columns(list(x_features.columns), cfg))

    manifest_features = get_manifest_features(manifest)
    manifest_present = [
        col for col in manifest_features
        if col in x_features.columns and col not in non_features and col not in drug_dummy_cols
    ]

    candidate_source = "manifest"

    if manifest_present:
        feature_cols = manifest_present
    else:
        # fallback only when manifest cannot guide selection
        candidate_source = "numeric_fallback"
        feature_cols = [
            col for col in x_features.columns
            if col not in non_features and col not in drug_dummy_cols
        ]

    x_numeric = safe_numeric_frame(x_features[feature_cols])
    numeric_cols = [col for col in x_numeric.columns if x_numeric[col].notna().any()]

    dropped_non_numeric = sorted(set(feature_cols) - set(numeric_cols))
    dropped_drug_dummy = sorted(drug_dummy_cols)
    dropped_non_feature = sorted([col for col in x_features.columns if col in non_features])

    report_rows = []

    for col in feature_cols:
        report_rows.append(
            {
                "column": col,
                "selected_source": candidate_source,
                "selected": col in numeric_cols,
                "drop_reason": "" if col in numeric_cols else "not_numeric_or_all_missing",
            }
        )

    for col in dropped_drug_dummy:
        report_rows.append(
            {
                "column": col,
                "selected_source": "drug_dummy_filter",
                "selected": False,
                "drop_reason": "drug_identity_constant_within_treatment",
            }
        )

    for col in dropped_non_feature:
        report_rows.append(
            {
                "column": col,
                "selected_source": "non_feature_filter",
                "selected": False,
                "drop_reason": "identifier_target_metadata_or_leakage",
            }
        )

    feature_report = pd.DataFrame(report_rows)

    if not numeric_cols:
        raise ValueError("No usable numeric spatial feature columns found")

    return numeric_cols, feature_report


def build_modeling_frame(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """combine step 02 tables
    aligned modeling frame plus feature list"""
    modeling = tables["modeling_table"].copy()
    x_features = tables["x_features"].copy()
    y_target = tables["y_target"].copy()
    manifest = tables["model_feature_manifest"].copy()
    sample_split = tables["sample_split"].copy()

    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    required = [sample_col, drug_col, drug_key_col]

    for col in required:
        if col not in modeling.columns:
            raise ValueError(f"modeling_table missing required column: {col}")

    if len(modeling) != len(x_features):
        raise ValueError(
            f"Row count mismatch between modeling_table and X_features: {len(modeling)} versus {len(x_features)}"
        )

    y = resolve_target_vector(modeling, y_target, cfg)

    if len(y) != len(modeling):
        raise ValueError(
            f"Target length mismatch: {len(y)} versus modeling_table rows {len(modeling)}"
        )

    feature_cols, feature_report = select_feature_columns(x_features, manifest, cfg)

    # combine metadata and selected feature matrix by row order
    frame = modeling[[sample_col, drug_col, drug_key_col]].copy()
    frame[target_col] = y.to_numpy()

    # optional metadata copied when present
    for col in ["slide_id", "cancer_type", "dataset_id"]:
        if col in modeling.columns and col not in frame.columns:
            frame[col] = modeling[col].to_numpy()

    # split comes from sample_split, not row random split
    if "split" not in sample_split.columns:
        raise ValueError("sample_split.tsv missing split column")

    if sample_col not in sample_split.columns:
        raise ValueError(f"sample_split.tsv missing sample column: {sample_col}")

    split_map = sample_split[[sample_col, "split"]].drop_duplicates(sample_col)
    frame = frame.merge(split_map, on=sample_col, how="left")

    if frame["split"].isna().any():
        missing = frame.loc[frame["split"].isna(), sample_col].drop_duplicates().tolist()
        raise ValueError("Samples missing split assignment: " + compact_list(missing))

    # numeric features copied after metadata to preserve clean layout
    x_numeric = safe_numeric_frame(x_features[feature_cols])
    for col in feature_cols:
        frame[col] = x_numeric[col].to_numpy()

    # normalized treatment key for grouping stability
    frame["_drug_key_norm"] = frame[drug_key_col].map(normalize_key)

    return frame, feature_cols, feature_report


# ============================================================
# MODEL BUILDING
# ============================================================


def get_task(cfg: dict[str, Any]) -> str:
    """resolve task setting
    regression or classification"""
    task = clean_text(get_cfg(cfg, "task", "regression")).lower()

    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task: {task}")

    return task


def get_model_type(cfg: dict[str, Any]) -> str:
    """resolve model type
    per treatment override supported"""
    model_type = clean_text(get_cfg(cfg, "per_treatment_model_type", ""))

    if not model_type:
        model_type = clean_text(get_cfg(cfg, "model_type", "random_forest"))

    return model_type.lower()


def get_model_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """read model parameter block
    per treatment override supported"""
    params = get_cfg(cfg, "per_treatment_model_params", None)

    if isinstance(params, dict):
        return dict(params)

    params = get_cfg(cfg, "model_params", None)

    if isinstance(params, dict):
        return dict(params)

    return {}


def build_estimator(cfg: dict[str, Any], task: str) -> Any:
    """build sklearn estimator
    model type controlled by YAML"""
    model_type = get_model_type(cfg)
    params = get_model_params(cfg)
    random_state = int(get_cfg(cfg, "random_state", 42))

    # defaults chosen for small sample robustness
    if model_type in {"random_forest", "rf"}:
        defaults = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(params)

        if task == "classification":
            defaults.setdefault("class_weight", "balanced")
            return RandomForestClassifier(**defaults)

        return RandomForestRegressor(**defaults)

    if model_type in {"extra_trees", "extratrees"}:
        defaults = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(params)

        if task == "classification":
            defaults.setdefault("class_weight", "balanced")
            return ExtraTreesClassifier(**defaults)

        return ExtraTreesRegressor(**defaults)

    if model_type in {"gradient_boosting", "gbrt"}:
        defaults = {
            "random_state": random_state,
        }
        defaults.update(params)

        if task == "classification":
            return GradientBoostingClassifier(**defaults)

        return GradientBoostingRegressor(**defaults)

    if model_type in {"ridge", "ridge_regression"}:
        defaults = {
            "alpha": 1.0,
        }
        defaults.update(params)
        return Ridge(**defaults)

    if model_type in {"logistic", "logistic_regression"}:
        defaults = {
            "max_iter": 3000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
        defaults.update(params)
        return LogisticRegression(**defaults)

    if model_type in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except Exception as exc:
            raise ImportError(
                "xgboost requested but not available. Use random_forest or install xgboost."
            ) from exc

        defaults = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(params)

        if task == "classification":
            defaults.setdefault("eval_metric", "logloss")
            return XGBClassifier(**defaults)

        return XGBRegressor(**defaults)

    raise ValueError(f"Unsupported model_type: {model_type}")


def build_pipeline(cfg: dict[str, Any], task: str) -> Pipeline:
    """build impute plus model pipeline
    median imputation for spatial features"""
    estimator = build_estimator(cfg, task)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


# ============================================================
# SPLIT AND ELIGIBILITY
# ============================================================


def split_mask(frame: pd.DataFrame, split_name: str) -> pd.Series:
    """boolean mask for split
    case insensitive"""
    return frame["split"].astype(str).str.lower().eq(split_name.lower())


def treatment_summary_counts(
    treatment_df: pd.DataFrame,
    cfg: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """count treatment data
    total train test target variation"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))

    y = safe_numeric_series(treatment_df[target_col])
    train_df = treatment_df[split_mask(treatment_df, "train")]
    test_df = treatment_df[split_mask(treatment_df, "test")]

    out: dict[str, Any] = {
        "n_total": int(len(treatment_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_target_nonmissing": int(y.notna().sum()),
        "target_mean": float(y.mean()) if y.notna().any() else np.nan,
        "target_std": float(y.std()) if y.notna().sum() > 1 else 0.0,
        "target_min": float(y.min()) if y.notna().any() else np.nan,
        "target_max": float(y.max()) if y.notna().any() else np.nan,
        "target_unique": int(y.nunique(dropna=True)),
    }

    if task == "classification":
        y_binary = (y >= threshold).astype(float)
        out["n_binary_classes_total"] = int(y_binary.dropna().nunique())
        out["n_binary_classes_train"] = int((safe_numeric_series(train_df[target_col]) >= threshold).astype(float).nunique())
    else:
        out["n_binary_classes_total"] = np.nan
        out["n_binary_classes_train"] = np.nan

    return out


def treatment_skip_reason(
    treatment_df: pd.DataFrame,
    cfg: dict[str, Any],
    task: str,
) -> tuple[bool, str, dict[str, Any]]:
    """check treatment eligibility
    returns should_train and reason"""
    counts = treatment_summary_counts(treatment_df, cfg, task)

    min_total = int(get_cfg(cfg, "min_samples_per_treatment", 30))
    min_train = int(get_cfg(cfg, "min_train_samples_per_treatment", max(5, math.ceil(min_total * 0.5))))
    min_test = int(get_cfg(cfg, "min_test_samples_per_treatment", 2))
    min_std = float(get_cfg(cfg, "min_target_std", 0.02))
    min_unique = int(get_cfg(cfg, "min_target_unique", 2))

    reasons: list[str] = []

    if counts["n_total"] < min_total:
        reasons.append(f"n_total_lt_{min_total}")

    if counts["n_train"] < min_train:
        reasons.append(f"n_train_lt_{min_train}")

    if counts["n_test"] < min_test:
        reasons.append(f"n_test_lt_{min_test}")

    if counts["n_target_nonmissing"] < min_total:
        reasons.append("target_missing")

    if counts["target_unique"] < min_unique:
        reasons.append(f"target_unique_lt_{min_unique}")

    if float(counts["target_std"]) < min_std:
        reasons.append(f"target_std_lt_{min_std}")

    if task == "classification" and int(counts.get("n_binary_classes_train", 0)) < 2:
        reasons.append("train_binary_class_count_lt_2")

    should_train = len(reasons) == 0
    reason = "eligible" if should_train else ";".join(reasons)

    return should_train, reason, counts


def build_split_summary(frame: pd.DataFrame, cfg: dict[str, Any], task: str) -> pd.DataFrame:
    """summarize split counts by treatment
    useful before fitting"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    rows: list[dict[str, Any]] = []

    for drug_key_norm, group in frame.groupby("_drug_key_norm", sort=True):
        should_train, reason, counts = treatment_skip_reason(group, cfg, task)

        rows.append(
            {
                "drug_key_norm": drug_key_norm,
                "drug_key": group[drug_key_col].iloc[0],
                "drug": group[drug_col].iloc[0],
                "eligible": should_train,
                "reason": reason,
                **counts,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation
    NaN when undefined"""
    if len(y_true) < 2:
        return float("nan")

    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """rank correlation
    pandas based"""
    if len(y_true) < 2:
        return float("nan")

    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")

    if a.std() == 0 or b.std() == 0:
        return float("nan")

    return float(a.corr(b))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, Any]:
    """regression metrics
    robust to small test sets"""
    out: dict[str, Any] = {}

    if len(y_true) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_pearson": np.nan,
            f"{prefix}_spearman": np.nan,
        }

    mse = mean_squared_error(y_true, y_pred)

    out[f"{prefix}_n"] = int(len(y_true))
    out[f"{prefix}_mae"] = float(mean_absolute_error(y_true, y_pred))
    out[f"{prefix}_rmse"] = float(np.sqrt(mse))
    out[f"{prefix}_r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    out[f"{prefix}_pearson"] = pearson_corr(y_true, y_pred)
    out[f"{prefix}_spearman"] = spearman_corr(y_true, y_pred)

    return out


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, prefix: str, threshold: float) -> dict[str, Any]:
    """classification metrics
    probability first"""
    out: dict[str, Any] = {}

    if len(y_true) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_balanced_accuracy": np.nan,
            f"{prefix}_precision": np.nan,
            f"{prefix}_recall": np.nan,
            f"{prefix}_f1": np.nan,
            f"{prefix}_roc_auc": np.nan,
            f"{prefix}_average_precision": np.nan,
        }

    y_pred = (y_prob >= threshold).astype(int)
    classes = np.unique(y_true)

    out[f"{prefix}_n"] = int(len(y_true))
    out[f"{prefix}_accuracy"] = float(accuracy_score(y_true, y_pred))
    out[f"{prefix}_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out[f"{prefix}_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out[f"{prefix}_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out[f"{prefix}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if len(classes) >= 2:
        out[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out[f"{prefix}_average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        out[f"{prefix}_roc_auc"] = np.nan
        out[f"{prefix}_average_precision"] = np.nan

    return out


def predict_values(model: Pipeline, x: pd.DataFrame, task: str) -> np.ndarray:
    """get prediction values
    probabilities for classification"""
    if task == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]

        # fallback for unusual classifier
        return model.predict(x).astype(float)

    return model.predict(x).astype(float)


def compute_metrics(
    model: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """compute train and test metrics
    regression or classification"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))

    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]

    y_train_raw = safe_numeric_series(train_df[target_col]).to_numpy()
    y_test_raw = safe_numeric_series(test_df[target_col]).to_numpy()

    train_pred = predict_values(model, x_train, task)
    test_pred = predict_values(model, x_test, task)

    if task == "classification":
        y_train = (y_train_raw >= threshold).astype(int)
        y_test = (y_test_raw >= threshold).astype(int)

        out = {}
        out.update(classification_metrics(y_train, train_pred, "train", threshold))
        out.update(classification_metrics(y_test, test_pred, "test", threshold))
        return out

    out = {}
    out.update(regression_metrics(y_train_raw, train_pred, "train"))
    out.update(regression_metrics(y_test_raw, test_pred, "test"))
    return out


# ============================================================
# PREDICTIONS AND FEATURE IMPORTANCE
# ============================================================


def build_prediction_table(
    model: Pipeline,
    treatment_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    task: str,
) -> pd.DataFrame:
    """build predictions for one treatment
    all rows with train test label"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))

    y_true = safe_numeric_series(treatment_df[target_col])
    y_pred = predict_values(model, treatment_df[feature_cols], task)

    out_cols = [sample_col, drug_col, drug_key_col, "split"]

    for optional in ["slide_id", "dataset_id", "cancer_type"]:
        if optional in treatment_df.columns:
            out_cols.append(optional)

    pred = treatment_df[out_cols].copy()
    pred["y_true"] = y_true.to_numpy()
    pred["y_pred"] = y_pred

    if task == "classification":
        pred["y_true_binary"] = (pred["y_true"] >= threshold).astype(int)
        pred["y_pred_binary"] = (pred["y_pred"] >= threshold).astype(int)
        pred["y_pred_probability"] = pred["y_pred"]
    else:
        pred["residual"] = pred["y_true"] - pred["y_pred"]
        pred["absolute_error"] = np.abs(pred["residual"])

    return pred


def get_inner_model(model: Pipeline) -> Any:
    """extract estimator from pipeline
    model step by convention"""
    return model.named_steps.get("model", model)


def build_feature_importance(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """extract model feature importance
    supports trees and coefficients"""
    estimator = get_inner_model(model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "feature_importances"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)

        if coef.ndim > 1:
            coef = coef[0]

        values = np.abs(coef.astype(float))
        importance_type = "absolute_coefficient"
    else:
        values = np.repeat(np.nan, len(feature_cols))
        importance_type = "unavailable"

    out = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "importance": values,
            "importance_type": importance_type,
        }
    )

    out["rank"] = out["importance"].rank(method="first", ascending=False)
    out = out.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    return out


def top_feature_rows(
    importance: pd.DataFrame,
    drug_key: str,
    drug: str,
    top_n: int,
) -> pd.DataFrame:
    """tag top feature rows
    combined summary output"""
    out = importance.head(top_n).copy()
    out.insert(0, "drug_key", drug_key)
    out.insert(0, "drug", drug)

    return out


# ============================================================
# TRAINING LOOP
# ============================================================


def train_one_treatment(
    treatment_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    task: str,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """train model for one treatment
    returns model metrics predictions importance"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))

    train_df = treatment_df[split_mask(treatment_df, "train")].copy()
    test_df = treatment_df[split_mask(treatment_df, "test")].copy()

    x_train = train_df[feature_cols]
    y_train_raw = safe_numeric_series(train_df[target_col])

    if task == "classification":
        y_train = (y_train_raw >= threshold).astype(int)
    else:
        y_train = y_train_raw.astype(float)

    model = build_pipeline(cfg, task)

    # fit only train rows, split already sample grouped
    model.fit(x_train, y_train)

    metrics = compute_metrics(
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        cfg=cfg,
        task=task,
    )

    predictions = build_prediction_table(
        model=model,
        treatment_df=treatment_df,
        feature_cols=feature_cols,
        cfg=cfg,
        task=task,
    )

    importance = build_feature_importance(model, feature_cols)

    return model, metrics, predictions, importance


def model_package(
    model: Pipeline,
    cfg: dict[str, Any],
    feature_cols: list[str],
    treatment_meta: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """package model for joblib
    estimator plus metadata"""
    return {
        "model": model,
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "task": task,
        "model_type": get_model_type(cfg),
        "feature_columns": feature_cols,
        "treatment": treatment_meta,
        "target_col": get_cfg(cfg, "target_col", "fused_prob_responder"),
        "sample_col": get_cfg(cfg, "sample_col", "sample_id"),
        "drug_col": get_cfg(cfg, "drug_col", "drug"),
        "drug_key_col": get_cfg(cfg, "drug_key_col", "drug_key"),
        "random_state": get_cfg(cfg, "random_state", 42),
    }


def write_disabled_outputs(
    out_dir: Path,
    cfg: dict[str, Any],
    input_report: pd.DataFrame | None = None,
) -> None:
    """write empty outputs when disabled
    preserves pipeline contract"""
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "feature_importance")

    summary_cols = [
        "drug",
        "drug_key",
        "status",
        "reason",
        "n_total",
        "n_train",
        "n_test",
    ]

    pd.DataFrame(columns=summary_cols).to_csv(out_dir / "per_treatment_model_summary.tsv", sep="\t", index=False)
    pd.DataFrame(columns=summary_cols).to_csv(out_dir / "skipped_treatments.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "per_treatment_predictions_all.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "per_treatment_feature_importance_top.tsv", sep="\t", index=False)

    if input_report is not None:
        input_report.to_csv(out_dir / "input_file_report.tsv", sep="\t", index=False)

    lines = [
        "Per treatment model summary",
        "",
        "Status",
        "  disabled by config",
        "",
        "Reason",
        "  run_per_treatment_models is false",
        "",
        "Output contract",
        "  empty summary files written",
    ]
    write_text(out_dir / "per_treatment_model_summary.txt", lines)

    save_json(
        {
            "script_name": SCRIPT_NAME,
            "step_name": STEP_NAME,
            "run_per_treatment_models": False,
            "output_dir": str(out_dir),
        },
        out_dir / "run_config.json",
    )


def train_all_treatments(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train eligible treatment models
    writes per treatment artifacts"""
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "feature_importance")

    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    top_n = int(get_cfg(cfg, "top_n_features_per_treatment", 25))

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_top_importance: list[pd.DataFrame] = []

    for drug_key_norm, treatment_df in frame.groupby("_drug_key_norm", sort=True):
        drug = clean_text(treatment_df[drug_col].iloc[0])
        drug_key = clean_text(treatment_df[drug_key_col].iloc[0])
        safe_key = safe_filename(drug_key)

        should_train, reason, counts = treatment_skip_reason(treatment_df, cfg, task)

        base_row: dict[str, Any] = {
            "drug": drug,
            "drug_key": drug_key,
            "drug_key_norm": drug_key_norm,
            "status": "trained" if should_train else "skipped",
            "reason": reason,
            **counts,
        }

        if not should_train:
            summary_rows.append(base_row)
            continue

        try:
            model, metrics, predictions, importance = train_one_treatment(
                treatment_df=treatment_df.copy(),
                feature_cols=feature_cols,
                cfg=cfg,
                task=task,
            )

            model_path = out_dir / "models" / f"{safe_key}.joblib"
            pred_path = out_dir / "predictions" / f"{safe_key}_predictions.tsv"
            importance_path = out_dir / "feature_importance" / f"{safe_key}_feature_importance.tsv"

            treatment_meta = {
                "drug": drug,
                "drug_key": drug_key,
                "drug_key_norm": drug_key_norm,
                **counts,
            }

            joblib.dump(
                model_package(
                    model=model,
                    cfg=cfg,
                    feature_cols=feature_cols,
                    treatment_meta=treatment_meta,
                    task=task,
                ),
                model_path,
            )

            predictions.to_csv(pred_path, sep="\t", index=False)
            importance.to_csv(importance_path, sep="\t", index=False)

            top_imp = top_feature_rows(importance, drug_key=drug_key, drug=drug, top_n=top_n)
            all_predictions.append(predictions)
            all_top_importance.append(top_imp)

            base_row.update(metrics)
            base_row["model_path"] = str(model_path)
            base_row["prediction_path"] = str(pred_path)
            base_row["feature_importance_path"] = str(importance_path)
            summary_rows.append(base_row)

        except Exception as exc:
            # keep other treatments running after one failure
            failed = dict(base_row)
            failed["status"] = "failed"
            failed["reason"] = f"{type(exc).__name__}: {exc}"
            summary_rows.append(failed)

    summary = pd.DataFrame(summary_rows)

    if all_predictions:
        predictions_all = pd.concat(all_predictions, ignore_index=True)
    else:
        predictions_all = pd.DataFrame()

    if all_top_importance:
        top_importance = pd.concat(all_top_importance, ignore_index=True)
    else:
        top_importance = pd.DataFrame()

    return summary, predictions_all, top_importance


# ============================================================
# REPORTING
# ============================================================


def write_summary_text(
    path: Path,
    cfg: dict[str, Any],
    frame: pd.DataFrame,
    feature_cols: list[str],
    split_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    task: str,
) -> None:
    """write human summary
    compact modeling report"""
    n_treatments = int(frame["_drug_key_norm"].nunique()) if "_drug_key_norm" in frame.columns else 0
    n_trained = int((model_summary["status"] == "trained").sum()) if "status" in model_summary.columns else 0
    n_skipped = int((model_summary["status"] == "skipped").sum()) if "status" in model_summary.columns else 0
    n_failed = int((model_summary["status"] == "failed").sum()) if "status" in model_summary.columns else 0

    lines: list[str] = []
    lines.append("Per treatment spatial response model summary")
    lines.append("")
    lines.append("Run settings")
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  task: {task}")
    lines.append(f"  model_type: {get_model_type(cfg)}")
    lines.append(f"  output_root: {get_cfg(cfg, 'output_root', '')}")
    lines.append("")
    lines.append("Input shape")
    lines.append(f"  modeling rows: {len(frame)}")
    lines.append(f"  treatments: {n_treatments}")
    lines.append(f"  features used: {len(feature_cols)}")
    lines.append("")
    lines.append("Training status")
    lines.append(f"  trained: {n_trained}")
    lines.append(f"  skipped: {n_skipped}")
    lines.append(f"  failed: {n_failed}")
    lines.append("")

    if not split_summary.empty:
        lines.append("Eligibility thresholds")
        lines.append(f"  min_samples_per_treatment: {get_cfg(cfg, 'min_samples_per_treatment', 30)}")
        lines.append(f"  min_train_samples_per_treatment: {get_cfg(cfg, 'min_train_samples_per_treatment', '')}")
        lines.append(f"  min_test_samples_per_treatment: {get_cfg(cfg, 'min_test_samples_per_treatment', 2)}")
        lines.append(f"  min_target_std: {get_cfg(cfg, 'min_target_std', 0.02)}")
        lines.append(f"  min_target_unique: {get_cfg(cfg, 'min_target_unique', 2)}")
        lines.append("")

    if n_skipped > 0 and "reason" in model_summary.columns:
        lines.append("Top skip reasons")
        reason_counts = model_summary.loc[model_summary["status"] == "skipped", "reason"].value_counts().head(12)

        for reason, count in reason_counts.items():
            lines.append(f"  {reason}: {count}")

        lines.append("")

    if n_trained > 0:
        lines.append("Metric note")
        lines.append("  train and test metrics are written per treatment")
        lines.append("  small treatment test sets should be interpreted cautiously")
    else:
        lines.append("Metric note")
        lines.append("  no eligible treatments trained")

    write_text(path, lines)


def write_outputs(
    out_dir: Path,
    cfg: dict[str, Any],
    input_report: pd.DataFrame,
    feature_report: pd.DataFrame,
    split_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    predictions_all: pd.DataFrame,
    top_importance: pd.DataFrame,
    frame: pd.DataFrame,
    feature_cols: list[str],
    task: str,
) -> None:
    """write all step 04 reports
    stable output names"""
    input_report.to_csv(out_dir / "input_file_report.tsv", sep="\t", index=False)
    feature_report.to_csv(out_dir / "input_feature_report.tsv", sep="\t", index=False)
    split_summary.to_csv(out_dir / "split_summary_by_treatment.tsv", sep="\t", index=False)
    model_summary.to_csv(out_dir / "per_treatment_model_summary.tsv", sep="\t", index=False)

    skipped = model_summary[model_summary["status"].isin(["skipped", "failed"])].copy()
    skipped.to_csv(out_dir / "skipped_treatments.tsv", sep="\t", index=False)

    predictions_all.to_csv(out_dir / "per_treatment_predictions_all.tsv", sep="\t", index=False)
    top_importance.to_csv(out_dir / "per_treatment_feature_importance_top.tsv", sep="\t", index=False)

    write_summary_text(
        path=out_dir / "per_treatment_model_summary.txt",
        cfg=cfg,
        frame=frame,
        feature_cols=feature_cols,
        split_summary=split_summary,
        model_summary=model_summary,
        task=task,
    )

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "task": task,
        "model_type": get_model_type(cfg),
        "output_dir": str(out_dir),
        "n_rows": int(len(frame)),
        "n_features": int(len(feature_cols)),
        "n_treatments": int(frame["_drug_key_norm"].nunique()),
        "n_trained": int((model_summary["status"] == "trained").sum()) if "status" in model_summary.columns else 0,
        "n_skipped": int((model_summary["status"] == "skipped").sum()) if "status" in model_summary.columns else 0,
        "n_failed": int((model_summary["status"] == "failed").sum()) if "status" in model_summary.columns else 0,
        "feature_columns": feature_cols,
    }
    save_json(run_info, out_dir / "run_config.json")


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    """run per treatment modeling
    optional YAML controlled step"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    out_dir = get_output_dir(cfg)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "feature_importance")

    run_enabled = bool_like(get_cfg(cfg, "run_per_treatment_models", False)) or args.force

    print("Training per treatment spatial response models")
    print(f"Config: {Path(args.config)}")
    print(f"Output: {out_dir}")
    print(f"Enabled: {run_enabled}")

    if not run_enabled:
        write_disabled_outputs(out_dir, cfg)
        print("\nDONE")
        print("Per treatment models disabled by config")
        print(f"Wrote empty contract outputs: {out_dir}")
        return 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        tables, input_report = load_step02_inputs(cfg)
        task = get_task(cfg)

        frame, feature_cols, feature_report = build_modeling_frame(tables, cfg)

        split_summary = build_split_summary(frame, cfg, task)

        model_summary, predictions_all, top_importance = train_all_treatments(
            frame=frame,
            feature_cols=feature_cols,
            cfg=cfg,
            out_dir=out_dir,
            task=task,
        )

        write_outputs(
            out_dir=out_dir,
            cfg=cfg,
            input_report=input_report,
            feature_report=feature_report,
            split_summary=split_summary,
            model_summary=model_summary,
            predictions_all=predictions_all,
            top_importance=top_importance,
            frame=frame,
            feature_cols=feature_cols,
            task=task,
        )

    n_trained = int((model_summary["status"] == "trained").sum()) if "status" in model_summary.columns else 0
    n_skipped = int((model_summary["status"] == "skipped").sum()) if "status" in model_summary.columns else 0
    n_failed = int((model_summary["status"] == "failed").sum()) if "status" in model_summary.columns else 0

    print("\nDONE")
    print(f"Rows: {len(frame):,}")
    print(f"Treatments: {frame['_drug_key_norm'].nunique():,}")
    print(f"Features: {len(feature_cols):,}")
    print(f"Trained: {n_trained:,}")
    print(f"Skipped: {n_skipped:,}")
    print(f"Failed: {n_failed:,}")
    print(f"Wrote: {out_dir / 'per_treatment_model_summary.tsv'}")
    print(f"Wrote: {out_dir / 'skipped_treatments.tsv'}")
    print(f"Wrote: {out_dir / 'per_treatment_predictions_all.tsv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
