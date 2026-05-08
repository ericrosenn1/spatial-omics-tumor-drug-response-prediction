"""
Script:
    03_train_global_spatial_response_model.py

Purpose:
    Train the global spatial response prediction model.

Role:
    Third step in spatial_prediction_model.
    Consumes the leakage safe modeling dataset from step 02.
    Trains one pooled model across all sample treatment rows.
    Uses spatial features plus optional drug dummy features.
    Writes model object, train/test predictions, metrics, and feature importance.

Pipeline position:
    01_validate_prediction_inputs.py
        verifies teacher_builder handoff files and basic data integrity.

    02_build_spatial_modeling_dataset.py
        builds modeling_table.tsv, X_features.csv, y_target.csv,
        model_feature_manifest.csv, leakage_excluded_columns.tsv,
        and sample_split.tsv.

    03_train_global_spatial_response_model.py
        reads those step 02 outputs, fits the global predictive model,
        evaluates grouped train/test performance, and writes reusable model
        artifacts for downstream explanation and prediction steps.

Modeling idea:
    The global model learns:
        spatial features + treatment identity -> fused teacher response

    The target is usually fused_prob_responder.
    For the first 10 sample run, the model is a smoke test and alignment check.
    For the 102 sample run, the same script should run without code edits.

Expected inputs from step 02:
    outputs/<run_name>/02_modeling_dataset/modeling_table.tsv
        Metadata, target, spatial features, and optional drug dummy features.

    outputs/<run_name>/02_modeling_dataset/X_features.csv
        Numeric feature matrix aligned row by row to modeling_table.tsv.

    outputs/<run_name>/02_modeling_dataset/y_target.csv
        Target vector aligned row by row to modeling_table.tsv.

    outputs/<run_name>/02_modeling_dataset/model_feature_manifest.csv
        Feature names and optional annotations such as feature group/type.

    outputs/<run_name>/02_modeling_dataset/sample_split.tsv
        Group split assignment, usually one row per sample_id.

Primary outputs:
    outputs/<run_name>/03_global_model/model.joblib
        Trained sklearn style pipeline plus metadata bundle.

    outputs/<run_name>/03_global_model/predictions_train.tsv
        Row level train predictions.

    outputs/<run_name>/03_global_model/predictions_test.tsv
        Row level test predictions.

    outputs/<run_name>/03_global_model/predictions_all_labeled.tsv
        Row level predictions for all labeled rows used by the global model.

    outputs/<run_name>/03_global_model/metrics.tsv
        Train/test metrics.

    outputs/<run_name>/03_global_model/feature_importance.tsv
        Model feature importance if the fitted estimator exposes it.

    outputs/<run_name>/03_global_model/run_config.json
        Reproducibility record for this run.

    outputs/<run_name>/03_global_model/global_model_summary.txt
        Human readable summary.

Design contract:
    YAML driven paths, columns, model choices, and hyperparameters.
    No hard coded project specific sample counts.
    Group split by sample_id unless step 02 provides explicit row splits.
    Median imputation used before tree models because sklearn forests do not
    reliably accept missing values across all environments.
    No feature scaling, because tree models do not require it.

Notes:
    RandomForest is the default because it is stable, interpretable enough for
    feature importance, and available in standard sklearn environments.
    XGBoost support is optional and only used when requested in YAML.
    SHAP is intentionally not computed here; that belongs in step 05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


SCRIPT_NAME = "03_train_global_spatial_response_model.py"
STEP_NAME = "03_train_global_spatial_response_model"
DEFAULT_MODELING_DATASET_SUBDIR = "02_modeling_dataset"
DEFAULT_GLOBAL_MODEL_SUBDIR = "03_global_model"


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI args
    config path and overwrite controls"""
    parser = argparse.ArgumentParser(description="Train global spatial response model")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    parser.add_argument(
        "--allow-missing-test",
        action="store_true",
        help="Allow training when no test split rows are available",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config
    UTF8 BOM tolerant"""
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """get config value
    fallback aware"""
    return cfg[key] if key in cfg else default


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve path value
    absolute or project relative"""
    if value in [None, ""]:
        return None

    path = Path(str(value))

    if path.is_absolute():
        return path

    return project_dir / path


def get_output_root(cfg: dict[str, Any]) -> Path:
    """resolve output root
    YAML output_root required"""
    return Path(str(cfg["output_root"]))


def get_step_dir(cfg: dict[str, Any], key: str, default_subdir: str) -> Path:
    """resolve step output folder
    output_root plus output_subdirs key"""
    output_root = get_output_root(cfg)
    output_subdirs = get_cfg(cfg, "output_subdirs", {}) or {}
    subdir = output_subdirs.get(key, default_subdir)

    return output_root / subdir


def ensure_dir(path: Path) -> None:
    """create folder
    parents included"""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write text file
    newline joined"""
    path.write_text("\n".join(lines), encoding="utf-8")


def save_json(data: dict[str, Any], path: Path) -> None:
    """write JSON file
    readable indent"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def clean_text(value: Any) -> str:
    """clean scalar text
    empty string for missing"""
    if pd.isna(value):
        return ""

    return str(value).strip()


def bool_like(value: Any) -> bool:
    """truth parser
    tolerant text handling"""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "included"}


def safe_float(value: Any, default: float | None = None) -> float | None:
    """parse float
    fallback for null config values"""
    if value in [None, "", "null", "None"]:
        return default

    return float(value)


def safe_int(value: Any, default: int | None = None) -> int | None:
    """parse int
    fallback for null config values"""
    if value in [None, "", "null", "None"]:
        return default

    return int(value)


# ============================================================
# INPUT DISCOVERY AND LOADING
# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """load CSV or TSV
    extension based parser"""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    # fallback for uncommon suffixes
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def get_modeling_input_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """resolve step 02 outputs
    expected model matrix files"""
    modeling_dir = get_step_dir(
        cfg=cfg,
        key="modeling_dataset",
        default_subdir=DEFAULT_MODELING_DATASET_SUBDIR,
    )

    return {
        "modeling_dir": modeling_dir,
        "modeling_table": modeling_dir / "modeling_table.tsv",
        "x_features": modeling_dir / "X_features.csv",
        "y_target": modeling_dir / "y_target.csv",
        "model_feature_manifest": modeling_dir / "model_feature_manifest.csv",
        "sample_split": modeling_dir / "sample_split.tsv",
        "split_assignments": modeling_dir / "split_assignments.tsv",
    }


def validate_input_files(paths: dict[str, Path]) -> pd.DataFrame:
    """check expected files
    sample split aliases supported"""
    required = ["modeling_table", "x_features"]
    optional = ["y_target", "model_feature_manifest", "sample_split", "split_assignments"]

    rows: list[dict[str, Any]] = []

    for key in required + optional:
        path = paths[key]
        exists = bool(path.exists())
        is_file = bool(exists and path.is_file())

        rows.append(
            {
                "input_key": key,
                "path": str(path),
                "required": key in required,
                "exists": exists,
                "is_file": is_file,
                "size_bytes": int(path.stat().st_size) if is_file else np.nan,
            }
        )

    check = pd.DataFrame(rows)

    missing_required = check.loc[(check["required"] == True) & (check["is_file"] != True)]
    if not missing_required.empty:
        missing = "; ".join(missing_required["input_key"].astype(str).tolist())
        raise FileNotFoundError(f"Missing required step 02 outputs: {missing}")

    return check


def drop_unwanted_feature_columns(X: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """remove accidental metadata columns
    leakage and id columns never train"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    slide_col = str(get_cfg(cfg, "slide_col", "slide_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    leakage_cols = set(str(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or [])
    passthrough_cols = set(str(x) for x in get_cfg(cfg, "metadata_passthrough_columns", []) or [])

    # explicit safety set, catches accidental carry through from step 02
    blocked = set([sample_col, slide_col, drug_col, drug_key_col, target_col])
    blocked |= leakage_cols
    blocked |= passthrough_cols
    blocked |= {"row_id", "split", "model_split", "Unnamed: 0"}

    rows: list[dict[str, Any]] = []
    keep_cols: list[str] = []

    for col in X.columns:
        blocked_reason = ""

        if col in blocked:
            blocked_reason = "metadata_or_leakage"

        rows.append(
            {
                "column": col,
                "kept_as_feature": blocked_reason == "",
                "drop_reason": blocked_reason,
            }
        )

        if blocked_reason == "":
            keep_cols.append(col)

    return X[keep_cols].copy(), pd.DataFrame(rows)


def numeric_feature_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """coerce features to numeric
    records coercion missingness"""
    rows: list[dict[str, Any]] = []
    converted_cols: dict[str, pd.Series] = {}

    for col in X.columns:
        original = X[col]
        converted = pd.to_numeric(original, errors="coerce")

        # coercion gap helps detect unexpected strings in step 02 output
        original_nonmissing = original.notna() & original.astype(str).str.strip().ne("")
        converted_nonmissing = converted.notna()
        lost_to_coercion = int((original_nonmissing & ~converted_nonmissing).sum())

        rows.append(
            {
                "feature_name": col,
                "original_dtype": str(original.dtype),
                "numeric_dtype": str(converted.dtype),
                "n_rows": int(len(converted)),
                "n_missing_after_numeric": int(converted.isna().sum()),
                "missing_fraction_after_numeric": float(converted.isna().mean()) if len(converted) else np.nan,
                "n_lost_to_numeric_coercion": lost_to_coercion,
                "n_unique_numeric": int(converted.nunique(dropna=True)),
            }
        )

        converted_cols[col] = converted

    X_numeric = pd.DataFrame(converted_cols, index=X.index)
    return X_numeric, pd.DataFrame(rows)


def load_y_target(paths: dict[str, Path], modeling_table: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """load target vector
    y_target preferred, table fallback"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    y_path = paths["y_target"]

    if y_path.exists():
        y_df = load_table(y_path)

        if target_col in y_df.columns:
            y = y_df[target_col]
        elif "target" in y_df.columns:
            y = y_df["target"]
        elif y_df.shape[1] == 1:
            y = y_df.iloc[:, 0]
        else:
            raise ValueError(
                f"Could not identify target column in {y_path}. "
                f"Expected {target_col}, target, or single column."
            )

        return pd.to_numeric(y, errors="coerce")

    if target_col not in modeling_table.columns:
        raise ValueError(f"Target not found in y_target.csv or modeling_table.tsv: {target_col}")

    return pd.to_numeric(modeling_table[target_col], errors="coerce")


def load_feature_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    """load model feature manifest
    empty frame if absent"""
    manifest_path = paths["model_feature_manifest"]

    if not manifest_path.exists():
        return pd.DataFrame()

    return load_table(manifest_path)


def resolve_manifest_feature_col(manifest: pd.DataFrame, cfg: dict[str, Any]) -> str | None:
    """find feature name column
    YAML setting plus common aliases"""
    preferred = clean_text(get_cfg(cfg, "feature_name_col", "feature_name"))
    aliases = [preferred, "feature_name", "feature", "column", "column_name"]

    for col in aliases:
        if col in manifest.columns:
            return col

    return None


def filter_features_by_manifest(
    X: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """restrict X by manifest
    keeps X order when manifest absent"""
    if manifest.empty:
        report = pd.DataFrame(
            {
                "feature_name": list(X.columns),
                "in_manifest": False,
                "used_for_training": True,
            }
        )
        return X.copy(), report

    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        report = pd.DataFrame(
            {
                "feature_name": list(X.columns),
                "in_manifest": False,
                "used_for_training": True,
            }
        )
        return X.copy(), report

    manifest_work = manifest.copy()

    if "included" in manifest_work.columns:
        manifest_work = manifest_work[manifest_work["included"].map(bool_like)].copy()

    manifest_features = manifest_work[feature_col].dropna().astype(str).map(clean_text).tolist()
    manifest_features = [f for f in manifest_features if f]

    # preserve feature order from X to match fitted estimator input
    manifest_set = set(manifest_features)
    training_cols = [col for col in X.columns if col in manifest_set]

    # if manifest has only spatial features and X has drug dummies, keep drug dummies too
    drug_prefix = str(get_cfg(cfg, "drug_dummy_prefix", "drug__"))
    drug_dummy_cols = [col for col in X.columns if col.startswith(drug_prefix)]

    if bool(get_cfg(cfg, "include_drug_dummies", True)):
        for col in drug_dummy_cols:
            if col not in training_cols:
                training_cols.append(col)

    # if the manifest produces an empty feature list, fail early
    if len(training_cols) == 0:
        raise ValueError("No training features remain after manifest filtering")

    rows: list[dict[str, Any]] = []
    for col in X.columns:
        rows.append(
            {
                "feature_name": col,
                "in_manifest": col in manifest_set,
                "is_drug_dummy": col.startswith(drug_prefix),
                "used_for_training": col in training_cols,
            }
        )

    return X[training_cols].copy(), pd.DataFrame(rows)


def load_modeling_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    """load step 02 products
    aligned matrices and reports"""
    paths = get_modeling_input_paths(cfg)
    path_report = validate_input_files(paths)

    modeling_table = load_table(paths["modeling_table"])
    X_raw = load_table(paths["x_features"])
    y = load_y_target(paths, modeling_table, cfg)
    manifest = load_feature_manifest(paths)

    if len(X_raw) != len(modeling_table):
        raise ValueError(
            f"X_features row count does not match modeling_table: "
            f"X={len(X_raw)}; modeling_table={len(modeling_table)}"
        )

    if len(y) != len(modeling_table):
        raise ValueError(
            f"Target row count does not match modeling_table: "
            f"y={len(y)}; modeling_table={len(modeling_table)}"
        )

    X_no_blocked, feature_drop_report = drop_unwanted_feature_columns(X_raw, cfg)
    X_numeric, numeric_report = numeric_feature_matrix(X_no_blocked)
    X_final, manifest_filter_report = filter_features_by_manifest(X_numeric, manifest, cfg)

    if X_final.shape[1] == 0:
        raise ValueError("No usable model features found")

    return {
        "paths": paths,
        "path_report": path_report,
        "modeling_table": modeling_table.reset_index(drop=True),
        "X": X_final.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "manifest": manifest,
        "feature_drop_report": feature_drop_report,
        "numeric_report": numeric_report,
        "manifest_filter_report": manifest_filter_report,
    }


# ============================================================
# SPLIT HANDLING
# ============================================================


def find_split_file(paths: dict[str, Path]) -> Path | None:
    """find split file
    primary name then alias"""
    for key in ["sample_split", "split_assignments"]:
        path = paths.get(key)
        if path is not None and path.exists():
            return path

    return None


def normalize_split_label(value: Any) -> str:
    """normalize split label
    train test validation aliases"""
    text = clean_text(value).lower()

    if text in {"train", "training"}:
        return "train"

    if text in {"test", "holdout", "heldout"}:
        return "test"

    if text in {"val", "valid", "validation"}:
        return "validation"

    return text


def generate_group_holdout_split(modeling_table: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """create group holdout split
    fallback when step 02 split absent"""
    group_col = str(get_cfg(cfg, "split_group_col", get_cfg(cfg, "sample_col", "sample_id")))
    random_state = int(get_cfg(cfg, "random_state", 42))
    test_size = float(get_cfg(cfg, "test_size", 0.20))

    if group_col not in modeling_table.columns:
        raise ValueError(f"Split group column missing from modeling_table: {group_col}")

    groups = modeling_table[group_col].astype(str)
    unique_groups = groups.dropna().unique()

    # one group cannot support holdout without losing all training rows
    if len(unique_groups) < 2:
        return pd.Series(["train"] * len(modeling_table), index=modeling_table.index)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    dummy_y = np.zeros(len(modeling_table))
    train_idx, test_idx = next(splitter.split(modeling_table, dummy_y, groups=groups))

    split = pd.Series(["train"] * len(modeling_table), index=modeling_table.index)
    split.iloc[test_idx] = "test"
    split.iloc[train_idx] = "train"

    return split


def attach_split(modeling_table: pd.DataFrame, paths: dict[str, Path], cfg: dict[str, Any]) -> tuple[pd.Series, str]:
    """attach row split labels
    sample split preferred"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    group_col = str(get_cfg(cfg, "split_group_col", sample_col))
    split_path = find_split_file(paths)

    if split_path is None:
        return generate_group_holdout_split(modeling_table, cfg), "generated_group_holdout"

    split_df = load_table(split_path)

    if "split" not in split_df.columns:
        raise ValueError(f"Split file has no split column: {split_path}")

    split_work = split_df.copy()
    split_work["split"] = split_work["split"].map(normalize_split_label)

    # row_id joins are most precise when step 02 writes row level splits
    if "row_id" in split_work.columns and "row_id" in modeling_table.columns:
        merged = modeling_table[["row_id"]].merge(
            split_work[["row_id", "split"]],
            on="row_id",
            how="left",
        )
        split = merged["split"]
        source = "row_id_split_file"

    # sample level split is the intended grouped split contract
    elif group_col in split_work.columns and group_col in modeling_table.columns:
        merged = modeling_table[[group_col]].merge(
            split_work[[group_col, "split"]].drop_duplicates(group_col),
            on=group_col,
            how="left",
        )
        split = merged["split"]
        source = "group_split_file"

    # final alignment fallback for split files written row by row
    elif len(split_work) == len(modeling_table):
        split = split_work["split"].reset_index(drop=True)
        source = "row_order_split_file"

    else:
        raise ValueError(
            f"Could not align split file to modeling_table: {split_path}. "
            f"Need row_id, {group_col}, or matching row count."
        )

    # missing split labels become train to avoid losing rows silently
    split = split.fillna("train").map(normalize_split_label)

    return split.reset_index(drop=True), source


def build_split_summary(modeling_table: pd.DataFrame, split: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize split rows
    counts by samples and treatments"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    work = modeling_table.copy()
    work["split"] = split.values

    rows: list[dict[str, Any]] = []

    for split_name, df in work.groupby("split", dropna=False):
        row = {
            "split": split_name,
            "n_rows": int(len(df)),
            "n_samples": int(df[sample_col].nunique()) if sample_col in df.columns else np.nan,
            "n_treatments": int(df[drug_key_col].nunique()) if drug_key_col in df.columns else np.nan,
            "target_mean": float(pd.to_numeric(df[target_col], errors="coerce").mean()) if target_col in df.columns else np.nan,
            "target_std": float(pd.to_numeric(df[target_col], errors="coerce").std()) if target_col in df.columns else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


# ============================================================
# MODEL CONSTRUCTION
# ============================================================


def infer_task(cfg: dict[str, Any]) -> str:
    """resolve modeling task
    regression or classification"""
    task = clean_text(get_cfg(cfg, "task", "regression")).lower()

    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task: {task}")

    return task


def make_model(cfg: dict[str, Any], task: str):
    """build estimator
    RandomForest default, XGBoost optional"""
    model_type = clean_text(get_cfg(cfg, "model_type", "random_forest")).lower()
    random_state = int(get_cfg(cfg, "random_state", 42))
    n_jobs = int(get_cfg(cfg, "n_jobs", -1))

    if model_type in {"random_forest", "rf", "randomforest"}:
        rf_cfg = get_cfg(cfg, "random_forest", {}) or {}

        # values mirror YAML defaults and sklearn defaults where sensible
        n_estimators = int(rf_cfg.get("n_estimators", 500))
        max_depth = safe_int(rf_cfg.get("max_depth", None), default=None)
        min_samples_leaf = int(rf_cfg.get("min_samples_leaf", 1))
        min_samples_split = int(rf_cfg.get("min_samples_split", 2))
        max_features = rf_cfg.get("max_features", "sqrt")
        bootstrap = bool(rf_cfg.get("bootstrap", True))

        if task == "classification":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight="balanced",
            )

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_type in {"xgboost", "xgb"}:
        xgb_cfg = get_cfg(cfg, "xgboost", {}) or {}

        try:
            from xgboost import XGBClassifier, XGBRegressor
        except Exception as exc:
            raise ImportError(
                "xgboost was requested but could not be imported. "
                "Set model_type: random_forest or install xgboost."
            ) from exc

        common = {
            "n_estimators": int(xgb_cfg.get("n_estimators", 500)),
            "max_depth": int(xgb_cfg.get("max_depth", 4)),
            "learning_rate": float(xgb_cfg.get("learning_rate", 0.03)),
            "subsample": float(xgb_cfg.get("subsample", 0.85)),
            "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.85)),
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        if task == "classification":
            return XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric="logloss",
            )

        return XGBRegressor(
            **common,
            objective=str(xgb_cfg.get("objective", "reg:squarederror")),
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def make_pipeline(cfg: dict[str, Any], task: str) -> Pipeline:
    """build sklearn pipeline
    imputer plus estimator"""
    estimator = make_model(cfg, task)

    # median is robust to outliers and works well for tree inputs
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def prepare_target_for_task(y: pd.Series, cfg: dict[str, Any], task: str) -> pd.Series:
    """prepare y vector
    threshold only for classification"""
    y_num = pd.to_numeric(y, errors="coerce")

    if task == "classification":
        threshold = float(get_cfg(cfg, "binary_threshold", 0.5))
        return (y_num >= threshold).astype(int)

    return y_num.astype(float)


# ============================================================
# PREDICTIONS AND METRICS
# ============================================================


def predict_model(pipeline: Pipeline, X: pd.DataFrame, task: str) -> dict[str, np.ndarray]:
    """predict with pipeline
    returns task aware arrays"""
    if len(X) == 0:
        return {"prediction": np.array([]), "probability": np.array([])}

    if task == "classification":
        pred_label = pipeline.predict(X)

        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X)[:, 1]
        else:
            prob = pred_label.astype(float)

        return {"prediction": pred_label, "probability": prob}

    pred = pipeline.predict(X)
    return {"prediction": pred.astype(float), "probability": pred.astype(float)}


def safe_pearson(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """pearson correlation
    NaN if undefined"""
    a = pd.to_numeric(pd.Series(y_true), errors="coerce")
    b = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = a.notna() & b.notna()

    if int(mask.sum()) < 2:
        return float("nan")

    if float(a[mask].std()) == 0 or float(b[mask].std()) == 0:
        return float("nan")

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def safe_spearman(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """spearman correlation
    pandas rank fallback"""
    a = pd.to_numeric(pd.Series(y_true), errors="coerce")
    b = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = a.notna() & b.notna()

    if int(mask.sum()) < 2:
        return float("nan")

    return safe_pearson(a[mask].rank(method="average"), b[mask].rank(method="average"))


def safe_auc(y_binary: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    """binary AUC
    NaN if one class only"""
    y = pd.Series(y_binary).dropna()
    s = pd.Series(score).loc[y.index]

    if y.nunique(dropna=True) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, split_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """compute regression metrics
    includes threshold AUC"""
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.Series(y_pred, index=y.index).astype(float)
    mask = y.notna() & p.notna()

    if int(mask.sum()) == 0:
        return {"split": split_name, "task": "regression", "n_rows": 0}

    y_m = y[mask]
    p_m = p[mask]
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))
    y_binary = (y_m >= threshold).astype(int)

    rmse = float(math.sqrt(mean_squared_error(y_m, p_m)))

    row = {
        "split": split_name,
        "task": "regression",
        "n_rows": int(len(y_m)),
        "target_mean": float(y_m.mean()),
        "target_std": float(y_m.std()),
        "prediction_mean": float(p_m.mean()),
        "prediction_std": float(p_m.std()),
        "mae": float(mean_absolute_error(y_m, p_m)),
        "rmse": rmse,
        "r2": float(r2_score(y_m, p_m)) if len(y_m) >= 2 else np.nan,
        "pearson": safe_pearson(y_m, p_m),
        "spearman": safe_spearman(y_m, p_m),
        "auc_at_binary_threshold": safe_auc(y_binary, p_m),
        "binary_threshold": threshold,
    }

    return row


def classification_metrics(
    y_true: pd.Series,
    y_pred_label: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    """compute classification metrics
    accuracy balanced accuracy AUC"""
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.Series(y_pred_label, index=y.index)
    prob = pd.Series(y_prob, index=y.index)
    mask = y.notna() & p.notna()

    if int(mask.sum()) == 0:
        return {"split": split_name, "task": "classification", "n_rows": 0}

    y_m = y[mask].astype(int)
    p_m = p[mask].astype(int)
    prob_m = prob[mask].astype(float)

    row = {
        "split": split_name,
        "task": "classification",
        "n_rows": int(len(y_m)),
        "n_positive": int((y_m == 1).sum()),
        "n_negative": int((y_m == 0).sum()),
        "accuracy": float(accuracy_score(y_m, p_m)),
        "balanced_accuracy": float(balanced_accuracy_score(y_m, p_m)) if y_m.nunique() > 1 else np.nan,
        "auc": safe_auc(y_m, prob_m),
        "prediction_positive_rate": float((p_m == 1).mean()),
        "probability_mean": float(prob_m.mean()),
    }

    return row


def build_prediction_table(
    modeling_table: pd.DataFrame,
    y_true_raw: pd.Series,
    y_train_task: pd.Series,
    split: pd.Series,
    predictions: dict[str, np.ndarray],
    cfg: dict[str, Any],
    task: str,
) -> pd.DataFrame:
    """build row prediction table
    metadata plus prediction columns"""
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))
    passthrough_cols = [str(x) for x in get_cfg(cfg, "metadata_passthrough_columns", []) or []]

    keep_cols = [col for col in passthrough_cols if col in modeling_table.columns]

    # preserve row identity even if step 02 did not write row_id
    out = modeling_table[keep_cols].copy()
    out.insert(0, "row_id", np.arange(len(modeling_table)))
    out["split"] = split.values
    out[target_col] = pd.to_numeric(y_true_raw, errors="coerce")
    out["target_for_training"] = y_train_task.values
    out["target_binary_at_threshold"] = (out[target_col] >= threshold).astype(int)

    if task == "classification":
        out["predicted_label"] = predictions["prediction"]
        out["predicted_prob_responder"] = predictions["probability"]
        out["prediction_error"] = out["predicted_prob_responder"] - out["target_binary_at_threshold"]
    else:
        out["predicted_fused_prob_responder"] = predictions["prediction"]
        out["predicted_binary_at_threshold"] = (out["predicted_fused_prob_responder"] >= threshold).astype(int)
        out["residual"] = out["predicted_fused_prob_responder"] - out[target_col]

    return out


def compute_metrics_for_splits(
    y_true_raw: pd.Series,
    y_task: pd.Series,
    split: pd.Series,
    prediction_table: pd.DataFrame,
    cfg: dict[str, Any],
    task: str,
) -> pd.DataFrame:
    """compute metrics by split
    train test and all labeled"""
    rows: list[dict[str, Any]] = []

    split_labels = ["train", "test", "validation", "all_labeled"]

    for split_name in split_labels:
        if split_name == "all_labeled":
            mask = pd.Series([True] * len(split), index=split.index)
        else:
            mask = split == split_name

        if int(mask.sum()) == 0:
            continue

        if task == "classification":
            row = classification_metrics(
                y_true=y_task[mask],
                y_pred_label=prediction_table.loc[mask, "predicted_label"].to_numpy(),
                y_prob=prediction_table.loc[mask, "predicted_prob_responder"].to_numpy(),
                split_name=split_name,
            )
        else:
            row = regression_metrics(
                y_true=y_true_raw[mask],
                y_pred=prediction_table.loc[mask, "predicted_fused_prob_responder"].to_numpy(),
                split_name=split_name,
                cfg=cfg,
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def manifest_annotations(manifest: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """extract manifest annotations
    safe empty annotations"""
    if manifest.empty:
        return pd.DataFrame(columns=["feature_name"])

    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        return pd.DataFrame(columns=["feature_name"])

    out = manifest.copy()
    out["feature_name"] = out[feature_col].astype(str).map(clean_text)

    # keep only useful annotation columns, avoid duplicate feature name aliases
    keep_cols = ["feature_name"]
    for col in ["feature_group", "feature_type", "source", "included", "reason"]:
        if col in out.columns and col not in keep_cols:
            keep_cols.append(col)

    return out[keep_cols].drop_duplicates("feature_name")


def build_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """extract feature importance
    tree estimator support"""
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": np.nan,
                "importance_rank": np.nan,
            }
        )

    importance = np.asarray(model.feature_importances_, dtype=float)

    if len(importance) != len(feature_names):
        raise ValueError("Feature importance length does not match feature names")

    out = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importance,
        }
    )

    out["importance_rank"] = out["importance"].rank(ascending=False, method="dense").astype(int)
    out["is_drug_dummy"] = out["feature_name"].str.startswith(str(get_cfg(cfg, "drug_dummy_prefix", "drug__")))

    annotations = manifest_annotations(manifest, cfg)
    if not annotations.empty:
        out = out.merge(annotations, on="feature_name", how="left")

    out = out.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)

    return out


# ============================================================
# SUMMARY WRITING
# ============================================================


def write_summary_text(
    path: Path,
    cfg: dict[str, Any],
    input_report: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_names: list[str],
    split_source: str,
) -> None:
    """write readable model summary
    compact run report"""
    lines: list[str] = []
    lines.append("Global spatial response model summary")
    lines.append("")

    lines.append("Run settings")
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  task: {get_cfg(cfg, 'task', 'regression')}")
    lines.append(f"  model_type: {get_cfg(cfg, 'model_type', 'random_forest')}")
    lines.append(f"  random_state: {get_cfg(cfg, 'random_state', 42)}")
    lines.append(f"  split_source: {split_source}")
    lines.append("")

    lines.append("Inputs")
    for _, row in input_report.iterrows():
        status = "exists" if bool(row.get("exists")) else "missing"
        req = "required" if bool(row.get("required")) else "optional"
        lines.append(f"  {row['input_key']}: {status}; {req}; {row['path']}")
    lines.append("")

    lines.append("Feature matrix")
    lines.append(f"  n_features: {len(feature_names)}")
    lines.append(f"  n_drug_dummy_features: {sum(str(f).startswith(str(get_cfg(cfg, 'drug_dummy_prefix', 'drug__'))) for f in feature_names)}")
    lines.append("")

    if not split_summary.empty:
        lines.append("Split summary")
        for _, row in split_summary.iterrows():
            lines.append(
                "  "
                + f"{row['split']}: rows={row['n_rows']}; "
                + f"samples={row['n_samples']}; treatments={row['n_treatments']}; "
                + f"target_mean={row['target_mean']:.4f}"
            )
        lines.append("")

    if not metrics.empty:
        lines.append("Metrics")
        for _, row in metrics.iterrows():
            if row.get("task") == "regression":
                lines.append(
                    "  "
                    + f"{row['split']}: n={row['n_rows']}; "
                    + f"mae={row.get('mae', np.nan):.4f}; "
                    + f"rmse={row.get('rmse', np.nan):.4f}; "
                    + f"r2={row.get('r2', np.nan):.4f}; "
                    + f"pearson={row.get('pearson', np.nan):.4f}"
                )
            else:
                lines.append(
                    "  "
                    + f"{row['split']}: n={row['n_rows']}; "
                    + f"accuracy={row.get('accuracy', np.nan):.4f}; "
                    + f"balanced_accuracy={row.get('balanced_accuracy', np.nan):.4f}; "
                    + f"auc={row.get('auc', np.nan):.4f}"
                )
        lines.append("")

    if not feature_importance.empty and "importance" in feature_importance.columns:
        top = feature_importance.head(15)
        lines.append("Top feature importance")
        for _, row in top.iterrows():
            lines.append(f"  {row['feature_name']}: {row['importance']:.6f}")
        lines.append("")

    lines.append("Notes")
    lines.append("  10 sample mode is a smoke test, not a final performance estimate")
    lines.append("  Group split protects against same sample appearing in train and test")
    lines.append("  SHAP is handled by step 05")

    write_text(path, lines)


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================


def main() -> int:
    """run global model training
    write artifacts and reports"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    out_dir = get_step_dir(
        cfg=cfg,
        key="global_model",
        default_subdir=DEFAULT_GLOBAL_MODEL_SUBDIR,
    )
    ensure_dir(out_dir)

    print("Training global spatial response model")
    print(f"Config: {Path(args.config)}")
    print(f"Output: {out_dir}")

    loaded = load_modeling_inputs(cfg)

    modeling_table = loaded["modeling_table"]
    X = loaded["X"]
    y_raw = loaded["y"]
    manifest = loaded["manifest"]
    paths = loaded["paths"]

    split, split_source = attach_split(modeling_table, paths, cfg)
    split_summary = build_split_summary(modeling_table, split, cfg)

    task = infer_task(cfg)
    y_task = prepare_target_for_task(y_raw, cfg, task)

    valid_target_mask = y_task.notna()

    if int(valid_target_mask.sum()) == 0:
        raise ValueError("No nonmissing target values available for training")

    # train/test masks after target filtering
    train_mask = (split == "train") & valid_target_mask
    test_mask = (split == "test") & valid_target_mask

    if int(train_mask.sum()) == 0:
        raise ValueError("No training rows available after split and target filtering")

    if int(test_mask.sum()) == 0 and not args.allow_missing_test:
        allow_small = bool(get_cfg(cfg, "allow_small_test_mode_metrics", False))
        if not allow_small:
            raise ValueError("No test rows available. Use --allow-missing-test or set allow_small_test_mode_metrics true.")

    print(f"Rows loaded: {len(modeling_table):,}")
    print(f"Features: {X.shape[1]:,}")
    print(f"Train rows: {int(train_mask.sum()):,}")
    print(f"Test rows: {int(test_mask.sum()):,}")
    print(f"Task: {task}")
    print(f"Model type: {get_cfg(cfg, 'model_type', 'random_forest')}")

    pipeline = make_pipeline(cfg, task)
    pipeline.fit(X.loc[train_mask], y_task.loc[train_mask])

    all_predictions = predict_model(pipeline, X, task)

    prediction_table = build_prediction_table(
        modeling_table=modeling_table,
        y_true_raw=y_raw,
        y_train_task=y_task,
        split=split,
        predictions=all_predictions,
        cfg=cfg,
        task=task,
    )

    metrics = compute_metrics_for_splits(
        y_true_raw=y_raw,
        y_task=y_task,
        split=split,
        prediction_table=prediction_table,
        cfg=cfg,
        task=task,
    )

    feature_names = X.columns.astype(str).tolist()
    feature_importance = build_feature_importance(
        pipeline=pipeline,
        feature_names=feature_names,
        manifest=manifest,
        cfg=cfg,
    )

    # primary model bundle, kept as dict for downstream steps
    model_bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "target_col": str(get_cfg(cfg, "target_col", "fused_prob_responder")),
        "task": task,
        "model_type": str(get_cfg(cfg, "model_type", "random_forest")),
        "split_source": split_source,
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
    }

    model_path = out_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)

    # split specific predictions for expected downstream file names
    train_predictions = prediction_table.loc[prediction_table["split"] == "train"].copy()
    test_predictions = prediction_table.loc[prediction_table["split"] == "test"].copy()

    train_predictions.to_csv(out_dir / "predictions_train.tsv", sep="\t", index=False)
    test_predictions.to_csv(out_dir / "predictions_test.tsv", sep="\t", index=False)
    prediction_table.to_csv(out_dir / "predictions_all_labeled.tsv", sep="\t", index=False)

    metrics.to_csv(out_dir / "metrics.tsv", sep="\t", index=False)
    feature_importance.to_csv(out_dir / "feature_importance.tsv", sep="\t", index=False)
    split_summary.to_csv(out_dir / "split_summary.tsv", sep="\t", index=False)

    loaded["path_report"].to_csv(out_dir / "input_file_report.tsv", sep="\t", index=False)
    loaded["feature_drop_report"].to_csv(out_dir / "feature_drop_report.tsv", sep="\t", index=False)
    loaded["numeric_report"].to_csv(out_dir / "feature_numeric_report.tsv", sep="\t", index=False)
    loaded["manifest_filter_report"].to_csv(out_dir / "feature_manifest_filter_report.tsv", sep="\t", index=False)

    write_summary_text(
        path=out_dir / "global_model_summary.txt",
        cfg=cfg,
        input_report=loaded["path_report"],
        split_summary=split_summary,
        metrics=metrics,
        feature_importance=feature_importance,
        feature_names=feature_names,
        split_source=split_source,
    )

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(Path(args.config)),
        "output_dir": str(out_dir),
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "task": task,
        "model_type": get_cfg(cfg, "model_type", "random_forest"),
        "random_state": get_cfg(cfg, "random_state", 42),
        "n_rows_loaded": int(len(modeling_table)),
        "n_features": int(X.shape[1]),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "split_source": split_source,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "outputs": {
            "model": str(model_path),
            "predictions_train": str(out_dir / "predictions_train.tsv"),
            "predictions_test": str(out_dir / "predictions_test.tsv"),
            "metrics": str(out_dir / "metrics.tsv"),
            "feature_importance": str(out_dir / "feature_importance.tsv"),
        },
    }
    save_json(run_info, out_dir / "run_config.json")

    print("\nDONE")
    print(f"Model: {model_path}")
    print(f"Train predictions: {out_dir / 'predictions_train.tsv'}")
    print(f"Test predictions: {out_dir / 'predictions_test.tsv'}")
    print(f"Metrics: {out_dir / 'metrics.tsv'}")
    print(f"Feature importance: {out_dir / 'feature_importance.tsv'}")
    print(f"Summary: {out_dir / 'global_model_summary.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
