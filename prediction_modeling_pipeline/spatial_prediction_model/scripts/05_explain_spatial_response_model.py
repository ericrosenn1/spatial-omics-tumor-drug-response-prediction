"""
Script:
    05_explain_spatial_response_model.py

Purpose:
    Explain trained spatial response models with global and local feature attribution.

Role:
    Fifth step in spatial_prediction_model.
    Consumes the model ready data from step 02 and the fitted model artifacts from
    steps 03 and 04. Produces feature level, feature group, sample level, and
    treatment level explanation reports.

Pipeline position:
    01_validate_prediction_inputs.py
        validates teacher_builder handoff files.

    02_build_spatial_modeling_dataset.py
        builds modeling_table.tsv, X_features.csv, y_target.csv,
        model_feature_manifest.csv, and sample_split.tsv.

    03_train_global_spatial_response_model.py
        trains the pooled spatial response model and writes model.joblib,
        predictions, metrics, and model feature importance.

    04_train_per_treatment_models.py
        optionally trains treatment specific models and writes treatment level
        model summaries and feature importance files.

    05_explain_spatial_response_model.py
        explains the global model and integrates optional treatment specific
        explanations into publication ready reports.

Explanation strategy:
    SHAP is used when the package is available and the fitted model supports it.
    The script explains the preprocessing transformed matrix used by the fitted
    model, while reporting original biological feature annotations from the
    step 02 manifest. If SHAP cannot run, the script falls back to permutation
    importance and model native importance.

Expected inputs:
    outputs/<run_name>/02_modeling_dataset/modeling_table.tsv
    outputs/<run_name>/02_modeling_dataset/X_features.csv
    outputs/<run_name>/02_modeling_dataset/y_target.csv
    outputs/<run_name>/02_modeling_dataset/model_feature_manifest.csv
    outputs/<run_name>/02_modeling_dataset/sample_split.tsv

    outputs/<run_name>/03_global_model/model.joblib
    outputs/<run_name>/03_global_model/feature_importance.tsv
    outputs/<run_name>/03_global_model/predictions_all_labeled.tsv
    outputs/<run_name>/03_global_model/metrics.tsv

    outputs/<run_name>/04_per_treatment_models/per_treatment_model_summary.tsv
    outputs/<run_name>/04_per_treatment_models/per_treatment_feature_importance_top.tsv

Primary outputs:
    outputs/<run_name>/05_model_explanations/global_feature_explanations.tsv
    outputs/<run_name>/05_model_explanations/global_feature_group_summary.tsv
    outputs/<run_name>/05_model_explanations/global_feature_axis_summary.tsv
    outputs/<run_name>/05_model_explanations/global_local_explanations_top.tsv
    outputs/<run_name>/05_model_explanations/global_sample_explanation_summary.tsv
    outputs/<run_name>/05_model_explanations/global_treatment_explanation_summary.tsv
    outputs/<run_name>/05_model_explanations/per_treatment_explanation_summary.tsv
    outputs/<run_name>/05_model_explanations/per_treatment_global_concordance.tsv
    outputs/<run_name>/05_model_explanations/model_explanation_summary.txt
    outputs/<run_name>/05_model_explanations/run_config.json

Design contract:
    YAML driven paths, columns, thresholds, and explanation settings.
    Supports 10 sample smoke test and future full cohort run without code edits.
    Never splits by row for evaluation; it only reuses the sample split from step 02.
    Does not retrain models.
    Writes empty but schema valid reports when optional step 04 outputs are absent.

Notes:
    The top explanations are model explanations, not causal claims.
    Drug dummy features are reported separately from spatial biology features.
    SHAP values for tree ensembles can be expensive on the full run, so row and
    feature report sizes are controlled through YAML.
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

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score


SCRIPT_NAME = "05_explain_spatial_response_model.py"
STEP_NAME = "05_explain_spatial_response_model"
DEFAULT_MODELING_DATASET_SUBDIR = "02_modeling_dataset"
DEFAULT_GLOBAL_MODEL_SUBDIR = "03_global_model"
DEFAULT_PER_TREATMENT_SUBDIR = "04_per_treatment_models"
DEFAULT_EXPLANATION_SUBDIR = "05_model_explanations"


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI arguments
    config path plus optional SHAP override"""
    parser = argparse.ArgumentParser(description="Explain spatial response model")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP and use permutation or native importance only",
    )
    parser.add_argument(
        "--force-permutation",
        action="store_true",
        help="Run permutation importance even when SHAP succeeds",
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
    """read config value
    small fallback helper"""
    return cfg[key] if key in cfg else default


def is_windows_absolute_path(value: str | Path) -> bool:
    """detect Windows drive path
    helpful when running checks outside Windows"""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(value)))


def path_is_absolute_like(value: str | Path) -> bool:
    """detect absolute paths
    supports current OS and Windows paths"""
    text = str(value)
    return Path(text).is_absolute() or is_windows_absolute_path(text)


def infer_model_root(config_path: Path, cfg: dict[str, Any]) -> Path:
    """infer spatial model root
    config path parent is preferred"""
    configured = get_cfg(cfg, "spatial_prediction_model_dir", None) or get_cfg(cfg, "model_root", None)

    if configured:
        return Path(str(configured))

    if config_path.parent.name.lower() == "configs":
        return config_path.parent.parent.resolve()

    return Path.cwd().resolve()


def resolve_path(value: str | Path | None, base: Path) -> Path | None:
    """resolve path from config
    absolute values preserved"""
    if value in [None, ""]:
        return None

    path = Path(str(value))

    if path_is_absolute_like(path):
        return path

    return base / path


def get_output_root(cfg: dict[str, Any], model_root: Path) -> Path:
    """resolve configured output root
    model relative when needed"""
    value = get_cfg(cfg, "output_root", "outputs")
    path = Path(str(value))

    if path_is_absolute_like(path):
        return path

    return model_root / path


def get_output_subdir(cfg: dict[str, Any], key: str, default: str) -> str:
    """resolve step output subdir
    output_subdirs mapping supported"""
    subdirs = get_cfg(cfg, "output_subdirs", {}) or {}

    if isinstance(subdirs, dict) and key in subdirs:
        return str(subdirs[key])

    return default


def candidate_step_dirs(
    cfg: dict[str, Any],
    model_root: Path,
    key: str,
    default_subdir: str,
) -> list[Path]:
    """build possible step dirs
    handles output_root with or without run_name"""
    output_root = get_output_root(cfg, model_root)
    run_name = str(get_cfg(cfg, "run_name", "")).strip()
    subdir = get_output_subdir(cfg, key, default_subdir)
    subdir_path = Path(subdir)

    candidates: list[Path] = []

    if path_is_absolute_like(subdir_path):
        candidates.append(subdir_path)
    else:
        candidates.append(output_root / subdir_path)
        if run_name:
            candidates.append(output_root / run_name / subdir_path)

    # avoid duplicate paths while preserving order
    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        text = str(path)
        if text not in seen:
            unique.append(path)
            seen.add(text)

    return unique


def find_existing_step_dir(
    cfg: dict[str, Any],
    model_root: Path,
    key: str,
    default_subdir: str,
) -> Path:
    """find existing step folder
    falls back to first candidate"""
    candidates = candidate_step_dirs(cfg, model_root, key, default_subdir)

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def get_output_step_dir(
    cfg: dict[str, Any],
    model_root: Path,
    key: str,
    default_subdir: str,
) -> Path:
    """resolve output step folder
    prefers active run folder"""
    output_root = get_output_root(cfg, model_root)
    run_name = str(get_cfg(cfg, "run_name", "")).strip()
    subdir = get_output_subdir(cfg, key, default_subdir)
    subdir_path = Path(subdir)

    if path_is_absolute_like(subdir_path):
        return subdir_path

    # if root already points at run folder, write directly under it
    if run_name and output_root.name == run_name:
        return output_root / subdir_path

    # if step 02 exists under root, root is already the run folder
    if (output_root / DEFAULT_MODELING_DATASET_SUBDIR).exists():
        return output_root / subdir_path

    # normal scaffold: outputs plus run_name plus step folder
    if run_name:
        return output_root / run_name / subdir_path

    return output_root / subdir_path


def ensure_dir(path: Path) -> None:
    """create directory
    parents included"""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write text report
    newline joined"""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: dict[str, Any], path: Path) -> None:
    """write JSON output
    readable indent"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


# ============================================================
# BASIC HELPERS
# ============================================================


def clean_text(value: Any) -> str:
    """clean scalar text
    empty string for missing"""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_key(value: Any) -> str:
    """normalize grouping key
    lowercase compact spaces"""
    text = clean_text(value).lower()
    return " ".join(text.split())


def bool_like(value: Any) -> bool:
    """parse boolean like values
    tolerant table and YAML handling"""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on", "included"}


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """coerce series to numeric
    invalid values become missing"""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """coerce frame to numeric
    invalid values become missing"""
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def compact_list(values: list[Any], max_items: int = 12) -> str:
    """compact list for reports
    trims long value sets"""
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    shown = cleaned[:max_items]

    if len(cleaned) > max_items:
        shown.append(f"plus {len(cleaned) - max_items} more")

    return "; ".join(shown)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """find first matching column
    exact name order preserved"""
    for col in candidates:
        if col in df.columns:
            return col

    return None


def rank_desc(series: pd.Series) -> pd.Series:
    """rank descending values
    missing values sorted last"""
    return series.rank(ascending=False, method="dense", na_option="bottom")


def safe_divide(numerator: float, denominator: float) -> float:
    """safe numeric division
    NaN for zero denominator"""
    if denominator == 0 or pd.isna(denominator):
        return float("nan")

    return float(numerator) / float(denominator)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation
    rank based pandas fallback"""
    x = safe_numeric_series(a)
    y = safe_numeric_series(b)
    mask = x.notna() & y.notna()

    if int(mask.sum()) < 2:
        return float("nan")

    rx = x[mask].rank(method="average")
    ry = y[mask].rank(method="average")

    if float(rx.std()) == 0 or float(ry.std()) == 0:
        return float("nan")

    return float(rx.corr(ry))


# ============================================================
# INPUT LOADING
# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """load CSV or TSV table
    suffix based parser"""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def find_existing_file(candidates: list[Path], required: bool = True) -> Path | None:
    """find first existing file
    optional mode returns None"""
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    if required:
        joined = "; ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No expected file found. Checked: {joined}")

    return None


def get_input_paths(cfg: dict[str, Any], model_root: Path) -> dict[str, Path | None]:
    """resolve step input paths
    supports canonical and legacy names"""
    step02 = find_existing_step_dir(cfg, model_root, "modeling_dataset", DEFAULT_MODELING_DATASET_SUBDIR)
    step03 = find_existing_step_dir(cfg, model_root, "global_model", DEFAULT_GLOBAL_MODEL_SUBDIR)
    step04 = find_existing_step_dir(cfg, model_root, "per_treatment_models", DEFAULT_PER_TREATMENT_SUBDIR)

    paths: dict[str, Path | None] = {
        "step02_dir": step02,
        "step03_dir": step03,
        "step04_dir": step04,
        "modeling_table": find_existing_file(
            [step02 / "modeling_table.tsv", step02 / "merged_model_input.tsv"],
            required=True,
        ),
        "x_features": find_existing_file(
            [step02 / "X_features.csv", step02 / "X_features.tsv"],
            required=True,
        ),
        "y_target": find_existing_file(
            [step02 / "y_target.csv", step02 / "y_target.tsv"],
            required=False,
        ),
        "feature_manifest": find_existing_file(
            [step02 / "model_feature_manifest.csv", step02 / "modeling_feature_manifest.csv", step02 / "modeling_feature_manifest.tsv"],
            required=True,
        ),
        "sample_split": find_existing_file(
            [step02 / "sample_split.tsv", step02 / "split_assignments.tsv"],
            required=False,
        ),
        "global_model": find_existing_file(
            [step03 / "model.joblib", step03 / "global_model.joblib"],
            required=True,
        ),
        "global_feature_importance": find_existing_file(
            [step03 / "feature_importance.tsv", step03 / "feature_importance.csv"],
            required=False,
        ),
        "global_predictions": find_existing_file(
            [step03 / "predictions_all_labeled.tsv", step03 / "predictions_test.tsv"],
            required=False,
        ),
        "global_metrics": find_existing_file(
            [step03 / "metrics.tsv", step03 / "metrics.csv"],
            required=False,
        ),
        "per_treatment_summary": find_existing_file(
            [step04 / "per_treatment_model_summary.tsv", step04 / "per_treatment_model_summary.csv"],
            required=False,
        ),
        "per_treatment_importance": find_existing_file(
            [step04 / "per_treatment_feature_importance_top.tsv", step04 / "per_treatment_feature_importance_top.csv"],
            required=False,
        ),
    }

    return paths


def build_input_file_report(paths: dict[str, Path | None]) -> pd.DataFrame:
    """summarize input files
    path existence and size"""
    rows: list[dict[str, Any]] = []

    for key, path in paths.items():
        exists = bool(path is not None and path.exists())
        is_file = bool(exists and path.is_file())
        is_dir = bool(exists and path.is_dir())

        rows.append(
            {
                "input_name": key,
                "path": str(path) if path is not None else "",
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "size_bytes": int(path.stat().st_size) if is_file else np.nan,
            }
        )

    return pd.DataFrame(rows)


def load_optional_table(path: Path | None) -> pd.DataFrame:
    """load optional table
    empty frame when absent or headerless"""
    if path is None or not path.exists():
        return pd.DataFrame()

    # disabled optional steps may write zero-column contract files
    if path.is_file() and path.stat().st_size <= 2:
        return pd.DataFrame()

    try:
        return load_table(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_model_bundle(path: Path) -> dict[str, Any]:
    """load joblib model artifact
    normalizes direct pipeline or dict bundle"""
    loaded = joblib.load(path)

    if isinstance(loaded, dict):
        bundle = dict(loaded)
    else:
        bundle = {"pipeline": loaded}

    if "pipeline" not in bundle and "model" in bundle:
        bundle["pipeline"] = bundle["model"]

    if "pipeline" not in bundle:
        raise ValueError("Loaded model artifact has no pipeline or model entry")

    return bundle


def resolve_target_vector(
    modeling_table: pd.DataFrame,
    y_target: pd.DataFrame,
    cfg: dict[str, Any],
    bundle: dict[str, Any],
) -> pd.Series:
    """resolve target vector
    y_target file preferred when present"""
    target_col = str(bundle.get("target_col") or get_cfg(cfg, "target_col", "fused_prob_responder"))

    if not y_target.empty:
        if target_col in y_target.columns:
            return safe_numeric_series(y_target[target_col])

        if "target" in y_target.columns:
            return safe_numeric_series(y_target["target"])

        if "y" in y_target.columns:
            return safe_numeric_series(y_target["y"])

        if y_target.shape[1] == 1:
            return safe_numeric_series(y_target.iloc[:, 0])

    if target_col in modeling_table.columns:
        return safe_numeric_series(modeling_table[target_col])

    raise ValueError(f"Could not resolve target vector from y_target or modeling_table: {target_col}")


def load_all_inputs(cfg: dict[str, Any], model_root: Path) -> dict[str, Any]:
    """load all required inputs
    returns tables model bundle and reports"""
    paths = get_input_paths(cfg, model_root)
    input_report = build_input_file_report(paths)

    modeling_table = load_table(paths["modeling_table"])  # type: ignore[arg-type]
    x_features = load_table(paths["x_features"])  # type: ignore[arg-type]
    feature_manifest = load_table(paths["feature_manifest"])  # type: ignore[arg-type]
    y_target = load_optional_table(paths["y_target"])
    sample_split = load_optional_table(paths["sample_split"])
    global_importance = load_optional_table(paths["global_feature_importance"])
    global_predictions = load_optional_table(paths["global_predictions"])
    global_metrics = load_optional_table(paths["global_metrics"])
    per_treatment_summary = load_optional_table(paths["per_treatment_summary"])
    per_treatment_importance = load_optional_table(paths["per_treatment_importance"])
    bundle = load_model_bundle(paths["global_model"])  # type: ignore[arg-type]

    if len(modeling_table) != len(x_features):
        raise ValueError(
            f"modeling_table and X_features have different rows: {len(modeling_table)} versus {len(x_features)}"
        )

    y = resolve_target_vector(modeling_table, y_target, cfg, bundle)

    if len(y) != len(modeling_table):
        raise ValueError(f"Target length does not match modeling table: {len(y)} versus {len(modeling_table)}")

    return {
        "paths": paths,
        "input_report": input_report,
        "modeling_table": modeling_table.reset_index(drop=True),
        "x_features_raw": x_features.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "feature_manifest": feature_manifest,
        "sample_split": sample_split,
        "global_importance": global_importance,
        "global_predictions": global_predictions,
        "global_metrics": global_metrics,
        "per_treatment_summary": per_treatment_summary,
        "per_treatment_importance": per_treatment_importance,
        "bundle": bundle,
    }


# ============================================================
# FEATURE AND SPLIT PREPARATION
# ============================================================


def get_model_feature_names(bundle: dict[str, Any], x_features: pd.DataFrame) -> list[str]:
    """resolve feature names used by model
    bundle list preferred"""
    names = bundle.get("feature_names") or bundle.get("feature_columns")

    if names is not None:
        return [str(x) for x in list(names)]

    return [str(x) for x in x_features.columns]


def align_x_to_model_features(x_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """align X columns to fitted model
    errors on missing columns"""
    missing = [col for col in feature_names if col not in x_features.columns]

    if missing:
        raise ValueError("X_features is missing model columns: " + compact_list(missing, max_items=20))

    out = x_features[feature_names].copy()
    return safe_numeric_frame(out)


def prepare_target_for_task(y: pd.Series, cfg: dict[str, Any], task: str) -> pd.Series:
    """prepare target by task
    threshold for classification only"""
    y_num = safe_numeric_series(y)

    if task == "classification":
        threshold = float(get_cfg(cfg, "binary_threshold", get_cfg(cfg, "classification_threshold", 0.5)))
        return (y_num >= threshold).astype(int)

    return y_num.astype(float)


def attach_split(
    modeling_table: pd.DataFrame,
    sample_split: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """attach split labels
    sample split preferred"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    split_group_col = str(get_cfg(cfg, "split_group_col", sample_col))

    if "split" in modeling_table.columns:
        return modeling_table["split"].astype(str).str.lower().reset_index(drop=True)

    if sample_split.empty:
        return pd.Series(["all_labeled"] * len(modeling_table), index=modeling_table.index)

    if "split" not in sample_split.columns:
        return pd.Series(["all_labeled"] * len(modeling_table), index=modeling_table.index)

    if "row_id" in sample_split.columns and "row_id" in modeling_table.columns:
        merged = modeling_table[["row_id"]].merge(
            sample_split[["row_id", "split"]],
            on="row_id",
            how="left",
        )
        return merged["split"].fillna("train").astype(str).str.lower().reset_index(drop=True)

    if split_group_col in sample_split.columns and split_group_col in modeling_table.columns:
        split_map = sample_split[[split_group_col, "split"]].drop_duplicates(split_group_col)
        merged = modeling_table[[split_group_col]].merge(split_map, on=split_group_col, how="left")
        return merged["split"].fillna("train").astype(str).str.lower().reset_index(drop=True)

    if len(sample_split) == len(modeling_table):
        return sample_split["split"].fillna("train").astype(str).str.lower().reset_index(drop=True)

    return pd.Series(["all_labeled"] * len(modeling_table), index=modeling_table.index)


def choose_explanation_indices(
    modeling_table: pd.DataFrame,
    split: pd.Series,
    y: pd.Series,
    cfg: dict[str, Any],
) -> list[int]:
    """choose rows to explain
    test rows preferred when available"""
    random_state = int(get_cfg(cfg, "random_state", 42))
    max_rows = int(get_cfg(cfg, "max_explanation_rows", 200))
    mode = str(get_cfg(cfg, "explanation_rows", "test_if_available_else_all")).lower()

    valid = y.notna()

    if mode == "test":
        mask = valid & split.eq("test")
    elif mode == "train":
        mask = valid & split.eq("train")
    elif mode == "all" or mode == "all_labeled":
        mask = valid
    else:
        test_mask = valid & split.eq("test")
        mask = test_mask if int(test_mask.sum()) > 0 else valid

    indices = np.where(mask.to_numpy())[0].tolist()

    if len(indices) <= max_rows:
        return indices

    rng = np.random.default_rng(random_state)
    sampled = rng.choice(indices, size=max_rows, replace=False)
    return sorted(int(x) for x in sampled.tolist())


def infer_task(cfg: dict[str, Any], bundle: dict[str, Any]) -> str:
    """infer modeling task
    bundle value overrides YAML"""
    task = clean_text(bundle.get("task") or get_cfg(cfg, "task", "regression")).lower()

    if task not in {"regression", "classification"}:
        task = "regression"

    return task


def identify_drug_dummy_columns(feature_names: list[str], cfg: dict[str, Any]) -> set[str]:
    """identify treatment identity features
    used for separate reporting"""
    prefixes = get_cfg(
        cfg,
        "drug_dummy_prefixes",
        ["drug_dummy__", "drug_key__", "drug__", "drug_key_", "drug_", "treatment__"],
    )

    out: set[str] = set()

    for feature in feature_names:
        low = feature.lower()
        if any(low.startswith(str(prefix).lower()) for prefix in prefixes):
            out.add(feature)

    return out


# ============================================================
# MODEL PREDICTION AND PREPROCESSING
# ============================================================


def get_pipeline(bundle: dict[str, Any]) -> Any:
    """extract fitted pipeline
    compatible with step 03 bundle"""
    if "pipeline" in bundle:
        return bundle["pipeline"]

    if "model" in bundle:
        return bundle["model"]

    raise ValueError("Model bundle has no pipeline or model")


def get_estimator_from_pipeline(pipeline: Any) -> Any:
    """extract final estimator
    pipeline or direct estimator"""
    if isinstance(pipeline, Pipeline):
        if "model" in pipeline.named_steps:
            return pipeline.named_steps["model"]

        return pipeline.steps[-1][1]

    return pipeline


def transform_for_estimator(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """apply preprocessing before estimator
    stops before final model step"""
    if not isinstance(pipeline, Pipeline):
        return X.to_numpy()

    data: Any = X.copy()

    for name, step in pipeline.steps:
        # final model is not a transformer
        if name == "model":
            break

        # if final unnamed estimator appears last, stop there
        if step is pipeline.steps[-1][1] and not hasattr(step, "transform"):
            break

        if hasattr(step, "transform"):
            data = step.transform(data)

    return np.asarray(data)


def predict_for_task(pipeline: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    """make task aware predictions
    probabilities for classifiers"""
    if len(X) == 0:
        return np.array([])

    if task == "classification" and hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)

    return np.asarray(pipeline.predict(X), dtype=float)


# ============================================================
# SHAP AND PERMUTATION EXPLANATIONS
# ============================================================


def shap_available() -> bool:
    """check SHAP import
    avoids hard dependency"""
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def normalize_shap_values(values: Any, task: str) -> np.ndarray:
    """normalize SHAP output shape
    binary class uses positive class"""
    if isinstance(values, list):
        if len(values) > 1:
            return np.asarray(values[1], dtype=float)
        return np.asarray(values[0], dtype=float)

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        if task == "classification" and arr.shape[2] > 1:
            return arr[:, :, 1]
        return arr[:, :, 0]

    return arr


def compute_shap_values(
    pipeline: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    task: str,
) -> tuple[pd.DataFrame, str]:
    """compute SHAP values
    tree explainer first, generic fallback second"""
    import shap

    estimator = get_estimator_from_pipeline(pipeline)
    X_transformed = transform_for_estimator(pipeline, X_sample)

    # tree explainer is fast for forests and boosted trees
    try:
        explainer = shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(X_transformed)
        values = normalize_shap_values(raw_values, task)
        method = "tree_shap"
    except Exception:
        # generic explainer fallback, usually slower but broader
        explainer = shap.Explainer(estimator, X_transformed)
        explanation = explainer(X_transformed)
        values = normalize_shap_values(explanation.values, task)
        method = "generic_shap"

    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature count mismatch: {values.shape[1]} values versus {len(feature_names)} feature names"
        )

    shap_df = pd.DataFrame(values, columns=feature_names, index=X_sample.index)
    return shap_df, method


def build_shap_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
    """summarize SHAP by feature
    mean absolute and signed values"""
    rows: list[dict[str, Any]] = []

    for col in shap_df.columns:
        values = safe_numeric_series(shap_df[col])
        rows.append(
            {
                "feature_name": col,
                "mean_abs_shap": float(values.abs().mean()),
                "mean_shap": float(values.mean()),
                "median_abs_shap": float(values.abs().median()),
                "max_abs_shap": float(values.abs().max()),
                "shap_nonzero_fraction": float((values.abs() > 0).mean()),
            }
        )

    out = pd.DataFrame(rows)
    out["shap_rank"] = rank_desc(out["mean_abs_shap"]).astype(int)
    return out.sort_values("shap_rank").reset_index(drop=True)


def choose_permutation_scoring(task: str, y_task: pd.Series) -> str:
    """choose permutation metric
    safe for small class sets"""
    if task == "classification":
        if int(y_task.nunique(dropna=True)) >= 2:
            return "roc_auc"
        return "accuracy"

    return "neg_mean_absolute_error"


def compute_permutation_summary(
    pipeline: Any,
    X_sample: pd.DataFrame,
    y_task: pd.Series,
    cfg: dict[str, Any],
    task: str,
) -> pd.DataFrame:
    """compute permutation importance
    model agnostic fallback"""
    n_repeats = int(get_cfg(cfg, "permutation_repeats", 10))
    random_state = int(get_cfg(cfg, "random_state", 42))
    n_jobs = int(get_cfg(cfg, "n_jobs", -1))
    scoring = str(get_cfg(cfg, "permutation_scoring", "")).strip()

    if not scoring:
        scoring = choose_permutation_scoring(task, y_task)

    mask = y_task.notna()
    X_use = X_sample.loc[mask].copy()
    y_use = y_task.loc[mask].copy()

    if len(X_use) == 0:
        return pd.DataFrame(columns=["feature_name", "permutation_importance_mean", "permutation_importance_std"])

    result = permutation_importance(
        pipeline,
        X_use,
        y_use,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    out = pd.DataFrame(
        {
            "feature_name": list(X_sample.columns),
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "permutation_scoring": scoring,
            "permutation_repeats": n_repeats,
        }
    )

    out["permutation_rank"] = rank_desc(out["permutation_importance_mean"]).astype(int)
    return out.sort_values("permutation_rank").reset_index(drop=True)


def native_feature_importance(
    pipeline: Any,
    feature_names: list[str],
    global_importance: pd.DataFrame,
) -> pd.DataFrame:
    """resolve native importance
    step 03 file preferred"""
    if not global_importance.empty and "feature_name" in global_importance.columns:
        out = global_importance.copy()
        if "importance" not in out.columns:
            imp_col = first_existing_column(out, ["feature_importance", "gain", "weight"])
            if imp_col is not None:
                out = out.rename(columns={imp_col: "importance"})
        return out

    estimator = get_estimator_from_pipeline(pipeline)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        values = np.repeat(np.nan, len(feature_names))

    out = pd.DataFrame({"feature_name": feature_names, "importance": values})
    out["importance_rank"] = rank_desc(out["importance"]) if out["importance"].notna().any() else np.nan
    return out


# ============================================================
# FEATURE ANNOTATION AND SUMMARY TABLES
# ============================================================


def resolve_manifest_feature_col(manifest: pd.DataFrame) -> str | None:
    """find manifest feature column
    common aliases supported"""
    return first_existing_column(
        manifest,
        ["feature_name", "feature_clean", "feature", "column", "column_name"],
    )


def annotate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """normalize manifest annotations
    creates feature_name column"""
    if manifest.empty:
        return pd.DataFrame(columns=["feature_name"])

    feature_col = resolve_manifest_feature_col(manifest)

    if feature_col is None:
        return pd.DataFrame(columns=["feature_name"])

    out = manifest.copy()
    out["feature_name"] = out[feature_col].astype(str).map(clean_text)

    keep_cols = ["feature_name"]

    for col in [
        "feature_original",
        "feature_group",
        "feature_axis",
        "feature_type",
        "source",
        "status",
        "included",
        "kept_for_modeling",
        "missing_fraction",
        "nonmissing_count",
        "training_missing_fraction",
        "training_nonmissing_count",
        "training_unique_values",
    ]:
        if col in out.columns and col not in keep_cols:
            keep_cols.append(col)

    return out[keep_cols].drop_duplicates("feature_name")


def add_feature_classes(
    feature_table: pd.DataFrame,
    drug_dummy_cols: set[str],
) -> pd.DataFrame:
    """add spatial versus drug class
    keeps unannotated features readable"""
    out = feature_table.copy()

    out["feature_class"] = np.where(out["feature_name"].isin(drug_dummy_cols), "drug_identity", "spatial_feature")

    if "feature_group" not in out.columns:
        out["feature_group"] = ""

    if "feature_axis" not in out.columns:
        out["feature_axis"] = ""

    out.loc[out["feature_class"] == "drug_identity", "feature_group"] = "drug_identity"
    out.loc[out["feature_class"] == "drug_identity", "feature_axis"] = "treatment_context"

    out["feature_group"] = out["feature_group"].replace("", "unannotated")
    out["feature_axis"] = out["feature_axis"].replace("", "unannotated")

    return out


def build_global_feature_table(
    feature_names: list[str],
    manifest: pd.DataFrame,
    shap_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    native_importance: pd.DataFrame,
    drug_dummy_cols: set[str],
) -> pd.DataFrame:
    """combine all feature explanations
    SHAP takes primary priority"""
    base = pd.DataFrame({"feature_name": feature_names})
    annotations = annotate_manifest(manifest)

    out = base.merge(annotations, on="feature_name", how="left")

    if not shap_summary.empty:
        out = out.merge(shap_summary, on="feature_name", how="left")

    if not permutation_summary.empty:
        out = out.merge(permutation_summary, on="feature_name", how="left")

    if not native_importance.empty and "feature_name" in native_importance.columns:
        native = native_importance.copy()
        native_cols = [c for c in ["feature_name", "importance", "importance_rank", "is_drug_dummy"] if c in native.columns]
        out = out.merge(native[native_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

    out = add_feature_classes(out, drug_dummy_cols)

    out["primary_explanation_score"] = np.nan
    out["primary_explanation_method"] = "none"

    if "mean_abs_shap" in out.columns and out["mean_abs_shap"].notna().any():
        out["primary_explanation_score"] = out["mean_abs_shap"]
        out["primary_explanation_method"] = "mean_abs_shap"
    elif "permutation_importance_mean" in out.columns and out["permutation_importance_mean"].notna().any():
        out["primary_explanation_score"] = out["permutation_importance_mean"]
        out["primary_explanation_method"] = "permutation_importance"
    elif "importance" in out.columns and out["importance"].notna().any():
        out["primary_explanation_score"] = out["importance"]
        out["primary_explanation_method"] = "model_native_importance"

    out["primary_explanation_rank"] = rank_desc(out["primary_explanation_score"])

    return out.sort_values(
        ["primary_explanation_rank", "feature_name"],
        ascending=[True, True],
    ).reset_index(drop=True)


def summarize_by_group(feature_table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """summarize explanation by groups
    sums primary scores"""
    if feature_table.empty:
        return pd.DataFrame()

    score_col = "primary_explanation_score"

    work = feature_table.copy()
    work[score_col] = safe_numeric_series(work[score_col]).fillna(0.0)

    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_features=("feature_name", "nunique"),
            total_explanation_score=(score_col, "sum"),
            mean_explanation_score=(score_col, "mean"),
            max_explanation_score=(score_col, "max"),
        )
        .reset_index()
    )

    total = float(summary["total_explanation_score"].sum())
    summary["fraction_of_total_score"] = summary["total_explanation_score"].map(lambda x: safe_divide(x, total))
    summary["rank"] = rank_desc(summary["total_explanation_score"]).astype(int)

    return summary.sort_values("rank").reset_index(drop=True)


def build_spatial_vs_drug_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    """summarize spatial and drug identity contribution
    separate model feature classes"""
    return summarize_by_group(feature_table, ["feature_class"])


# ============================================================
# LOCAL AND TREATMENT SHAP TABLES
# ============================================================


def metadata_columns_for_output(modeling_table: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """choose metadata columns for explanation rows
    keeps outputs interpretable"""
    candidates = [
        "row_id",
        str(get_cfg(cfg, "sample_col", "sample_id")),
        str(get_cfg(cfg, "slide_col", "slide_id")),
        str(get_cfg(cfg, "drug_col", "drug")),
        str(get_cfg(cfg, "drug_key_col", "drug_key")),
        "split",
        str(get_cfg(cfg, "target_col", "fused_prob_responder")),
        "response_binary",
        "baseline_response",
        "residual_response",
        "modality_used",
        "fused_confidence",
        "dataset_id",
        "cancer_type",
    ]

    return [col for col in candidates if col in modeling_table.columns]


def build_local_shap_table(
    shap_df: pd.DataFrame,
    X_sample: pd.DataFrame,
    modeling_sample: pd.DataFrame,
    feature_table: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """build local top feature explanations
    top absolute SHAP values per row"""
    if shap_df.empty:
        return pd.DataFrame()

    top_n = int(get_cfg(cfg, "top_local_features_per_row", 20))
    metadata_cols = metadata_columns_for_output(modeling_sample, cfg)
    annotation_cols = [
        col for col in ["feature_name", "feature_original", "feature_group", "feature_axis", "feature_class"]
        if col in feature_table.columns
    ]
    annotations = feature_table[annotation_cols].drop_duplicates("feature_name") if annotation_cols else pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for row_index in shap_df.index:
        values = shap_df.loc[row_index]
        ordered = values.abs().sort_values(ascending=False).head(top_n).index.tolist()
        meta = modeling_sample.loc[row_index, metadata_cols].to_dict() if metadata_cols else {}

        for rank, feature in enumerate(ordered, start=1):
            rows.append(
                {
                    **meta,
                    "explanation_row_index": int(row_index),
                    "local_rank": int(rank),
                    "feature_name": feature,
                    "feature_value": X_sample.loc[row_index, feature] if feature in X_sample.columns else np.nan,
                    "shap_value": float(shap_df.loc[row_index, feature]),
                    "abs_shap_value": float(abs(shap_df.loc[row_index, feature])),
                    "effect_direction": "positive" if shap_df.loc[row_index, feature] >= 0 else "negative",
                }
            )

    out = pd.DataFrame(rows)

    if not annotations.empty and not out.empty:
        out = out.merge(annotations, on="feature_name", how="left")

    return out


def summarize_local_by_sample(local_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize local explanations by sample
    aggregates absolute SHAP"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if local_df.empty or sample_col not in local_df.columns:
        return pd.DataFrame()

    summary = (
        local_df.groupby([sample_col, "feature_name"], dropna=False)
        .agg(
            mean_abs_shap=("abs_shap_value", "mean"),
            mean_shap=("shap_value", "mean"),
            n_rows=("feature_name", "size"),
        )
        .reset_index()
    )

    summary["rank_within_sample"] = summary.groupby(sample_col)["mean_abs_shap"].rank(
        ascending=False,
        method="dense",
    )

    max_rank = int(get_cfg(cfg, "top_features_per_sample", 25))
    summary = summary[summary["rank_within_sample"] <= max_rank].copy()

    return summary.sort_values([sample_col, "rank_within_sample"]).reset_index(drop=True)


def summarize_shap_by_treatment(
    shap_df: pd.DataFrame,
    modeling_sample: pd.DataFrame,
    feature_table: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """summarize SHAP by treatment
    top features within each drug"""
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    top_n = int(get_cfg(cfg, "top_features_per_treatment_explanation", 30))

    if shap_df.empty or drug_key_col not in modeling_sample.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for drug_key, idx in modeling_sample.groupby(drug_key_col).groups.items():
        idx_list = list(idx)
        sub = shap_df.loc[idx_list]
        mean_abs = sub.abs().mean(axis=0)
        mean_signed = sub.mean(axis=0)
        top_features = mean_abs.sort_values(ascending=False).head(top_n)
        drug_name = modeling_sample.loc[idx_list, drug_col].iloc[0] if drug_col in modeling_sample.columns else drug_key

        for rank, (feature, value) in enumerate(top_features.items(), start=1):
            rows.append(
                {
                    "drug_key": drug_key,
                    "drug": drug_name,
                    "feature_name": feature,
                    "rank_within_treatment": int(rank),
                    "mean_abs_shap": float(value),
                    "mean_shap": float(mean_signed[feature]),
                    "n_explained_rows": int(len(idx_list)),
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    annotation_cols = [
        col for col in ["feature_name", "feature_original", "feature_group", "feature_axis", "feature_class", "primary_explanation_rank"]
        if col in feature_table.columns
    ]

    if annotation_cols:
        out = out.merge(feature_table[annotation_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

    return out.sort_values(["drug_key", "rank_within_treatment"]).reset_index(drop=True)


# ============================================================
# PER TREATMENT MODEL INTEGRATION
# ============================================================


def normalize_per_treatment_importance(per_imp: pd.DataFrame) -> pd.DataFrame:
    """normalize step 04 importance table
    returns empty when absent"""
    if per_imp.empty:
        return pd.DataFrame()

    out = per_imp.copy()

    if "feature_name" not in out.columns:
        feature_col = first_existing_column(out, ["feature", "column", "column_name"])
        if feature_col is not None:
            out = out.rename(columns={feature_col: "feature_name"})

    if "importance" not in out.columns:
        imp_col = first_existing_column(out, ["feature_importance", "mean_abs_shap", "gain", "weight"])
        if imp_col is not None:
            out = out.rename(columns={imp_col: "importance"})

    if "drug_key" not in out.columns:
        drug_key_col = first_existing_column(out, ["treatment_key", "drug_key_norm"])
        if drug_key_col is not None:
            out = out.rename(columns={drug_key_col: "drug_key"})

    required = ["drug_key", "feature_name"]
    if any(col not in out.columns for col in required):
        return pd.DataFrame()

    if "importance" not in out.columns:
        out["importance"] = np.nan

    return out


def build_per_treatment_summary(
    per_imp: pd.DataFrame,
    per_summary: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """combine step 04 and global annotations
    top treatment specific features"""
    norm = normalize_per_treatment_importance(per_imp)

    if norm.empty:
        cols = [
            "drug_key",
            "drug",
            "feature_name",
            "importance",
            "feature_original",
            "feature_group",
            "feature_axis",
            "primary_explanation_rank",
        ]
        return pd.DataFrame(columns=cols)

    annotation_cols = [
        col for col in ["feature_name", "feature_original", "feature_group", "feature_axis", "feature_class", "primary_explanation_rank", "primary_explanation_score"]
        if col in feature_table.columns
    ]

    out = norm.merge(feature_table[annotation_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

    if not per_summary.empty and "drug_key" in per_summary.columns:
        keep = [col for col in ["drug_key", "status", "reason", "n_total", "n_train", "n_test"] if col in per_summary.columns]
        out = out.merge(per_summary[keep].drop_duplicates("drug_key"), on="drug_key", how="left")

    return out


def build_per_treatment_concordance(
    per_treatment_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """compare treatment and global ranks
    overlap and rank correlation"""
    if per_treatment_table.empty:
        return pd.DataFrame(
            columns=[
                "drug_key",
                "n_treatment_features",
                "top_global_overlap_count",
                "top_global_overlap_fraction",
                "spearman_treatment_vs_global_rank",
            ]
        )

    top_n = int(get_cfg(cfg, "concordance_top_n", 25))
    global_top = set(
        feature_table.sort_values("primary_explanation_rank").head(top_n)["feature_name"].astype(str).tolist()
    )

    rows: list[dict[str, Any]] = []

    work = per_treatment_table.copy()

    if "rank" not in work.columns:
        if "importance" in work.columns:
            work["rank"] = work.groupby("drug_key")["importance"].rank(ascending=False, method="dense")
        else:
            work["rank"] = np.nan

    for drug_key, group in work.groupby("drug_key", dropna=False):
        group_top = group.sort_values("rank").head(top_n).copy()
        treatment_features = set(group_top["feature_name"].astype(str).tolist())
        overlap = treatment_features & global_top

        corr = safe_spearman(group["rank"], group["primary_explanation_rank"]) if "primary_explanation_rank" in group.columns else np.nan

        rows.append(
            {
                "drug_key": drug_key,
                "n_treatment_features": int(group["feature_name"].nunique()),
                "top_n": int(top_n),
                "top_global_overlap_count": int(len(overlap)),
                "top_global_overlap_fraction": safe_divide(len(overlap), max(1, len(treatment_features))),
                "overlapping_features": compact_list(sorted(overlap), max_items=20),
                "spearman_treatment_vs_global_rank": corr,
            }
        )

    return pd.DataFrame(rows).sort_values("drug_key").reset_index(drop=True)


# ============================================================
# PLOTS
# ============================================================


def plot_available() -> bool:
    """check matplotlib import
    avoids hard dependency failures"""
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception:
        return False


def make_bar_plot(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, top_n: int = 25) -> None:
    """write horizontal bar plot
    one figure only"""
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    import matplotlib.pyplot as plt

    work = df[[label_col, value_col]].dropna().copy()
    work[value_col] = safe_numeric_series(work[value_col])
    work = work.sort_values(value_col, ascending=False).head(top_n)

    if work.empty:
        return

    labels = work[label_col].astype(str).iloc[::-1]
    values = work[value_col].astype(float).iloc[::-1]

    height = max(4.5, min(14.0, 0.35 * len(work) + 2.0))
    plt.figure(figsize=(10, height))
    plt.barh(labels, values)
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_plots(out_dir: Path, feature_table: pd.DataFrame, group_summary: pd.DataFrame, axis_summary: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """write explanation plots
    returns generated paths"""
    if not bool_like(get_cfg(cfg, "write_explanation_plots", True)):
        return []

    if not plot_available():
        return []

    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    top_n = int(get_cfg(cfg, "plot_top_n", 25))

    outputs: list[str] = []

    p1 = fig_dir / "top_global_features.png"
    make_bar_plot(
        feature_table,
        label_col="feature_label",
        value_col="primary_explanation_score",
        title="Top global model explanations",
        path=p1,
        top_n=top_n,
    )
    if p1.exists():
        outputs.append(str(p1))

    p2 = fig_dir / "top_feature_groups.png"
    make_bar_plot(
        group_summary,
        label_col="feature_group",
        value_col="total_explanation_score",
        title="Top explained feature groups",
        path=p2,
        top_n=top_n,
    )
    if p2.exists():
        outputs.append(str(p2))

    p3 = fig_dir / "top_feature_axes.png"
    make_bar_plot(
        axis_summary,
        label_col="feature_axis",
        value_col="total_explanation_score",
        title="Top explained feature axes",
        path=p3,
        top_n=top_n,
    )
    if p3.exists():
        outputs.append(str(p3))

    return outputs


# ============================================================
# SUMMARY TEXT
# ============================================================


def add_feature_labels(feature_table: pd.DataFrame) -> pd.DataFrame:
    """add readable labels
    original biology name preferred"""
    out = feature_table.copy()

    if "feature_original" in out.columns:
        out["feature_label"] = out["feature_original"].fillna("").astype(str)
        out.loc[out["feature_label"].str.strip() == "", "feature_label"] = out["feature_name"]
    else:
        out["feature_label"] = out["feature_name"]

    return out


def build_model_performance_summary(pipeline: Any, X: pd.DataFrame, y_task: pd.Series, split: pd.Series, task: str) -> pd.DataFrame:
    """compute simple explain set metrics
    sanity check only"""
    rows: list[dict[str, Any]] = []

    for split_name in ["train", "test", "all_labeled"]:
        if split_name == "all_labeled":
            mask = y_task.notna()
        else:
            mask = y_task.notna() & split.eq(split_name)

        if int(mask.sum()) == 0:
            continue

        pred = predict_for_task(pipeline, X.loc[mask], task)
        y_true = y_task.loc[mask]

        if task == "classification":
            if int(y_true.nunique()) >= 2:
                metric = float(roc_auc_score(y_true, pred))
                metric_name = "roc_auc"
            else:
                metric = float(accuracy_score(y_true, (pred >= 0.5).astype(int)))
                metric_name = "accuracy"
        else:
            metric = float(r2_score(y_true, pred)) if len(y_true) >= 2 else np.nan
            metric_name = "r2"

        rows.append({"split": split_name, "n_rows": int(mask.sum()), "metric_name": metric_name, "metric_value": metric})

    return pd.DataFrame(rows)


def write_summary_text(
    path: Path,
    cfg: dict[str, Any],
    inputs: dict[str, Any],
    feature_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    spatial_vs_drug: pd.DataFrame,
    per_concordance: pd.DataFrame,
    performance_summary: pd.DataFrame,
    explanation_method: str,
    shap_status: str,
    permutation_status: str,
    plot_paths: list[str],
) -> None:
    """write human explanation summary
    concise run report"""
    loaded = inputs
    modeling_table = loaded["modeling_table"]
    X = loaded["X"]
    split = loaded["split"]

    lines: list[str] = []
    lines.append("Spatial response model explanation summary")
    lines.append("")
    lines.append("Run settings")
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  task: {loaded['task']}")
    lines.append(f"  explanation_method: {explanation_method}")
    lines.append(f"  shap_status: {shap_status}")
    lines.append(f"  permutation_status: {permutation_status}")
    lines.append("")

    lines.append("Input shape")
    lines.append(f"  modeling rows: {len(modeling_table)}")
    lines.append(f"  model features: {X.shape[1]}")
    lines.append(f"  explained rows: {len(loaded['explanation_indices'])}")
    lines.append(f"  split labels: {compact_list(sorted(split.unique().tolist()))}")
    lines.append("")

    if not performance_summary.empty:
        lines.append("Prediction sanity metrics")
        for _, row in performance_summary.iterrows():
            lines.append(
                f"  {row['split']}: n={row['n_rows']}; {row['metric_name']}={row['metric_value']:.4f}"
            )
        lines.append("")

    lines.append("Top global features")
    top_features = feature_table.head(20)
    for _, row in top_features.iterrows():
        label = clean_text(row.get("feature_label", row.get("feature_name", "")))
        score = row.get("primary_explanation_score", np.nan)
        group = clean_text(row.get("feature_group", ""))
        axis = clean_text(row.get("feature_axis", ""))
        lines.append(f"  {row['primary_explanation_rank']:.0f}. {label}: score={score:.6g}; group={group}; axis={axis}")
    lines.append("")

    if not group_summary.empty:
        lines.append("Top feature groups")
        for _, row in group_summary.head(12).iterrows():
            lines.append(
                f"  {row['feature_group']}: total_score={row['total_explanation_score']:.6g}; fraction={row['fraction_of_total_score']:.3f}"
            )
        lines.append("")

    if not spatial_vs_drug.empty:
        lines.append("Spatial versus treatment identity")
        for _, row in spatial_vs_drug.iterrows():
            lines.append(
                f"  {row['feature_class']}: n_features={row['n_features']}; fraction={row['fraction_of_total_score']:.3f}"
            )
        lines.append("")

    if not per_concordance.empty:
        lines.append("Per treatment concordance")
        lines.append(f"  treatments summarized: {per_concordance.shape[0]}")
        mean_overlap = safe_numeric_series(per_concordance["top_global_overlap_fraction"]).mean()
        lines.append(f"  mean top feature overlap with global model: {mean_overlap:.3f}")
        lines.append("")

    if plot_paths:
        lines.append("Plots")
        for p in plot_paths:
            lines.append(f"  {p}")
        lines.append("")

    lines.append("Interpretation notes")
    lines.append("  Explanations describe model behavior, not causal drug response biology")
    lines.append("  Drug identity features are separated from spatial biology features")
    lines.append("  Small test runs should be used as pipeline checks, not final inference")

    write_text(path, lines)


# ============================================================
# MAIN WORKFLOW
# ============================================================


def run_explanation(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """run model explanation workflow
    returns loaded state and outputs"""
    config_path = Path(args.config)
    cfg = load_config(config_path)
    model_root = infer_model_root(config_path, cfg)
    out_dir = get_output_step_dir(cfg, model_root, "model_explanations", DEFAULT_EXPLANATION_SUBDIR)
    ensure_dir(out_dir)

    loaded = load_all_inputs(cfg, model_root)
    bundle = loaded["bundle"]
    pipeline = get_pipeline(bundle)
    task = infer_task(cfg, bundle)

    feature_names = get_model_feature_names(bundle, loaded["x_features_raw"])
    X = align_x_to_model_features(loaded["x_features_raw"], feature_names)
    y = loaded["y"]
    y_task = prepare_target_for_task(y, cfg, task)
    split = attach_split(loaded["modeling_table"], loaded["sample_split"], cfg)
    explanation_indices = choose_explanation_indices(loaded["modeling_table"], split, y, cfg)

    X_sample = X.iloc[explanation_indices].copy()
    X_sample.index = explanation_indices
    y_sample_task = y_task.iloc[explanation_indices].copy()
    y_sample_task.index = explanation_indices
    modeling_sample = loaded["modeling_table"].iloc[explanation_indices].copy()
    modeling_sample.index = explanation_indices
    modeling_sample["split"] = split.iloc[explanation_indices].to_numpy()

    drug_dummy_cols = identify_drug_dummy_columns(feature_names, cfg)
    shap_df = pd.DataFrame()
    shap_summary = pd.DataFrame()
    permutation_summary = pd.DataFrame()
    shap_status = "not_requested"
    permutation_status = "not_requested"
    explanation_method = "model_native_importance"

    use_shap = bool_like(get_cfg(cfg, "use_shap", True)) and not args.no_shap

    if use_shap and shap_available():
        try:
            shap_df, shap_method = compute_shap_values(pipeline, X_sample, feature_names, task)
            shap_summary = build_shap_summary(shap_df)
            shap_status = f"success:{shap_method}"
            explanation_method = "shap"
        except Exception as exc:
            shap_status = f"failed:{type(exc).__name__}:{exc}"
    elif use_shap:
        shap_status = "failed:shap_not_installed"

    run_permutation = bool_like(get_cfg(cfg, "run_permutation_importance", False)) or args.force_permutation

    if run_permutation or shap_df.empty:
        try:
            permutation_summary = compute_permutation_summary(pipeline, X_sample, y_sample_task, cfg, task)
            permutation_status = "success"
            if shap_df.empty:
                explanation_method = "permutation_importance"
        except Exception as exc:
            permutation_status = f"failed:{type(exc).__name__}:{exc}"

    native_importance = native_feature_importance(pipeline, feature_names, loaded["global_importance"])

    feature_table = build_global_feature_table(
        feature_names=feature_names,
        manifest=loaded["feature_manifest"],
        shap_summary=shap_summary,
        permutation_summary=permutation_summary,
        native_importance=native_importance,
        drug_dummy_cols=drug_dummy_cols,
    )
    feature_table = add_feature_labels(feature_table)

    group_summary = summarize_by_group(feature_table, ["feature_group"])
    axis_summary = summarize_by_group(feature_table, ["feature_axis"])
    class_summary = build_spatial_vs_drug_summary(feature_table)
    group_axis_summary = summarize_by_group(feature_table, ["feature_group", "feature_axis", "feature_class"])

    local_shap = build_local_shap_table(shap_df, X_sample, modeling_sample, feature_table, cfg)
    sample_summary = summarize_local_by_sample(local_shap, cfg)
    treatment_shap_summary = summarize_shap_by_treatment(shap_df, modeling_sample, feature_table, cfg)

    per_treatment_table = build_per_treatment_summary(
        loaded["per_treatment_importance"],
        loaded["per_treatment_summary"],
        feature_table,
    )
    per_concordance = build_per_treatment_concordance(per_treatment_table, feature_table, cfg)

    performance_summary = build_model_performance_summary(pipeline, X, y_task, split, task)

    # write machine readable outputs
    loaded["input_report"].to_csv(out_dir / "input_file_report.tsv", sep="\t", index=False)
    feature_table.to_csv(out_dir / "global_feature_explanations.tsv", sep="\t", index=False)
    group_summary.to_csv(out_dir / "global_feature_group_summary.tsv", sep="\t", index=False)
    axis_summary.to_csv(out_dir / "global_feature_axis_summary.tsv", sep="\t", index=False)
    group_axis_summary.to_csv(out_dir / "global_feature_group_axis_summary.tsv", sep="\t", index=False)
    class_summary.to_csv(out_dir / "global_spatial_vs_drug_summary.tsv", sep="\t", index=False)
    local_shap.to_csv(out_dir / "global_local_explanations_top.tsv", sep="\t", index=False)
    sample_summary.to_csv(out_dir / "global_sample_explanation_summary.tsv", sep="\t", index=False)
    treatment_shap_summary.to_csv(out_dir / "global_treatment_explanation_summary.tsv", sep="\t", index=False)
    per_treatment_table.to_csv(out_dir / "per_treatment_explanation_summary.tsv", sep="\t", index=False)
    per_concordance.to_csv(out_dir / "per_treatment_global_concordance.tsv", sep="\t", index=False)
    performance_summary.to_csv(out_dir / "prediction_sanity_metrics.tsv", sep="\t", index=False)

    if not shap_df.empty and bool_like(get_cfg(cfg, "write_shap_matrix", False)):
        shap_df.to_csv(out_dir / "global_shap_values_matrix.tsv", sep="\t", index=True, index_label="row_index")

    if not permutation_summary.empty:
        permutation_summary.to_csv(out_dir / "global_permutation_importance.tsv", sep="\t", index=False)

    if not native_importance.empty:
        native_importance.to_csv(out_dir / "global_native_feature_importance.tsv", sep="\t", index=False)

    plot_paths = write_plots(out_dir, feature_table, group_summary, axis_summary, cfg)

    loaded_state = {
        **loaded,
        "X": X,
        "y_task": y_task,
        "split": split,
        "task": task,
        "out_dir": out_dir,
        "model_root": model_root,
        "explanation_indices": explanation_indices,
    }

    write_summary_text(
        path=out_dir / "model_explanation_summary.txt",
        cfg=cfg,
        inputs=loaded_state,
        feature_table=feature_table,
        group_summary=group_summary,
        spatial_vs_drug=class_summary,
        per_concordance=per_concordance,
        performance_summary=performance_summary,
        explanation_method=explanation_method,
        shap_status=shap_status,
        permutation_status=permutation_status,
        plot_paths=plot_paths,
    )

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "task": task,
        "n_rows": int(len(loaded["modeling_table"])),
        "n_features": int(X.shape[1]),
        "n_explained_rows": int(len(explanation_indices)),
        "explanation_method": explanation_method,
        "shap_status": shap_status,
        "permutation_status": permutation_status,
        "input_paths": {key: str(path) if path is not None else "" for key, path in loaded["paths"].items()},
        "outputs": {
            "global_feature_explanations": str(out_dir / "global_feature_explanations.tsv"),
            "global_local_explanations_top": str(out_dir / "global_local_explanations_top.tsv"),
            "model_explanation_summary": str(out_dir / "model_explanation_summary.txt"),
        },
    }
    save_json(run_info, out_dir / "run_config.json")

    outputs = {
        "out_dir": out_dir,
        "feature_table": feature_table,
        "group_summary": group_summary,
        "local_shap": local_shap,
        "sample_summary": sample_summary,
        "treatment_summary": treatment_shap_summary,
        "per_treatment_table": per_treatment_table,
        "per_concordance": per_concordance,
        "explanation_method": explanation_method,
        "shap_status": shap_status,
        "permutation_status": permutation_status,
    }

    return loaded_state, outputs


def main() -> int:
    """run explanation step
    writes reports and exits"""
    args = parse_args()

    print("Explaining spatial response model")
    print(f"Config: {Path(args.config)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        loaded, outputs = run_explanation(args)

    out_dir = outputs["out_dir"]
    feature_table = outputs["feature_table"]

    print("\nDONE")
    print(f"Rows loaded: {len(loaded['modeling_table']):,}")
    print(f"Model features: {loaded['X'].shape[1]:,}")
    print(f"Explained rows: {len(loaded['explanation_indices']):,}")
    print(f"Explanation method: {outputs['explanation_method']}")
    print(f"SHAP status: {outputs['shap_status']}")
    print(f"Permutation status: {outputs['permutation_status']}")
    print(f"Top feature: {feature_table.iloc[0]['feature_label'] if not feature_table.empty else 'none'}")
    print(f"Wrote: {out_dir / 'global_feature_explanations.tsv'}")
    print(f"Wrote: {out_dir / 'global_feature_group_summary.tsv'}")
    print(f"Wrote: {out_dir / 'model_explanation_summary.txt'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
