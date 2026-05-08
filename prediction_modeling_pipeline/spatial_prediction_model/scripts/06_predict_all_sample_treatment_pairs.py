"""
Script:
    06_predict_all_sample_treatment_pairs.py

Purpose:
    Predict spatial response scores for every requested sample treatment pair.

Role:
    Sixth step in spatial_prediction_model.
    Consumes the trained global model from step 03.
    Reconstructs the model feature matrix for unlabeled and labeled spatial samples.
    Applies the global model to all selected treatments.
    Writes sample treatment predictions, rankings, summaries, and teacher overlap checks.

Pipeline position:
    01_validate_prediction_inputs.py
        validates the teacher_builder handoff files.

    02_build_spatial_modeling_dataset.py
        builds the leakage-safe modeling table, feature matrix, target vector,
        feature manifest, and grouped sample split.

    03_train_global_spatial_response_model.py
        trains the pooled spatial response model and writes model.joblib.

    04_train_per_treatment_models.py
        optionally trains treatment-specific models.

    05_explain_spatial_response_model.py
        explains model behavior with feature importance and SHAP outputs.

    06_predict_all_sample_treatment_pairs.py
        uses the global model to score sample treatment combinations beyond the
        labeled teacher rows.

Prediction idea:
    The global model learned:
        spatial features + treatment identity -> fused teacher response

    This script reconstructs the same feature columns used by that model.
    For each selected sample and treatment, it supplies:
        1) spatial features for the sample
        2) one-hot treatment identity features, when the model used them

    For the current 10-sample smoke run, YAML usually requests only the labeled
    test samples. For the full 102-sample run, YAML should switch to all spatial
    samples without changing code.

Expected inputs:
    outputs/<run>/02_modeling_dataset/model_feature_manifest.csv
        Maps clean feature names, such as f_0, back to original spatial columns.

    outputs/<run>/02_modeling_dataset/modeling_table.tsv
        Labeled sample treatment rows used to train the model.
        Used here to recover treatment list and teacher overlap labels.

    outputs/<run>/02_modeling_dataset/X_features.csv
        Training feature matrix.
        Used here to recover exact drug dummy column mapping.

    outputs/<run>/03_global_model/model.joblib
        Trained model bundle from step 03.

    teacher_builder/outputs/05_prediction_ready_teacher/model_input_numeric.csv
        Spatial feature table for all available spatial samples.

Primary outputs:
    outputs/<run>/06_all_sample_predictions/all_sample_treatment_predictions.tsv
        One row per predicted sample treatment pair.

    outputs/<run>/06_all_sample_predictions/prediction_summary_by_sample.tsv
        Sample-level prediction distribution summary.

    outputs/<run>/06_all_sample_predictions/prediction_summary_by_treatment.tsv
        Treatment-level prediction distribution summary.

    outputs/<run>/06_all_sample_predictions/top_treatment_per_sample.tsv
        Highest predicted treatment for every sample.

    outputs/<run>/06_all_sample_predictions/teacher_labeled_prediction_comparison.tsv
        Prediction versus teacher label on labeled rows, when available.

    outputs/<run>/06_all_sample_predictions/prediction_matrix_sample_by_treatment.tsv
        Wide matrix of predicted response values.

    outputs/<run>/06_all_sample_predictions/run_config.json
        Reproducibility record.

Design contract:
    YAML driven paths, columns, sample selection mode, treatment source, and
    output behavior.
    No hard-coded sample counts.
    The same code supports the 10-sample test run and future 102-sample run.
    Model feature order is taken from model.joblib and is treated as mandatory.
    Missing spatial feature values are left as NaN, because the trained model
    pipeline includes median imputation.

Notes:
    This script does not retrain the model.
    This script does not recompute SHAP.
    Step 05 may already be complete, but step 06 only needs step 02 and step 03
    artifacts to generate predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
import warnings

# governed smoke patch: ignore pandas fragmentation warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import yaml


SCRIPT_NAME = "06_predict_all_sample_treatment_pairs.py"
STEP_NAME = "06_predict_all_sample_treatment_pairs"
DEFAULT_MODELING_DATASET_SUBDIR = "02_modeling_dataset"
DEFAULT_GLOBAL_MODEL_SUBDIR = "03_global_model"
DEFAULT_MODEL_EXPLANATION_SUBDIR = "05_model_explanation"
DEFAULT_OUTPUT_SUBDIR = "06_all_sample_predictions"


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI arguments
    config path plus optional force flag"""
    parser = argparse.ArgumentParser(description="Predict all sample treatment pairs")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when run_all_sample_predictions is false",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config
    UTF8 with BOM tolerant"""
    if not path.exists():
        raise FileNotFoundError(path)

    # utf-8-sig handles occasional Windows BOM in YAML files
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """read config value
    simple fallback wrapper"""
    return cfg[key] if key in cfg else default


def clean_text(value: Any) -> str:
    """clean scalar text
    empty string for missing"""
    if pd.isna(value):
        return ""

    return str(value).strip()


def bool_like(value: Any) -> bool:
    """parse truth-like values
    tolerant for YAML or text"""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on", "included"}


def none_like(value: Any) -> bool:
    """detect null-like config value
    supports YAML and text nulls"""
    if value is None:
        return True

    text = clean_text(value).lower()
    return text in {"", "none", "null", "nan"}


def is_windows_absolute_path(value: str) -> bool:
    """detect Windows drive-letter path
    useful when running path logic cross-platform"""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(value)))


def is_absolute_path_like(value: str) -> bool:
    """check absolute path
    supports native and Windows absolute paths"""
    text = str(value)
    return Path(text).is_absolute() or is_windows_absolute_path(text)


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve input path
    absolute paths preserved"""
    if value in [None, ""]:
        return None

    path = Path(str(value))

    if is_absolute_path_like(str(path)):
        return path

    return project_dir / path


def get_output_root(cfg: dict[str, Any]) -> Path:
    """resolve output root
    expected to point to current run folder"""
    return Path(str(cfg["output_root"]))


def get_output_subdir(cfg: dict[str, Any], key: str, default: str) -> str:
    """resolve named output subdir
    output_subdirs mapping aware"""
    subdirs = get_cfg(cfg, "output_subdirs", {}) or {}

    if isinstance(subdirs, dict) and key in subdirs:
        return str(subdirs[key])

    return default


def get_step_dir(cfg: dict[str, Any], key: str, default: str) -> Path:
    """resolve step directory
    output_root plus configured subfolder"""
    return get_output_root(cfg) / get_output_subdir(cfg, key, default)


def get_step02_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 02 directory
    modeling dataset inputs"""
    return get_step_dir(cfg, "modeling_dataset", DEFAULT_MODELING_DATASET_SUBDIR)


def get_step03_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 03 directory
    trained global model inputs"""
    return get_step_dir(cfg, "global_model", DEFAULT_GLOBAL_MODEL_SUBDIR)


def get_step05_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 05 directory
    optional explanation artifacts"""
    return get_step_dir(cfg, "model_explanation", DEFAULT_MODEL_EXPLANATION_SUBDIR)


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 06 output directory
    all sample predictions folder"""
    return get_step_dir(cfg, "all_sample_predictions", DEFAULT_OUTPUT_SUBDIR)


def ensure_dir(path: Path) -> None:
    """create directory
    includes parents"""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write text report
    newline joined"""
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    """convert values for JSON
    Path and numpy aware"""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return str(value)


def save_json(data: dict[str, Any], path: Path) -> None:
    """write JSON file
    readable indent"""
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


# ============================================================
# TEXT AND NUMERIC HELPERS
# ============================================================


def normalize_key(value: Any) -> str:
    """normalize treatment key
    lowercase compact whitespace"""
    text = clean_text(value).lower()
    return " ".join(text.split())


def drug_key_to_dummy_suffix(value: Any) -> str:
    """convert drug key to dummy suffix
    mirrors common one-hot naming"""
    text = normalize_key(value)

    # replace non-alphanumeric runs with underscore
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def safe_filename(value: Any, max_len: int = 180) -> str:
    """safe filename token
    compact and Windows safe"""
    text = drug_key_to_dummy_suffix(value)

    if not text:
        text = "unknown"

    return text[:max_len]


def compact_list(values: list[Any], max_items: int = 12) -> str:
    """compact list text
    useful in reports"""
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    shown = cleaned[:max_items]

    if len(cleaned) > max_items:
        shown.append(f"... {len(cleaned) - max_items} more")

    return "; ".join(shown)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """coerce series to numeric
    invalid values become missing"""
    return pd.to_numeric(series, errors="coerce")


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """coerce frame to numeric
    invalid cells become missing"""
    return df.apply(pd.to_numeric, errors="coerce")


def clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    """clip predictions to 0..1
    probability-like output guard"""
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """find first available column
    candidate order preserved"""
    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# IO HELPERS
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

    # flexible fallback for unusual extension
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def write_table(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    """write table file
    parent folder created"""
    ensure_dir(path.parent)
    df.to_csv(path, sep=sep, index=False)


def expected_input_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """build expected input path map
    step 02, step 03, and teacher handoff"""
    project_dir = Path(str(get_cfg(cfg, "project_dir", ".")))
    step02 = get_step02_dir(cfg)
    step03 = get_step03_dir(cfg)
    step05 = get_step05_dir(cfg)

    model_input = get_cfg(cfg, "model_input_numeric", get_cfg(cfg, "spatial_feature_table", ""))
    teacher_table = get_cfg(cfg, "teacher_table", "")
    training_table = get_cfg(cfg, "training_table", "")

    return {
        "step02_dir": step02,
        "step03_dir": step03,
        "step05_dir": step05,
        "modeling_table": step02 / "modeling_table.tsv",
        "x_features": step02 / "X_features.csv",
        "model_feature_manifest": step02 / "model_feature_manifest.csv",
        "teacher_target_table": step02 / "teacher_target_table.tsv",
        "sample_split": step02 / "sample_split.tsv",
        "global_model": step03 / "model.joblib",
        "global_metrics": step03 / "metrics.tsv",
        "shap_summary": step05 / "shap_summary.tsv",
        "filtered_spatial_shap_summary": step05 / "filtered_spatial_shap_summary.tsv",
        "model_input_numeric": resolve_path(project_dir, model_input) or Path(""),
        "teacher_table": resolve_path(project_dir, teacher_table) or Path(""),
        "training_table": resolve_path(project_dir, training_table) or Path(""),
    }


def build_input_file_report(paths: dict[str, Path]) -> pd.DataFrame:
    """summarize input files
    required and optional artifacts"""
    required = {
        "modeling_table",
        "x_features",
        "model_feature_manifest",
        "global_model",
        "model_input_numeric",
    }

    rows: list[dict[str, Any]] = []

    for name, path in paths.items():
        exists = bool(path.exists()) if str(path) not in {"", "."} else False
        is_file = bool(exists and path.is_file())
        is_dir = bool(exists and path.is_dir())

        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "required": name in required,
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "size_bytes": int(path.stat().st_size) if is_file else np.nan,
            }
        )

    report = pd.DataFrame(rows)
    missing = report.loc[(report["required"] == True) & (report["exists"] != True), "input_name"].tolist()

    if missing:
        raise FileNotFoundError("Missing required input files: " + compact_list(missing))

    return report


# ============================================================
# MODEL LOADING AND FEATURE DISCOVERY
# ============================================================


def load_model_bundle(path: Path) -> dict[str, Any]:
    """load global model bundle
    step 03 joblib object"""
    if not path.exists():
        raise FileNotFoundError(path)

    obj = joblib.load(path)

    # preferred step 03 output, dictionary with pipeline and feature names
    if isinstance(obj, dict):
        if "pipeline" not in obj:
            raise ValueError("model.joblib dictionary has no 'pipeline' entry")
        if "feature_names" not in obj:
            raise ValueError("model.joblib dictionary has no 'feature_names' entry")
        return obj

    # fallback for bare sklearn pipeline, feature names must come from elsewhere
    raise ValueError("Unsupported model.joblib format. Expected step 03 model bundle dictionary.")


def get_pipeline_from_bundle(bundle: dict[str, Any]) -> Any:
    """extract fitted pipeline
    standard step 03 key"""
    return bundle["pipeline"]


def get_feature_names_from_bundle(bundle: dict[str, Any]) -> list[str]:
    """extract ordered model features
    model input order is mandatory"""
    features = bundle.get("feature_names")

    if not isinstance(features, list) or len(features) == 0:
        raise ValueError("Model bundle feature_names is empty or invalid")

    return [str(x) for x in features]


def get_task_from_bundle(bundle: dict[str, Any], cfg: dict[str, Any]) -> str:
    """resolve model task
    bundle value preferred"""
    task = clean_text(bundle.get("task", get_cfg(cfg, "task", "regression"))).lower()

    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task in model bundle: {task}")

    return task


def identify_drug_dummy_columns(feature_names: list[str], cfg: dict[str, Any]) -> list[str]:
    """find treatment identity features
    exact model feature names only"""
    primary_prefix = str(get_cfg(cfg, "drug_dummy_prefix", "drug__"))
    extra_prefixes = get_cfg(
        cfg,
        "drug_dummy_prefixes",
        [primary_prefix, "drug_dummy__", "drug_key__", "treatment__", "treatment_key__"],
    )

    prefixes = [str(x).lower() for x in extra_prefixes]

    out: list[str] = []
    for feature in feature_names:
        low = feature.lower()
        if any(low.startswith(prefix) for prefix in prefixes):
            out.append(feature)

    return out


def spatial_model_features(feature_names: list[str], drug_dummy_cols: list[str]) -> list[str]:
    """return non-drug model features
    spatial feature block"""
    drug_set = set(drug_dummy_cols)
    return [feature for feature in feature_names if feature not in drug_set]


# ============================================================
# MANIFEST AND SPATIAL FEATURE RECONSTRUCTION
# ============================================================


def resolve_manifest_feature_name_col(manifest: pd.DataFrame) -> str | None:
    """find clean feature name column
    supports old and new step 02 schemas"""
    return first_existing_column(
        manifest,
        ["feature_name", "feature_clean", "feature", "column", "column_name"],
    )


def resolve_manifest_original_col(manifest: pd.DataFrame) -> str | None:
    """find original spatial feature column
    biology source mapping"""
    return first_existing_column(
        manifest,
        ["feature_original", "original_feature", "source_column", "upstream_manifest_key", "original_column"],
    )


def build_feature_source_map(manifest: pd.DataFrame) -> dict[str, str]:
    """map model feature to source column
    clean feature -> original feature"""
    if manifest.empty:
        return {}

    feature_col = resolve_manifest_feature_name_col(manifest)
    original_col = resolve_manifest_original_col(manifest)

    if feature_col is None:
        return {}

    mapping: dict[str, str] = {}

    for _, row in manifest.iterrows():
        clean_feature = clean_text(row.get(feature_col, ""))
        if not clean_feature:
            continue

        original = clean_text(row.get(original_col, "")) if original_col is not None else ""

        # if original missing, assume feature name is already source column
        mapping[clean_feature] = original if original else clean_feature

    return mapping


def build_spatial_prediction_matrix(
    spatial_df: pd.DataFrame,
    selected_samples: list[str],
    spatial_features: list[str],
    feature_source_map: dict[str, str],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """build sample x spatial-feature matrix
    model feature names as columns"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if sample_col not in spatial_df.columns:
        raise ValueError(f"Spatial feature table missing sample column: {sample_col}")

    work = spatial_df.copy()
    work[sample_col] = work[sample_col].astype(str).map(clean_text)

    # one row per sample expected from teacher builder handoff
    work = work.drop_duplicates(sample_col).copy()
    work = work[work[sample_col].isin(set(selected_samples))].copy()

    # preserve selected sample order from sample selection step
    order = pd.DataFrame({sample_col: selected_samples, "_sample_order": range(len(selected_samples))})
    work = order.merge(work, on=sample_col, how="left").sort_values("_sample_order").drop(columns=["_sample_order"])

    rows: list[dict[str, Any]] = []
    out = pd.DataFrame({sample_col: work[sample_col].tolist()})

    for feature in spatial_features:
        source_col = feature_source_map.get(feature, feature)

        if feature in work.columns:
            source_used = feature
            status = "direct_feature_column"
            values = work[feature]
        elif source_col in work.columns:
            source_used = source_col
            status = "manifest_source_column"
            values = work[source_col]
        else:
            source_used = source_col
            status = "missing_source_column"
            values = pd.Series([np.nan] * len(work), index=work.index)

        numeric = safe_numeric_series(values)
        out[feature] = numeric.to_numpy()

        rows.append(
            {
                "model_feature": feature,
                "source_column": source_used,
                "status": status,
                "n_rows": int(len(numeric)),
                "n_missing": int(numeric.isna().sum()),
                "missing_fraction": float(numeric.isna().mean()) if len(numeric) else np.nan,
                "n_unique": int(numeric.nunique(dropna=True)),
            }
        )

    report = pd.DataFrame(rows)

    missing_sources = report.loc[report["status"] == "missing_source_column", "model_feature"].tolist()
    if missing_sources:
        # model imputer can handle missing values, but missing whole feature sources should be visible
        print("WARNING: spatial source columns missing for features:", compact_list(missing_sources), file=sys.stderr)

    return out, report


# ============================================================
# SAMPLE AND TREATMENT SELECTION
# ============================================================


def load_spatial_feature_table(paths: dict[str, Path]) -> pd.DataFrame:
    """load all-sample spatial features
    teacher builder model_input_numeric"""
    return load_table(paths["model_input_numeric"])


def load_modeling_context(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """load step 02 context tables
    empty frames when optional files absent"""
    out: dict[str, pd.DataFrame] = {}

    for key in ["modeling_table", "x_features", "model_feature_manifest", "teacher_target_table", "sample_split"]:
        path = paths.get(key)
        if path is not None and path.exists() and path.is_file():
            out[key] = load_table(path)
        else:
            out[key] = pd.DataFrame()

    return out


def load_teacher_overlap_table(paths: dict[str, Path], cfg: dict[str, Any]) -> pd.DataFrame:
    """load teacher labels for overlap annotation
    step 02 table preferred"""
    if paths["teacher_target_table"].exists():
        return load_table(paths["teacher_target_table"])

    if paths["teacher_table"].exists():
        return load_table(paths["teacher_table"])

    if paths["training_table"].exists():
        return load_table(paths["training_table"])

    return pd.DataFrame()


def selected_sample_ids(
    spatial_df: pd.DataFrame,
    modeling_table: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[str]:
    """select samples to predict
    YAML prediction_sample_mode controlled"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    mode = clean_text(get_cfg(cfg, "prediction_sample_mode", "all_spatial_samples")).lower()
    max_samples_value = get_cfg(cfg, "max_prediction_samples", None)

    if sample_col not in spatial_df.columns:
        raise ValueError(f"Spatial table missing sample column: {sample_col}")

    all_samples = spatial_df[sample_col].dropna().astype(str).map(clean_text).drop_duplicates().tolist()

    if mode in {"test_labeled_samples", "labeled_samples", "training_labeled_samples"}:
        if modeling_table.empty or sample_col not in modeling_table.columns:
            raise ValueError(f"prediction_sample_mode={mode} requires step 02 modeling_table with {sample_col}")
        samples = modeling_table[sample_col].dropna().astype(str).map(clean_text).drop_duplicates().tolist()
    elif mode in {"all_spatial_samples", "all_samples", "all"}:
        samples = all_samples
    elif mode in {"first_n_spatial_samples", "first_n_samples"}:
        samples = all_samples
    else:
        raise ValueError(
            "Unsupported prediction_sample_mode. Use test_labeled_samples, "
            "labeled_samples, all_spatial_samples, or first_n_spatial_samples."
        )

    # keep only samples with spatial rows
    spatial_set = set(all_samples)
    samples = [sample for sample in samples if sample in spatial_set]

    if not none_like(max_samples_value):
        max_samples = int(max_samples_value)
        samples = samples[:max_samples]

    if len(samples) == 0:
        raise ValueError("No prediction samples selected")

    return samples


def treatments_from_modeling_table(modeling_table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """extract treatment list from step 02
    preferred treatment source"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    if modeling_table.empty:
        return pd.DataFrame(columns=[drug_col, drug_key_col, "drug_key_norm"])

    if drug_col not in modeling_table.columns or drug_key_col not in modeling_table.columns:
        raise ValueError(f"modeling_table must contain {drug_col} and {drug_key_col}")

    out = modeling_table[[drug_col, drug_key_col]].drop_duplicates().copy()
    out[drug_col] = out[drug_col].map(clean_text)
    out[drug_key_col] = out[drug_key_col].map(clean_text)
    out["drug_key_norm"] = out[drug_key_col].map(normalize_key)

    return out.drop_duplicates("drug_key_norm").sort_values(drug_col).reset_index(drop=True)


def treatments_from_teacher_or_training(paths: dict[str, Path], cfg: dict[str, Any]) -> pd.DataFrame:
    """extract treatment list from configured teacher tables
    fallback treatment source"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    source = clean_text(get_cfg(cfg, "prediction_treatment_source", "training_table_unique_drugs")).lower()

    if source.startswith("teacher") and paths["teacher_table"].exists():
        df = load_table(paths["teacher_table"])
    elif paths["training_table"].exists():
        df = load_table(paths["training_table"])
    elif paths["teacher_table"].exists():
        df = load_table(paths["teacher_table"])
    else:
        return pd.DataFrame(columns=[drug_col, drug_key_col, "drug_key_norm"])

    if drug_col not in df.columns or drug_key_col not in df.columns:
        return pd.DataFrame(columns=[drug_col, drug_key_col, "drug_key_norm"])

    out = df[[drug_col, drug_key_col]].drop_duplicates().copy()
    out[drug_col] = out[drug_col].map(clean_text)
    out[drug_key_col] = out[drug_key_col].map(clean_text)
    out["drug_key_norm"] = out[drug_key_col].map(normalize_key)

    return out.drop_duplicates("drug_key_norm").sort_values(drug_col).reset_index(drop=True)


def treatments_from_drug_dummies(drug_dummy_cols: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """recover treatments from dummy names
    last-resort fallback"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    prefix = str(get_cfg(cfg, "drug_dummy_prefix", "drug__"))

    rows: list[dict[str, Any]] = []

    for col in drug_dummy_cols:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
        else:
            suffix = col.split("__", 1)[-1]

        key = suffix.replace("_", " ")
        rows.append({drug_col: key, drug_key_col: key, "drug_key_norm": normalize_key(key)})

    return pd.DataFrame(rows).drop_duplicates("drug_key_norm").reset_index(drop=True)


def select_treatments(
    paths: dict[str, Path],
    modeling_table: pd.DataFrame,
    drug_dummy_cols: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """select treatments to predict
    source controlled by YAML"""
    source = clean_text(get_cfg(cfg, "prediction_treatment_source", "training_table_unique_drugs")).lower()

    if source in {"modeling_table", "step02_modeling_table", "training_table_unique_drugs"}:
        treatments = treatments_from_modeling_table(modeling_table, cfg)
        if treatments.empty:
            treatments = treatments_from_teacher_or_training(paths, cfg)
    elif source in {"teacher_table_unique_drugs", "teacher_table"}:
        treatments = treatments_from_teacher_or_training(paths, cfg)
    elif source in {"model_drug_dummies", "drug_dummies"}:
        treatments = treatments_from_drug_dummies(drug_dummy_cols, cfg)
    else:
        raise ValueError(f"Unsupported prediction_treatment_source: {source}")

    if treatments.empty:
        treatments = treatments_from_drug_dummies(drug_dummy_cols, cfg)

    if treatments.empty:
        raise ValueError("No prediction treatments selected")

    return treatments.reset_index(drop=True)


# ============================================================
# DRUG DUMMY MAPPING
# ============================================================


def derive_drug_dummy_map_from_training(
    modeling_table: pd.DataFrame,
    x_features: pd.DataFrame,
    drug_dummy_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[dict[str, str], pd.DataFrame]:
    """map drug_key to one-hot column
    uses actual step 02 training matrix"""
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))

    rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}

    if modeling_table.empty or x_features.empty or not drug_dummy_cols:
        return mapping, pd.DataFrame(rows)

    if drug_key_col not in modeling_table.columns:
        return mapping, pd.DataFrame(rows)

    dummy_present = [col for col in drug_dummy_cols if col in x_features.columns]
    if not dummy_present:
        return mapping, pd.DataFrame(rows)

    work = modeling_table[[drug_key_col]].copy()
    if drug_col in modeling_table.columns:
        work[drug_col] = modeling_table[drug_col].values
    else:
        work[drug_col] = modeling_table[drug_key_col].values

    work["drug_key_norm"] = work[drug_key_col].map(normalize_key)
    dummy_numeric = safe_numeric_frame(x_features[dummy_present]).fillna(0.0)

    for drug_key_norm, idx in work.groupby("drug_key_norm").groups.items():
        subset = dummy_numeric.loc[list(idx), dummy_present]
        means = subset.mean(axis=0)
        best_col = str(means.idxmax()) if not means.empty else ""
        best_mean = float(means.max()) if not means.empty else 0.0
        active_cols = means[means > 0.5].index.astype(str).tolist()

        if len(active_cols) == 1:
            mapped_col = active_cols[0]
            status = "exact_active_dummy"
        elif best_mean > 0:
            mapped_col = best_col
            status = "best_nonzero_dummy"
        else:
            mapped_col = ""
            status = "no_active_dummy"

        if mapped_col:
            mapping[drug_key_norm] = mapped_col

        first_row = work.loc[list(idx)[0]]
        rows.append(
            {
                "drug": clean_text(first_row.get(drug_col, "")),
                "drug_key": clean_text(first_row.get(drug_key_col, "")),
                "drug_key_norm": drug_key_norm,
                "mapped_dummy_column": mapped_col,
                "status": status,
                "best_dummy_mean": best_mean,
                "n_active_dummy_columns": int(len(active_cols)),
            }
        )

    return mapping, pd.DataFrame(rows)


def fallback_drug_dummy_column(drug_key: str, drug_dummy_cols: list[str], cfg: dict[str, Any]) -> str | None:
    """guess dummy column from drug key
    used only when training map missing"""
    prefix = str(get_cfg(cfg, "drug_dummy_prefix", "drug__"))
    candidate = prefix + drug_key_to_dummy_suffix(drug_key)

    if candidate in set(drug_dummy_cols):
        return candidate

    # tolerant suffix search for older one-hot naming variants
    suffix = drug_key_to_dummy_suffix(drug_key)
    for col in drug_dummy_cols:
        if col.endswith(suffix):
            return col

    return None


def complete_drug_dummy_map(
    treatments: pd.DataFrame,
    base_map: dict[str, str],
    drug_dummy_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[dict[str, str], pd.DataFrame]:
    """complete dummy map for treatment list
    training-derived first, fallback second"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    rows: list[dict[str, Any]] = []
    out_map: dict[str, str] = {}

    for _, row in treatments.iterrows():
        drug = clean_text(row.get(drug_col, ""))
        drug_key = clean_text(row.get(drug_key_col, drug))
        norm = normalize_key(drug_key)

        if norm in base_map:
            dummy_col = base_map[norm]
            status = "training_map"
        else:
            dummy_col = fallback_drug_dummy_column(drug_key, drug_dummy_cols, cfg)
            status = "fallback_name_match" if dummy_col else "missing_dummy"

        if dummy_col:
            out_map[norm] = dummy_col

        rows.append(
            {
                "drug": drug,
                "drug_key": drug_key,
                "drug_key_norm": norm,
                "dummy_column": dummy_col or "",
                "status": status,
            }
        )

    return out_map, pd.DataFrame(rows)


# ============================================================
# PREDICTION MATRIX AND MODEL APPLICATION
# ============================================================


def predict_model(pipeline: Any, X: pd.DataFrame, task: str) -> dict[str, np.ndarray]:
    """apply fitted model
    task-aware outputs"""
    if task == "classification":
        label = pipeline.predict(X)

        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X)[:, 1]
        else:
            prob = label.astype(float)

        return {"prediction": label, "probability": prob}

    pred = pipeline.predict(X).astype(float)
    return {"prediction": pred, "probability": pred}


def build_prediction_rows_for_treatment(
    treatment_row: pd.Series,
    spatial_matrix: pd.DataFrame,
    feature_names: list[str],
    drug_dummy_cols: list[str],
    dummy_map: dict[str, str],
    pipeline: Any,
    task: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """predict one treatment across samples
    returns predictions and feature status"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    drug = clean_text(treatment_row.get(drug_col, ""))
    drug_key = clean_text(treatment_row.get(drug_key_col, drug))
    drug_key_norm = normalize_key(drug_key)

    # start from sample-level spatial matrix
    X = spatial_matrix.drop(columns=[sample_col], errors="ignore").copy()

    # drug dummies reset per treatment
    for col in drug_dummy_cols:
        X[col] = 0.0

    dummy_col = dummy_map.get(drug_key_norm)
    dummy_status = "not_used_no_drug_dummies"

    if drug_dummy_cols:
        if dummy_col and dummy_col in X.columns:
            X[dummy_col] = 1.0
            dummy_status = "mapped"
        else:
            dummy_status = "missing_dummy_column"

    # add missing features as NaN, then enforce exact training order
    missing_model_features = [feature for feature in feature_names if feature not in X.columns]
    for feature in missing_model_features:
        X[feature] = np.nan

    X = X[feature_names]

    pred = predict_model(pipeline, X, task)
    pred_value = pred["probability"] if task == "classification" else pred["prediction"]

    if bool_like(get_cfg(cfg, "clip_predictions_to_01", True)):
        pred_value = clip01(pred_value)

    out = pd.DataFrame(
        {
            sample_col: spatial_matrix[sample_col].values,
            drug_col: drug,
            drug_key_col: drug_key,
            "drug_key_norm": drug_key_norm,
            "prediction_task": task,
            "dummy_column": dummy_col or "",
            "dummy_status": dummy_status,
            "predicted_fused_prob_responder": pred_value,
        }
    )

    if task == "classification":
        out["predicted_label"] = pred["prediction"]
        out["predicted_prob_responder"] = pred_value
    else:
        threshold = float(get_cfg(cfg, "binary_threshold", 0.5))
        out["predicted_binary_at_threshold"] = (out["predicted_fused_prob_responder"] >= threshold).astype(int)
        out["prediction_threshold"] = threshold

    feature_status = pd.DataFrame(
        {
            "drug": [drug],
            "drug_key": [drug_key],
            "drug_key_norm": [drug_key_norm],
            "dummy_column": [dummy_col or ""],
            "dummy_status": [dummy_status],
            "n_missing_model_features_added": [len(missing_model_features)],
            "missing_model_feature_examples": [compact_list(missing_model_features)],
        }
    )

    return out, feature_status


def predict_all_pairs(
    treatments: pd.DataFrame,
    spatial_matrix: pd.DataFrame,
    feature_names: list[str],
    drug_dummy_cols: list[str],
    dummy_map: dict[str, str],
    pipeline: Any,
    task: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """predict selected samples by treatments
    loops treatment-wise for clarity"""
    prediction_frames: list[pd.DataFrame] = []
    feature_status_frames: list[pd.DataFrame] = []

    for _, treatment_row in treatments.iterrows():
        pred_df, status_df = build_prediction_rows_for_treatment(
            treatment_row=treatment_row,
            spatial_matrix=spatial_matrix,
            feature_names=feature_names,
            drug_dummy_cols=drug_dummy_cols,
            dummy_map=dummy_map,
            pipeline=pipeline,
            task=task,
            cfg=cfg,
        )
        prediction_frames.append(pred_df)
        feature_status_frames.append(status_df)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    feature_status = pd.concat(feature_status_frames, ignore_index=True) if feature_status_frames else pd.DataFrame()

    return predictions, feature_status


# ============================================================
# TEACHER OVERLAP AND SUMMARY TABLES
# ============================================================


def build_teacher_overlap(teacher_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """prepare teacher labels for merge
    one row per sample treatment"""
    if teacher_df.empty:
        return pd.DataFrame()

    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    slide_col = str(get_cfg(cfg, "slide_col", "slide_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    # support tables that use slide_id instead of sample_id
    if sample_col not in teacher_df.columns and slide_col in teacher_df.columns:
        teacher_df = teacher_df.rename(columns={slide_col: sample_col}).copy()

    required = [sample_col, drug_key_col]
    if any(col not in teacher_df.columns for col in required):
        return pd.DataFrame()

    keep_cols = [sample_col, drug_key_col]

    for col in [drug_col, target_col, "fused_confidence", "modality_used", "expression_available", "histology_available"]:
        if col in teacher_df.columns and col not in keep_cols:
            keep_cols.append(col)

    out = teacher_df[keep_cols].copy()
    out[sample_col] = out[sample_col].astype(str).map(clean_text)
    out[drug_key_col] = out[drug_key_col].map(clean_text)
    out["drug_key_norm"] = out[drug_key_col].map(normalize_key)

    rename_map = {}
    if target_col in out.columns:
        rename_map[target_col] = "teacher_fused_prob_responder"
    if "fused_confidence" in out.columns:
        rename_map["fused_confidence"] = "teacher_fused_confidence"
    if "modality_used" in out.columns:
        rename_map["modality_used"] = "teacher_modality_used"

    out = out.rename(columns=rename_map)
    out = out.drop_duplicates([sample_col, "drug_key_norm"]).copy()
    out["is_teacher_labeled"] = True

    return out


def attach_teacher_overlap(predictions: pd.DataFrame, teacher_overlap: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """merge teacher labels into predictions
    labeled row error columns"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if teacher_overlap.empty:
        out = predictions.copy()
        out["is_teacher_labeled"] = False
        out["teacher_fused_prob_responder"] = np.nan
        out["prediction_error_vs_teacher"] = np.nan
        out["absolute_error_vs_teacher"] = np.nan
        return out

    overlap_cols = [col for col in teacher_overlap.columns if col not in {"drug", "drug_key"}]
    out = predictions.merge(
        teacher_overlap[overlap_cols],
        on=[sample_col, "drug_key_norm"],
        how="left",
    )

    out["is_teacher_labeled"] = out["is_teacher_labeled"].fillna(False).astype(bool)

    if "teacher_fused_prob_responder" in out.columns:
        out["teacher_fused_prob_responder"] = safe_numeric_series(out["teacher_fused_prob_responder"])
        out["prediction_error_vs_teacher"] = out["predicted_fused_prob_responder"] - out["teacher_fused_prob_responder"]
        out["absolute_error_vs_teacher"] = out["prediction_error_vs_teacher"].abs()
    else:
        out["teacher_fused_prob_responder"] = np.nan
        out["prediction_error_vs_teacher"] = np.nan
        out["absolute_error_vs_teacher"] = np.nan

    return out


def build_summary_by_sample(predictions: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize predictions per sample
    mean and range across treatments"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if predictions.empty:
        return pd.DataFrame()

    return (
        predictions.groupby(sample_col, as_index=False)
        .agg(
            n_treatments_predicted=("drug_key_norm", "nunique"),
            mean_predicted_response=("predicted_fused_prob_responder", "mean"),
            median_predicted_response=("predicted_fused_prob_responder", "median"),
            min_predicted_response=("predicted_fused_prob_responder", "min"),
            max_predicted_response=("predicted_fused_prob_responder", "max"),
            std_predicted_response=("predicted_fused_prob_responder", "std"),
            n_teacher_labeled=("is_teacher_labeled", "sum"),
        )
        .sort_values(["mean_predicted_response", sample_col], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_summary_by_treatment(predictions: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize predictions per treatment
    cohort-level treatment ranking"""
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    if predictions.empty:
        return pd.DataFrame()

    summary = (
        predictions.groupby([drug_col, drug_key_col, "drug_key_norm"], as_index=False)
        .agg(
            n_samples_predicted=("predicted_fused_prob_responder", "size"),
            mean_predicted_response=("predicted_fused_prob_responder", "mean"),
            median_predicted_response=("predicted_fused_prob_responder", "median"),
            min_predicted_response=("predicted_fused_prob_responder", "min"),
            max_predicted_response=("predicted_fused_prob_responder", "max"),
            std_predicted_response=("predicted_fused_prob_responder", "std"),
            n_teacher_labeled=("is_teacher_labeled", "sum"),
            mean_teacher_response=("teacher_fused_prob_responder", "mean"),
            mean_absolute_error_on_labeled=("absolute_error_vs_teacher", "mean"),
        )
        .sort_values(["mean_predicted_response", drug_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary["prediction_rank_by_mean"] = np.arange(1, len(summary) + 1)
    return summary


def build_top_treatment_per_sample(predictions: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """select top treatment per sample
    highest predicted response"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    if predictions.empty:
        return pd.DataFrame()

    work = predictions.sort_values(
        [sample_col, "predicted_fused_prob_responder", "drug_key_norm"],
        ascending=[True, False, True],
    ).copy()

    work["prediction_rank_within_sample"] = work.groupby(sample_col)["predicted_fused_prob_responder"].rank(
        ascending=False,
        method="first",
    ).astype(int)

    return work[work["prediction_rank_within_sample"] == 1].reset_index(drop=True)


def build_teacher_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """filter labeled prediction rows
    prediction versus teacher check"""
    if predictions.empty or "is_teacher_labeled" not in predictions.columns:
        return pd.DataFrame()

    cols = [
        col for col in predictions.columns
        if col in {
            "sample_id",
            "slide_id",
            "drug",
            "drug_key",
            "drug_key_norm",
            "predicted_fused_prob_responder",
            "teacher_fused_prob_responder",
            "prediction_error_vs_teacher",
            "absolute_error_vs_teacher",
            "teacher_fused_confidence",
            "teacher_modality_used",
            "expression_available",
            "histology_available",
        }
    ]

    out = predictions[predictions["is_teacher_labeled"] == True].copy()

    if cols:
        out = out[cols].copy()

    return out.reset_index(drop=True)


def build_prediction_matrix(predictions: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """wide sample by treatment matrix
    predicted response values"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    if predictions.empty:
        return pd.DataFrame()

    matrix = predictions.pivot_table(
        index=sample_col,
        columns=drug_key_col,
        values="predicted_fused_prob_responder",
        aggfunc="mean",
    ).reset_index()

    matrix.columns.name = None
    return matrix


# ============================================================
# MANIFESTS AND REPORTS
# ============================================================


def build_prediction_sample_manifest(spatial_df: pd.DataFrame, selected_samples: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """sample manifest for predictions
    selected versus all spatial samples"""
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    all_samples = spatial_df[[sample_col]].drop_duplicates().copy()
    all_samples[sample_col] = all_samples[sample_col].astype(str).map(clean_text)
    selected_set = set(selected_samples)

    all_samples["selected_for_prediction"] = all_samples[sample_col].isin(selected_set)
    all_samples["selection_order"] = all_samples[sample_col].map({sample: i for i, sample in enumerate(selected_samples)})

    return all_samples.sort_values(["selected_for_prediction", "selection_order", sample_col], ascending=[False, True, True]).reset_index(drop=True)


def build_prediction_treatment_manifest(treatments: pd.DataFrame, dummy_map_report: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """treatment manifest for predictions
    includes dummy mapping status"""
    if dummy_map_report.empty:
        return treatments.copy()

    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    work = treatments.copy()
    work["drug_key_norm"] = work[drug_key_col].map(normalize_key)

    report_cols = ["drug_key_norm", "dummy_column", "status"]
    report_cols = [col for col in report_cols if col in dummy_map_report.columns]

    if report_cols:
        return work.merge(dummy_map_report[report_cols].drop_duplicates("drug_key_norm"), on="drug_key_norm", how="left")

    return work


def write_disabled_outputs(out_dir: Path, cfg: dict[str, Any]) -> None:
    """write empty outputs when disabled
    keeps pipeline contract"""
    ensure_dir(out_dir)

    pd.DataFrame().to_csv(out_dir / "all_sample_treatment_predictions.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "prediction_summary_by_sample.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "prediction_summary_by_treatment.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "top_treatment_per_sample.tsv", sep="\t", index=False)
    pd.DataFrame().to_csv(out_dir / "teacher_labeled_prediction_comparison.tsv", sep="\t", index=False)

    write_text(
        out_dir / "all_sample_prediction_summary.txt",
        [
            "All sample treatment prediction summary",
            "",
            "Status",
            "  disabled by config",
            "",
            "Reason",
            "  run_all_sample_predictions is false",
        ],
    )

    save_json(
        {
            "script_name": SCRIPT_NAME,
            "step_name": STEP_NAME,
            "run_all_sample_predictions": False,
            "output_dir": str(out_dir),
        },
        out_dir / "run_config.json",
    )


def write_summary_text(
    path: Path,
    cfg: dict[str, Any],
    input_report: pd.DataFrame,
    predictions: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    treatment_manifest: pd.DataFrame,
    spatial_feature_report: pd.DataFrame,
    drug_dummy_report: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> None:
    """write human-readable prediction summary
    compact output report"""
    lines: list[str] = []
    lines.append("All sample treatment prediction summary")
    lines.append("")

    lines.append("Run settings")
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  prediction_sample_mode: {get_cfg(cfg, 'prediction_sample_mode', '')}")
    lines.append(f"  max_prediction_samples: {get_cfg(cfg, 'max_prediction_samples', '')}")
    lines.append(f"  prediction_treatment_source: {get_cfg(cfg, 'prediction_treatment_source', '')}")
    lines.append(f"  clip_predictions_to_01: {get_cfg(cfg, 'clip_predictions_to_01', '')}")
    lines.append("")

    lines.append("Model")
    lines.append(f"  task: {model_bundle.get('task', '')}")
    lines.append(f"  model_type: {model_bundle.get('model_type', '')}")
    lines.append(f"  n_model_features: {len(model_bundle.get('feature_names', []))}")
    lines.append(f"  split_source: {model_bundle.get('split_source', '')}")
    lines.append("")

    lines.append("Inputs")
    for _, row in input_report.iterrows():
        status = "exists" if bool(row.get("exists")) else "missing"
        required = "required" if bool(row.get("required")) else "optional"
        lines.append(f"  {row['input_name']}: {status}; {required}; {row['path']}")
    lines.append("")

    n_samples_selected = int(sample_manifest["selected_for_prediction"].sum()) if "selected_for_prediction" in sample_manifest.columns else 0
    n_treatments = int(len(treatment_manifest))
    n_rows = int(len(predictions))
    n_labeled = int(predictions["is_teacher_labeled"].sum()) if "is_teacher_labeled" in predictions.columns else 0

    lines.append("Prediction shape")
    lines.append(f"  selected samples: {n_samples_selected}")
    lines.append(f"  selected treatments: {n_treatments}")
    lines.append(f"  prediction rows: {n_rows}")
    lines.append(f"  teacher-labeled overlap rows: {n_labeled}")
    lines.append("")

    if not predictions.empty:
        y = safe_numeric_series(predictions["predicted_fused_prob_responder"])
        lines.append("Prediction distribution")
        lines.append(f"  mean: {float(y.mean()):.4f}")
        lines.append(f"  median: {float(y.median()):.4f}")
        lines.append(f"  min: {float(y.min()):.4f}")
        lines.append(f"  max: {float(y.max()):.4f}")
        lines.append(f"  std: {float(y.std()):.4f}")
        lines.append("")

    if "status" in spatial_feature_report.columns:
        missing_sources = int((spatial_feature_report["status"] == "missing_source_column").sum())
        lines.append("Spatial feature reconstruction")
        lines.append(f"  spatial model features checked: {len(spatial_feature_report)}")
        lines.append(f"  missing source columns: {missing_sources}")
        lines.append("")

    if "status" in drug_dummy_report.columns:
        lines.append("Drug dummy mapping")
        for status, count in drug_dummy_report["status"].value_counts().items():
            lines.append(f"  {status}: {count}")
        lines.append("")

    lines.append("Notes")
    lines.append("  Predictions are model-derived estimates, not new teacher labels")
    lines.append("  Teacher overlap rows are included only where fused teacher labels exist")
    lines.append("  10-sample run is an integration smoke test")

    write_text(path, lines)


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main() -> int:
    """run all sample treatment prediction
    writes prediction outputs and reports"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    out_dir = get_output_dir(cfg)
    ensure_dir(out_dir)

    run_enabled = bool_like(get_cfg(cfg, "run_all_sample_predictions", True)) or args.force

    print("Predicting all sample treatment pairs")
    print(f"Config: {Path(args.config)}")
    print(f"Output: {out_dir}")
    print(f"Enabled: {run_enabled}")

    if not run_enabled:
        write_disabled_outputs(out_dir, cfg)
        print("\nDONE")
        print("All sample treatment predictions disabled by config")
        return 0

    paths = expected_input_paths(cfg)
    input_report = build_input_file_report(paths)

    # load trained global model bundle
    model_bundle = load_model_bundle(paths["global_model"])
    pipeline = get_pipeline_from_bundle(model_bundle)
    feature_names = get_feature_names_from_bundle(model_bundle)
    task = get_task_from_bundle(model_bundle, cfg)

    drug_dummy_cols = identify_drug_dummy_columns(feature_names, cfg)
    spatial_features = spatial_model_features(feature_names, drug_dummy_cols)

    # load step 02 context and all-sample spatial features
    context = load_modeling_context(paths)
    spatial_df = load_spatial_feature_table(paths)
    teacher_df = load_teacher_overlap_table(paths, cfg)

    modeling_table = context["modeling_table"]
    x_features = context["x_features"]
    manifest = context["model_feature_manifest"]

    selected_samples = selected_sample_ids(spatial_df, modeling_table, cfg)
    treatments = select_treatments(paths, modeling_table, drug_dummy_cols, cfg)

    feature_source_map = build_feature_source_map(manifest)
    spatial_matrix, spatial_feature_report = build_spatial_prediction_matrix(
        spatial_df=spatial_df,
        selected_samples=selected_samples,
        spatial_features=spatial_features,
        feature_source_map=feature_source_map,
        cfg=cfg,
    )

    base_dummy_map, base_dummy_report = derive_drug_dummy_map_from_training(
        modeling_table=modeling_table,
        x_features=x_features,
        drug_dummy_cols=drug_dummy_cols,
        cfg=cfg,
    )

    dummy_map, dummy_map_report = complete_drug_dummy_map(
        treatments=treatments,
        base_map=base_dummy_map,
        drug_dummy_cols=drug_dummy_cols,
        cfg=cfg,
    )

    predictions_raw, prediction_feature_status = predict_all_pairs(
        treatments=treatments,
        spatial_matrix=spatial_matrix,
        feature_names=feature_names,
        drug_dummy_cols=drug_dummy_cols,
        dummy_map=dummy_map,
        pipeline=pipeline,
        task=task,
        cfg=cfg,
    )

    teacher_overlap = build_teacher_overlap(teacher_df, cfg)
    predictions = attach_teacher_overlap(predictions_raw, teacher_overlap, cfg)

    # stable sort for readable outputs
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    predictions = predictions.sort_values([sample_col, drug_col]).reset_index(drop=True)

    sample_manifest = build_prediction_sample_manifest(spatial_df, selected_samples, cfg)
    treatment_manifest = build_prediction_treatment_manifest(treatments, dummy_map_report, cfg)
    summary_by_sample = build_summary_by_sample(predictions, cfg)
    summary_by_treatment = build_summary_by_treatment(predictions, cfg)
    top_treatment = build_top_treatment_per_sample(predictions, cfg)
    teacher_comparison = build_teacher_comparison(predictions)
    prediction_matrix = build_prediction_matrix(predictions, cfg)

    # -------------------------
    # Write outputs
    # -------------------------

    write_table(input_report, out_dir / "input_file_report.tsv")
    write_table(sample_manifest, out_dir / "prediction_sample_manifest.tsv")
    write_table(treatment_manifest, out_dir / "prediction_treatment_manifest.tsv")
    write_table(spatial_feature_report, out_dir / "prediction_spatial_feature_report.tsv")
    write_table(base_dummy_report, out_dir / "training_drug_dummy_map.tsv")
    write_table(dummy_map_report, out_dir / "prediction_drug_dummy_map.tsv")
    write_table(prediction_feature_status, out_dir / "prediction_feature_status_by_treatment.tsv")

    write_table(predictions, out_dir / "all_sample_treatment_predictions.tsv")
    write_table(predictions, out_dir / "all_sample_treatment_predictions.csv", sep=",")
    write_table(summary_by_sample, out_dir / "prediction_summary_by_sample.tsv")
    write_table(summary_by_treatment, out_dir / "prediction_summary_by_treatment.tsv")
    write_table(top_treatment, out_dir / "top_treatment_per_sample.tsv")
    write_table(teacher_comparison, out_dir / "teacher_labeled_prediction_comparison.tsv")
    write_table(prediction_matrix, out_dir / "prediction_matrix_sample_by_treatment.tsv")

    write_summary_text(
        path=out_dir / "all_sample_prediction_summary.txt",
        cfg=cfg,
        input_report=input_report,
        predictions=predictions,
        sample_manifest=sample_manifest,
        treatment_manifest=treatment_manifest,
        spatial_feature_report=spatial_feature_report,
        drug_dummy_report=dummy_map_report,
        model_bundle=model_bundle,
    )

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(Path(args.config)),
        "output_dir": str(out_dir),
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "prediction_sample_mode": get_cfg(cfg, "prediction_sample_mode", ""),
        "prediction_treatment_source": get_cfg(cfg, "prediction_treatment_source", ""),
        "task": task,
        "model_type": model_bundle.get("model_type", ""),
        "n_model_features": int(len(feature_names)),
        "n_spatial_model_features": int(len(spatial_features)),
        "n_drug_dummy_features": int(len(drug_dummy_cols)),
        "n_selected_samples": int(len(selected_samples)),
        "n_selected_treatments": int(len(treatments)),
        "n_prediction_rows": int(len(predictions)),
        "n_teacher_labeled_overlap_rows": int(predictions["is_teacher_labeled"].sum()) if "is_teacher_labeled" in predictions.columns else 0,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "outputs": {
            "all_sample_treatment_predictions": str(out_dir / "all_sample_treatment_predictions.tsv"),
            "prediction_summary_by_sample": str(out_dir / "prediction_summary_by_sample.tsv"),
            "prediction_summary_by_treatment": str(out_dir / "prediction_summary_by_treatment.tsv"),
            "top_treatment_per_sample": str(out_dir / "top_treatment_per_sample.tsv"),
            "teacher_labeled_prediction_comparison": str(out_dir / "teacher_labeled_prediction_comparison.tsv"),
        },
    }
    save_json(run_info, out_dir / "run_config.json")

    print("\nDONE")
    print(f"Selected samples: {len(selected_samples):,}")
    print(f"Selected treatments: {len(treatments):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Teacher-labeled overlap rows: {run_info['n_teacher_labeled_overlap_rows']:,}")
    print(f"Wrote: {out_dir / 'all_sample_treatment_predictions.tsv'}")
    print(f"Wrote: {out_dir / 'prediction_summary_by_sample.tsv'}")
    print(f"Wrote: {out_dir / 'prediction_summary_by_treatment.tsv'}")
    print(f"Wrote: {out_dir / 'top_treatment_per_sample.tsv'}")
    print(f"Wrote: {out_dir / 'all_sample_prediction_summary.txt'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

