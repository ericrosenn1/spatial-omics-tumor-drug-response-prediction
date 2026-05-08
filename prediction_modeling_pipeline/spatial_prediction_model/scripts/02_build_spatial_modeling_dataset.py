"""
Script:
    02_build_spatial_modeling_dataset.py

Purpose:
    Build the leakage safe modeling dataset for the spatial response model.

Role:
    Second step in spatial_prediction_model.
    Consumes the validated teacher_builder handoff files from step 01.
    Creates canonical model input files for steps 03 and 04.
    Does not train a model.

Pipeline position:
    01_validate_prediction_inputs.py
        checks paths, table shapes, required columns, target values,
        sample overlap, sample treatment overlap, duplicate keys,
        feature manifest availability, and leakage column presence.

    02_build_spatial_modeling_dataset.py
        builds a row aligned model table, feature matrix, target vector,
        grouped sample split, and feature manifest.

    03_train_global_spatial_response_model.py
        reads modeling_table.tsv, X_features.csv, y_target.csv,
        model_feature_manifest.csv, and sample_split.tsv.

    04_train_per_treatment_models.py
        reads the same step 02 handoff and trains optional treatment specific
        spatial models when enough samples exist.

Modeling idea:
    The teacher table has one row per sample treatment pair.
    The spatial table has one row per sample.
    This script merges teacher rows onto spatial rows by sample_id.
    The result is one modeling row per labeled sample treatment pair.

Canonical outputs required by downstream steps:
    outputs/<run>/02_modeling_dataset/modeling_table.tsv
        Full row aligned modeling table with metadata, target, split, and
        clean spatial features.

    outputs/<run>/02_modeling_dataset/X_features.csv
        Numeric feature matrix aligned row by row to modeling_table.tsv.
        Contains clean spatial features and optional drug dummy features.

    outputs/<run>/02_modeling_dataset/X_features_spatial_only.csv
        Numeric spatial feature matrix without drug dummy columns.

    outputs/<run>/02_modeling_dataset/y_target.csv
        Target vector aligned row by row to modeling_table.tsv.

    outputs/<run>/02_modeling_dataset/model_feature_manifest.csv
        Canonical feature manifest with feature_name, feature_original,
        included, status, feature_group, feature_axis, and missingness stats.

    outputs/<run>/02_modeling_dataset/sample_split.tsv
        One row per sample_id with train or test assignment.

    outputs/<run>/02_modeling_dataset/split_assignments.tsv
        One row per modeling row with row_id, sample_id, drug_key, and split.

    outputs/<run>/02_modeling_dataset/leakage_excluded_columns.tsv
        Leakage configuration and presence report.

Additional rich outputs:
    feature_quality_report.tsv
    modeling_feature_manifest.tsv
    validated_spatial_features.tsv
    validated_teacher_table.tsv
    teacher_target_table.tsv
    merged_model_input.tsv
    merged_model_input.csv
    drug_dummy_manifest.tsv
    sample_manifest.tsv
    drug_summary.tsv
    target_summary.tsv
    split_summary.tsv
    unmatched_spatial_samples.tsv
    unmatched_teacher_rows.tsv
    dataset_build_summary.txt
    run_config.json

Design contract:
    YAML controls all paths, run scope, feature filtering, target column,
    split settings, and drug dummy behavior.
    No hard coded sample counts.
    Split is grouped by sample_id, not by row.
    Feature filtering is checked on both all spatial samples and the labeled
    samples used in the current run.
    Correlated biological features are not removed here.
    Drug dummy columns are added for the global model and excluded by step 04.

Notes:
    The current 10 sample run is a smoke test.
    The same script should run on the full 102 sample setup by editing YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import yaml


SCRIPT_NAME = "02_build_spatial_modeling_dataset.py"
STEP_NAME = "02_build_spatial_modeling_dataset"
DEFAULT_INPUT_VALIDATION_SUBDIR = "01_input_validation"
DEFAULT_MODELING_DATASET_SUBDIR = "02_modeling_dataset"


# ============================================================
# CONFIG AND PATH HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI args
    config path and optional validation override"""
    parser = argparse.ArgumentParser(description="Build spatial prediction modeling dataset")
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    parser.add_argument(
        "--allow-validation-errors",
        action="store_true",
        help="Continue even if step 01 validation_issues.tsv contains error rows",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config
    Windows BOM tolerant"""
    if not path.exists():
        raise FileNotFoundError(path)

    # utf 8 sig handles occasional PowerShell/Notepad BOM
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """get YAML value
    simple fallback helper"""
    return cfg[key] if key in cfg else default


def bool_like(value: Any) -> bool:
    """parse truth like values
    supports bools and common strings"""
    if isinstance(value, bool):
        return value

    text = clean_text(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on", "included"}


def is_windows_absolute_path(value: str) -> bool:
    """detect Windows drive paths
    useful when running checks outside Windows"""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(value)))


def is_absolute_path_like(value: Any) -> bool:
    """detect platform or Windows absolute path
    keeps C:/ paths intact"""
    text = str(value)
    return Path(text).is_absolute() or is_windows_absolute_path(text)


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve absolute or project relative path
    empty values stay None"""
    if value in [None, ""]:
        return None

    text = str(value).strip()

    if text == "":
        return None

    if is_absolute_path_like(text):
        return Path(text)

    return project_dir / text


def get_output_root(cfg: dict[str, Any]) -> Path:
    """resolve run output root
    output_root points at output_run_10 or output_run_102"""
    return Path(str(cfg["output_root"]))


def get_output_subdir(cfg: dict[str, Any], key: str, default: str) -> str:
    """resolve step subdir name
    output_subdirs mapping takes priority"""
    subdirs = get_cfg(cfg, "output_subdirs", {}) or {}

    if isinstance(subdirs, dict) and key in subdirs:
        return str(subdirs[key])

    return default


def get_validation_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 01 output folder
    canonical validation location"""
    output_root = get_output_root(cfg)
    subdir = get_output_subdir(cfg, "input_validation", DEFAULT_INPUT_VALIDATION_SUBDIR)
    return output_root / subdir


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """resolve step 02 output folder
    canonical modeling dataset location"""
    output_root = get_output_root(cfg)
    subdir = get_output_subdir(cfg, "modeling_dataset", DEFAULT_MODELING_DATASET_SUBDIR)
    return output_root / subdir


def ensure_dir(path: Path) -> None:
    """create directory
    parents included"""
    path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """JSON fallback converter
    handles Path and numpy scalar values"""
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
    readable reproducibility output"""
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def write_text(path: Path, lines: Sequence[str]) -> None:
    """write text file
    newline joined report"""
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# BASIC TEXT AND TABLE HELPERS
# ============================================================


def clean_text(value: Any) -> str:
    """clean scalar text
    empty string for missing values"""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_sample_id(value: Any) -> str:
    """normalize sample identifiers
    no case folding for SAMPLE_ ids"""
    return clean_text(value)


def normalize_drug_key(value: Any) -> str:
    """normalize drug key
    lowercase and compact whitespace"""
    text = clean_text(value).lower()
    return " ".join(text.split())


def clean_column_name(value: Any) -> str:
    """normalize column name for matching
    lowercase underscore form"""
    text = clean_text(value).lower()
    text = re.sub(r"\s+", "_", text)
    return text


def safe_feature_token(value: Any) -> str:
    """make safe feature token
    used for drug dummy column names"""
    text = normalize_drug_key(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if text == "":
        text = "unknown"

    return text


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """coerce series to numeric
    invalid and infinite values become missing"""
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """coerce frame to numeric
    cell level invalid values become missing"""
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def fraction(numerator: int | float, denominator: int | float) -> float:
    """safe fraction
    zero denominator gives NaN"""
    if denominator == 0:
        return float("nan")

    return float(numerator) / float(denominator)


def compact_list(values: Iterable[Any], max_items: int = 12) -> str:
    """compact list for reports
    avoids giant issue messages"""
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    shown = cleaned[:max_items]

    if len(cleaned) > max_items:
        shown.append(f"... {len(cleaned) - max_items} more")

    return "; ".join(shown)


def detect_separator(path: Path) -> str:
    """infer separator from suffix
    tsv otherwise csv"""
    if path.suffix.lower() == ".tsv":
        return "\t"

    return ","


def read_table(path: Path | None, label: str, required: bool = True) -> pd.DataFrame:
    """read CSV or TSV table
    optional mode returns empty frame"""
    if path is None:
        if required:
            raise FileNotFoundError(f"Missing configured path for {label}")
        return pd.DataFrame()

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required {label} not found: {path}")
        return pd.DataFrame()

    sep = detect_separator(path)

    # low_memory False avoids mixed dtype surprises in omics tables
    return pd.read_csv(path, sep=sep, low_memory=False)


def write_table(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    """write table
    parent folder created first"""
    ensure_dir(path.parent)
    df.to_csv(path, sep=sep, index=False)


def write_csv_if_requested(df: pd.DataFrame, path: Path, cfg: dict[str, Any]) -> None:
    """write optional csv copy
    controlled by write_csv_copies"""
    if bool_like(get_cfg(cfg, "write_csv_copies", True)):
        write_table(df, path, sep=",")


def first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    """find first matching column
    case sensitive first, lower fallback"""
    column_list = list(columns)

    for candidate in candidates:
        if candidate in column_list:
            return candidate

    lower_map = {str(col).lower(): col for col in column_list}

    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]

    return None


# ============================================================
# INPUT DISCOVERY AND VALIDATION GATE
# ============================================================


def get_input_paths(cfg: dict[str, Any]) -> dict[str, Path | None]:
    """resolve upstream input files
    teacher builder handoff plus validation folder"""
    project_dir = Path(str(cfg["project_dir"]))

    return {
        "model_input_numeric": resolve_path(project_dir, get_cfg(cfg, "model_input_numeric")),
        "teacher_table": resolve_path(project_dir, get_cfg(cfg, "teacher_table")),
        "feature_manifest": resolve_path(project_dir, get_cfg(cfg, "feature_manifest")),
        "training_table": resolve_path(project_dir, get_cfg(cfg, "training_table")),
        "teacher_builder_output_dir": resolve_path(project_dir, get_cfg(cfg, "teacher_builder_output_dir")),
        "validation_dir": get_validation_dir(cfg),
    }


def build_resolved_input_report(paths: dict[str, Path | None]) -> pd.DataFrame:
    """summarize resolved paths
    useful for reproducibility and debugging"""
    rows: list[dict[str, Any]] = []

    for name, path in paths.items():
        exists = bool(path is not None and path.exists())
        is_file = bool(exists and path.is_file())
        is_dir = bool(exists and path.is_dir())

        rows.append(
            {
                "name": name,
                "path": str(path) if path is not None else "",
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "size_bytes": int(path.stat().st_size) if is_file else np.nan,
            }
        )

    return pd.DataFrame(rows)


def require_core_inputs(paths: dict[str, Path | None]) -> None:
    """validate required upstream files
    stop before partial output state"""
    required = ["model_input_numeric", "teacher_table", "feature_manifest"]

    missing: list[str] = []

    for name in required:
        path = paths.get(name)

        if path is None or not path.exists() or not path.is_file():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError("Missing required input files: " + compact_list(missing))


def load_validation_issues(validation_dir: Path) -> pd.DataFrame:
    """load step 01 issues
    empty frame if unavailable"""
    path = validation_dir / "validation_issues.tsv"
    return read_table(path, "validation issues", required=False)


def validation_has_errors(issue_df: pd.DataFrame) -> bool:
    """detect error rows in validation issues
    severity column expected from step 01"""
    if issue_df.empty or "severity" not in issue_df.columns:
        return False

    levels = issue_df["severity"].astype(str).str.lower().str.strip()
    return bool(levels.isin(["error", "fail", "failed"]).any())


def load_available_labeled_samples(validation_dir: Path) -> pd.DataFrame:
    """load labeled sample list from step 01
    optional but preferred"""
    path = validation_dir / "available_labeled_samples.tsv"
    return read_table(path, "available labeled samples", required=False)


def available_sample_ids(available_df: pd.DataFrame, sample_col: str) -> list[str]:
    """extract available sample ids
    empty list if file missing or malformed"""
    if available_df.empty:
        return []

    col = first_existing_column(available_df.columns, [sample_col, "sample_id", "slide_id"])

    if col is None:
        return []

    ids = available_df[col].map(normalize_sample_id).tolist()
    return sorted({x for x in ids if x})


def enforce_validation_gate(issue_df: pd.DataFrame, args: argparse.Namespace) -> None:
    """stop on validation errors
    override allowed for development"""
    if validation_has_errors(issue_df) and not args.allow_validation_errors:
        raise ValueError(
            "Step 01 validation_issues.tsv contains error rows. "
            "Fix validation or rerun with --allow-validation-errors."
        )


# ============================================================
# INPUT TABLE LOADING AND COLUMN RESOLUTION
# ============================================================


def resolve_required_column(df: pd.DataFrame, configured: str, aliases: Sequence[str], label: str) -> str:
    """resolve required column name
    configured name first, aliases second"""
    col = first_existing_column(df.columns, [configured] + list(aliases))

    if col is None:
        raise ValueError(f"Could not find {label} column. Configured={configured}; aliases={list(aliases)}")

    return col


def load_input_tables(cfg: dict[str, Any], paths: dict[str, Path | None]) -> dict[str, pd.DataFrame]:
    """load upstream handoff tables
    training table optional for reports"""
    tables = {
        "spatial": read_table(paths["model_input_numeric"], "model_input_numeric", required=True),
        "teacher": read_table(paths["teacher_table"], "teacher_table", required=True),
        "feature_manifest": read_table(paths["feature_manifest"], "feature_manifest", required=True),
    }

    training_path = paths.get("training_table")
    tables["training_table"] = read_table(training_path, "training_table", required=False)

    return tables


def normalize_core_columns(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, str]:
    """resolve and normalize key columns
    updates tables in place"""
    spatial = tables["spatial"]
    teacher = tables["teacher"]

    sample_col = resolve_required_column(
        spatial,
        str(get_cfg(cfg, "sample_col", "sample_id")),
        ["sample_id", "slide_id", "sample", "slide"],
        "spatial sample",
    )

    teacher_sample_col = resolve_required_column(
        teacher,
        str(get_cfg(cfg, "sample_col", "sample_id")),
        ["sample_id", "slide_id", "sample", "slide"],
        "teacher sample",
    )

    drug_col = resolve_required_column(
        teacher,
        str(get_cfg(cfg, "drug_col", "drug")),
        ["drug", "treatment", "regimen", "drug_name"],
        "drug",
    )

    drug_key_col = resolve_required_column(
        teacher,
        str(get_cfg(cfg, "drug_key_col", "drug_key")),
        ["drug_key", "treatment_key", "regimen_key"],
        "drug key",
    )

    target_col = resolve_required_column(
        teacher,
        str(get_cfg(cfg, "target_col", "fused_prob_responder")),
        ["fused_prob_responder", "prob_responder", "target", "response"],
        "target",
    )

    # normalize spatial sample ids once before overlap logic
    spatial[sample_col] = spatial[sample_col].map(normalize_sample_id)

    # normalize teacher keys once before all joins and summaries
    teacher[teacher_sample_col] = teacher[teacher_sample_col].map(normalize_sample_id)
    teacher[drug_col] = teacher[drug_col].map(clean_text)
    teacher[drug_key_col] = teacher[drug_key_col].map(normalize_drug_key)
    teacher[target_col] = safe_numeric_series(teacher[target_col])

    return {
        "sample_col": sample_col,
        "teacher_sample_col": teacher_sample_col,
        "drug_col": drug_col,
        "drug_key_col": drug_key_col,
        "target_col": target_col,
    }


def resolve_manifest_feature_col(manifest: pd.DataFrame, cfg: dict[str, Any]) -> str | None:
    """resolve manifest feature column
    supports upstream and canonical names"""
    preferred = clean_text(get_cfg(cfg, "feature_name_col", "feature_name"))
    aliases = [preferred, "feature_name", "feature", "column", "column_name", "feature_original"]
    return first_existing_column(manifest.columns, aliases)


def get_manifest_included_features(manifest: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """extract included upstream features
    preserves manifest order"""
    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        return []

    work = manifest.copy()

    # included column is common in teacher builder feature_manifest.csv
    if "included" in work.columns:
        work = work[work["included"].map(bool_like)].copy()

    features = work[feature_col].dropna().astype(str).map(clean_text).tolist()

    # stable de-duplication while preserving order
    seen: set[str] = set()
    out: list[str] = []

    for feature in features:
        if feature and feature not in seen:
            seen.add(feature)
            out.append(feature)

    return out


# ============================================================
# SAMPLE SELECTION AND SPLITTING
# ============================================================


def matched_sample_ids(
    spatial: pd.DataFrame,
    teacher: pd.DataFrame,
    sample_col: str,
    teacher_sample_col: str,
) -> list[str]:
    """find samples with spatial features and labels
    one sample set for current modeling run"""
    spatial_ids = set(spatial[sample_col].dropna().map(normalize_sample_id))
    teacher_ids = set(teacher[teacher_sample_col].dropna().map(normalize_sample_id))

    spatial_ids.discard("")
    teacher_ids.discard("")

    return sorted(spatial_ids & teacher_ids)


def select_run_samples(
    matched_ids: Sequence[str],
    available_ids: Sequence[str],
    cfg: dict[str, Any],
) -> list[str]:
    """select samples for this run
    validation list preferred in test mode"""
    selected = set(matched_ids)

    # Step 01 captures the intended 10 labeled sample set
    if bool_like(get_cfg(cfg, "use_validation_sample_filter", True)) and available_ids:
        selected = selected & set(available_ids)

    # Fallback if step 01 sample list is absent
    elif bool_like(get_cfg(cfg, "test_mode", False)) and bool_like(get_cfg(cfg, "limit_training_to_test_samples", True)):
        n = get_cfg(cfg, "test_n_samples", None)
        if n not in [None, ""]:
            selected = set(sorted(selected)[: int(n)])

    return sorted(selected)


def build_sample_overlap_report(
    spatial: pd.DataFrame,
    teacher: pd.DataFrame,
    sample_col: str,
    teacher_sample_col: str,
    selected_samples: Sequence[str],
) -> pd.DataFrame:
    """summarize sample overlap
    selected flag tracks run cohort"""
    spatial_ids = set(spatial[sample_col].dropna().map(normalize_sample_id))
    teacher_ids = set(teacher[teacher_sample_col].dropna().map(normalize_sample_id))
    all_ids = sorted((spatial_ids | teacher_ids) - {""})
    selected = set(selected_samples)

    rows = []

    for sample_id in all_ids:
        in_spatial = sample_id in spatial_ids
        in_teacher = sample_id in teacher_ids
        status = "matched" if in_spatial and in_teacher else "spatial_only" if in_spatial else "teacher_only"

        rows.append(
            {
                "sample_id": sample_id,
                "status": status,
                "in_spatial": int(in_spatial),
                "in_teacher": int(in_teacher),
                "selected_for_modeling": int(sample_id in selected),
            }
        )

    return pd.DataFrame(rows)


def create_sample_split(selected_samples: Sequence[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """create grouped sample split
    one split label per sample_id"""
    samples = sorted({normalize_sample_id(x) for x in selected_samples if normalize_sample_id(x)})
    random_state = int(get_cfg(cfg, "random_state", 42))
    test_size = float(get_cfg(cfg, "test_size", 0.20))

    if len(samples) == 0:
        raise ValueError("No selected samples available for split")

    # With one sample, holdout is impossible without losing train data
    if len(samples) == 1:
        return pd.DataFrame({"sample_id": samples, "split": ["train"]})

    rng = np.random.default_rng(random_state)
    shuffled = np.array(samples, dtype=object)
    rng.shuffle(shuffled)

    # ceil gives 2 test samples for 10 at test_size 0.2
    n_test = int(math.ceil(len(samples) * test_size))
    n_test = max(1, min(n_test, len(samples) - 1))

    test_samples = set(shuffled[:n_test].tolist())

    rows = []
    for sample_id in samples:
        rows.append(
            {
                "sample_id": sample_id,
                "split": "test" if sample_id in test_samples else "train",
                "split_group_col": str(get_cfg(cfg, "split_group_col", "sample_id")),
                "split_strategy": str(get_cfg(cfg, "split_strategy", "group_holdout")),
                "random_state": random_state,
                "test_size": test_size,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FEATURE FILTERING AND MANIFEST BUILDING
# ============================================================


def configured_leakage_columns(cfg: dict[str, Any]) -> set[str]:
    """configured leakage set
    exact names from YAML"""
    return {clean_text(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or [] if clean_text(x)}


def nonfeature_columns(cfg: dict[str, Any], cols: dict[str, str]) -> set[str]:
    """identifier and metadata columns
    never eligible as spatial features"""
    passthrough = {clean_text(x) for x in get_cfg(cfg, "metadata_passthrough_columns", []) or [] if clean_text(x)}

    base = {
        cols["sample_col"],
        cols["teacher_sample_col"],
        cols["drug_col"],
        cols["drug_key_col"],
        cols["target_col"],
        str(get_cfg(cfg, "slide_col", "slide_id")),
        "row_id",
        "split",
        "index",
        "Unnamed: 0",
    }

    return base | passthrough | configured_leakage_columns(cfg)


def leakage_name_terms(cfg: dict[str, Any]) -> list[str]:
    """name based leakage terms
    catches teacher derived columns"""
    terms = get_cfg(
        cfg,
        "leakage_name_terms",
        [
            "teacher",
            "target",
            "response",
            "responder",
            "prob_responder",
            "fused_prob",
            "histology_prob",
            "expression_prob",
            "label",
            "drug",
            "treatment",
        ],
    )

    return [clean_column_name(x) for x in terms]


def nonfeature_name_terms(cfg: dict[str, Any]) -> list[str]:
    """metadata like name terms
    avoids paths and loader notes"""
    terms = get_cfg(
        cfg,
        "nonfeature_name_terms",
        [
            "path",
            "file",
            "h5ad",
            "loaded_from",
            "status",
            "error",
            "notes",
            "summary",
            "metadata",
            "format",
            "timepoint",
        ],
    )

    return [clean_column_name(x) for x in terms]


def feature_status_by_name(feature: str, cfg: dict[str, Any], cols: dict[str, str]) -> str:
    """classify feature by name
    exact leakage checked before terms"""
    clean = clean_column_name(feature)
    exact_nonfeatures = {clean_column_name(x) for x in nonfeature_columns(cfg, cols)}
    exact_leakage = {clean_column_name(x) for x in configured_leakage_columns(cfg)}

    if clean in exact_nonfeatures:
        return "drop_metadata_or_identifier"

    if clean in exact_leakage:
        return "drop_configured_leakage"

    if any(term in clean for term in leakage_name_terms(cfg)):
        return "drop_leakage_like_name"

    if any(term in clean for term in nonfeature_name_terms(cfg)):
        return "drop_metadata_like_name"

    return "candidate"


def infer_feature_group(feature: str) -> str:
    """infer broad feature group
    label used for reports and interpretation"""
    f = clean_column_name(feature)

    if f.startswith("f_") and f[2:].isdigit():
        return "clean_numeric_feature"

    if f.startswith("filtering__") or f in {"n_spots", "n_genes", "n_clusters"}:
        return "qc"

    if f.startswith(("mean__", "median__", "spot_fraction__")):
        return "program_summary"

    if f.startswith(("program__", "label__")):
        return "program_label"

    if "simple__" in f or "ucell__" in f or "gsva" in f:
        return "signature_score"

    if f.startswith("access_") or "accessibility" in f:
        return "accessibility"

    if f.startswith("hotspot__") or "hotspot" in f:
        return "hotspot"

    if f.startswith("metabolic_module__"):
        return "metabolic_module"

    if f.startswith("context_module__"):
        return "context_module"

    if f.startswith("concordance__"):
        return "context_alignment"

    if f.startswith("motif_"):
        return "motif"

    if f.startswith("pair_"):
        return "pair_relationship"

    if "slope_vs_depth" in f or "r2_vs_depth" in f:
        return "gradient"

    return "other_numeric"


def infer_feature_axis(feature: str) -> str:
    """infer rough biology axis
    simple rule based label"""
    f = clean_column_name(feature)

    if "tumor_epithelial" in f or "tumor_mask" in f:
        return "tumor_epithelial"

    if any(x in f for x in ["stromal", "stroma", "ecm", "collagen", "integrin"]):
        return "stromal_ecm"

    if "hypoxi" in f:
        return "hypoxia"

    if any(x in f for x in ["vascular", "angiogenic", "endothelial", "vegf"]):
        return "vascular"

    if any(x in f for x in ["t_cell", "interferon", "immune", "tcr"]):
        return "immune"

    if "myeloid" in f or "macrophage" in f:
        return "myeloid"

    if "b_plasma" in f or "b_cell" in f or "bcr" in f:
        return "b_cell_plasma"

    if any(x in f for x in ["prolifer", "cell_cycle", "mitotic", "g2m"]):
        return "proliferation"

    if "glycolysis" in f:
        return "glycolysis"

    if "oxidative_phosphorylation" in f or "oxphos" in f:
        return "oxidative_phosphorylation"

    if "fatty_acid" in f:
        return "fatty_acid_metabolism"

    if "glutamine" in f:
        return "glutamine_metabolism"

    if "tryptophan" in f or "kynurenine" in f:
        return "immune_suppression_metabolism"

    if "barrier" in f or "impermeable" in f:
        return "barrier"

    if any(x in f for x in ["distance", "nearest", "centroid"]):
        return "spatial_distance"

    if any(x in f for x in ["overlap", "interface", "adjacent"]):
        return "spatial_relationship"

    if any(x in f for x in ["component", "fragmentation", "topology"]):
        return "spatial_topology"

    return "other"


def numeric_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    """compute numeric feature stats
    prefix separates all versus labeled samples"""
    y = safe_numeric_series(values)
    nonmissing = y.dropna()
    n = int(len(y))
    n_nonmissing = int(nonmissing.shape[0])

    return {
        f"{prefix}_n_rows": n,
        f"{prefix}_nonmissing_count": n_nonmissing,
        f"{prefix}_missing_count": int(y.isna().sum()),
        f"{prefix}_missing_fraction": fraction(int(y.isna().sum()), n),
        f"{prefix}_unique_values": int(nonmissing.nunique()) if n_nonmissing else 0,
        f"{prefix}_mean": float(nonmissing.mean()) if n_nonmissing else np.nan,
        f"{prefix}_median": float(nonmissing.median()) if n_nonmissing else np.nan,
        f"{prefix}_std": float(nonmissing.std()) if n_nonmissing > 1 else np.nan,
        f"{prefix}_min": float(nonmissing.min()) if n_nonmissing else np.nan,
        f"{prefix}_max": float(nonmissing.max()) if n_nonmissing else np.nan,
    }


def choose_candidate_features(spatial: pd.DataFrame, manifest: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """choose initial feature candidates
    manifest preferred, numeric fallback"""
    manifest_features = get_manifest_included_features(manifest, cfg)
    present_manifest_features = [feature for feature in manifest_features if feature in spatial.columns]

    if present_manifest_features:
        return present_manifest_features

    # fallback only if manifest is missing or unusable
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    return [col for col in spatial.columns if col != sample_col]


def build_feature_quality_report(
    spatial: pd.DataFrame,
    matched_spatial: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
    cols: dict[str, str],
) -> pd.DataFrame:
    """profile and filter spatial features
    checks all samples and labeled samples"""
    candidate_features = choose_candidate_features(spatial, manifest, cfg)

    max_missing = float(get_cfg(cfg, "max_missing_fraction", 0.80))
    min_nonmissing = int(get_cfg(cfg, "min_nonmissing_values", 2))
    min_unique = int(get_cfg(cfg, "min_unique_values", 2))

    # labeled sample thresholds guard current train matrix usability
    train_max_missing = float(get_cfg(cfg, "max_training_missing_fraction", 0.95))
    train_min_nonmissing = int(get_cfg(cfg, "min_training_nonmissing_values", 2))
    train_min_unique = int(get_cfg(cfg, "min_training_unique_values", 2))

    rows: list[dict[str, Any]] = []

    for feature in candidate_features:
        name_status = feature_status_by_name(feature, cfg, cols)

        if feature not in spatial.columns:
            row = {
                "feature_original": feature,
                "status": "drop_missing_from_spatial_table",
                "included": False,
                "drop_reason": "missing_from_spatial_table",
                "feature_group": infer_feature_group(feature),
                "feature_axis": infer_feature_axis(feature),
            }
            rows.append(row)
            continue

        all_stats = numeric_stats(spatial[feature], "all_samples")
        train_stats = numeric_stats(matched_spatial[feature], "labeled_samples") if feature in matched_spatial.columns else {}

        status = "keep"
        reason = "usable_numeric_feature"

        # name based exclusion before numeric tests
        if name_status != "candidate":
            status = name_status
            reason = name_status
        elif all_stats["all_samples_nonmissing_count"] == 0:
            status = "drop_all_missing"
            reason = "all_samples_all_missing"
        elif all_stats["all_samples_nonmissing_count"] < min_nonmissing:
            status = "drop_too_few_nonmissing"
            reason = f"all_samples_nonmissing_lt_{min_nonmissing}"
        elif all_stats["all_samples_missing_fraction"] > max_missing:
            status = "drop_mostly_missing"
            reason = f"all_samples_missing_fraction_gt_{max_missing}"
        elif all_stats["all_samples_unique_values"] < min_unique:
            status = "drop_constant"
            reason = f"all_samples_unique_lt_{min_unique}"
        elif train_stats.get("labeled_samples_nonmissing_count", 0) == 0:
            status = "drop_labeled_all_missing"
            reason = "labeled_samples_all_missing"
        elif train_stats.get("labeled_samples_nonmissing_count", 0) < train_min_nonmissing:
            status = "drop_labeled_too_few_nonmissing"
            reason = f"labeled_samples_nonmissing_lt_{train_min_nonmissing}"
        elif train_stats.get("labeled_samples_missing_fraction", 1.0) > train_max_missing:
            status = "drop_labeled_mostly_missing"
            reason = f"labeled_samples_missing_fraction_gt_{train_max_missing}"
        elif train_stats.get("labeled_samples_unique_values", 0) < train_min_unique:
            status = "drop_labeled_constant"
            reason = f"labeled_samples_unique_lt_{train_min_unique}"

        row = {
            "feature_original": feature,
            "status": status,
            "included": status == "keep",
            "drop_reason": "" if status == "keep" else reason,
            "feature_group": infer_feature_group(feature),
            "feature_axis": infer_feature_axis(feature),
            **all_stats,
            **train_stats,
        }

        rows.append(row)

    report = pd.DataFrame(rows)

    if report.empty:
        return report

    return report.reset_index(drop=True)


def selected_features(feature_quality: pd.DataFrame) -> list[str]:
    """return kept features in original order
    order defines f_0 to f_n"""
    if feature_quality.empty:
        return []

    keep = feature_quality[feature_quality["included"].map(bool_like)].copy()
    return keep["feature_original"].tolist()


def build_feature_name_map(features: Sequence[str], cfg: dict[str, Any]) -> dict[str, str]:
    """map original features to model names
    f_# names keep matrices compact"""
    if not bool_like(get_cfg(cfg, "rename_features_to_clean_ids", True)):
        return {feature: feature for feature in features}

    # avoid renaming if all features are already clean f_# names
    if features and all(bool(re.match(r"^f_\d+$", str(feature))) for feature in features):
        return {feature: feature for feature in features}

    return {feature: f"f_{i}" for i, feature in enumerate(features)}


def build_model_feature_manifest(
    feature_quality: pd.DataFrame,
    feature_map: dict[str, str],
    upstream_manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """build canonical feature manifest
    downstream scripts read feature_name"""
    if feature_quality.empty:
        return pd.DataFrame()

    out = feature_quality.copy()
    out["feature_name"] = out["feature_original"].map(feature_map).fillna("")
    out["feature_clean"] = out["feature_name"]
    out["kept_for_modeling"] = out["included"].astype(int)
    out["source"] = "spatial_features"

    # preserve key upstream manifest fields with explicit prefix
    feature_col = resolve_manifest_feature_col(upstream_manifest, cfg) if not upstream_manifest.empty else None

    if feature_col is not None:
        upstream = upstream_manifest.copy()
        upstream[feature_col] = upstream[feature_col].astype(str).map(clean_text)
        keep_cols = [feature_col]

        for col in ["included", "reason", "nonmissing_fraction", "n_unique", "n_samples", "feature_group", "feature_axis"]:
            if col in upstream.columns and col not in keep_cols:
                keep_cols.append(col)

        upstream = upstream[keep_cols].drop_duplicates(feature_col)

        # upstream feature_name can collide with canonical feature_name
        upstream_key = "upstream_manifest_key"
        upstream = upstream.rename(columns={feature_col: upstream_key})

        rename = {col: f"upstream_{col}" for col in upstream.columns if col != upstream_key}
        upstream = upstream.rename(columns=rename)

        out = out.merge(upstream, left_on="feature_original", right_on=upstream_key, how="left")

    first_cols = [
        "feature_name",
        "feature_clean",
        "feature_original",
        "included",
        "kept_for_modeling",
        "status",
        "drop_reason",
        "feature_group",
        "feature_axis",
        "source",
    ]
    rest_cols = [col for col in out.columns if col not in first_cols]

    return out[first_cols + rest_cols].reset_index(drop=True)


def build_validated_spatial(
    spatial: pd.DataFrame,
    selected_samples: Sequence[str],
    selected: Sequence[str],
    feature_map: dict[str, str],
    cols: dict[str, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """build one row per selected sample
    clean feature names and numeric values"""
    sample_col = cols["sample_col"]
    selected_set = set(selected_samples)

    meta_cols = [sample_col]
    for col in ["slide_id", "dataset_id", "cancer_type"]:
        if col in spatial.columns and col not in meta_cols:
            meta_cols.append(col)

    keep_cols = meta_cols + [feature for feature in selected if feature in spatial.columns]
    out = spatial[keep_cols].copy()
    out = out[out[sample_col].isin(selected_set)].copy()

    # one spatial row per sample is required by many to one merge
    out = out.drop_duplicates(subset=[sample_col]).copy()

    for feature in selected:
        if feature in out.columns:
            out[feature] = safe_numeric_series(out[feature])

    rename_map = {old: new for old, new in feature_map.items() if old in out.columns}
    out = out.rename(columns=rename_map)

    return out.reset_index(drop=True)


# ============================================================
# TEACHER TARGET PREPARATION
# ============================================================


def clip01(series: pd.Series) -> pd.Series:
    """clip values to 0 to 1
    teacher probabilities should be bounded"""
    return series.clip(lower=0.0, upper=1.0)


def add_teacher_targets(teacher: pd.DataFrame, cols: dict[str, str], cfg: dict[str, Any]) -> pd.DataFrame:
    """add binary and residual targets
    leaves main target unchanged"""
    out = teacher.copy()
    sample_col = cols["teacher_sample_col"]
    drug_col = cols["drug_col"]
    drug_key_col = cols["drug_key_col"]
    target_col = cols["target_col"]

    threshold = float(get_cfg(cfg, "binary_threshold", 0.5))
    residual_mode = clean_text(get_cfg(cfg, "residual_mode", "drug_mean")).lower()

    out[target_col] = safe_numeric_series(out[target_col])

    if bool_like(get_cfg(cfg, "clip_target_01", True)):
        out[target_col] = clip01(out[target_col])

    # normalize drug key again after filtering
    out[drug_key_col] = out[drug_key_col].map(normalize_drug_key)

    out["response_binary"] = np.where(out[target_col].notna(), (out[target_col] >= threshold).astype(int), np.nan)
    out["response_threshold"] = threshold

    if residual_mode == "none":
        out["baseline_response"] = np.nan
        out["residual_response"] = out[target_col]
    elif residual_mode == "drug_mean":
        out["baseline_response"] = out.groupby(drug_key_col)[target_col].transform("mean")
        out["residual_response"] = out[target_col] - out["baseline_response"]
    elif residual_mode == "drug_median":
        out["baseline_response"] = out.groupby(drug_key_col)[target_col].transform("median")
        out["residual_response"] = out[target_col] - out["baseline_response"]
    elif residual_mode == "sample_mean":
        out["baseline_response"] = out.groupby(sample_col)[target_col].transform("mean")
        out["residual_response"] = out[target_col] - out["baseline_response"]
    elif residual_mode == "global_mean":
        out["baseline_response"] = out[target_col].mean()
        out["residual_response"] = out[target_col] - out["baseline_response"]
    else:
        raise ValueError("Unknown residual_mode: " + residual_mode)

    out["residual_mode"] = residual_mode
    out["residual_direction"] = np.where(out["residual_response"] >= 0, "above_expected", "below_expected")

    return out


def prepare_teacher_table(
    teacher: pd.DataFrame,
    selected_samples: Sequence[str],
    cols: dict[str, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """filter teacher rows and add targets
    selected labeled samples only"""
    sample_col = cols["teacher_sample_col"]
    drug_col = cols["drug_col"]
    target_col = cols["target_col"]
    selected_set = set(selected_samples)

    out = teacher[teacher[sample_col].isin(selected_set)].copy()
    out = out.dropna(subset=[sample_col, drug_col, target_col]).copy()
    out = add_teacher_targets(out, cols, cfg)

    return out.reset_index(drop=True)


def select_teacher_columns(teacher: pd.DataFrame, cols: dict[str, str], cfg: dict[str, Any]) -> pd.DataFrame:
    """keep stable teacher columns
    avoids carrying unexpected leakage fields"""
    preferred = list(get_cfg(cfg, "teacher_columns_to_keep", []) or [])

    required = [
        cols["teacher_sample_col"],
        cols["drug_col"],
        cols["drug_key_col"],
        cols["target_col"],
        "response_binary",
        "response_threshold",
        "baseline_response",
        "residual_response",
        "residual_direction",
        "residual_mode",
    ]

    keep: list[str] = []

    for col in preferred + required:
        if col in teacher.columns and col not in keep:
            keep.append(col)

    # preserve confidence and availability annotations for QC only
    for col in teacher.columns:
        low = clean_column_name(col)
        if any(token in low for token in ["confidence", "ci_low", "ci_high", "available"]):
            if col not in keep:
                keep.append(col)

    return teacher[keep].copy()


# ============================================================
# MODEL TABLE AND MATRICES
# ============================================================


def build_modeling_table(
    teacher: pd.DataFrame,
    validated_spatial: pd.DataFrame,
    sample_split: pd.DataFrame,
    cols: dict[str, str],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """merge teacher rows with spatial features
    many teacher rows to one spatial row"""
    teacher_sample_col = cols["teacher_sample_col"]
    sample_col = cols["sample_col"]

    merged = teacher.merge(
        validated_spatial,
        left_on=teacher_sample_col,
        right_on=sample_col,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_spatial"),
    )

    if merged.empty:
        raise ValueError("Merged modeling table is empty")

    # canonical sample_id column expected by steps 03 and 04
    if "sample_id" not in merged.columns:
        merged["sample_id"] = merged[teacher_sample_col]

    split_map = sample_split[["sample_id", "split"]].drop_duplicates("sample_id")
    merged = merged.merge(split_map, on="sample_id", how="left")

    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), "sample_id"].drop_duplicates().tolist()
        raise ValueError("Rows missing sample split: " + compact_list(missing))

    # row_id gives exact row alignment for matrices and predictions
    merged = merged.reset_index(drop=True)
    merged.insert(0, "row_id", np.arange(len(merged), dtype=int))

    # stable column layout: ids, teacher targets, then features
    id_cols = ["row_id", "sample_id"]
    for col in ["slide_id", cols["teacher_sample_col"], cols["drug_col"], cols["drug_key_col"], "split"]:
        if col in merged.columns and col not in id_cols:
            id_cols.append(col)

    target_cols = [
        cols["target_col"],
        "response_binary",
        "response_threshold",
        "baseline_response",
        "residual_response",
        "residual_direction",
        "residual_mode",
    ]
    target_cols = [col for col in target_cols if col in merged.columns and col not in id_cols]

    annotation_cols = []
    for col in merged.columns:
        if col in id_cols or col in target_cols or col in feature_names:
            continue
        if any(token in clean_column_name(col) for token in ["confidence", "available", "modality", "ci_low", "ci_high", "tiles", "dataset", "cancer"]):
            annotation_cols.append(col)

    ordered = id_cols + target_cols + annotation_cols + list(feature_names)
    ordered = [col for col in ordered if col in merged.columns]
    remaining = [col for col in merged.columns if col not in ordered]

    return merged[ordered + remaining].copy()


def build_drug_dummies(modeling_table: pd.DataFrame, cols: dict[str, str], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """make treatment identity features
    global model uses these columns"""
    drug_key_col = cols["drug_key_col"]
    prefix = str(get_cfg(cfg, "drug_dummy_prefix", "drug__"))

    keys = modeling_table[drug_key_col].map(normalize_drug_key)
    unique_keys = sorted(keys.dropna().unique().tolist())

    dummy_map = {drug_key: f"{prefix}{safe_feature_token(drug_key)}" for drug_key in unique_keys}

    dummy_data: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for drug_key, dummy_col in dummy_map.items():
        dummy_data[dummy_col] = (keys == drug_key).astype(int)
        rows.append({"drug_key": drug_key, "dummy_feature": dummy_col})

    dummies = pd.DataFrame(dummy_data, index=modeling_table.index)
    manifest = pd.DataFrame(rows)

    return dummies, manifest


def build_feature_matrices(
    modeling_table: pd.DataFrame,
    clean_feature_names: Sequence[str],
    cols: dict[str, str],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """build X matrices
    spatial only plus optional drug dummies"""
    spatial_X = safe_numeric_frame(modeling_table[list(clean_feature_names)].copy())

    if bool_like(get_cfg(cfg, "include_drug_dummies", True)):
        drug_X, drug_manifest = build_drug_dummies(modeling_table, cols, cfg)
        X = pd.concat([spatial_X, drug_X], axis=1)
    else:
        drug_manifest = pd.DataFrame(columns=["drug_key", "dummy_feature"])
        X = spatial_X.copy()

    return X, spatial_X, drug_manifest


def build_target_vector(modeling_table: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """build aligned target output
    includes useful target variants"""
    target_col = cols["target_col"]

    keep = ["row_id", "sample_id", cols["drug_col"], cols["drug_key_col"], target_col]
    for col in ["response_binary", "baseline_response", "residual_response"]:
        if col in modeling_table.columns:
            keep.append(col)

    keep = [col for col in keep if col in modeling_table.columns]
    return modeling_table[keep].copy()


def build_split_assignments(modeling_table: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """build row level split table
    exact alignment helper for downstream"""
    keep = ["row_id", "sample_id", cols["drug_col"], cols["drug_key_col"], "split"]
    keep = [col for col in keep if col in modeling_table.columns]
    return modeling_table[keep].copy()


# ============================================================
# REPORTS AND SUMMARIES
# ============================================================


def build_sample_manifest(
    spatial: pd.DataFrame,
    teacher: pd.DataFrame,
    modeling_table: pd.DataFrame,
    cols: dict[str, str],
) -> pd.DataFrame:
    """summarize per sample coverage
    spatial teacher modeling counts"""
    sample_col = cols["sample_col"]
    teacher_sample_col = cols["teacher_sample_col"]
    drug_key_col = cols["drug_key_col"]

    spatial_ids = spatial[[sample_col]].drop_duplicates().rename(columns={sample_col: "sample_id"})
    spatial_ids["in_spatial"] = 1

    teacher_counts = (
        teacher.groupby(teacher_sample_col)
        .agg(teacher_rows=(drug_key_col, "size"), n_treatments=(drug_key_col, "nunique"))
        .reset_index()
        .rename(columns={teacher_sample_col: "sample_id"})
    )
    teacher_counts["in_teacher"] = 1

    model_counts = (
        modeling_table.groupby("sample_id")
        .agg(modeling_rows=(drug_key_col, "size"), modeling_treatments=(drug_key_col, "nunique"))
        .reset_index()
    )
    model_counts["in_modeling_dataset"] = 1

    out = spatial_ids.merge(teacher_counts, on="sample_id", how="outer")
    out = out.merge(model_counts, on="sample_id", how="outer")

    for col in ["in_spatial", "in_teacher", "in_modeling_dataset"]:
        out[col] = out[col].fillna(0).astype(int)

    for col in ["teacher_rows", "n_treatments", "modeling_rows", "modeling_treatments"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    return out.sort_values("sample_id").reset_index(drop=True)


def build_drug_summary(modeling_table: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """summarize target coverage by treatment
    one row per drug key"""
    drug_col = cols["drug_col"]
    drug_key_col = cols["drug_key_col"]
    target_col = cols["target_col"]

    if modeling_table.empty:
        return pd.DataFrame()

    return (
        modeling_table.groupby(drug_key_col)
        .agg(
            drug=(drug_col, "first"),
            n_rows=(target_col, "size"),
            n_samples=("sample_id", "nunique"),
            mean_response=(target_col, "mean"),
            median_response=(target_col, "median"),
            std_response=(target_col, "std"),
            min_response=(target_col, "min"),
            max_response=(target_col, "max"),
            mean_residual=("residual_response", "mean"),
            std_residual=("residual_response", "std"),
        )
        .reset_index()
        .sort_values(["n_samples", "n_rows", drug_key_col], ascending=[False, False, True])
    )


def build_target_summary(modeling_table: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """summarize target distributions
    continuous binary residual views"""
    target_col = cols["target_col"]
    rows: list[dict[str, Any]] = []

    for col in [target_col, "response_binary", "baseline_response", "residual_response"]:
        if col not in modeling_table.columns:
            continue

        values = safe_numeric_series(modeling_table[col]).dropna()

        if values.empty:
            rows.append({"target": col, "n": 0})
            continue

        rows.append(
            {
                "target": col,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else np.nan,
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
                "n_unique": int(values.nunique()),
            }
        )

    return pd.DataFrame(rows)


def build_split_summary(modeling_table: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """summarize grouped split
    confirms train test sample separation"""
    target_col = cols["target_col"]
    drug_key_col = cols["drug_key_col"]

    rows: list[dict[str, Any]] = []

    for split_name, df in modeling_table.groupby("split", dropna=False):
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(df)),
                "n_samples": int(df["sample_id"].nunique()),
                "n_treatments": int(df[drug_key_col].nunique()),
                "target_mean": float(df[target_col].mean()),
                "target_std": float(df[target_col].std()),
            }
        )

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def build_leakage_report(cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """report configured leakage fields
    documents what cannot enter X"""
    leakage_cols = sorted(configured_leakage_columns(cfg))
    rows: list[dict[str, Any]] = []

    for col in leakage_cols:
        row: dict[str, Any] = {"column": col, "configured_as_leakage": True}

        for table_name, df in tables.items():
            row[f"present_in_{table_name}"] = bool(col in df.columns)

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_matrix_report(X: pd.DataFrame, spatial_X: pd.DataFrame) -> pd.DataFrame:
    """summarize final X columns
    distinguishes spatial and drug dummy features"""
    rows: list[dict[str, Any]] = []

    spatial_cols = set(spatial_X.columns)

    for col in X.columns:
        values = safe_numeric_series(X[col])
        rows.append(
            {
                "feature_name": col,
                "feature_source": "spatial" if col in spatial_cols else "drug_dummy",
                "n_rows": int(len(values)),
                "n_missing": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()) if len(values) else np.nan,
                "n_unique": int(values.nunique(dropna=True)),
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def unmatched_tables(
    spatial: pd.DataFrame,
    teacher: pd.DataFrame,
    selected_samples: Sequence[str],
    cols: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """build unmatched sample reports
    helps audit test versus full scope"""
    selected = set(selected_samples)
    sample_col = cols["sample_col"]
    teacher_sample_col = cols["teacher_sample_col"]

    spatial_unmatched = spatial.loc[~spatial[sample_col].isin(selected), [sample_col]].drop_duplicates().copy()
    spatial_unmatched = spatial_unmatched.rename(columns={sample_col: "sample_id"})

    teacher_unmatched = teacher.loc[~teacher[teacher_sample_col].isin(selected)].copy()

    return spatial_unmatched.reset_index(drop=True), teacher_unmatched.reset_index(drop=True)


def summarize_build(
    cfg: dict[str, Any],
    paths: dict[str, Path | None],
    spatial: pd.DataFrame,
    teacher: pd.DataFrame,
    selected_samples: Sequence[str],
    feature_quality: pd.DataFrame,
    selected: Sequence[str],
    modeling_table: pd.DataFrame,
    X: pd.DataFrame,
    spatial_X: pd.DataFrame,
    sample_split: pd.DataFrame,
) -> list[str]:
    """create human readable summary
    compact run audit"""
    n_drug_dummy = int(X.shape[1] - spatial_X.shape[1])

    lines: list[str] = []
    lines.append("Spatial modeling dataset build summary")
    lines.append("")

    lines.append("Run settings")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  test_mode: {get_cfg(cfg, 'test_mode', '')}")
    lines.append(f"  test_n_samples: {get_cfg(cfg, 'test_n_samples', '')}")
    lines.append("")

    lines.append("Inputs")
    for key in ["model_input_numeric", "teacher_table", "feature_manifest", "validation_dir"]:
        lines.append(f"  {key}: {paths.get(key)}")
    lines.append("")

    lines.append("Loaded tables")
    lines.append(f"  spatial rows: {spatial.shape[0]}")
    lines.append(f"  spatial columns: {spatial.shape[1]}")
    lines.append(f"  teacher rows: {teacher.shape[0]}")
    lines.append(f"  teacher columns: {teacher.shape[1]}")
    lines.append("")

    lines.append("Sample matching")
    lines.append(f"  selected samples: {len(selected_samples)}")
    lines.append(f"  train samples: {int((sample_split['split'] == 'train').sum())}")
    lines.append(f"  test samples: {int((sample_split['split'] == 'test').sum())}")
    lines.append("")

    lines.append("Feature filtering")
    lines.append(f"  candidate features checked: {feature_quality.shape[0]}")
    lines.append(f"  kept spatial features: {len(selected)}")

    if not feature_quality.empty:
        for status, count in feature_quality["status"].value_counts().items():
            lines.append(f"  {status}: {count}")

    lines.append("")
    lines.append("Final canonical outputs")
    lines.append(f"  modeling_table rows: {modeling_table.shape[0]}")
    lines.append(f"  modeling_table columns: {modeling_table.shape[1]}")
    lines.append(f"  X_features rows: {X.shape[0]}")
    lines.append(f"  X_features columns: {X.shape[1]}")
    lines.append(f"  spatial feature columns: {spatial_X.shape[1]}")
    lines.append(f"  drug dummy columns: {n_drug_dummy}")
    lines.append("")

    lines.append("Downstream contract")
    lines.append("  03 reads modeling_table.tsv, X_features.csv, y_target.csv, model_feature_manifest.csv, sample_split.tsv")
    lines.append("  04 reads the same files and removes drug dummy columns internally")
    lines.append("  split is grouped by sample_id")

    return lines


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main() -> int:
    """run dataset build
    write canonical and rich outputs"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    out_dir = get_output_dir(cfg)
    ensure_dir(out_dir)

    paths = get_input_paths(cfg)
    input_report = build_resolved_input_report(paths)

    print("Building spatial modeling dataset")
    print(f"Config: {Path(args.config)}")
    print(f"Output: {out_dir}")
    print()

    require_core_inputs(paths)

    # Save resolved paths before heavier processing
    write_table(input_report, out_dir / "resolved_input_paths.tsv")

    issue_df = load_validation_issues(paths["validation_dir"])
    enforce_validation_gate(issue_df, args)

    available_df = load_available_labeled_samples(paths["validation_dir"])
    available_ids = available_sample_ids(available_df, str(get_cfg(cfg, "sample_col", "sample_id")))

    tables = load_input_tables(cfg, paths)
    cols = normalize_core_columns(tables, cfg)

    spatial = tables["spatial"]
    teacher = tables["teacher"]
    manifest = tables["feature_manifest"]

    matched_ids = matched_sample_ids(spatial, teacher, cols["sample_col"], cols["teacher_sample_col"])
    selected_samples = select_run_samples(matched_ids, available_ids, cfg)

    if len(selected_samples) == 0:
        raise ValueError("No selected samples available after spatial teacher overlap and validation filtering")

    sample_split = create_sample_split(selected_samples, cfg)
    overlap_report = build_sample_overlap_report(spatial, teacher, cols["sample_col"], cols["teacher_sample_col"], selected_samples)

    matched_spatial_raw = spatial[spatial[cols["sample_col"]].isin(selected_samples)].copy()

    feature_quality = build_feature_quality_report(
        spatial=spatial,
        matched_spatial=matched_spatial_raw,
        manifest=manifest,
        cfg=cfg,
        cols=cols,
    )

    kept_original_features = selected_features(feature_quality)

    if len(kept_original_features) == 0:
        raise ValueError("No valid spatial features remained after all sample and labeled sample filtering")

    feature_map = build_feature_name_map(kept_original_features, cfg)
    clean_feature_names = [feature_map[feature] for feature in kept_original_features]

    model_manifest = build_model_feature_manifest(feature_quality, feature_map, manifest, cfg)

    validated_spatial = build_validated_spatial(
        spatial=spatial,
        selected_samples=selected_samples,
        selected=kept_original_features,
        feature_map=feature_map,
        cols=cols,
        cfg=cfg,
    )

    teacher_targets = prepare_teacher_table(teacher, selected_samples, cols, cfg)
    validated_teacher = select_teacher_columns(teacher_targets, cols, cfg)

    modeling_table = build_modeling_table(
        teacher=validated_teacher,
        validated_spatial=validated_spatial,
        sample_split=sample_split,
        cols=cols,
        feature_names=clean_feature_names,
    )

    X, spatial_X, drug_dummy_manifest = build_feature_matrices(modeling_table, clean_feature_names, cols, cfg)
    y_target = build_target_vector(modeling_table, cols)
    split_assignments = build_split_assignments(modeling_table, cols)

    sample_manifest = build_sample_manifest(spatial, teacher_targets, modeling_table, cols)
    drug_summary = build_drug_summary(modeling_table, cols)
    target_summary = build_target_summary(modeling_table, cols)
    split_summary = build_split_summary(modeling_table, cols)
    leakage_report = build_leakage_report(cfg, tables)
    feature_matrix_report = build_feature_matrix_report(X, spatial_X)
    unmatched_spatial, unmatched_teacher = unmatched_tables(spatial, teacher, selected_samples, cols)

    # Canonical outputs required by steps 03 and 04
    write_table(modeling_table, out_dir / "modeling_table.tsv")
    write_table(X, out_dir / "X_features.csv", sep=",")
    write_table(spatial_X, out_dir / "X_features_spatial_only.csv", sep=",")
    write_table(y_target, out_dir / "y_target.csv", sep=",")
    write_table(model_manifest, out_dir / "model_feature_manifest.csv", sep=",")
    write_table(sample_split, out_dir / "sample_split.tsv")
    write_table(split_assignments, out_dir / "split_assignments.tsv")
    write_table(leakage_report, out_dir / "leakage_excluded_columns.tsv")

    # Rich compatibility outputs retained from original step 02 style
    write_table(feature_quality, out_dir / "feature_quality_report.tsv")
    write_table(model_manifest, out_dir / "modeling_feature_manifest.tsv")
    write_csv_if_requested(model_manifest, out_dir / "modeling_feature_manifest.csv", cfg)
    write_table(validated_spatial, out_dir / "validated_spatial_features.tsv")
    write_csv_if_requested(validated_spatial, out_dir / "validated_spatial_features.csv", cfg)
    write_table(validated_teacher, out_dir / "validated_teacher_table.tsv")
    write_csv_if_requested(validated_teacher, out_dir / "validated_teacher_table.csv", cfg)
    write_table(teacher_targets, out_dir / "teacher_target_table.tsv")
    write_table(modeling_table, out_dir / "merged_model_input.tsv")
    write_csv_if_requested(modeling_table, out_dir / "merged_model_input.csv", cfg)
    write_table(drug_dummy_manifest, out_dir / "drug_dummy_manifest.tsv")
    write_table(sample_manifest, out_dir / "sample_manifest.tsv")
    write_table(drug_summary, out_dir / "drug_summary.tsv")
    write_table(target_summary, out_dir / "target_summary.tsv")
    write_table(split_summary, out_dir / "split_summary.tsv")
    write_table(feature_matrix_report, out_dir / "feature_matrix_report.tsv")
    write_table(overlap_report, out_dir / "sample_overlap_report.tsv")
    write_table(unmatched_spatial, out_dir / "unmatched_spatial_samples.tsv")
    write_table(unmatched_teacher, out_dir / "unmatched_teacher_rows.tsv")

    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(Path(args.config)),
        "output_dir": str(out_dir),
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "test_mode": get_cfg(cfg, "test_mode", ""),
        "selected_samples": selected_samples,
        "n_spatial_rows": int(spatial.shape[0]),
        "n_teacher_rows": int(teacher.shape[0]),
        "n_selected_samples": int(len(selected_samples)),
        "n_modeling_rows": int(modeling_table.shape[0]),
        "n_spatial_features": int(spatial_X.shape[1]),
        "n_drug_dummy_features": int(X.shape[1] - spatial_X.shape[1]),
        "n_total_features": int(X.shape[1]),
        "input_paths": {key: str(value) if value is not None else "" for key, value in paths.items()},
        "outputs": {
            "modeling_table": str(out_dir / "modeling_table.tsv"),
            "X_features": str(out_dir / "X_features.csv"),
            "y_target": str(out_dir / "y_target.csv"),
            "model_feature_manifest": str(out_dir / "model_feature_manifest.csv"),
            "sample_split": str(out_dir / "sample_split.tsv"),
        },
    }
    save_json(run_info, out_dir / "run_config.json")

    summary_lines = summarize_build(
        cfg=cfg,
        paths=paths,
        spatial=spatial,
        teacher=teacher,
        selected_samples=selected_samples,
        feature_quality=feature_quality,
        selected=kept_original_features,
        modeling_table=modeling_table,
        X=X,
        spatial_X=spatial_X,
        sample_split=sample_split,
    )
    write_text(out_dir / "dataset_build_summary.txt", summary_lines)

    print("DONE")
    print(f"Spatial rows loaded: {spatial.shape[0]:,}")
    print(f"Teacher rows loaded: {teacher.shape[0]:,}")
    print(f"Selected samples: {len(selected_samples):,}")
    print(f"Kept spatial features: {spatial_X.shape[1]:,}")
    print(f"Drug dummy features: {X.shape[1] - spatial_X.shape[1]:,}")
    print(f"Merged model rows: {modeling_table.shape[0]:,}")
    print(f"X feature columns: {X.shape[1]:,}")
    print(f"Wrote: {out_dir / 'modeling_table.tsv'}")
    print(f"Wrote: {out_dir / 'X_features.csv'}")
    print(f"Wrote: {out_dir / 'y_target.csv'}")
    print(f"Wrote: {out_dir / 'model_feature_manifest.csv'}")
    print(f"Wrote: {out_dir / 'sample_split.tsv'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
