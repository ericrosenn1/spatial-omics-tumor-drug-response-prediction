"""
Script:
    01_validate_prediction_inputs.py

Purpose:
    Validate the teacher_builder handoff files before spatial response modeling.

Role:
    First step in spatial_prediction_model.
    No model fitting.
    No feature engineering beyond validation summaries.
    No SHAP or prediction generation.

Pipeline position:
    teacher_builder/05_prediction_ready_teacher
        produces model_input_numeric.csv, visium_fused_teacher_table.tsv,
        feature_manifest.csv, and prediction_ready_training_table.tsv.

    spatial_prediction_model/01_validate_prediction_inputs.py
        verifies those files, confirms required columns, checks sample and
        sample treatment overlap, records feature availability, flags leakage
        fields, and writes validation reports.

    spatial_prediction_model/02_build_spatial_modeling_dataset.py
        consumes the validated handoff and builds the leakage safe modeling
        table used by downstream training steps.

Design contract:
    YAML driven paths and columns.
    Project relative and absolute paths supported.
    10 sample test mode and future 102 sample mode controlled only by YAML.
    Required handoff files are treated as hard requirements.
    Warnings are written for suspicious but nonfatal states.
    Errors are written and then stop the step with a nonzero exit code.

Validated inputs:
    model_input_numeric:
        One row per spatial sample.
        Contains sample_id and numeric spatial feature columns.

    teacher_table:
        One row per labeled sample treatment pair.
        Contains fused teacher response labels.

    feature_manifest:
        Feature table from teacher_builder step 05.
        Used to confirm feature names and included feature availability.

    training_table:
        Prediction ready sample treatment table.
        Contains teacher labels joined with spatial features.

Primary outputs:
    input_validation_summary.txt:
        Human readable validation report.

    input_table_shapes.tsv:
        Row, column, sample, treatment, and target counts.

    input_column_report.tsv:
        Per column data type, missingness, uniqueness, and role flags.

Additional outputs:
    resolved_input_paths.tsv
    required_column_check.tsv
    input_duplicate_report.tsv
    input_sample_overlap.tsv
    input_pair_overlap.tsv
    target_summary.tsv
    feature_manifest_check.tsv
    leakage_column_report.tsv
    available_labeled_samples.tsv
    validation_issues.tsv
    run_config.json

Notes:
    This script mirrors the teacher_builder style.
    It is intentionally conservative because the next script will create the
    actual model matrix.
"""

# future annotations: cleaner type hints on older Python builds
from __future__ import annotations

# pathlib: Windows paths and project relative paths handled safely
from pathlib import Path
from typing import Any
import argparse
import json
import sys

# numpy: NaN values for report fields that do not apply
import numpy as np
import pandas as pd
import yaml


# stable labels used in run_config.json and logs
SCRIPT_NAME = "01_validate_prediction_inputs.py"
STEP_NAME = "01_validate_prediction_inputs"
# default folder name mirrors numbered pipeline step
DEFAULT_OUTPUT_SUBDIR = "01_input_validation"


# ============================================================
# CONFIG HELPERS
# ============================================================


def parse_args() -> argparse.Namespace:
    """parse CLI args
    config path plus optional nonfatal mode"""
    # CLI parser, same pattern as teacher_builder step scripts
    parser = argparse.ArgumentParser(description="Validate spatial prediction inputs")
    # required config path, no hard coded project paths here
    parser.add_argument("--config", required=True, help="Path to spatial_prediction_model.yaml")
    # debug escape hatch, reports still written even with errors
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write reports but do not stop on validation errors",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """load YAML config
    UTF8 with BOM tolerant"""
    # hard fail before YAML parsing, clearer path error
    if not path.exists():
        raise FileNotFoundError(path)

    # utf-8-sig: tolerates Windows BOM from Notepad edits
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    # YAML root must be key value mapping, not list or scalar
    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")

    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """get config value
    fallback aware"""
    # explicit helper avoids repeated cfg.get calls downstream
    return cfg[key] if key in cfg else default


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve path value
    absolute or project relative"""
    # blank config values treated as unavailable paths
    if value in [None, ""]:
        return None

    # cast to string first, supports Path or YAML scalar input
    path = Path(str(value))

    # absolute paths kept unchanged, useful for external model files
    if path.is_absolute():
        return path

    # relative paths anchored at project_dir from YAML
    return project_dir / path


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """resolve step output folder
    output_root plus configured subdir"""
    # output_root is required, selected by run_name in YAML
    output_root = Path(str(cfg["output_root"]))
    # optional subdir overrides, safe empty dict fallback
    output_subdirs = get_cfg(cfg, "output_subdirs", {}) or {}
    # input_validation key maps to this step folder
    output_subdir = output_subdirs.get("input_validation", DEFAULT_OUTPUT_SUBDIR)

    return output_root / output_subdir


def ensure_dir(path: Path) -> None:
    """create folder
    parents included"""
    # parents=True creates full nested output path
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write text report
    newline joined"""
    # compact human report, one list item per line
    path.write_text("\n".join(lines), encoding="utf-8")


def save_json(data: dict[str, Any], path: Path) -> None:
    """write JSON report
    readable indent"""
    # standard JSON writer, UTF8 for Windows paths
    with open(path, "w", encoding="utf-8") as handle:
    # default=str handles Path and numpy scalar objects
        json.dump(data, handle, indent=2, default=str)


# ============================================================
# BASIC HELPERS
# ============================================================


def clean_text(value: Any) -> str:
    """clean scalar text
    empty for missing"""
    # pandas missing check, covers NaN and None
    if pd.isna(value):
        return ""

    # normalized text, avoids accidental whitespace mismatches
    return str(value).strip()


def normalize_key(value: Any) -> str:
    """normalize treatment key
    lowercase compact whitespace"""
    # drug keys compared in lowercase form
    text = clean_text(value).lower()
    # split plus join collapses repeated whitespace
    return " ".join(text.split())


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """numeric coercion
    invalid to missing"""
    # invalid strings become NaN, safer than raising
    return pd.to_numeric(series, errors="coerce")


def bool_like(value: Any) -> bool:
    """truthy parser
    tolerant string handling"""
    # already boolean, preserve exact value
    if isinstance(value, bool):
        return value

    # lower case string parser for YAML and table values
    text = clean_text(value).lower()

    # accepted truthy tokens, includes manifest included flag
    return text in {"1", "true", "t", "yes", "y", "included"}


def is_numeric_like(series: pd.Series, min_fraction: float = 0.80) -> bool:
    """numeric column check
    dtype or coercion fraction"""
    # true numeric dtype needs no coercion test
    if pd.api.types.is_numeric_dtype(series):
        return True

    # empty strings excluded from numeric fraction denominator
    nonmissing = series.notna() & series.astype(str).str.strip().ne("")

    # all missing column should not count as numeric
    if int(nonmissing.sum()) == 0:
        return False

    # coercion test, catches numeric strings in CSV output
    converted = pd.to_numeric(series[nonmissing], errors="coerce")
    # 0.80 threshold, tolerant of a few malformed entries
    return float(converted.notna().mean()) >= float(min_fraction)


def fraction(numerator: int | float, denominator: int | float) -> float:
    """safe fraction
    zero denominator returns NaN"""
    # NaN preferred over divide by zero crash in reports
    if denominator == 0:
        return float("nan")

    return float(numerator) / float(denominator)


def compact_list(values: list[Any], max_items: int = 12) -> str:
    """compact list display
    stable report text"""
    # remove empty values before display trimming
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    # report preview cap, avoids huge issue files
    shown = cleaned[:max_items]

    # append count summary when examples are truncated
    if len(cleaned) > max_items:
        shown.append(f"... {len(cleaned) - max_items} more")

    # semicolon separator, readable inside TSV cells
    return "; ".join(shown)


# ============================================================
# VALIDATION ISSUE TRACKING
# ============================================================


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    table: str,
    check_name: str,
    message: str,
    detail: Any = "",
) -> None:
    """append validation issue
    severity table check message"""
    # centralized issue schema, easier report writing later
    issues.append(
        {
            # severity values: error or warn
            "severity": severity,
            # table or logical source of failed check
            "table": table,
            # short machine readable check name
            "check_name": check_name,
            # human readable problem text
            "message": message,
            # optional examples or numeric details
            "detail": detail,
        }
    )


def count_issues(issues: list[dict[str, Any]], severity: str) -> int:
    """count issues by severity
    case insensitive"""
    # case insensitive matching, robust to future severity labels
    target = severity.lower()
    # generator expression avoids building temporary list
    return sum(1 for row in issues if clean_text(row.get("severity")).lower() == target)


# ============================================================
# TABLE LOADING
# ============================================================


def load_table(path: Path) -> pd.DataFrame:
    """load CSV or TSV
    low memory disabled"""
    # required file existence checked again before pandas load
    if not path.exists():
        raise FileNotFoundError(path)

    # suffix controls delimiter choice for normal pipeline files
    suffix = path.suffix.lower()

    # TSV handoff files from teacher_builder
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    # CSV handoff files, mainly model_input and manifest
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    # fallback for nonstandard names
    # delimiter sniffing fallback for unusual extensions
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def get_input_path_map(cfg: dict[str, Any]) -> dict[str, Path | None]:
    """resolve configured inputs
    teacher builder handoff keys"""
    # project_dir anchors all relative YAML inputs
    project_dir = Path(str(cfg["project_dir"]))

    # fixed logical names, used throughout validation outputs
    return {
        # optional folder, helpful provenance but not loaded as table
        "teacher_builder_output_dir": resolve_path(project_dir, get_cfg(cfg, "teacher_builder_output_dir")),
        # spatial features, one row per sample
        "model_input_numeric": resolve_path(project_dir, get_cfg(cfg, "model_input_numeric")),
        # fused teacher labels, one row per sample treatment
        "teacher_table": resolve_path(project_dir, get_cfg(cfg, "teacher_table")),
        # included feature names and feature groups
        "feature_manifest": resolve_path(project_dir, get_cfg(cfg, "feature_manifest")),
        # joined label plus spatial feature table
        "training_table": resolve_path(project_dir, get_cfg(cfg, "training_table")),
    }


def build_resolved_path_table(path_map: dict[str, Path | None]) -> pd.DataFrame:
    """summarize resolved paths
    existence and size"""
    # rows collected then converted once to DataFrame
    rows: list[dict[str, Any]] = []

    # each configured path gets one summary row
    for key, path in path_map.items():
        # None safe existence check
        exists = bool(path is not None and path.exists())
        # file versus folder distinction, important for inputs
        is_file = bool(exists and path.is_file())
        is_dir = bool(exists and path.is_dir())
        # size only meaningful for files, not directories
        size_bytes = int(path.stat().st_size) if is_file else np.nan

        # row schema mirrors resolved_input_paths.tsv
        rows.append(
            {
                "config_key": key,
                "resolved_path": str(path) if path is not None else "",
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "suffix": path.suffix if path is not None else "",
                "size_bytes": size_bytes,
            }
        )

    return pd.DataFrame(rows)


def validate_input_paths(path_map: dict[str, Path | None], issues: list[dict[str, Any]]) -> None:
    """validate required input paths
    missing files as errors"""
    # hard requirements for downstream modeling
    required_files = [
        "model_input_numeric",
        "teacher_table",
        "feature_manifest",
        "training_table",
    ]

    # each required file checked independently
    for key in required_files:
        path = path_map.get(key)

        # None means key absent or blank in YAML
        if path is None:
            add_issue(issues, "error", key, "path_config", f"Missing config key: {key}")
            continue

        # missing file blocks this step
        if not path.exists():
            add_issue(issues, "error", key, "path_exists", f"Input file not found: {path}")
            continue

        # directory passed where file expected
        if not path.is_file():
            add_issue(issues, "error", key, "path_is_file", f"Input path is not a file: {path}")
            continue

        # zero byte file often means interrupted write
        if path.stat().st_size == 0:
            add_issue(issues, "error", key, "path_nonempty", f"Input file is empty: {path}")

    # teacher output folder is provenance, not required table load
    teacher_dir = path_map.get("teacher_builder_output_dir")

    # warning only, individual files are checked above
    if teacher_dir is not None and not teacher_dir.exists():
        add_issue(
            issues,
            "warn",
            "teacher_builder_output_dir",
            "path_exists",
            f"Teacher builder output folder not found: {teacher_dir}",
        )


def load_all_inputs(path_map: dict[str, Path | None], issues: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """load all required input tables
    records load errors"""
    # loaded tables keyed by logical input name
    loaded: dict[str, pd.DataFrame] = {}

    # only real table files are attempted
    for key in ["model_input_numeric", "teacher_table", "feature_manifest", "training_table"]:
        path = path_map.get(key)

        # path issue already recorded, skip pandas load
        if path is None or not path.exists() or not path.is_file():
            continue

        # pandas load can still fail from delimiter or corruption
        try:
            loaded[key] = load_table(path)
            # load exception captured as validation error
        except Exception as exc:
            add_issue(
                issues,
                "error",
                key,
                "load_table",
                f"Could not load table: {path}",
                f"{type(exc).__name__}: {exc}",
            )

    return loaded


# ============================================================
# COLUMN EXPECTATIONS
# ============================================================


def get_required_columns(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """required columns by table
    config driven names"""
    # column names all configurable for future schema changes
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    slide_col = str(get_cfg(cfg, "slide_col", "slide_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    # fused_prob_responder is default continuous teacher target
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    # confidence optional, useful later but not model target
    confidence_col = str(get_cfg(cfg, "confidence_col", "fused_confidence"))

    # only minimal required columns enforced in step 01
    return {
        "model_input_numeric": [sample_col],
        "teacher_table": [sample_col, drug_col, drug_key_col, target_col],
        "training_table": [sample_col, drug_col, drug_key_col, target_col],
        "feature_manifest": [],
        "optional_teacher_table": [slide_col, confidence_col],
        "optional_training_table": [slide_col, confidence_col],
    }


def resolve_manifest_feature_col(manifest: pd.DataFrame, cfg: dict[str, Any]) -> str | None:
    """find manifest feature column
    YAML name then known aliases"""
    # YAML can override manifest feature column name
    preferred = clean_text(get_cfg(cfg, "feature_name_col", "feature_name"))
    # aliases cover earlier pipeline naming variants
    aliases = [preferred, "feature", "feature_name", "column", "column_name"]

    # first matching alias wins, deterministic order
    for col in aliases:
        if col in manifest.columns:
            return col

    return None


def validate_required_columns(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """validate required columns
    required and optional records"""
    # expected columns by table from config aware helper
    required = get_required_columns(cfg)
    # output rows for required_column_check.tsv
    rows: list[dict[str, Any]] = []

    # required checks only for the three core data tables
    for table_name in ["model_input_numeric", "teacher_table", "training_table"]:
        df = tables.get(table_name)
        expected = required.get(table_name, [])

        for col in expected:
            # df None safe for failed table loads
            present = bool(df is not None and col in df.columns)

            # report row written for every expected column
            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "required": True,
                    "present": present,
                }
            )

            # missing required column blocks downstream steps
            if not present:
                add_issue(
                    issues,
                    "error",
                    table_name,
                    "required_column",
                    f"Required column missing: {col}",
                )

    # optional columns, report only
    # optional columns, report only
    # optional columns tracked but not fatal
    optional_map = {
        "teacher_table": required.get("optional_teacher_table", []),
        "training_table": required.get("optional_training_table", []),
    }

    # optional table checks use same output schema
    for table_name, expected in optional_map.items():
        df = tables.get(table_name)

        for col in expected:
            # optional presence captured for transparency
            present = bool(df is not None and col in df.columns)

            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "required": False,
                    "present": present,
                }
            )

            # warning only, step 02 can still continue
            if not present:
                add_issue(
                    issues,
                    "warn",
                    table_name,
                    "optional_column",
                    f"Optional column missing: {col}",
                )

    # feature manifest needs separate alias based check
    manifest = tables.get("feature_manifest")

    if manifest is not None:
        # resolved column may differ from configured preferred name
        feature_col = resolve_manifest_feature_col(manifest, cfg)
        present = feature_col is not None

        # explicit record for feature name column
        rows.append(
            {
                "table": "feature_manifest",
                "column": clean_text(get_cfg(cfg, "feature_name_col", "feature_name")),
                "required": True,
                "present": present,
                "resolved_column": feature_col or "",
            }
        )

            # no feature name column means manifest cannot be trusted
        if not present:
            add_issue(
                issues,
                "error",
                "feature_manifest",
                "feature_column",
                "Feature manifest has no recognized feature name column",
                "Expected one of feature_name, feature, column, column_name",
            )

        # feature group optional, used for summaries only
        feature_group_col = clean_text(get_cfg(cfg, "feature_group_col", "feature_group"))

        # group presence reported for transparency
        rows.append(
            {
                "table": "feature_manifest",
                "column": feature_group_col,
                "required": False,
                "present": feature_group_col in manifest.columns,
                "resolved_column": feature_group_col if feature_group_col in manifest.columns else "",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# TABLE SUMMARIES
# ============================================================


def build_table_shapes(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize table sizes
    samples treatments targets"""
    # shared schema names, all YAML configurable
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    # shape rows become input_table_shapes.tsv
    rows: list[dict[str, Any]] = []

    # summarize every loaded input table
    for table_name, df in tables.items():
        # conditional counts, NaN when column not relevant
        row: dict[str, Any] = {
            "table": table_name,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            # unique samples if sample column exists
            "n_sample_id": int(df[sample_col].nunique()) if sample_col in df.columns else np.nan,
            # unique treatments if drug key exists
            "n_drug_key": int(df[drug_key_col].nunique()) if drug_key_col in df.columns else np.nan,
            "n_target_nonmissing": int(df[target_col].notna().sum()) if target_col in df.columns else np.nan,
            "n_target_missing": int(df[target_col].isna().sum()) if target_col in df.columns else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# column role labels support leakage review in reports
def infer_column_role(column: str, cfg: dict[str, Any], manifest_features: set[str]) -> str:
    """infer column role
    required metadata leakage feature"""
    # expected ID and target names from YAML
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    slide_col = str(get_cfg(cfg, "slide_col", "slide_id"))
    drug_col = str(get_cfg(cfg, "drug_col", "drug"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    # configured leakage exclusions, normalized to strings
    leakage_cols = set(str(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or [])
    # metadata columns allowed through later but not features
    passthrough_cols = set(str(x) for x in get_cfg(cfg, "metadata_passthrough_columns", []) or [])

    # target identified first, most important role
    if column == target_col:
        return "target"

    # ID fields, useful for joins but not model features
    if column in {sample_col, slide_col, drug_col, drug_key_col}:
        return "required_id"

    # teacher label or modality fields, exclude from model matrix
    if column in leakage_cols:
        return "leakage_excluded"

    # metadata kept for reports or passthrough
    if column in passthrough_cols:
        return "metadata_passthrough"

    # feature listed by teacher_builder manifest
    if column in manifest_features:
        return "manifest_feature"

    # everything else, inspected but not selected here
    return "other"


def build_column_report(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """build per column report
    dtype missingness uniqueness"""
    # manifest feature set used for role assignment
    manifest_features = get_manifest_feature_set(tables.get("feature_manifest"), cfg)
    rows: list[dict[str, Any]] = []

    # one report block per loaded table
    for table_name, df in tables.items():
        n_rows = len(df)

        # one row per column in input_column_report.tsv
        for col in df.columns:
            series = df[col]
            # missing values from pandas perspective
            n_missing = int(series.isna().sum())
            n_nonmissing = int(series.notna().sum())
            # unique values excluding missing, quick constant check
            n_unique = int(series.nunique(dropna=True))

            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "role": infer_column_role(col, cfg, manifest_features),
                    "dtype": str(series.dtype),
                    # strict dtype check, no coercion
                    "is_numeric_dtype": bool(pd.api.types.is_numeric_dtype(series)),
                    # tolerant numeric check for CSV string numbers
                    "is_numeric_like": bool(is_numeric_like(series)),
                    "n_missing": n_missing,
                    "n_nonmissing": n_nonmissing,
                    # safe fraction helper prevents zero division
                    "missing_fraction": fraction(n_missing, n_rows),
                    "n_unique": n_unique,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# DUPLICATE AND OVERLAP CHECKS
# ============================================================


def build_duplicate_report(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """check duplicate keys
    sample and sample treatment"""
    # keys expected to be unique in each table
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    # model input unique by sample, label tables unique by pair
    checks = [
        ("model_input_numeric", [sample_col]),
        ("teacher_table", [sample_col, drug_key_col]),
        ("training_table", [sample_col, drug_key_col]),
    ]

    # duplicate report rows accumulated here
    rows: list[dict[str, Any]] = []

    # run every uniqueness check independently
    for table_name, key_cols in checks:
        df = tables.get(table_name)

        if df is None:
            continue

        # missing key columns prevents duplicate check
        missing_keys = [col for col in key_cols if col not in df.columns]

        if missing_keys:
            rows.append(
                {
                    "table": table_name,
                    "key_columns": ",".join(key_cols),
                    "n_duplicate_rows": np.nan,
                    "n_duplicate_groups": np.nan,
                    "status": "not_checked_missing_keys",
                    "detail": compact_list(missing_keys),
                }
            )
            continue

        # keep=False marks every row in duplicate groups
        duplicated_mask = df.duplicated(key_cols, keep=False)
        # duplicate row count, not duplicate group count
        n_duplicate_rows = int(duplicated_mask.sum())

        # duplicate examples collected only when needed
        if n_duplicate_rows > 0:
            # one row per duplicated key group
            duplicate_groups = df.loc[duplicated_mask, key_cols].drop_duplicates()
            n_duplicate_groups = int(len(duplicate_groups))
            # compact preview of problematic key values
            detail = compact_list(
                ["|".join(map(str, values)) for values in duplicate_groups.head(12).to_numpy().tolist()]
            )

            # duplicate keys warning, downstream can still decide
            add_issue(
                issues,
                "warn",
                table_name,
                "duplicate_keys",
                f"Duplicate rows detected for key: {','.join(key_cols)}",
                detail,
            )
        else:
            # explicit zero values for clean reports
            n_duplicate_groups = 0
            detail = ""

        # status string simplifies manual inspection
        rows.append(
            {
                "table": table_name,
                "key_columns": ",".join(key_cols),
                "n_duplicate_rows": n_duplicate_rows,
                "n_duplicate_groups": n_duplicate_groups,
                "status": "ok" if n_duplicate_rows == 0 else "duplicates_found",
                "detail": detail,
            }
        )

    return pd.DataFrame(rows)


def get_sample_set(df: pd.DataFrame | None, sample_col: str) -> set[str]:
    """extract sample set
    empty if unavailable"""
    # unavailable table or missing column gives empty set
    if df is None or sample_col not in df.columns:
        return set()

    # cast to string, clean whitespace, drop missing samples
    return set(df[sample_col].dropna().astype(str).map(clean_text))


def build_sample_overlap(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """compare sample sets
    model teacher training"""
    # sample ID name shared by all three tables
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))

    # model samples may include unlabeled future prediction samples
    model_samples = get_sample_set(tables.get("model_input_numeric"), sample_col)
    # teacher samples are labeled by fused teacher
    teacher_samples = get_sample_set(tables.get("teacher_table"), sample_col)
    # training samples are labeled and joined with features
    training_samples = get_sample_set(tables.get("training_table"), sample_col)

    # directional checks, left should be contained in right
    comparisons = [
        ("teacher_in_model", teacher_samples, model_samples),
        ("training_in_model", training_samples, model_samples),
        ("training_in_teacher", training_samples, teacher_samples),
        ("teacher_in_training", teacher_samples, training_samples),
    ]

    rows: list[dict[str, Any]] = []

    # set arithmetic gives overlap and missing samples
    for name, left, right in comparisons:
        # samples present on left but absent on right
        missing = sorted(left - right)
        # samples shared by both sides
        overlap = sorted(left & right)

        rows.append(
            {
                "comparison": name,
                "n_left": int(len(left)),
                "n_right": int(len(right)),
                "n_overlap": int(len(overlap)),
                "n_missing_from_right": int(len(missing)),
                "missing_examples": compact_list(missing),
            }
        )

        # missing sample examples become issues only when present
        if missing:
            # fatal when labels lack spatial features
            severity = "error" if name in {"teacher_in_model", "training_in_model"} else "warn"
            add_issue(
                issues,
                severity,
                "sample_overlap",
                name,
                f"Samples missing in right side of comparison: {name}",
                compact_list(missing),
            )

    return pd.DataFrame(rows)


# pair identity: sample_id plus normalized drug_key
def pair_set(df: pd.DataFrame | None, sample_col: str, drug_key_col: str) -> set[tuple[str, str]]:
    """extract sample treatment pairs
    normalized drug key"""
    # no table means no pairs to compare
    if df is None:
        return set()

    # both columns required for pair extraction
    if sample_col not in df.columns or drug_key_col not in df.columns:
        return set()

    # drop incomplete pair records before set conversion
    subset = df[[sample_col, drug_key_col]].dropna().copy()
    # clean sample IDs before tuple creation
    subset[sample_col] = subset[sample_col].astype(str).map(clean_text)
    # normalize drugs to avoid case or whitespace mismatches
    subset[drug_key_col] = subset[drug_key_col].map(normalize_key)

    # numpy rows converted into hashable tuple pairs
    return set(map(tuple, subset.to_numpy().tolist()))


def build_pair_overlap(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """compare sample treatment pairs
    teacher versus training"""
    # pair columns configured in YAML
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))

    # teacher pairs before feature join
    teacher_pairs = pair_set(tables.get("teacher_table"), sample_col, drug_key_col)
    # training pairs after feature join
    training_pairs = pair_set(tables.get("training_table"), sample_col, drug_key_col)

    # teacher labels lost before training table
    missing_training = sorted(teacher_pairs - training_pairs)
    # training rows without teacher label provenance
    extra_training = sorted(training_pairs - teacher_pairs)

    # two directional overlap summaries
    rows = [
        {
            "comparison": "teacher_pairs_in_training",
            "n_left": int(len(teacher_pairs)),
            "n_right": int(len(training_pairs)),
                # same overlap value used for both directions
            "n_overlap": int(len(teacher_pairs & training_pairs)),
            "n_missing_from_right": int(len(missing_training)),
            "missing_examples": compact_list([f"{a}|{b}" for a, b in missing_training]),
        },
        {
            "comparison": "training_pairs_in_teacher",
            "n_left": int(len(training_pairs)),
            "n_right": int(len(teacher_pairs)),
            "n_overlap": int(len(teacher_pairs & training_pairs)),
            "n_missing_from_right": int(len(extra_training)),
            "missing_examples": compact_list([f"{a}|{b}" for a, b in extra_training]),
        },
    ]

    # missing teacher pairs are fatal for model training
    if missing_training:
        add_issue(
            issues,
            "error",
            "pair_overlap",
            "teacher_pairs_in_training",
            "Teacher sample treatment pairs missing from training table",
            compact_list([f"{a}|{b}" for a, b in missing_training]),
        )

    # extra training pairs suspicious but not always fatal
    if extra_training:
        add_issue(
            issues,
            "warn",
            "pair_overlap",
            "training_pairs_in_teacher",
            "Training sample treatment pairs not present in teacher table",
            compact_list([f"{a}|{b}" for a, b in extra_training]),
        )

    return pd.DataFrame(rows)


# ============================================================
# TARGET AND FEATURE CHECKS
# ============================================================


def summarize_target(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """summarize target columns
    range and missingness checks"""
    # target column from YAML, default fused teacher probability
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))
    rows: list[dict[str, Any]] = []

    # check both raw fused labels and joined training table
    for table_name in ["teacher_table", "training_table"]:
        df = tables.get(table_name)

        # absent table or target already covered by earlier checks
        if df is None or target_col not in df.columns:
            continue

        # coercion protects stats from string-typed CSV values
        y = safe_numeric_series(df[target_col])
        n_rows = int(len(y))
        n_nonmissing = int(y.notna().sum())
        n_missing = int(y.isna().sum())
        # probability-like target should stay inside 0 to 1
        n_outside_01 = int(((y < 0) | (y > 1)).sum())

        # one compact row for target_summary.tsv
        row = {
            "table": table_name,
            "target_col": target_col,
            "n_rows": n_rows,
            "n_nonmissing": n_nonmissing,
            "n_missing": n_missing,
            "missing_fraction": fraction(n_missing, n_rows),
            # conditional stats avoid reductions on empty targets
            "min": float(y.min()) if n_nonmissing else np.nan,
            "q25": float(y.quantile(0.25)) if n_nonmissing else np.nan,
            "mean": float(y.mean()) if n_nonmissing else np.nan,
            "median": float(y.median()) if n_nonmissing else np.nan,
            "q75": float(y.quantile(0.75)) if n_nonmissing else np.nan,
            "max": float(y.max()) if n_nonmissing else np.nan,
            "std": float(y.std()) if n_nonmissing else np.nan,
            "n_unique": int(y.nunique(dropna=True)),
            "n_outside_0_1": n_outside_01,
        }
        rows.append(row)

        # no labels means downstream training cannot run
        if n_nonmissing == 0:
            add_issue(
                issues,
                "error",
                table_name,
                "target_nonmissing",
                f"Target has no nonmissing values: {target_col}",
            )

        # warning only, step 02 can decide whether to clip or drop
        if n_outside_01 > 0:
            add_issue(
                issues,
                "warn",
                table_name,
                "target_range",
                f"Target values outside 0 to 1: {target_col}",
                n_outside_01,
            )

        # constant target makes model metrics uninformative
        if int(y.nunique(dropna=True)) <= 1 and n_nonmissing > 0:
            add_issue(
                issues,
                "warn",
                table_name,
                "target_variation",
                f"Target has one or fewer unique values: {target_col}",
            )

    return pd.DataFrame(rows)


def get_manifest_feature_set(manifest: pd.DataFrame | None, cfg: dict[str, Any]) -> set[str]:
    """feature set from manifest
    empty if unresolved"""
    if manifest is None:
        return set()

    # manifest column resolved through preferred name plus aliases
    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        return set()

    # set form supports fast membership checks in column role logic
    return set(manifest[feature_col].dropna().astype(str).map(clean_text))


def get_included_manifest_features(manifest: pd.DataFrame | None, cfg: dict[str, Any]) -> list[str]:
    """included manifest features
    supports included column"""
    if manifest is None:
        return []

    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        return []

    # copy before filtering, keep loaded manifest unchanged
    out = manifest.copy()

    # bool_like handles True, 1, yes, included, and related strings
    if "included" in out.columns:
        out = out[out["included"].map(bool_like)].copy()

    # clean feature names and drop blanks below
    features = out[feature_col].dropna().astype(str).map(clean_text).tolist()
    return [f for f in features if f]


def build_feature_manifest_check(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """validate feature manifest
    feature presence and numeric status"""
    # local aliases keep later checks readable
    manifest = tables.get("feature_manifest")
    model_input = tables.get("model_input_numeric")
    training_table = tables.get("training_table")

    # no manifest, no feature-level report
    if manifest is None:
        return pd.DataFrame()

    feature_col = resolve_manifest_feature_col(manifest, cfg)

    if feature_col is None:
        return pd.DataFrame()

    # leakage list is YAML controlled, not inferred here
    leakage_cols = set(str(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or [])
    included_features = get_included_manifest_features(manifest, cfg)

    rows: list[dict[str, Any]] = []

    # one report row per included feature
    for feature in included_features:
        # feature should appear in both spatial-only and training tables
        in_model = bool(model_input is not None and feature in model_input.columns)
        in_training = bool(training_table is not None and feature in training_table.columns)
        in_leakage = feature in leakage_cols

        # numeric-like allows values stored as strings but coercible
        model_numeric = bool(in_model and is_numeric_like(model_input[feature])) if model_input is not None else False
        training_numeric = bool(in_training and is_numeric_like(training_table[feature])) if training_table is not None else False

        rows.append(
            {
                "feature_name": feature,
                "in_model_input_numeric": in_model,
                "in_training_table": in_training,
                "model_input_numeric_like": model_numeric,
                "training_table_numeric_like": training_numeric,
                "in_leakage_excluded_columns": in_leakage,
            }
        )

    # table form used for report writing and issue extraction
    check = pd.DataFrame(rows)

    # empty included set is suspicious but not necessarily fatal
    if check.empty:
        add_issue(
            issues,
            "warn",
            "feature_manifest",
            "included_features",
            "No included features found in feature manifest",
        )
        return check

    # pandas masks pull failing feature names into concise lists
    missing_model = check.loc[check["in_model_input_numeric"] != True, "feature_name"].tolist()
    missing_training = check.loc[check["in_training_table"] != True, "feature_name"].tolist()
    # '&' combines masks, parentheses required by pandas precedence
    nonnumeric_model = check.loc[
        (check["in_model_input_numeric"] == True) & (check["model_input_numeric_like"] != True),
        "feature_name",
    ].tolist()
    leakage_features = check.loc[check["in_leakage_excluded_columns"] == True, "feature_name"].tolist()

    # warning, since step 02 may subset features before modeling
    if missing_model:
        add_issue(
            issues,
            "warn",
            "feature_manifest",
            "features_in_model_input",
            "Included manifest features missing from model_input_numeric",
            compact_list(missing_model),
        )

    # training table should carry every selected spatial feature
    if missing_training:
        add_issue(
            issues,
            "warn",
            "feature_manifest",
            "features_in_training_table",
            "Included manifest features missing from training_table",
            compact_list(missing_training),
        )

    # tree models need numeric matrix columns
    if nonnumeric_model:
        add_issue(
            issues,
            "warn",
            "model_input_numeric",
            "feature_numeric_type",
            "Included model input features not numeric like",
            compact_list(nonnumeric_model),
        )

    # leakage feature is fatal, model would learn teacher artifacts
    if leakage_features:
        add_issue(
            issues,
            "error",
            "feature_manifest",
            "feature_leakage",
            "Leakage columns appear as included model features",
            compact_list(leakage_features),
        )

    return check


def build_leakage_column_report(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """report leakage column presence
    inputs for step 02 exclusion"""
    # same YAML list used later by dataset builder for exclusion
    leakage_cols = [str(x) for x in get_cfg(cfg, "leakage_excluded_columns", []) or []]
    rows: list[dict[str, Any]] = []

    # report where each forbidden column appears
    for col in leakage_cols:
        row: dict[str, Any] = {"column": col}

        # dynamic field names create one flag per input table
        for table_name in ["model_input_numeric", "teacher_table", "training_table", "feature_manifest"]:
            df = tables.get(table_name)
            row[f"present_in_{table_name}"] = bool(df is not None and col in df.columns)

        rows.append(row)

    return pd.DataFrame(rows)


def build_available_labeled_samples(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """summarize labeled samples
    one row per sample"""
    # labeled table, one row per sample-treatment pair
    training_table = tables.get("training_table")

    if training_table is None:
        return pd.DataFrame()

    # YAML-controlled names preserve schema flexibility
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    drug_key_col = str(get_cfg(cfg, "drug_key_col", "drug_key"))
    target_col = str(get_cfg(cfg, "target_col", "fused_prob_responder"))

    if sample_col not in training_table.columns:
        return pd.DataFrame()

    # pandas named aggregation: output name -> source column and reducer
    agg_dict: dict[str, Any] = {
        "n_training_rows": (sample_col, "size"),
    }

    # treatment count optional, depends on drug_key availability
    if drug_key_col in training_table.columns:
        agg_dict["n_treatments"] = (drug_key_col, "nunique")

    # target range by sample helps spot label collapse
    if target_col in training_table.columns:
        agg_dict["mean_target"] = (target_col, "mean")
        agg_dict["min_target"] = (target_col, "min")
        agg_dict["max_target"] = (target_col, "max")

    # sample-level view of labeled training coverage
    return training_table.groupby(sample_col, as_index=False).agg(**agg_dict)


# ============================================================
# RUN SCOPE CHECKS
# ============================================================


def validate_run_scope(
    tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    """validate test versus full scope
    YAML controlled expectations"""
    # labeled training samples define test scope, not all spatial samples
    training_table = tables.get("training_table")

    if training_table is None:
        return

    # smoke-test expectations come from YAML only
    sample_col = str(get_cfg(cfg, "sample_col", "sample_id"))
    test_mode = bool(get_cfg(cfg, "test_mode", False))
    test_n_samples = get_cfg(cfg, "test_n_samples", None)

    if sample_col not in training_table.columns:
        return

    # unique labeled samples after sample-treatment expansion
    n_training_samples = int(training_table[sample_col].nunique())

    # full run ignores this test-mode sample-count check
    if test_mode and test_n_samples is not None:
        # robust parse, YAML may store numeric values as strings
        try:
            test_n = int(test_n_samples)
        except Exception:
            test_n = None

        # too many samples suggests stale outputs or wrong config
        if test_n is not None and n_training_samples > test_n:
            add_issue(
                issues,
                "warn",
                "training_table",
                "test_mode_sample_count",
                "Training table has more labeled samples than configured test_n_samples",
                f"observed={n_training_samples}; configured={test_n}",
            )

        # too few samples suggests incomplete teacher-builder handoff
        if test_n is not None and n_training_samples < test_n:
            add_issue(
                issues,
                "warn",
                "training_table",
                "test_mode_sample_count",
                "Training table has fewer labeled samples than configured test_n_samples",
                f"observed={n_training_samples}; configured={test_n}",
            )


# ============================================================
# SUMMARY TEXT
# ============================================================


def write_summary_text(
    path: Path,
    cfg: dict[str, Any],
    path_table: pd.DataFrame,
    shape_table: pd.DataFrame,
    target_summary: pd.DataFrame,
    feature_check: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    sample_overlap: pd.DataFrame,
    pair_overlap: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    """write human validation summary
    compact but complete"""
    # totals calculated once for summary header and footer
    n_errors = count_issues(issues, "error")
    n_warnings = count_issues(issues, "warn")

    # accumulate lines, then write report in one call
    lines: list[str] = []
    lines.append("Spatial prediction input validation summary")
    lines.append("")
    lines.append("Run settings")
    # provenance copied directly from YAML
    lines.append(f"  pipeline_name: {get_cfg(cfg, 'pipeline_name', '')}")
    lines.append(f"  run_name: {get_cfg(cfg, 'run_name', '')}")
    lines.append(f"  run_scope: {get_cfg(cfg, 'run_scope', '')}")
    lines.append(f"  test_mode: {get_cfg(cfg, 'test_mode', '')}")
    lines.append(f"  test_n_samples: {get_cfg(cfg, 'test_n_samples', '')}")
    lines.append(f"  output_root: {get_cfg(cfg, 'output_root', '')}")
    lines.append("")

    # resolved paths show exactly what files were checked
    lines.append("Input paths")
    # small report table, iterrows is fine here
    for _, row in path_table.iterrows():
        status = "exists" if bool(row.get("exists")) else "missing"
        lines.append(f"  {row['config_key']}: {status}: {row['resolved_path']}")
    lines.append("")

    # table dimensions provide first sanity check
    lines.append("Table shapes")
    for _, row in shape_table.iterrows():
        # split expression keeps long summary line readable
        lines.append(
            "  "
            + f"{row['table']}: rows={row['n_rows']}; columns={row['n_columns']}; "
            + f"samples={row.get('n_sample_id', np.nan)}; treatments={row.get('n_drug_key', np.nan)}"
        )
    lines.append("")

    # optional section, omitted if target report empty
    if not target_summary.empty:
        lines.append("Target summary")
        for _, row in target_summary.iterrows():
            lines.append(
                "  "
                + f"{row['table']}: nonmissing={row['n_nonmissing']}; "
                + f"mean={row['mean']:.4f}; min={row['min']:.4f}; max={row['max']:.4f}; "
                + f"unique={row['n_unique']}"
            )
        lines.append("")

    # confirms selected features are present and non-leakage
    if not feature_check.empty:
        # boolean columns sum to counts in pandas
        n_features = int(len(feature_check))
        n_in_model = int(feature_check["in_model_input_numeric"].sum())
        n_in_training = int(feature_check["in_training_table"].sum())
        n_leakage = int(feature_check["in_leakage_excluded_columns"].sum())
        lines.append("Feature manifest check")
        lines.append(f"  included features checked: {n_features}")
        lines.append(f"  present in model_input_numeric: {n_in_model}")
        lines.append(f"  present in training_table: {n_in_training}")
        lines.append(f"  leakage listed as features: {n_leakage}")
        lines.append("")

    # duplicate keys can silently break joins
    if not duplicate_report.empty:
        lines.append("Duplicate key check")
        for _, row in duplicate_report.iterrows():
            lines.append(
                "  "
                + f"{row['table']} on {row['key_columns']}: "
                + f"duplicate_rows={row['n_duplicate_rows']}; status={row['status']}"
            )
        lines.append("")

    # sample overlap across model input, teacher labels, training rows
    if not sample_overlap.empty:
        lines.append("Sample overlap")
        for _, row in sample_overlap.iterrows():
            lines.append(
                "  "
                + f"{row['comparison']}: overlap={row['n_overlap']}; "
                + f"missing={row['n_missing_from_right']}"
            )
        lines.append("")

    # stricter check, sample plus treatment pair agreement
    if not pair_overlap.empty:
        lines.append("Sample treatment pair overlap")
        for _, row in pair_overlap.iterrows():
            lines.append(
                "  "
                + f"{row['comparison']}: overlap={row['n_overlap']}; "
                + f"missing={row['n_missing_from_right']}"
            )
        lines.append("")

    # final pass/fail counts
    lines.append("Validation status")
    lines.append(f"  errors: {n_errors}")
    lines.append(f"  warnings: {n_warnings}")

    # detailed issue block only when needed
    if issues:
        lines.append("")
        lines.append("Issues")
        for issue in issues:
            lines.append(
                "  "
                + f"{issue['severity']} | {issue['table']} | {issue['check_name']} | "
                + f"{issue['message']} | {issue.get('detail', '')}"
            )
    else:
        lines.append("")
        lines.append("No validation issues detected")

    # shared writer handles newline joining
    write_text(path, lines)


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    """run validation step
    write reports then exit"""
    # direct CLI path, also used by the pipeline runner
    args = parse_args()
    cfg = load_config(Path(args.config))

    # output folder derived from YAML output_root
    out_dir = get_output_dir(cfg)
    ensure_dir(out_dir)

    # shared issue list, validators append dictionaries
    issues: list[dict[str, Any]] = []

    print("Validating spatial prediction inputs")
    print(f"Config: {Path(args.config)}")
    print(f"Output: {out_dir}")

    # path resolution and existence checks
    path_map = get_input_path_map(cfg)
    path_table = build_resolved_path_table(path_map)
    validate_input_paths(path_map, issues)

    # path report written even if later checks fail
    path_table.to_csv(out_dir / "resolved_input_paths.tsv", sep="\t", index=False)

    # load tables only after path validation
    tables = load_all_inputs(path_map, issues)

    # core table checks
    required_column_check = validate_required_columns(tables, cfg, issues)
    shape_table = build_table_shapes(tables, cfg)
    column_report = build_column_report(tables, cfg)
    duplicate_report = build_duplicate_report(tables, cfg, issues)
    sample_overlap = build_sample_overlap(tables, cfg, issues)
    pair_overlap = build_pair_overlap(tables, cfg, issues)
    # target and feature checks after schema checks
    target_summary = summarize_target(tables, cfg, issues)
    feature_check = build_feature_manifest_check(tables, cfg, issues)
    leakage_report = build_leakage_column_report(tables, cfg)
    labeled_samples = build_available_labeled_samples(tables, cfg)

    # run scope checks last, after table state known
    validate_run_scope(tables, cfg, issues)

    # write required reports
    shape_table.to_csv(out_dir / "input_table_shapes.tsv", sep="\t", index=False)
    column_report.to_csv(out_dir / "input_column_report.tsv", sep="\t", index=False)

    # write additional reports
    required_column_check.to_csv(out_dir / "required_column_check.tsv", sep="\t", index=False)
    duplicate_report.to_csv(out_dir / "input_duplicate_report.tsv", sep="\t", index=False)
    sample_overlap.to_csv(out_dir / "input_sample_overlap.tsv", sep="\t", index=False)
    pair_overlap.to_csv(out_dir / "input_pair_overlap.tsv", sep="\t", index=False)
    target_summary.to_csv(out_dir / "target_summary.tsv", sep="\t", index=False)
    feature_check.to_csv(out_dir / "feature_manifest_check.tsv", sep="\t", index=False)
    leakage_report.to_csv(out_dir / "leakage_column_report.tsv", sep="\t", index=False)
    labeled_samples.to_csv(out_dir / "available_labeled_samples.tsv", sep="\t", index=False)

    # fixed columns keep empty issue table readable
    issue_df = pd.DataFrame(
        issues,
        columns=["severity", "table", "check_name", "message", "detail"],
    )
    issue_df.to_csv(out_dir / "validation_issues.tsv", sep="\t", index=False)

    # human summary assembled from machine-readable reports
    write_summary_text(
        path=out_dir / "input_validation_summary.txt",
        cfg=cfg,
        path_table=path_table,
        shape_table=shape_table,
        target_summary=target_summary,
        feature_check=feature_check,
        duplicate_report=duplicate_report,
        sample_overlap=sample_overlap,
        pair_overlap=pair_overlap,
        issues=issues,
    )

    # compact provenance record for rerun audit
    run_info = {
        "script_name": SCRIPT_NAME,
        "step_name": STEP_NAME,
        "config_path": str(Path(args.config)),
        "output_dir": str(out_dir),
        "run_name": get_cfg(cfg, "run_name", ""),
        "run_scope": get_cfg(cfg, "run_scope", ""),
        "test_mode": get_cfg(cfg, "test_mode", ""),
        "test_n_samples": get_cfg(cfg, "test_n_samples", ""),
        # Path objects stringified for JSON serialization
        "input_paths": {key: str(value) if value is not None else "" for key, value in path_map.items()},
        "n_tables_loaded": int(len(tables)),
        "n_errors": int(count_issues(issues, "error")),
        "n_warnings": int(count_issues(issues, "warn")),
    }
    # snapshot includes final table and issue counts
    save_json(run_info, out_dir / "run_config.json")

    # counts reused for console output and exit behavior
    n_errors = count_issues(issues, "error")
    n_warnings = count_issues(issues, "warn")

    print("\nDONE")
    print(f"Tables loaded: {len(tables)}")
    print(f"Errors: {n_errors}")
    print(f"Warnings: {n_warnings}")
    print(f"Wrote: {out_dir / 'input_validation_summary.txt'}")
    print(f"Wrote: {out_dir / 'input_table_shapes.tsv'}")
    print(f"Wrote: {out_dir / 'input_column_report.tsv'}")

    # nonzero exit stops automated pipeline on validation errors
    if n_errors > 0 and not args.allow_errors:
        print("\nValidation errors found. See validation_issues.tsv.")
        return 1

    return 0


# standard script guard, prevents import-time execution
if __name__ == "__main__":
    sys.exit(main())
