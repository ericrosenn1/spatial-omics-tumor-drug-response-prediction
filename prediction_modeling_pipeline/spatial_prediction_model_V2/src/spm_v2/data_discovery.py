from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import build_source_manifest, read_table


REQUIRED_HANDOFF_FILES = {
    "teacher_table": "visium_fused_teacher_table.tsv",
    "spatial_numeric": "model_input_numeric.csv",
    "feature_manifest": "feature_manifest.csv",
}


def handoff_sample_coverage(teacher: pd.DataFrame, spatial: pd.DataFrame) -> dict[str, int]:
    """Expose the existing inner join's cohort coverage without changing it."""
    if any("sample_id" not in table or table.columns.duplicated().any() for table in (teacher, spatial)):
        return {}
    teacher_ids = set(teacher.sample_id.dropna().astype(str))
    spatial_ids = set(spatial.sample_id.dropna().astype(str))
    return {
        "matched_sample_count": len(teacher_ids & spatial_ids),
        "teacher_only_sample_count": len(teacher_ids - spatial_ids),
        "spatial_only_sample_count": len(spatial_ids - teacher_ids),
    }


def handoff_validation_errors(
    teacher: pd.DataFrame,
    spatial: pd.DataFrame,
    manifest: pd.DataFrame,
    residual_col: str = "fused_residual_vs_prior",
) -> list[str]:
    """Check table identities before joins can multiply records or be empty.

    Missing residual measurements remain allowed; they are omitted by the
    existing target builder. This check does not select a new cohort or infer
    missing feature/response values.
    """
    errors = []
    for label, table, required in [
        ("teacher", teacher, ["sample_id", "drug_key", residual_col]),
        ("spatial_numeric", spatial, ["sample_id"]),
        ("feature_manifest", manifest, []),
    ]:
        if table.empty:
            errors.append(f"{label}: table is empty")
        if table.columns.duplicated().any():
            errors.append(f"{label}: duplicate column names")
            continue
        missing = [column for column in required if column not in table]
        if missing:
            errors.append(f"{label}: missing required columns: {', '.join(missing)}")
        for column in ("sample_id", "drug_key"):
            if column in required and column in table:
                values = table[column]
                if values.isna().any() or values.astype(str).str.strip().eq("").any():
                    errors.append(f"{label}: missing or blank {column}")

    if not teacher.columns.duplicated().any():
        if {"sample_id", "drug_key"} <= set(teacher):
            identities = teacher[["sample_id", "drug_key"]].astype(str)
            if identities.duplicated().any():
                errors.append("teacher: duplicate sample_id/drug_key pairs")
        if residual_col in teacher:
            values = pd.to_numeric(teacher[residual_col], errors="coerce")
            malformed = teacher[residual_col].notna() & values.isna()
            if malformed.any() or values.isin([float("inf"), -float("inf")]).any():
                errors.append(f"teacher: nonnumeric or infinite {residual_col}")
            if not values.notna().any():
                errors.append(f"teacher: no observed {residual_col} values")
    if not spatial.columns.duplicated().any() and "sample_id" in spatial:
        if spatial.sample_id.astype(str).duplicated().any():
            errors.append("spatial_numeric: duplicate sample_id values")
    coverage = handoff_sample_coverage(teacher, spatial)
    if coverage and coverage["matched_sample_count"] == 0:
        errors.append("teacher and spatial_numeric: no matching sample_id values")
    if not manifest.empty and not manifest.columns.duplicated().any():
        try:
            feature_col = find_feature_column(manifest)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            names = manifest[feature_col]
            if names.isna().any() or names.astype(str).str.strip().eq("").any():
                errors.append("feature_manifest: missing or blank feature names")
            if names.astype(str).duplicated().any():
                errors.append("feature_manifest: duplicate feature names")
    return errors


def handoff_paths(handoff_root: str | Path) -> dict[str, Path]:
    root = Path(handoff_root)
    return {key: root / rel for key, rel in REQUIRED_HANDOFF_FILES.items()}


def validate_handoff_files(handoff_root: str | Path) -> pd.DataFrame:
    paths = handoff_paths(handoff_root)
    return build_source_manifest(paths)


def load_handoff_tables(handoff_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = handoff_paths(handoff_root)

    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing handoff files: " + ", ".join(missing))

    teacher = read_table(paths["teacher_table"])
    spatial = read_table(paths["spatial_numeric"])
    manifest = read_table(paths["feature_manifest"])

    return teacher, spatial, manifest, paths


def table_shape_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        rows.append({
            "table_name": name,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "n_numeric_columns": int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
            "n_object_columns": int(sum(pd.api.types.is_object_dtype(df[c]) for c in df.columns)),
            "n_bool_columns": int(sum(pd.api.types.is_bool_dtype(df[c]) for c in df.columns)),
        })
    return pd.DataFrame(rows)


def column_presence_summary(df: pd.DataFrame, required_columns: list[str], table_name: str) -> pd.DataFrame:
    rows = []
    for col in required_columns:
        rows.append({
            "table_name": table_name,
            "column": col,
            "present": col in df.columns,
        })
    return pd.DataFrame(rows)


def find_feature_column(manifest: pd.DataFrame) -> str:
    candidates = [
        "feature_name",
        "feature",
        "feature_id",
        "model_feature",
        "feature_clean",
        "original_feature",
        "feature_original",
    ]

    for candidate in candidates:
        if candidate in manifest.columns:
            return candidate

    feature_like = [c for c in manifest.columns if "feature" in str(c).lower()]
    if feature_like:
        return feature_like[0]

    raise ValueError("Could not identify feature column in feature manifest")
