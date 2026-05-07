from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import build_source_manifest, read_table


REQUIRED_HANDOFF_FILES = {
    "teacher_table": "visium_fused_teacher_table.tsv",
    "spatial_numeric": "model_input_numeric.csv",
    "feature_manifest": "feature_manifest.csv",
}


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
