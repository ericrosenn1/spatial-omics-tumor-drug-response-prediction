from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_discovery import find_feature_column
from .feature_governance import classify_feature, infer_theme, strict_biology_manifest
from .target_building import build_broad_residual_targets, treatment_eligibility


def numeric_feature_columns(spatial: pd.DataFrame) -> list[str]:
    cols = []
    for col in spatial.columns:
        if col == "sample_id":
            continue
        values = pd.to_numeric(spatial[col], errors="coerce")
        if values.notna().sum() > 0:
            cols.append(col)
    return cols


def build_all_feature_governance_manifest(spatial: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    spatial_cols = set(spatial.columns)
    feature_col = find_feature_column(manifest)

    governed = strict_biology_manifest(manifest, feature_col=feature_col)
    governed = governed.copy()

    governed["available_in_spatial_matrix"] = governed["feature_name"].astype(str).isin(spatial_cols)
    governed["include_for_broad_governed_candidate_pool"] = (
        governed["include_for_primary_biology"].astype(bool)
        & governed["available_in_spatial_matrix"].astype(bool)
    )

    governed["include_for_v2_strict_biology_registry"] = False
    governed["v2_strict_registry_status"] = "not_generated_until_residual_model_evidence"

    return governed


def build_broad_candidate_pool(spatial: pd.DataFrame, governed: pd.DataFrame) -> list[str]:
    spatial_cols = set(spatial.columns)

    broad = governed.loc[
        governed["include_for_broad_governed_candidate_pool"] == True,
        "feature_name",
    ].astype(str).tolist()

    broad = [f for f in broad if f in spatial_cols]
    broad = list(dict.fromkeys(broad))

    if len(broad) >= 10:
        return broad

    fallback_rows = []
    for col in numeric_feature_columns(spatial):
        governance_class, governance_reason = classify_feature(col, col, "")
        fallback_rows.append({
            "feature_name": col,
            "governance_class": governance_class,
            "governance_reason": governance_reason,
        })

    fallback = pd.DataFrame(fallback_rows)
    broad = fallback.loc[fallback["governance_class"] == "include_biology", "feature_name"].astype(str).tolist()
    return list(dict.fromkeys(broad))


def coerce_spatial_features(spatial: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = spatial[["sample_id"] + feature_cols].copy()
    out["sample_id"] = out["sample_id"].astype(str)

    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def build_pair_level_residual_dataset(
    teacher: pd.DataFrame,
    spatial_features: pd.DataFrame,
    feature_cols: list[str],
    residual_col: str = "fused_residual_vs_prior",
) -> pd.DataFrame:
    required = ["sample_id", "drug_key", residual_col]
    missing = [c for c in required if c not in teacher.columns]

    if missing:
        raise ValueError("Teacher table missing required columns: " + ", ".join(missing))

    keep_cols = ["sample_id", "drug_key", residual_col]

    for col in [
        "drug",
        "drug_label",
        "treatment",
        "treatment_name",
        "fused_prob_responder",
        "treatment_prior",
        "fused_prior_prob",
    ]:
        if col in teacher.columns and col not in keep_cols:
            keep_cols.append(col)

    df = teacher[keep_cols].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["drug_key"] = df["drug_key"].astype(str)
    df[residual_col] = pd.to_numeric(df[residual_col], errors="coerce")
    df = df.dropna(subset=[residual_col])

    merged = df.merge(spatial_features[["sample_id"] + feature_cols], on="sample_id", how="inner")
    return merged


def build_broad_residual_dataset(
    teacher: pd.DataFrame,
    spatial_features: pd.DataFrame,
    feature_cols: list[str],
    residual_col: str = "fused_residual_vs_prior",
) -> pd.DataFrame:
    broad = build_broad_residual_targets(teacher, residual_col=residual_col)
    merged = broad.merge(spatial_features[["sample_id"] + feature_cols], on="sample_id", how="inner")
    return merged


def build_dataset_bundle(
    teacher: pd.DataFrame,
    spatial: pd.DataFrame,
    manifest: pd.DataFrame,
    residual_col: str = "fused_residual_vs_prior",
) -> dict:
    teacher = teacher.copy()
    spatial = spatial.copy()

    teacher["sample_id"] = teacher["sample_id"].astype(str)
    spatial["sample_id"] = spatial["sample_id"].astype(str)

    governance = build_all_feature_governance_manifest(spatial, manifest)
    broad_feature_cols = build_broad_candidate_pool(spatial, governance)

    if len(broad_feature_cols) < 10:
        raise ValueError(f"Too few broad governed candidate features selected: {len(broad_feature_cols)}")

    spatial_features = coerce_spatial_features(spatial, broad_feature_cols)

    pair_dataset = build_pair_level_residual_dataset(
        teacher=teacher,
        spatial_features=spatial_features,
        feature_cols=broad_feature_cols,
        residual_col=residual_col,
    )

    broad_dataset = build_broad_residual_dataset(
        teacher=teacher,
        spatial_features=spatial_features,
        feature_cols=broad_feature_cols,
        residual_col=residual_col,
    )

    eligibility = treatment_eligibility(
        teacher=teacher,
        residual_col=residual_col,
    )

    feature_set_policy = {
        "broad_governed_candidate_feature_count": int(len(broad_feature_cols)),
        "strict_biology_registry_status": "not_generated_yet",
        "strict_biology_registry_generation_step": "Step 05 residual biology registry using Step 04 residual pair model evidence",
        "production_dependency_on_v1_outputs": "no",
        "v1_strict_feature_table_role": "reference_only_for_migration_comparison",
        "primary_interpretation_feature_set_after_step05": "v2_generated_strict_biology_registry",
        "sensitivity_feature_set": "broad_governed_candidate_pool",
    }

    return {
        "all_feature_governance_manifest": governance,
        "broad_feature_cols": broad_feature_cols,
        "spatial_features": spatial_features,
        "pair_level_residual_dataset": pair_dataset,
        "broad_residual_dataset": broad_dataset,
        "treatment_eligibility": eligibility,
        "feature_set_policy": feature_set_policy,
    }

