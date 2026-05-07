from __future__ import annotations

import re

import pandas as pd


ARTIFACT_PATTERNS = [
    "filtering__",
    "spatial_x",
    "spatial_y",
    "array_row",
    "array_col",
    "barcode",
    "total_counts",
    "pct_counts_mt",
    "n_genes_by_counts",
]

SIZE_OR_COUNT_PATTERNS = [
    "n_spots",
    "n_spots_scored",
    "spot_count",
    "largest_component_spots",
    "access_tumor_spots",
]

CAUTION_PATTERNS = [
    "method_structure_agreement",
    "structure_region_consensus_fraction",
    "metabolic_best_matching_state_score",
]


def infer_theme(text: str) -> str:
    text = str(text).lower()

    if "tryptophan" in text or "kynurenine" in text:
        return "tryptophan kynurenine immune suppression"
    if "myeloid" in text or "macrophage" in text:
        return "myeloid macrophage tumor ecology"
    if "hypoxi" in text:
        return "hypoxia immune stress context"
    if "t_cell" in text or "interferon" in text or "immune" in text or "b_plasma" in text:
        return "immune inflammation and t cell organization"
    if "access" in text or "boundary" in text or "penetration" in text:
        return "tumor access and boundary penetration"
    if "stromal" in text or "stroma" in text or "ecm" in text or "collagen" in text:
        return "stromal ecm barrier architecture"
    if "tumor_proliferative" in text or "proliferation" in text or "cell_cycle" in text:
        return "tumor proliferation state"
    if "vascular" in text or "angiogenic" in text or "endothelial" in text:
        return "vascular angiogenic context"
    if "fatty_acid" in text:
        return "fatty acid metabolism"
    if "glycolysis" in text or "oxphos" in text or "oxidative" in text or "metabolic" in text:
        return "metabolic spatial context"
    if "pair_" in text or "centroid_distance" in text or "overlap" in text:
        return "pairwise spatial relationship"
    return "other interpretable spatial signal"


def classify_feature(feature_name: str, feature_original: str = "", feature_group: str = "") -> tuple[str, str]:
    text = " ".join([str(feature_name), str(feature_original), str(feature_group)]).lower()

    if str(feature_group).lower() == "qc":
        return "exclude_qc", "QC feature group"

    for pattern in ARTIFACT_PATTERNS:
        if pattern in text:
            return "exclude_coordinate_or_artifact", f"artifact pattern: {pattern}"

    for pattern in SIZE_OR_COUNT_PATTERNS:
        if pattern in text:
            return "exclude_size_or_count", f"size or count pattern: {pattern}"

    for pattern in CAUTION_PATTERNS:
        if pattern in text:
            return "exclude_caution", f"caution pattern: {pattern}"

    return "include_biology", "strict biology feature"


def strict_biology_manifest(manifest: pd.DataFrame, feature_col: str = "feature_name") -> pd.DataFrame:
    if manifest is None or manifest.empty:
        return pd.DataFrame(columns=["feature_name", "feature_original", "biological_theme", "governance_class", "governance_reason"])

    out = manifest.copy()

    if feature_col not in out.columns:
        candidates = [c for c in out.columns if "feature" in str(c).lower()]
        if not candidates:
            raise ValueError("No feature column found in manifest")
        feature_col = candidates[0]

    out["feature_name"] = out[feature_col].astype(str)

    if "feature_original" not in out.columns:
        out["feature_original"] = out["feature_name"]

    if "feature_group" not in out.columns:
        out["feature_group"] = ""

    classes = out.apply(
        lambda r: classify_feature(r["feature_name"], r.get("feature_original", ""), r.get("feature_group", "")),
        axis=1,
    )
    out["governance_class"] = [x[0] for x in classes]
    out["governance_reason"] = [x[1] for x in classes]
    out["biological_theme"] = out["feature_original"].map(infer_theme)
    out["include_for_primary_biology"] = out["governance_class"].eq("include_biology")
    return out
