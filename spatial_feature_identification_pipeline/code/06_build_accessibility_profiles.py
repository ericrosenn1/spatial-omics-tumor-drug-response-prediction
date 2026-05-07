"""
Script: 06_build_accessibility_profiles.py

Purpose:
Build spatial accessibility profiles for each processed Visium sample.

This script estimates how accessible tumor regions are based on:
    1. tumor boundary structure
    2. stromal / ECM / hypoxia barrier scores
    3. immune and vascular proximity
    4. graph-based penetration depth from accessible tumor boundary
    5. spatial gradients from tumor boundary into tumor core
    6. accessible vs impermeable boundary topology

Inputs:
    processed_samples/<sample_id>/adata/03_final_pipeline_output.h5ad
    signature_scores/slide_features_with_signature_scores.csv
    scored_labels/slide_features_scored_labeled.csv
    merged_features/merged_slide_features.csv
    processing/processing_report.csv

Outputs:
    accessibility_profiles/per_sample/
    accessibility_profiles/slide_accessibility_profiles.csv
    accessibility_profiles/accessibility_status.csv
    accessibility_profiles/slide_features_with_accessibility.csv
    accessibility_profiles/accessibility_profile_summary.txt

Usage:
    python scripts/06_build_accessibility_profiles.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


# =========================
# Project imports
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config


# =========================
# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# Config constants
# =========================

# These weights define the biological interpretation of the local accessibility score.
# Positive weights increase barrier strength.
# Negative weights represent features that make entry or penetration easier.
WEIGHTS = {
    "stromal": 1.0,
    "hypoxia": 1.0,
    "ecm_remodeling": 0.75,
    "angiogenic": -0.75,
    "vascular_proximity": -0.75,
    "immune_proximity": -0.50,
}

# Top and bottom tumor-boundary quantiles become accessible vs impermeable entry zones.
ACCESSIBLE_BOUNDARY_QUANTILE = 0.80
IMPERMEABLE_BOUNDARY_QUANTILE = 0.20

# Minimum spot counts prevent unstable graph metrics on tiny tumor regions.
MIN_TUMOR_SPOTS = 20
MIN_BOUNDARY_SPOTS = 5

# Use spatial graph neighbors from Squidpy when missing.
SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"

# Prefer final processed object, but allow fallback if a sample lacks later files.
H5AD_CANDIDATES = [
    "03_final_pipeline_output.h5ad",
    "02_after_clustering_and_annotation.h5ad",
    "01_after_filtering.h5ad",
]


# =========================
# Argument and config helpers
# =========================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def get_output_root(config_path):
    """Load the YAML config and return output_root."""
    cfg = validate_config(load_config(config_path))
    return Path(cfg["output_root"])


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Input discovery helpers
# =========================

def read_processing_report(output_root):
    """Read the processing report if it exists."""
    path = output_root / "output_02_process_samples_reports" / "processing_report.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_ok_samples(processing_report):
    """Return sample IDs with OK processing status."""
    if processing_report.empty:
        return None

    if "sample_id" not in processing_report.columns or "status" not in processing_report.columns:
        return None

    ok = processing_report.loc[
        processing_report["status"].astype(str).str.upper() == "OK",
        "sample_id",
    ]

    return set(ok.astype(str))


def discover_sample_dirs(output_root, ok_samples=None):
    """Find processed sample folders available for accessibility profiling."""
    processed_root = output_root / "output_02_01_process_samples_data"

    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed sample folder: {processed_root}")

    sample_dirs = sorted([p for p in processed_root.iterdir() if p.is_dir()])

    # Use processing_report as the source of truth when available.
    if ok_samples is not None:
        sample_dirs = [p for p in sample_dirs if p.name in ok_samples]

    return sample_dirs


def find_h5ad(sample_dir):
    """Find the best available h5ad file for one processed sample."""
    adata_dir = sample_dir / "adata"

    for name in H5AD_CANDIDATES:
        path = adata_dir / name
        if path.exists():
            return path

    if adata_dir.exists():
        h5ads = sorted(adata_dir.glob("*.h5ad"))
        if h5ads:
            return h5ads[-1]

    return None




def find_consensus_h5ad(output_root, sample_id):
    """Prefer the step 05 labeled h5ad so accessibility uses consensus structure labels."""
    path = Path(output_root) / "output_05_build_multi_axis_transcriptome_labels" / "per_sample_h5ad" / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    return path if path.exists() else None


def choose_base_table(output_root):
    """Choose the richest existing slide table to merge accessibility features into."""
    candidates = [
        output_root / "output_05_build_multi_axis_transcriptome_labels" / "slide_features_with_multi_axis_labels.csv",
        output_root / "output_04_score_and_label_slides" / "slide_features_scored_labeled.csv",
        output_root / "output_03_merge_slide_features" / "merged_slide_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find signature, scored, or merged slide table")


# =========================
# Safe numeric helpers
# =========================

def finite_values(values):
    """Return finite numeric values from an array-like object."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(values):
    """Return the mean of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.mean(vals))


def safe_median(values):
    """Return the median of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.median(vals))


def safe_std(values):
    """Return the standard deviation of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.std(vals))


def safe_min(values):
    """Return the minimum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.min(vals))


def safe_max(values):
    """Return the maximum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.max(vals))


def safe_quantile(values, q):
    """Return a quantile of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.quantile(vals, q))


def normalize_01(values):
    """Min-max normalize values to 0 to 1 while handling constants."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)

    mask = np.isfinite(arr)

    if mask.sum() == 0:
        return out

    vmin = np.nanmin(arr[mask])
    vmax = np.nanmax(arr[mask])

    # If all values are identical, there is no spatial contrast to normalize.
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return out

    out[mask] = (arr[mask] - vmin) / (vmax - vmin)
    return out


def zscore_within_slide(values):
    """Z score values within one slide while handling constants."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)

    mask = np.isfinite(arr)

    if mask.sum() <= 1:
        return out

    mu = np.nanmean(arr[mask])
    sd = np.nanstd(arr[mask])

    # Constant or invalid scores should not create artificial barriers.
    if not np.isfinite(sd) or sd == 0:
        return out

    out[mask] = (arr[mask] - mu) / sd
    return out


def fit_linear_slope(x, y):
    """Fit a simple linear slope and R2 between two numeric vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        return np.nan, np.nan
    slope = float(coeffs[0])

    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = float(1 - ss_res / ss_tot)

    return slope, r2


# =========================
# Column detection helpers
# =========================

def get_first_existing_col(df_or_obs, candidates):
    """Return the first candidate column that exists."""
    cols = list(df_or_obs.columns)

    for col in candidates:
        if col in cols:
            return col

    return None


def get_score_vector(obs_df, base_name, n):
    """Find and return a spot-level score vector from obs using flexible column names."""
    # Multiple scripts and packages name scores differently.
    # Try a broad but controlled set of possibilities.
    candidates = [
        f"structure_score__{base_name}",
        f"function_score__{base_name}",
        f"metabolic_score__{base_name}",
        f"simple_mean__{base_name}",
        f"rank_percentile__{base_name}",
        f"custom_gsva__{base_name}",
        f"ucell__{base_name}",
        base_name,
        f"{base_name}_score",
        f"{base_name}_ucell",
        f"ucell_{base_name}",
        f"{base_name}_UCell",
        f"UCell_{base_name}",
        f"{base_name}_gsva",
        f"gsva_{base_name}",
        f"simple__{base_name}",
        f"ucell__{base_name}",
        f"gsva_custom__{base_name}",
    ]

    col = get_first_existing_col(obs_df, candidates)

    if col is None:
        # Missing score vectors become zeros so the metric can still run.
        # The column-used metadata records that this was not directly measured.
        return np.zeros(n, dtype=float), None

    vals = pd.to_numeric(obs_df[col], errors="coerce").fillna(0.0).values.astype(float)
    return vals, col


def get_annotation_vector(obs_df):
    """Return the best available spot annotation vector."""
    candidates = [
        "structure_region_label_smoothed",
        "structure_region_label",
        "structure_dominant_label_raw",
        "structure_dominant_label",
        "cluster_annotation_final",
        "cluster_annotation_primary",
        "cluster_annotation_secondary",
        "final_annotation",
        "cluster_scored_architecture_label",
        "cluster_annotation",
        "manual_annotation",
        "annotation",
        "leiden_annotation",
        "leiden",
    ]

    col = get_first_existing_col(obs_df, candidates)

    if col is None:
        return pd.Series(["unknown"] * obs_df.shape[0], index=obs_df.index), None

    return obs_df[col].astype(str), col


# =========================
# Annotation mask helpers
# =========================

def build_masks_from_annotations(annotation_series):
    """Convert spot annotations into tumor, stromal, immune, vascular, and hypoxic masks."""
    ann = annotation_series.fillna("unknown").astype(str).str.lower()

    # These terms are intentionally broad because labels differ across datasets.
    tumor_terms = [
        "tumor_epithelial",
        "tumor_proliferative",
        "tumor_associated_state",
        "epithelial_or_tumor_like",
        "tumor",
        "epithelial",
        "malignant",
        "cancer",
    ]

    stromal_terms = [
        "stromal_ecm",
        "ecm_remodeling",
        "fibroblast",
        "stroma",
        "stromal",
        "collagen",
        "matrix",
    ]

    immune_terms = [
        "t_cell",
        "immune_b_plasma",
        "myeloid_macrophage",
        "interferon_inflamed",
        "immune",
        "lymphocyte",
        "macrophage",
        "myeloid",
        "b_cell",
        "plasma",
    ]

    vascular_terms = [
        "angiogenic_vascular",
        "vascular",
        "endothelial",
        "blood_vessel",
        "vessel",
    ]

    hypoxic_terms = [
        "hypoxic_stress",
        "hypoxic",
        "hypoxia",
    ]

    tumor_mask = ann.apply(lambda x: any(term in x for term in tumor_terms)).values
    stromal_mask = ann.apply(lambda x: any(term in x for term in stromal_terms)).values
    immune_mask = ann.apply(lambda x: any(term in x for term in immune_terms)).values
    vascular_mask = ann.apply(lambda x: any(term in x for term in vascular_terms)).values
    hypoxic_mask = ann.apply(lambda x: any(term in x for term in hypoxic_terms)).values

    return {
        "tumor": tumor_mask,
        "stromal": stromal_mask,
        "immune": immune_mask,
        "vascular": vascular_mask,
        "hypoxic": hypoxic_mask,
    }


def fallback_tumor_mask_from_scores(obs_df, n):
    """Build a fallback tumor mask from tumor score columns if annotations are missing."""
    tumor_score, tumor_col = get_score_vector(obs_df, "tumor_epithelial", n)

    if tumor_col is None:
        return np.zeros(n, dtype=bool), None

    finite = tumor_score[np.isfinite(tumor_score)]

    if finite.size == 0:
        return np.zeros(n, dtype=bool), tumor_col

    # Use upper quartile as a conservative tumor-enriched fallback.
    threshold = np.nanquantile(finite, 0.75)
    return tumor_score >= threshold, tumor_col


# =========================
# Spatial graph helpers
# =========================

def ensure_spatial_graph(adata):
    """Ensure the AnnData object has a spatial neighbor graph."""
    if SPATIAL_CONNECTIVITY_KEY in adata.obsp:
        return

    # Squidpy builds a neighbor graph from Visium spatial coordinates.
    # This graph is used for boundary detection and graph penetration depth.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sq.gr.spatial_neighbors(adata)


def get_spatial_coordinates(adata):
    """Return spatial coordinates from AnnData or raise a clear error."""
    if "spatial" not in adata.obsm:
        raise ValueError("missing_spatial_coordinates")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    if coords.shape[0] != adata.n_obs:
        raise ValueError("spatial_coordinate_length_mismatch")

    return coords


def get_spatial_adjacency(adata):
    """Return the spatial connectivity matrix as CSR."""
    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        raise ValueError("missing_spatial_connectivities")

    return csr_matrix(adata.obsp[SPATIAL_CONNECTIVITY_KEY])


def compute_boundary_mask(tumor_mask, adj):
    """Identify tumor spots touching at least one non-tumor neighbor."""
    tumor_idx = np.where(tumor_mask)[0]
    boundary = np.zeros(len(tumor_mask), dtype=bool)

    if len(tumor_idx) == 0:
        return boundary

    for idx in tumor_idx:
        neighbors = adj[idx].indices

        if len(neighbors) == 0:
            continue

        # A tumor spot is a boundary spot if any neighbor is not tumor.
        if np.any(~tumor_mask[neighbors]):
            boundary[idx] = True

    return boundary

def nearest_distance(src_coords, ref_coords):
    """Compute nearest Euclidean distance from each source coordinate to reference coordinates."""
    # If either set is empty, there is no meaningful distance to compute.
    if src_coords.shape[0] == 0 or ref_coords.shape[0] == 0:
        return np.array([], dtype=float)

    # KDTree makes nearest-neighbor distance much faster than pairwise distance matrices.
    tree = cKDTree(ref_coords)
    dists, _ = tree.query(src_coords, k=1)

    return np.asarray(dists, dtype=float)


def compute_connected_component_features(mask, adj, n_total):
    """Summarize spatial connected components for a boolean spot mask."""
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return {
            "n_components": 0,
            "largest_component_spots": 0,
            "largest_component_fraction_of_slide": np.nan,
            "largest_component_fraction_of_mask": np.nan,
        }

    # Restrict the spatial graph to only the selected spots.
    subgraph = adj[idx][:, idx]
    n_comp, labels = connected_components(
        subgraph,
        directed=False,
        connection="weak",
    )

    if len(labels) == 0:
        largest = 0
    else:
        largest = int(pd.Series(labels).value_counts().max())

    return {
        "n_components": int(n_comp),
        "largest_component_spots": int(largest),
        "largest_component_fraction_of_slide": largest / n_total if n_total > 0 else np.nan,
        "largest_component_fraction_of_mask": largest / len(idx) if len(idx) > 0 else np.nan,
    }


def compute_graph_depth_from_sources(tumor_mask, source_mask, adj):
    """Compute shortest graph distance from source tumor spots to all tumor spots."""
    tumor_idx = np.where(tumor_mask)[0]
    source_idx = np.where(source_mask & tumor_mask)[0]

    depth = np.full(adj.shape[0], np.nan, dtype=float)

    if len(tumor_idx) == 0 or len(source_idx) == 0:
        return depth

    # Dijkstra gives graph distance through the tissue adjacency graph.
    # We use unweighted graph distance because each edge is one spatial-neighbor step.
    dist_mat = dijkstra(
        csgraph=adj,
        directed=False,
        indices=source_idx,
        unweighted=True,
    )

    if dist_mat.ndim == 1:
        min_dist = dist_mat
    else:
        min_dist = np.nanmin(dist_mat, axis=0)

    min_dist[~np.isfinite(min_dist)] = np.nan
    depth[tumor_idx] = min_dist[tumor_idx]

    return depth


def summarize_profile(values, prefix):
    """Summarize a spatial profile vector into slide-level features."""
    vals = finite_values(values)

    if len(vals) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
        }

    return {
        f"{prefix}_mean": safe_mean(vals),
        f"{prefix}_median": safe_median(vals),
        f"{prefix}_std": safe_std(vals),
        f"{prefix}_min": safe_min(vals),
        f"{prefix}_max": safe_max(vals),
        f"{prefix}_q10": safe_quantile(vals, 0.10),
        f"{prefix}_q25": safe_quantile(vals, 0.25),
        f"{prefix}_q75": safe_quantile(vals, 0.75),
        f"{prefix}_q90": safe_quantile(vals, 0.90),
    }


def build_barrier_and_accessibility_scores(stromal_score, hypoxia_score, ecm_score,
                                           angiogenic_score, vascular_prox_score,
                                           immune_prox_score):
    """Build composite barrier and accessibility vectors from local score components."""
    # Z-score within each slide so one feature scale does not dominate the composite.
    stromal_z = zscore_within_slide(stromal_score)
    hypoxia_z = zscore_within_slide(hypoxia_score)
    ecm_z = zscore_within_slide(ecm_score)
    angiogenic_z = zscore_within_slide(angiogenic_score)
    vascular_prox_z = zscore_within_slide(vascular_prox_score)
    immune_prox_z = zscore_within_slide(immune_prox_score)

    # Barrier emphasizes stromal, hypoxic, and ECM remodeling signals.
    barrier_score = (
        WEIGHTS["stromal"] * stromal_z +
        WEIGHTS["hypoxia"] * hypoxia_z +
        WEIGHTS["ecm_remodeling"] * ecm_z
    )

    # Accessibility is the conceptual inverse of barrier, plus vascular/immune proximity.
    accessibility_score = (
        -WEIGHTS["stromal"] * stromal_z +
        -WEIGHTS["hypoxia"] * hypoxia_z +
        -WEIGHTS["ecm_remodeling"] * ecm_z +
        (-WEIGHTS["angiogenic"]) * angiogenic_z +
        (-WEIGHTS["vascular_proximity"]) * vascular_prox_z +
        (-WEIGHTS["immune_proximity"]) * immune_prox_z
    )

    return barrier_score, accessibility_score


def assign_accessibility_profile(row):
    """Assign a coarse spatial accessibility label from computed metrics."""
    if row.get("access_tumor_fraction_of_slide", 0) >= 0.90:
        return "tumor_only_section", np.nan, "no_non_tumor_context"

    frac_access = row.get("access_accessible_boundary_fraction_of_tumor_boundary", np.nan)
    frac_imperm = row.get("access_impermeable_boundary_fraction_of_tumor_boundary", np.nan)
    frac_within_2 = row.get("access_frac_tumor_within_graph_depth_2", np.nan)
    frac_deep_4 = row.get("access_frac_tumor_deeper_than_graph_depth_4", np.nan)
    barrier_slope = row.get("access_barrier_depth_slope", np.nan)
    access_slope = row.get("access_accessibility_depth_slope", np.nan)
    boundary_std = row.get("access_boundary_accessibility_std", np.nan)
    boundary_range = row.get("access_boundary_accessibility_range", np.nan)
    tumor_access_mean = row.get("access_accessibility_score_tumor_mean", np.nan)
    core_access_mean = row.get("access_core_accessibility_mean", np.nan)
    tumor_barrier_mean = row.get("access_barrier_score_tumor_mean", np.nan)
    core_barrier_mean = row.get("access_core_barrier_mean", np.nan)

    required = [
        frac_access,
        frac_imperm,
        frac_within_2,
        frac_deep_4,
        barrier_slope,
        boundary_std,
    ]

    # If key metrics are missing, avoid making a confident spatial interpretation.
    if not all(np.isfinite(x) for x in required):
        return "undetermined", np.nan, "missing_core_metrics"

    heterogeneous_boundary = (
        (np.isfinite(boundary_std) and boundary_std >= 0.75) or
        (np.isfinite(boundary_range) and boundary_range >= 3.0)
    )

    strongly_accessible_boundary = frac_access >= 0.60
    weakly_accessible_boundary = frac_access <= 0.25
    strongly_impermeable_boundary = frac_imperm >= 0.60
    weakly_impermeable_boundary = frac_imperm <= 0.25

    shallow_penetration = frac_within_2 >= 0.70
    poor_penetration = frac_within_2 <= 0.25
    deep_core_present = frac_deep_4 >= 0.30

    barrier_rises_inward = barrier_slope > 0.05
    barrier_falls_inward = barrier_slope < -0.05
    access_falls_inward = access_slope < -0.05 if np.isfinite(access_slope) else False

    core_more_barriered = (
        np.isfinite(core_barrier_mean) and
        np.isfinite(tumor_barrier_mean) and
        core_barrier_mean > tumor_barrier_mean + 0.25
    )

    core_more_accessible = (
        np.isfinite(core_access_mean) and
        np.isfinite(tumor_access_mean) and
        core_access_mean > tumor_access_mean + 0.25
    )

    profile_hits = []

    # These rules are intentionally interpretable, not machine-learned.
    # They provide a readable spatial phenotype for each slide.
    if (
        strongly_accessible_boundary and
        weakly_impermeable_boundary and
        shallow_penetration and
        not deep_core_present and
        (barrier_falls_inward or not barrier_rises_inward) and
        not heterogeneous_boundary
    ):
        profile_hits.append("uniformly_accessible")

    if (
        weakly_accessible_boundary and
        strongly_impermeable_boundary and
        poor_penetration and
        (barrier_rises_inward or core_more_barriered) and
        not heterogeneous_boundary
    ):
        profile_hits.append("uniformly_inaccessible")

    if (
        weakly_accessible_boundary and
        poor_penetration and
        (barrier_rises_inward or core_more_barriered)
    ):
        profile_hits.append("boundary_limited_barrier_rising")

    if (
        frac_within_2 > 0.25 and
        deep_core_present and
        (barrier_rises_inward or access_falls_inward or core_more_barriered)
    ):
        profile_hits.append("partial_penetration_with_deep_core")

    if heterogeneous_boundary and 0.20 < frac_access < 0.80:
        profile_hits.append("heterogeneous_accessibility")

    if (
        frac_access >= 0.40 and
        frac_within_2 >= 0.45 and
        not strongly_impermeable_boundary and
        not barrier_rises_inward and
        not deep_core_present
    ):
        profile_hits.append("mixed_but_permissive")

    if (
        frac_imperm >= 0.40 and
        (barrier_rises_inward or access_falls_inward or deep_core_present)
    ):
        profile_hits.append("mixed_but_barrier_skewed")

    if (
        heterogeneous_boundary and
        0.20 <= frac_access <= 0.50 and
        frac_within_2 >= 0.35 and
        deep_core_present
    ):
        profile_hits.append("patchy_entry_channels")

    if len(profile_hits) == 0:
        if (
            frac_access >= 0.40 and
            frac_within_2 >= 0.40 and
            not barrier_rises_inward and
            not deep_core_present
        ):
            label = "mixed_permissive"
        elif (
            frac_imperm >= 0.35 or
            barrier_rises_inward or
            deep_core_present or
            core_more_barriered
        ):
            label = "mixed_barrier_skewed"
        elif core_more_accessible:
            label = "mixed_core_accessible"
        else:
            label = "mixed_balanced"

        notes = "fallback_refined_mixed"
    else:
        priority = [
            "uniformly_accessible",
            "uniformly_inaccessible",
            "boundary_limited_barrier_rising",
            "partial_penetration_with_deep_core",
            "heterogeneous_accessibility",
            "patchy_entry_channels",
            "mixed_but_barrier_skewed",
            "mixed_but_permissive",
        ]

        label = next((x for x in priority if x in profile_hits), profile_hits[0])
        notes = "; ".join(profile_hits)

    confidence_terms = [
        abs(frac_access - 0.5) * 2.0,
        abs(frac_imperm - 0.5) * 2.0,
        abs(frac_within_2 - 0.5) * 2.0,
        min(1.0, abs(frac_deep_4 - 0.2) * 2.0),
        min(1.0, abs(barrier_slope) / 0.25),
        min(1.0, boundary_std / 1.0),
    ]

    confidence = float(np.nanmean(confidence_terms))
    confidence = min(1.0, confidence) if np.isfinite(confidence) else np.nan

    return label, confidence, notes


def score_one_sample(sample_id, h5ad_path, per_sample_dir):
    """Compute accessibility metrics for one sample."""
    adata = sc.read_h5ad(h5ad_path)

    coords = get_spatial_coordinates(adata)
    ensure_spatial_graph(adata)
    adj = get_spatial_adjacency(adata)

    obs = adata.obs.copy()
    annotation_vec, annotation_col = get_annotation_vector(obs)
    masks = build_masks_from_annotations(annotation_vec)

    tumor_mask = masks["tumor"]

    # If annotations do not identify tumor, try a conservative tumor-score fallback.
    tumor_fallback_col = None
    # fallback if annotation-based tumor mask is too small
    if int(tumor_mask.sum()) < MIN_TUMOR_SPOTS:
        fallback_mask, tumor_fallback_col = fallback_tumor_mask_from_scores(obs, adata.n_obs)

        if int(fallback_mask.sum()) >= MIN_TUMOR_SPOTS:
            tumor_mask = fallback_mask

    # SECOND fallback: use top fraction of tumor score even if annotation exists
    if int(tumor_mask.sum()) < MIN_TUMOR_SPOTS:
        tumor_score, tumor_col = get_score_vector(obs, "tumor_epithelial", adata.n_obs)

        finite = tumor_score[np.isfinite(tumor_score)]
        if len(finite) > 0:
            thr = np.nanquantile(finite, 0.60)  # less strict than 0.75
            tumor_mask = tumor_score >= thr

    stromal_mask = masks["stromal"]
    immune_mask = masks["immune"]
    vascular_mask = masks["vascular"]
    hypoxic_mask = masks["hypoxic"]

    tumor_n = int(tumor_mask.sum())

    if tumor_n < MIN_TUMOR_SPOTS:
        raise ValueError(f"too_few_tumor_spots_{tumor_n}")

    tumor_boundary_mask = compute_boundary_mask(tumor_mask, adj)
    boundary_n = int(tumor_boundary_mask.sum())

    if boundary_n < MIN_BOUNDARY_SPOTS:
        # do not fail â€” mark as low-quality boundary
        boundary_flag = "weak_boundary"
    else:
        boundary_flag = "ok"

    # Pull spot-level score vectors. Missing scores are recorded as "none" and treated as zeros.
    stromal_score, stromal_col = get_score_vector(obs, "stromal_ecm", adata.n_obs)
    hypoxia_score, hypoxia_col = get_score_vector(obs, "hypoxic_stress", adata.n_obs)
    ecm_score, ecm_col = get_score_vector(obs, "ecm_remodeling", adata.n_obs)
    angiogenic_score, angiogenic_col = get_score_vector(obs, "angiogenic_vascular", adata.n_obs)
    tumor_score, tumor_col = get_score_vector(obs, "tumor_epithelial", adata.n_obs)
    inflamed_score, inflamed_col = get_score_vector(obs, "interferon_inflamed", adata.n_obs)
    tcell_score, tcell_col = get_score_vector(obs, "t_cell", adata.n_obs)
    bcell_score, bcell_col = get_score_vector(obs, "immune_b_plasma", adata.n_obs)
    myeloid_score, myeloid_col = get_score_vector(obs, "myeloid_macrophage", adata.n_obs)

    # Immune score is a compact immune proximity signal for access modeling.
    immune_score = tcell_score + bcell_score + 0.5 * inflamed_score

    tumor_coords = coords[tumor_mask]
    immune_coords = coords[immune_mask]
    vascular_coords = coords[vascular_mask]
    non_tumor_coords = coords[~tumor_mask]

    dist_tumor_to_immune = np.full(adata.n_obs, np.nan, dtype=float)
    dist_tumor_to_vascular = np.full(adata.n_obs, np.nan, dtype=float)
    dist_tumor_to_nontumor = np.full(adata.n_obs, np.nan, dtype=float)

    d = nearest_distance(tumor_coords, immune_coords)
    if len(d) > 0:
        dist_tumor_to_immune[tumor_mask] = d

    d = nearest_distance(tumor_coords, vascular_coords)
    if len(d) > 0:
        dist_tumor_to_vascular[tumor_mask] = d

    d = nearest_distance(tumor_coords, non_tumor_coords)
    if len(d) > 0:
        dist_tumor_to_nontumor[tumor_mask] = d

    vascular_prox_score = np.zeros(adata.n_obs, dtype=float)
    immune_prox_score = np.zeros(adata.n_obs, dtype=float)

    if np.isfinite(dist_tumor_to_vascular[tumor_mask]).sum() > 0:
        vascular_prox_score[tumor_mask] = 1.0 - normalize_01(dist_tumor_to_vascular[tumor_mask])

    if np.isfinite(dist_tumor_to_immune[tumor_mask]).sum() > 0:
        immune_prox_score[tumor_mask] = 1.0 - normalize_01(dist_tumor_to_immune[tumor_mask])

    barrier_score, accessibility_score = build_barrier_and_accessibility_scores(
        stromal_score=stromal_score,
        hypoxia_score=hypoxia_score,
        ecm_score=ecm_score,
        angiogenic_score=angiogenic_score,
        vascular_prox_score=vascular_prox_score,
        immune_prox_score=immune_prox_score,
    )

    boundary_access_values = accessibility_score[tumor_boundary_mask]

    acc_thr = safe_quantile(boundary_access_values, ACCESSIBLE_BOUNDARY_QUANTILE)
    imp_thr = safe_quantile(boundary_access_values, IMPERMEABLE_BOUNDARY_QUANTILE)

    accessible_boundary_mask = np.zeros(adata.n_obs, dtype=bool)
    impermeable_boundary_mask = np.zeros(adata.n_obs, dtype=bool)

    if np.isfinite(acc_thr):
        accessible_boundary_mask[tumor_boundary_mask] = (
            accessibility_score[tumor_boundary_mask] >= acc_thr
        )

    if np.isfinite(imp_thr):
        impermeable_boundary_mask[tumor_boundary_mask] = (
            accessibility_score[tumor_boundary_mask] <= imp_thr
        )

    penetration_depth = compute_graph_depth_from_sources(
        tumor_mask=tumor_mask,
        source_mask=accessible_boundary_mask,
        adj=adj,
    )

    impermeable_depth = compute_graph_depth_from_sources(
        tumor_mask=tumor_mask,
        source_mask=impermeable_boundary_mask,
        adj=adj,
    )

    tumor_depth = penetration_depth[tumor_mask]
    tumor_depth_finite = finite_values(tumor_depth)
    max_depth = safe_max(tumor_depth_finite)

    normalized_penetration_depth = np.full(adata.n_obs, np.nan, dtype=float)

    if np.isfinite(max_depth) and max_depth > 0:
        normalized_penetration_depth[tumor_mask] = penetration_depth[tumor_mask] / max_depth
    else:
        normalized_penetration_depth[tumor_mask] = 0.0

    frac_within_1 = float(np.mean(tumor_depth_finite <= 1)) if len(tumor_depth_finite) else np.nan
    frac_within_2 = float(np.mean(tumor_depth_finite <= 2)) if len(tumor_depth_finite) else np.nan
    frac_within_3 = float(np.mean(tumor_depth_finite <= 3)) if len(tumor_depth_finite) else np.nan
    frac_deeper_4 = float(np.mean(tumor_depth_finite >= 4)) if len(tumor_depth_finite) else np.nan

    stromal_depth_slope, stromal_depth_r2 = fit_linear_slope(tumor_depth, stromal_score[tumor_mask])
    hypoxia_depth_slope, hypoxia_depth_r2 = fit_linear_slope(tumor_depth, hypoxia_score[tumor_mask])
    angiogenic_depth_slope, angiogenic_depth_r2 = fit_linear_slope(tumor_depth, angiogenic_score[tumor_mask])
    immune_depth_slope, immune_depth_r2 = fit_linear_slope(tumor_depth, immune_score[tumor_mask])
    barrier_depth_slope, barrier_depth_r2 = fit_linear_slope(tumor_depth, barrier_score[tumor_mask])
    access_depth_slope, access_depth_r2 = fit_linear_slope(tumor_depth, accessibility_score[tumor_mask])

    boundary_access_std = safe_std(boundary_access_values)
    boundary_access_range = (
        safe_max(boundary_access_values) - safe_min(boundary_access_values)
        if len(finite_values(boundary_access_values)) > 0
        else np.nan
    )

    accessible_cc = compute_connected_component_features(
        accessible_boundary_mask,
        adj,
        adata.n_obs,
    )

    impermeable_cc = compute_connected_component_features(
        impermeable_boundary_mask,
        adj,
        adata.n_obs,
    )

    euclid_penetration = np.full(adata.n_obs, np.nan, dtype=float)
    d = nearest_distance(tumor_coords, coords[accessible_boundary_mask])
    if len(d) > 0:
        euclid_penetration[tumor_mask] = d

    core_cutoff = safe_quantile(tumor_depth_finite, 0.75)
    core_mask = np.zeros(adata.n_obs, dtype=bool)

    if np.isfinite(core_cutoff):
        core_mask[tumor_mask] = penetration_depth[tumor_mask] >= core_cutoff

    row = {
        "sample_id": sample_id,
        "access_h5ad_used": str(h5ad_path),
        "access_annotation_column_used": annotation_col if annotation_col is not None else "none",
        "access_tumor_fallback_score_col_used": tumor_fallback_col if tumor_fallback_col is not None else "none",

        "access_stromal_score_col_used": stromal_col if stromal_col is not None else "none",
        "access_hypoxia_score_col_used": hypoxia_col if hypoxia_col is not None else "none",
        "access_ecm_remodel_score_col_used": ecm_col if ecm_col is not None else "none",
        "access_angiogenic_score_col_used": angiogenic_col if angiogenic_col is not None else "none",
        "access_tumor_score_col_used": tumor_col if tumor_col is not None else "none",
        "access_inflamed_score_col_used": inflamed_col if inflamed_col is not None else "none",
        "access_tcell_score_col_used": tcell_col if tcell_col is not None else "none",
        "access_bcell_score_col_used": bcell_col if bcell_col is not None else "none",
        "access_myeloid_score_col_used": myeloid_col if myeloid_col is not None else "none",

        "access_total_spots": int(adata.n_obs),
        "access_tumor_spots": int(tumor_mask.sum()),
        "access_tumor_fraction_of_slide": float(tumor_mask.mean()),
        "access_boundary_tumor_spots": int(tumor_boundary_mask.sum()),
        "access_boundary_tumor_fraction_of_tumor": float(tumor_boundary_mask[tumor_mask].mean()),

        "access_immune_spots": int(immune_mask.sum()),
        "access_stromal_spots": int(stromal_mask.sum()),
        "access_vascular_spots": int(vascular_mask.sum()),
        "access_hypoxic_spots": int(hypoxic_mask.sum()),

        "access_accessible_boundary_threshold": acc_thr,
        "access_impermeable_boundary_threshold": imp_thr,

        "access_accessible_boundary_spots": int(accessible_boundary_mask.sum()),
        "access_impermeable_boundary_spots": int(impermeable_boundary_mask.sum()),

        "access_accessible_boundary_fraction_of_tumor_boundary": float(accessible_boundary_mask[tumor_boundary_mask].mean()),
        "access_impermeable_boundary_fraction_of_tumor_boundary": float(impermeable_boundary_mask[tumor_boundary_mask].mean()),

        "access_boundary_accessibility_mean": safe_mean(boundary_access_values),
        "access_boundary_accessibility_median": safe_median(boundary_access_values),
        "access_boundary_accessibility_std": boundary_access_std,
        "access_boundary_accessibility_range": boundary_access_range,

        "access_barrier_score_tumor_mean": safe_mean(barrier_score[tumor_mask]),
        "access_barrier_score_tumor_median": safe_median(barrier_score[tumor_mask]),
        "access_accessibility_score_tumor_mean": safe_mean(accessibility_score[tumor_mask]),
        "access_accessibility_score_tumor_median": safe_median(accessibility_score[tumor_mask]),

        "access_frac_tumor_within_graph_depth_1": frac_within_1,
        "access_frac_tumor_within_graph_depth_2": frac_within_2,
        "access_frac_tumor_within_graph_depth_3": frac_within_3,
        "access_frac_tumor_deeper_than_graph_depth_4": frac_deeper_4,

        "access_penetration_graph_depth_mean": safe_mean(tumor_depth_finite),
        "access_penetration_graph_depth_median": safe_median(tumor_depth_finite),
        "access_penetration_graph_depth_std": safe_std(tumor_depth_finite),
        "access_penetration_graph_depth_max": safe_max(tumor_depth_finite),

        "access_penetration_euclid_mean": safe_mean(euclid_penetration[tumor_mask]),
        "access_penetration_euclid_median": safe_median(euclid_penetration[tumor_mask]),
        "access_penetration_euclid_std": safe_std(euclid_penetration[tumor_mask]),
        "access_penetration_euclid_max": safe_max(euclid_penetration[tumor_mask]),

        "access_core_spots": int(core_mask.sum()),
        "access_core_fraction_of_tumor": float(core_mask[tumor_mask].mean()) if tumor_mask.sum() > 0 else np.nan,
        "access_core_barrier_mean": safe_mean(barrier_score[core_mask]),
        "access_core_accessibility_mean": safe_mean(accessibility_score[core_mask]),

        "access_stromal_depth_slope": stromal_depth_slope,
        "access_stromal_depth_r2": stromal_depth_r2,
        "access_hypoxia_depth_slope": hypoxia_depth_slope,
        "access_hypoxia_depth_r2": hypoxia_depth_r2,
        "access_angiogenic_depth_slope": angiogenic_depth_slope,
        "access_angiogenic_depth_r2": angiogenic_depth_r2,
        "access_immune_depth_slope": immune_depth_slope,
        "access_immune_depth_r2": immune_depth_r2,
        "access_barrier_depth_slope": barrier_depth_slope,
        "access_barrier_depth_r2": barrier_depth_r2,
        "access_accessibility_depth_slope": access_depth_slope,
        "access_accessibility_depth_r2": access_depth_r2,

        "access_mean_tumor_to_immune_distance": safe_mean(dist_tumor_to_immune[tumor_mask]),
        "access_median_tumor_to_immune_distance": safe_median(dist_tumor_to_immune[tumor_mask]),
        "access_mean_tumor_to_vascular_distance": safe_mean(dist_tumor_to_vascular[tumor_mask]),
        "access_median_tumor_to_vascular_distance": safe_median(dist_tumor_to_vascular[tumor_mask]),
        "access_mean_tumor_to_nontumor_distance": safe_mean(dist_tumor_to_nontumor[tumor_mask]),
        "access_median_tumor_to_nontumor_distance": safe_median(dist_tumor_to_nontumor[tumor_mask]),
    }

    row.update({f"access_accessible_boundary_{k}": v for k, v in accessible_cc.items()})
    row.update({f"access_impermeable_boundary_{k}": v for k, v in impermeable_cc.items()})

    row.update(summarize_profile(accessibility_score[tumor_mask], "access_accessibility_score_tumor"))
    row.update(summarize_profile(barrier_score[tumor_mask], "access_barrier_score_tumor_profile"))
    row.update(summarize_profile(immune_score[tumor_mask], "access_immune_score_tumor"))
    row.update(summarize_profile(stromal_score[tumor_mask], "access_stromal_score_tumor"))
    row.update(summarize_profile(hypoxia_score[tumor_mask], "access_hypoxia_score_tumor"))
    row.update(summarize_profile(angiogenic_score[tumor_mask], "access_angiogenic_score_tumor"))

    profile_label, profile_confidence, profile_notes = assign_accessibility_profile(row)

    row["access_profile_label"] = profile_label
    row["access_profile_confidence"] = profile_confidence
    row["access_profile_rule_hits"] = profile_notes
    row["access_boundary_quality"] = boundary_flag

    spot_df = pd.DataFrame({
        "spot_id": adata.obs_names.astype(str),
        "tumor_mask": tumor_mask.astype(int),
        "tumor_boundary_mask": tumor_boundary_mask.astype(int),
        "accessible_boundary_mask": accessible_boundary_mask.astype(int),
        "impermeable_boundary_mask": impermeable_boundary_mask.astype(int),
        "core_mask": core_mask.astype(int),
        "stromal_score": stromal_score,
        "hypoxia_score": hypoxia_score,
        "ecm_remodeling_score": ecm_score,
        "angiogenic_score": angiogenic_score,
        "tumor_epithelial_score": tumor_score,
        "immune_score": immune_score,
        "myeloid_score": myeloid_score,
        "barrier_score": barrier_score,
        "accessibility_score": accessibility_score,
        "graph_penetration_depth": penetration_depth,
        "impermeable_graph_depth": impermeable_depth,
        "normalized_penetration_depth": normalized_penetration_depth,
        "euclid_penetration_distance": euclid_penetration,
        "tumor_to_immune_distance": dist_tumor_to_immune,
        "tumor_to_vascular_distance": dist_tumor_to_vascular,
        "tumor_to_nontumor_distance": dist_tumor_to_nontumor,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })

    spot_path = per_sample_dir / f"{sample_id}_accessibility_spot_profile.csv"
    spot_df.to_csv(spot_path, index=False)

    return row, {
        "sample_id": sample_id,
        "status": "ok",
        "h5ad_path": str(h5ad_path),
        "tumor_spots": int(tumor_mask.sum()),
        "boundary_spots": int(tumor_boundary_mask.sum()),
        "accessible_boundary_spots": int(accessible_boundary_mask.sum()),
        "impermeable_boundary_spots": int(impermeable_boundary_mask.sum()),
        "profile_label": profile_label,
        "error": "",
    }


def merge_accessibility_summary(base_path, access_df):
    """Merge accessibility features into the richest available slide-level table."""
    base_df = pd.read_csv(base_path)

    if "sample_id" not in base_df.columns:
        raise ValueError(f"{base_path} must contain sample_id")

    if access_df.empty:
        return base_df

    keep_cols = [
        col for col in access_df.columns
        if col == "sample_id" or col not in base_df.columns
    ]

    return base_df.merge(access_df[keep_cols], on="sample_id", how="left")


def build_summary_text(status_df, access_df):
    """Build a text summary for the accessibility profiling step."""
    lines = []
    lines.append("Accessibility profile summary")
    lines.append("")
    lines.append(f"Samples attempted: {len(status_df)}")
    lines.append(f"Accessibility rows written: {len(access_df)}")
    lines.append("")

    if not status_df.empty and "status" in status_df.columns:
        lines.append("Status counts:")
        for key, value in status_df["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if not access_df.empty and "access_profile_label" in access_df.columns:
        lines.append("")
        lines.append("Accessibility profile label counts:")
        for key, value in access_df["access_profile_label"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    failed = status_df[status_df["status"].astype(str).str.lower() != "ok"] if not status_df.empty else pd.DataFrame()

    lines.append("")
    lines.append(f"Non-OK samples: {len(failed)}")

    if len(failed) > 0:
        lines.append("")
        lines.append("Non-OK sample details:")
        for _, row in failed.iterrows():
            lines.append(f"  {row.get('sample_id', '')}: {row.get('status', '')} {row.get('error', '')}")

    return "\n".join(lines)


def main():
    """Run spatial accessibility profiling across all processed samples."""
    args = parse_args()
    output_root = get_output_root(args.config)

    out_dir = output_root / "output_06_build_accessibility_profiles"
    per_sample_dir = out_dir / "per_sample"

    ensure_dir(out_dir)
    ensure_dir(per_sample_dir)

    processing_report = read_processing_report(output_root)
    ok_samples = get_ok_samples(processing_report)
    sample_dirs = discover_sample_dirs(output_root, ok_samples=ok_samples)

    slide_rows = []
    status_rows = []

    print("=== Build accessibility profiles ===")
    print("Output root:", output_root)
    print("Samples discovered:", len(sample_dirs))
    print()

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        h5ad_path = find_consensus_h5ad(output_root, sample_id)
        if h5ad_path is None:
            h5ad_path = find_h5ad(sample_dir)

        if h5ad_path is None:
            status_rows.append({
                "sample_id": sample_id,
                "status": "missing_h5ad",
                "h5ad_path": "",
                "error": "",
            })
            continue

        print(f"Profiling {sample_id}")

        try:
            row, status = score_one_sample(
                sample_id=sample_id,
                h5ad_path=h5ad_path,
                per_sample_dir=per_sample_dir,
            )

            slide_rows.append(row)
            status_rows.append(status)

        except Exception as error:
            status_rows.append({
                "sample_id": sample_id,
                "status": "failed",
                "h5ad_path": str(h5ad_path),
                "error": f"{type(error).__name__}: {error}",
            })

    access_df = pd.DataFrame(slide_rows)
    status_df = pd.DataFrame(status_rows)

    base_path = choose_base_table(output_root)
    merged = merge_accessibility_summary(base_path, access_df)

    access_path = out_dir / "slide_accessibility_profiles.csv"
    status_path = out_dir / "accessibility_status.csv"
    merged_path = out_dir / "slide_features_with_accessibility.csv"
    summary_path = out_dir / "accessibility_profile_summary.txt"

    access_df.to_csv(access_path, index=False)
    status_df.to_csv(status_path, index=False)
    merged.to_csv(merged_path, index=False)

    summary_text = build_summary_text(status_df, access_df)
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print("DONE")
    print("Base table used:", base_path)
    print("Accessibility profiles:", access_path)
    print("Status:", status_path)
    print("Merged table:", merged_path)
    print("Summary:", summary_path)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()



