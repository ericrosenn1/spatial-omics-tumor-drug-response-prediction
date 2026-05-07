"""
Script: 07_append_hotspot_metrics.py

Purpose:
Append spatial hotspot metrics to the cohort level feature table.

This script identifies high-signal spatial hotspots for key biological programs
and summarizes their size, fragmentation, connectedness, overlap, and distance
relationships.

Inputs:
    processed_samples/<sample_id>/adata/03_final_pipeline_output.h5ad
    accessibility_profiles/slide_features_with_accessibility.csv
    signature_scores/slide_features_with_signature_scores.csv
    scored_labels/slide_features_scored_labeled.csv
    merged_features/merged_slide_features.csv
    processing/processing_report.csv

Outputs:
    hotspot_metrics/per_sample/
    hotspot_metrics/hotspot_slide_summary.csv
    hotspot_metrics/hotspot_status.csv
    hotspot_metrics/slide_features_with_hotspot_metrics.csv
    hotspot_metrics/hotspot_metrics_summary.txt

Usage:
    python scripts/07_append_hotspot_metrics.py --config configs/visium_cohort_clean.yaml
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
from scipy.sparse.csgraph import connected_components
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

# High-score hotspots are defined within each slide.
# 0.80 means the top 20 percent of valid spots for a feature.
HOTSPOT_QUANTILE = 0.80

# Minimum number of valid scored spots required before calling a hotspot meaningful.
MIN_VALID_SPOTS_FOR_HOTSPOT = 20

# Minimum number of hotspot spots required before spatial metrics are trusted.
MIN_HOTSPOT_SPOTS = 5

# Spatial graph key created by Squidpy.
SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"

# Prefer the final processed object, but allow fallbacks for incomplete samples.
H5AD_CANDIDATES = [
    "03_final_pipeline_output.h5ad",
    "02_after_clustering_and_annotation.h5ad",
    "01_after_filtering.h5ad",
]


# =========================
# Hotspot feature definitions
# =========================

# Each feature can have many possible column names because scores may come from
# simple scoring, UCell, GSVA, or earlier annotation steps.
HOTSPOT_FEATURES = {
    "tumor_epithelial": [
        "tumor_epithelial",
        "tumor_epithelial_score",
        "simple__tumor_epithelial",
        "ucell__tumor_epithelial",
        "gsva_custom__tumor_epithelial",
    ],
    "stromal_ecm": [
        "stromal_ecm",
        "stromal_ecm_score",
        "fibroblast_stroma_score",
        "simple__stromal_ecm",
        "ucell__stromal_ecm",
        "gsva_custom__stromal_ecm",
    ],
    "ecm_remodeling": [
        "ecm_remodeling",
        "ecm_remodeling_score",
        "simple__ecm_remodeling",
        "ucell__ecm_remodeling",
        "gsva_custom__ecm_remodeling",
    ],
    "hypoxic_stress": [
        "hypoxic_stress",
        "hypoxic_stress_score",
        "hypoxia_score",
        "simple__hypoxic_stress",
        "ucell__hypoxic_stress",
        "gsva_custom__hypoxic_stress",
    ],
    "angiogenic_vascular": [
        "angiogenic_vascular",
        "angiogenic_vascular_score",
        "endothelial_score",
        "simple__angiogenic_vascular",
        "ucell__angiogenic_vascular",
        "gsva_custom__angiogenic_vascular",
    ],
    "t_cell": [
        "t_cell",
        "t_cell_score",
        "simple__t_cell",
        "ucell__t_cell",
        "gsva_custom__t_cell",
    ],
    "immune_b_plasma": [
        "immune_b_plasma",
        "immune_b_plasma_score",
        "simple__immune_b_plasma",
        "ucell__immune_b_plasma",
        "gsva_custom__immune_b_plasma",
    ],
    "myeloid_macrophage": [
        "myeloid_macrophage",
        "myeloid_macrophage_score",
        "myeloid_score",
        "simple__myeloid_macrophage",
        "ucell__myeloid_macrophage",
        "gsva_custom__myeloid_macrophage",
    ],
    "interferon_inflamed": [
        "interferon_inflamed",
        "interferon_inflamed_score",
        "immune_general_score",
        "simple__interferon_inflamed",
        "ucell__interferon_inflamed",
        "gsva_custom__interferon_inflamed",
    ],
}


# Structural features use consensus region labels when available.
STRUCTURE_REGION_FEATURES = {
    "tumor_epithelial",
    "tumor_proliferative",
    "stromal_ecm",
    "ecm_remodeling",
    "angiogenic_vascular",
    "t_cell",
    "immune_b_plasma",
    "myeloid_macrophage",
}


# These annotation terms are used only as broad spatial reference masks.
REFERENCE_MASK_TERMS = {
    "tumor": [
        "tumor",
        "tumor_epithelial",
        "epithelial",
        "malignant",
        "cancer",
    ],
    "stromal": [
        "stromal",
        "stroma",
        "fibroblast",
        "ecm",
        "collagen",
        "matrix",
    ],
    "immune": [
        "immune",
        "t_cell",
        "b_cell",
        "plasma",
        "myeloid",
        "macrophage",
        "lymphocyte",
    ],
    "vascular": [
        "vascular",
        "endothelial",
        "angiogenic",
        "vessel",
        "blood_vessel",
    ],
    "hypoxic": [
        "hypoxic",
        "hypoxia",
        "hypoxic_stress",
    ],
}


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
    """Find processed sample folders available for hotspot scoring."""
    processed_root = output_root / "output_02_01_process_samples_data"

    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed sample folder: {processed_root}")

    sample_dirs = sorted([p for p in processed_root.iterdir() if p.is_dir()])

    # Use the processing report to avoid scoring skipped or failed samples.
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
    """Prefer the step 05 labeled h5ad so hotspots can use consensus structure labels."""
    path = Path(output_root) / "output_05_build_multi_axis_transcriptome_labels" / "per_sample_h5ad" / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    return path if path.exists() else None


def choose_base_table(output_root):
    """Choose the richest existing table to merge hotspot metrics into."""
    candidates = [
        output_root / "output_06_build_accessibility_profiles" / "slide_features_with_accessibility.csv",
        output_root / "output_05_build_multi_axis_transcriptome_labels" / "slide_features_with_multi_axis_labels.csv",
        output_root / "output_04_score_and_label_slides" / "slide_features_scored_labeled.csv",
        output_root / "output_03_merge_slide_features" / "merged_slide_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find accessibility, signature, scored, or merged slide table")


# =========================
# Safe numeric helpers
# =========================

def finite_values(values):
    """Return finite numeric values from an array-like object."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(values):
    """Return mean of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.mean(vals))


def safe_median(values):
    """Return median of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.median(vals))


def safe_std(values):
    """Return standard deviation of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.std(vals))


def safe_min(values):
    """Return minimum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.min(vals))


def safe_max(values):
    """Return maximum of finite values or NaN if empty."""
    vals = finite_values(values)

    if vals.size == 0:
        return np.nan

    return float(np.max(vals))


def safe_quantile(values, q):
    """Return quantile of finite values or NaN if empty."""
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

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return out

    out[mask] = (arr[mask] - vmin) / (vmax - vmin)

    return out


# =========================
# Spatial graph helpers
# =========================

def ensure_spatial_graph(adata):
    """Ensure AnnData has a spatial neighbor graph."""
    if SPATIAL_CONNECTIVITY_KEY in adata.obsp:
        return

    # Squidpy builds spatial neighbors from adata.obsm["spatial"].
    # This is required for connected components and adjacency metrics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sq.gr.spatial_neighbors(adata)


def get_spatial_coordinates(adata):
    """Return spatial coordinates from AnnData."""
    if "spatial" not in adata.obsm:
        raise ValueError("missing_spatial_coordinates")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    if coords.shape[0] != adata.n_obs:
        raise ValueError("spatial_coordinate_length_mismatch")

    return coords


def get_spatial_adjacency(adata):
    """Return spatial connectivity matrix as CSR."""
    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        raise ValueError("missing_spatial_connectivities")

    return csr_matrix(adata.obsp[SPATIAL_CONNECTIVITY_KEY])


def nearest_distance(src_coords, ref_coords):
    """Compute nearest Euclidean distance from source points to reference points."""
    if src_coords.shape[0] == 0 or ref_coords.shape[0] == 0:
        return np.array([], dtype=float)

    # KDTree avoids building a full pairwise distance matrix.
    tree = cKDTree(ref_coords)
    dists, _ = tree.query(src_coords, k=1)

    return np.asarray(dists, dtype=float)


def compute_connected_component_features(mask, adj, n_total):
    """Summarize connected components for one hotspot mask."""
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return {
            "n_components": 0,
            "largest_component_spots": 0,
            "largest_component_fraction_of_slide": np.nan,
            "largest_component_fraction_of_hotspot": np.nan,
            "fragmentation_index": np.nan,
        }

    # Restrict graph to hotspot spots only.
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

    largest_fraction_hotspot = largest / len(idx) if len(idx) > 0 else np.nan

    # Higher fragmentation means the hotspot is broken into many pieces.
    fragmentation_index = n_comp / len(idx) if len(idx) > 0 else np.nan

    return {
        "n_components": int(n_comp),
        "largest_component_spots": int(largest),
        "largest_component_fraction_of_slide": largest / n_total if n_total > 0 else np.nan,
        "largest_component_fraction_of_hotspot": largest_fraction_hotspot,
        "fragmentation_index": fragmentation_index,
    }


def compute_boundary_contact(mask_a, mask_b, adj):
    """Count spots in mask A touching mask B through the spatial graph."""
    idx_a = np.where(mask_a)[0]

    if len(idx_a) == 0 or mask_b.sum() == 0:
        return {
            "contact_spots": 0,
            "contact_fraction": np.nan,
        }

    contact = np.zeros(len(mask_a), dtype=bool)

    for idx in idx_a:
        neighbors = adj[idx].indices

        if len(neighbors) == 0:
            continue

        if np.any(mask_b[neighbors]):
            contact[idx] = True

    contact_spots = int(contact[mask_a].sum())

    return {
        "contact_spots": contact_spots,
        "contact_fraction": contact_spots / int(mask_a.sum()) if mask_a.sum() > 0 else np.nan,
    }


# =========================
# Column and annotation helpers
# =========================

def get_first_existing_col(df_or_obs, candidates):
    """Return first existing column from a candidate list."""
    cols = list(df_or_obs.columns)

    for col in candidates:
        if col in cols:
            return col

    return None


def get_score_vector(obs_df, feature_name, n):
    """Return a spot-level score vector for a hotspot feature."""
    candidates = [
        f"structure_score__{feature_name}",
        f"function_score__{feature_name}",
        f"metabolic_score__{feature_name}",
        f"simple_mean__{feature_name}",
        f"rank_percentile__{feature_name}",
        f"custom_gsva__{feature_name}",
        f"ucell__{feature_name}",
    ] + HOTSPOT_FEATURES.get(feature_name, [feature_name])

    col = get_first_existing_col(obs_df, candidates)

    if col is None:
        return np.full(n, np.nan, dtype=float), None

    values = pd.to_numeric(obs_df[col], errors="coerce").values.astype(float)

    # keep NaNs for filtering, but replace them with -inf for thresholding
    values_clean = values.copy()
    values_clean[~np.isfinite(values_clean)] = -np.inf

    return values_clean, col

# =========================
# Annotation mask helpers (for reference spatial context)
# =========================

def get_annotation_vector(obs_df):
    """Return best available annotation column."""
    candidates = [
        "structure_region_label_smoothed",
        "structure_region_label",
        "structure_dominant_label_raw",
        "structure_dominant_label",
        "cluster_annotation_final",
        "cluster_annotation_primary",
        "cluster_annotation_secondary",
        "annotation",
        "leiden",
    ]

    col = get_first_existing_col(obs_df, candidates)

    if col is None:
        return pd.Series(["unknown"] * obs_df.shape[0], index=obs_df.index), None

    return obs_df[col].astype(str), col


def build_reference_masks(annotation_series):
    """Build coarse spatial masks (tumor, stromal, immune, vascular, hypoxic)."""
    ann = annotation_series.fillna("unknown").str.lower()

    masks = {}

    for key, terms in REFERENCE_MASK_TERMS.items():
        masks[key] = ann.apply(lambda x: any(term in x for term in terms)).values

    return masks


# =========================
# Hotspot detection
# =========================

def compute_hotspot_mask(values):
    """Compute binary hotspot mask from a score vector."""
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]

    # If not enough valid values, skip hotspot detection.
    if len(finite) < MIN_VALID_SPOTS_FOR_HOTSPOT:
        return np.zeros_like(vals, dtype=bool), np.nan

    # Threshold is defined per-slide using quantile.
    threshold = np.nanquantile(finite, HOTSPOT_QUANTILE)

    # if threshold is invalid, skip hotspot detection entirely
    if not np.isfinite(threshold):
        return np.zeros_like(vals, dtype=bool), np.nan

    mask = np.zeros_like(vals, dtype=bool)
    mask[np.isfinite(vals)] = vals[np.isfinite(vals)] >= threshold

    return mask, threshold


def summarize_hotspot(mask, coords, adj, ref_masks, feature_name):
    """Summarize spatial properties of a hotspot."""
    n_total = len(mask)
    hotspot_idx = np.where(mask)[0]

    # Basic counts
    n_hotspot = int(mask.sum())
    frac_hotspot = n_hotspot / n_total if n_total > 0 else np.nan

    # If hotspot is too small, metrics are unstable â†’ return minimal info
    if n_hotspot < MIN_HOTSPOT_SPOTS:
        return {
            f"hotspot__{feature_name}_n_spots": n_hotspot,
            f"hotspot__{feature_name}_fraction": frac_hotspot,
            f"hotspot__{feature_name}_available": 0,
        }

    # Connected components
    cc = compute_connected_component_features(mask, adj, n_total)

    # Distance relationships
    coords_hotspot = coords[mask]

    dist_metrics = {}

    for ref_name, ref_mask in ref_masks.items():
        ref_coords = coords[ref_mask]

        d = nearest_distance(coords_hotspot, ref_coords)

        dist_metrics[f"hotspot__{feature_name}_dist_to_{ref_name}_mean"] = safe_mean(d)
        dist_metrics[f"hotspot__{feature_name}_dist_to_{ref_name}_median"] = safe_median(d)

    # Boundary contact
    contact_metrics = {}

    for ref_name, ref_mask in ref_masks.items():
        contact = compute_boundary_contact(mask, ref_mask, adj)

        contact_metrics[f"hotspot__{feature_name}_contact_{ref_name}_spots"] = contact["contact_spots"]
        contact_metrics[f"hotspot__{feature_name}_contact_{ref_name}_fraction"] = contact["contact_fraction"]

    return {
        f"hotspot__{feature_name}_n_spots": n_hotspot,
        f"hotspot__{feature_name}_fraction": frac_hotspot,
        f"hotspot__{feature_name}_available": 1,

        f"hotspot__{feature_name}_n_components": cc["n_components"],
        f"hotspot__{feature_name}_largest_component_spots": cc["largest_component_spots"],
        f"hotspot__{feature_name}_largest_component_fraction_of_slide": cc["largest_component_fraction_of_slide"],
        f"hotspot__{feature_name}_largest_component_fraction_of_hotspot": cc["largest_component_fraction_of_hotspot"],
        f"hotspot__{feature_name}_fragmentation_index": cc["fragmentation_index"],

        **dist_metrics,
        **contact_metrics,
    }


# =========================
# Per-sample processing
# =========================

def score_one_sample(sample_id, h5ad_path, per_sample_dir):
    """Compute hotspot metrics for one sample."""
    adata = sc.read_h5ad(h5ad_path)

    coords = get_spatial_coordinates(adata)
    ensure_spatial_graph(adata)
    adj = get_spatial_adjacency(adata)

    obs = adata.obs.copy()

    annotation_vec, annotation_col = get_annotation_vector(obs)
    ref_masks = build_reference_masks(annotation_vec)

    region_label_col = get_first_existing_col(
        obs,
        ["structure_region_label_smoothed", "structure_region_label"],
    )
    region_labels = obs[region_label_col].astype(str) if region_label_col is not None else None

    # Store per-spot masks for output
    spot_df = pd.DataFrame({
        "spot_id": adata.obs_names.astype(str),
        "sample_id": sample_id,
    })

    row = {
        "sample_id": sample_id,
        "hotspot_annotation_column_used": annotation_col if annotation_col else "none",
        "hotspot_total_spots": int(adata.n_obs),
    }

    # Iterate over defined hotspot feature spaces
    for feature_name in HOTSPOT_FEATURES.keys():

        values, col = get_score_vector(obs, feature_name, adata.n_obs)

        row[f"hotspot__{feature_name}_available"] = int(col is not None)
        row[f"hotspot__{feature_name}_score_column_used"] = col if col else "none"

        if region_labels is not None and feature_name in STRUCTURE_REGION_FEATURES:
            mask = region_labels.eq(feature_name).values
            threshold = np.nan
            col = region_label_col
        else:
            mask, threshold = compute_hotspot_mask(values)

        row[f"hotspot__{feature_name}_threshold"] = threshold

        # Store spot-level binary mask
        spot_df[f"{feature_name}_hotspot"] = mask.astype(int)

        # Summarize hotspot structure
        metrics = summarize_hotspot(
            mask=mask,
            coords=coords,
            adj=adj,
            ref_masks=ref_masks,
            feature_name=feature_name,
        )

        row.update(metrics)

    # Save per-sample spot table
    spot_path = per_sample_dir / f"{sample_id}_hotspot_spot_masks.csv"
    spot_df.to_csv(spot_path, index=False)

    return row, {
        "sample_id": sample_id,
        "status": "ok",
        "h5ad_path": str(h5ad_path),
        "error": "",
    }


# =========================
# Merge + summary
# =========================

def merge_hotspot_summary(base_path, hotspot_df):
    """Merge hotspot features into existing slide-level table."""
    base_df = pd.read_csv(base_path)

    if "sample_id" not in base_df.columns:
        raise ValueError("Base table missing sample_id")

    if hotspot_df.empty:
        return base_df

    keep_cols = [
        col for col in hotspot_df.columns
        if col == "sample_id" or col not in base_df.columns
    ]

    return base_df.merge(hotspot_df[keep_cols], on="sample_id", how="left")


def build_summary_text(status_df, hotspot_df):
    """Build text summary of hotspot metrics."""
    lines = []

    lines.append("Hotspot metrics summary")
    lines.append("")
    lines.append(f"Samples attempted: {len(status_df)}")
    lines.append(f"Hotspot rows written: {len(hotspot_df)}")

    if not status_df.empty:
        lines.append("")
        lines.append("Status counts:")
        for k, v in status_df["status"].value_counts().items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main():
    """Run hotspot metric extraction across all samples."""
    args = parse_args()
    output_root = get_output_root(args.config)

    out_dir = output_root / "output_07_append_hotspot_metrics"
    per_sample_dir = out_dir / "per_sample"

    ensure_dir(out_dir)
    ensure_dir(per_sample_dir)

    processing_report = read_processing_report(output_root)
    ok_samples = get_ok_samples(processing_report)
    sample_dirs = discover_sample_dirs(output_root, ok_samples)

    slide_rows = []
    status_rows = []

    print("=== Append hotspot metrics ===")
    print("Samples:", len(sample_dirs))

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        h5ad_path = find_consensus_h5ad(output_root, sample_id)
        if h5ad_path is None:
            h5ad_path = find_h5ad(sample_dir)

        if h5ad_path is None:
            status_rows.append({
                "sample_id": sample_id,
                "status": "missing_h5ad"
            })
            continue

        print(f"Processing {sample_id}")

        try:
            row, status = score_one_sample(
                sample_id=sample_id,
                h5ad_path=h5ad_path,
                per_sample_dir=per_sample_dir,
            )

            slide_rows.append(row)
            status_rows.append(status)

        except Exception as e:
            status_rows.append({
                "sample_id": sample_id,
                "status": f"failed: {e}"
            })

    hotspot_df = pd.DataFrame(slide_rows)
    status_df = pd.DataFrame(status_rows)

    base_path = choose_base_table(output_root)
    merged = merge_hotspot_summary(base_path, hotspot_df)

    hotspot_path = out_dir / "hotspot_slide_summary.csv"
    status_path = out_dir / "hotspot_status.csv"
    merged_path = out_dir / "slide_features_with_hotspot_metrics.csv"
    summary_path = out_dir / "hotspot_metrics_summary.txt"

    hotspot_df.to_csv(hotspot_path, index=False)
    status_df.to_csv(status_path, index=False)
    merged.to_csv(merged_path, index=False)

    summary_text = build_summary_text(status_df, hotspot_df)
    summary_path.write_text(summary_text)

    print("DONE")

if __name__ == "__main__":
    main()



