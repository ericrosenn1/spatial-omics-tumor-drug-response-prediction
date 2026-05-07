"""
Script: 09_build_motif_tables.py

Purpose:
Build motif-level and pairwise motif relationship tables for each processed Visium sample.

This script uses hotspot masks and accessibility profiles from earlier steps to quantify:
    1. individual motif size, position, compactness, and fragmentation
    2. pairwise distances, overlaps, adjacency, and interface structure between motifs
    3. gradient relationships such as hypoxia depth, immune exclusion, and barrier gradients
    4. slide-level wide summary features for downstream modeling

Inputs:
    processed_samples/<sample_id>/adata/03_final_pipeline_output.h5ad
    hotspot_metrics/per_sample/<sample_id>_hotspot_spot_masks.csv
    accessibility_profiles/per_sample/<sample_id>_accessibility_spot_profile.csv
    metabolic_concordance/slide_features_with_metabolic_concordance.csv
    hotspot_metrics/slide_features_with_hotspot_metrics.csv
    accessibility_profiles/slide_features_with_accessibility.csv

Outputs:
    motif_tables/per_sample/
    motif_tables/all_motif_table.csv
    motif_tables/all_pair_table.csv
    motif_tables/all_gradient_table.csv
    motif_tables/slide_motif_summary.csv
    motif_tables/slide_features_with_motif_tables.csv
    motif_tables/motif_table_status.csv
    motif_tables/motif_table_summary.txt

Usage:
    python scripts/09_build_motif_tables.py --config configs/visium_cohort_clean.yaml
"""

from pathlib import Path
import argparse
import itertools
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

SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"

H5AD_CANDIDATES = [
    "03_final_pipeline_output.h5ad",
    "02_after_clustering_and_annotation.h5ad",
    "01_after_filtering.h5ad",
]

# Minimum spots needed before a motif is treated as spatially meaningful.
MIN_MOTIF_SPOTS = 5

# Motifs smaller than this still get recorded, but detailed spatial metrics become less reliable.
MIN_MOTIF_SPOTS_FOR_PAIRS = 5

# If two motifs overlap at this fraction, we flag possible nesting or shared region structure.
NESTED_OVERLAP_FRACTION = 0.50

# Graph-depth gradients need enough finite paired observations to fit a slope.
MIN_POINTS_FOR_GRADIENT = 10


# =========================
# Motif definitions
# =========================

# These motif names should match the columns created by Script 07.
# Script 09 is intentionally built around the hotspot masks, because those masks are
# already the high-signal spatial niches we want to compare.
HOTSPOT_MOTIF_COLUMNS = {
    "tumor_epithelial": "tumor_epithelial_hotspot",
    "stromal_ecm": "stromal_ecm_hotspot",
    "ecm_remodeling": "ecm_remodeling_hotspot",
    "hypoxic_stress": "hypoxic_stress_hotspot",
    "angiogenic_vascular": "angiogenic_vascular_hotspot",
    "t_cell": "t_cell_hotspot",
    "immune_b_plasma": "immune_b_plasma_hotspot",
    "myeloid_macrophage": "myeloid_macrophage_hotspot",
    "interferon_inflamed": "interferon_inflamed_hotspot",
}

# Accessibility masks are useful motifs too, because they encode entry zones,
# impermeable regions, tumor boundary, and tumor core from Script 06.
ACCESSIBILITY_MOTIF_COLUMNS = {
    "tumor_mask": "tumor_mask",
    "tumor_boundary": "tumor_boundary_mask",
    "accessible_boundary": "accessible_boundary_mask",
    "impermeable_boundary": "impermeable_boundary_mask",
    "tumor_core": "core_mask",
}

# Score columns from Script 06 per-sample outputs.
# These are used for gradients and per-motif score summaries.
ACCESSIBILITY_SCORE_COLUMNS = [
    "stromal_score",
    "hypoxia_score",
    "ecm_remodeling_score",
    "angiogenic_score",
    "tumor_epithelial_score",
    "immune_score",
    "myeloid_score",
    "barrier_score",
    "accessibility_score",
    "graph_penetration_depth",
    "impermeable_graph_depth",
    "normalized_penetration_depth",
    "euclid_penetration_distance",
    "tumor_to_immune_distance",
    "tumor_to_vascular_distance",
    "tumor_to_nontumor_distance",
]

# Pairwise relationships worth emphasizing in the slide-level wide summary.
# The all_pair_table will still contain every pair, but these become named features.
IMPORTANT_PAIRS = [
    ("t_cell", "tumor_epithelial"),
    ("interferon_inflamed", "tumor_epithelial"),
    ("myeloid_macrophage", "tumor_epithelial"),
    ("stromal_ecm", "tumor_epithelial"),
    ("ecm_remodeling", "tumor_epithelial"),
    ("hypoxic_stress", "tumor_epithelial"),
    ("angiogenic_vascular", "tumor_epithelial"),
    ("hypoxic_stress", "angiogenic_vascular"),
    ("stromal_ecm", "t_cell"),
    ("stromal_ecm", "accessible_boundary"),
    ("hypoxic_stress", "tumor_core"),
    ("accessible_boundary", "tumor_core"),
    ("impermeable_boundary", "tumor_core"),
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
    """Read processing report if available."""
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
    """Find processed sample folders available for motif building."""
    processed_root = output_root / "output_02_01_process_samples_data"

    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed sample folder: {processed_root}")

    sample_dirs = sorted([p for p in processed_root.iterdir() if p.is_dir()])

    if ok_samples is not None:
        sample_dirs = [p for p in sample_dirs if p.name in ok_samples]

    return sample_dirs


def find_h5ad(sample_dir):
    """Find the best processed h5ad file for one sample."""
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


def find_hotspot_file(output_root, sample_id):
    """Find per-sample hotspot mask file from Script 07."""
    path = output_root / "output_07_append_hotspot_metrics" / "per_sample" / f"{sample_id}_hotspot_spot_masks.csv"

    if path.exists():
        return path

    return None


def find_accessibility_file(output_root, sample_id):
    """Find per-sample accessibility spot profile from Script 06."""
    path = output_root / "output_06_build_accessibility_profiles" / "per_sample" / f"{sample_id}_accessibility_spot_profile.csv"

    if path.exists():
        return path

    return None




def find_consensus_h5ad(output_root, sample_id):
    """Prefer the step 05 labeled h5ad when available."""
    path = Path(output_root) / "output_05_build_multi_axis_transcriptome_labels" / "per_sample_h5ad" / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    return path if path.exists() else None


def choose_base_table(output_root):
    """Choose richest slide-level table to merge motif outputs into."""
    candidates = [
        output_root / "output_08_context_alignment_and_metabolic_concordance" / "slide_features_with_metabolic_concordance.csv",
        output_root / "output_07_append_hotspot_metrics" / "slide_features_with_hotspot_metrics.csv",
        output_root / "output_06_build_accessibility_profiles" / "slide_features_with_accessibility.csv",
        output_root / "output_05_build_multi_axis_transcriptome_labels" / "slide_features_with_multi_axis_labels.csv",
        output_root / "output_04_score_and_label_slides" / "slide_features_scored_labeled.csv",
        output_root / "output_03_merge_slide_features" / "merged_slide_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find a usable slide-level feature table")


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


def safe_fraction(numerator, denominator):
    """Return numerator divided by denominator with zero protection."""
    if denominator is None or denominator == 0:
        return np.nan

    return float(numerator) / float(denominator)


def fit_linear_slope(x, y):
    """Fit simple slope and R2 for gradient summaries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < MIN_POINTS_FOR_GRADIENT:
        return np.nan, np.nan

    # Avoid unstable fits when all x or all y values are the same.
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
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
# Spatial graph helpers
# =========================

def ensure_spatial_graph(adata):
    """Ensure AnnData has a spatial neighbor graph."""
    if SPATIAL_CONNECTIVITY_KEY in adata.obsp:
        return

    # Squidpy creates the spatial graph from adata.obsm["spatial"].
    # We need this graph for adjacency, interface edges, and graph distances.
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
    """Compute nearest Euclidean distance from source coordinates to reference coordinates."""
    if src_coords.shape[0] == 0 or ref_coords.shape[0] == 0:
        return np.array([], dtype=float)

    # KDTree keeps this fast even when motifs have many spots.
    tree = cKDTree(ref_coords)
    dists, _ = tree.query(src_coords, k=1)

    return np.asarray(dists, dtype=float)


def centroid(coords):
    """Compute centroid of a coordinate matrix."""
    if coords.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=float)

    return np.nanmean(coords, axis=0)


def euclidean_distance(point_a, point_b):
    """Compute Euclidean distance between two 2D points."""
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)

    if not np.all(np.isfinite(point_a)) or not np.all(np.isfinite(point_b)):
        return np.nan

    return float(np.linalg.norm(point_a - point_b))


def compute_connected_component_features(mask, adj, n_total):
    """Summarize connected components for a motif mask."""
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return {
            "n_components": 0,
            "largest_component_spots": 0,
            "largest_component_fraction_of_slide": np.nan,
            "largest_component_fraction_of_motif": np.nan,
            "fragmentation_index": np.nan,
        }

    # Restrict graph to only the motif spots.
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
        "largest_component_fraction_of_slide": safe_fraction(largest, n_total),
        "largest_component_fraction_of_motif": safe_fraction(largest, len(idx)),
        "fragmentation_index": safe_fraction(n_comp, len(idx)),
    }


def count_interface_edges(mask_a, mask_b, adj):
    """Count spatial graph edges connecting motif A to motif B."""
    idx_a = np.where(mask_a)[0]

    if len(idx_a) == 0 or mask_b.sum() == 0:
        return 0

    edge_count = 0

    for idx in idx_a:
        neighbors = adj[idx].indices

        if len(neighbors) == 0:
            continue

        # Count how many neighboring spots belong to motif B.
        edge_count += int(mask_b[neighbors].sum())

    return int(edge_count)

def get_pairable_motifs(motif_masks):
    """Select motif masks for pairwise analysis using adaptive fallback."""
    motif_sizes = {
        name: int(mask.sum())
        for name, mask in motif_masks.items()
    }

    # First pass: normal threshold
    selected = [
        name for name, size in motif_sizes.items()
        if size >= MIN_MOTIF_SPOTS_FOR_PAIRS
    ]

    if len(selected) >= 2:
        return selected, "standard"

    # Second pass: relaxed threshold
    relaxed_min = max(2, MIN_MOTIF_SPOTS_FOR_PAIRS // 2)

    selected = [
        name for name, size in motif_sizes.items()
        if size >= relaxed_min
    ]

    if len(selected) >= 2:
        return selected, "relaxed"

    # Final fallback: keep largest nonzero motifs
    nonzero = [
        name for name, size in sorted(
            motif_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if size > 0
    ]

    selected = nonzero[:4]

    if len(selected) >= 2:
        return selected, "top_nonzero"

    return selected, "insufficient"

# =========================
# Per-motif table
# =========================

def build_motif_table(sample_id, coords, adj, motif_masks, score_df):
    """Build per-motif table for one sample."""
    rows = []

    for motif_name, mask in motif_masks.items():

        idx = np.where(mask)[0]
        n_spots = len(idx)

        if n_spots == 0:
            continue

        coords_motif = coords[mask]
        cent = centroid(coords_motif)

        cc = compute_connected_component_features(mask, adj, len(mask))

        row = {
            "sample_id": sample_id,
            "motif": motif_name,
            "n_spots": n_spots,
            "fraction_of_slide": safe_fraction(n_spots, len(mask)),
            "centroid_x": cent[0],
            "centroid_y": cent[1],
        }

        row.update({f"cc_{k}": v for k, v in cc.items()})

        # Add score summaries (only for relevant columns present)
        for col in score_df.columns:
            values = score_df[col].values[mask]

            row[f"{col}_mean"] = safe_mean(values)
            row[f"{col}_median"] = safe_median(values)
            row[f"{col}_max"] = safe_max(values)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Pairwise motif table
# =========================

def build_pair_table(sample_id, coords, adj, motif_masks):
    """Build pairwise motif relationships for one sample."""
    rows = []

    pairable_motifs, pair_quality = get_pairable_motifs(motif_masks)

    for a, b in itertools.combinations(pairable_motifs, 2):

        mask_a = motif_masks[a]
        mask_b = motif_masks[b]

        coords_a = coords[mask_a]
        coords_b = coords[mask_b]

        centroid_a = centroid(coords_a)
        centroid_b = centroid(coords_b)

        centroid_dist = euclidean_distance(centroid_a, centroid_b)

        # nearest neighbor distance
        nn_dist = safe_mean(nearest_distance(coords_a, coords_b))

        # overlap
        overlap = (mask_a & mask_b).sum()
        overlap_fraction = safe_fraction(overlap, min(mask_a.sum(), mask_b.sum()))

        # adjacency
        interface_edges = count_interface_edges(mask_a, mask_b, adj)

        row = {
            "sample_id": sample_id,
            "motif_a": a,
            "motif_b": b,
            "centroid_distance": centroid_dist,
            "nearest_distance": nn_dist,
            "overlap_fraction": overlap_fraction,
            "interface_edges": interface_edges,
            "pair_selection_quality": pair_quality,
        }

        # nesting
        if overlap_fraction >= NESTED_OVERLAP_FRACTION:
            row["relationship"] = "nested"
        elif overlap > 0:
            row["relationship"] = "overlapping"
        elif interface_edges > 0:
            row["relationship"] = "adjacent"
        else:
            row["relationship"] = "separate"

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Gradient table
# =========================

def build_gradient_table(sample_id, score_df):
    """Build gradient relationships for one sample."""
    rows = []

    if "graph_penetration_depth" not in score_df.columns:
        return pd.DataFrame()

    depth = score_df["graph_penetration_depth"].values

    if np.isfinite(depth).sum() < MIN_POINTS_FOR_GRADIENT:
        return pd.DataFrame()
    
    for col in score_df.columns:

        if col == "graph_penetration_depth":
            continue

        values = score_df[col].values

        slope, r2 = fit_linear_slope(depth, values)

        rows.append({
            "sample_id": sample_id,
            "feature": col,
            "slope_vs_depth": slope,
            "r2_vs_depth": r2,
        })

    return pd.DataFrame(rows)


# =========================
# Combine motif masks
# =========================

def build_all_motif_masks(hotspot_df, access_df):
    """Combine hotspot and accessibility masks."""
    motif_masks = {}

    for name, col in HOTSPOT_MOTIF_COLUMNS.items():
        if col in hotspot_df.columns:
            motif_masks[name] = hotspot_df[col].astype(bool).values

    for name, col in ACCESSIBILITY_MOTIF_COLUMNS.items():
        if col in access_df.columns:
            motif_masks[name] = access_df[col].astype(bool).values

    return motif_masks


# =========================
# Per-sample processing
# =========================

def process_sample(sample_id, h5ad_path, hotspot_path, access_path, output_dir):
    """Process one sample into motif, pair, and gradient tables."""
    adata = sc.read_h5ad(h5ad_path)

    coords = get_spatial_coordinates(adata)
    ensure_spatial_graph(adata)
    adj = get_spatial_adjacency(adata)

    hotspot_df = pd.read_csv(hotspot_path)
    access_df = pd.read_csv(access_path)

    # Combine score columns
    score_df = access_df[[c for c in ACCESSIBILITY_SCORE_COLUMNS if c in access_df.columns]].copy()
    if score_df.shape[1] == 0:
        score_df = pd.DataFrame(index=access_df.index)

    motif_masks = build_all_motif_masks(hotspot_df, access_df)

    motif_table = build_motif_table(sample_id, coords, adj, motif_masks, score_df)
    pair_table = build_pair_table(sample_id, coords, adj, motif_masks)
    gradient_table = build_gradient_table(sample_id, score_df)

    # Save per-sample tables
    motif_table.to_csv(output_dir / f"{sample_id}_motif_table.csv", index=False)
    pair_table.to_csv(output_dir / f"{sample_id}_pair_table.csv", index=False)
    gradient_table.to_csv(output_dir / f"{sample_id}_gradient_table.csv", index=False)

    return motif_table, pair_table, gradient_table


# =========================
# Slide-level summary
# =========================

def build_slide_summary(motif_df, pair_df):
    """Build wide-format slide summary."""
    summary = {}

    # motif size summaries
    for _, row in motif_df.iterrows():
        key = f"motif_{row['motif']}_fraction"
        summary[key] = row["fraction_of_slide"]

    # If no valid pair rows exist, skip pair summary safely.
    if pair_df.empty or "motif_a" not in pair_df.columns or "motif_b" not in pair_df.columns:
        return summary

    # important pairs
    for a, b in IMPORTANT_PAIRS:
        subset = pair_df[
            ((pair_df["motif_a"] == a) & (pair_df["motif_b"] == b)) |
            ((pair_df["motif_a"] == b) & (pair_df["motif_b"] == a))
        ]

        if subset.empty:
            continue

        summary[f"pair_{a}_{b}_centroid_distance"] = subset["centroid_distance"].mean()
        summary[f"pair_{a}_{b}_overlap"] = subset["overlap_fraction"].mean()

    return summary


# =========================
# Main
# =========================

def main():
    """Run Step 09 motif table construction across all processed samples."""
    args = parse_args()
    output_root = get_output_root(args.config)

    out_dir = output_root / "output_09_build_motif_tables"
    per_sample_dir = out_dir / "per_sample"

    ensure_dir(out_dir)
    ensure_dir(per_sample_dir)

    processing_report = read_processing_report(output_root)
    ok_samples = get_ok_samples(processing_report)
    sample_dirs = discover_sample_dirs(output_root, ok_samples)

    all_motif = []
    all_pair = []
    all_gradient = []
    slide_rows = []
    status_rows = []

    print("=== Build motif tables ===")
    print("Samples:", len(sample_dirs))

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        h5ad_path = find_consensus_h5ad(output_root, sample_id)
        if h5ad_path is None:
            h5ad_path = find_h5ad(sample_dir)
        hotspot_path = find_hotspot_file(output_root, sample_id)
        access_path = find_accessibility_file(output_root, sample_id)

        if h5ad_path is None or hotspot_path is None or access_path is None:
            status_rows.append({"sample_id": sample_id, "status": "missing_inputs"})
            continue

        try:
            motif_df, pair_df, grad_df = process_sample(
                sample_id,
                h5ad_path,
                hotspot_path,
                access_path,
                per_sample_dir,
            )

            all_motif.append(motif_df)
            all_pair.append(pair_df)
            all_gradient.append(grad_df)

            slide_rows.append({
                "sample_id": sample_id,
                **build_slide_summary(motif_df, pair_df)
            })

            status_rows.append({"sample_id": sample_id, "status": "ok"})

        except Exception as e:
            status_rows.append({"sample_id": sample_id, "status": f"failed: {e}"})

    motif_all = pd.concat(all_motif, ignore_index=True) if all_motif else pd.DataFrame()
    pair_all = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    grad_all = pd.concat(all_gradient, ignore_index=True) if all_gradient else pd.DataFrame()

    slide_df = pd.DataFrame(slide_rows)
    status_df = pd.DataFrame(status_rows)

    motif_all.to_csv(out_dir / "all_motif_table.csv", index=False)
    pair_all.to_csv(out_dir / "all_pair_table.csv", index=False)
    grad_all.to_csv(out_dir / "all_gradient_table.csv", index=False)
    slide_df.to_csv(out_dir / "slide_motif_summary.csv", index=False)
    status_df.to_csv(out_dir / "motif_table_status.csv", index=False)

    base_path = choose_base_table(output_root)
    merged = pd.read_csv(base_path).merge(slide_df, on="sample_id", how="left")
    merged.to_csv(out_dir / "slide_features_with_motif_tables.csv", index=False)

    print("DONE")

if __name__ == "__main__":
    main()



