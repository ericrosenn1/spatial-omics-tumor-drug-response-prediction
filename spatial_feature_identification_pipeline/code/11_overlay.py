"""
Script: 11_overlay.py

Purpose:
Generate per-sample Visium histology overlays from the completed spatial feature
identification pipeline.

This is an add-on visualization script, not a core feature generation step.

It is designed to look like the original overlay script:
    solid filled regions for broad structural regions
    solid outline regions for functional programs
    dotted outline regions for metabolic programs
    faint background spots for tissue context
    interactive HTML with buttons for structure, function, metabolism, and all
    separate static-style HTML and optional PNG files

Inputs:
    processed_samples/<sample_id>/adata/*.h5ad
    hotspot_metrics/per_sample/<sample_id>_hotspot_spot_masks.csv
    accessibility_profiles/per_sample/<sample_id>_accessibility_spot_profile.csv
    signature_scores/per_sample/<sample_id>_signature_spot_scores.csv
    cleaned Visium image files from input_root/SAMPLE_*/spatial/

Outputs:
    overlays/<sample_id>/interactive_visium_region_overlay.html
    overlays/<sample_id>/combined_region_overlay.html
    overlays/<sample_id>/fine_structure_region_overlay.html
    overlays/<sample_id>/functional_region_overlay.html
    overlays/<sample_id>/metabolic_region_overlay.html
    overlays/<sample_id>/combined_region_overlay.png
    overlays/<sample_id>/fine_structure_region_overlay.png
    overlays/<sample_id>/functional_region_overlay.png
    overlays/<sample_id>/metabolic_region_overlay.png
    overlays/overlay_status.csv
    overlays/overlay_summary.txt

Typical usage:
    python scripts/11_overlay.py --config configs/visium_cohort_clean.yaml --sample SAMPLE_0000 --no-png
    python scripts/11_overlay.py --config configs/visium_cohort_clean.yaml --sample SAMPLE_0000
    python scripts/11_overlay.py --config configs/visium_cohort_clean.yaml --limit 5
    python scripts/11_overlay.py --config configs/visium_cohort_clean.yaml

Dependencies / Environment Setup:
    python -m pip install plotly pillow scipy

Optional for PNG export:
    python -m pip install kaleido

Notes:
    Plotly is required for interactive HTML overlays.
    Pillow is used for image loading.
    SciPy is used only if available for convex hull fallback support, but this
    script also includes a built-in convex hull helper.
    Kaleido is only needed for PNG export. HTML output works without it.
"""

from pathlib import Path
from collections import deque
import argparse
import json
import sys
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import io
import base64

from PIL import Image

import plotly.graph_objects as go


# =========================
# Project imports
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config


# =========================
# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# General settings
# =========================

H5AD_CANDIDATES = [
    "03_final_pipeline_output.h5ad",
    "02_after_clustering_and_annotation.h5ad",
    "02_processed.h5ad",
    "01_after_filtering.h5ad",
    "01_loaded.h5ad",
]

IMAGE_CANDIDATES_HIRES = [
    "tissue_hires_image.png",
    "*tissue_hires_image.png",
    "*hires*.png",
]

IMAGE_CANDIDATES_LOWRES = [
    "tissue_lowres_image.png",
    "*tissue_lowres_image.png",
    "*lowres*.png",
]

SCALEFACTOR_CANDIDATES = [
    "scalefactors_json.json",
    "*scalefactors_json.json",
]

SHOW_SPOTS_FAINTLY = True
FAINT_SPOT_SIZE = 4
FAINT_SPOT_OPACITY = 0.14

MIN_EXCLUSIVE_STRUCTURE_COMPONENT_SIZE = 5
MIN_OVERLAP_STRUCTURE_COMPONENT_SIZE = 4
MIN_FUNCTION_COMPONENT_SIZE = 4
MIN_METAB_COMPONENT_SIZE = 4

NEIGHBOR_DISTANCE_MULTIPLIER = 2.0

EXCLUSIVE_STRUCTURE_FILL_OPACITY = 0.24
EXCLUSIVE_STRUCTURE_LINE_WIDTH = 2
OVERLAP_STRUCTURE_LINE_WIDTH = 4
FUNCTION_LINE_WIDTH = 4
METAB_LINE_WIDTH = 4

STATIC_SCALE = 3
FIG_WIDTH = 1400
FIG_HEIGHT = 1100


# =========================
# Layer definitions
# =========================

# Filled structural regions. These get translucent fill, like the old script.
EXCLUSIVE_STRUCTURE_MASKS = {
    "tumor_mask": "tumor_mask",
    "stromal_ecm": "stromal_ecm_hotspot",
    "immune_b_plasma": "immune_b_plasma_hotspot",
    "myeloid_macrophage": "myeloid_macrophage_hotspot",
    "t_cell": "t_cell_hotspot",
    "angiogenic_vascular": "angiogenic_vascular_hotspot",
}

# Structural overlap regions. These get solid outline only.
OVERLAP_STRUCTURE_MASKS = {
    "tumor_epithelial": "tumor_epithelial_hotspot",
    "tumor_boundary": "tumor_boundary_mask",
    "tumor_core": "core_mask",
    "ecm_remodeling": "ecm_remodeling_hotspot",
    "hypoxic_stress": "hypoxic_stress_hotspot",
    "interferon_inflamed": "interferon_inflamed_hotspot",
    "accessible_boundary": "accessible_boundary_mask",
    "impermeable_boundary": "impermeable_boundary_mask",
}

# Functional regions. These get solid outlines.
FUNCTION_MASKS = {
    "hypoxic_stress": "hypoxic_stress_hotspot",
    "angiogenic_vascular": "angiogenic_vascular_hotspot",
    "interferon_inflamed": "interferon_inflamed_hotspot",
    "tumor_epithelial": "tumor_epithelial_hotspot",
    "ecm_remodeling": "ecm_remodeling_hotspot",
    "t_cell": "t_cell_hotspot",
    "myeloid_macrophage": "myeloid_macrophage_hotspot",
    "immune_b_plasma": "immune_b_plasma_hotspot",
}

# Metabolic regions. These are made from high-score spots and get dotted outlines.
METABOLIC_SCORE_CANDIDATES = {
    "glycolysis": [
        "metabolic_score__glycolysis",
        "simple_mean__glycolysis",
        "rank_percentile__glycolysis",
        "ucell__glycolysis",
        "custom_gsva__glycolysis",
    ],
    "oxphos": [
        "metabolic_score__oxphos",
        "simple_mean__oxphos",
        "rank_percentile__oxphos",
        "ucell__oxphos",
        "custom_gsva__oxphos",
        "hallmark_gsva__HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "reactome_gsva__REACTOME_RESPIRATORY_ELECTRON_TRANSPORT",
    ],
    "fatty_acid_oxidation": [
        "metabolic_score__fatty_acid_oxidation",
        "simple_mean__fatty_acid_oxidation",
        "rank_percentile__fatty_acid_oxidation",
        "ucell__fatty_acid_oxidation",
        "custom_gsva__fatty_acid_oxidation",
    ],
    "fatty_acid_synthesis": [
        "metabolic_score__fatty_acid_synthesis",
        "simple_mean__fatty_acid_synthesis",
        "rank_percentile__fatty_acid_synthesis",
        "ucell__fatty_acid_synthesis",
        "custom_gsva__fatty_acid_synthesis",
    ],
    "nucleotide_synthesis": [
        "metabolic_score__nucleotide_synthesis",
        "simple_mean__nucleotide_synthesis",
        "rank_percentile__nucleotide_synthesis",
        "ucell__nucleotide_synthesis",
        "custom_gsva__nucleotide_synthesis",
    ],
    "glutamine_metabolism": [
        "metabolic_score__glutamine_metabolism",
        "simple_mean__glutamine_metabolism",
        "rank_percentile__glutamine_metabolism",
        "ucell__glutamine_metabolism",
        "custom_gsva__glutamine_metabolism",
    ],
    "proline_collagen_support": [
        "metabolic_score__proline_collagen_support",
        "simple_mean__proline_collagen_support",
        "rank_percentile__proline_collagen_support",
        "ucell__proline_collagen_support",
        "custom_gsva__proline_collagen_support",
    ],
    "tryptophan_kynurenine": [
        "metabolic_score__tryptophan_kynurenine",
        "simple_mean__tryptophan_kynurenine",
        "rank_percentile__tryptophan_kynurenine",
        "ucell__tryptophan_kynurenine",
        "custom_gsva__tryptophan_kynurenine",
    ],
}

METABOLIC_QUANTILE = 0.75


# =========================
# Colors
# =========================

EXCLUSIVE_STRUCTURE_COLORS = {
    "tumor_mask": "#d62728",
    "stromal_ecm": "#2ca02c",
    "immune_b_plasma": "#1f77b4",
    "myeloid_macrophage": "#8c564b",
    "t_cell": "#9467bd",
    "angiogenic_vascular": "#17becf",
    "other": "#7f7f7f",
}

OVERLAP_STRUCTURE_COLORS = {
    "tumor_epithelial": "#d62728",
    "tumor_boundary": "#ff7f0e",
    "tumor_core": "#8b0000",
    "ecm_remodeling": "#98df8a",
    "hypoxic_stress": "#ff7f0e",
    "interferon_inflamed": "#e377c2",
    "accessible_boundary": "#10b981",
    "impermeable_boundary": "#111827",
    "other": "#7f7f7f",
}

FUNCTION_COLORS = {
    "hypoxic_stress": "#ff7f0e",
    "angiogenic_vascular": "#17becf",
    "interferon_inflamed": "#e377c2",
    "tumor_epithelial": "#d62728",
    "ecm_remodeling": "#98df8a",
    "t_cell": "#9467bd",
    "myeloid_macrophage": "#8c564b",
    "immune_b_plasma": "#1f77b4",
    "other": "#7f7f7f",
}

METABOLIC_COLORS = {
    "glycolysis": "#d62728",
    "oxidative_phosphorylation": "#1f77b4",
    "fatty_acid_oxidation": "#2ca02c",
    "fatty_acid_synthesis": "#ff7f0e",
    "nucleotide_synthesis": "#9467bd",
    "glutamine_metabolism": "#8c564b",
    "proline_collagen_support": "#17becf",
    "tryptophan_kynurenine": "#e377c2",
    "other": "#7f7f7f",
}


# =========================
# Argument and config helpers
# =========================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write HTML only. This avoids Kaleido dependency during testing.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate overlays even if output folder already exists.",
    )

    return parser.parse_args()


def get_paths(config_path):
    """Load config and return important root paths."""
    cfg = validate_config(load_config(config_path))

    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])

    return cfg, input_root, output_root


def ensure_dir(path):
    """Create directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


# =========================
# Input discovery helpers
# =========================

def read_processing_report(output_root):
    """Read processing report if available."""
    path = Path(output_root) / "output_02_process_samples_reports" / "processing_report.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_ok_samples(processing_report):
    """Return successfully processed sample IDs."""
    if processing_report.empty:
        return None

    if "sample_id" not in processing_report.columns:
        return None

    if "status" not in processing_report.columns:
        return None

    ok = processing_report.loc[
        processing_report["status"].astype(str).str.upper() == "OK",
        "sample_id",
    ]

    return set(ok.astype(str))


def discover_sample_dirs(output_root, sample=None, limit=None):
    """Find processed sample folders."""
    processing_report = read_processing_report(output_root)
    ok_samples = get_ok_samples(processing_report)

    processed_root = Path(output_root) / "output_02_01_process_samples_data"

    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed sample folder: {processed_root}")

    sample_dirs = sorted(
        p for p in processed_root.glob("SAMPLE_*")
        if p.is_dir()
    )

    if ok_samples is not None:
        sample_dirs = [p for p in sample_dirs if p.name in ok_samples]

    if sample is not None:
        sample_dirs = [p for p in sample_dirs if p.name == sample]

    if limit is not None:
        sample_dirs = sample_dirs[:limit]

    return sample_dirs


def find_h5ad(sample_dir):
    """Find best available AnnData file for one sample."""
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    try:
        output_root = sample_dir.parent.parent
        labeled = output_root / "output_05_build_multi_axis_transcriptome_labels" / "per_sample_h5ad" / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
        if labeled.exists():
            return labeled
    except Exception:
        pass

    adata_dir = Path(sample_dir) / "adata"

    for name in H5AD_CANDIDATES:
        path = adata_dir / name
        if path.exists():
            return path

    if adata_dir.exists():
        h5ads = sorted(adata_dir.glob("*.h5ad"))
        if h5ads:
            return h5ads[-1]

    return None


def find_first_glob(folder, patterns):
    """Find first file matching one of several patterns."""
    folder = Path(folder)

    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

    return None


def read_json(path):
    """Read JSON safely."""
    if path is None or not Path(path).exists():
        return {}

    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return {}


def find_sample_spatial_files(input_root, sample_id):
    """Find image and scalefactor files from cleaned Visium folder."""
    sample_input = Path(input_root) / sample_id
    spatial_dir = sample_input / "spatial"

    if not spatial_dir.exists():
        return {
            "sample_input": sample_input,
            "spatial_dir": spatial_dir,
            "image_path": None,
            "image_kind": "none",
            "scalefactors_path": None,
        }

    image_path = find_first_glob(spatial_dir, IMAGE_CANDIDATES_HIRES)
    image_kind = "hires"

    if image_path is None:
        image_path = find_first_glob(spatial_dir, IMAGE_CANDIDATES_LOWRES)
        image_kind = "lowres" if image_path is not None else "none"

    scalefactors_path = find_first_glob(spatial_dir, SCALEFACTOR_CANDIDATES)

    return {
        "sample_input": sample_input,
        "spatial_dir": spatial_dir,
        "image_path": image_path,
        "image_kind": image_kind,
        "scalefactors_path": scalefactors_path,
    }


def find_hotspot_file(output_root, sample_id):
    """Find hotspot mask table from script 07."""
    path = Path(output_root) / "output_07_append_hotspot_metrics" / "per_sample" / f"{sample_id}_hotspot_spot_masks.csv"
    return path if path.exists() else None


def find_accessibility_file(output_root, sample_id):
    """Find accessibility spot profile table from script 06."""
    path = Path(output_root) / "output_06_build_accessibility_profiles" / "per_sample" / f"{sample_id}_accessibility_spot_profile.csv"
    return path if path.exists() else None


def find_signature_file(output_root, sample_id):
    """Find signature spot score table from script 05."""
    path = Path(output_root) / "output_05_build_multi_axis_transcriptome_labels" / "per_sample" / f"{sample_id}_spot_scores.csv"
    return path if path.exists() else None


# =========================
# Image and coordinate helpers
# =========================

def to_rgb_uint8(img):
    """Convert image array to uint8 RGB style."""
    arr = np.asarray(img)

    if arr.dtype != np.uint8:
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def load_image(image_path):
    """Load image as PIL image."""
    if image_path is None or not Path(image_path).exists():
        return None

    img = Image.open(image_path).convert("RGB")
    return img


def choose_scale_factor(image_kind, scalefactors):
    """Choose coordinate scale factor matching selected image resolution."""
    if image_kind == "hires":
        return float(scalefactors.get("tissue_hires_scalef", 1.0))

    if image_kind == "lowres":
        return float(scalefactors.get("tissue_lowres_scalef", 1.0))

    return 1.0


def get_sample_id_from_adata(adata, fallback):
    """Get sample ID from AnnData if possible."""
    if "sample_id" in adata.obs.columns:
        vals = adata.obs["sample_id"].dropna().astype(str).unique()
        if len(vals) > 0:
            return vals[0]

    if "sample_id" in adata.uns:
        return str(adata.uns["sample_id"])

    return str(fallback)


def build_coordinate_table(adata, scale_factor):
    """Build scaled image coordinate table from AnnData spatial coordinates."""
    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial']")

    coords = pd.DataFrame(
        np.asarray(adata.obsm["spatial"], dtype=float),
        columns=["x_fullres", "y_fullres"],
        index=adata.obs_names.astype(str),
    )

    coords = coords.reset_index().rename(columns={"index": "spot_id"})

    # This is the key fix: use image coordinate scale, not arbitrary normalization.
    coords["x"] = coords["x_fullres"] * scale_factor
    coords["y"] = coords["y_fullres"] * scale_factor

    return coords


# =========================
# Table merge helpers
# =========================

def read_optional_csv(path):
    """Read CSV or return empty DataFrame."""
    if path is None or not Path(path).exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def standardize_spot_id(df):
    """Ensure a spot_id column exists and is string typed."""
    if df.empty:
        return df

    out = df.copy()

    if "spot_id" not in out.columns:
        if "barcode" in out.columns:
            out = out.rename(columns={"barcode": "spot_id"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "spot_id"})

    if "spot_id" in out.columns:
        out["spot_id"] = out["spot_id"].astype(str)

    return out


def merge_spot_tables(coords, hotspot_df, access_df, signature_df):
    """Merge all per-spot sources onto coordinate table."""
    df = coords.copy()
    df["spot_id"] = df["spot_id"].astype(str)

    for extra in [hotspot_df, access_df, signature_df]:
        extra = standardize_spot_id(extra)

        if extra.empty or "spot_id" not in extra.columns:
            continue

        keep_cols = [
            c for c in extra.columns
            if c == "spot_id" or c not in df.columns
        ]

        df = df.merge(extra[keep_cols], on="spot_id", how="left")

    return df


def get_first_existing_col(df, candidates):
    """Return first candidate column that exists."""
    for col in candidates:
        if col in df.columns:
            return col

    return None


# =========================
# Geometry helpers
# =========================

def nearest_neighbor_distance(xy):
    """Estimate spot spacing from nearest-neighbor distance."""
    xy = np.asarray(xy, dtype=float)

    if len(xy) < 2:
        return 1.0

    dists = []

    for i in range(len(xy)):
        diff = xy - xy[i]
        dist2 = np.sum(diff * diff, axis=1)
        dist2[i] = np.inf
        dists.append(np.sqrt(np.min(dist2)))

    return float(np.nanmedian(dists))


def connected_components(points_df, threshold):
    """Split selected spots into spatially connected components."""
    if points_df.empty:
        return []

    xy = points_df[["x", "y"]].to_numpy(dtype=float)
    n = len(xy)

    visited = np.zeros(n, dtype=bool)
    comps = []

    for start in range(n):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        idxs = []

        while queue:
            i = queue.popleft()
            idxs.append(i)

            diff = xy - xy[i]
            dist2 = np.sum(diff * diff, axis=1)

            neighbors = np.where((dist2 <= threshold * threshold) & (~visited))[0]

            for j in neighbors:
                visited[j] = True
                queue.append(j)

        comps.append(points_df.iloc[idxs].copy())

    return comps


def convex_hull(points):
    """Compute convex hull using monotonic chain algorithm."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)

    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        """Return the 2D cross product used by the convex hull helper."""
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def expand_polygon(poly, scale=1.10):
    """Expand polygon slightly so boundaries are visible around spots."""
    if len(poly) < 3:
        return poly

    center = poly.mean(axis=0)
    expanded = center + (poly - center) * scale

    return expanded


def close_polygon(poly):
    """Close polygon by repeating first point."""
    if len(poly) == 0:
        return poly

    if np.allclose(poly[0], poly[-1]):
        return poly

    return np.vstack([poly, poly[0]])


def rgba_from_hex(hex_color, alpha):
    """Convert hex color to rgba string."""
    hex_color = str(hex_color).lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"

# =========================
# Trace helpers (core visual logic)
# =========================

def add_faint_spots(fig, df):
    """Add faint background spots for context."""
    if not SHOW_SPOTS_FAINTLY:
        return 0

    fig.add_trace(
        go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=FAINT_SPOT_SIZE,
                color="rgba(60,60,60,0.25)",
                opacity=FAINT_SPOT_OPACITY,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="spots",
        )
    )
    return 1


def region_hover(label, size, region_type):
    """Build hover text for one rendered overlay region."""
    return f"{region_type}: {label}<br>spots_in_region: {size}"


def add_region_traces(fig, df, mask_dict, color_map, min_size, line_width, fill=False, dash=None, region_type="region"):
    """Generic region builder used for structure / function / metabolic."""
    count = 0

    xy = df[["x", "y"]].to_numpy()
    spacing = nearest_neighbor_distance(xy)
    threshold = spacing * NEIGHBOR_DISTANCE_MULTIPLIER

    for label, mask_source in mask_dict.items():

        # mask_source can be either a column name or a boolean Series
        if isinstance(mask_source, str):
            if mask_source not in df.columns:
                continue
            mask = df[mask_source].astype(bool)
        else:
            mask = pd.Series(mask_source, index=df.index).astype(bool)

        sub = df[mask].copy()

        if sub.empty:
            continue

        comps = connected_components(sub, threshold)

        first = True

        for comp in comps:
            if len(comp) < min_size:
                continue

            poly = convex_hull(comp[["x", "y"]].to_numpy())

            if len(poly) < 3:
                continue

            poly = expand_polygon(poly, scale=1.12)
            poly = close_polygon(poly)

            color = color_map.get(label, color_map.get("other", "#7f7f7f"))

            fig.add_trace(
                go.Scatter(
                    x=poly[:, 0],
                    y=poly[:, 1],
                    mode="lines",
                    name=f"{region_type}: {label}",
                    legendgroup=f"{region_type}_{label}",
                    showlegend=first,
                    line=dict(
                        color=color,
                        width=line_width,
                        dash=dash,
                    ),
                    fill="toself" if fill else None,
                    fillcolor=rgba_from_hex(color, EXCLUSIVE_STRUCTURE_FILL_OPACITY) if fill else None,
                    hovertemplate=region_hover(label, len(comp), region_type) + "<extra></extra>",
                )
            )

            count += 1
            first = False

    return count


# =========================
# Metabolic thresholding
# =========================

def compute_metabolic_masks(df):
    """Create boolean masks for metabolic regions using quantiles."""
    out = {}

    for label, candidates in METABOLIC_SCORE_CANDIDATES.items():

        col = get_first_existing_col(df, candidates)

        if col is None:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")

        if vals.isna().all():
            continue

        cutoff = vals.quantile(METABOLIC_QUANTILE)

        out[label] = vals >= cutoff

    return out


# =========================
# Build figures
# =========================

def image_to_data_uri(img):
    """Embed PIL image into Plotly HTML."""
    if img is None:
        return None

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def make_base_figure(img):
    """Make base figure with image locked to image coordinates."""
    fig = go.Figure()

    if img is not None:
        img_w, img_h = img.size
        img_source = image_to_data_uri(img)

        fig.add_layout_image(
            dict(
                source=img_source,
                x=0,
                y=0,
                sizex=img_w,
                sizey=img_h,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )

        fig.update_xaxes(visible=False, range=[0, img_w])
        fig.update_yaxes(
            visible=False,
            range=[img_h, 0],
            scaleanchor="x",
            scaleratio=1,
        )

    else:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
    )

    return fig


# =========================
# Per-sample processing
# =========================

def save_figure_pair(fig, out_dir, stem, write_png=True):
    """Write one overlay as HTML and optionally PNG."""
    fig.write_html(str(out_dir / f"{stem}.html"))

    if write_png:
        try:
            fig.write_image(str(out_dir / f"{stem}.png"), scale=STATIC_SCALE)
        except Exception as error:
            print(f"  PNG export failed for {stem}: {type(error).__name__}: {error}")


def build_combined_figure(img, df, metab_masks):
    """Build combined structure, function, and metabolic overlay."""
    fig = make_base_figure(img)
    add_faint_spots(fig, df)

    add_region_traces(
        fig,
        df,
        EXCLUSIVE_STRUCTURE_MASKS,
        EXCLUSIVE_STRUCTURE_COLORS,
        MIN_EXCLUSIVE_STRUCTURE_COMPONENT_SIZE,
        EXCLUSIVE_STRUCTURE_LINE_WIDTH,
        fill=True,
        region_type="Structure",
    )

    add_region_traces(
        fig,
        df,
        OVERLAP_STRUCTURE_MASKS,
        OVERLAP_STRUCTURE_COLORS,
        MIN_OVERLAP_STRUCTURE_COMPONENT_SIZE,
        OVERLAP_STRUCTURE_LINE_WIDTH,
        fill=False,
        region_type="Structure overlap",
    )

    add_region_traces(
        fig,
        df,
        FUNCTION_MASKS,
        FUNCTION_COLORS,
        MIN_FUNCTION_COMPONENT_SIZE,
        FUNCTION_LINE_WIDTH,
        fill=False,
        region_type="Function",
    )

    add_region_traces(
        fig,
        df,
        metab_masks,
        METABOLIC_COLORS,
        MIN_METAB_COMPONENT_SIZE,
        METAB_LINE_WIDTH,
        fill=False,
        dash="dot",
        region_type="Metabolic",
    )

    fig.update_layout(title="Combined structural, functional, and metabolic overlay")
    return fig


def build_structure_figure(img, df):
    """Build structure-only overlay."""
    fig = make_base_figure(img)
    add_faint_spots(fig, df)

    add_region_traces(
        fig,
        df,
        EXCLUSIVE_STRUCTURE_MASKS,
        EXCLUSIVE_STRUCTURE_COLORS,
        MIN_EXCLUSIVE_STRUCTURE_COMPONENT_SIZE,
        EXCLUSIVE_STRUCTURE_LINE_WIDTH,
        fill=True,
        region_type="Structure",
    )

    add_region_traces(
        fig,
        df,
        OVERLAP_STRUCTURE_MASKS,
        OVERLAP_STRUCTURE_COLORS,
        MIN_OVERLAP_STRUCTURE_COMPONENT_SIZE,
        OVERLAP_STRUCTURE_LINE_WIDTH,
        fill=False,
        region_type="Structure overlap",
    )

    fig.update_layout(title="Fine structure region overlay")
    return fig


def build_function_figure(img, df):
    """Build function-only overlay."""
    fig = make_base_figure(img)
    add_faint_spots(fig, df)

    add_region_traces(
        fig,
        df,
        FUNCTION_MASKS,
        FUNCTION_COLORS,
        MIN_FUNCTION_COMPONENT_SIZE,
        FUNCTION_LINE_WIDTH,
        fill=False,
        region_type="Function",
    )

    fig.update_layout(title="Functional region overlay")
    return fig


def build_metabolic_figure(img, df, metab_masks):
    """Build metabolic-only overlay."""
    fig = make_base_figure(img)
    add_faint_spots(fig, df)

    add_region_traces(
        fig,
        df,
        metab_masks,
        METABOLIC_COLORS,
        MIN_METAB_COMPONENT_SIZE,
        METAB_LINE_WIDTH,
        fill=False,
        dash="dot",
        region_type="Metabolic",
    )

    fig.update_layout(title="Metabolic region overlay")
    return fig


def process_one_sample(sample_dir, input_root, output_root, write_png=True):
    """Generate combined and separated overlays for one sample."""
    sample_id = sample_dir.name

    out_dir = Path(output_root) / "output_11_overlay" / sample_id
    ensure_dir(out_dir)

    h5ad_path = find_h5ad(sample_dir)
    if h5ad_path is None:
        return {"sample_id": sample_id, "status": "missing_h5ad"}

    adata = sc.read_h5ad(h5ad_path)

    image_info = find_sample_spatial_files(input_root, sample_id)
    scalefactors = read_json(image_info["scalefactors_path"])
    scale = choose_scale_factor(image_info["image_kind"], scalefactors)

    img = load_image(image_info["image_path"])

    coords = build_coordinate_table(adata, scale)

    hotspot_df = read_optional_csv(find_hotspot_file(output_root, sample_id))
    access_df = read_optional_csv(find_accessibility_file(output_root, sample_id))
    sig_df = read_optional_csv(find_signature_file(output_root, sample_id))

    df = merge_spot_tables(coords, hotspot_df, access_df, sig_df)

    metab_masks = compute_metabolic_masks(df)

    figures = {
        "combined_region_overlay": build_combined_figure(img, df, metab_masks),
        "fine_structure_region_overlay": build_structure_figure(img, df),
        "functional_region_overlay": build_function_figure(img, df),
        "metabolic_region_overlay": build_metabolic_figure(img, df, metab_masks),
    }

    for stem, fig in figures.items():
        save_figure_pair(fig, out_dir, stem, write_png=write_png)

    return {
        "sample_id": sample_id,
        "status": "ok",
        "image_used": image_info["image_kind"],
        "n_spots": int(len(df)),
        "n_metabolic_masks": int(len(metab_masks)),
        "combined_html": str(out_dir / "combined_region_overlay.html"),
        "structure_html": str(out_dir / "fine_structure_region_overlay.html"),
        "function_html": str(out_dir / "functional_region_overlay.html"),
        "metabolic_html": str(out_dir / "metabolic_region_overlay.html"),
    }


# =========================
# Main
# =========================

def main():

    """Run Step 11 overlay generation across selected samples."""
    args = parse_args()

    cfg, input_root, output_root = get_paths(args.config)

    write_png = not args.no_png

    sample_dirs = discover_sample_dirs(
        output_root=output_root,
        sample=args.sample,
        limit=args.limit,
    )

    ensure_dir(Path(output_root) / "output_11_overlay")

    rows = []

    for s in sample_dirs:
        print(f"Processing {s.name}")

        try:
            row = process_one_sample(s, input_root, output_root, write_png)
        except Exception as e:
            row = {
                "sample_id": s.name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        Path(output_root) / "output_11_overlay" / "overlay_status.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()




