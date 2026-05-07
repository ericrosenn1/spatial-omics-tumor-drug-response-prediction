"""
Script: 12_data_analysis_and_visuals.py

Purpose:
    Create Visium region overlays from canonical Step 05 multi axis transcriptome outputs.

Project context:
    spatial_feature_identification_pipeline Step 12 visualization and review layer.

Documentation status:
    This active script has been documentation polished for GitHub review, publication
    methods support, and future maintenance. It remains the canonical Step 12
    visualization script.

Expected use:
    1. Run this script as the Step 12 visualization and review layer.
    2. Read the comments and docstrings when auditing overlay implementation details.
    3. Use the reports in code/_documentation_audit to confirm documentation only edits.

Safety notes:
    Documentation polishing used compile checks and AST comparisons with docstrings
    ignored. Added material is limited to comments, section headers, and docstrings.
"""


# =========================
# Imports
# =========================

from pathlib import Path
from collections import deque
import argparse
import base64
import io
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from PIL import Image

# Optional dependency block; script should still run with reduced output when missing.

# =========================
# Optional dependencies
# =========================

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False


# Resolve local project roots before importing pipeline helpers.

# =========================
# Project path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config validation centralizes paths and input root checks.

# =========================
# Pipeline helper imports
# =========================

from lib.config import load_config, validate_config


# =========================
# Pipeline input and output locations
# =========================

# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# Step 12 reads canonical Step 05 labels rather than recalculating labels.
INPUT_SUBDIR = "output_05_build_multi_axis_transcriptome_labels"
# Overlay review products are written under the Step 12 output tree.
OUT_SUBDIR = Path("output_12_data_analysis_and_visuals") / "12_visium_region_overlays"

# Static image settings keep PNG review output consistent across samples.

# =========================
# Static rendering settings
# =========================

DPI = 300
# Static image settings keep PNG review output consistent across samples.
STATIC_FIGSIZE = (10, 8)

# Component size thresholds suppress tiny fragmented regions in overlays.
MIN_EXCLUSIVE_STRUCTURE_COMPONENT_SIZE = 8
# Component size thresholds suppress tiny fragmented regions in overlays.
MIN_OVERLAP_STRUCTURE_COMPONENT_SIZE = 6
# Component size thresholds suppress tiny fragmented regions in overlays.
MIN_FUNCTION_COMPONENT_SIZE = 6
# Component size thresholds suppress tiny fragmented regions in overlays.
MIN_METAB_COMPONENT_SIZE = 3

# Neighbor distance scales contour smoothing to the Visium grid spacing.
NEIGHBOR_DISTANCE_MULTIPLIER = 1.35

# Faint spot settings preserve tissue context behind colored overlays.
SHOW_FAINT_SPOTS = True
FAINT_SPOT_SIZE = 8
FAINT_SPOT_ALPHA = 0.18

EXCLUSIVE_FILL_ALPHA = 0.18
EXCLUSIVE_LINE_WIDTH = 2.0
OVERLAP_LINE_WIDTH = 2.8
FUNCTION_LINE_WIDTH = 2.8
METAB_LINE_WIDTH = 2.8

# =========================
# Candidate score column definitions
# =========================

STRUCTURE_SCORE_BASES = {
    "tumor_epithelial": [
        "simple_tumor_epithelial",
        "score__tumor_epithelial",
        "tumor_epithelial_score",
        "tumor_epithelial",
    ],
    "stromal_ecm": [
        "simple_stromal_ecm",
        "score__fibroblast_stroma",
        "score__stromal_ecm",
        "stromal_ecm_score",
        "stromal_ecm",
    ],
    "immune_b_plasma": [
        "simple_immune_b_plasma",
        "immune_b_plasma_score",
        "immune_b_plasma",
    ],
    "myeloid_macrophage": [
        "simple_myeloid_macrophage",
        "score__myeloid",
        "myeloid_macrophage_score",
        "myeloid_macrophage",
    ],
    "t_cell": [
        "simple_t_cell",
        "t_cell_score",
        "t_cell",
    ],
}

OVERLAP_SCORE_BASES = {
    "tumor_proliferative": [
        "simple_tumor_proliferative",
        "score__proliferation",
        "tumor_proliferative_score",
        "tumor_proliferative",
    ],
    "ecm_remodeling": [
        "simple_ecm_remodeling",
        "ecm_remodeling_score",
        "ecm_remodeling",
    ],
    "hypoxic_stress": [
        "simple_hypoxic_stress",
        "score__hypoxia",
        "hypoxic_stress_score",
        "hypoxic_stress",
    ],
    "angiogenic_vascular": [
        "simple_angiogenic_vascular",
        "score__endothelial",
        "angiogenic_vascular_score",
        "angiogenic_vascular",
    ],
    "interferon_inflamed": [
        "simple_interferon_inflamed",
        "interferon_inflamed_score",
        "interferon_inflamed",
    ],
}

FUNCTION_SCORE_BASES = {
    "hypoxic_stress": OVERLAP_SCORE_BASES["hypoxic_stress"],
    "angiogenic_vascular": OVERLAP_SCORE_BASES["angiogenic_vascular"],
    "interferon_inflamed": OVERLAP_SCORE_BASES["interferon_inflamed"],
    "tumor_proliferative": OVERLAP_SCORE_BASES["tumor_proliferative"],
    "ecm_remodeling": OVERLAP_SCORE_BASES["ecm_remodeling"],
    "t_cell": STRUCTURE_SCORE_BASES["t_cell"],
    "myeloid_macrophage": STRUCTURE_SCORE_BASES["myeloid_macrophage"],
    "immune_b_plasma": STRUCTURE_SCORE_BASES["immune_b_plasma"],
}

METABOLIC_SCORE_BASES = {
    "glycolysis": [
        "simple_glycolysis",
        "metabolic_module__glycolysis",
        "metab_glycolysis_score",
        "glycolysis_score",
        "glycolysis",
    ],
    "oxphos": [
        "oxphos",
        "metabolic_score__oxphos",
        "simple_mean__oxphos",
        "rank_percentile__oxphos",
        "ucell__oxphos",
        "custom_gsva__oxphos",
        "simple_oxphos",
        "simple_oxidative_phosphorylation",
        "metabolic_module__oxidative_phosphorylation",
        "metab_oxphos_score",
        "oxphos_score",
        "oxidative_phosphorylation",
    ],
    "fatty_acid_oxidation": [
        "simple_fatty_acid_oxidation",
        "metabolic_module__fatty_acid_oxidation",
        "metab_fatty_acid_oxidation_score",
        "fatty_acid_oxidation_score",
    ],
    "fatty_acid_synthesis": [
        "simple_fatty_acid_synthesis",
        "metab_fatty_acid_synthesis_score",
        "fatty_acid_synthesis_score",
    ],
    "nucleotide_synthesis": [
        "simple_nucleotide_synthesis",
        "metab_nucleotide_synthesis_score",
        "nucleotide_synthesis_score",
    ],
    "glutamine_metabolism": [
        "simple_glutamine_metabolism",
        "metab_glutamine_metabolism_score",
        "glutamine_metabolism_score",
    ],
    "proline_collagen_support": [
        "simple_proline_collagen_support",
        "metab_proline_collagen_support_score",
        "proline_collagen_support_score",
    ],
    "tryptophan_kynurenine": [
        "simple_tryptophan_kynurenine",
        "metab_tryptophan_kynurenine_score",
        "tryptophan_kynurenine_score",
    ],
}

# =========================
# Overlay color palettes
# =========================

STRUCTURE_COLORS = {
    "tumor_epithelial": "#d62728",
    "tumor_proliferative": "#ff9896",
    "stromal_ecm": "#2ca02c",
    "ecm_remodeling": "#98df8a",
    "immune_b_plasma": "#1f77b4",
    "myeloid_macrophage": "#8c564b",
    "t_cell": "#9467bd",
    "angiogenic_vascular": "#17becf",
    "other": "#7f7f7f",
}

OVERLAP_COLORS = {
    "tumor_proliferative": "#ff9896",
    "ecm_remodeling": "#98df8a",
    "hypoxic_stress": "#ff7f0e",
    "angiogenic_vascular": "#17becf",
    "interferon_inflamed": "#e377c2",
    "other": "#7f7f7f",
}

FUNCTION_COLORS = {
    "hypoxic_stress": "#ff7f0e",
    "angiogenic_vascular": "#17becf",
    "interferon_inflamed": "#e377c2",
    "tumor_proliferative": "#ff9896",
    "ecm_remodeling": "#98df8a",
    "t_cell": "#9467bd",
    "myeloid_macrophage": "#8c564b",
    "immune_b_plasma": "#1f77b4",
    "other": "#7f7f7f",
}

METABOLIC_COLORS = {
    "glycolysis": "#d62728",
    "oxphos": "#1f77b4",
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
    """Parse args inputs for the command line workflow."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-interactive", action="store_true")
    return parser.parse_args()


def get_output_root(config_path):
    """Return output root required by later visualization logic."""

    cfg = validate_config(load_config(config_path))
    return Path(cfg["output_root"])


def ensure_dir(path):
    """Helper for ensure dir within the Step 12 overlay workflow."""

    Path(path).mkdir(parents=True, exist_ok=True)


# =========================
# Input discovery helpers
# =========================

def discover_samples(input_root, sample=None, max_samples=None):
    """Helper for discover samples within the Step 12 overlay workflow."""

    h5ad_root = input_root / "per_sample_h5ad"
    samples = []

    if h5ad_root.exists():
        for p in sorted(h5ad_root.glob("*_with_multi_axis_transcriptome_labels.h5ad")):
            samples.append(p.name.replace("_with_multi_axis_transcriptome_labels.h5ad", ""))

    if sample is not None:
        samples = [s for s in samples if s == sample]

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def find_h5ad(input_root, output_root, sample_id):
    """Helper for find h5ad within the Step 12 overlay workflow."""

    candidates = [
        input_root / "per_sample_h5ad" / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad",
        output_root / "output_02_01_process_samples_data" / sample_id / "adata" / "02_processed.h5ad",
        output_root / "output_02_01_process_samples_data" / sample_id / "adata" / "01_loaded.h5ad",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


# =========================
# Per spot table loading helpers
# =========================

def read_optional_csv(path):
    """Load optional csv data used by the overlay workflow."""

    if path.exists():
        # CSV inputs are pipeline products, not external relabeling sources.
        return pd.read_csv(path)
    return pd.DataFrame()


def choose_spot_col(df):
    """Helper for choose spot col within the Step 12 overlay workflow."""

    for c in ["spot_id", "barcode", "obs_name"]:
        if c in df.columns:
            return c
    return None


def attach_spot_tables(adata, input_root, sample_id):
    """Helper for attach spot tables within the Step 12 overlay workflow."""

    adata = adata.copy()

    label_path = input_root / "per_sample" / f"{sample_id}_spot_labels.csv"
    score_path = input_root / "per_sample" / f"{sample_id}_spot_scores.csv"

    for table in [read_optional_csv(label_path), read_optional_csv(score_path)]:
        if table.empty:
            continue

        spot_col = choose_spot_col(table)

        if spot_col is None:
            if len(table) == adata.n_obs:
                table.index = adata.obs_names
            else:
                continue
        else:
            table[spot_col] = table[spot_col].astype(str)
            table = table.drop_duplicates(subset=[spot_col])
            table = table.set_index(spot_col)

        table = table.reindex(adata.obs_names.astype(str))

        for col in table.columns:
            if col in ["spot_id", "barcode", "obs_name"]:
                continue
            if col not in adata.obs.columns:
                adata.obs[col] = table[col].values

    return adata


# =========================
# Image and spatial coordinate helpers
# =========================

def choose_library_id(adata):
    """Helper for choose library id within the Step 12 overlay workflow."""

    if "spatial" not in adata.uns:
        raise ValueError("adata.uns['spatial'] is missing")
    keys = list(adata.uns["spatial"].keys())
    if len(keys) == 0:
        raise ValueError("adata.uns['spatial'] has no library ids")
    return keys[0]


def choose_image_and_scale(adata):
    """Helper for choose image and scale within the Step 12 overlay workflow."""

    lib = choose_library_id(adata)
    entry = adata.uns["spatial"][lib]
    images = entry.get("images", {})
    scalefactors = entry.get("scalefactors", {})

    if "hires" in images:
        return images["hires"], float(scalefactors.get("tissue_hires_scalef", 1.0)), "hires", lib

    if "lowres" in images:
        return images["lowres"], float(scalefactors.get("tissue_lowres_scalef", 1.0)), "lowres", lib

    raise ValueError("No hires or lowres image found")


def to_rgb_uint8(img):
    """Helper for to rgb uint8 within the Step 12 overlay workflow."""

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def valid_tissue_mask(adata):
    """Helper for valid tissue mask within the Step 12 overlay workflow."""

    mask = np.ones(adata.n_obs, dtype=bool)

    if "in_tissue" in adata.obs.columns:
        vals = pd.to_numeric(adata.obs["in_tissue"], errors="coerce").fillna(1).values
        mask &= vals > 0

    if "total_counts" in adata.obs.columns:
        vals = pd.to_numeric(adata.obs["total_counts"], errors="coerce").fillna(0).values
        mask &= vals > 0

    return mask


def build_plot_df(adata):
    """Compute build plot df values used by Step 12 visualization products."""

    img_array, scale, image_name, library_id = choose_image_and_scale(adata)
    img = Image.fromarray(to_rgb_uint8(img_array))
    img_w, img_h = img.size

    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    df = pd.DataFrame({
        "spot_id": adata.obs_names.astype(str),
        "x_fullres": coords[:, 0],
        "y_fullres": coords[:, 1],
    })

    df["x"] = df["x_fullres"] * scale
    df["y"] = df["y_fullres"] * scale
    df["valid_tissue"] = valid_tissue_mask(adata)

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)

    for col in obs.columns:
        if col not in df.columns:
            df[col] = obs[col].values

    df = df[df["valid_tissue"]].copy()

    return df, img, img_w, img_h, image_name, library_id


# =========================
# Column resolution helpers
# =========================

def first_existing_col(df, candidates):
    """Helper for first existing col within the Step 12 overlay workflow."""

    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================
# Column resolution helper
# =========================

def resolve_with_ma_prefix(df, candidates):
    """
    Try original column names first.
    If none found, try adding 'ma_' variants automatically.
    """
    # 1. original names
    for c in candidates:
        if c in df.columns:
            return c

    # 2. try ma_ versions
    for c in candidates:
        ma_variants = [
            f"structure_score__{c}",
            f"function_score__{c}",
            f"metabolic_score__{c}",
            f"simple_mean__{c}",
            f"rank_percentile__{c}",
            f"ucell__{c}",
            f"custom_gsva__{c}",
            f"hallmark_gsva__{c}",
            f"reactome_gsva__{c}",
            f"ma_score_structure_score__{c}",
            f"ma_score_function_score__{c}",
            f"ma_score_metabolic_score__{c}",
            f"ma_score_simple__{c}",
            f"ma_score_rank__{c}",
            f"ma_score_spatial_smooth__{c}",
            f"ma_score_cluster_smooth__{c}",
            f"ma_score_local_z__{c}",
        ]

        for m in ma_variants:
            if m in df.columns:
                return m

    return None


# =========================
# Numeric score helpers
# =========================

def numeric_series(df, candidates):
    """Helper for numeric series within the Step 12 overlay workflow."""

    col = resolve_with_ma_prefix(df, candidates)

    if col is None:
        return None, None

    values = pd.to_numeric(df[col], errors="coerce")

    if values.notna().sum() == 0:
        return None, None

    return values, col


def robust_high_mask(values, quantile=0.75, z_min=None):
    """Helper for robust high mask within the Step 12 overlay workflow."""

    values = pd.to_numeric(values, errors="coerce")
    vals = values.dropna()

    if len(vals) == 0:
        return np.zeros(len(values), dtype=bool), np.nan

    cutoff = float(vals.quantile(quantile))

    if z_min is not None:
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if np.isfinite(sd) and sd > 0:
            cutoff = max(cutoff, mu + z_min * sd)

    mask = values.fillna(-np.inf).values >= cutoff
    return mask, cutoff


# =========================
# Structure label helpers
# =========================

def assign_exclusive_structure(df):
    """Helper for assign exclusive structure within the Step 12 overlay workflow."""

    score_cols = {}
    for label, candidates in STRUCTURE_SCORE_BASES.items():
        values, col = numeric_series(df, candidates)
        if values is not None:
            score_cols[label] = values.values

    if not score_cols:
        return np.array(["other"] * len(df), dtype=object), {}

    mat = pd.DataFrame(score_cols, index=df.index)
    labels = choose_allowed_runner_up(
        mat,
        banned_terms=["overlap"],
        max_fraction=0.80,
    )

    return labels, score_cols

def choose_allowed_runner_up(score_df, banned_terms=None, max_fraction=0.80):
    """Helper for choose allowed runner up within the Step 12 overlay workflow."""

    if banned_terms is None:
        banned_terms = ["overlap"]

    labels = []

    for _, row in score_df.iterrows():
        ranked = row.sort_values(ascending=False)

        chosen = None
        for label in ranked.index:
            low = str(label).lower()
            if any(term in low for term in banned_terms):
                continue
            chosen = label
            break

        labels.append(chosen if chosen is not None else "other")

    labels = pd.Series(labels, index=score_df.index).astype(str)

    changed = True
    while changed:
        changed = False
        freqs = labels.value_counts(normalize=True)

        too_broad = [
            label for label, frac in freqs.items()
            if label != "other" and frac > max_fraction
        ]

        if not too_broad:
            break

        for broad_label in too_broad:
            idx = labels[labels.eq(broad_label)].index

            for i in idx:
                ranked = score_df.loc[i].sort_values(ascending=False)

                replacement = "other"
                for label in ranked.index:
                    low = str(label).lower()
                    if label == broad_label:
                        continue
                    if any(term in low for term in banned_terms):
                        continue
                    replacement = label
                    break

                labels.loc[i] = replacement

            changed = True

    return labels.values

# =========================
# Spatial geometry helpers
# =========================

def nearest_neighbor_distance(xy):
    """Helper for nearest neighbor distance within the Step 12 overlay workflow."""

    if len(xy) < 2:
        return 1.0

    sample = xy
    if len(sample) > 1200:
        rng = np.random.default_rng(7)
        sample = sample[rng.choice(len(sample), size=1200, replace=False)]

    dists = []
    # Iterate samples independently so failures can be logged without losing the full cohort.
    for i in range(len(sample)):
        diff = sample - sample[i]
        dist2 = np.sum(diff * diff, axis=1)
        dist2[i] = np.inf
        dists.append(np.sqrt(np.min(dist2)))

    return float(np.nanmedian(dists))


def connected_components_local(points_df, threshold):
    """Helper for connected components local within the Step 12 overlay workflow."""

    if points_df.empty:
        return []

    xy = points_df[["x", "y"]].to_numpy(dtype=float)
    n = len(xy)
    visited = np.zeros(n, dtype=bool)
    comps = []

    for start in range(n):
        if visited[start]:
            continue

        q = deque([start])
        visited[start] = True
        idxs = []

        while q:
            i = q.popleft()
            idxs.append(i)

            diff = xy - xy[i]
            dist2 = np.sum(diff * diff, axis=1)
            neighbors = np.where((dist2 <= threshold * threshold) & (~visited))[0]

            for j in neighbors:
                visited[j] = True
                q.append(j)

        comps.append(points_df.iloc[idxs].copy())

    return comps


def convex_hull(points):
    """Helper for convex hull within the Step 12 overlay workflow."""

    pts = np.unique(np.asarray(points, dtype=float), axis=0)

    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        """Helper for cross within the Step 12 overlay workflow."""

        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

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

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)

def auto_min_component_size(n_spots):
    """Helper for auto min component size within the Step 12 overlay workflow."""

    if n_spots < 1500:
        return 4
    elif n_spots < 4000:
        return 6
    elif n_spots < 8000:
        return 8
    else:
        return 10

def adaptive_conf_threshold(conf_series):
    """Helper for adaptive conf threshold within the Step 12 overlay workflow."""

    return conf_series.quantile(0.65)

def expand_polygon(poly, scale=1.10):
    """Helper for expand polygon within the Step 12 overlay workflow."""

    if len(poly) < 3:
        return poly

    center = poly.mean(axis=0)
    return center + (poly - center) * scale


def close_polygon(poly):
    """Helper for close polygon within the Step 12 overlay workflow."""

    if len(poly) == 0:
        return poly
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[0]])


# =========================
# Color and hover helpers
# =========================

def rgba_from_hex(hex_color, alpha):
    """Helper for rgba from hex within the Step 12 overlay workflow."""

    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def image_to_data_uri(img):
    """Helper for image to data uri within the Step 12 overlay workflow."""

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    enc = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + enc


def region_hover_text(kind, label, n_spots, source_col=None, cutoff=None):
    """Helper for region hover text within the Step 12 overlay workflow."""

    txt = f"{kind}: {label}<br>spots_in_region: {n_spots}"
    if source_col is not None:
        txt += f"<br>source_column: {source_col}"
    if cutoff is not None and np.isfinite(cutoff):
        txt += f"<br>cutoff: {cutoff:.3f}"
    return txt


# =========================
# Region layer construction
# =========================

def build_region_layers(df):
    """Compute build region layers values used by Step 12 visualization products."""

    df = df.copy()
    structure_consensus_col = first_existing_col(
        df,
        [
            "structure_region_label_smoothed",
            "structure_region_label",
            "structure_dominant_label_raw",
            "structure_dominant_label",
        ],
    )

    if structure_consensus_col is not None:
        exclusive_labels = df[structure_consensus_col].astype(str).fillna("other").values
        score_cols = {}
        using_structure_consensus = True
    else:
        exclusive_labels, score_cols = assign_exclusive_structure(df)
        using_structure_consensus = False

    df["exclusive_structure_label"] = exclusive_labels

    layers = []

    for label in sorted(pd.Series(exclusive_labels).dropna().unique()):
        if label == "other":
            continue
        mask = df["exclusive_structure_label"].astype(str).eq(label).values

        top1_col = first_existing_col(df, ["structure_top1_score", "ma_structure_top1_score"])
        top2_col = first_existing_col(df, ["structure_top2_score", "ma_structure_top2_score"])

        if (not using_structure_consensus) and top1_col is not None and top2_col is not None:
            top1 = pd.to_numeric(df[top1_col], errors="coerce").fillna(0)
            top2 = pd.to_numeric(df[top2_col], errors="coerce").fillna(0)
            mask = mask & top1.ge(0.55).values & (top1 - top2).ge(0.06).values
            mask = mask & (top1 > np.percentile(top1, 25))
        layers.append({
            "group": "structure",
            "kind": "Structure",
            "label": label,
            "mask": mask,
            "color": STRUCTURE_COLORS.get(label, STRUCTURE_COLORS["other"]),
            "fill": True,
            "line_dash": "solid",
            "line_width": EXCLUSIVE_LINE_WIDTH,
            "min_size": MIN_EXCLUSIVE_STRUCTURE_COMPONENT_SIZE,
            "expand": 1.02,
            "source_col": structure_consensus_col if using_structure_consensus else "exclusive argmax",
            "cutoff": np.nan,
        })

#    for label, candidates in OVERLAP_SCORE_BASES.items():
 #       values, col = numeric_series(df, candidates)
  #      if values is None:
   #         continue
    #    mask, cutoff = robust_high_mask(values, quantile=0.85)
      #  layers.append({
     #       "group": "structure",
      #      "kind": "Structure overlap",
      #      "label": label,
      #      "mask": mask,
      #      "color": OVERLAP_COLORS.get(label, OVERLAP_COLORS["other"]),
       #     "fill": False,
        #    "line_dash": "solid",
         #   "line_width": OVERLAP_LINE_WIDTH,
          #  "min_size": MIN_OVERLAP_STRUCTURE_COMPONENT_SIZE,
           # "expand": 1.16,
            # "cutoff": cutoff,
       # })

    function_scores = {}

    for label, candidates in FUNCTION_SCORE_BASES.items():
        values, col = numeric_series(df, candidates)
        if values is not None:
            function_scores[label] = values.values

    if function_scores:
        fmat = pd.DataFrame(function_scores, index=df.index)

        f_top1 = pd.Series(
            choose_allowed_runner_up(
                fmat,
                banned_terms=["overlap"],
                max_fraction=0.80,
            ),
            index=fmat.index,
        )

        f_top1_score = fmat.max(axis=1)
        f_top2_score = fmat.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
        f_margin = f_top1_score - f_top2_score

        df["function_top1_label"] = f_top1
        df["function_top1_score"] = f_top1_score
        df["function_top2_score"] = f_top2_score
        df["function_margin"] = f_margin

        for label in sorted(fmat.columns):
            mask = (
                df["function_top1_label"].eq(label).values
                & df["function_top1_score"].ge(df["function_top1_score"].quantile(0.55)).values
                & df["function_margin"].ge(df["function_margin"].quantile(0.25)).values
            )

            layers.append({
                "group": "function",
                "kind": "Function",
                "label": label,
                "mask": mask,
                "color": FUNCTION_COLORS.get(label, FUNCTION_COLORS["other"]),
                "fill": False,
                "line_dash": "solid",
                "line_width": FUNCTION_LINE_WIDTH,
                "min_size": 4,
                "expand": 1.08,
                "source_col": "competitive_function_top1",
                "cutoff": np.nan,
            })

    for label, candidates in METABOLIC_SCORE_BASES.items():
        active_col = first_existing_col(
            df,
            [
                f"metabolic__{label}_active",
                f"ma_metabolic__{label}_active",
            ],
        )

        conf_col = first_existing_col(
            df,
            [
                f"metabolic__{label}_confidence",
                f"ma_metabolic__{label}_confidence",
            ],
        )

        if active_col is not None:
            mask = pd.to_numeric(df[active_col], errors="coerce").fillna(0).gt(0).values
            cutoff = np.nan
            col = active_col

            if conf_col is not None:
                conf = pd.to_numeric(df[conf_col], errors="coerce").fillna(0)
                threshold = adaptive_conf_threshold(conf)
                mask = mask & conf.ge(threshold).values
        else:
            values, col = numeric_series(df, candidates)
            if values is None:
                continue
            mask, cutoff = robust_high_mask(values, quantile=0.75)
        layers.append({
            "group": "metabolism",
            "kind": "Metabolic",
            "label": label,
            "mask": mask,
            "color": METABOLIC_COLORS.get(label, METABOLIC_COLORS["other"]),
            "fill": False,
            "line_dash": "dot",
            "line_width": METAB_LINE_WIDTH,
            "min_size": MIN_METAB_COMPONENT_SIZE,
            "expand": 1.20,
            "source_col": col,
            "cutoff": cutoff,
        })

    return df, layers


# =========================
# Connected component and polygon construction
# =========================

def build_components(df, layers, neighbor_threshold):
    """Compute build components values used by Step 12 visualization products."""

    traces = []

    for layer in layers:
        sub = df[np.asarray(layer["mask"])].copy()

        if sub.empty:
            continue

        comps = connected_components_local(sub, neighbor_threshold)
        first = True

        for comp in comps:
            if len(comp) < layer["min_size"]:
                continue

            poly = convex_hull(comp[["x", "y"]].to_numpy())
            if len(poly) < 3:
                continue

            poly = expand_polygon(poly, scale=layer["expand"])
            poly = close_polygon(poly)

            traces.append({
                **layer,
                "x": poly[:, 0],
                "y": poly[:, 1],
                "n_spots": int(len(comp)),
                "showlegend": first,
            })

            first = False

    return traces


# =========================
# Interactive Plotly helpers
# =========================

def add_background_plotly(fig, img, img_w, img_h):
    """Helper for add background plotly within the Step 12 overlay workflow."""

    fig.add_layout_image(
        dict(
            source=image_to_data_uri(img),
            x=0,
            y=0,
            sizex=img_w,
            sizey=img_h,
            xref="x",
            yref="y",
            layer="below",
            sizing="stretch",
        )
    )


def add_faint_spots_plotly(fig, df):
    """Helper for add faint spots plotly within the Step 12 overlay workflow."""

    if not SHOW_FAINT_SPOTS:
        return 0

    fig.add_trace(
        go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode="markers",
            name="All tissue spots",
            marker=dict(
                size=FAINT_SPOT_SIZE,
                color="rgba(80,80,80,0.35)",
                opacity=FAINT_SPOT_ALPHA,
            ),
            hoverinfo="skip",
            showlegend=False,
            legendgroup="spots",
        )
    )

    return 1


def add_trace_plotly(fig, tr, visible=True):
    """Helper for add trace plotly within the Step 12 overlay workflow."""

    fill = "toself" if tr["fill"] else None
    fillcolor = rgba_from_hex(tr["color"], EXCLUSIVE_FILL_ALPHA) if tr["fill"] else None

    fig.add_trace(
        go.Scatter(
            x=tr["x"],
            y=tr["y"],
            mode="lines",
            name=f"{tr['kind']}: {tr['label']}",
            legendgroup=tr["group"],
            visible=visible,
            showlegend=tr["showlegend"],
            line=dict(
                color=tr["color"],
                width=tr["line_width"],
                dash=tr["line_dash"],
            ),
            fill=fill,
            fillcolor=fillcolor,
            hovertemplate=region_hover_text(
                tr["kind"],
                tr["label"],
                tr["n_spots"],
                tr["source_col"],
                tr["cutoff"],
            ) + "<extra></extra>",
        )
    )


def visibility_by_group(fig, group_name):
    """Helper for visibility by group within the Step 12 overlay workflow."""

    out = []

    for tr in fig.data:
        if getattr(tr, "legendgroup", None) == "spots":
            out.append(True)
        else:
            out.append(getattr(tr, "legendgroup", None) == group_name)

    return out


def visibility_all(fig):
    """Helper for visibility all within the Step 12 overlay workflow."""

    return [True] * len(fig.data)


def make_plotly_figure(df, traces, img, img_w, img_h, title, visible_group=None):
    """Create make plotly figure visual output while preserving pipeline labels."""

    fig = go.Figure()
    add_background_plotly(fig, img, img_w, img_h)
    add_faint_spots_plotly(fig, df)

    for tr in traces:
        visible = True if visible_group is None else tr["group"] == visible_group
        add_trace_plotly(fig, tr, visible=visible)

    fig.update_layout(
        title=title,
        width=1400,
        height=1100,
        template="plotly_white",
        margin=dict(l=20, r=280, t=75, b=20),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.82)",
            font=dict(size=10),
        ),
    )

    fig.update_xaxes(visible=False, range=[0, img_w])
    fig.update_yaxes(visible=False, range=[img_h, 0], scaleanchor="x", scaleratio=1)

    return fig


def add_buttons(fig):
    """Helper for add buttons within the Step 12 overlay workflow."""

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                buttons=[
                    dict(label="Fine structural", method="update", args=[{"visible": visibility_by_group(fig, "structure")}]),
                    dict(label="Functional", method="update", args=[{"visible": visibility_by_group(fig, "function")}]),
                    dict(label="Metabolic", method="update", args=[{"visible": visibility_by_group(fig, "metabolism")}]),
                    dict(label="Show all", method="update", args=[{"visible": visibility_all(fig)}]),
                ],
            )
        ]
    )


# =========================
# Static PNG and HTML index helpers
# =========================

def save_static_png(df, traces, img, img_w, img_h, title, out_path, group=None):
    """Write static png output for downstream review."""

    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=STATIC_FIGSIZE)

    ax.imshow(img)

    if SHOW_FAINT_SPOTS:
        ax.scatter(
            df["x"],
            df["y"],
            s=FAINT_SPOT_SIZE,
            c="black",
            alpha=FAINT_SPOT_ALPHA,
            linewidths=0,
        )

    handles = []
    labels = []
    seen = set()

    for tr in traces:
        if group is not None and tr["group"] != group:
            continue

        if tr["fill"]:
            ax.fill(
                tr["x"],
                tr["y"],
                color=tr["color"],
                alpha=EXCLUSIVE_FILL_ALPHA,
                linewidth=0,
            )

        line, = ax.plot(
            tr["x"],
            tr["y"],
            color=tr["color"],
            linewidth=tr["line_width"],
            linestyle=":" if tr["line_dash"] == "dot" else "-",
            alpha=0.98,
        )

        legend_label = f"{tr['kind']}: {tr['label']}"

        if legend_label not in seen:
            handles.append(line)
            labels.append(legend_label)
            seen.add(legend_label)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=False,
        )

    plt.tight_layout()
    # Persist review artifact for downstream inspection.
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_index(sample_dir, sample_id):
    """Write index output for downstream review."""

    html = f"""
<html>
<head>
<title>{sample_id} Visium region overlays</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
}}
h1 {{
    color: #8a5570;
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 26px;
}}
img {{
    max-width: 100%;
    border: 1px solid #ddd;
}}
a {{
    font-size: 18px;
}}
</style>
</head>
<body>
<h1>{sample_id} Visium region overlays</h1>
<p><a href="interactive_visium_region_overlay.html">interactive_visium_region_overlay.html</a></p>
<div class="grid">
<div><h3>Combined</h3><img src="combined_region_overlay.png"></div>
<div><h3>Fine structural</h3><img src="fine_structure_region_overlay.png"></div>
<div><h3>Functional</h3><img src="functional_region_overlay.png"></div>
<div><h3>Metabolic</h3><img src="metabolic_region_overlay.png"></div>
</div>
</body>
</html>
"""
    # Persist review artifact for downstream inspection.
    (sample_dir / "index.html").write_text(html, encoding="utf-8")


# =========================
# Per sample overlay processing
# =========================

def process_one_sample(sample_id, input_root, output_root, overlay_root, overwrite, skip_interactive):
    """Helper for process one sample within the Step 12 overlay workflow."""

    sample_dir = overlay_root / sample_id
    ensure_dir(sample_dir)

    done = sample_dir / "_overlay_done.txt"

    if done.exists() and not overwrite:
        return {"sample_id": sample_id, "status": "skipped_existing", "error": ""}

    h5ad_path = find_h5ad(input_root, output_root, sample_id)

    if h5ad_path is None:
        raise FileNotFoundError(f"No h5ad found for {sample_id}")

    adata = sc.read_h5ad(h5ad_path)

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] missing")

    if "spatial" not in adata.uns:
        raise ValueError("adata.uns['spatial'] missing")

    adata = attach_spot_tables(adata, input_root, sample_id)

    df, img, img_w, img_h, image_name, library_id = build_plot_df(adata)

    if df.empty:
        raise ValueError("No valid tissue spots after filtering")

    df, layers = build_region_layers(df)

    spot_spacing = nearest_neighbor_distance(df[["x", "y"]].to_numpy(dtype=float))
    # adaptive multiplier based on density
    if spot_spacing < 50:
        multiplier = 1.8
    elif spot_spacing < 100:
        multiplier = 1.6
    else:
        multiplier = 1.4

    neighbor_threshold = spot_spacing * multiplier
    min_component_size = auto_min_component_size(len(df))
    for layer in layers:
        if layer["group"] in ["function", "metabolism"]:
            layer["min_size"] = min_component_size

    traces = build_components(df, layers, neighbor_threshold)

    total_spots = len(df)

    filtered_traces = []
    for tr in traces:
        frac = tr["n_spots"] / total_spots

        if frac > 0.80:
            continue

        filtered_traces.append(tr)

    traces = filtered_traces

    if len(traces) > 200:
        neighbor_threshold *= 1.15
        traces = build_components(df, layers, neighbor_threshold)
    elif len(traces) < 20:
        neighbor_threshold *= 0.85
        traces = build_components(df, layers, neighbor_threshold)

    if len(traces) == 0:
        raise ValueError("No region components were generated")

    save_static_png(
        df,
        traces,
        img,
        img_w,
        img_h,
        f"{sample_id} combined regions on Visium histology",
        sample_dir / "combined_region_overlay.png",
        group=None,
    )

    save_static_png(
        df,
        traces,
        img,
        img_w,
        img_h,
        f"{sample_id} fine structural regions on Visium histology",
        sample_dir / "fine_structure_region_overlay.png",
        group="structure",
    )

    save_static_png(
        df,
        traces,
        img,
        img_w,
        img_h,
        f"{sample_id} functional regions on Visium histology",
        sample_dir / "functional_region_overlay.png",
        group="function",
    )

    save_static_png(
        df,
        traces,
        img,
        img_w,
        img_h,
        f"{sample_id} metabolic regions on Visium histology",
        sample_dir / "metabolic_region_overlay.png",
        group="metabolism",
    )

    if HAS_PLOTLY and not skip_interactive:
        fig_all = make_plotly_figure(
            df,
            traces,
            img,
            img_w,
            img_h,
            f"{sample_id} Visium region overlay | {library_id} | image={image_name}",
            visible_group=None,
        )
        add_buttons(fig_all)
        fig_all.write_html(str(sample_dir / "interactive_visium_region_overlay.html"))
        fig_all.write_html(str(sample_dir / "combined_region_overlay.html"))

        fig_structure = make_plotly_figure(
            df,
            traces,
            img,
            img_w,
            img_h,
            f"{sample_id} fine structural regions on Visium histology",
            visible_group="structure",
        )
        fig_structure.write_html(str(sample_dir / "fine_structure_region_overlay.html"))

        fig_function = make_plotly_figure(
            df,
            traces,
            img,
            img_w,
            img_h,
            f"{sample_id} functional regions on Visium histology",
            visible_group="function",
        )
        fig_function.write_html(str(sample_dir / "functional_region_overlay.html"))

        fig_metabolic = make_plotly_figure(
            df,
            traces,
            img,
            img_w,
            img_h,
            f"{sample_id} metabolic regions on Visium histology",
            visible_group="metabolism",
        )
        fig_metabolic.write_html(str(sample_dir / "metabolic_region_overlay.html"))

    write_index(sample_dir, sample_id)
    # Persist review artifact for downstream inspection.
    done.write_text("done", encoding="utf-8")

    rows = []
    for tr in traces:
        rows.append({
            "sample_id": sample_id,
            "group": tr["group"],
            "kind": tr["kind"],
            "label": tr["label"],
            "n_spots": tr["n_spots"],
            "source_col": tr["source_col"],
            "cutoff": tr["cutoff"],
        })

    # Persist review artifact for downstream inspection.
    pd.DataFrame(rows).to_csv(sample_dir / "region_overlay_manifest.csv", index=False)

    return {
        "sample_id": sample_id,
        "status": "ok",
        "h5ad_path": str(h5ad_path),
        "n_tissue_spots": int(len(df)),
        "n_layers": int(len(layers)),
        "n_region_components": int(len(traces)),
        "spot_spacing": float(spot_spacing),
        "neighbor_threshold": float(neighbor_threshold),
        "error": "",
    }


# Main orchestration keeps sample selection, loading, rendering, and status writing together.
def main():
    """Run the Step 12 overlay workflow from parsed command line arguments."""

    args = parse_args()

    output_root = get_output_root(args.config)
    input_root = output_root / INPUT_SUBDIR
    overlay_root = output_root / OUT_SUBDIR

    ensure_dir(overlay_root)

    samples = discover_samples(
        input_root=input_root,
        sample=args.sample,
        max_samples=args.max_samples,
    )

    if not samples:
        raise ValueError("No samples found")

    print("=== 12 Visium region overlays ===")
    print("Output root:", output_root)
    print("Input root:", input_root)
    print("Overlay root:", overlay_root)
    print("Samples:", len(samples))
    print("Plotly available:", HAS_PLOTLY)
    print()

    rows = []

    # Iterate samples independently so failures can be logged without losing the full cohort.
    for sample_id in samples:
        print("Generating:", sample_id)

        # Optional dependency block; script should still run with reduced output when missing.
        try:
            row = process_one_sample(
                sample_id=sample_id,
                input_root=input_root,
                output_root=output_root,
                overlay_root=overlay_root,
                overwrite=args.overwrite,
                skip_interactive=args.skip_interactive,
            )
            print("  ok")

        except Exception as e:
            row = {
                "sample_id": sample_id,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
            print("  failed:", row["error"])

        rows.append(row)

    status = pd.DataFrame(rows)
    # Persist review artifact for downstream inspection.
    status.to_csv(overlay_root / "overlay_generation_status.csv", index=False)

    print()
    print("DONE")
    print("Status:", overlay_root / "overlay_generation_status.csv")


# Command line entry point.
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


