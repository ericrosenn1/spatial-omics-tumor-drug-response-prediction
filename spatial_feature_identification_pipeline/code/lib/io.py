"""
io.py

Purpose:
Provide reusable input and output utilities for loading 10x Visium-style
spatial transcriptomics samples into AnnData objects.

This module supports:
    1. 10x H5 input files
    2. MTX folders with matrix.mtx, barcodes.tsv, and features.tsv
    3. Visium spatial coordinates
    4. Scanpy-compatible adata.uns["spatial"] metadata
    5. metadata.json reading
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
from anndata import AnnData

try:
    import matplotlib.image as mpimg
except Exception:
    mpimg = None


# =========================
# File discovery
# =========================

def find_h5(raw_dir):
    """Return the best H5 expression file from a raw directory."""
    raw_dir = Path(raw_dir)
    h5s = sorted(raw_dir.glob("*.h5"))

    if not h5s:
        return None

    filtered = [p for p in h5s if "filtered_feature_bc_matrix" in p.name]

    if filtered:
        return filtered[0]

    raw_feature = [p for p in h5s if "raw_feature_bc_matrix" in p.name]

    if raw_feature:
        return raw_feature[0]

    return h5s[0]


def find_mtx_files(raw_dir):
    """Return matrix, barcode, and feature files from a raw directory."""
    raw_dir = Path(raw_dir)

    matrix = sorted(raw_dir.glob("*.mtx"))
    barcodes = sorted(raw_dir.glob("*barcodes.tsv"))
    features = sorted(raw_dir.glob("*features.tsv"))

    if matrix and barcodes and features:
        return matrix[0], barcodes[0], features[0]

    return None, None, None


def get_raw_dir(sample_dir):
    """Return raw input directory, with fallback to sample directory."""
    sample_dir = Path(sample_dir)
    raw_dir = sample_dir / "raw"

    if raw_dir.exists():
        return raw_dir

    return sample_dir


def get_spatial_dir(sample_dir):
    """Return spatial directory, with fallback to sample directory."""
    sample_dir = Path(sample_dir)
    spatial_dir = sample_dir / "spatial"

    if spatial_dir.exists():
        return spatial_dir

    return sample_dir


def detect_expression_format(sample_dir):
    """Detect whether a sample uses h5, mtx, or unknown expression format."""
    sample_dir = Path(sample_dir)
    raw_dir = get_raw_dir(sample_dir)

    h5_path = find_h5(raw_dir)
    matrix, barcodes, features = find_mtx_files(raw_dir)

    if h5_path is not None:
        return "h5"

    if matrix is not None and barcodes is not None and features is not None:
        return "mtx"

    return "unknown"


# =========================
# Metadata and spatial checks
# =========================

def read_metadata(sample_dir):
    """Read sample metadata.json if present."""
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "metadata.json"

    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_spatial_files(sample_dir):
    """Check whether common Visium spatial files are present."""
    spatial_dir = get_spatial_dir(sample_dir)

    if not spatial_dir.exists():
        return {
            "has_positions": False,
            "has_scalefactors": False,
            "has_hires_image": False,
            "has_lowres_image": False,
            "has_image": False,
        }

    files = list(spatial_dir.glob("*"))

    has_positions = any("position" in p.name.lower() for p in files)
    has_scalefactors = any("scalefactors" in p.name.lower() for p in files)
    has_hires_image = any("tissue_hires_image" in p.name.lower() for p in files)
    has_lowres_image = any("tissue_lowres_image" in p.name.lower() for p in files)

    return {
        "has_positions": has_positions,
        "has_scalefactors": has_scalefactors,
        "has_hires_image": has_hires_image,
        "has_lowres_image": has_lowres_image,
        "has_image": has_hires_image or has_lowres_image,
    }


def find_positions_file(sample_dir):
    """Return the Visium tissue positions CSV file if present."""
    spatial_dir = get_spatial_dir(sample_dir)

    candidates = sorted(
        p for p in spatial_dir.glob("*.csv")
        if "position" in p.name.lower()
    )

    if candidates:
        preferred = [
            p for p in candidates
            if "tissue_positions" in p.name.lower()
        ]

        if preferred:
            return preferred[0]

        return candidates[0]

    return None


def find_scalefactors_file(sample_dir):
    """Return scalefactors_json.json if present."""
    spatial_dir = get_spatial_dir(sample_dir)

    candidates = sorted(
        p for p in spatial_dir.glob("*.json")
        if "scalefactors" in p.name.lower()
    )

    if candidates:
        return candidates[0]

    return None


def find_image_file(sample_dir, image_type):
    """Return hires or lowres tissue image if present."""
    spatial_dir = get_spatial_dir(sample_dir)
    image_type = str(image_type).lower()

    candidates = []

    for suffix in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        candidates.extend(spatial_dir.glob(suffix))

    candidates = sorted(candidates)

    if image_type == "hires":
        matches = [
            p for p in candidates
            if "tissue_hires_image" in p.name.lower()
        ]
    else:
        matches = [
            p for p in candidates
            if "tissue_lowres_image" in p.name.lower()
        ]

    if matches:
        return matches[0]

    return None


def read_positions_table(sample_dir):
    """Read old or new 10x Visium tissue position files."""
    positions_file = find_positions_file(sample_dir)

    if positions_file is None:
        return None

    if "tissue_positions_list" in positions_file.name:
        positions = pd.read_csv(positions_file, header=None)
        positions.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        return positions

    positions = pd.read_csv(positions_file)

    rename_map = {
        "barcode": "barcode",
        "in_tissue": "in_tissue",
        "array_row": "array_row",
        "array_col": "array_col",
        "pxl_row_in_fullres": "pxl_row_in_fullres",
        "pxl_col_in_fullres": "pxl_col_in_fullres",
    }

    positions = positions.rename(columns=rename_map)

    return positions


def read_scalefactors(sample_dir):
    """Read Visium scalefactors JSON if present."""
    path = find_scalefactors_file(sample_dir)

    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_image_or_none(path):
    """Read image as numpy array if possible."""
    if path is None:
        return None

    if mpimg is None:
        return None

    try:
        return mpimg.imread(str(path))
    except Exception:
        return None


def add_spatial_coordinates(adata, sample_dir):
    """Add Visium spatial coordinates to adata.obs and adata.obsm."""
    positions = read_positions_table(sample_dir)

    if positions is None:
        adata.uns["spatial_coordinates_loaded"] = False
        adata.uns["spatial_coordinates_error"] = "No positions CSV found"
        return adata

    required = ["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    missing = [col for col in required if col not in positions.columns]

    if missing:
        adata.uns["spatial_coordinates_loaded"] = False
        adata.uns["spatial_coordinates_error"] = f"Missing columns: {missing}"
        return adata

    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.set_index("barcode")

    adata.obs_names = adata.obs_names.astype(str)

    shared = adata.obs_names.intersection(positions.index)

    if len(shared) == 0:
        adata.uns["spatial_coordinates_loaded"] = False
        adata.uns["spatial_coordinates_error"] = "No matching barcodes"
        return adata

    aligned = positions.reindex(adata.obs_names)

    for col in [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]:
        if col in aligned.columns:
            adata.obs[col] = aligned[col].values

    adata.obs["spatial_x"] = pd.to_numeric(
        adata.obs["pxl_col_in_fullres"],
        errors="coerce",
    ).astype(float)

    adata.obs["spatial_y"] = pd.to_numeric(
        adata.obs["pxl_row_in_fullres"],
        errors="coerce",
    ).astype(float)

    adata.obsm["spatial"] = adata.obs[
        ["spatial_x", "spatial_y"]
    ].to_numpy()

    adata.uns["spatial_coordinates_loaded"] = True
    adata.uns["spatial_barcodes_matched"] = int(len(shared))
    adata.uns["spatial_barcodes_total"] = int(len(positions))

    return adata


def add_scanpy_spatial_metadata(adata, sample_dir):
    """
    Add Scanpy-compatible Visium metadata to adata.uns["spatial"].

    This is required for sc.pl.spatial to use tissue images, scalefactors,
    and spot diameter correctly.
    """
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name

    scalefactors = read_scalefactors(sample_dir)

    hires_path = find_image_file(sample_dir, "hires")
    lowres_path = find_image_file(sample_dir, "lowres")

    hires_img = read_image_or_none(hires_path)
    lowres_img = read_image_or_none(lowres_path)

    images = {}

    if hires_img is not None:
        images["hires"] = hires_img

    if lowres_img is not None:
        images["lowres"] = lowres_img

    if not scalefactors and not images:
        adata.uns["scanpy_spatial_metadata_loaded"] = False
        adata.uns["scanpy_spatial_metadata_error"] = "No scalefactors or tissue images found"
        return adata

    if "spot_diameter_fullres" not in scalefactors:
        scalefactors["spot_diameter_fullres"] = 1.0

    if "tissue_hires_scalef" not in scalefactors:
        scalefactors["tissue_hires_scalef"] = 1.0

    if "tissue_lowres_scalef" not in scalefactors:
        scalefactors["tissue_lowres_scalef"] = scalefactors.get("tissue_hires_scalef", 1.0)

    adata.uns["spatial"] = {
        sample_id: {
            "images": images,
            "scalefactors": scalefactors,
            "metadata": {
                "source_sample_dir": str(sample_dir),
                "source_image_path": str(hires_path) if hires_path else str(lowres_path) if lowres_path else "",
                "hires_image_path": str(hires_path) if hires_path else "",
                "lowres_image_path": str(lowres_path) if lowres_path else "",
                "scalefactors_path": str(find_scalefactors_file(sample_dir)) if find_scalefactors_file(sample_dir) else "",
            },
        }
    }

    adata.uns["scanpy_spatial_metadata_loaded"] = True
    adata.uns["scanpy_spatial_library_id"] = sample_id
    adata.uns["scanpy_spatial_has_hires"] = bool("hires" in images)
    adata.uns["scanpy_spatial_has_lowres"] = bool("lowres" in images)
    adata.uns["scanpy_spatial_has_scalefactors"] = bool(scalefactors)

    return adata


# =========================
# Load expression data
# =========================

def load_h5(sample_dir):
    """Load a 10x H5 expression matrix into AnnData."""
    sample_dir = Path(sample_dir)
    raw_dir = get_raw_dir(sample_dir)

    h5_path = find_h5(raw_dir)

    if h5_path is None:
        raise FileNotFoundError(f"No H5 file found in {raw_dir}")

    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    return adata, h5_path.name


def load_mtx(sample_dir):
    """Load an uncompressed 10x MTX expression matrix into AnnData."""
    sample_dir = Path(sample_dir)
    raw_dir = get_raw_dir(sample_dir)

    matrix, barcodes, features = find_mtx_files(raw_dir)

    if matrix is None:
        raise FileNotFoundError(
            f"Missing matrix.mtx, barcodes.tsv, or features.tsv in {raw_dir}"
        )

    X = scipy.io.mmread(matrix).T.tocsr()

    barcodes_df = pd.read_csv(barcodes, header=None, sep="\t")
    features_df = pd.read_csv(features, header=None, sep="\t")

    obs = pd.DataFrame(index=barcodes_df[0].astype(str))
    obs.index.name = None

    if features_df.shape[1] >= 2:
        gene_names = features_df[1].astype(str)
        gene_ids = features_df[0].astype(str)
    else:
        gene_names = features_df[0].astype(str)
        gene_ids = features_df[0].astype(str)

    var = pd.DataFrame(
        {
            "gene_ids": gene_ids.values,
        },
        index=gene_names.values,
    )
    var.index.name = None

    adata = AnnData(X=X, obs=obs, var=var)

    if adata.n_obs != len(obs):
        raise ValueError("Mismatch between matrix rows and barcodes")

    if adata.n_vars != len(var):
        raise ValueError("Mismatch between matrix columns and features")

    adata.var_names_make_unique()

    return adata, "mtx_manual"


def load_expression(sample_dir):
    """Load expression data from h5 or mtx format."""
    sample_dir = Path(sample_dir)

    expression_format = detect_expression_format(sample_dir)

    if expression_format == "h5":
        adata, loaded_from = load_h5(sample_dir)

    elif expression_format == "mtx":
        adata, loaded_from = load_mtx(sample_dir)

    else:
        raise FileNotFoundError(
            f"No usable expression input found in {get_raw_dir(sample_dir)}"
        )

    return adata, {
        "format": expression_format,
        "loaded_from": loaded_from,
    }


# =========================
# Public sample loader
# =========================

def load_sample(sample_dir):
    """Load one sample and return AnnData plus metadata about loading."""
    sample_dir = Path(sample_dir)

    adata, load_info = load_expression(sample_dir)

    adata = add_spatial_coordinates(adata, sample_dir)
    adata = add_scanpy_spatial_metadata(adata, sample_dir)

    meta = read_metadata(sample_dir)
    spatial_info = check_spatial_files(sample_dir)

    adata.obs["sample_id"] = sample_dir.name
    adata.uns["sample_id"] = sample_dir.name
    adata.uns["source_sample_dir"] = str(sample_dir)

    for key, value in meta.items():
        adata.uns[f"metadata_{key}"] = value

    info = {
        "sample_id": sample_dir.name,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        **load_info,
        **spatial_info,
        **meta,
    }

    return adata, info


# =========================
# Output helpers
# =========================

def ensure_dir(path):
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataframe(df, path):
    """Write a DataFrame to CSV after creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def log(message):
    """Simple print wrapper for consistent logging."""
    print(f"[IO] {message}")


def save_h5ad(adata, path):
    """Save AnnData object safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(path)
    return path


def summarize_adata(adata):
    """Return basic summary stats for an AnnData object."""
    return {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

