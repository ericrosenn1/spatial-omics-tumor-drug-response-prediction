"""
Script: 02_process_samples.py

Purpose:
Process Visium style sample folders into cleaned, normalized, clustered AnnData
objects and per sample slide level feature tables.

This script is the main sample processing step of the spatial feature
identification pipeline. It assumes that input validation has already passed.

Inputs:
    A YAML config file containing:
        input_root
        output_root
        sample_glob

Outputs:
    processed_samples/<sample_id>/adata/01_loaded.h5ad
    processed_samples/<sample_id>/adata/02_processed.h5ad
    processed_samples/<sample_id>/tables/slide_level_feature_row.csv
    processed_samples/<sample_id>/tables/cluster_summary.csv
    processing/processing_report.csv
    processing/processing_summary.txt

Typical usage:
    python scripts/02_process_samples.py --config configs/visium_cohort_clean.yaml
    python scripts/02_process_samples.py --config configs/visium_cohort_clean.yaml --limit 3
"""

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
import pandas as pd
import scanpy as sc

from tqdm import tqdm

# =========================
# Import project modules
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config
from lib.io import load_sample, save_h5ad, write_dataframe


# =========================
# Default marker programs
# =========================

MARKER_PROGRAMS = {
    "tumor_epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"
    ],
    "immune_general": [
        "PTPRC", "CD3D", "CD3E", "CD2", "LCK"
    ],
    "myeloid": [
        "LYZ", "CD68", "FCGR3A", "LST1", "AIF1"
    ],
    "fibroblast_stroma": [
        "COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"
    ],
    "endothelial": [
        "PECAM1", "VWF", "KDR", "ENG", "PLVAP"
    ],
    "hypoxia": [
        "CA9", "VEGFA", "SLC2A1", "ENO1", "LDHA"
    ],
    "proliferation": [
        "MKI67", "TOP2A", "PCNA", "MCM2", "UBE2C"
    ],
}


# =========================
# General helpers
# =========================

def safe_median(values):
    """Return median as float, or NaN if values cannot be summarized."""
    try:
        return float(np.nanmedian(values))
    except Exception:
        return np.nan


def get_existing_genes(adata, genes):
    """Return genes from a marker list that exist in adata.var_names."""
    available = set(adata.var_names)
    return [gene for gene in genes if gene in available]


def make_sample_output_dirs(output_root, sample_id):
    """Create and return output directories for one sample."""
    sample_root = Path(output_root) / "output_02_01_process_samples_data" / sample_id
    adata_dir = sample_root / "adata"
    table_dir = sample_root / "tables"
    plot_dir = sample_root / "plots"

    adata_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return {
        "sample_root": sample_root,
        "adata_dir": adata_dir,
        "table_dir": table_dir,
        "plot_dir": plot_dir,
    }


# =========================
# QC and preprocessing
# =========================

def add_qc_metrics(adata):
    """Add standard QC metrics to AnnData.obs."""
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    return adata


def filter_adata(
    adata,
    min_counts=500,
    min_genes=200,
    max_pct_mt=25,
    min_cells_per_gene=3,
):
    """Apply basic spot and gene filters."""
    before_spots = adata.n_obs
    before_genes = adata.n_vars

    keep_spots = (
        (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["pct_counts_mt"] <= max_pct_mt)
    )

    adata = adata[keep_spots].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    adata.uns["filtering_summary"] = {
        "spots_before": int(before_spots),
        "spots_after": int(adata.n_obs),
        "genes_before": int(before_genes),
        "genes_after": int(adata.n_vars),
        "min_counts": min_counts,
        "min_genes": min_genes,
        "max_pct_mt": max_pct_mt,
        "min_cells_per_gene": min_cells_per_gene,
    }

    return adata


def preprocess_adata(
    adata,
    target_sum=1e4,
    n_top_genes=3000,
    n_pcs=30,
):
    """Normalize, log transform, select variable genes, and compute PCA graph."""
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    adata.raw = adata.copy()

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
    )

    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() > 0:
        adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.scale(adata, max_value=10)

    max_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)

    if max_pcs >= 2:
        sc.tl.pca(adata, n_comps=max_pcs)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=max_pcs)

        try:
            sc.tl.umap(adata)
        except Exception:
            adata.uns["umap_error"] = "UMAP failed"

        try:
            sc.tl.leiden(adata, resolution=0.6, key_added="leiden")
        except Exception:
            adata.obs["leiden"] = "0"
            adata.uns["leiden_error"] = "Leiden failed. Check leidenalg installation."
    else:
        adata.obs["leiden"] = "0"
        adata.uns["pca_error"] = "Too few spots or genes for PCA"

    return adata


# =========================
# Marker scoring and annotation
# =========================

def score_marker_programs(adata, marker_programs=None):
    """Score simple biological marker programs using Scanpy score_genes."""
    if marker_programs is None:
        marker_programs = MARKER_PROGRAMS

    score_columns = []

    for program_name, genes in marker_programs.items():
        existing_genes = get_existing_genes(adata, genes)
        score_col = f"{program_name}_score"

        if len(existing_genes) >= 2:
            sc.tl.score_genes(
                adata,
                gene_list=existing_genes,
                score_name=score_col,
                use_raw=True,
            )
            adata.uns[f"{program_name}_genes_used"] = existing_genes
            score_columns.append(score_col)
        else:
            adata.obs[score_col] = np.nan
            adata.uns[f"{program_name}_genes_used"] = existing_genes

    return adata, score_columns


def assign_spot_labels(adata, score_columns):
    """Assign each spot a simple label based on the highest marker score."""
    if not score_columns:
        adata.obs["simple_label"] = "unknown"
        return adata

    score_frame = adata.obs[score_columns].copy()

    if score_frame.notna().sum().sum() == 0:
        adata.obs["simple_label"] = "unknown"
        return adata

    best_score_col = score_frame.idxmax(axis=1)
    labels = best_score_col.str.replace("_score", "", regex=False)

    adata.obs["simple_label"] = labels.fillna("unknown")

    return adata


def summarize_clusters(adata):
    """Create a cluster summary table with label composition and spot counts."""
    if "leiden" not in adata.obs.columns:
        return pd.DataFrame()

    rows = []

    for cluster_id, sub_obs in adata.obs.groupby("leiden"):
        row = {
            "cluster": cluster_id,
            "n_spots": int(len(sub_obs)),
            "fraction_spots": float(len(sub_obs) / adata.n_obs),
        }

        if "simple_label" in sub_obs.columns:
            label_counts = sub_obs["simple_label"].value_counts(normalize=True)
            for label, value in label_counts.items():
                row[f"label_fraction__{label}"] = float(value)

            row["dominant_label"] = str(label_counts.idxmax())
        else:
            row["dominant_label"] = "unknown"

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Spatial coordinate helpers
# =========================

def find_spatial_coordinate_columns(adata):
    """Find likely spatial coordinate columns in AnnData."""
    if "spatial" in adata.obsm:
        return "obsm_spatial"

    possible_pairs = [
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
    ]

    for x_col, y_col in possible_pairs:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return x_col, y_col

    return None


def add_basic_spatial_features(adata):
    """Add simple spatial coordinate features if spatial coordinates are present."""
    coord_source = find_spatial_coordinate_columns(adata)
    adata.uns["spatial_coordinate_source"] = str(coord_source)

    if coord_source == "obsm_spatial":
        coords = np.asarray(adata.obsm["spatial"])
        adata.obs["spatial_x"] = coords[:, 0]
        adata.obs["spatial_y"] = coords[:, 1]
        return adata

    if isinstance(coord_source, tuple):
        x_col, y_col = coord_source
        adata.obs["spatial_x"] = adata.obs[x_col].astype(float)
        adata.obs["spatial_y"] = adata.obs[y_col].astype(float)
        return adata

    adata.obs["spatial_x"] = np.nan
    adata.obs["spatial_y"] = np.nan

    return adata


# =========================
# Slide level feature row
# =========================

def build_slide_feature_row(adata, sample_info):
    """Build one slide level feature row from processed AnnData."""
    row = {
        "sample_id": sample_info.get("sample_id", adata.uns.get("sample_id", "")),
        "format": sample_info.get("format", ""),
        "loaded_from": sample_info.get("loaded_from", ""),
        "dataset_id": sample_info.get("dataset_id", ""),
        "cancer_type": sample_info.get("cancer_type", ""),
        "timepoint": sample_info.get("timepoint", ""),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

    if "filtering_summary" in adata.uns:
        for key, value in adata.uns["filtering_summary"].items():
            row[f"filtering__{key}"] = value

    qc_columns = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    ]

    for col in qc_columns:
        if col in adata.obs.columns:
            row[f"median__{col}"] = safe_median(adata.obs[col])

    score_cols = [col for col in adata.obs.columns if col.endswith("_score")]

    for col in score_cols:
        row[f"mean__{col}"] = float(np.nanmean(adata.obs[col]))
        row[f"median__{col}"] = safe_median(adata.obs[col])

    if "simple_label" in adata.obs.columns:
        label_fracs = adata.obs["simple_label"].value_counts(normalize=True)
        for label, frac in label_fracs.items():
            row[f"spot_fraction__{label}"] = float(frac)

    if "leiden" in adata.obs.columns:
        row["n_clusters"] = int(adata.obs["leiden"].nunique())

    if "spatial_x" in adata.obs.columns and "spatial_y" in adata.obs.columns:
        x_vals = adata.obs["spatial_x"].values
        y_vals = adata.obs["spatial_y"].values

        # handle NaN-safe min/max
        row["spatial_x_min"] = float(np.nanmin(x_vals)) if not np.all(np.isnan(x_vals)) else np.nan
        row["spatial_x_max"] = float(np.nanmax(x_vals)) if not np.all(np.isnan(x_vals)) else np.nan
        row["spatial_y_min"] = float(np.nanmin(y_vals)) if not np.all(np.isnan(y_vals)) else np.nan
        row["spatial_y_max"] = float(np.nanmax(y_vals)) if not np.all(np.isnan(y_vals)) else np.nan

    return pd.DataFrame([row])


# =========================
# Per sample processing
# =========================

def process_one_sample(sample_dir, output_root, overwrite=False):
    """Process one sample and write processed outputs."""
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name

    dirs = make_sample_output_dirs(output_root, sample_id)

    final_h5ad = dirs["adata_dir"] / "02_processed.h5ad"
    feature_path = dirs["table_dir"] / "slide_level_feature_row.csv"
    cluster_path = dirs["table_dir"] / "cluster_summary.csv"

    if final_h5ad.exists() and feature_path.exists() and not overwrite:
        return {
            "sample_id": sample_id,
            "status": "SKIPPED",
            "reason": "outputs_exist",
            "processed_h5ad": str(final_h5ad),
            "slide_feature_row": str(feature_path),
            "error": "",
        }

    adata, sample_info = load_sample(sample_dir)

    save_h5ad(adata, dirs["adata_dir"] / "01_loaded.h5ad")

    adata = add_qc_metrics(adata)
    adata = filter_adata(adata)

    # handle empty dataset after filtering
    if adata.n_obs == 0 or adata.n_vars == 0:
        return {
            "sample_id": sample_id,
            "status": "SKIPPED",
            "reason": "empty_after_filtering",
            "processed_h5ad": "",
            "slide_feature_row": "",
            "cluster_summary": "",
            "n_spots": 0,
            "n_genes": 0,
            "n_clusters": 0,
            "error": "",
        }

    adata = preprocess_adata(adata)

    adata, score_columns = score_marker_programs(adata)
    adata = assign_spot_labels(adata, score_columns)
    adata = add_basic_spatial_features(adata)

    cluster_summary = summarize_clusters(adata)
    slide_feature_row = build_slide_feature_row(adata, sample_info)

    save_h5ad(adata, final_h5ad)
    write_dataframe(slide_feature_row, feature_path)

    if len(cluster_summary) > 0:
        write_dataframe(cluster_summary, cluster_path)

    return {
        "sample_id": sample_id,
        "status": "OK",
        "reason": "",
        "processed_h5ad": str(final_h5ad),
        "slide_feature_row": str(feature_path),
        "cluster_summary": str(cluster_path),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_clusters": int(adata.obs["leiden"].nunique()) if "leiden" in adata.obs.columns else 0,
        "error": "",
    }


# =========================
# Report helpers
# =========================

def build_processing_summary(report):
    """Build a readable text summary of the processing run."""
    lines = []

    lines.append("Processing summary")
    lines.append("")
    lines.append(f"Total samples attempted: {len(report)}")

    if "status" in report.columns:
        lines.append("")
        lines.append("Status counts:")
        for key, value in report["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    errors = report[report["status"] == "ERROR"] if "status" in report.columns else pd.DataFrame()

    lines.append("")
    lines.append(f"Failed samples: {len(errors)}")

    if len(errors) > 0:
        lines.append("")
        lines.append("Failure details:")
        for _, row in errors.iterrows():
            lines.append(f"  {row['sample_id']}: {row['error']}")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main():
    """Run sample processing across all sample folders."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = validate_config(load_config(args.config))

    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    sample_glob = cfg.get("sample_glob", "SAMPLE_*")

    processing_dir = output_root / "output_02_process_samples_reports"
    processing_dir.mkdir(parents=True, exist_ok=True)

    report_path = processing_dir / "processing_report.csv"
    summary_path = processing_dir / "processing_summary.txt"

    sample_dirs = sorted(
        sample_dir for sample_dir in input_root.glob(sample_glob)
        if sample_dir.is_dir()
    )

    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]

    print("=== Sample processing ===")
    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Sample glob:", sample_glob)
    print("Samples found:", len(sample_dirs))
    print("Overwrite:", args.overwrite)
    print()

    rows = []

    for i, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples"), start=1):
        tqdm.write(f"Processing {sample_dir.name}")

        try:
            row = process_one_sample(
                sample_dir=sample_dir,
                output_root=output_root,
                overwrite=args.overwrite,
            )

            rows.append(row)

            if row["status"] == "OK":
                print(
                    f"  OK: spots={row.get('n_spots', '')} | "
                    f"genes={row.get('n_genes', '')} | "
                    f"clusters={row.get('n_clusters', '')}"
                )

            elif row["status"] == "SKIPPED":
                print(f"  SKIPPED: {row.get('reason', '')}")

            else:
                print(f"  {row['status']}: {row.get('error', '')}")

        except Exception as error:
            error_row = {
                "sample_id": sample_dir.name,
                "status": "ERROR",
                "reason": "",
                "processed_h5ad": "",
                "slide_feature_row": "",
                "cluster_summary": "",
                "n_spots": "",
                "n_genes": "",
                "n_clusters": "",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

            rows.append(error_row)
            print(f"  ERROR: {error_row['error']}")

        # Save progress after every sample so the run is recoverable.
        pd.DataFrame(rows).to_csv(report_path, index=False)

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)

    summary_text = build_processing_summary(report)
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print("DONE")
    print("Report:", report_path)
    print("Summary:", summary_path)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()


