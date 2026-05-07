"""
Script: 13a_external_study_validation.py

Purpose:
Run external study validation for selected cohorts using already generated pipeline outputs.

Project context:
This script is a validation and review layer. It reads Step 05 labeled AnnData files,
creates local validation maps, copies matched study figures when available, and builds
manual review manifests. It does not modify canonical pipeline outputs or model inputs.

Primary modes:
    1. Hepatoblastoma validation using matched samples and marker based review maps.
    2. PDAC validation delegated to lib.external_validation_pdac when --study pdac is used.

Important:
External marker maps are validation overlays only. They are not added to the main
pipeline label ontology and should not be interpreted as training labels.
"""

from __future__ import annotations


# =========================
# Imports
# =========================

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.spatial import cKDTree

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False



# =========================
# Project path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, validate_config



# =========================
# Script constants and output folders
# =========================

SCRIPT_NAME = "13a_external_study_validation.py"
DEFAULT_OUTPUT_SUBDIR = "output_13_external_study_validation"
STEP05_SUBDIR = "output_05_build_multi_axis_transcriptome_labels"

STRUCTURE_LABEL_COLUMNS = [
    "structure_region_label_smoothed",
    "structure_region_label",
    "structure_dominant_label_raw",
    "structure_dominant_label",
]


# =========================
# Hepatoblastoma sample and marker definitions
# =========================

HB_DEFAULT_SAMPLES = {
    "H1190": "SAMPLE_0012",
    "H391": "SAMPLE_0026",
    "H1226": "SAMPLE_0016",
    "H388": "SAMPLE_0024",
    "H691": "SAMPLE_0028",
    "H1180": "SAMPLE_0010",
}

HB_MARKER_SETS = {
    "normal_liver_like": [
        "ALB", "APOA1", "APOA2", "APOB", "APOC1", "APOC2", "APOC3",
        "FABP1", "CYP2E1", "LIPC", "AGT", "FGB", "FGG", "FGA",
    ],
    "hb_metabolic_program": [
        "LIPC", "APOC2", "APOE", "APOC1", "AGT", "APOB",
        "CYP2E1", "APOC3", "FABP1", "APOA1",
    ],
    "hb_developmental_wnt_mdk": [
        "MDK", "DKK1", "PDE4D", "TRPS1", "AXIN2", "PTK7", "STRA6",
        "SOX11", "AHI1", "PTCH1", "PTPRS", "SPRY2", "CTNNB1", "ID3",
    ],
    "hb_cycling_proliferative": [
        "MKI67", "BIRC5", "MAD2L1", "TYMS", "CCNB2", "HMMR",
        "CCNA2", "AURKA", "NUSAP1", "CENPW", "CDKN3", "TOP2A",
    ],
    "stroma_ecm": [
        "COL1A1", "COL1A2", "COL3A1", "COL18A1", "DCN", "LUM",
        "VIM", "TGFBI", "SERPINE1", "POSTN", "FN1", "MMP11",
    ],
    "myeloid_macrophage": [
        "LYZ", "CD14", "FCGR3A", "CD68", "CD163", "MSR1", "MARCO",
        "C1QA", "C1QB", "APOE", "S100A10", "CXCL8", "EGR1", "VSIR", "SIRPA",
    ],
    "lymphoid": [
        "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "MS4A1", "CD79A",
        "CD79B", "NKG7", "GNLY", "GZMB", "TRAC",
    ],
    "endothelial": [
        "PECAM1", "VWF", "KDR", "CDH5", "RGS5", "COL15A1",
    ],
}


# =========================
# Plot color definitions
# =========================

PIPELINE_REGION_COLORS = {
    "tumor_like": "#d64f4f",
    "stroma_like": "#75b878",
    "immune_like": "#8b6fc6",
    "vascular_like": "#46b9c7",
    "unmapped_or_low_signal": "#bdbdbd",
}

VALIDATION_REGION_COLORS = {
    "tumor_like": "#d64f4f",
    "stroma_like": "#75b878",
    "normal_liver_like": "#3b8bc2",
    "immune_like": "#8b6fc6",
    "vascular_like": "#46b9c7",
    "unmapped_or_low_signal": "#bdbdbd",
}

CORE_BORDER_COLORS = {
    "tumor_core": "#d64f4f",
    "tumor_border": "#f2e394",
    "non_tumor": "#3b8bc2",
}



# =========================
# Argument and config helpers
# =========================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for external study validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--study", default="hepatoblastoma")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--study-png-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_value, pipeline_root: Path, output_root: Path) -> Path:
    """Resolve relative or absolute config paths against pipeline and output roots."""
    if path_value is None or str(path_value).strip() == "":
        return Path("")

    p = Path(str(path_value))

    if p.is_absolute():
        return p

    p1 = pipeline_root / p
    if p1.exists():
        return p1

    p2 = output_root / p
    if p2.exists():
        return p2

    return p1


def find_latest_study_png_dir(output_root: Path, study: str) -> Path | None:
    """Find the newest available study PNG folder for a validation study."""
    base = output_root / "_external_study_validation" / study

    if not base.exists():
        return None

    candidates = sorted(
        [p for p in base.rglob("study_pngs_to_create") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        return candidates[0]

    candidates = sorted(
        [p for p in base.rglob("study_pngs") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        return candidates[0]

    return None


def get_config_study_block(cfg: dict, study: str) -> dict:
    """Return the external validation config block for one study."""
    ev = cfg.get("external_validation", {}) or {}

    if study in ev and isinstance(ev[study], dict):
        return ev[study]

    studies = ev.get("studies", {}) or {}
    if study in studies and isinstance(studies[study], dict):
        return studies[study]

    return {}


def get_output_dir(output_root: Path, args: argparse.Namespace, cfg: dict) -> Path:
    """Resolve the Step 13 output directory for the selected study."""
    ev = cfg.get("external_validation", {}) or {}

    subdir = (
        args.output_subdir
        or ev.get("output_subdir")
        or DEFAULT_OUTPUT_SUBDIR
    )

    return output_root / subdir / args.study


def get_sample_map(cfg: dict, study: str) -> dict[str, str]:
    """Return the configured or default study sample map."""
    block = get_config_study_block(cfg, study)
    samples = block.get("samples", None)

    if isinstance(samples, dict) and samples:
        return {str(k): str(v) for k, v in samples.items()}

    if study == "hepatoblastoma":
        return HB_DEFAULT_SAMPLES.copy()

    raise ValueError(f"No sample map configured for study: {study}")


def get_study_png_dir(cfg: dict, output_root: Path, args: argparse.Namespace) -> Path | None:
    """Resolve the directory containing study reference PNGs."""
    if args.study_png_dir:
        p = Path(args.study_png_dir)
        return p if p.exists() else p

    ev = cfg.get("external_validation", {}) or {}
    block = get_config_study_block(cfg, args.study)

    configured = (
        block.get("study_png_dir")
        or ev.get("study_png_root")
        or ev.get("study_png_dir")
    )

    if configured:
        p = resolve_path(configured, PROJECT_ROOT, output_root)
        if p.exists():
            return p

    return find_latest_study_png_dir(output_root, args.study)



# =========================
# Output and sample lookup helpers
# =========================

def h5ad_path(output_root: Path, sample_id: str) -> Path:
    """Return the Step 05 labeled AnnData path for one sample."""
    p = (
        output_root
        / STEP05_SUBDIR
        / "per_sample_h5ad"
        / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    )

    if not p.exists():
        raise FileNotFoundError(f"Missing step 05 h5ad: {p}")

    return p


def get_coords(adata) -> np.ndarray:
    """Extract spatial coordinates from an AnnData object."""
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData is missing obsm['spatial']")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Invalid spatial coordinates")

    return coords[:, :2]



# =========================
# Expression and marker scoring helpers
# =========================

def expression_source(adata):
    """Return raw expression when available, otherwise use active AnnData expression."""
    if adata.raw is not None:
        src = adata.raw.to_adata()
    else:
        src = adata.copy()

    src.obs_names = src.obs_names.astype(str)
    src.var_names = src.var_names.astype(str)
    src.var_names_make_unique()
    return src


def vector_for_gene(src, gene: str):
    """Return one gene expression vector from an AnnData expression source."""
    gene = gene.upper()
    var_upper = pd.Index(src.var_names.astype(str).str.upper())
    idx = np.where(var_upper == gene)[0]

    if len(idx) == 0:
        return None

    x = src.X[:, idx]

    if sparse.issparse(x):
        arr = x.toarray()
    else:
        arr = np.asarray(x)

    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    return np.asarray(arr, dtype=float).reshape(-1)


def rank01(values) -> np.ndarray:
    """Convert values into percentile ranks scaled from zero to one."""
    s = pd.Series(values)

    if s.notna().sum() <= 1:
        return np.zeros(len(s), dtype=float)

    if s.nunique(dropna=True) <= 1:
        return np.full(len(s), 0.5, dtype=float)

    return s.rank(pct=True).fillna(0.0).to_numpy(dtype=float)


def signature_score(src, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute a marker signature score from genes present in the sample."""
    found = []
    values = []

    for gene in genes:
        v = vector_for_gene(src, gene)

        if v is not None:
            found.append(gene)
            values.append(rank01(v))

    if not values:
        return np.zeros(src.n_obs, dtype=float), found

    return np.nanmean(np.vstack(values), axis=0), found


def obs_score(adata, columns: list[str]) -> np.ndarray:
    """Compute a rank averaged score from AnnData observation columns."""
    values = []

    for col in columns:
        if col in adata.obs.columns:
            values.append(rank01(pd.to_numeric(adata.obs[col], errors="coerce")))

    if not values:
        return np.zeros(adata.n_obs, dtype=float)

    return np.nanmean(np.vstack(values), axis=0)



# =========================
# Pipeline region crosswalk helpers
# =========================

def choose_structure_label_column(obs: pd.DataFrame) -> str | None:
    """Choose the best available structure label column."""
    for col in STRUCTURE_LABEL_COLUMNS:
        if col in obs.columns:
            return col
    return None


def pipeline_region_crosswalk(adata) -> np.ndarray:
    """Collapse pipeline structure labels into validation region classes."""
    obs = adata.obs.copy()
    col = choose_structure_label_column(obs)

    if col is None:
        return np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    labels = obs[col].astype(str).str.lower()

    out = np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    tumor = labels.str.contains("tumor", na=False).to_numpy()
    stroma = labels.str.contains("stromal|ecm", na=False).to_numpy()
    immune = labels.str.contains("t_cell|immune|plasma|myeloid|macrophage", na=False).to_numpy()
    vascular = labels.str.contains("vascular|angiogenic|endothelial", na=False).to_numpy()

    out[stroma] = "stroma_like"
    out[immune] = "immune_like"
    out[vascular] = "vascular_like"
    out[tumor] = "tumor_like"

    return out


def validation_region_proxy(adata, scores: dict[str, np.ndarray]) -> np.ndarray:
    """Add validation only normal or stromal proxies without changing pipeline labels."""
    pure = pipeline_region_crosswalk(adata).copy()

    normal = scores.get("normal_liver_like", np.zeros(adata.n_obs))
    stroma = scores.get("stroma_ecm", np.zeros(adata.n_obs))

    normal_thr = np.nanquantile(normal, 0.75)
    stroma_thr = np.nanquantile(stroma, 0.75)

    out = pure.copy()

    normal_mask = (normal >= normal_thr) & (pure != "tumor_like")
    stroma_mask = (stroma >= stroma_thr) & (pure != "tumor_like")

    out[normal_mask] = "normal_liver_like"
    out[stroma_mask] = "stroma_like"

    return out



# =========================
# Spatial proxy and plotting helpers
# =========================

def core_border_from_region(coords: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Create tumor core and border proxy labels from spatial tumor regions."""
    tumor = region == "tumor_like"
    out = np.array(["non_tumor"] * len(region), dtype=object)

    if tumor.sum() == 0:
        return out

    if len(region) < 3:
        out[tumor] = "tumor_core"
        return out

    tree = cKDTree(coords)
    d2, _ = tree.query(coords, k=2)
    radius = float(np.nanmedian(d2[:, 1]) * 1.35)

    tumor_indices = np.where(tumor)[0]

    for i in tumor_indices:
        neighbors = tree.query_ball_point(coords[i], r=radius)
        touches_non_tumor = any(not tumor[j] for j in neighbors if j != i)
        out[i] = "tumor_border" if touches_non_tumor else "tumor_core"

    return out


def plot_categorical(coords, labels, title: str, dest: Path, colors: dict[str, str]) -> None:
    """Write a categorical spatial map as a static PNG."""
    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    for label, color in colors.items():
        mask = labels == label
        if int(mask.sum()) == 0:
            continue

        ax.scatter(
            x[mask],
            y[mask],
            s=9,
            c=color,
            label=label,
            linewidths=0,
            alpha=0.95,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(dest, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score(coords, score, title: str, dest: Path) -> None:
    """Write a continuous score spatial map as a static PNG."""
    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(
        x,
        y,
        s=9,
        c=score,
        cmap="viridis",
        linewidths=0,
        alpha=0.95,
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(dest, dpi=220, bbox_inches="tight")
    plt.close(fig)



# =========================
# Study figure matching helpers
# =========================

def find_study_png(study_png_dir: Path | None, requested_name: str) -> Path | None:
    """Find a requested study PNG by exact or approximate name matching."""
    if study_png_dir is None or not study_png_dir.exists():
        return None

    exact = study_png_dir / requested_name
    if exact.exists():
        return exact

    requested_words = [
        x.lower()
        for x in requested_name.replace(".png", "").replace("_", " ").split()
        if len(x) > 1
    ]

    scored = []

    for p in study_png_dir.glob("*.png"):
        low = p.name.lower()
        score = sum(1 for w in requested_words if w in low)

        if score > 0:
            scored.append((score, p))

    if not scored:
        return None

    scored = sorted(scored, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]


def side_by_side(local_img: Path, study_img: Path, dest: Path, title: str) -> str:
    """Create a side by side local map and study figure comparison image."""
    if not HAS_PIL:
        return "pillow_not_available"

    if not local_img.exists() or not study_img.exists():
        return "missing_input"

    left = Image.open(local_img).convert("RGB")
    right = Image.open(study_img).convert("RGB")

    target_h = 900

    def resize_to_height(img):
        """Document resize to height within the 13a_external validation workflow."""
        w, h = img.size
        return img.resize((int(w * target_h / h), target_h))

    left = resize_to_height(left)
    right = resize_to_height(right)

    pad = 35
    title_h = 90
    width = left.width + right.width + pad * 3
    height = target_h + title_h + pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 20), title, fill="black")
    draw.text((pad, 60), "LOCAL VALIDATION MAP", fill="black")
    draw.text((pad * 2 + left.width, 60), "STUDY FIGURE", fill="black")
    canvas.paste(left, (pad, title_h))
    canvas.paste(right, (pad * 2 + left.width, title_h))
    canvas.save(dest)

    return "created"



# =========================
# Hepatoblastoma comparison definitions
# =========================

def get_comparisons(study: str) -> list[dict]:
    """Return predefined comparison specifications for one validation study."""
    if study != "hepatoblastoma":
        raise ValueError("Only hepatoblastoma is implemented in this script version.")

    return [
        {
            "id": "01",
            "sample": "H1190",
            "kind": "pipeline_region_crosswalk",
            "study_png": "Supplement Fig. S4b H1190.png",
            "purpose": "Pure pipeline labels collapsed to paper style regions",
            "look_at": "tumor like versus stroma or normal liver area",
        },
        {
            "id": "02",
            "sample": "H1190",
            "kind": "validation_region_proxy",
            "study_png": "Supplement Fig. S4b H1190.png",
            "purpose": "Validation proxy with normal liver like added",
            "look_at": "normal liver like, stroma like, tumor like",
        },
        {
            "id": "03",
            "sample": "H1190",
            "kind": "pipeline_metabolic_score",
            "study_png": "Supplement Fig. S5c H1190.png",
            "purpose": "Pipeline metabolic score proxy",
            "look_at": "metabolic signal in tumor region",
        },
        {
            "id": "04",
            "sample": "H1190",
            "kind": "hb_metabolic_program",
            "study_png": "Supplement Fig. S5c H1190.png",
            "purpose": "Study marker metabolic validation score",
            "look_at": "same sample expression pattern, not canonical pipeline label",
        },
        {
            "id": "05",
            "sample": "H1190",
            "kind": "MDK",
            "study_png": "Main Fig. 4E H1190.png",
            "purpose": "MDK expression validation",
            "look_at": "MDK high in tumor and low in normal liver",
        },
        {
            "id": "06",
            "sample": "H391",
            "kind": "pipeline_region_crosswalk",
            "study_png": "Supplement Fig. S4b H391.png",
            "purpose": "Pure pipeline labels collapsed to paper style regions",
            "look_at": "large tumor island plus non tumor region",
        },
        {
            "id": "07",
            "sample": "H391",
            "kind": "core_border_pipeline",
            "study_png": "Main Fig. 3f H391.png",
            "purpose": "Tumor core and border from pipeline tumor like labels",
            "look_at": "tumor core, border, and non tumor geometry",
        },
        {
            "id": "08",
            "sample": "H391",
            "kind": "myeloid_macrophage",
            "study_png": "Supplement Fig. S8c H391.png",
            "purpose": "Myeloid macrophage validation score",
            "look_at": "myeloid signal near interface and outside tumor",
        },
        {
            "id": "09",
            "sample": "H1226",
            "kind": "core_border_pipeline",
            "study_png": "Supplement Fig. S13a H1226.png",
            "purpose": "Fragmented tumor core and border from pipeline labels",
            "look_at": "small separated tumor islands",
        },
        {
            "id": "10",
            "sample": "H388",
            "kind": "core_border_pipeline",
            "study_png": "Supplement Fig. S13a H388.png",
            "purpose": "Large tumor versus non tumor boundary from pipeline labels",
            "look_at": "large tumor block and diagonal boundary",
        },
    ]



# =========================
# Sample layer computation
# =========================

def compute_sample_layers(adata) -> dict:
    """Compute local validation layers and marker scores for one AnnData sample."""
    coords = get_coords(adata)
    src = expression_source(adata)

    scores = {}
    genes_found = {}

    for name, genes in HB_MARKER_SETS.items():
        scores[name], genes_found[name] = signature_score(src, genes)

    mdk = vector_for_gene(src, "MDK")
    if mdk is None:
        scores["MDK"] = np.zeros(adata.n_obs)
        genes_found["MDK"] = []
    else:
        scores["MDK"] = rank01(mdk)
        genes_found["MDK"] = ["MDK"]

    pipeline_region = pipeline_region_crosswalk(adata)
    validation_region = validation_region_proxy(adata, scores)
    core_border_pipeline = core_border_from_region(coords, pipeline_region)

    pipeline_metabolic = obs_score(
        adata,
        [
            "metabolic_score__fatty_acid_oxidation",
            "metabolic_score__oxphos",
            "metabolic_score__glutamine_metabolism",
            "metabolic_score__nucleotide_synthesis",
        ],
    )

    pipeline_proliferative = obs_score(
        adata,
        [
            "structure_score__tumor_proliferative",
            "function_score__tumor_proliferative_function",
        ],
    )

    scores["pipeline_metabolic_score"] = pipeline_metabolic
    scores["pipeline_proliferative_score"] = pipeline_proliferative

    return {
        "coords": coords,
        "scores": scores,
        "genes_found": genes_found,
        "pipeline_region": pipeline_region,
        "validation_region": validation_region,
        "core_border_pipeline": core_border_pipeline,
    }


def plot_kind(sample_name: str, sample_id: str, kind: str, layers: dict, dest: Path) -> None:
    """Render one requested validation map type for one sample."""
    coords = layers["coords"]

    if kind == "pipeline_region_crosswalk":
        plot_categorical(
            coords,
            layers["pipeline_region"],
            f"{sample_name} pipeline label crosswalk",
            dest,
            PIPELINE_REGION_COLORS,
        )
        return

    if kind == "validation_region_proxy":
        plot_categorical(
            coords,
            layers["validation_region"],
            f"{sample_name} validation region proxy",
            dest,
            VALIDATION_REGION_COLORS,
        )
        return

    if kind == "core_border_pipeline":
        plot_categorical(
            coords,
            layers["core_border_pipeline"],
            f"{sample_name} pipeline tumor core and border proxy",
            dest,
            CORE_BORDER_COLORS,
        )
        return

    if kind in layers["scores"]:
        plot_score(
            coords,
            layers["scores"][kind],
            f"{sample_name} {kind}",
            dest,
        )
        return

    raise ValueError(f"Unknown plot kind: {kind}")



# =========================
# Validation report writer
# =========================

def write_validation_report(out_dir: Path, rows: list[dict], genes_rows: list[dict], cfg: dict) -> None:
    """Write the external validation summary report."""
    report = out_dir / "external_validation_summary.txt"

    n_created = sum(1 for r in rows if r["side_by_side_status"] == "created")
    n_rows = len(rows)

    lines = [
        "External study validation summary",
        "",
        f"Script: {SCRIPT_NAME}",
        f"Study: hepatoblastoma",
        f"Comparisons: {n_rows}",
        f"Side by side figures created: {n_created}",
        "",
        "Interpretation:",
        "pipeline_region_crosswalk uses only the pipeline structure labels collapsed to paper compatible regions.",
        "validation_region_proxy adds a hepatoblastoma normal liver like marker layer for validation only.",
        "core_border_pipeline uses pipeline tumor like labels and spatial neighbors to reproduce the paper core and border concept.",
        "Marker maps such as MDK and hepatoblastoma metabolic program are same sample validation maps, not canonical pipeline labels.",
        "",
        "Recommended use:",
        "Score each comparison as strong match, partial match, weak match, or not assessable.",
        "",
        "Output files:",
        "manual_review_manifest.csv",
        "marker_genes_found_by_sample.csv",
        "side_by_side_pairs",
        "local_validation_maps",
        "study_pngs_used",
    ]

    report.write_text("\n".join(lines), encoding="utf-8")



# =========================
# Main workflow
# =========================

def main() -> None:
    """Run external study validation and write maps, manifests, and review outputs."""
    args = parse_args()

    if args.study == "pdac":
        from lib.external_validation_pdac import run_pdac_validation
        run_pdac_validation(args=args, project_root=PROJECT_ROOT)
        return

    # Step 13 is validation only and does not require input_root.
    # Avoid validate_config here because input_root may point to the old computer.
    cfg = load_config(args.config)

    output_root_from_config = Path(str(cfg.get("output_root", "")))
    if not output_root_from_config.exists():
        cfg["output_root"] = str(PROJECT_ROOT / "outputs")

    output_root = Path(cfg["output_root"])
    sample_map = get_sample_map(cfg, args.study)

    out_base = get_output_dir(output_root, args, cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"{args.study}_validation_{stamp}"

    local_dir = out_dir / "local_validation_maps"
    study_copy_dir = out_dir / "study_pngs_used"
    pair_dir = out_dir / "side_by_side_pairs"

    local_dir.mkdir(parents=True, exist_ok=True)
    study_copy_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    study_png_dir = get_study_png_dir(cfg, output_root, args)

    print("=== External study validation ===")
    print("Study:", args.study)
    print("Output root:", output_root)
    print("Output folder:", out_dir)
    print("Study PNG folder:", study_png_dir if study_png_dir else "not found")
    print()

    loaded = {}
    genes_rows = []

    for sample_name, sample_id in sample_map.items():
        p = h5ad_path(output_root, sample_id)
        a = ad.read_h5ad(p)
        layers = compute_sample_layers(a)

        loaded[sample_name] = {
            "sample_id": sample_id,
            "adata_path": str(p),
            "layers": layers,
        }

        for theme, genes in layers["genes_found"].items():
            genes_rows.append({
                "sample": sample_name,
                "sample_id": sample_id,
                "theme": theme,
                "n_genes_found": len(genes),
                "genes_found": ";".join(genes),
            })

    rows = []

    for comp in get_comparisons(args.study):
        sample_name = comp["sample"]
        sample_id = sample_map[sample_name]
        layers = loaded[sample_name]["layers"]

        local_name = f"{comp['id']}_LOCAL_{sample_name}_{comp['kind']}.png"
        local_path = local_dir / local_name

        plot_kind(sample_name, sample_id, comp["kind"], layers, local_path)

        study_src = find_study_png(study_png_dir, comp["study_png"])
        study_copy = ""

        if study_src is not None and study_src.exists():
            study_dest = study_copy_dir / f"{comp['id']}_STUDY_{sample_name}_{study_src.name}"
            shutil.copy2(study_src, study_dest)
            study_copy = str(study_dest)
            study_status = "copied"
        else:
            study_dest = None
            study_status = "missing"

        side_path = pair_dir / f"{comp['id']}_{sample_name}_{comp['kind']}_side_by_side.png"

        if study_dest is not None:
            side_status = side_by_side(
                local_path,
                study_dest,
                side_path,
                f"{comp['id']} {sample_name} {comp['purpose']}",
            )
        else:
            side_status = "missing_study_png"

        rows.append({
            "id": comp["id"],
            "sample": sample_name,
            "sample_id": sample_id,
            "comparison_kind": comp["kind"],
            "purpose": comp["purpose"],
            "look_at": comp["look_at"],
            "local_validation_map": str(local_path),
            "study_png_requested": comp["study_png"],
            "study_png_found": str(study_src) if study_src is not None else "",
            "study_png_copy": study_copy,
            "side_by_side": str(side_path) if side_path.exists() else "",
            "study_status": study_status,
            "side_by_side_status": side_status,
            "review_score": "",
            "review_notes": "",
        })

    manifest = out_dir / "manual_review_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    genes_path = out_dir / "marker_genes_found_by_sample.csv"
    pd.DataFrame(genes_rows).to_csv(genes_path, index=False)

    metadata = {
        "script": SCRIPT_NAME,
        "study": args.study,
        "output_folder": str(out_dir),
        "study_png_dir": str(study_png_dir) if study_png_dir is not None else "",
        "output_root": str(output_root),
        "sample_map": sample_map,
        "note": "Validation only. Does not alter pipeline outputs or model_input_numeric.csv.",
    }

    with open(out_dir / "external_validation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_validation_report(out_dir, rows, genes_rows, cfg)

    print("DONE")
    print("Output folder:", out_dir)
    print("Manifest:", manifest)
    print("Side by side folder:", pair_dir)
    print("Summary:", out_dir / "external_validation_summary.txt")



# =========================
# Command line entry point
# =========================

if __name__ == "__main__":
    main()


