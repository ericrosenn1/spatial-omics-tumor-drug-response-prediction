
r"""
Script: 05_build_multi_axis_transcriptome_labels.py

Purpose:
Build the canonical multi axis transcriptome label layer for the spatial feature
identification pipeline.

This script replaces the older split logic from:
    05_append_signature_scores.py
    05.1_better_transcriptome_scoring.py
    05.2_multi_axis_transcriptome_scoring.py

Design:
    1. Score curated structure, function, and metabolic signatures.
    2. Add optional exact UCell, custom GSVA, Hallmark GSVA, and Reactome GSVA.
    3. Fuse methods after percentile normalization.
    4. Assign one dominant structural label while preserving runner up labels.
    5. Allow functional and metabolic labels to overlap.
    6. Track method agreement, confidence, low signal, and mismatch patterns.
    7. Write spot level, slide level, h5ad, gene support, and metadata outputs.

Typical usage:
    python .\code\05_build_multi_axis_transcriptome_labels.py --config .\configs\visium_cohort_clean.yaml

Fast test:
    python .\code\05_build_multi_axis_transcriptome_labels.py --config .\configs\visium_cohort_clean.yaml --limit 5 --overwrite

Safer laptop run without exact UCell:
    python .\code\05_build_multi_axis_transcriptome_labels.py --config .\configs\visium_cohort_clean.yaml --skip-ucell --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree


# =========================
# Project imports
# =========================

CODE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = CODE_ROOT.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lib.config import load_config, validate_config


# =========================
# Optional imports
# =========================

try:
    import pyucell as uc
    HAS_UCELL = True
except Exception:
    uc = None
    HAS_UCELL = False

try:
    import gseapy as gp
    from gseapy import Msigdb
    HAS_GSEAPY = True
except Exception:
    gp = None
    Msigdb = None
    HAS_GSEAPY = False


# =========================
# Constants
# =========================

DEFAULT_OUT_SUBDIR = "output_05_build_multi_axis_transcriptome_labels"
DEFAULT_PROCESSED_SUBDIR = "output_02_01_process_samples_data"

BASE_TABLE_CANDIDATES = [
    ("output_04_score_and_label_slides", "slide_features_scored_labeled.csv"),
    ("output_03_merge_slide_features", "merged_slide_features.csv"),
]

H5AD_CANDIDATES = [
    "03_final_pipeline_output.h5ad",
    "02_after_clustering_and_annotation.h5ad",
    "02_processed.h5ad",
    "01_after_filtering.h5ad",
    "01_loaded.h5ad",
]

MSIGDB_DBVER = "2023.1.Hs"

MIN_GENES_PER_SIGNATURE = 2
MIN_GENES_EXTERNAL = 10

MIN_TOTAL_COUNTS = 500
MIN_N_GENES = 150
MAX_PCT_MT = 35.0

STRUCTURE_ACTIVE_QUANTILE = 0.75
FUNCTION_ACTIVE_QUANTILE = 0.80
METABOLIC_ACTIVE_QUANTILE = 0.80

ACTIVE_CONFIDENCE_MIN = 0.25
N_TOP_STRUCTURE_LABELS = 3

SMOOTH_K = 10
LOCAL_Z_K = 18
MIN_CLUSTER_SPOTS = 8

CUSTOM_GSVA_MIN_SIZE = 2
EXTERNAL_GSVA_MIN_SIZE = 10
GSVA_MAX_SIZE = 5000

HIGH_CONFIDENCE_CONCORDANCE = 0.60
STRONG_MISMATCH_BEST_SCORE = 0.70
LOW_SIGNAL_BEST_SCORE = 0.60

# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# Raw spot-level structure labels are preserved, but downstream structural regions
# use a cluster-consensus layer for coherent architecture interpretation.
STRUCTURE_CONSENSUS_MIN_FRACTION = 0.50
STRUCTURE_CONSENSUS_LABEL_COL = "structure_region_label"
STRUCTURE_CONSENSUS_SMOOTHED_LABEL_COL = "structure_region_label_smoothed"


# =========================
# Signature definitions
# =========================

STRUCTURE_SIGNATURES = {
    "tumor_epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "KRT17",
        "MUC1", "ELF3", "PIGR", "TACSTD2",
    ],
    "tumor_proliferative": [
        "MKI67", "TOP2A", "STMN1", "PCNA", "MCM2", "MCM5",
        "UBE2C", "CCNB1", "CCNA2", "TYMS",
    ],
    "stromal_ecm": [
        "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2",
        "DCN", "LUM", "FAP", "ACTA2", "VIM",
    ],
    "ecm_remodeling": [
        "MMP2", "MMP9", "MMP11", "MMP14", "TIMP1", "TIMP2",
        "LOX", "POSTN", "TAGLN", "IGFBP7",
    ],
    "immune_b_plasma": [
        "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN",
        "IGHG1", "IGKC", "IGLC2", "IGHA1",
    ],
    "t_cell": [
        "CD3D", "CD3E", "CD2", "TRAC", "TRBC1", "TRBC2",
        "IL7R", "CCR7", "CD8A", "CD8B",
    ],
    "myeloid_macrophage": [
        "CD68", "CD163", "LST1", "C1QA", "C1QB", "C1QC",
        "TYROBP", "AIF1", "FCGR3A",
    ],
    "angiogenic_vascular": [
        "PECAM1", "VWF", "KDR", "FLT1", "ENG", "EMCN", "PLVAP",
        "ESAM", "CLDN5", "RAMP2",
    ],
}

FUNCTION_SIGNATURES = {
    "hypoxic_stress": [
        "CA9", "VEGFA", "SLC2A1", "ENO1", "LDHA", "BNIP3",
        "NDRG1", "ADM", "ANGPTL4", "HIF1A",
    ],
    "interferon_inflamed": [
        "STAT1", "IRF1", "IFIT1", "IFIT2", "IFIT3", "ISG15",
        "CXCL9", "CXCL10", "GBP1", "MX1",
    ],
    "antigen_presentation": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "NLRC5",
    ],
    "checkpoint_exhaustion": [
        "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2",
        "TIGIT", "TOX", "ENTPD1",
    ],
    "ecm_remodeling_function": [
        "MMP2", "MMP9", "MMP11", "MMP14", "TIMP1", "TIMP2",
        "LOX", "POSTN", "COL1A1", "COL3A1",
    ],
    "tumor_proliferative_function": [
        "MKI67", "TOP2A", "UBE2C", "MCM2", "MCM5", "CCNB1",
        "CCNA2", "E2F1", "E2F2",
    ],
    "inflammatory_tnf_nfkb": [
        "NFKB1", "NFKBIA", "TNFAIP3", "TNFAIP6", "RELA",
        "IL6", "CXCL1", "CXCL2", "CCL2",
    ],
}

METABOLIC_SIGNATURES = {
    "glycolysis": [
        "SLC2A1", "HK1", "HK2", "GPI", "PFKP", "PFKM", "PFKL",
        "ALDOA", "GAPDH", "PGK1", "ENO1", "ENO2", "PKM", "LDHA", "LDHB",
    ],
    "oxphos": [
        "NDUFA1", "NDUFA2", "NDUFS1", "NDUFS2", "SDHA", "SDHB",
        "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "COX6A1",
        "ATP5F1A", "ATP5F1B", "ATP5MC1",
    ],
    "fatty_acid_oxidation": [
        "CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ACAA2",
    ],
    "fatty_acid_synthesis": [
        "ACACA", "ACACB", "FASN", "SCD", "ELOVL1", "ELOVL6", "FADS1", "FADS2",
    ],
    "nucleotide_synthesis": [
        "RRM1", "RRM2", "TYMS", "DHODH", "CAD", "UMPS", "PPAT",
        "GMPS", "IMPDH1", "IMPDH2",
    ],
    "glutamine_metabolism": [
        "GLS", "GLS2", "GLUD1", "GLUD2", "SLC1A5", "GOT1", "GOT2",
    ],
    "proline_collagen_support": [
        "PYCR1", "PYCR2", "ALDH18A1", "P4HA1", "P4HA2", "PLOD1", "PLOD2",
    ],
    "tryptophan_kynurenine": [
        "IDO1", "TDO2", "KYNU", "KMO", "QPRT",
    ],
}

ALL_SIGNATURES = {}
ALL_SIGNATURES.update(STRUCTURE_SIGNATURES)
ALL_SIGNATURES.update(FUNCTION_SIGNATURES)
ALL_SIGNATURES.update(METABOLIC_SIGNATURES)

SIGNATURE_AXIS = {}
SIGNATURE_AXIS.update({k: "structure" for k in STRUCTURE_SIGNATURES})
SIGNATURE_AXIS.update({k: "function" for k in FUNCTION_SIGNATURES})
SIGNATURE_AXIS.update({k: "metabolism" for k in METABOLIC_SIGNATURES})

STRUCTURE_KEYS = list(STRUCTURE_SIGNATURES)
FUNCTION_KEYS = list(FUNCTION_SIGNATURES)
METABOLIC_KEYS = list(METABOLIC_SIGNATURES)


HALLMARK_KEEP = [
    "HALLMARK_HYPOXIA",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_E2F_TARGETS",
]

REACTOME_KEYWORDS = [
    "INTERFERON", "IMMUNE", "CYTOKINE", "CHEMOKINE", "HYPOXIA",
    "VEGF", "ANGIOGENESIS", "EXTRACELLULAR_MATRIX", "ECM", "COLLAGEN",
    "INTEGRIN", "EPITHELIAL", "MYELOID", "TCR", "BCR", "ANTIGEN",
    "MACROPHAGE", "NEUTROPHIL", "CELL_CYCLE", "MITOTIC",
    "GLYCOLYSIS", "OXIDATIVE_PHOSPHORYLATION", "RESPIRATORY_ELECTRON",
    "FATTY_ACID", "GLUTAMINE", "TRYPTOPHAN", "NUCLEOTIDE",
]

HALLMARK_TO_AXIS_LABEL = {
    "HALLMARK_HYPOXIA": [("function", "hypoxic_stress")],
    "HALLMARK_ANGIOGENESIS": [("structure", "angiogenic_vascular")],
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": [("function", "interferon_inflamed")],
    "HALLMARK_INFLAMMATORY_RESPONSE": [("function", "inflammatory_tnf_nfkb")],
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": [("function", "inflammatory_tnf_nfkb")],
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": [("function", "inflammatory_tnf_nfkb")],
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": [
        ("structure", "stromal_ecm"),
        ("structure", "ecm_remodeling"),
        ("function", "ecm_remodeling_function"),
    ],
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": [("metabolism", "oxphos")],
    "HALLMARK_G2M_CHECKPOINT": [
        ("structure", "tumor_proliferative"),
        ("function", "tumor_proliferative_function"),
        ("metabolism", "nucleotide_synthesis"),
    ],
    "HALLMARK_E2F_TARGETS": [
        ("structure", "tumor_proliferative"),
        ("function", "tumor_proliferative_function"),
        ("metabolism", "nucleotide_synthesis"),
    ],
    "HALLMARK_APOPTOSIS": [("function", "inflammatory_tnf_nfkb")],
}

REACTOME_RULES = [
    (["INTERFERON"], [("function", "interferon_inflamed")]),
    (["ANTIGEN", "MHC", "CROSS_PRESENTATION"], [("function", "antigen_presentation")]),
    (["TCR", "T_CELL", "T CELL"], [("structure", "t_cell")]),
    (["BCR", "B_CELL", "B CELL"], [("structure", "immune_b_plasma")]),
    (["MACROPHAGE", "MYELOID", "NEUTROPHIL"], [("structure", "myeloid_macrophage")]),
    (["COLLAGEN"], [("structure", "stromal_ecm"), ("function", "ecm_remodeling_function")]),
    (["EXTRACELLULAR_MATRIX", "ECM", "INTEGRIN"], [
        ("structure", "stromal_ecm"),
        ("structure", "ecm_remodeling"),
        ("function", "ecm_remodeling_function"),
    ]),
    (["VEGF", "ANGIOGENESIS"], [("structure", "angiogenic_vascular")]),
    (["CELL_CYCLE", "CELL CYCLE", "MITOTIC", "G2M"], [
        ("structure", "tumor_proliferative"),
        ("function", "tumor_proliferative_function"),
        ("metabolism", "nucleotide_synthesis"),
    ]),
    (["HYPOXIA"], [("function", "hypoxic_stress"), ("metabolism", "glycolysis")]),
    (["GLYCOLYSIS"], [("metabolism", "glycolysis")]),
    (["OXIDATIVE_PHOSPHORYLATION", "RESPIRATORY_ELECTRON"], [("metabolism", "oxphos")]),
    (["FATTY_ACID", "FATTY ACID"], [("metabolism", "fatty_acid_oxidation")]),
    (["GLUTAMINE"], [("metabolism", "glutamine_metabolism")]),
    (["TRYPTOPHAN", "KYNURENINE"], [("metabolism", "tryptophan_kynurenine")]),
    (["NUCLEOTIDE", "PURINE", "PYRIMIDINE"], [("metabolism", "nucleotide_synthesis")]),
]

FALLBACK_HALLMARK_GENESETS = {
    "HALLMARK_HYPOXIA": FUNCTION_SIGNATURES["hypoxic_stress"] + ["NDRG1", "BNIP3"],
    "HALLMARK_ANGIOGENESIS": STRUCTURE_SIGNATURES["angiogenic_vascular"] + ["VEGFA"],
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": FUNCTION_SIGNATURES["interferon_inflamed"],
    "HALLMARK_INFLAMMATORY_RESPONSE": FUNCTION_SIGNATURES["inflammatory_tnf_nfkb"] + ["CXCL10"],
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": STRUCTURE_SIGNATURES["stromal_ecm"] + STRUCTURE_SIGNATURES["ecm_remodeling"],
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": METABOLIC_SIGNATURES["oxphos"],
    "HALLMARK_G2M_CHECKPOINT": STRUCTURE_SIGNATURES["tumor_proliferative"],
    "HALLMARK_E2F_TARGETS": STRUCTURE_SIGNATURES["tumor_proliferative"],
}

FALLBACK_REACTOME_GENESETS = {
    "REACTOME_INTERFERON_SIGNALING": FUNCTION_SIGNATURES["interferon_inflamed"],
    "REACTOME_ANTIGEN_PROCESSING_CROSS_PRESENTATION": FUNCTION_SIGNATURES["antigen_presentation"],
    "REACTOME_TCR_SIGNALING": STRUCTURE_SIGNATURES["t_cell"],
    "REACTOME_BCR_SIGNALING": STRUCTURE_SIGNATURES["immune_b_plasma"],
    "REACTOME_COLLAGEN_FORMATION": STRUCTURE_SIGNATURES["stromal_ecm"],
    "REACTOME_INTEGRIN_ECM_INTERACTION": STRUCTURE_SIGNATURES["ecm_remodeling"],
    "REACTOME_VEGFA_VEGFR2_PATHWAY": STRUCTURE_SIGNATURES["angiogenic_vascular"],
    "REACTOME_CELL_CYCLE_MITOTIC": STRUCTURE_SIGNATURES["tumor_proliferative"],
    "REACTOME_GLYCOLYSIS": METABOLIC_SIGNATURES["glycolysis"],
    "REACTOME_RESPIRATORY_ELECTRON_TRANSPORT": METABOLIC_SIGNATURES["oxphos"],
    "REACTOME_FATTY_ACID_METABOLISM": METABOLIC_SIGNATURES["fatty_acid_oxidation"],
    "REACTOME_GLUTAMINE_METABOLISM": METABOLIC_SIGNATURES["glutamine_metabolism"],
    "REACTOME_TRYPTOPHAN_CATABOLISM": METABOLIC_SIGNATURES["tryptophan_kynurenine"],
}

EXPECTED_STATE_PROFILES = {
    "hypoxic_tumor": {
        "glycolysis": 1.0,
        "oxphos": -1.0,
        "fatty_acid_oxidation": -0.3,
        "fatty_acid_synthesis": 0.3,
        "nucleotide_synthesis": 0.4,
        "glutamine_metabolism": 0.3,
        "proline_collagen_support": 0.0,
        "tryptophan_kynurenine": 0.0,
    },
    "proliferative_tumor": {
        "glycolysis": 0.7,
        "oxphos": 0.2,
        "fatty_acid_oxidation": 0.0,
        "fatty_acid_synthesis": 0.6,
        "nucleotide_synthesis": 1.0,
        "glutamine_metabolism": 0.7,
        "proline_collagen_support": 0.0,
        "tryptophan_kynurenine": 0.0,
    },
    "stromal_ecm": {
        "glycolysis": 0.1,
        "oxphos": 0.3,
        "fatty_acid_oxidation": 0.2,
        "fatty_acid_synthesis": 0.3,
        "nucleotide_synthesis": -0.4,
        "glutamine_metabolism": 0.3,
        "proline_collagen_support": 1.0,
        "tryptophan_kynurenine": 0.0,
    },
    "angiogenic_vascular": {
        "glycolysis": 0.5,
        "oxphos": 0.2,
        "fatty_acid_oxidation": 0.0,
        "fatty_acid_synthesis": 0.3,
        "nucleotide_synthesis": 0.2,
        "glutamine_metabolism": 0.2,
        "proline_collagen_support": 0.1,
        "tryptophan_kynurenine": 0.0,
    },
    "inflamed_immune": {
        "glycolysis": 0.5,
        "oxphos": 0.2,
        "fatty_acid_oxidation": 0.0,
        "fatty_acid_synthesis": 0.0,
        "nucleotide_synthesis": 0.1,
        "glutamine_metabolism": 0.3,
        "proline_collagen_support": 0.0,
        "tryptophan_kynurenine": 0.8,
    },
    "quiescent_or_normal": {
        "glycolysis": -0.2,
        "oxphos": 0.5,
        "fatty_acid_oxidation": 0.2,
        "fatty_acid_synthesis": -0.1,
        "nucleotide_synthesis": -0.5,
        "glutamine_metabolism": 0.0,
        "proline_collagen_support": 0.0,
        "tryptophan_kynurenine": 0.0,
    },
}

STRUCTURE_TO_EXPECTED_STATE = {
    "tumor_epithelial": "proliferative_tumor",
    "tumor_proliferative": "proliferative_tumor",
    "stromal_ecm": "stromal_ecm",
    "ecm_remodeling": "stromal_ecm",
    "angiogenic_vascular": "angiogenic_vascular",
    "t_cell": "inflamed_immune",
    "immune_b_plasma": "inflamed_immune",
    "myeloid_macrophage": "inflamed_immune",
}


# =========================
# Data containers
# =========================

@dataclass
class ExternalLibraries:
    hallmark: dict[str, list[str]]
    reactome: dict[str, list[str]]
    status: str
    source: str


@dataclass
class SampleResult:
    status: str
    sample_id: str
    slide_row: dict[str, Any]
    status_row: dict[str, Any]
    support: pd.DataFrame


# =========================
# General helpers
# =========================

def ensure_dir(path: Path) -> None:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = np.nan) -> float:
    """Convert a value to float while returning a default on missing or invalid input."""
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean_label(text: Any) -> str:
    """Normalize free text labels into lowercase underscore labels."""
    return str(text).strip().lower().replace(" ", "_").replace("-", "_")


def percentile_series(values: pd.Series) -> pd.Series:
    """Convert one numeric series into percentile ranks with constant value handling."""
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=s.index)
    return s.rank(method="average", pct=True)


def percentile_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply percentile ranking column wise to a dataframe."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = percentile_series(df[col])
    return out


def normalize_01(values: Any) -> np.ndarray:
    """Normalize a numeric vector to the 0 to 1 range while handling constants."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros(len(arr), dtype=float)
    if finite.sum() == 0:
        return out
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    if hi == lo:
        out[finite] = 0.5
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def confidence_from_score(values: Any) -> np.ndarray:
    """Convert raw scores into a simple 0 to 1 confidence scale."""
    return normalize_01(values)


def summarize_vector(values: Any, prefix: str) -> dict[str, float]:
    """Summarize one numeric vector into common slide level statistics."""
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return {
        f"{prefix}_mean": safe_float(s.mean()),
        f"{prefix}_median": safe_float(s.median()),
        f"{prefix}_std": safe_float(s.std(ddof=0)),
        f"{prefix}_q75": safe_float(s.quantile(0.75)),
        f"{prefix}_q90": safe_float(s.quantile(0.90)),
        f"{prefix}_max": safe_float(s.max()),
    }


def sparse_or_dense_to_array(x: Any) -> np.ndarray:
    """Convert sparse or dense matrix input into a NumPy array."""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def get_spatial_coordinates(adata: Any) -> np.ndarray | None:
    """Extract spatial coordinates from AnnData using common Visium column conventions."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].astype(float)

    pairs = [
        ("spatial_x", "spatial_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
    ]

    for x_col, y_col in pairs:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return adata.obs[[x_col, y_col]].astype(float).to_numpy()

    return None


def choose_cluster_column(obs: pd.DataFrame) -> str | None:
    """Choose the best available clustering column from AnnData observations."""
    for col in ["leiden", "cluster", "clusters", "cluster_id"]:
        if col in obs.columns:
            return col
    return None


def find_h5ad_for_sample(processed_root: Path, sample_id: str) -> Path | None:
    """Find the best available processed AnnData file for one sample."""
    sample_dir = processed_root / sample_id
    adata_dir = sample_dir / "adata"

    for name in H5AD_CANDIDATES:
        path = adata_dir / name
        if path.exists():
            return path

    return None


def read_base_table(output_root: Path) -> pd.DataFrame:
    """Read the richest upstream slide level table available before Step 05."""
    for subdir, filename in BASE_TABLE_CANDIDATES:
        path = output_root / subdir / filename
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


# =========================
# Gene set helpers
# =========================

def uppercase_gene_sets(gene_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Normalize gene set names to uppercase gene symbols."""
    out = {}
    for key, genes in gene_sets.items():
        out[key] = sorted({str(g).upper() for g in genes if str(g).strip()})
    return out


def keyword_filter_terms(gene_sets: dict[str, list[str]], keywords: list[str]) -> dict[str, list[str]]:
    """Filter external gene set terms by keyword matches."""
    filtered = {}
    for term, genes in gene_sets.items():
        term_up = str(term).upper()
        if any(k in term_up for k in keywords):
            filtered[term] = genes
    return filtered


def cap_gene_sets(gene_sets: dict[str, list[str]], max_terms: int) -> dict[str, list[str]]:
    """Limit an external gene set dictionary to a maximum number of terms."""
    if max_terms is None or max_terms <= 0:
        return gene_sets
    return dict(list(gene_sets.items())[:max_terms])


def load_external_libraries(max_reactome_terms: int) -> ExternalLibraries:
    """Load Hallmark and Reactome libraries, falling back to curated local gene sets when needed."""
    if not HAS_GSEAPY or Msigdb is None:
        return ExternalLibraries(
            hallmark=uppercase_gene_sets(FALLBACK_HALLMARK_GENESETS),
            reactome=cap_gene_sets(uppercase_gene_sets(FALLBACK_REACTOME_GENESETS), max_reactome_terms),
            status="gseapy_not_available_using_fallback",
            source="fallback_curated_lite",
        )

    try:
        msig = Msigdb()
        hallmark = msig.get_gmt(category="h.all", dbver=MSIGDB_DBVER)
        reactome = msig.get_gmt(category="c2.cp.reactome", dbver=MSIGDB_DBVER)

        hallmark = {k: v for k, v in hallmark.items() if k in HALLMARK_KEEP}
        reactome = keyword_filter_terms(reactome, REACTOME_KEYWORDS)
        reactome = cap_gene_sets(reactome, max_reactome_terms)

        return ExternalLibraries(
            hallmark=uppercase_gene_sets(hallmark),
            reactome=uppercase_gene_sets(reactome),
            status="ok",
            source=f"msigdb_{MSIGDB_DBVER}",
        )

    except Exception as error:
        return ExternalLibraries(
            hallmark=uppercase_gene_sets(FALLBACK_HALLMARK_GENESETS),
            reactome=cap_gene_sets(uppercase_gene_sets(FALLBACK_REACTOME_GENESETS), max_reactome_terms),
            status=f"msigdb_failed_using_fallback: {type(error).__name__}: {error}",
            source="fallback_curated_lite",
        )


def filter_gene_sets_to_present(
    gene_sets: dict[str, list[str]],
    present_genes: set[str],
    min_genes: int,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Keep gene sets with enough genes present in the current sample."""
    filtered = {}
    rows = []

    for name, genes in gene_sets.items():
        genes_up = [str(g).upper() for g in genes]
        present = [g for g in genes_up if g in present_genes]
        passed = len(present) >= min_genes

        if passed:
            filtered[name] = present

        rows.append({
            "signature": name,
            "n_defined": int(len(genes_up)),
            "n_present": int(len(present)),
            "passed": int(passed),
            "present_genes": ";".join(present),
            "missing_genes": ";".join([g for g in genes_up if g not in present_genes]),
        })

    return filtered, pd.DataFrame(rows)


# =========================
# Expression helpers
# =========================

def get_expression_source(adata: Any) -> Any:
    """Return raw expression when available, otherwise return the active AnnData object."""
    return adata.raw.to_adata() if adata.raw is not None else adata.copy()


def get_present_gene_map(source: Any) -> dict[str, list[int]]:
    """Build a mapping from uppercase gene names to expression matrix column indices."""
    names = [str(g).upper() for g in source.var_names.astype(str)]
    gene_map: dict[str, list[int]] = {}
    for i, g in enumerate(names):
        gene_map.setdefault(g, []).append(i)
    return gene_map


def make_expr_df_upper(source: Any, genes: list[str] | set[str] | None = None) -> pd.DataFrame:
    """Build a spot by gene expression dataframe using uppercase gene names."""
    gene_map = get_present_gene_map(source)

    if genes is None:
        wanted = sorted(gene_map.keys())
    else:
        wanted = sorted({str(g).upper() for g in genes if str(g).upper() in gene_map})

    if not wanted:
        return pd.DataFrame(index=source.obs_names.astype(str))

    cols = []
    arrays = []

    X = source.X

    for gene in wanted:
        idxs = gene_map[gene]
        sub = X[:, idxs]
        arr = sparse_or_dense_to_array(sub).astype(np.float32)

        if arr.ndim == 1:
            vals = arr
        elif arr.shape[1] == 1:
            vals = arr[:, 0]
        else:
            vals = np.nanmean(arr, axis=1)

        cols.append(gene)
        arrays.append(vals)

    mat = np.vstack(arrays).T if arrays else np.zeros((source.n_obs, 0), dtype=np.float32)
    return pd.DataFrame(mat, index=source.obs_names.astype(str), columns=cols)


def apply_spot_qc(adata: Any, expr_index: pd.Index) -> pd.Series:
    """Filter spots using total count, gene count, and mitochondrial percentage thresholds."""
    obs = adata.obs.copy()
    keep = pd.Series(True, index=expr_index)

    if "total_counts" in obs.columns:
        s = pd.to_numeric(obs["total_counts"], errors="coerce").reindex(expr_index)
        keep &= s.fillna(MIN_TOTAL_COUNTS) >= MIN_TOTAL_COUNTS

    if "n_genes_by_counts" in obs.columns:
        s = pd.to_numeric(obs["n_genes_by_counts"], errors="coerce").reindex(expr_index)
        keep &= s.fillna(MIN_N_GENES) >= MIN_N_GENES

    if "pct_counts_mt" in obs.columns:
        s = pd.to_numeric(obs["pct_counts_mt"], errors="coerce").reindex(expr_index)
        keep &= s.fillna(0.0) <= MAX_PCT_MT

    return keep.fillna(False)


# =========================
# Scoring helpers
# =========================

def compute_mean_signature_scores(
    expr_df: pd.DataFrame,
    filtered: dict[str, list[str]],
) -> pd.DataFrame:
    """Score signatures by averaging expression across present genes."""
    out = pd.DataFrame(index=expr_df.index)

    for name, genes in filtered.items():
        present = [g for g in genes if g in expr_df.columns]
        if present:
            out[name] = expr_df[present].mean(axis=1)
        else:
            out[name] = np.nan

    return out


def compute_rank_signature_scores(
    expr_df: pd.DataFrame,
    filtered: dict[str, list[str]],
) -> pd.DataFrame:
    """Score signatures using percentile ranked gene expression."""
    out = pd.DataFrame(index=expr_df.index)

    if expr_df.empty:
        return out

    gene_percentiles = expr_df.rank(axis=0, pct=True, method="average")

    for name, genes in filtered.items():
        present = [g for g in genes if g in gene_percentiles.columns]
        if present:
            out[name] = gene_percentiles[present].mean(axis=1)
        else:
            out[name] = np.nan

    return out


def prepare_adata_for_ucell(source: Any) -> Any:
    """Prepare an AnnData copy with uppercase gene names for UCell scoring."""
    adata_uc = source.copy()
    adata_uc.var_names = pd.Index([str(g).upper() for g in adata_uc.var_names])
    adata_uc.var_names_make_unique()
    return adata_uc


def get_ucell_score_column(obs: pd.DataFrame, signature_name: str) -> str | None:
    """Find the UCell score column corresponding to a signature."""
    candidates = [
        signature_name,
        f"{signature_name}_ucell",
        f"ucell_{signature_name}",
        f"{signature_name}_UCell",
        f"UCell_{signature_name}",
    ]

    for c in candidates:
        if c in obs.columns:
            return c

    sig_low = signature_name.lower()
    for c in obs.columns:
        c_low = str(c).lower()
        if sig_low in c_low and ("ucell" in c_low or "score" in c_low):
            return c

    return None


def compute_ucell_scores(
    source: Any,
    filtered: dict[str, list[str]],
    enabled: bool,
    n_jobs: int,
) -> tuple[pd.DataFrame, str]:
    """Compute optional UCell scores for filtered signatures."""
    out = pd.DataFrame(index=source.obs_names.astype(str))

    if not enabled:
        return out, "skipped"

    if not HAS_UCELL:
        return out, "ucell_not_available"

    if not filtered:
        return out, "no_signatures"

    try:
        adata_uc = prepare_adata_for_ucell(source)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uc.compute_ucell_scores(
                adata_uc,
                signatures=filtered,
                n_jobs=n_jobs,
            )

        for sig in filtered:
            col = get_ucell_score_column(adata_uc.obs, sig)
            if col is not None:
                out[sig] = pd.to_numeric(adata_uc.obs[col], errors="coerce").reindex(out.index).values

        if out.shape[1] == 0:
            return out, "no_ucell_columns_found"

        return out, "ok"

    except Exception as error:
        return out, f"failed: {type(error).__name__}: {error}"


def parse_gseapy_scores(gsva_obj: Any, sample_names: list[str]) -> pd.DataFrame:
    """Parse GSEApy GSVA output into a spot by term score table."""
    res = gsva_obj.res2d.copy()

    if res.empty:
        return pd.DataFrame(index=sample_names)

    cols_lower = {str(c).lower(): c for c in res.columns}

    if "name" in cols_lower and "term" in cols_lower:
        sample_col = cols_lower["name"]
        term_col = cols_lower["term"]

        score_col = None
        for candidate in ["es", "nes", "score"]:
            if candidate in cols_lower:
                score_col = cols_lower[candidate]
                break

        if score_col is None:
            numeric = res.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError("No numeric GSVA score column found")
            score_col = numeric[0]

        out = res.pivot(index=sample_col, columns=term_col, values=score_col)
        out.index = out.index.astype(str)
        return out.reindex(sample_names)

    first_col = res.columns[0]
    if str(first_col).lower() in ["term", "pathway", "name"]:
        res = res.set_index(first_col)

    res.index = res.index.astype(str)

    if set(sample_names).issubset(set(res.index)):
        return res.reindex(sample_names)

    overlap = [c for c in res.columns.astype(str) if c in sample_names]
    if overlap:
        return res[overlap].T.reindex(sample_names)

    raise ValueError("Could not parse GSVA output format")


def compute_gsva_scores(
    expr_df: pd.DataFrame,
    filtered: dict[str, list[str]],
    enabled: bool,
    min_size: int,
    max_size: int,
) -> tuple[pd.DataFrame, str]:
    """Compute optional GSVA scores for custom or external gene sets."""
    out = pd.DataFrame(index=expr_df.index)

    if not enabled:
        return out, "skipped"

    if not HAS_GSEAPY:
        return out, "gseapy_not_available"

    if not filtered:
        return out, "no_gene_sets"

    if expr_df.shape[0] < 2:
        return out, "too_few_spots"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gp.gsva(
                data=expr_df.T,
                gene_sets=filtered,
                outdir=None,
                verbose=False,
                min_size=min_size,
                max_size=max_size,
            )

        parsed = parse_gseapy_scores(res, list(expr_df.index))
        if parsed.shape[1] == 0:
            return parsed, "empty_output"

        return parsed, "ok"

    except Exception as error:
        return out, f"failed: {type(error).__name__}: {error}"


def map_reactome_term(term: str) -> list[tuple[str, str]]:
    """Map a Reactome term name onto one or more pipeline axis labels."""
    term_up = str(term).upper().replace(" ", "_")
    hits: list[tuple[str, str]] = []

    for keywords, mappings in REACTOME_RULES:
        if any(k.replace(" ", "_") in term_up for k in keywords):
            hits.extend(mappings)

    unique = []
    seen = set()
    for m in hits:
        if m not in seen:
            unique.append(m)
            seen.add(m)

    return unique


def map_external_scores_to_axis_labels(
    score_df: pd.DataFrame,
    mapping: dict[str, list[tuple[str, str]]] | None,
    is_reactome: bool = False,
) -> dict[str, pd.DataFrame]:
    """Collapse external pathway scores onto structure, function, and metabolism axes."""
    axis_frames = {
        "structure": pd.DataFrame(index=score_df.index),
        "function": pd.DataFrame(index=score_df.index),
        "metabolism": pd.DataFrame(index=score_df.index),
    }

    collected: dict[tuple[str, str], list[pd.Series]] = {}

    for term in score_df.columns:
        if is_reactome:
            maps = map_reactome_term(term)
        else:
            maps = mapping.get(term, []) if mapping is not None else []

        for axis, label in maps:
            if axis not in axis_frames:
                continue
            collected.setdefault((axis, label), []).append(pd.to_numeric(score_df[term], errors="coerce"))

    for (axis, label), series_list in collected.items():
        mat = pd.concat(series_list, axis=1)
        axis_frames[axis][label] = mat.mean(axis=1)

    return axis_frames


# =========================
# Smoothing and fusion
# =========================

def compute_spatial_smoothed_scores(score_df: pd.DataFrame, coords: np.ndarray | None, k: int) -> pd.DataFrame:
    """Smooth score vectors across spatial neighbors."""
    if score_df.empty or coords is None or len(coords) != len(score_df):
        return score_df.copy()

    if len(score_df) <= 2:
        return score_df.copy()

    kk = min(k + 1, len(score_df))

    try:
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]

        values = score_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        smoothed = np.nanmean(values[idx], axis=1)
        return pd.DataFrame(smoothed, index=score_df.index, columns=score_df.columns)

    except Exception:
        return score_df.copy()


def compute_local_z_scores(score_df: pd.DataFrame, coords: np.ndarray | None, k: int) -> pd.DataFrame:
    """Compute local spatial z scores relative to neighboring spots."""
    if score_df.empty or coords is None or len(coords) != len(score_df):
        return pd.DataFrame(index=score_df.index, columns=score_df.columns, data=0.0)

    if len(score_df) <= 2:
        return pd.DataFrame(index=score_df.index, columns=score_df.columns, data=0.0)

    kk = min(k + 1, len(score_df))

    try:
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]

        values = score_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        local_mean = np.nanmean(values[idx], axis=1)
        local_std = np.nanstd(values[idx], axis=1)
        local_std[local_std == 0] = np.nan

        z = (values - local_mean) / local_std
        z[~np.isfinite(z)] = 0.0
        return pd.DataFrame(z, index=score_df.index, columns=score_df.columns)

    except Exception:
        return pd.DataFrame(index=score_df.index, columns=score_df.columns, data=0.0)


def compute_cluster_smoothed_scores(score_df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Smooth score vectors within cluster groups when cluster labels are available."""
    if score_df.empty:
        return score_df.copy()

    col = choose_cluster_column(obs)
    if col is None:
        return score_df.copy()

    clusters = obs[col].astype(str).reindex(score_df.index).fillna("NA")
    cluster_counts = clusters.value_counts()
    counts = clusters.map(cluster_counts)

    out = pd.DataFrame(index=score_df.index)

    for score_col in score_df.columns:
        vals = pd.to_numeric(score_df[score_col], errors="coerce").fillna(0.0)
        means = vals.groupby(clusters).transform("mean")
        out[score_col] = np.where(counts >= MIN_CLUSTER_SPOTS, means, vals)

    return out


def combine_method_scores(
    method_frames: dict[str, pd.DataFrame],
    allowed_labels: list[str],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Fuse score matrices from multiple methods into one axis specific score table."""
    out = pd.DataFrame(index=next(iter(method_frames.values())).index) if method_frames else pd.DataFrame()

    for label in allowed_labels:
        vals = []
        wts = []

        for method_name, df in method_frames.items():
            if df is None or df.empty or label not in df.columns:
                continue

            norm = percentile_series(df[label])
            vals.append(norm)
            wts.append(float(weights.get(method_name, 1.0)))

        if vals:
            mat = pd.concat(vals, axis=1)
            weights_arr = np.asarray(wts, dtype=float)
            out[label] = np.average(mat.fillna(0.5).to_numpy(dtype=float), axis=1, weights=weights_arr)

    return out


def refine_axis_scores(base_scores: pd.DataFrame, coords: np.ndarray | None, obs: pd.DataFrame) -> pd.DataFrame:
    """Combine base, spatial, cluster, and local z versions of axis scores."""
    if base_scores.empty:
        return base_scores.copy()

    frames = {
        "base": base_scores,
        "spatial": compute_spatial_smoothed_scores(base_scores, coords, SMOOTH_K),
        "cluster": compute_cluster_smoothed_scores(base_scores, obs),
        "local_z": compute_local_z_scores(base_scores, coords, LOCAL_Z_K),
    }

    weights = {
        "base": 1.0,
        "spatial": 0.70,
        "cluster": 0.55,
        "local_z": 0.35,
    }

    out = pd.DataFrame(index=base_scores.index)

    for label in base_scores.columns:
        vals = []
        wts = []

        for name, df in frames.items():
            if label not in df.columns:
                continue
            vals.append(percentile_series(df[label]))
            wts.append(weights[name])

        mat = pd.concat(vals, axis=1)
        out[label] = np.average(mat.fillna(0.5).to_numpy(dtype=float), axis=1, weights=np.asarray(wts))

    return out


# =========================
# Label construction
# =========================

def build_structure_labels(structure_scores: pd.DataFrame) -> pd.DataFrame:
    """Assign dominant and runner up structural labels from structure scores."""
    idx = structure_scores.index
    vals = structure_scores.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if vals.empty:
        return pd.DataFrame(index=idx)

    sorted_cols = np.argsort(-vals.to_numpy(dtype=float), axis=1)
    col_names = list(vals.columns)

    top_labels = []
    top_scores = []

    for j in range(N_TOP_STRUCTURE_LABELS):
        if vals.shape[1] > j:
            labs = [col_names[i] for i in sorted_cols[:, j]]
            scores = vals.to_numpy()[np.arange(len(idx)), sorted_cols[:, j]]
        else:
            labs = [""] * len(idx)
            scores = np.zeros(len(idx))

        top_labels.append(labs)
        top_scores.append(scores)

    margin = top_scores[0] - top_scores[1] if len(top_scores) > 1 else top_scores[0]
    conf = confidence_from_score(top_scores[0]) * normalize_01(margin)

    out = pd.DataFrame({
        "structure_dominant_label": top_labels[0],
        "structure_top1": top_labels[0],
        "structure_top2": top_labels[1] if len(top_labels) > 1 else "",
        "structure_top3": top_labels[2] if len(top_labels) > 2 else "",
        "structure_top1_score": top_scores[0],
        "structure_top2_score": top_scores[1] if len(top_scores) > 1 else 0.0,
        "structure_top3_score": top_scores[2] if len(top_scores) > 2 else 0.0,
        "structure_margin_top1_top2": margin,
        "structure_confidence": conf,
    }, index=idx)

    active_masks = {}
    for col in vals.columns:
        cutoff = vals[col].quantile(STRUCTURE_ACTIVE_QUANTILE)
        active_masks[f"structure_overlap__{col}_active"] = (
            (vals[col] >= cutoff) &
            (confidence_from_score(vals[col].values) >= ACTIVE_CONFIDENCE_MIN)
        ).astype(int).values

    return pd.concat([out, pd.DataFrame(active_masks, index=idx)], axis=1)



# STRUCTURE_REGION_CONSENSUS_PATCH_V1
# =========================
# Structure region consensus
# =========================

def add_structure_region_consensus(
    labels: pd.DataFrame,
    obs: pd.DataFrame,
    raw_label_col: str = "structure_dominant_label",
    confidence_col: str = "structure_confidence",
) -> pd.DataFrame:
    """
    Preserve raw spot-level structure calls while adding a coherent structural
    region label based on the dominant label within each Leiden or cluster group.

    The raw structure label remains available as structure_dominant_label_raw.
    The region label is intended for overlays, accessibility, motif features,
    and other region-level summaries.
    """
    out = labels.copy()

    if raw_label_col not in out.columns:
        out["structure_dominant_label_raw"] = "unknown"
        out["structure_confidence_raw"] = np.nan
        out["structure_consensus_group"] = "unassigned"
        out["structure_region_label"] = "unknown"
        out["structure_region_label_smoothed"] = "unknown"
        out["structure_region_consensus_fraction"] = np.nan
        out["structure_region_status"] = "missing_raw_structure_label"
        out["structure_region_source"] = "none"
        return out

    out["structure_dominant_label_raw"] = out[raw_label_col].astype(str)

    if confidence_col in out.columns:
        out["structure_confidence_raw"] = pd.to_numeric(out[confidence_col], errors="coerce")
    else:
        out["structure_confidence_raw"] = np.nan

    group_col = choose_cluster_column(obs)

    if group_col is not None:
        groups = obs[group_col].astype(str).reindex(out.index).fillna("unassigned")
        source = f"{group_col}_consensus"
    else:
        groups = pd.Series(out.index.astype(str), index=out.index)
        source = "spot_fallback"

    out["structure_consensus_group"] = groups.astype(str).values

    region_label = pd.Series("unknown", index=out.index, dtype=object)
    region_fraction = pd.Series(np.nan, index=out.index, dtype=float)
    region_status = pd.Series("unknown", index=out.index, dtype=object)

    for group_value, sub in out.groupby("structure_consensus_group", dropna=False):
        idx = sub.index
        raw_labels = sub["structure_dominant_label_raw"].astype(str)

        if confidence_col in sub.columns:
            weights = pd.to_numeric(sub[confidence_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            if float(weights.sum()) <= 0:
                weights = pd.Series(1.0, index=sub.index)
        else:
            weights = pd.Series(1.0, index=sub.index)

        weighted_counts: dict[str, float] = {}
        for label_value, weight_value in zip(raw_labels, weights):
            weighted_counts[label_value] = weighted_counts.get(label_value, 0.0) + float(weight_value)

        if not weighted_counts:
            chosen_label = "unknown"
            consensus_fraction = np.nan
            status = "empty_group"
        else:
            chosen_label = max(weighted_counts, key=weighted_counts.get)
            total_weight = sum(weighted_counts.values())
            consensus_fraction = weighted_counts[chosen_label] / total_weight if total_weight > 0 else np.nan
            status = "ok" if consensus_fraction >= STRUCTURE_CONSENSUS_MIN_FRACTION else "mixed_low_consensus"

        region_label.loc[idx] = chosen_label
        region_fraction.loc[idx] = consensus_fraction
        region_status.loc[idx] = status

    out["structure_region_label"] = region_label.astype(str)
    out["structure_region_label_smoothed"] = out["structure_region_label"]
    out["structure_region_consensus_fraction"] = region_fraction.astype(float)
    out["structure_region_status"] = region_status.astype(str)
    out["structure_region_source"] = source

    return out


def build_multilabel_axis_labels(scores: pd.DataFrame, axis: str, quantile: float) -> pd.DataFrame:
    """Assign active multilabel function or metabolism calls using score thresholds."""
    idx = scores.index
    vals = scores.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    out = pd.DataFrame(index=idx)
    active_cols = []

    for col in vals.columns:
        cutoff = vals[col].quantile(quantile)
        conf = confidence_from_score(vals[col].values)
        active = ((vals[col] >= cutoff) & (conf >= ACTIVE_CONFIDENCE_MIN)).astype(int)

        active_col = f"{axis}__{col}_active"
        conf_col = f"{axis}__{col}_confidence"

        out[active_col] = active.values
        out[conf_col] = conf
        active_cols.append((col, active_col))

    active_labels = []
    for spot_id, row in out.iterrows():
        labels = [label for label, active_col in active_cols if int(row.get(active_col, 0)) == 1]
        active_labels.append(";".join(labels))

    out[f"{axis}_active_labels"] = active_labels
    out[f"{axis}_n_active_labels"] = [len(x.split(";")) if x else 0 for x in active_labels]

    if not vals.empty:
        best_col = vals.idxmax(axis=1)
        best_score = vals.max(axis=1)
        out[f"{axis}_dominant_label"] = best_col.values
        out[f"{axis}_dominant_score"] = best_score.values

    return out


def infer_expected_metabolic_state(structure_label: str) -> str:
    """Map a structure label to the expected metabolic state profile."""
    return STRUCTURE_TO_EXPECTED_STATE.get(str(structure_label), "quiescent_or_normal")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between observed and expected metabolic profiles."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan

    a = a[mask]
    b = b[mask]

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(np.dot(a, b) / (na * nb))


def similarity_to_0_1(x: float) -> float:
    """Rescale cosine similarity from negative to positive range into 0 to 1."""
    if pd.isna(x):
        return np.nan
    return float((x + 1.0) / 2.0)


def build_metabolic_concordance(structure_labels: pd.DataFrame, metabolic_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare observed metabolic scores with expected metabolic state profiles."""
    idx = metabolic_scores.index
    vals = metabolic_scores.reindex(columns=METABOLIC_KEYS).apply(pd.to_numeric, errors="coerce")

    z = pd.DataFrame(index=idx)
    for col in vals.columns:
        s = vals[col]
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            z[col] = 0.0
        else:
            z[col] = (s - s.mean()) / std

    rows = []

    for spot_id in idx:
        structure_label = str(structure_labels.loc[spot_id, "structure_dominant_label"]) if "structure_dominant_label" in structure_labels.columns else "unknown"
        expected_state = infer_expected_metabolic_state(structure_label)
        observed = z.loc[spot_id, METABOLIC_KEYS].to_numpy(dtype=float)

        state_scores = {}
        for state_name, profile in EXPECTED_STATE_PROFILES.items():
            expected_vec = np.array([profile.get(k, 0.0) for k in METABOLIC_KEYS], dtype=float)
            state_scores[state_name] = cosine_similarity(observed, expected_vec)

        best_state = max(
            state_scores,
            key=lambda k: -999 if pd.isna(state_scores[k]) else state_scores[k],
        )

        expected_raw = state_scores.get(expected_state, np.nan)
        best_raw = state_scores.get(best_state, np.nan)
        expected_score = similarity_to_0_1(expected_raw)
        best_score = similarity_to_0_1(best_raw)
        agreement = int(expected_state == best_state)

        rows.append({
            "expected_metabolic_state": expected_state,
            "metabolic_label_concordance_raw": expected_raw,
            "metabolic_label_concordance_score": expected_score,
            "metabolic_best_matching_state": best_state,
            "metabolic_best_matching_state_raw": best_raw,
            "metabolic_best_matching_state_score": best_score,
            "metabolic_state_agreement": agreement,
        })

    out = pd.DataFrame(rows, index=idx)

    out["metabolic_stratification_group"] = "intermediate"

    high_conf = (
        (out["metabolic_state_agreement"] == 1) &
        (pd.to_numeric(out["metabolic_label_concordance_score"], errors="coerce") >= HIGH_CONFIDENCE_CONCORDANCE)
    )

    mismatch = (
        (out["metabolic_state_agreement"] == 0) &
        (pd.to_numeric(out["metabolic_best_matching_state_score"], errors="coerce") >= STRONG_MISMATCH_BEST_SCORE)
    )

    low_signal = pd.to_numeric(out["metabolic_best_matching_state_score"], errors="coerce") < LOW_SIGNAL_BEST_SCORE

    out.loc[high_conf, "metabolic_stratification_group"] = "high_confidence"
    out.loc[mismatch, "metabolic_stratification_group"] = "mismatch_but_strong"
    out.loc[low_signal, "metabolic_stratification_group"] = "low_signal"

    return out


def method_best_structure_calls(method_frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Find each scoring method's best structural call per spot."""
    calls = {}

    for method, df in method_frames.items():
        if df is None or df.empty:
            continue

        structure_cols = [c for c in STRUCTURE_KEYS if c in df.columns]
        if not structure_cols:
            continue

        vals = df[structure_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        calls[method] = vals.idxmax(axis=1)

    return calls


def build_method_agreement(
    method_frames: dict[str, pd.DataFrame],
    structure_labels: pd.DataFrame,
    metabolic_concordance: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize method agreement and confidence for structural labels."""
    idx = structure_labels.index
    calls = method_best_structure_calls(method_frames)

    rows = []

    for spot_id in idx:
        final_label = structure_labels.loc[spot_id, "structure_dominant_label"]
        method_labels = {}
        agree = []

        for method, series in calls.items():
            label = series.reindex(idx).loc[spot_id]
            method_labels[method] = label
            agree.append(int(label == final_label))

        method_count = len(agree)
        agreement_fraction = float(np.mean(agree)) if agree else np.nan

        structure_conf = safe_float(structure_labels.loc[spot_id, "structure_confidence"])
        metab_best = safe_float(metabolic_concordance.loc[spot_id, "metabolic_best_matching_state_score"])
        metab_conc = safe_float(metabolic_concordance.loc[spot_id, "metabolic_label_concordance_score"])
        metab_agree = int(metabolic_concordance.loc[spot_id, "metabolic_state_agreement"])

        confidence_group = "intermediate"

        if method_count == 0:
            confidence_group = "method_unavailable"
        elif structure_conf < 0.10 and (pd.isna(metab_best) or metab_best < LOW_SIGNAL_BEST_SCORE):
            confidence_group = "low_signal"
        elif metab_agree == 0 and metab_best >= STRONG_MISMATCH_BEST_SCORE:
            confidence_group = "mismatch_but_strong"
        elif agreement_fraction >= 0.50 and structure_conf >= 0.25 and metab_conc >= HIGH_CONFIDENCE_CONCORDANCE:
            confidence_group = "high_confidence"

        row = {
            "method_structure_agreement_fraction": agreement_fraction,
            "method_structure_count": method_count,
            "method_confidence_group": confidence_group,
        }

        for method, label in method_labels.items():
            row[f"method_structure_call__{method}"] = label
            row[f"method_structure_agree__{method}"] = int(label == final_label)

        rows.append(row)

    return pd.DataFrame(rows, index=idx)


# =========================
# Slide level summaries
# =========================

def summarize_spot_outputs(
    sample_id: str,
    final_scores_by_axis: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    method_status: dict[str, str],
) -> dict[str, Any]:
    """Create one slide level summary row from spot level scores and labels."""
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "n_spots_scored": int(len(labels)),
    }

    for method, status in method_status.items():
        row[f"{method}_status"] = status

    if "structure_dominant_label" in labels.columns:
        counts = labels["structure_dominant_label"].value_counts(normalize=True)
        for label, frac in counts.items():
            row[f"structure_fraction__{label}"] = float(frac)

    if "structure_dominant_label_raw" in labels.columns:
        counts = labels["structure_dominant_label_raw"].value_counts(normalize=True)
        for label, frac in counts.items():
            row[f"structure_raw_fraction__{label}"] = float(frac)

    if "structure_region_label" in labels.columns:
        counts = labels["structure_region_label"].value_counts(normalize=True)
        for label, frac in counts.items():
            row[f"structure_region_fraction__{label}"] = float(frac)

    if "structure_region_status" in labels.columns:
        counts = labels["structure_region_status"].value_counts(normalize=True)
        for status, frac in counts.items():
            row[f"structure_region_status_fraction__{status}"] = float(frac)

    if "structure_region_consensus_fraction" in labels.columns:
        row.update(summarize_vector(labels["structure_region_consensus_fraction"], "structure_region_consensus_fraction"))

    for axis, scores in final_scores_by_axis.items():
        for col in scores.columns:
            prefix = f"{axis}_score__{col}"
            row.update(summarize_vector(scores[col], prefix))

    for col in labels.columns:
        if col.endswith("_active"):
            row[f"fraction__{col}"] = safe_float(pd.to_numeric(labels[col], errors="coerce").mean())

    for col in [
        "structure_confidence",
        "method_structure_agreement_fraction",
        "metabolic_label_concordance_score",
        "metabolic_best_matching_state_score",
    ]:
        if col in labels.columns:
            row.update(summarize_vector(labels[col], col))

    if "method_confidence_group" in labels.columns:
        vc = labels["method_confidence_group"].value_counts(normalize=True)
        for group, frac in vc.items():
            row[f"method_confidence_fraction__{group}"] = float(frac)

    if "metabolic_stratification_group" in labels.columns:
        vc = labels["metabolic_stratification_group"].value_counts(normalize=True)
        for group, frac in vc.items():
            row[f"metabolic_stratification_fraction__{group}"] = float(frac)

    return row


def build_agreement_summary(all_labels: pd.DataFrame) -> pd.DataFrame:
    """Summarize method agreement metrics by dominant structure label."""
    if all_labels.empty or "structure_dominant_label" not in all_labels.columns:
        return pd.DataFrame()

    grp = all_labels.groupby("structure_dominant_label", dropna=False)

    summary = grp.agg(
        n_spots=("spot_id", "count"),
        mean_structure_confidence=("structure_confidence", "mean"),
        mean_method_structure_agreement_fraction=("method_structure_agreement_fraction", "mean"),
        mean_metabolic_label_concordance_score=("metabolic_label_concordance_score", "mean"),
        mean_metabolic_best_matching_state_score=("metabolic_best_matching_state_score", "mean"),
    ).reset_index()

    if "method_confidence_group" in all_labels.columns:
        counts = (
            all_labels.groupby(["structure_dominant_label", "method_confidence_group"], dropna=False)
            .size()
            .reset_index(name="n")
        )

        wide = counts.pivot(
            index="structure_dominant_label",
            columns="method_confidence_group",
            values="n",
        ).fillna(0).reset_index()

        summary = summary.merge(wide, on="structure_dominant_label", how="left")

    return summary


# =========================
# Per sample processing
# =========================

def process_one_sample(
    sample_id: str,
    h5ad_path: Path,
    out_dirs: dict[str, Path],
    external: ExternalLibraries,
    args: argparse.Namespace,
) -> SampleResult:
    """Score one sample and write Step 05 per sample labels, scores, support tables, and h5ad output."""
    method_status: dict[str, str] = {}

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    source = get_expression_source(adata)
    source.obs_names = source.obs_names.astype(str)
    source.var_names = source.var_names.astype(str)
    source.var_names_make_unique()

    present_genes = set(get_present_gene_map(source).keys())

    all_sig_upper = uppercase_gene_sets(ALL_SIGNATURES)
    custom_filtered, custom_support = filter_gene_sets_to_present(
        all_sig_upper,
        present_genes,
        min_genes=MIN_GENES_PER_SIGNATURE,
    )

    hallmark_filtered, hallmark_support = filter_gene_sets_to_present(
        external.hallmark,
        present_genes,
        min_genes=EXTERNAL_GSVA_MIN_SIZE,
    )

    reactome_filtered, reactome_support = filter_gene_sets_to_present(
        external.reactome,
        present_genes,
        min_genes=EXTERNAL_GSVA_MIN_SIZE,
    )

    for df, collection in [
        (custom_support, "custom"),
        (hallmark_support, "hallmark"),
        (reactome_support, "reactome"),
    ]:
        if not df.empty:
            df.insert(0, "sample_id", sample_id)
            df.insert(1, "collection", collection)

    support = pd.concat([custom_support, hallmark_support, reactome_support], ignore_index=True)

    needed_genes = set()
    for gene_set in list(custom_filtered.values()) + list(hallmark_filtered.values()) + list(reactome_filtered.values()):
        needed_genes.update(gene_set)

    expr_df = make_expr_df_upper(source, needed_genes)
    expr_df = expr_df.reindex(source.obs_names.astype(str))

    keep = apply_spot_qc(adata, expr_df.index)
    expr_df = expr_df.loc[keep].copy()

    if expr_df.shape[0] == 0:
        raise ValueError("No spots passed transcriptome QC")

    adata_for_output = adata[expr_df.index].copy()

    coords_full = get_spatial_coordinates(adata)
    if coords_full is not None:
        idx_lookup = pd.Series(np.arange(len(adata.obs_names)), index=adata.obs_names.astype(str))
        idx = idx_lookup.reindex(expr_df.index).dropna().astype(int).to_numpy()
        coords = coords_full[idx]
    else:
        coords = None

    obs = adata.obs.reindex(expr_df.index).copy()

    simple_scores = compute_mean_signature_scores(expr_df, custom_filtered)
    rank_scores = compute_rank_signature_scores(expr_df, custom_filtered)
    ucell_scores, ucell_status = compute_ucell_scores(
        source[source.obs_names.astype(str).isin(expr_df.index)].copy(),
        custom_filtered,
        enabled=not args.skip_ucell,
        n_jobs=args.ucell_n_jobs,
    )
    ucell_scores = ucell_scores.reindex(expr_df.index)

    custom_gsva_scores, custom_gsva_status = compute_gsva_scores(
        expr_df,
        custom_filtered,
        enabled=not args.skip_custom_gsva,
        min_size=CUSTOM_GSVA_MIN_SIZE,
        max_size=GSVA_MAX_SIZE,
    )

    hallmark_scores, hallmark_status = compute_gsva_scores(
        expr_df,
        hallmark_filtered,
        enabled=not args.skip_hallmark_gsva,
        min_size=EXTERNAL_GSVA_MIN_SIZE,
        max_size=GSVA_MAX_SIZE,
    )

    reactome_scores, reactome_status = compute_gsva_scores(
        expr_df,
        reactome_filtered,
        enabled=not args.skip_reactome_gsva,
        min_size=EXTERNAL_GSVA_MIN_SIZE,
        max_size=GSVA_MAX_SIZE,
    )

    method_status.update({
        "simple_mean": "ok" if not simple_scores.empty else "empty",
        "rank_percentile": "ok" if not rank_scores.empty else "empty",
        "ucell": ucell_status,
        "custom_gsva": custom_gsva_status,
        "hallmark_gsva": hallmark_status,
        "reactome_gsva": reactome_status,
    })

    hallmark_axis = map_external_scores_to_axis_labels(
        hallmark_scores,
        HALLMARK_TO_AXIS_LABEL,
        is_reactome=False,
    )

    reactome_axis = map_external_scores_to_axis_labels(
        reactome_scores,
        None,
        is_reactome=True,
    )

    base_method_frames = {
        "simple_mean": simple_scores,
        "rank_percentile": rank_scores,
        "ucell": ucell_scores,
        "custom_gsva": custom_gsva_scores,
    }

    method_weights = {
        "simple_mean": 1.00,
        "rank_percentile": 0.90,
        "ucell": 1.20,
        "custom_gsva": 1.10,
        "hallmark_gsva": 1.00,
        "reactome_gsva": 1.00,
    }

    axis_method_frames = {
        "structure": {
            **base_method_frames,
            "hallmark_gsva": hallmark_axis["structure"],
            "reactome_gsva": reactome_axis["structure"],
        },
        "function": {
            **base_method_frames,
            "hallmark_gsva": hallmark_axis["function"],
            "reactome_gsva": reactome_axis["function"],
        },
        "metabolism": {
            **base_method_frames,
            "hallmark_gsva": hallmark_axis["metabolism"],
            "reactome_gsva": reactome_axis["metabolism"],
        },
    }

    structure_base = combine_method_scores(axis_method_frames["structure"], STRUCTURE_KEYS, method_weights)
    function_base = combine_method_scores(axis_method_frames["function"], FUNCTION_KEYS, method_weights)
    metabolic_base = combine_method_scores(axis_method_frames["metabolism"], METABOLIC_KEYS, method_weights)

    structure_scores = refine_axis_scores(structure_base, coords, obs)
    function_scores = refine_axis_scores(function_base, coords, obs)
    metabolic_scores = refine_axis_scores(metabolic_base, coords, obs)

    structure_labels = build_structure_labels(structure_scores)
    function_labels = build_multilabel_axis_labels(function_scores, "function", FUNCTION_ACTIVE_QUANTILE)
    metabolic_labels = build_multilabel_axis_labels(metabolic_scores, "metabolic", METABOLIC_ACTIVE_QUANTILE)
    metabolic_concordance = build_metabolic_concordance(structure_labels, metabolic_scores)

    method_agreement = build_method_agreement(
        base_method_frames,
        structure_labels,
        metabolic_concordance,
    )

    labels = pd.concat(
        [
            structure_labels,
            function_labels,
            metabolic_labels,
            metabolic_concordance,
            method_agreement,
        ],
        axis=1,
    )

    labels = add_structure_region_consensus(
        labels=labels,
        obs=obs,
        raw_label_col="structure_dominant_label",
        confidence_col="structure_confidence",
    )

    labels.insert(0, "sample_id", sample_id)
    labels.insert(1, "spot_id", labels.index.astype(str))

    score_frames = []

    for axis, scores in [
        ("structure", structure_scores),
        ("function", function_scores),
        ("metabolic", metabolic_scores),
    ]:
        temp = scores.copy()
        temp.columns = [f"{axis}_score__{c}" for c in temp.columns]
        score_frames.append(temp)

    method_score_frames = []
    for method, df in base_method_frames.items():
        if df is not None and not df.empty:
            temp = df.copy()
            temp.columns = [f"{method}__{c}" for c in temp.columns]
            method_score_frames.append(temp)

    if not hallmark_scores.empty:
        temp = hallmark_scores.copy()
        temp.columns = [f"hallmark_gsva__{c}" for c in temp.columns]
        method_score_frames.append(temp)

    if not reactome_scores.empty:
        temp = reactome_scores.copy()
        temp.columns = [f"reactome_gsva__{c}" for c in temp.columns]
        method_score_frames.append(temp)

    spot_scores = pd.concat(score_frames + method_score_frames, axis=1)
    spot_scores.insert(0, "sample_id", sample_id)
    spot_scores.insert(1, "spot_id", spot_scores.index.astype(str))

    labels_out = out_dirs["per_sample"] / f"{sample_id}_spot_labels.csv"
    scores_out = out_dirs["per_sample"] / f"{sample_id}_spot_scores.csv"
    h5ad_out = out_dirs["per_sample_h5ad"] / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    support_out = out_dirs["per_sample_support"] / f"{sample_id}_gene_support.csv"

    labels.to_csv(labels_out, index=False)
    spot_scores.to_csv(scores_out, index=False)
    support.to_csv(support_out, index=False)

    obs_append = labels.set_index("spot_id").reindex(adata_for_output.obs_names.astype(str))
    for col in obs_append.columns:
        if col == "sample_id":
            continue
        adata_for_output.obs[col] = obs_append[col].values

    score_append = spot_scores.set_index("spot_id").reindex(adata_for_output.obs_names.astype(str))
    for col in score_append.columns:
        if col == "sample_id":
            continue
        if col.startswith(("structure_score__", "function_score__", "metabolic_score__")):
            adata_for_output.obs[col] = pd.to_numeric(score_append[col], errors="coerce").values

    adata_for_output.uns["multi_axis_transcriptome_labels"] = {
        "script": "05_build_multi_axis_transcriptome_labels.py",
        "sample_id": sample_id,
        "method_status": method_status,
        "external_library_status": external.status,
        "external_library_source": external.source,
    }

    adata_for_output.write_h5ad(h5ad_out)

    final_scores_by_axis = {
        "structure": structure_scores,
        "function": function_scores,
        "metabolic": metabolic_scores,
    }

    slide_row = summarize_spot_outputs(sample_id, final_scores_by_axis, labels.set_index("spot_id"), method_status)

    status_row = {
        "sample_id": sample_id,
        "status": "ok",
        "h5ad_input": str(h5ad_path),
        "n_spots_input": int(adata.n_obs),
        "n_spots_scored": int(expr_df.shape[0]),
        "n_genes_available": int(len(present_genes)),
        "n_custom_signatures_scored": int(len(custom_filtered)),
        "n_hallmark_terms_scored": int(len(hallmark_filtered)),
        "n_reactome_terms_scored": int(len(reactome_filtered)),
        **{f"{k}_status": v for k, v in method_status.items()},
    }

    return SampleResult(
        status="ok",
        sample_id=sample_id,
        slide_row=slide_row,
        status_row=status_row,
        support=support,
    )


# =========================
# Output and summary
# =========================

def build_summary_text(status_df: pd.DataFrame, slide_df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    """Build the Step 05 cohort summary text."""
    lines = []
    lines.append("Multi axis transcriptome label summary")
    lines.append("")
    lines.append(f"Script: 05_build_multi_axis_transcriptome_labels.py")
    lines.append(f"Output folder: {metadata.get('out_subdir', '')}")
    lines.append(f"External library status: {metadata.get('external_library_status', '')}")
    lines.append(f"External library source: {metadata.get('external_library_source', '')}")
    lines.append("")
    lines.append(f"Samples attempted: {len(status_df)}")

    if not status_df.empty and "status" in status_df.columns:
        lines.append("")
        lines.append("Sample status counts:")
        for key, value in status_df["status"].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    method_cols = [c for c in status_df.columns if c.endswith("_status") and c != "status"]
    for col in method_cols:
        lines.append("")
        lines.append(f"{col} counts:")
        for key, value in status_df[col].value_counts(dropna=False).items():
            lines.append(f"  {key}: {value}")

    if not slide_df.empty:
        lines.append("")
        lines.append(f"Slide level rows written: {len(slide_df)}")
        lines.append(f"Slide level columns written: {slide_df.shape[1]}")

    return "\n".join(lines)


def write_metadata(out_dir: Path, args: argparse.Namespace, external: ExternalLibraries, status_df: pd.DataFrame) -> dict[str, Any]:
    """Write Step 05 metadata describing settings, libraries, and status counts."""
    metadata = {
        "script": "05_build_multi_axis_transcriptome_labels.py",
        "purpose": "canonical multi axis transcriptome labels with method fusion",
        "out_subdir": args.out_subdir,
        "settings": {
            "min_genes_per_signature": MIN_GENES_PER_SIGNATURE,
            "min_genes_external": MIN_GENES_EXTERNAL,
            "structure_active_quantile": STRUCTURE_ACTIVE_QUANTILE,
            "function_active_quantile": FUNCTION_ACTIVE_QUANTILE,
            "metabolic_active_quantile": METABOLIC_ACTIVE_QUANTILE,
            "active_confidence_min": ACTIVE_CONFIDENCE_MIN,
            "n_top_structure_labels": N_TOP_STRUCTURE_LABELS,
            "smooth_k": SMOOTH_K,
            "local_z_k": LOCAL_Z_K,
            "custom_gsva_min_size": CUSTOM_GSVA_MIN_SIZE,
            "external_gsva_min_size": EXTERNAL_GSVA_MIN_SIZE,
            "gsva_max_size": GSVA_MAX_SIZE,
            "reactome_max_terms": args.reactome_max_terms,
            "skip_ucell": args.skip_ucell,
            "skip_custom_gsva": args.skip_custom_gsva,
            "skip_hallmark_gsva": args.skip_hallmark_gsva,
            "skip_reactome_gsva": args.skip_reactome_gsva,
            "ucell_n_jobs": args.ucell_n_jobs,
        },
        "available_packages": {
            "pyucell": HAS_UCELL,
            "gseapy": HAS_GSEAPY,
        },
        "external_library_status": external.status,
        "external_library_source": external.source,
        "n_hallmark_terms_loaded": len(external.hallmark),
        "n_reactome_terms_loaded": len(external.reactome),
        "signature_axis": SIGNATURE_AXIS,
        "hallmark_to_axis_label": HALLMARK_TO_AXIS_LABEL,
        "reactome_rules": [
            {"keywords": keywords, "mappings": mappings}
            for keywords, mappings in REACTOME_RULES
        ],
        "status_counts": status_df["status"].value_counts(dropna=False).to_dict() if "status" in status_df.columns else {},
    }

    with open(out_dir / "multi_axis_label_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def write_one_sample_tracking_files(
    out_dirs: dict[str, Path],
    sample_id: str,
    slide_row: dict[str, Any] | None,
    status_row: dict[str, Any],
) -> None:
    """Write per-sample status and slide rows for safe parallel merging."""
    status_path = out_dirs["per_sample_status"] / f"{sample_id}_status.csv"
    pd.DataFrame([status_row]).to_csv(status_path, index=False)

    if slide_row:
        slide_path = out_dirs["per_sample_slide"] / f"{sample_id}_slide_summary.csv"
        pd.DataFrame([slide_row]).to_csv(slide_path, index=False)


def read_many_csv(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate many csv files safely."""
    frames = []

    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def build_final_merged_outputs(
    out_dir: Path,
    out_dirs: dict[str, Path],
    output_root: Path,
    args: argparse.Namespace,
    external: ExternalLibraries,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-sample outputs into final cohort-level summary tables."""
    status_paths = sorted(out_dirs["per_sample_status"].glob("*_status.csv"))
    slide_paths = sorted(out_dirs["per_sample_slide"].glob("*_slide_summary.csv"))
    support_paths = sorted(out_dirs["per_sample_support"].glob("*_gene_support.csv"))
    label_paths = sorted(out_dirs["per_sample"].glob("*_spot_labels.csv"))

    status_df = read_many_csv(status_paths)
    slide_df = read_many_csv(slide_paths)
    support_df = read_many_csv(support_paths)
    all_labels = read_many_csv(label_paths)

    status_df.to_csv(out_dir / "multi_axis_label_status.csv", index=False)
    slide_df.to_csv(out_dir / "multi_axis_slide_summary.csv", index=False)
    support_df.to_csv(out_dir / "multi_axis_gene_support_all_samples.csv", index=False)

    agreement_summary = build_agreement_summary(all_labels)
    agreement_summary.to_csv(out_dir / "multi_axis_method_agreement_summary.csv", index=False)

    base_table = read_base_table(output_root)

    if not base_table.empty and not slide_df.empty and "sample_id" in base_table.columns:
        merged = base_table.merge(slide_df, on="sample_id", how="left")
    else:
        merged = slide_df.copy()

    merged.to_csv(out_dir / "slide_features_with_multi_axis_labels.csv", index=False)

    metadata = write_metadata(out_dir, args, external, status_df)
    summary_text = build_summary_text(status_df, slide_df, metadata)
    (out_dir / "multi_axis_label_summary.txt").write_text(summary_text, encoding="utf-8")

    return status_df, slide_df

# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Step 05 multi axis transcriptome labeling."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--processed-subdir", default=DEFAULT_PROCESSED_SUBDIR)

    parser.add_argument("--skip-ucell", action="store_true")
    parser.add_argument("--skip-custom-gsva", action="store_true")
    parser.add_argument("--skip-hallmark-gsva", action="store_true")
    parser.add_argument("--skip-reactome-gsva", action="store_true")

    parser.add_argument("--ucell-n-jobs", type=int, default=int(os.environ.get("UCELL_N_JOBS", "1")))
    parser.add_argument("--reactome-max-terms", type=int, default=75)

    parser.add_argument(
        "--per-sample-only",
        action="store_true",
        help="Write only per-sample outputs. Do not write shared merged cohort tables.",
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing per-sample outputs into final cohort summary tables without scoring samples.",
    )

    return parser.parse_args()


def main() -> None:
    """Run Step 05 scoring, per sample output generation, or merge only aggregation."""
    args = parse_args()

    if args.per_sample_only and args.merge_only:
        raise ValueError("Use either --per-sample-only or --merge-only, not both.")

    cfg = validate_config(load_config(args.config))

    output_root = Path(cfg["output_root"])
    processed_root = output_root / args.processed_subdir
    out_dir = output_root / args.out_subdir

    out_dirs = {
        "root": out_dir,
        "per_sample": out_dir / "per_sample",
        "per_sample_h5ad": out_dir / "per_sample_h5ad",
        "per_sample_support": out_dir / "per_sample_support",
        "per_sample_status": out_dir / "per_sample_status",
        "per_sample_slide": out_dir / "per_sample_slide",
    }

    for path in out_dirs.values():
        ensure_dir(path)

    print("=== Multi axis transcriptome label build ===")
    print("Output root:", output_root)
    print("Processed root:", processed_root)
    print("Output folder:", out_dir)
    print("UCell available:", HAS_UCELL)
    print("GSEApy available:", HAS_GSEAPY)
    print("UCell jobs:", args.ucell_n_jobs)
    print("Reactome max terms:", args.reactome_max_terms)
    print("Per-sample only:", args.per_sample_only)
    print("Merge only:", args.merge_only)
    print()

    external = load_external_libraries(args.reactome_max_terms)

    print("External library status:", external.status)
    print("Hallmark terms loaded:", len(external.hallmark))
    print("Reactome terms loaded:", len(external.reactome))
    print()

    if args.merge_only:
        status_df, slide_df = build_final_merged_outputs(
            out_dir=out_dir,
            out_dirs=out_dirs,
            output_root=output_root,
            args=args,
            external=external,
        )

        print("DONE merge-only")
        print("Status:", out_dir / "multi_axis_label_status.csv")
        print("Slide summary:", out_dir / "multi_axis_slide_summary.csv")
        print("Merged slide table:", out_dir / "slide_features_with_multi_axis_labels.csv")
        print(f"Rows merged: status={len(status_df)}, slide={len(slide_df)}")
        return

    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed root: {processed_root}")

    sample_dirs = sorted([
        p for p in processed_root.iterdir()
        if p.is_dir() and p.name.startswith("SAMPLE_")
    ])

    if args.sample_id:
        requested = set(args.sample_id)
        sample_dirs = [p for p in sample_dirs if p.name in requested]

    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]

    status_rows = []
    slide_rows = []

    for i, sample_dir in enumerate(sample_dirs, start=1):
        sample_id = sample_dir.name
        print(f"[{i}/{len(sample_dirs)}] {sample_id}")

        h5ad_path = find_h5ad_for_sample(processed_root, sample_id)

        if h5ad_path is None:
            status_row = {
                "sample_id": sample_id,
                "status": "missing_h5ad",
                "error": "No h5ad candidate found",
            }

            write_one_sample_tracking_files(
                out_dirs=out_dirs,
                sample_id=sample_id,
                slide_row=None,
                status_row=status_row,
            )

            status_rows.append(status_row)
            print("  missing h5ad")
            continue

        labels_out = out_dirs["per_sample"] / f"{sample_id}_spot_labels.csv"
        h5ad_out = out_dirs["per_sample_h5ad"] / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"

        if labels_out.exists() and h5ad_out.exists() and not args.overwrite:
            status_row = {
                "sample_id": sample_id,
                "status": "already_present",
                "h5ad_input": str(h5ad_path),
            }

            write_one_sample_tracking_files(
                out_dirs=out_dirs,
                sample_id=sample_id,
                slide_row=None,
                status_row=status_row,
            )

            status_rows.append(status_row)
            print("  already present")
            continue

        try:
            result = process_one_sample(
                sample_id=sample_id,
                h5ad_path=h5ad_path,
                out_dirs=out_dirs,
                external=external,
                args=args,
            )

            write_one_sample_tracking_files(
                out_dirs=out_dirs,
                sample_id=sample_id,
                slide_row=result.slide_row,
                status_row=result.status_row,
            )

            slide_rows.append(result.slide_row)
            status_rows.append(result.status_row)

            print("  ok")

        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"

            status_row = {
                "sample_id": sample_id,
                "status": "failed",
                "h5ad_input": str(h5ad_path),
                "error": error_text,
                "traceback": traceback.format_exc(),
            }

            write_one_sample_tracking_files(
                out_dirs=out_dirs,
                sample_id=sample_id,
                slide_row=None,
                status_row=status_row,
            )

            status_rows.append(status_row)
            print("  failed:", error_text)

        if not args.per_sample_only:
            pd.DataFrame(status_rows).to_csv(out_dir / "multi_axis_label_status.csv", index=False)

    if args.per_sample_only:
        print()
        print("DONE per-sample-only")
        print("Per-sample outputs written under:", out_dir)
        print("Run merge after all parallel jobs finish:")
        print(
            f"python .\\code\\05_build_multi_axis_transcriptome_labels.py "
            f"--config {args.config} --merge-only "
            f"--reactome-max-terms {args.reactome_max_terms}"
        )
        return

    status_df, slide_df = build_final_merged_outputs(
        out_dir=out_dir,
        out_dirs=out_dirs,
        output_root=output_root,
        args=args,
        external=external,
    )

    metadata = write_metadata(out_dir, args, external, status_df)
    summary_text = build_summary_text(status_df, slide_df, metadata)

    print()
    print("DONE")
    print("Status:", out_dir / "multi_axis_label_status.csv")
    print("Slide summary:", out_dir / "multi_axis_slide_summary.csv")
    print("Merged slide table:", out_dir / "slide_features_with_multi_axis_labels.csv")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
