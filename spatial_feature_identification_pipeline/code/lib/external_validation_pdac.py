"""PDAC external validation helper library.

This module supports Step 13 PDAC validation by parsing study supplementary
marker tables, scoring validation marker programs, generating local validation
maps, and preparing manual review outputs. The marker scores produced here are
validation overlays only and are not added to canonical pipeline labels or model
input features."""

from __future__ import annotations


# =========================
# Imports
# =========================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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
# Validation category definitions
# =========================

CATEGORY_KEYWORDS = {
    "normal_acinar": ["acinar"],
    "normal_endocrine": ["endocrine"],
    "normal_ductal": ["ductal"],
    "tumor_classical": ["classical"],
    "tumor_basal": ["basal"],
    "tumor_proliferative": ["proliferative", "proliferation"],
    "tumor_fibroblast_high": ["fibroblast-high", "fib high", "fib-high", "fibroblast high"],
    "stroma_fibroblast": ["fibroblast", "caf"],
    "stroma_stellate": ["stellate"],
    "vascular_endothelial": ["endothelial"],
    "immune_t_cell": ["t cell", "tcell"],
    "immune_b_plasma": ["b cell", "plasma"],
    "immune_macrophage": ["macrophage"],
    "immune_dendritic": ["dendritic"],
    "immune_mast": ["mast"],
    "hypoxia": ["hypoxia"],
}


# =========================
# Curated gene hint definitions
# =========================

GENE_HINTS = {
    "normal_acinar": ["PRSS2", "REG1A", "CPB1", "CPA1", "AMY2A", "SPINK1"],
    "normal_endocrine": ["INS", "GCG", "TTR", "CHGA", "CHGB"],
    "normal_ductal": ["KRT19", "KRT7", "CRP", "SPINK1", "MMP7"],
    "tumor_classical": ["TFF1", "TFF3", "CEACAM6", "CEACAM5", "AGR2", "S100P", "KRT7", "KRT19", "EPCAM"],
    "tumor_basal": ["KRT14", "KRT17", "KRT6A", "S100A2", "SERPINB3", "SERPINB4", "KRT16"],
    "tumor_proliferative": ["MKI67", "TOP2A", "STMN1", "TUBB", "HMGB2", "AURKA", "BIRC5"],
    "stroma_fibroblast": ["COL1A1", "COL1A2", "COL3A1", "COL6A2", "COL6A3", "DCN", "LUM", "FN1", "BGN"],
    "stroma_stellate": ["TAGLN", "ACTA2", "MYL9", "CALD1", "MYH11", "TPM2"],
    "vascular_endothelial": ["VWF", "PECAM1", "COL15A1", "RGS5", "ENG"],
    "immune_t_cell": ["CD3D", "CD3E", "CD2", "IL7R", "CXCR4", "GZMB", "CD8A"],
    "immune_b_plasma": ["XBP1", "IGKC", "IGHG1", "IGHG2", "MZB1", "MS4A1", "CD79A", "JCHAIN"],
    "immune_macrophage": ["C1QA", "C1QB", "C1QC", "LYZ", "CD68", "CD163", "MARCO", "FCGR3A"],
    "immune_dendritic": ["LAMP3", "CCR7", "IDO1", "CD80", "CSF2RA"],
    "immune_mast": ["TPSB2", "TPSAB1", "CPA3", "KIT"],
    "hypoxia": ["HIF1A", "CA9", "VEGFA", "SLC2A1", "LDHA", "DDIT4", "ENO1"],
}

PIPELINE_REGION_COLORS = {
    "tumor_like": "#d64f4f",
    "stroma_like": "#75b878",
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
# Config helpers
# =========================

def load_yaml(path: Path) -> dict:
    """Load one YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_bool(value) -> bool:
    """Support as bool within the PDAC external validation workflow."""
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def get_output_root(cfg: dict, project_root: Path) -> Path:
    """Resolve the configured output root for PDAC validation."""
    p = Path(str(cfg.get("output_root", "")))

    if p.exists():
        return p

    return project_root / "outputs"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supplementary table column names for parser compatibility."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_optional(row, name, default=""):
    """Return a named row value or a default when missing."""
    try:
        val = row.get(name, default)
    except Exception:
        return default

    if pd.isna(val):
        return default

    return val


def row_text(row, cols) -> str:
    """Build searchable text from selected supplementary table columns."""
    vals = []
    for c in cols:
        if c in row.index:
            val = row[c]
            if not pd.isna(val):
                vals.append(str(val))
    return " ".join(vals).lower()


def category_from_text(text: str) -> list[str]:
    """Infer validation marker categories from row annotation text."""
    cats = []

    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k.lower() in text for k in keywords):
            cats.append(cat)

    return cats


def category_from_gene(symbol: str) -> list[str]:
    """Infer validation marker categories from curated gene hints."""
    symbol = str(symbol).upper()
    cats = []

    for cat, genes in GENE_HINTS.items():
        if symbol in set(g.upper() for g in genes):
            cats.append(cat)

    return cats



# =========================
# Supplementary marker table parsing
# =========================

def parse_pdac_supplementary_tables(supp_dir: Path, out_dir: Path) -> dict:
    """Parse PDAC supplementary XLSX tables into validation marker dictionaries."""
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(supp_dir.glob("mmc*.xlsx"))

    if not files:
        raise FileNotFoundError(
            "No mmc*.xlsx files found. Copy the PDAC supplementary Excel files to: "
            + str(supp_dir)
        )

    rows = []

    for xlsx in files:
        try:
            xls = pd.ExcelFile(xlsx)
        except Exception as e:
            rows.append({
                "source_file": xlsx.name,
                "sheet": "",
                "status": "read_error",
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet)
            except Exception as e:
                rows.append({
                    "source_file": xlsx.name,
                    "sheet": sheet,
                    "status": "sheet_read_error",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue

            df = normalize_column_names(df)

            if "Symbol" not in df.columns:
                continue

            for _, r in df.iterrows():
                symbol = str(r.get("Symbol", "")).strip().upper()

                if symbol == "" or symbol == "NAN":
                    continue

                text = row_text(
                    r,
                    [
                        "cluster_name",
                        "cluster",
                        "Sig_Name",
                        "Signature_name",
                        "Signature",
                        "direction",
                    ],
                )

                cats = category_from_text(text)
                gene_cats = category_from_gene(symbol)

                for cat in gene_cats:
                    if cat not in cats:
                        cats.append(cat)

                if not cats:
                    continue

                avg_logfc = pd.to_numeric(get_optional(r, "avg_logFC", np.nan), errors="coerce")
                pscore = pd.to_numeric(get_optional(r, "p_val_adj_neg_log10", np.nan), errors="coerce")
                pct_1 = pd.to_numeric(get_optional(r, "pct_1", np.nan), errors="coerce")
                pct_2 = pd.to_numeric(get_optional(r, "pct_2", np.nan), errors="coerce")

                for cat in cats:
                    rows.append({
                        "source_file": xlsx.name,
                        "sheet": sheet,
                        "status": "ok",
                        "category": cat,
                        "symbol": symbol,
                        "cluster": get_optional(r, "cluster", ""),
                        "cluster_name": get_optional(r, "cluster_name", ""),
                        "sig_name": get_optional(r, "Sig_Name", get_optional(r, "Signature_name", "")),
                        "direction": get_optional(r, "direction", ""),
                        "avg_logFC": avg_logfc,
                        "p_val_adj_neg_log10": pscore,
                        "pct_1": pct_1,
                        "pct_2": pct_2,
                    })

    source = pd.DataFrame(rows)

    source_path = out_dir / "pdac_supplementary_marker_source_rows.csv"
    source.to_csv(source_path, index=False)

    ok = source[source["status"] == "ok"].copy()

    if len(ok) == 0:
        raise RuntimeError("No marker rows could be parsed from the PDAC supplementary tables.")

    ok["avg_logFC_num"] = pd.to_numeric(ok["avg_logFC"], errors="coerce").fillna(0)
    ok["p_score_num"] = pd.to_numeric(ok["p_val_adj_neg_log10"], errors="coerce").fillna(0)
    ok["rank_score"] = ok["avg_logFC_num"].clip(lower=0) + 0.05 * ok["p_score_num"]

    dict_rows = []
    marker_dict = {}

    for cat in sorted(set(CATEGORY_KEYWORDS) | set(GENE_HINTS)):
        sub = ok[ok["category"] == cat].copy()

        if len(sub) == 0:
            marker_dict[cat] = []
            continue

        sub = (
            sub.sort_values(["rank_score", "p_score_num", "avg_logFC_num"], ascending=False)
            .drop_duplicates("symbol")
            .head(50)
        )

        genes = sub["symbol"].astype(str).str.upper().tolist()
        marker_dict[cat] = genes

        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            dict_rows.append({
                "category": cat,
                "rank": rank,
                "symbol": r["symbol"],
                "rank_score": r["rank_score"],
                "avg_logFC": r["avg_logFC"],
                "p_val_adj_neg_log10": r["p_val_adj_neg_log10"],
                "source_file": r["source_file"],
                "sheet": r["sheet"],
                "cluster": r["cluster"],
                "cluster_name": r["cluster_name"],
                "sig_name": r["sig_name"],
                "direction": r["direction"],
            })

    dict_table = pd.DataFrame(dict_rows)

    dict_csv = out_dir / "pdac_marker_dictionary_from_supplements.csv"
    dict_json = out_dir / "pdac_marker_dictionary_from_supplements.json"
    summary_txt = out_dir / "pdac_marker_dictionary_summary.txt"

    dict_table.to_csv(dict_csv, index=False)

    with open(dict_json, "w", encoding="utf-8") as f:
        json.dump(marker_dict, f, indent=2)

    lines = []
    lines.append("PDAC supplementary marker dictionary")
    lines.append("")
    lines.append("This dictionary is for external validation only.")
    lines.append("It is not added to steps 01 through 12 and does not alter model_input_numeric.csv.")
    lines.append("")
    lines.append("Source folder: " + str(supp_dir))
    lines.append("Files parsed:")
    for f in files:
        lines.append("  " + f.name)
    lines.append("")
    lines.append("Category counts:")

    for cat, genes in marker_dict.items():
        lines.append(f"  {cat}: {len(genes)}")

    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "marker_dict": marker_dict,
        "source_path": source_path,
        "dict_csv": dict_csv,
        "dict_json": dict_json,
        "summary_txt": summary_txt,
    }



# =========================
# PDAC sample selection
# =========================

def get_pdac_samples(output_root: Path) -> pd.DataFrame:
    """Select PDAC samples that are present in the final model input."""
    mapping_path = output_root / "_external_study_validation" / "sample_to_geo_study_mapping.csv"

    if not mapping_path.exists():
        raise FileNotFoundError("Missing sample mapping: " + str(mapping_path))

    mapping = pd.read_csv(mapping_path)

    needed = ["sample_id", "dataset_id", "cancer_type", "original_name", "in_final_model_input"]
    missing = [c for c in needed if c not in mapping.columns]

    if missing:
        raise ValueError("Sample mapping missing columns: " + ", ".join(missing))

    m = mapping[
        (mapping["dataset_id"].astype(str) == "GSE282302")
        & (mapping["cancer_type"].astype(str).str.lower() == "pdac")
        & (mapping["in_final_model_input"].apply(as_bool))
    ].copy()

    if len(m) == 0:
        raise RuntimeError("No final model PDAC samples found for GSE282302.")

    return m[needed].sort_values("sample_id")


def read_h5ad(output_root: Path, sample_id: str):
    """Support read h5ad within the PDAC external validation workflow."""
    path = (
        output_root
        / "output_05_build_multi_axis_transcriptome_labels"
        / "per_sample_h5ad"
        / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"
    )

    if not path.exists():
        raise FileNotFoundError("Missing h5ad: " + str(path))

    return ad.read_h5ad(path), path


def get_coords(adata) -> np.ndarray:
    """Support get coords within the PDAC external validation workflow."""
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData missing obsm['spatial']")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Invalid spatial coordinates")

    return coords[:, :2]



# =========================
# Expression and marker scoring
# =========================

def expression_source(adata):
    """Return raw expression when available, otherwise active AnnData expression."""
    if adata.raw is not None:
        src = adata.raw.to_adata()
    else:
        src = adata.copy()

    src.var_names = src.var_names.astype(str)
    src.var_names_make_unique()
    return src


def gene_vector(src, gene: str):
    """Extract one gene expression vector from an AnnData object."""
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
    """Convert numeric values into zero to one percentile ranks."""
    s = pd.Series(values)

    if s.notna().sum() <= 1:
        return np.zeros(len(s), dtype=float)

    if s.nunique(dropna=True) <= 1:
        return np.full(len(s), 0.5, dtype=float)

    return s.rank(pct=True).fillna(0.0).to_numpy(dtype=float)


def signature_score(src, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Score a marker set by averaging ranked expression across present genes."""
    vals = []
    found = []

    for gene in genes:
        v = gene_vector(src, gene)

        if v is not None:
            vals.append(rank01(v))
            found.append(gene)

    if not vals:
        return np.zeros(src.n_obs, dtype=float), found

    return np.nanmean(np.vstack(vals), axis=0), found


def choose_structure_column(obs: pd.DataFrame) -> str | None:
    """Choose the best available pipeline structure label column."""
    for c in [
        "structure_region_label_smoothed",
        "structure_region_label",
        "structure_dominant_label_raw",
        "structure_dominant_label",
    ]:
        if c in obs.columns:
            return c
    return None



# =========================
# Pipeline region crosswalk
# =========================

def pipeline_region_crosswalk(adata) -> np.ndarray:
    """Collapse pipeline structure labels into broad validation regions."""
    obs = adata.obs.copy()
    col = choose_structure_column(obs)

    if col is None:
        return np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    x = obs[col].astype(str).str.lower()

    out = np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    stroma = x.str.contains("stromal|ecm", na=False).to_numpy()
    immune = x.str.contains("t_cell|immune|plasma|myeloid|macrophage", na=False).to_numpy()
    vascular = x.str.contains("vascular|angiogenic|endothelial", na=False).to_numpy()
    tumor = x.str.contains("tumor", na=False).to_numpy()

    out[stroma] = "stroma_like"
    out[immune] = "immune_like"
    out[vascular] = "vascular_like"
    out[tumor] = "tumor_like"

    return out


def core_border_from_pipeline(coords: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Support core border from pipeline within the PDAC external validation workflow."""
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

    for i in np.where(tumor)[0]:
        neighbors = tree.query_ball_point(coords[i], r=radius)
        touches_non_tumor = any(not tumor[j] for j in neighbors if j != i)
        out[i] = "tumor_border" if touches_non_tumor else "tumor_core"

    return out



# =========================
# Spatial plotting helpers
# =========================

def plot_categorical(coords, labels, title: str, dest: Path, colors: dict[str, str]) -> None:
    """Write a categorical spatial validation map."""
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
    """Write a continuous marker score spatial validation map."""
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
# Study figure pairing helpers
# =========================

def side_by_side(left_path: Path, right_path: Path, dest: Path, title: str) -> str:
    """Create a side by side study and pipeline validation image."""
    if not HAS_PIL:
        return "pillow_not_available"

    if not left_path.exists() or not right_path.exists():
        return "missing_input"

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    target_h = 900

    def resize_h(img):
        """Support resize h within the PDAC external validation workflow."""
        w, h = img.size
        return img.resize((int(w * target_h / h), target_h))

    left = resize_h(left)
    right = resize_h(right)

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


def find_study_png(study_dir: Path | None, requested: str) -> Path | None:
    """Find a study PNG by exact or approximate filename matching."""
    if study_dir is None or not study_dir.exists():
        return None

    exact = study_dir / requested

    if exact.exists():
        return exact

    words = [
        w.lower()
        for w in requested.replace(".png", "").replace("_", " ").split()
        if len(w) > 1
    ]

    scored = []

    for p in study_dir.glob("*.png"):
        low = p.name.lower()
        score = sum(1 for w in words if w in low)

        if score > 0:
            scored.append((score, p))

    if not scored:
        return None

    scored = sorted(scored, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]



# =========================
# External validation path helpers
# =========================

def get_study_png_dir(cfg: dict, output_root: Path) -> Path | None:
    """Resolve the directory containing study PNG reference panels."""
    ev = cfg.get("external_validation", {}) or {}
    pdac = ev.get("pdac", {}) or {}

    raw = pdac.get("study_png_dir", "")

    if raw:
        p = Path(str(raw))

        if not p.is_absolute():
            p = output_root.parent / p

        if p.exists():
            return p

    fallback = output_root / "_external_study_validation" / "pdac"

    if not fallback.exists():
        return None

    candidates = sorted(
        [p for p in fallback.rglob("study_pngs_to_create") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return candidates[0] if candidates else None


def get_supplement_dir(cfg: dict, output_root: Path) -> Path:
    """Resolve or create the PDAC supplementary table directory."""
    ev = cfg.get("external_validation", {}) or {}
    pdac = ev.get("pdac", {}) or {}

    raw = pdac.get(
        "supplementary_table_dir",
        "outputs/_external_study_validation/pdac/source_data/supplementary_tables",
    )

    p = Path(str(raw))

    if not p.is_absolute():
        p = output_root.parent / p

    p.mkdir(parents=True, exist_ok=True)
    return p



# =========================
# Main PDAC validation workflow
# =========================

def run_pdac_validation(args, project_root: Path) -> None:
    """Run PDAC external validation and write maps, tables, and review manifests."""
    cfg = load_yaml(Path(args.config))
    output_root = get_output_root(cfg, project_root)

    ev_out_root = output_root / "output_13_external_study_validation" / "pdac"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ev_out_root / f"pdac_validation_{stamp}"

    tables_dir = out_dir / "tables"
    maps_dir = out_dir / "local_validation_maps"
    pair_dir = out_dir / "side_by_side_pairs"
    study_copy_dir = out_dir / "study_pngs_used"

    for p in [tables_dir, maps_dir, pair_dir, study_copy_dir]:
        p.mkdir(parents=True, exist_ok=True)

    supp_dir = get_supplement_dir(cfg, output_root)
    study_png_dir = get_study_png_dir(cfg, output_root)

    print("=== PDAC external validation ===")
    print("Output root:", output_root)
    print("Output folder:", out_dir)
    print("Supplementary table folder:", supp_dir)
    print("Study PNG folder:", study_png_dir if study_png_dir else "not found")
    print()

    parsed = parse_pdac_supplementary_tables(supp_dir, tables_dir)
    marker_dict = parsed["marker_dict"]

    pdac_samples = get_pdac_samples(output_root)
    pdac_samples.to_csv(tables_dir / "pdac_sample_mapping.csv", index=False)

    categories_to_plot = [
        "normal_acinar",
        "normal_ductal",
        "normal_endocrine",
        "tumor_classical",
        "tumor_basal",
        "tumor_proliferative",
        "tumor_fibroblast_high",
        "stroma_fibroblast",
        "stroma_stellate",
        "vascular_endothelial",
        "immune_macrophage",
        "immune_t_cell",
        "immune_b_plasma",
        "hypoxia",
    ]

    study_suggestions = {
        "pipeline_region_crosswalk": "Figure 1B or Supplementary Figure 1.2 pathology and de novo compartment maps",
        "core_border_pipeline": "Use any available pathology or tumor compartment spot map for the same C1_D10, C2_D11, or C3_D12 sample",
        "normal_acinar": "Supplementary Figure 1.2 pathology and signature composition, or normal parenchyma panels",
        "normal_ductal": "Supplementary Figure 2.1 and 2.3 ductal like or PanIN panels",
        "tumor_classical": "Supplementary Figure 2.1 classical PDAC signature panels",
        "tumor_basal": "Supplementary Figure 2.1 basal PDAC signature panels",
        "tumor_proliferative": "Supplementary Figure 2.1 or 2.3 proliferative panels",
        "stroma_fibroblast": "Figure 5 and Supplementary Figure 5.1 stromal neighborhood panels",
        "immune_macrophage": "Figure 4D or Supplementary Figure 4.1 tumor neighborhood immune panels",
        "vascular_endothelial": "Figure 4D and Figure 5G endothelial neighborhood panels",
        "hypoxia": "Figure 3G or basal hypoxia panels",
    }

    manifest_rows = []
    fraction_rows = []
    score_rows = []
    genes_found_rows = []

    for _, row in pdac_samples.iterrows():
        sample_id = str(row["sample_id"])
        original_name = str(row["original_name"])

        adata, adata_path = read_h5ad(output_root, sample_id)
        coords = get_coords(adata)
        src = expression_source(adata)

        pipeline_region = pipeline_region_crosswalk(adata)
        core_border = core_border_from_pipeline(coords, pipeline_region)

        for label in sorted(set(pipeline_region)):
            fraction_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "kind": "pipeline_region_crosswalk",
                "label": label,
                "fraction": float(np.mean(pipeline_region == label)),
            })

        for label in sorted(set(core_border)):
            fraction_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "kind": "core_border_pipeline",
                "label": label,
                "fraction": float(np.mean(core_border == label)),
            })

        local_path = maps_dir / f"{sample_id}_{original_name}_pipeline_region_crosswalk.png"
        plot_categorical(
            coords,
            pipeline_region,
            f"{sample_id} {original_name} pipeline region crosswalk",
            local_path,
            PIPELINE_REGION_COLORS,
        )

        manifest_rows.append({
            "sample_id": sample_id,
            "original_name": original_name,
            "comparison_kind": "pipeline_region_crosswalk",
            "validation_type": "pure_pipeline_labels",
            "local_validation_map": str(local_path),
            "study_figure_suggestion": study_suggestions["pipeline_region_crosswalk"],
            "study_png": "",
            "side_by_side": "",
            "review_score": "",
            "review_notes": "",
        })

        local_path = maps_dir / f"{sample_id}_{original_name}_core_border_pipeline.png"
        plot_categorical(
            coords,
            core_border,
            f"{sample_id} {original_name} tumor core and border from pipeline labels",
            local_path,
            CORE_BORDER_COLORS,
        )

        manifest_rows.append({
            "sample_id": sample_id,
            "original_name": original_name,
            "comparison_kind": "core_border_pipeline",
            "validation_type": "pure_pipeline_labels_plus_spatial_neighbors",
            "local_validation_map": str(local_path),
            "study_figure_suggestion": study_suggestions["core_border_pipeline"],
            "study_png": "",
            "side_by_side": "",
            "review_score": "",
            "review_notes": "",
        })

        for category in categories_to_plot:
            genes = marker_dict.get(category, [])

            if not genes:
                continue

            score, found = signature_score(src, genes)

            genes_found_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "category": category,
                "n_genes_in_dictionary": len(genes),
                "n_genes_found_in_sample": len(found),
                "genes_found": ";".join(found),
            })

            score_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "category": category,
                "mean_score": float(np.nanmean(score)),
                "median_score": float(np.nanmedian(score)),
                "q90_score": float(np.nanquantile(score, 0.90)),
                "max_score": float(np.nanmax(score)),
                "n_genes_found": len(found),
            })

            local_path = maps_dir / f"{sample_id}_{original_name}_validation_score__{category}.png"
            plot_score(
                coords,
                score,
                f"{sample_id} {original_name} validation score {category}",
                local_path,
            )

            manifest_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "comparison_kind": category,
                "validation_type": "supplement_marker_score_validation_only",
                "local_validation_map": str(local_path),
                "study_figure_suggestion": study_suggestions.get(category, ""),
                "study_png": "",
                "side_by_side": "",
                "review_score": "",
                "review_notes": "",
            })

    manifest = pd.DataFrame(manifest_rows)
    fractions = pd.DataFrame(fraction_rows)
    scores = pd.DataFrame(score_rows)
    genes_found = pd.DataFrame(genes_found_rows)

    manifest_path = out_dir / "pdac_manual_review_candidate_manifest.csv"
    fractions_path = tables_dir / "pdac_pipeline_region_fraction_summary.csv"
    scores_path = tables_dir / "pdac_validation_marker_score_summary.csv"
    genes_found_path = tables_dir / "pdac_validation_marker_genes_found.csv"

    manifest.to_csv(manifest_path, index=False)
    fractions.to_csv(fractions_path, index=False)
    scores.to_csv(scores_path, index=False)
    genes_found.to_csv(genes_found_path, index=False)

    summary_path = out_dir / "pdac_external_validation_summary.txt"

    lines = []
    lines.append("PDAC external validation summary")
    lines.append("")
    lines.append("This is validation only.")
    lines.append("No markers from the PDAC paper were added to the main pipeline.")
    lines.append("No step 05 through step 12 outputs were modified.")
    lines.append("model_input_numeric.csv was not modified.")
    lines.append("")
    lines.append("Pure pipeline validation maps:")
    lines.append("  pipeline_region_crosswalk")
    lines.append("  core_border_pipeline")
    lines.append("")
    lines.append("Supplement marker maps:")
    lines.append("  These are validation score maps only.")
    lines.append("  They are used to compare the same samples against paper biology.")
    lines.append("  They are not canonical labels.")
    lines.append("")
    lines.append("PDAC samples:")
    for _, r in pdac_samples.iterrows():
        lines.append(f"  {r['sample_id']} | {r['original_name']}")
    lines.append("")
    lines.append("Output files:")
    lines.append(str(manifest_path))
    lines.append(str(fractions_path))
    lines.append(str(scores_path))
    lines.append(str(genes_found_path))
    lines.append(str(parsed["dict_csv"]))
    lines.append(str(parsed["dict_json"]))
    lines.append("")
    lines.append("Next step:")
    lines.append("Inspect pipeline_region_crosswalk and core_border_pipeline first.")
    lines.append("Only then create study PNG files for the most relevant paper panels.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "study": "pdac",
        "samples": pdac_samples.to_dict(orient="records"),
        "output_folder": str(out_dir),
        "supplementary_table_folder": str(supp_dir),
        "study_png_dir": str(study_png_dir) if study_png_dir else "",
        "validation_only": True,
    }

    with open(out_dir / "pdac_external_validation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("")
    print("DONE PDAC validation")
    print("Output folder:", out_dir)
    print("Summary:", summary_path)
    print("Manifest:", manifest_path)
    print("Local maps:", maps_dir)


