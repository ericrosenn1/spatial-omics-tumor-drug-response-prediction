"""
Script: 13b_pdac_curated_non_image_audit.py

Purpose:
Run the curated PDAC non image validation audit against existing pipeline labels.

Project context:
This script tests whether curated PDAC marker programs are enriched in the expected
pipeline regions for the already processed PDAC samples. It is validation only.

Important:
The curated markers are not added to the pipeline. They are scored after the pipeline
has already assigned regions, then summarized by existing tumor_like, stroma_like,
immune_like, vascular_like, and unmapped_or_low_signal classes.
"""


# =========================
# Imports
# =========================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse



# =========================
# Project paths and output roots
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

PDAC_OUTPUT_ROOT = OUTPUT_ROOT / "output_13_external_study_validation" / "pdac"
STEP05_H5AD_ROOT = OUTPUT_ROOT / "output_05_build_multi_axis_transcriptome_labels" / "per_sample_h5ad"
MODEL_INPUT_PATH = OUTPUT_ROOT / "output_10_build_model_ready_table" / "model_input_numeric.csv"
FEATURE_MANIFEST_PATH = OUTPUT_ROOT / "output_10_build_model_ready_table" / "feature_manifest.csv"
SAMPLE_MAPPING_PATH = OUTPUT_ROOT / "_external_study_validation" / "sample_to_geo_study_mapping.csv"


# =========================
# Curated validation marker dictionary
# =========================

CURATED_MARKERS = {
    "normal_acinar": [
        "PRSS1", "PRSS2", "CPA1", "CPA2", "CPB1", "CTRC", "CTRB1", "CELA3A", "PNLIP", "AMY2A"
    ],
    "normal_endocrine": [
        "INS", "GCG", "SST", "CHGA", "CHGB", "TTR", "IAPP"
    ],
    "normal_ductal": [
        "KRT7", "KRT19", "SOX9", "MMP7", "CFTR", "SLC4A4"
    ],
    "tumor_classical": [
        "TFF1", "TFF3", "AGR2", "CEACAM5", "CEACAM6", "S100P", "EPCAM", "KRT8", "KRT18", "KRT19"
    ],
    "tumor_basal": [
        "KRT5", "KRT6A", "KRT14", "KRT17", "S100A2", "SERPINB3", "SERPINB4"
    ],
    "tumor_proliferative": [
        "MKI67", "TOP2A", "STMN1", "PCNA", "UBE2C", "TYMS", "AURKA", "BIRC5"
    ],
    "stroma_fibroblast": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FN1", "COL6A2", "COL6A3", "POSTN"
    ],
    "stellate_myofibroblast": [
        "ACTA2", "TAGLN", "MYL9", "TPM2", "CALD1", "CNN1"
    ],
    "endothelial": [
        "PECAM1", "VWF", "KDR", "ENG", "FLT1", "RAMP2", "ESAM"
    ],
    "macrophage": [
        "C1QA", "C1QB", "C1QC", "CD68", "CD163", "LYZ", "MARCO", "APOE", "FCGR3A"
    ],
    "T_cell": [
        "CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "GZMB"
    ],
    "B_plasma": [
        "MS4A1", "CD79A", "CD79B", "IGKC", "JCHAIN", "MZB1", "XBP1"
    ],
}


# =========================
# Expected validation regions
# =========================

EXPECTED_REGION = {
    "normal_acinar": "non_tumor_any",
    "normal_endocrine": "non_tumor_any",
    "normal_ductal": "non_tumor_any",
    "tumor_classical": "tumor_like",
    "tumor_basal": "tumor_like",
    "tumor_proliferative": "tumor_like",
    "stroma_fibroblast": "stroma_like",
    "stellate_myofibroblast": "stroma_like",
    "endothelial": "vascular_like",
    "macrophage": "immune_like",
    "T_cell": "immune_like",
    "B_plasma": "immune_like",
}

REGION_ORDER = [
    "tumor_like",
    "stroma_like",
    "immune_like",
    "vascular_like",
    "unmapped_or_low_signal",
]



# =========================
# Run discovery helpers
# =========================

def latest_pdac_run() -> Path:
    """Return the newest PDAC Step 13 validation run folder."""
    runs = sorted(
        [p for p in PDAC_OUTPUT_ROOT.glob("pdac_validation_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError(f"No pdac_validation_* folder found under {PDAC_OUTPUT_ROOT}")

    return runs[0]


def as_bool(value) -> bool:
    """Interpret string and numeric values as booleans for sample metadata."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}



# =========================
# Sample selection helpers
# =========================

def get_pdac_samples() -> pd.DataFrame:
    """Select final model PDAC samples from the sample mapping table."""
    if not SAMPLE_MAPPING_PATH.exists():
        raise FileNotFoundError(SAMPLE_MAPPING_PATH)

    df = pd.read_csv(SAMPLE_MAPPING_PATH)

    needed = ["sample_id", "dataset_id", "cancer_type", "original_name", "in_final_model_input"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError("Sample mapping missing columns: " + ", ".join(missing))

    pdac = df[
        (df["dataset_id"].astype(str) == "GSE282302")
        & (df["cancer_type"].astype(str).str.lower() == "pdac")
        & (df["in_final_model_input"].apply(as_bool))
    ].copy()

    if len(pdac) == 0:
        raise RuntimeError("No final model PDAC samples found for GSE282302.")

    return pdac[needed].sort_values("sample_id").reset_index(drop=True)



# =========================
# Pipeline region crosswalk helpers
# =========================

def choose_structure_column(obs: pd.DataFrame):
    """Choose the best available structure label column from AnnData observations."""
    for col in [
        "structure_region_label_smoothed",
        "structure_region_label",
        "structure_dominant_label_raw",
        "structure_dominant_label",
    ]:
        if col in obs.columns:
            return col
    return None


def pipeline_region_crosswalk(adata) -> np.ndarray:
    """Collapse pipeline structure labels into broad validation regions."""
    col = choose_structure_column(adata.obs)

    if col is None:
        return np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    labels = adata.obs[col].astype(str).str.lower()
    out = np.array(["unmapped_or_low_signal"] * adata.n_obs, dtype=object)

    stroma = labels.str.contains("stromal|ecm", na=False).to_numpy()
    immune = labels.str.contains("t_cell|immune|plasma|myeloid|macrophage", na=False).to_numpy()
    vascular = labels.str.contains("vascular|angiogenic|endothelial", na=False).to_numpy()
    tumor = labels.str.contains("tumor", na=False).to_numpy()

    out[stroma] = "stroma_like"
    out[immune] = "immune_like"
    out[vascular] = "vascular_like"
    out[tumor] = "tumor_like"

    return out



# =========================
# Expression and marker scoring helpers
# =========================

def expression_source(adata):
    """Return raw AnnData expression when available, otherwise active expression."""
    if adata.raw is not None:
        src = adata.raw.to_adata()
    else:
        src = adata.copy()

    src.var_names = src.var_names.astype(str)
    src.var_names_make_unique()
    return src


def gene_vector(src, gene: str):
    """Extract one gene expression vector from an AnnData expression source."""
    gene = str(gene).upper()
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
    """Convert expression values into percentile ranks from zero to one."""
    s = pd.Series(values)

    if s.notna().sum() <= 1:
        return np.zeros(len(s), dtype=float)

    if s.nunique(dropna=True) <= 1:
        return np.full(len(s), 0.5, dtype=float)

    return s.rank(pct=True).fillna(0.0).to_numpy(dtype=float)


def signature_score(src, genes):
    """Score a curated marker category using genes present in the sample."""
    values = []
    found = []

    for gene in genes:
        vec = gene_vector(src, gene)

        if vec is not None:
            values.append(rank01(vec))
            found.append(gene)

    if not values:
        return np.zeros(src.n_obs, dtype=float), found

    return np.nanmean(np.vstack(values), axis=0), found



# =========================
# Region enrichment summaries
# =========================

def expected_top_is_correct(category: str, top_region: str) -> bool:
    """Check whether a top region matches the expected validation region."""
    expected = EXPECTED_REGION.get(category, "")

    if expected == "non_tumor_any":
        return top_region != "tumor_like"

    return top_region == expected


def region_fraction_table(sample_id: str, original_name: str, region: np.ndarray) -> list[dict]:
    """Summarize spot fractions for each pipeline validation region."""
    rows = []

    for label in REGION_ORDER:
        mask = region == label

        rows.append({
            "sample_id": sample_id,
            "original_name": original_name,
            "pipeline_region": label,
            "n_spots": int(mask.sum()),
            "fraction": float(mask.mean()),
        })

    return rows


def score_category_by_region(sample_id, original_name, category, score, region):
    """Summarize one marker score across pipeline validation regions."""
    rows = []

    for label in REGION_ORDER:
        mask = region == label

        if int(mask.sum()) == 0:
            rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "category": category,
                "expected_region": EXPECTED_REGION.get(category, ""),
                "pipeline_region": label,
                "n_spots_region": 0,
                "fraction_spots_region": 0.0,
                "mean_score": np.nan,
                "median_score": np.nan,
                "q90_score": np.nan,
                "mean_score_rest": np.nan,
                "delta_vs_rest": np.nan,
            })
            continue

        rest = ~mask
        mean_score = float(np.nanmean(score[mask]))
        median_score = float(np.nanmedian(score[mask]))
        q90_score = float(np.nanquantile(score[mask], 0.90))

        if int(rest.sum()) > 0:
            mean_rest = float(np.nanmean(score[rest]))
            delta = mean_score - mean_rest
        else:
            mean_rest = np.nan
            delta = np.nan

        rows.append({
            "sample_id": sample_id,
            "original_name": original_name,
            "category": category,
            "expected_region": EXPECTED_REGION.get(category, ""),
            "pipeline_region": label,
            "n_spots_region": int(mask.sum()),
            "fraction_spots_region": float(mask.mean()),
            "mean_score": mean_score,
            "median_score": median_score,
            "q90_score": q90_score,
            "mean_score_rest": mean_rest,
            "delta_vs_rest": float(delta) if np.isfinite(delta) else np.nan,
        })

    return rows


def summarize_top_region(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Find the top scoring pipeline region for each sample and marker category."""
    top_rows = []

    group_cols = ["sample_id", "original_name", "category"]

    for keys, sub in enrichment.groupby(group_cols, dropna=False):
        sub2 = sub.dropna(subset=["mean_score"]).copy()

        if len(sub2) == 0:
            continue

        sub2 = sub2.sort_values("mean_score", ascending=False)
        top = sub2.iloc[0]

        if len(sub2) > 1:
            second = sub2.iloc[1]
            top_minus_second = float(top["mean_score"] - second["mean_score"])
            second_region = str(second["pipeline_region"])
        else:
            top_minus_second = np.nan
            second_region = ""

        category = str(top["category"])
        top_region = str(top["pipeline_region"])

        top_rows.append({
            "sample_id": str(top["sample_id"]),
            "original_name": str(top["original_name"]),
            "category": category,
            "expected_region": EXPECTED_REGION.get(category, ""),
            "top_pipeline_region_by_mean_score": top_region,
            "second_pipeline_region_by_mean_score": second_region,
            "top_region_mean_score": float(top["mean_score"]),
            "top_minus_second_region_mean_score": top_minus_second,
            "top_region_delta_vs_rest": float(top["delta_vs_rest"]) if pd.notna(top["delta_vs_rest"]) else np.nan,
            "top_region_n_spots": int(top["n_spots_region"]),
            "expected_region_is_top": expected_top_is_correct(category, top_region),
            "interpretation": "",
        })

    return pd.DataFrame(top_rows)


def summarize_category_pass_rates(top: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation pass rates by marker category."""
    rows = []

    for category, sub in top.groupby("category", dropna=False):
        rows.append({
            "category": category,
            "expected_region": EXPECTED_REGION.get(str(category), ""),
            "n_samples": int(len(sub)),
            "n_expected_top": int(sub["expected_region_is_top"].sum()),
            "fraction_expected_top": float(sub["expected_region_is_top"].mean()) if len(sub) else np.nan,
            "top_regions_observed": ";".join(
                f"{k}:{v}" for k, v in sub["top_pipeline_region_by_mean_score"].value_counts().to_dict().items()
            ),
        })

    return pd.DataFrame(rows).sort_values(["fraction_expected_top", "category"], ascending=[False, True])


def summarize_sample_pass_rates(top: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation pass rates by sample."""
    rows = []

    for sample_id, sub in top.groupby("sample_id", dropna=False):
        original_name = str(sub["original_name"].iloc[0])

        tumor_categories = ["tumor_classical", "tumor_basal", "tumor_proliferative"]
        context_categories = ["stroma_fibroblast", "stellate_myofibroblast", "endothelial", "macrophage", "T_cell", "B_plasma"]
        normal_categories = ["normal_acinar", "normal_endocrine", "normal_ductal"]

        def frac_for(cats):
            """Document frac for within the 13c_pdac validation workflow."""
            ss = sub[sub["category"].isin(cats)]
            if len(ss) == 0:
                return np.nan
            return float(ss["expected_region_is_top"].mean())

        rows.append({
            "sample_id": sample_id,
            "original_name": original_name,
            "n_categories": int(len(sub)),
            "n_expected_top": int(sub["expected_region_is_top"].sum()),
            "fraction_expected_top": float(sub["expected_region_is_top"].mean()) if len(sub) else np.nan,
            "tumor_category_fraction_expected_top": frac_for(tumor_categories),
            "context_category_fraction_expected_top": frac_for(context_categories),
            "normal_category_fraction_non_tumor_top": frac_for(normal_categories),
        })

    return pd.DataFrame(rows).sort_values("sample_id")



# =========================
# Model feature subset writers
# =========================

def write_model_subsets(audit_dir: Path, pdac_samples: list[str]):
    """Write PDAC specific model input and feature manifest subsets for audit."""
    if MODEL_INPUT_PATH.exists():
        model = pd.read_csv(MODEL_INPUT_PATH)
        model["sample_id"] = model["sample_id"].astype(str)
        subset = model[model["sample_id"].isin(pdac_samples)].copy()
        subset.to_csv(audit_dir / "pdac_curated_model_input_numeric_subset.csv", index=False)

        keywords = [
            "tumor", "stromal", "ecm", "immune", "myeloid", "macrophage",
            "t_cell", "plasma", "vascular", "angiogenic", "hypoxic",
            "metabolic", "proliferative", "boundary", "core", "accessibility",
            "hotspot", "motif", "distance", "contact",
        ]

        keep = [
            c for c in subset.columns
            if c == "sample_id" or any(k in c.lower() for k in keywords)
        ]

        subset[keep].to_csv(audit_dir / "pdac_curated_relevant_pipeline_feature_subset.csv", index=False)

    if FEATURE_MANIFEST_PATH.exists():
        manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
        manifest.to_csv(audit_dir / "pdac_curated_feature_manifest_copy.csv", index=False)



# =========================
# Main validation workflow
# =========================

def main():
    """Run the curated PDAC non image validation audit and write output tables."""
    latest_run = latest_pdac_run()
    audit_dir = latest_run / "curated_non_image_validation_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    pdac_samples = get_pdac_samples()

    enrichment_rows = []
    gene_rows = []
    fraction_rows = []

    for _, sample in pdac_samples.iterrows():
        sample_id = str(sample["sample_id"])
        original_name = str(sample["original_name"])

        h5ad_path = STEP05_H5AD_ROOT / f"{sample_id}_with_multi_axis_transcriptome_labels.h5ad"

        if not h5ad_path.exists():
            raise FileNotFoundError(h5ad_path)

        adata = ad.read_h5ad(h5ad_path)
        src = expression_source(adata)
        region = pipeline_region_crosswalk(adata)

        fraction_rows.extend(region_fraction_table(sample_id, original_name, region))

        for category, genes in CURATED_MARKERS.items():
            score, found = signature_score(src, genes)

            gene_rows.append({
                "sample_id": sample_id,
                "original_name": original_name,
                "category": category,
                "n_genes_dictionary": len(genes),
                "n_genes_found": len(found),
                "genes_found": ";".join(found),
                "genes_missing": ";".join([g for g in genes if g not in found]),
            })

            enrichment_rows.extend(
                score_category_by_region(
                    sample_id=sample_id,
                    original_name=original_name,
                    category=category,
                    score=score,
                    region=region,
                )
            )

    enrichment = pd.DataFrame(enrichment_rows)
    genes_found = pd.DataFrame(gene_rows)
    fractions = pd.DataFrame(fraction_rows)
    top = summarize_top_region(enrichment)
    category_pass = summarize_category_pass_rates(top)
    sample_pass = summarize_sample_pass_rates(top)

    marker_json_path = audit_dir / "pdac_curated_marker_dictionary.json"
    with open(marker_json_path, "w", encoding="utf-8") as f:
        json.dump(CURATED_MARKERS, f, indent=2)

    marker_table_rows = []
    for category, genes in CURATED_MARKERS.items():
        for rank, gene in enumerate(genes, start=1):
            marker_table_rows.append({
                "category": category,
                "rank": rank,
                "gene": gene,
                "expected_region": EXPECTED_REGION.get(category, ""),
            })

    pd.DataFrame(marker_table_rows).to_csv(
        audit_dir / "pdac_curated_marker_dictionary.csv",
        index=False,
    )

    fractions.to_csv(
        audit_dir / "pdac_curated_pipeline_region_fraction_summary.csv",
        index=False,
    )

    genes_found.to_csv(
        audit_dir / "pdac_curated_marker_genes_found_by_sample.csv",
        index=False,
    )

    enrichment.to_csv(
        audit_dir / "pdac_curated_marker_score_enrichment_by_pipeline_region.csv",
        index=False,
    )

    top.to_csv(
        audit_dir / "pdac_curated_marker_score_top_pipeline_region_summary.csv",
        index=False,
    )

    category_pass.to_csv(
        audit_dir / "pdac_curated_category_pass_rate_summary.csv",
        index=False,
    )

    sample_pass.to_csv(
        audit_dir / "pdac_curated_sample_pass_rate_summary.csv",
        index=False,
    )

    write_model_subsets(
        audit_dir=audit_dir,
        pdac_samples=pdac_samples["sample_id"].astype(str).tolist(),
    )

    lines = []
    lines.append("PDAC curated non image validation audit")
    lines.append("")
    lines.append("This audit is validation only.")
    lines.append("It does not add labels to the pipeline.")
    lines.append("It does not modify steps 05 through 12.")
    lines.append("It does not modify model_input_numeric.csv.")
    lines.append("")
    lines.append("Logic:")
    lines.append("1. Existing pipeline labels are collapsed into tumor_like, stroma_like, immune_like, vascular_like, and unmapped_or_low_signal.")
    lines.append("2. Curated PDAC marker scores are computed after that.")
    lines.append("3. Marker scores are tested for enrichment inside the existing pipeline regions.")
    lines.append("")
    lines.append("PDAC samples:")
    for _, sample in pdac_samples.iterrows():
        lines.append(f"  {sample['sample_id']} | {sample['original_name']}")
    lines.append("")
    lines.append("Pipeline region fractions:")
    lines.append(fractions.to_string(index=False))
    lines.append("")
    lines.append("Top region by curated marker score:")
    lines.append(top.to_string(index=False))
    lines.append("")
    lines.append("Category pass rates:")
    lines.append(category_pass.to_string(index=False))
    lines.append("")
    lines.append("Sample pass rates:")
    lines.append(sample_pass.to_string(index=False))
    lines.append("")
    lines.append("Output folder:")
    lines.append(str(audit_dir))

    summary_path = audit_dir / "pdac_curated_non_image_validation_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("PDAC curated non image validation audit complete.")
    print("Latest PDAC Step 13 run:")
    print(latest_run)
    print("")
    print("Audit folder:")
    print(audit_dir)
    print("")
    print("Open these files first:")
    print(summary_path)
    print(audit_dir / "pdac_curated_marker_score_top_pipeline_region_summary.csv")
    print(audit_dir / "pdac_curated_category_pass_rate_summary.csv")
    print(audit_dir / "pdac_curated_sample_pass_rate_summary.csv")



# =========================
# Command line entry point
# =========================

if __name__ == "__main__":
    main()


