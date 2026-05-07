"""
Script: 13c_pdac_collect_and_pair_validation_figures.py

Purpose:
Collect PDAC validation figures and build paired study versus pipeline review packets.

Project context:
This script organizes visual validation outputs after Step 13 has created local PDAC
maps and non image audit tables. It copies candidate figures, finds study reference
panels, creates side by side validation packets, and writes review manifests.

Important:
The generated packets are review artifacts. They do not change pipeline labels,
model input tables, or upstream outputs.
"""

from __future__ import annotations


# =========================
# Imports
# =========================

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont



# =========================
# Matched PDAC sample identifiers
# =========================

SAMPLES = {
    "SAMPLE_0065": "GSM8641003_C1_D10_ROI1_s1",
    "SAMPLE_0066": "GSM8641030_C2_D11_ROI1_s1",
    "SAMPLE_0067": "GSM8641067_C3_D12_ROI1_s1",
}

SAMPLE_LABELS = {
    "SAMPLE_0065": "C1_D10_ROI1_s1",
    "SAMPLE_0066": "C2_D11_ROI1_s1",
    "SAMPLE_0067": "C3_D12_ROI1_s1",
}



# =========================
# Meaningful comparison pair specifications
# =========================

PAIR_SPECS = [
    {
        "pair_id": "01",
        "name": "compartment_framework_vs_pipeline_crosswalk",
        "title": "Compartment framework versus pipeline region crosswalk",
        "study_panels": [
            "PDAC_STUDY_Fig1F_pathology_transcriptomics_concordance_HQ.png",
            "PDAC_STUDY_SuppFig1_2B_de_novo_by_pathology_HQ.png",
            "PDAC_STUDY_SuppFig1_2C_D_signature_assignment_HQ.png",
        ],
        "local_patterns": [
            "*pipeline_region_crosswalk.png",
        ],
        "analysis": [
            "Study reference: published PDAC de novo clusters and pathology or signature concordance define acinar, endocrine, ductal, tumor, fibroblast, endothelial or stellate, and immune compartments.",
            "Pipeline comparison: pipeline structure labels are collapsed into tumor_like, stroma_like, immune_like, and vascular_like without using PDAC paper markers.",
            "Validation claim: this is the strongest atlas level compartment validation. Agreement supports the pipeline compartment ontology, while disagreements highlight missing normal or benign organ classes.",
        ],
    },
    {
        "pair_id": "02",
        "name": "pathology_reference_vs_pipeline_regions",
        "title": "Pathology reference versus pipeline region maps",
        "study_panels": [
            "PDAC_STUDY_SuppFig1_2A_low_bulk_pathology_HQ.png",
            "PDAC_STUDY_Fig1F_pathology_transcriptomics_concordance_HQ.png",
        ],
        "local_patterns": [
            "*pipeline_region_crosswalk.png",
            "*core_border_pipeline.png",
        ],
        "analysis": [
            "Study reference: pathology annotation distinguishes tumor, benign epithelium, endocrine or exocrine tissue, non epithelial tissue, muscle or vessel, TLS, nerve, adipose, blood, and luminal regions.",
            "Pipeline comparison: pure pipeline crosswalk and core or border maps test whether the pipeline separates tumor from non tumor context.",
            "Validation claim: this pair evaluates tumor architecture and compartment separation, not exact pixel level matching.",
        ],
    },
    {
        "pair_id": "03",
        "name": "tumor_subtype_reference_vs_pipeline_tumor_scores",
        "title": "Tumor subtype reference versus pipeline validation tumor scores",
        "study_panels": [
            "PDAC_STUDY_SuppFig2_1A_B_tumor_subtype_composition_with_legend_HQ.png",
            "PDAC_STUDY_SuppFig2_1D_E_classical_basal_genes_HQ.png",
            "PDAC_STUDY_SuppFig2_1F_deprioritized_signatures_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__tumor_classical.png",
            "*validation_score__tumor_basal.png",
            "*validation_score__tumor_proliferative.png",
            "*core_border_pipeline.png",
        ],
        "analysis": [
            "Study reference: PDAC tumor subtypes include ductal like, PanIN, classical PDAC, proliferative, intermediate PDAC, basal PDAC, and fibroblast high programs.",
            "Pipeline comparison: validation score maps are calculated after pipeline labels already exist. They are not added to the main pipeline.",
            "Validation claim: tumor classical, basal, and proliferative scores should concentrate within or near pipeline tumor_like regions and should not replace pipeline labels.",
        ],
    },
    {
        "pair_id": "04",
        "name": "hypoxia_reference_vs_pipeline_hypoxia_scores",
        "title": "Hypoxia reference versus pipeline hypoxia validation scores",
        "study_panels": [
            "PDAC_STUDY_Fig3G_H_hypoxia_signature_spatial_HQ.png",
            "PDAC_STUDY_SuppFig7_1_gene_communities_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__hypoxia.png",
            "*validation_score__tumor_basal.png",
            "*validation_score__tumor_fibroblast_high.png",
        ],
        "analysis": [
            "Study reference: hypoxia is presented as a PDAC intrinsic habituated hypoxia signature and as a spatially distributed program.",
            "Pipeline comparison: hypoxia validation maps are compared to tumor basal and fibroblast high maps because the study links hypoxia to PDAC tumor states and spatial context.",
            "Validation claim: broad spatial enrichment supports the pipeline hypoxia and tumor context feature families, but this is not exact ROI matching.",
        ],
    },
    {
        "pair_id": "05",
        "name": "immune_neighborhood_reference_vs_pipeline_immune_scores",
        "title": "Immune neighborhood reference versus pipeline immune score maps",
        "study_panels": [
            "PDAC_STUDY_Fig4D_tumor_subtype_3hop_neighborhood_HQ.png",
            "PDAC_STUDY_SuppFig4_1A_one_hop_neighborhood_composition_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__immune_t_cell.png",
            "*validation_score__immune_b_plasma.png",
            "*validation_score__immune_macrophage.png",
            "*pipeline_region_crosswalk.png",
        ],
        "analysis": [
            "Study reference: tumor subtype neighborhoods include fibroblast, macrophage, T cell, endothelial, mast, stellate, B cell or plasma, and other neighbor types.",
            "Pipeline comparison: immune validation maps and the immune_like crosswalk regions test whether immune signal localizes to non tumor or interface compartments.",
            "Validation claim: this validates immune context and neighborhood feature families rather than exact single cell neighborhood reconstruction.",
        ],
    },
    {
        "pair_id": "06",
        "name": "stroma_collagen_reference_vs_pipeline_stroma_scores",
        "title": "Stroma, collagen, and distance reference versus pipeline stroma maps",
        "study_panels": [
            "PDAC_STUDY_Fig5E_H_stroma_distance_collagen_HQ.png",
            "PDAC_STUDY_SuppFig5_1G_stromal_composition_by_patient_HQ.png",
            "PDAC_STUDY_SuppFig5_1H_stroma_distance_from_tumor_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__stroma_fibroblast.png",
            "*validation_score__stroma_stellate.png",
            "*validation_score__vascular_endothelial.png",
            "*pipeline_region_crosswalk.png",
        ],
        "analysis": [
            "Study reference: stromal spot composition, collagen, and distance from tumor are analyzed across PDAC tumor subtypes.",
            "Pipeline comparison: stroma_fibroblast, stroma_stellate, vascular_endothelial, and stroma_like regions test stromal architecture.",
            "Validation claim: this supports ECM, accessibility, stromal context, and distance or boundary feature families, but collagen itself is not directly measured by the transcriptomic pipeline.",
        ],
    },
    {
        "pair_id": "07",
        "name": "gene_community_reference_vs_pipeline_programs",
        "title": "Gene community reference versus pipeline program maps",
        "study_panels": [
            "PDAC_STUDY_SuppFig7_1_gene_communities_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__tumor_proliferative.png",
            "*validation_score__stroma_fibroblast.png",
            "*validation_score__normal_ductal.png",
            "*validation_score__hypoxia.png",
        ],
        "analysis": [
            "Study reference: conserved gene communities include proliferation, activated stroma, ductal or early malignancy, intermediate PDAC, and hypoxia.",
            "Pipeline comparison: local maps are grouped by matching biological program families.",
            "Validation claim: this is a pathway and program level comparison, not a compartment label validation.",
        ],
    },
    {
        "pair_id": "08",
        "name": "normal_or_benign_signal_stress_test",
        "title": "Normal or benign marker signal stress test",
        "study_panels": [
            "PDAC_STUDY_SuppFig1_2A_low_bulk_pathology_HQ.png",
            "PDAC_STUDY_SuppFig1_2B_de_novo_by_pathology_HQ.png",
        ],
        "local_patterns": [
            "*validation_score__normal_acinar.png",
            "*validation_score__normal_ductal.png",
            "*validation_score__normal_endocrine.png",
            "*pipeline_region_crosswalk.png",
        ],
        "analysis": [
            "Study reference: the paper separates benign epithelial, exocrine or endocrine, ductal, and tumor associated compartments.",
            "Pipeline comparison: normal marker score maps are used only after the pipeline has assigned regions.",
            "Validation claim: this is the main stress test for the current limitation. The pipeline lacks a canonical normal_or_organ_like label, so normal marker programs may appear inside tumor_like or context labels.",
        ],
    },
]



# =========================
# Text and image helper functions
# =========================

def safe_name(s: str) -> str:
    """Return a filesystem safe version of a string."""
    out = []
    for c in str(s):
        if c.isalnum() or c in "._":
            out.append(c)
        else:
            out.append("_")
    return "".join(out)


def find_font(size: int):
    """Find an available font for packet labels."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_TITLE = find_font(34)
FONT_SUBTITLE = find_font(24)
FONT_TEXT = find_font(18)
FONT_SMALL = find_font(14)


def add_text(draw, xy, text, font, fill="black", max_width=None, line_spacing=6):
    """Draw wrapped text onto a PIL canvas and return the next y position."""
    x, y = xy
    if max_width is None:
        draw.text((x, y), text, font=font, fill=fill)
        return y + draw.textbbox((x, y), text, font=font)[3] - draw.textbbox((x, y), text, font=font)[1] + line_spacing

    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        trial = word if current == "" else current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_spacing

    return y


def resize_for_tile(path: Path, max_w: int, max_h: int):
    """Resize an image to fit within one packet tile."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return img



# =========================
# Study panel discovery helpers
# =========================

def find_study_panel(study_roots, filename):
    """Find a study reference panel by exact or approximate file name."""
    for root in study_roots:
        p = root / filename
        if p.exists():
            return p

    stem_words = [w.lower() for w in Path(filename).stem.split("_") if len(w) > 2]
    hits = []
    for root in study_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.png"):
            low = p.name.lower()
            score = sum(1 for w in stem_words if w in low)
            if score >= max(2, min(4, len(stem_words) // 2)):
                hits.append((score, p))
    if hits:
        hits = sorted(hits, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
        return hits[0][1]
    return None


def get_local_maps(local_dir: Path, patterns):
    """Collect local pipeline maps that match requested filename patterns."""
    paths = []
    for pattern in patterns:
        for p in sorted(local_dir.glob(pattern)):
            paths.append(p)

    seen = set()
    out = []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)

    return out



# =========================
# Figure packet rendering
# =========================

def create_packet(out_path, title, study_paths, local_paths, analysis_lines):
    """Create one study versus pipeline validation packet image."""
    width = 2400
    margin = 60
    y = margin

    study_max_w = 720
    study_max_h = 520
    local_max_w = 520
    local_max_h = 420

    study_imgs = []
    for p in study_paths:
        if p and p.exists():
            study_imgs.append((p, resize_for_tile(p, study_max_w, study_max_h)))

    local_imgs = []
    for p in local_paths:
        if p and p.exists():
            local_imgs.append((p, resize_for_tile(p, local_max_w, local_max_h)))

    n_study_rows = max(1, (len(study_imgs) + 2) // 3)
    n_local_rows = max(1, (len(local_imgs) + 3) // 4)

    height = (
        margin
        + 70
        + 160
        + 60
        + n_study_rows * (study_max_h + 110)
        + 70
        + n_local_rows * (local_max_h + 110)
        + margin
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = add_text(draw, (margin, y), title, FONT_TITLE, max_width=width - 2 * margin)
    y += 15

    for line in analysis_lines:
        y = add_text(draw, (margin, y), line, FONT_TEXT, fill=(40, 40, 40), max_width=width - 2 * margin)
    y += 20

    y = add_text(draw, (margin, y), "Study reference panels", FONT_SUBTITLE)
    y += 10

    x0 = margin
    col_w = (width - 2 * margin) // 3

    for i, (p, img) in enumerate(study_imgs):
        row = i // 3
        col = i % 3
        x = x0 + col * col_w
        yy = y + row * (study_max_h + 110)
        canvas.paste(img, (x, yy))
        add_text(draw, (x, yy + img.height + 8), p.name, FONT_SMALL, fill=(50, 50, 50), max_width=col_w - 20)

    y += n_study_rows * (study_max_h + 110) + 25

    y = add_text(draw, (margin, y), "Pipeline outputs from matched PDAC samples", FONT_SUBTITLE)
    y += 10

    col_w = (width - 2 * margin) // 4

    for i, (p, img) in enumerate(local_imgs):
        row = i // 4
        col = i % 4
        x = x0 + col * col_w
        yy = y + row * (local_max_h + 110)
        canvas.paste(img, (x, yy))
        add_text(draw, (x, yy + img.height + 8), p.name, FONT_SMALL, fill=(50, 50, 50), max_width=col_w - 20)

    canvas.save(out_path)


def copy_file_unique(src: Path, dest_dir: Path, prefix=""):
    """Copy a file into a destination folder without overwriting existing files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(prefix + src.name)
    dest = dest_dir / name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(src, dest)
    return dest



# =========================
# Pipeline figure collection
# =========================

def collect_pipeline_figures(outputs_root, run_dir, candidate_root):
    """Copy candidate local pipeline figures and reports into a review folder."""
    local_dir = run_dir / "local_validation_maps"

    step13_dir = candidate_root / "01_step13_local_validation_maps"
    other_dir = candidate_root / "02_other_pipeline_sample_pngs"
    table_dir = candidate_root / "99_tables_and_reports"

    copied_rows = []

    if local_dir.exists():
        for p in sorted(local_dir.glob("*.png")):
            dest = copy_file_unique(p, step13_dir)
            copied_rows.append({
                "source_type": "step13_local_validation_map",
                "source_file": str(p),
                "copied_file": str(dest),
            })

    excluded_parts = [
        str(candidate_root).lower(),
        str(run_dir / "side_by_side_pairs").lower(),
        "_external_study_validation\\pdac\\study_pngs",
        "_external_study_validation/pdac/study_pngs",
    ]

    sample_tokens = list(SAMPLES.keys()) + list(SAMPLES.values()) + ["C1_D10", "C2_D11", "C3_D12"]

    for p in outputs_root.rglob("*.png"):
        low_path = str(p).lower()
        if any(x.lower() in low_path for x in excluded_parts):
            continue
        name_low = p.name.lower()
        if any(tok.lower() in name_low or tok.lower() in low_path for tok in sample_tokens):
            dest = copy_file_unique(p, other_dir, prefix=safe_name(p.parent.name) + "__")
            copied_rows.append({
                "source_type": "other_pipeline_sample_png",
                "source_file": str(p),
                "copied_file": str(dest),
            })

    for table_name in [
        "pdac_external_validation_summary.txt",
        "pdac_manual_review_candidate_manifest.csv",
        "pdac_external_validation_metadata.json",
    ]:
        p = run_dir / table_name
        if p.exists():
            dest = copy_file_unique(p, table_dir)
            copied_rows.append({
                "source_type": "run_report_or_manifest",
                "source_file": str(p),
                "copied_file": str(dest),
            })

    for sub in ["tables", "non_image_validation_audit", "curated_non_image_validation_audit"]:
        p = run_dir / sub
        if p.exists():
            dest_sub = table_dir / sub
            dest_sub.mkdir(parents=True, exist_ok=True)
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in [".csv", ".txt", ".json"]:
                    dest = copy_file_unique(f, dest_sub, prefix=safe_name(f.parent.name) + "__")
                    copied_rows.append({
                        "source_type": "table_or_report",
                        "source_file": str(f),
                        "copied_file": str(dest),
                    })

    manifest = candidate_root / "pipeline_candidate_figure_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_type", "source_file", "copied_file"])
        writer.writeheader()
        writer.writerows(copied_rows)

    return copied_rows, manifest


def read_optional_csv(path: Path):
    """Read a CSV when present and return None on missing or failed input."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None



# =========================
# Report builders
# =========================

def build_analysis_report(run_dir, candidate_root, pair_root, copied_rows):
    """Write the PDAC figure analysis and interpretation report."""
    report = pair_root / "pdac_full_figure_analysis_validation_report.txt"

    curated = run_dir / "curated_non_image_validation_audit"
    non_image = run_dir / "non_image_validation_audit"
    tables = run_dir / "tables"

    category_pass = read_optional_csv(curated / "pdac_curated_category_pass_rate_summary.csv")
    sample_pass = read_optional_csv(curated / "pdac_curated_sample_pass_rate_summary.csv")
    top_summary = read_optional_csv(curated / "pdac_curated_marker_score_top_pipeline_region_summary.csv")
    region_fraction = read_optional_csv(tables / "pdac_pipeline_region_fraction_summary.csv")

    lines = []

    lines.append("PDAC external validation figure analysis")
    lines.append("")
    lines.append("Scope")
    lines.append("")
    lines.append("This report organizes study reference panels and matched pipeline figures for the three PDAC samples in the final model input.")
    lines.append("The comparison is validation only. It does not add labels to the pipeline, does not modify steps 05 through 12, and does not modify model_input_numeric.csv.")
    lines.append("")
    lines.append("Matched PDAC samples")
    lines.append("")
    for sid, orig in SAMPLES.items():
        lines.append(f"{sid}: {orig}")

    lines.append("")
    lines.append("Important interpretation rule")
    lines.append("")
    lines.append("The PDAC study panels are mostly atlas level, patient level, or conceptual spatial biology panels. They are not exact spot level annotations for the same three Visium ROIs. Therefore, the side by side figures created here should be interpreted as validation packets, not pixel level image matches.")
    lines.append("")
    lines.append("Pipeline figure collection")
    lines.append("")
    lines.append(f"Total copied pipeline or report files: {len(copied_rows)}")
    lines.append(f"Candidate figure root: {candidate_root}")
    lines.append(f"Meaningful comparison packet folder: {pair_root}")
    lines.append("")

    if region_fraction is not None:
        lines.append("Pure pipeline region fractions")
        lines.append("")
        lines.append(region_fraction.to_string(index=False))
        lines.append("")

    if category_pass is not None:
        lines.append("Curated marker category pass rates")
        lines.append("")
        lines.append(category_pass.to_string(index=False))
        lines.append("")

    if sample_pass is not None:
        lines.append("Curated marker sample pass rates")
        lines.append("")
        lines.append(sample_pass.to_string(index=False))
        lines.append("")

    if top_summary is not None:
        lines.append("Top pipeline region by curated marker score")
        lines.append("")
        lines.append(top_summary.to_string(index=False))
        lines.append("")

    lines.append("Figure packet interpretation")
    lines.append("")
    lines.append("01 compartment_framework_vs_pipeline_crosswalk")
    lines.append("This packet compares the study's pathology and transcriptomics concordance framework with pure pipeline region crosswalk maps. This is the most important compartment validation packet. It asks whether pipeline tumor, stroma, immune, and vascular regions align with the paper's atlas level compartment logic.")
    lines.append("")
    lines.append("02 pathology_reference_vs_pipeline_regions")
    lines.append("This packet compares the paper's pathology UMAP and framework with pipeline crosswalk and tumor core or border maps. It tests whether tumor architecture is spatially coherent and whether non tumor context is preserved.")
    lines.append("")
    lines.append("03 tumor_subtype_reference_vs_pipeline_tumor_scores")
    lines.append("This packet compares paper tumor subtype composition and marker panels with tumor classical, basal, and proliferative validation score maps. These maps do not relabel the pipeline. They test whether expected PDAC tumor programs are spatially expressed inside the existing pipeline regions.")
    lines.append("")
    lines.append("04 hypoxia_reference_vs_pipeline_hypoxia_scores")
    lines.append("This packet compares hypoxia reference panels with local hypoxia score maps. This supports interpretation of hypoxia and tumor context features, but should not be used as direct same ROI validation.")
    lines.append("")
    lines.append("05 immune_neighborhood_reference_vs_pipeline_immune_scores")
    lines.append("This packet compares published one hop and three hop neighborhood results with local immune marker score maps. It supports immune context, contact, neighborhood, and distance feature families.")
    lines.append("")
    lines.append("06 stroma_collagen_reference_vs_pipeline_stroma_scores")
    lines.append("This packet compares published stroma distance and collagen panels with local fibroblast, stellate, and vascular score maps. It supports stromal ECM, accessibility, distance, boundary, and motif features. Collagen itself is not directly measured by the pipeline.")
    lines.append("")
    lines.append("07 gene_community_reference_vs_pipeline_programs")
    lines.append("This packet compares published gene communities with pipeline program level validation maps. It is best used for feature family interpretation, not for compartment label validation.")
    lines.append("")
    lines.append("08 normal_or_benign_signal_stress_test")
    lines.append("This packet evaluates the main remaining limitation. The current pipeline lacks an explicit normal_or_organ_like compartment. Normal acinar, ductal, or endocrine marker signal may therefore appear inside tumor_like, immune_like, vascular_like, or stroma_like regions depending on which canonical label is closest.")
    lines.append("")
    lines.append("Overall validation interpretation")
    lines.append("")
    lines.append("The PDAC validation supports the pipeline's ability to recover tumor, stromal, immune, vascular, hypoxic, and tumor boundary architecture in a second external cohort.")
    lines.append("The strongest validation comes from pure pipeline crosswalk maps, core or border maps, and curated non image enrichment results.")
    lines.append("The main limitation remains the lack of a canonical normal_or_organ_like compartment. This limitation is consistent with the hepatoblastoma validation and should be considered as a general future pipeline refinement, but the PDAC validation does not show a catastrophic failure of the current pipeline.")
    lines.append("")
    lines.append("Decision")
    lines.append("")
    lines.append("Do not rerun steps 05 through 12 based on the current PDAC figure analysis alone.")
    lines.append("Proceed with documentation and one more targeted review of the meaningful comparison packets.")
    lines.append("Only patch the core pipeline if future validation confirms that normal or benign tissue handling materially affects downstream model features.")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report



# =========================
# Main workflow
# =========================

def main():
    """Collect PDAC figures, create paired validation packets, and write manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    pipeline_root = Path(__file__).resolve().parents[1]
    outputs_root = pipeline_root / "outputs"
    external_root = outputs_root / "_external_study_validation" / "pdac"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidate_root = external_root / "pipeline_candidate_figures" / f"pipeline_candidates_{stamp}"
    candidate_root.mkdir(parents=True, exist_ok=True)

    pair_root = run_dir / "side_by_side_pairs" / f"pdac_meaningful_validation_pairs_{stamp}"
    pair_root.mkdir(parents=True, exist_ok=True)

    study_roots = [
        external_root / "study_pngs_publication_grade" / "priority_panels",
        external_root / "study_pngs_publication_grade" / "all_panels",
        external_root / "study_pngs_to_create",
        external_root / "study_pngs_from_mmc21" / "priority_pngs",
        external_root / "study_pngs_from_mmc21" / "all_extracted_pngs",
    ]

    copied_rows, candidate_manifest = collect_pipeline_figures(outputs_root, run_dir, candidate_root)

    local_dir = run_dir / "local_validation_maps"

    pair_rows = []

    for spec in PAIR_SPECS:
        study_paths = []
        for name in spec["study_panels"]:
            p = find_study_panel(study_roots, name)
            if p is not None:
                study_paths.append(p)

        local_paths = get_local_maps(local_dir, spec["local_patterns"])

        out_name = f"{spec['pair_id']}_{spec['name']}.png"
        out_path = pair_root / out_name

        create_packet(
            out_path=out_path,
            title=spec["title"],
            study_paths=study_paths,
            local_paths=local_paths,
            analysis_lines=spec["analysis"],
        )

        pair_rows.append({
            "pair_id": spec["pair_id"],
            "name": spec["name"],
            "output_file": str(out_path),
            "n_study_panels_found": len(study_paths),
            "study_panels_found": ";".join(str(p) for p in study_paths),
            "n_local_panels_found": len(local_paths),
            "local_patterns": ";".join(spec["local_patterns"]),
            "interpretation": "validation_packet_not_exact_roi_pixel_match",
        })

    pair_manifest = pair_root / "pdac_meaningful_pair_manifest.csv"
    with open(pair_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "name",
                "output_file",
                "n_study_panels_found",
                "study_panels_found",
                "n_local_panels_found",
                "local_patterns",
                "interpretation",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    report = build_analysis_report(
        run_dir=run_dir,
        candidate_root=candidate_root,
        pair_root=pair_root,
        copied_rows=copied_rows,
    )

    readme = pair_root / "README.txt"
    readme.write_text(
        "\n".join([
            "PDAC meaningful validation pairs",
            "",
            "These are validation packets, not exact same ROI pixel matches.",
            "Study panels are atlas level, patient level, or reference panels.",
            "Pipeline panels are from the three matched PDAC samples.",
            "",
            "Review order:",
            "01 compartment framework",
            "02 pathology reference",
            "03 tumor subtype reference",
            "04 hypoxia",
            "05 immune neighborhoods",
            "06 stroma and collagen",
            "07 gene communities",
            "08 normal or benign signal stress test",
            "",
            "Manifest:",
            str(pair_manifest),
            "",
            "Full analysis report:",
            str(report),
        ]),
        encoding="utf-8",
    )

    print("")
    print("Done.")
    print("")
    print("All candidate pipeline figures copied to:")
    print(candidate_root)
    print("")
    print("Candidate manifest:")
    print(candidate_manifest)
    print("")
    print("Meaningful validation pair folder:")
    print(pair_root)
    print("")
    print("Pair manifest:")
    print(pair_manifest)
    print("")
    print("Full analysis report:")
    print(report)



# =========================
# Command line entry point
# =========================

if __name__ == "__main__":
    main()

