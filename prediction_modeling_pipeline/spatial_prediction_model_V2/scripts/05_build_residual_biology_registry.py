"""
Script: 05_build_residual_biology_registry.py

Purpose:
    Build the Step 05 residual-biology feature registry for
    spatial_prediction_model_V2.

Pipeline role:
    This step converts Step 04 residual-model feature evidence into a curated V2
    strict biology registry. The registry becomes the interpretable feature set
    used by later broad and per-treatment residual models.

Scientific role:
    The registry separates interpretable spatial-biology features from treatment
    identity, coordinate artifacts, QC fields, and other technical signals. This
    keeps downstream biological interpretation focused on spatial phenotypes
    rather than nuisance variables.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP05_DOC_POLISH_V2

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    imports, constants, thresholds, hyperparameters, feature-selection
    rules, output filenames, and return codes must remain unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
# Use a non-interactive backend so registry figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.io_utils import ensure_dir, read_table, write_json, write_table, write_text_report
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest


# =============================================================================
# Helper functions
# =============================================================================

def sha256(path: Path) -> str:
    """Return the SHA-256 hash for an existing file, or blank for missing paths."""

    if path is None or not Path(path).exists() or not Path(path).is_file():
        return ""

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest(paths: dict[str, Path | None]) -> pd.DataFrame:
    """Build a source-file manifest with existence, size, and hash metadata."""

    rows = []

    for key, path in paths.items():
        exists = bool(path is not None and Path(path).exists())
        is_file = bool(exists and Path(path).is_file())

        rows.append({
            "source_key": key,
            "path": str(path) if path is not None else "",
            "exists": exists,
            "is_file": is_file,
            "size_bytes": int(Path(path).stat().st_size) if is_file else "",
            "sha256": sha256(path) if is_file else "",
        })

    return pd.DataFrame(rows)


def first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path from a candidate list."""

    for path in paths:
        if path.exists():
            return path
    return None


def find_score_col(df: pd.DataFrame) -> str:
    """Identify the numeric evidence score column used for feature ranking."""

    preferred = [
        "mean_abs_shap",
        "total_mean_abs_shap",
        "mean_gain_importance",
        "gain_importance",
        "selection_frequency",
        "importance",
        "feature_importance",
    ]

    for col in preferred:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() > 0:
                return col

    numeric_cols = []

    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("Could not identify a numeric feature score column.")

    return numeric_cols[0]


def row_text(row: pd.Series) -> str:
    """Concatenate row metadata into lowercase text for rule-based feature classification."""

    parts = []

    for col in [
        "feature_name",
        "feature_original",
        "feature_group",
        "feature_axis",
        "model_feature_id",
        "feature_source",
        "governance_class",
        "governance_reason",
    ]:
        if col in row.index:
            parts.append(str(row.get(col, "")))

    return " ".join(parts).lower()


def classify_feature(row: pd.Series) -> tuple[str, str]:
    """Classify one residual feature as biology, caution, or exclusion."""

    text = row_text(row)

    drug_terms = [
        "treatment_identity",
        "drug_identity",
        "drug_dummy",
        "drug_key",
    ]

    for term in drug_terms:
        if term in text:
            return "exclude_drug_identity", f"drug or treatment identity feature: {term}"

    coordinate_terms = [
        "array_row",
        "array_col",
        "pxl_row",
        "pxl_col",
        "spatial_x",
        "spatial_y",
        "image_x",
        "image_y",
        "coord",
        "coordinate",
        "barcode",
    ]

    for term in coordinate_terms:
        if term in text:
            return "exclude_coordinate_or_artifact", f"coordinate or technical artifact term: {term}"

    qc_terms = [
        "pct_counts",
        "total_counts",
        "n_genes",
        "n_counts",
        "mitochond",
        "percent_mt",
        "qc_",
        "quality",
    ]

    for term in qc_terms:
        if term in text:
            return "exclude_qc", f"QC feature term: {term}"

    size_or_count_terms = [
        "largest_component_spots",
        "spot_count",
        "spots_count",
        "n_spots",
        "num_spots",
        "total_spots",
        "component_size",
        "component_area",
        "slide_area",
        "fraction_of_slide",
    ]

    for term in size_or_count_terms:
        if term in text:
            return "caution", f"size or count associated feature: {term}"

    caution_terms = [
        "structure_region_consensus_fraction",
        "metabolic_best_matching_state_score",
        "method_structure_agreement",
        "raw_fraction",
    ]

    for term in caution_terms:
        if term in text:
            return "caution", f"interpret with caution because of {term}"

    return "include_biology", "biological spatial feature"


def infer_theme(row: pd.Series) -> str:
    """Assign an interpretable biological theme from feature metadata text."""

    text = row_text(row)

    if "tryptophan" in text or "kynurenine" in text:
        return "tryptophan kynurenine immune suppression"

    if "myeloid" in text or "macrophage" in text:
        return "myeloid macrophage tumor ecology"

    if "hypoxi" in text:
        return "hypoxia immune stress context"

    if "checkpoint" in text or "exhaustion" in text:
        return "immune inflammation and t cell organization"

    if "t_cell" in text or "interferon" in text or "immune" in text or "b_plasma" in text:
        return "immune inflammation and t cell organization"

    if "access" in text or "boundary" in text or "penetration" in text or "barrier" in text or "graph_depth" in text:
        return "tumor access and boundary penetration"

    if "stromal" in text or "stroma" in text or "ecm" in text or "fibroblast" in text or "collagen" in text:
        return "stromal ecm barrier architecture"

    if "tumor_proliferative" in text or "proliferation" in text or "cell_cycle" in text:
        return "tumor proliferation state"

    if "vascular" in text or "angiogenic" in text or "endothelial" in text:
        return "vascular angiogenic context"

    if "fatty_acid" in text:
        return "fatty acid metabolism"

    if "glutamine" in text or "glycolysis" in text or "oxphos" in text or "oxidative" in text or "metabolic" in text:
        return "metabolic spatial context"

    if "pair_" in text or "centroid_distance" in text or "overlap" in text:
        return "pairwise spatial relationship"

    return "other interpretable spatial signal"


def merge_feature_metadata(evidence: pd.DataFrame, feature_manifest: pd.DataFrame) -> pd.DataFrame:
    """Merge feature evidence with the feature manifest while preserving model identifiers."""

    out = evidence.copy()

    if "feature_name" not in out.columns and "feature" in out.columns:
        out = out.rename(columns={"feature": "feature_name"})

    if "feature_name" not in out.columns:
        raise ValueError("Evidence table must contain feature_name or feature.")

    out["feature_name"] = out["feature_name"].astype(str)

    if "model_feature_id" not in out.columns:
        out["model_feature_id"] = out["feature_name"]

    for col in ["feature_original", "feature_group", "feature_axis"]:
        if col not in out.columns:
            out[col] = ""

    if feature_manifest is not None and not feature_manifest.empty:
        meta = feature_manifest.copy()

        if "feature_name" not in meta.columns:
            for candidate in ["feature", "feature_id", "model_feature", "feature_original", "original_feature"]:
                if candidate in meta.columns:
                    meta = meta.rename(columns={candidate: "feature_name"})
                    break

        if "feature_name" in meta.columns:
            keep_cols = ["feature_name"]

            for col in ["feature_original", "feature_group", "feature_axis", "governance_class", "governance_reason"]:
                if col in meta.columns:
                    keep_cols.append(col)

            meta = meta[keep_cols].drop_duplicates("feature_name")
            out = out.merge(meta, on="feature_name", how="left", suffixes=("", "_manifest"))

            for col in ["feature_original", "feature_group", "feature_axis", "governance_class", "governance_reason"]:
                manifest_col = f"{col}_manifest"

                if manifest_col in out.columns:
                    if col not in out.columns:
                        out[col] = out[manifest_col]
                    else:
                        current = out[col].fillna("").astype(str)
                        replacement = out[manifest_col].fillna("").astype(str)
                        out[col] = current.where(current.str.len() > 0, replacement)

    out["feature_original"] = out["feature_original"].fillna("").astype(str)
    out["feature_original"] = out["feature_original"].where(out["feature_original"].str.len() > 0, out["feature_name"])

    return out


def build_theme_summary(strict_biology: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Summarize strict-biology registry features by biological theme."""

    if strict_biology.empty:
        return pd.DataFrame(columns=[
            "biological_theme",
            "n_features",
            "total_mean_abs_score",
            "max_abs_score",
            "example_features",
        ])

    theme_summary = (
        strict_biology
        .groupby("biological_theme", dropna=False)
        .agg(
            n_features=("model_feature_id", "count"),
            total_mean_abs_score=(score_col, "sum"),
            max_abs_score=(score_col, "max"),
        )
        .reset_index()
        .sort_values("total_mean_abs_score", ascending=False)
    )

    examples = []

    for theme in theme_summary["biological_theme"].tolist():
        sub = strict_biology[strict_biology["biological_theme"] == theme].sort_values(score_col, ascending=False)
        names = sub.head(3)["feature_original"].astype(str).tolist()
        examples.append("; ".join(names))

    theme_summary["example_features"] = examples
    return theme_summary


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, xlabel: str, top_n: int = 30) -> None:
    """Save a ranked horizontal bar plot for model, feature, or theme evidence."""

    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    plot = df.copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)

    if plot.empty:
        return

    labels = plot[label_col].astype(str).tolist()
    labels = [x if len(x) <= 72 else x[:69] + "..." for x in labels]

    y = np.arange(len(plot))[::-1]
    values = plot[value_col].to_numpy()[::-1]
    labels = labels[::-1]

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(6, len(plot) * 0.32)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Run this spatial_prediction_model_V2 step and write tables, figures, reports, and provenance."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--step04-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-input-features", type=int, default=150)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    step04_root = Path(args.step04_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs_and_source_manifest")
    d02 = ensure_dir(output_root / "02_classified_residual_features")
    d03 = ensure_dir(output_root / "03_v2_strict_biology_registry")
    d04 = ensure_dir(output_root / "04_theme_summary")
    d05 = ensure_dir(output_root / "05_figures")
    d06 = ensure_dir(output_root / "06_reports")

    evidence_path = step04_root / "03_feature_evidence_for_step05" / "spatial_feature_evidence_for_step05.tsv"
    all_feature_evidence_path = step04_root / "03_feature_evidence_for_step05" / "pair_level_residual_feature_evidence_summary.tsv"
    contribution_path = step04_root / "03_feature_evidence_for_step05" / "pair_level_residual_spatial_vs_treatment_contribution.tsv"
    metric_summary_path = step04_root / "02_metrics" / "pair_level_residual_metric_summary.tsv"
    step04_summary_path = step04_root / "v2_step04_pair_level_residual_model_summary.json"
    feature_manifest_path = dataset_root / "02_feature_governance" / "v2_broad_governed_candidate_features.tsv"
    feature_policy_path = dataset_root / "02_feature_governance" / "v2_feature_set_policy.json"

    sources = {
        "run_root": run_root,
        "dataset_root": dataset_root,
        "step04_root": step04_root,
        "step04_spatial_feature_evidence": evidence_path,
        "step04_all_feature_evidence": all_feature_evidence_path,
        "step04_spatial_vs_treatment_contribution": contribution_path,
        "step04_metric_summary": metric_summary_path,
        "step04_summary": step04_summary_path,
        "feature_manifest": feature_manifest_path,
        "feature_policy": feature_policy_path,
    }

    source_manifest_df = source_manifest(sources)
    write_table(source_manifest_df, d01 / "source_manifest.tsv")

    evidence = read_table(evidence_path)
    feature_manifest = read_table(feature_manifest_path)
    contribution = read_table(contribution_path)
    metric_summary = read_table(metric_summary_path)

    if evidence is None or evidence.empty:
        raise FileNotFoundError(f"Step 04 spatial feature evidence is missing or empty: {evidence_path}")

    # Merge model evidence with feature metadata before applying biological interpretation rules.
    evidence = merge_feature_metadata(evidence, feature_manifest)

    score_col = find_score_col(evidence)
    evidence[score_col] = pd.to_numeric(evidence[score_col], errors="coerce").fillna(0.0)
    evidence = evidence.sort_values(score_col, ascending=False).copy()

    if args.max_input_features and args.max_input_features > 0:
        evidence = evidence.head(args.max_input_features).copy()

    # Classify features before registry construction so technical, coordinate, and QC artifacts are excluded.
    class_pairs = evidence.apply(classify_feature, axis=1)
    evidence["interpretation_class"] = [x[0] for x in class_pairs]
    evidence["interpretation_note"] = [x[1] for x in class_pairs]
    evidence["biological_theme"] = evidence.apply(infer_theme, axis=1)

    all_classified = evidence.copy()
    # Strict registry keeps only interpretable spatial-biology features for downstream residual models.
    strict_biology = evidence[evidence["interpretation_class"].eq("include_biology")].copy()
    caution_features = evidence[evidence["interpretation_class"].eq("caution")].copy()
    excluded_features = evidence[~evidence["interpretation_class"].isin(["include_biology", "caution"])].copy()

    strict_biology["include_for_v2_strict_biology_registry"] = True
    strict_biology["v2_registry_generation_source"] = "05_build_residual_biology_registry.py"
    strict_biology["production_dependency_on_v1_outputs"] = "no"

    theme_summary = build_theme_summary(strict_biology, score_col)

    write_table(all_classified, d02 / "v2_all_residual_spatial_features_classified.tsv")
    write_table(all_classified, d02 / "v2_all_residual_spatial_features_classified.csv")
    write_table(caution_features, d02 / "v2_top_residual_caution_features.tsv")
    write_table(caution_features, d02 / "v2_top_residual_caution_features.csv")
    write_table(excluded_features, d02 / "v2_residual_interpretation_excluded_features.tsv")
    write_table(excluded_features, d02 / "v2_residual_interpretation_excluded_features.csv")

    write_table(strict_biology, d03 / "v2_strict_biology_feature_registry.tsv")
    write_table(strict_biology, d03 / "v2_strict_biology_feature_registry.csv")
    write_table(strict_biology, d03 / "top_residual_biology_features_strict.csv")

    write_table(theme_summary, d04 / "v2_residual_biology_theme_summary.tsv")
    write_table(theme_summary, d04 / "v2_residual_biology_theme_summary.csv")

    save_bar(
        strict_biology,
        "feature_original",
        score_col,
        d05 / "fig_01_v2_top_residual_spatial_biology_features.png",
        "V2 Step 05 residual spatial biology features",
        score_col,
        top_n=30,
    )

    save_bar(
        theme_summary,
        "biological_theme",
        "total_mean_abs_score",
        d05 / "fig_02_v2_residual_biology_theme_contribution.png",
        "V2 Step 05 residual biology theme contribution",
        "Total feature score",
        top_n=15,
    )

    contribution_summary = {}
    if contribution is not None and not contribution.empty:
        for col in contribution.columns:
            val = contribution.iloc[0][col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                contribution_summary[f"contribution_{col}"] = float(val)
            else:
                contribution_summary[f"contribution_{col}"] = str(val)

    metric_summary_dict = {}
    if metric_summary is not None and not metric_summary.empty:
        for col in metric_summary.columns:
            if "mean" in col or col in ["model_family", "target_col"]:
                val = metric_summary.iloc[0][col]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    metric_summary_dict[f"step04_metric_{col}"] = float(val)
                else:
                    metric_summary_dict[f"step04_metric_{col}"] = str(val)

    # Track whether the registry size reflects a smoke run or a full-scale evidence run.
    registry_count_note = "smoke_registry_count_expected_to_be_lower_than_full_v1_count"
    if len(strict_biology) >= 100:
        registry_count_note = "registry_count_close_to_full_scale_strict_biology_count"

    run_summary = {
        "status": "pass",
        "official_step": "05_build_residual_biology_registry",
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "step04_root": str(step04_root),
        "output_root": str(output_root),
        "evidence_path": str(evidence_path),
        "feature_manifest": str(feature_manifest_path),
        "score_col": score_col,
        "max_input_features": int(args.max_input_features),
        "n_all_classified_features": int(len(all_classified)),
        "n_strict_biology_features": int(len(strict_biology)),
        "n_caution_features": int(len(caution_features)),
        "n_excluded_features": int(len(excluded_features)),
        "n_biology_themes": int(theme_summary["biological_theme"].nunique()) if "biological_theme" in theme_summary.columns else 0,
        "registry_count_note": registry_count_note,
        "production_dependency_on_v1_outputs": "no",
        "v1_outputs_role": "migration_reference_only",
        "v2_registry_generated_in_v2": "yes",
        **contribution_summary,
        **metric_summary_dict,
    }

    write_json(run_summary, output_root / "v2_step05_residual_biology_registry_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    show_cols = [
        "feature_name",
        "feature_original",
        "feature_group",
        "feature_axis",
        score_col,
        "selection_frequency",
        "mean_gain_importance",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]
    show_cols = [c for c in show_cols if c in strict_biology.columns]

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 05 RESIDUAL BIOLOGY REGISTRY REPORT")
    report_lines.append("")
    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    report_lines.append("1. Source files")
    report_lines.append(source_manifest_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("2. Design")
    report_lines.append("This step ports the V1 residual biology interpretation logic into V2.")
    report_lines.append("It reads Step 04 spatial residual feature evidence, maps available feature metadata, assigns interpretation classes, assigns biological themes, and writes the V2 strict biology registry.")
    report_lines.append("This step does not read V1 output tables.")
    report_lines.append("")
    report_lines.append("3. Strict residual biology features")
    if strict_biology.empty:
        report_lines.append("No strict biology features generated.")
    else:
        report_lines.append(strict_biology[show_cols].head(80).to_string(index=False))
    report_lines.append("")
    report_lines.append("4. Caution features")
    if caution_features.empty:
        report_lines.append("No caution features.")
    else:
        report_lines.append(caution_features[[c for c in show_cols if c in caution_features.columns]].head(50).to_string(index=False))
    report_lines.append("")
    report_lines.append("5. Excluded features")
    if excluded_features.empty:
        report_lines.append("No excluded features.")
    else:
        report_lines.append(excluded_features[[c for c in show_cols if c in excluded_features.columns]].head(50).to_string(index=False))
    report_lines.append("")
    report_lines.append("6. Biology theme summary")
    report_lines.append(theme_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("7. Interpretation")
    report_lines.append("The output top_residual_biology_features_strict.csv is a compatibility filename for downstream V1 logic ports.")
    report_lines.append("For a smoke run, the feature count may be lower than the prior V1 full scale registry because Step 04 used a subset of samples, treatments, and repeats.")
    report_lines.append("The full V2 run should regenerate this registry using full Step 04 evidence.")

    report_path = write_text_report(d06 / "v2_step05_residual_biology_registry_report.txt", "\n".join(report_lines))

    slide_lines = []
    slide_lines.append("V2 STEP 05 RESIDUAL BIOLOGY REGISTRY SLIDE NOTES")
    slide_lines.append("")
    slide_lines.append(f"Strict biology features: {len(strict_biology)}")
    slide_lines.append(f"Caution features: {len(caution_features)}")
    slide_lines.append(f"Excluded features: {len(excluded_features)}")
    slide_lines.append(f"Biology themes: {run_summary['n_biology_themes']}")
    slide_lines.append("")
    slide_lines.append("Top biology themes:")
    if theme_summary.empty:
        slide_lines.append("No themes generated.")
    else:
        for theme in theme_summary.head(8)["biological_theme"].astype(str).tolist():
            slide_lines.append(f"{theme}")
    slide_lines.append("")
    slide_lines.append("Caveat: SHAP and gain importance describe model behavior, not causal biology.")
    write_text_report(d06 / "v2_step05_residual_biology_slide_notes.txt", "\n".join(slide_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Dataset root: {dataset_root}",
        f"Step 04 root: {step04_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Score column: {score_col}",
        f"All classified features: {len(all_classified)}",
        f"Strict biology features: {len(strict_biology)}",
        f"Caution features: {len(caution_features)}",
        f"Excluded features: {len(excluded_features)}",
        f"Biology themes: {run_summary['n_biology_themes']}",
        "Production dependency on V1 outputs: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 05 RESIDUAL BIOLOGY REGISTRY COMPLETE", terminal_lines))
    print("")

    if strict_biology.empty:
        print("No strict biology features were generated.")
    else:
        print("Top strict biology features")
        print(strict_biology[show_cols].head(30).to_string(index=False))
        print("")

    if not theme_summary.empty:
        print("Biology theme summary")
        print(theme_summary.head(15).to_string(index=False))
        print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
