#!/usr/bin/env python
"""
Script:
    07_make_final_outputs.py

Description:
    Creates final publication-oriented outputs: final TSV tables, Excel workbook,
    final figures, figure captions, methods/results/discussion narrative, final
    report, and output manifests.

Instructions:
    Run after Step 06 passes. Inspect final table and figure manifests before
    Step 08 packaging.

Source-truth policy:
    Final outputs are reporting artifacts generated from Steps 01-06. They do not
    modify upstream data, retrain models, or convert associations into clinical advice.
"""

# =============================================================================
# PIM_DOCS_PATCH: RUN AND MAINTENANCE INSTRUCTIONS
# =============================================================================
# Run numbered scripts through 00_run_prediction_interpretation_model.py unless
# debugging a single step. Treat the V2 full-run root as read-only source truth.
# Every generated .txt report must start with FILEPATH, and terminal summaries
# should remain concise enough for copy/paste debugging.
# =============================================================================


# =============================================================================
# PIM_DOCS_SECTION: imports and dependencies
# =============================================================================
# Keep imports explicit and standard-library-first where practical. The pipeline
# expects local scripts to run from the scripts directory or through the orchestrator.

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import shutil
import traceback
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _pim_utils import (
    add_qc,
    ensure_dir,
    load_prepared_index,
    open_folder,
    read_table,
    save_output_manifest,
    write_json,
    write_text_report,
    write_tsv,
)


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    Defaults preserve local project paths while allowing explicit overrides."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\Adv_Omics_Fenyo\project")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def required_file(path: Path) -> Path:
    """Assert that an upstream file exists and return its path.
    Provides clear failure messages when a pipeline contract is incomplete."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not path.exists():
        raise FileNotFoundError(f"Required upstream file missing: {path}")
    return path


def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy an upstream file when it exists.
    Returns a boolean so final-output manifests can track copied artifacts."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def figure_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path, n: int = 20) -> bool:
    """Create a horizontal bar figure from a table.
    Returns False when required columns or numeric data are unavailable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return False

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col]).sort_values(y_col, ascending=True).tail(n)
    if plot_df.empty:
        return False

    ensure_dir(path.parent)
    height = max(4.5, min(12.0, 0.32 * len(plot_df) + 2.0))
    plt.figure(figsize=(10, height))
    plt.barh(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def figure_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, path: Path) -> bool:
    """Create a histogram figure from a numeric column.
    Uses matplotlib's noninteractive backend for reproducible batch runs."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if df.empty or col not in df.columns:
        return False
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return False

    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=min(30, max(8, int(np.sqrt(len(vals))))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def figure_heatmap(matrix: pd.DataFrame, title: str, path: Path, max_rows: int = 35) -> bool:
    """Create a heatmap figure from a matrix table.
    Limits rows when needed so final figures remain readable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if matrix.empty or matrix.shape[1] < 2:
        return False

    df = matrix.copy()
    label_col = df.columns[0]
    value_cols = [c for c in df.columns if c != label_col]
    data = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if len(df) > max_rows:
        row_strength = data.abs().sum(axis=1)
        df = df.loc[row_strength.sort_values(ascending=False).head(max_rows).index].copy()
        data = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if data.empty:
        return False

    ensure_dir(path.parent)
    plt.figure(figsize=(max(8, 0.6 * len(value_cols) + 4), max(5, 0.24 * len(df) + 3)))
    plt.imshow(data.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label="signed effect")
    plt.xticks(range(len(value_cols)), value_cols, rotation=90)
    plt.yticks(range(len(df)), df[label_col].astype(str))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def make_excel(path: Path, sheets: List[tuple[str, pd.DataFrame]]) -> None:
    """Write publication tables to an Excel workbook.
    Sanitizes sheet names and prevents duplicate sheet-name collisions."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        used = set()
        for sheet_name, df in sheets:
            safe_name = "".join(ch for ch in sheet_name if ch not in r'[]:*?/\\')[:31] or "Sheet"
            base = safe_name
            i = 2
            while safe_name in used:
                suffix = f"_{i}"
                safe_name = (base[:31 - len(suffix)] + suffix)[:31]
                i += 1
            used.add(safe_name)
            df.to_excel(writer, sheet_name=safe_name, index=False)


def build_final_table_manifest(publication_dir: Path) -> pd.DataFrame:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows = []
    for path in sorted(publication_dir.glob("*.tsv")):
        try:
            df = read_table(path)
            rows.append({
                "table_file": path.name,
                "path": str(path),
                "rows": len(df),
                "columns": len(df.columns),
            })
        except Exception as exc:
            rows.append({
                "table_file": path.name,
                "path": str(path),
                "rows": "",
                "columns": "",
                "read_error": str(exc),
            })
    return pd.DataFrame(rows)


# =============================================================================
# PIM_DOCS_SECTION: main entry point
# =============================================================================
# The main function wires inputs, output folders, QC checks, reports, and terminal summaries.

def main() -> int:
    """Run the script's command-line workflow.
    Writes outputs, QC checks, summaries, and terminal status messages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    args = parse_args()
    started = dt.datetime.now()
    output_root = Path(args.output_root)
    prepared_root, _ = load_prepared_index(output_root, Path(args.prepared_input_root) if args.prepared_input_root else None)

    step02 = output_root / "02_feature_and_treatment_dictionary"
    step03 = output_root / "03_signed_spatial_effects"
    step04 = output_root / "04_treatment_interpretation_cards"
    step05 = output_root / "05_sample_level_interpretations"
    step06 = output_root / "06_mechanism_atlas"

    files = {
        "feature_dictionary": required_file(step02 / "01_feature_dictionary" / "strict_spatial_feature_dictionary.tsv"),
        "theme_dictionary": required_file(step02 / "01_feature_dictionary" / "biology_theme_dictionary.tsv"),
        "treatment_dictionary": required_file(step02 / "02_treatment_dictionary" / "treatment_dictionary.tsv"),
        "signed_feature_effects": required_file(step03 / "01_treatment_feature_effects" / "signed_treatment_feature_effects.tsv"),
        "signed_theme_effects": required_file(step03 / "02_treatment_theme_effects" / "signed_treatment_theme_effects.tsv"),
        "treatment_cards": required_file(step04 / "02_cards_tsv" / "treatment_interpretation_cards.tsv"),
        "sample_scores": required_file(step05 / "01_sample_treatment_scores" / "sample_treatment_signed_interpretation_scores.tsv"),
        "sample_summary": required_file(step05 / "03_sample_summaries" / "sample_interpretation_summary.tsv"),
        "theme_atlas": required_file(step06 / "01_theme_atlas" / "cross_treatment_biology_theme_atlas.tsv"),
        "feature_atlas": required_file(step06 / "02_feature_atlas" / "cross_treatment_feature_effect_atlas.tsv"),
        "component_class_atlas": required_file(step06 / "04_component_atlas" / "component_class_mechanism_atlas.tsv"),
        "sample_mechanism_summary": required_file(step06 / "05_sample_mechanism_patterns" / "sample_mechanism_summary.tsv"),
        "treatment_similarity_edges": required_file(step06 / "03_treatment_similarity" / "treatment_theme_similarity_edges.tsv"),
        "treatment_theme_matrix": required_file(step06 / "01_theme_atlas" / "treatment_theme_signed_effect_matrix.tsv"),
    }

    step_root = output_root / "07_final_outputs"
    publication_dir = step_root / "01_publication_tables_tsv"
    workbook_dir = step_root / "02_publication_workbook"
    figure_dir = step_root / "03_final_figures"
    report_dir = step_root / "04_final_reports"
    manifest_dir = step_root / "05_manifests"
    supporting_dir = step_root / "06_supporting_files"

    for path in [publication_dir, workbook_dir, figure_dir, report_dir, manifest_dir, supporting_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []
    figure_rows: List[dict] = []

    try:
        feature_dictionary = read_table(files["feature_dictionary"])
        theme_dictionary = read_table(files["theme_dictionary"])
        treatment_dictionary = read_table(files["treatment_dictionary"])
        signed_feature_effects = read_table(files["signed_feature_effects"])
        signed_theme_effects = read_table(files["signed_theme_effects"])
        treatment_cards = read_table(files["treatment_cards"])
        sample_scores = read_table(files["sample_scores"])
        sample_summary = read_table(files["sample_summary"])
        theme_atlas = read_table(files["theme_atlas"])
        feature_atlas = read_table(files["feature_atlas"])
        component_class_atlas = read_table(files["component_class_atlas"])
        sample_mechanism_summary = read_table(files["sample_mechanism_summary"])
        treatment_similarity_edges = read_table(files["treatment_similarity_edges"])
        treatment_theme_matrix = read_table(files["treatment_theme_matrix"])

        final_tables = [
            ("Final_Feature_Dictionary.tsv", feature_dictionary),
            ("Final_Biology_Theme_Dictionary.tsv", theme_dictionary),
            ("Final_Treatment_Dictionary.tsv", treatment_dictionary),
            ("Final_Treatment_Interpretation_Cards.tsv", treatment_cards),
            ("Final_Signed_Treatment_Feature_Effects.tsv", signed_feature_effects),
            ("Final_Signed_Treatment_Theme_Effects.tsv", signed_theme_effects),
            ("Final_Cross_Treatment_Biology_Theme_Atlas.tsv", theme_atlas),
            ("Final_Cross_Treatment_Feature_Atlas.tsv", feature_atlas),
            ("Final_Component_Class_Mechanism_Atlas.tsv", component_class_atlas),
            ("Final_Sample_Treatment_Interpretation_Scores.tsv", sample_scores),
            ("Final_Sample_Interpretation_Summary.tsv", sample_summary),
            ("Final_Sample_Mechanism_Summary.tsv", sample_mechanism_summary),
            ("Final_Treatment_Theme_Similarity_Edges.tsv", treatment_similarity_edges),
            ("Final_Treatment_Theme_Signed_Effect_Matrix.tsv", treatment_theme_matrix),
        ]

        for filename, df in final_tables:
            write_tsv(publication_dir / filename, df)

        workbook_path = workbook_dir / "prediction_interpretation_model_final_publication_tables.xlsx"
        make_excel(
            workbook_path,
            [
                ("Treatment_Cards", treatment_cards),
                ("Theme_Atlas", theme_atlas),
                ("Feature_Atlas", feature_atlas.head(500)),
                ("Theme_Effects", signed_theme_effects),
                ("Feature_Effects", signed_feature_effects.head(1000)),
                ("Sample_Summary", sample_summary),
                ("Sample_Mechanisms", sample_mechanism_summary),
                ("Component_Atlas", component_class_atlas),
                ("Similarity_Edges", treatment_similarity_edges.head(500)),
            ],
        )

        source_card_dir = step04 / "01_cards_txt"
        final_card_dir = supporting_dir / "treatment_cards_txt"
        ensure_dir(final_card_dir)
        card_txt_count = 0
        if source_card_dir.exists():
            for src in sorted(source_card_dir.glob("*.txt")):
                if copy_if_exists(src, final_card_dir / src.name):
                    card_txt_count += 1

        figure_specs = [
            (
                "fig_01_top_biology_themes_by_absolute_effect.png",
                lambda p: figure_bar(theme_atlas, "biological_theme", "absolute_theme_effect_sum", "Top cross-treatment biology themes", "Absolute signed effect sum", "Biological theme", p, 20),
                "Top cross-treatment biology themes by summed absolute signed effect.",
            ),
            (
                "fig_02_net_biology_theme_direction.png",
                lambda p: figure_bar(theme_atlas, "biological_theme", "signed_theme_effect_sum", "Net signed biology theme direction", "Signed effect sum", "Biological theme", p, 20),
                "Net signed direction of recurrent biology themes across validated treatments.",
            ),
            (
                "fig_03_top_spatial_features_by_absolute_effect.png",
                lambda p: figure_bar(feature_atlas, "feature_name", "absolute_feature_effect_sum", "Top recurrent spatial features", "Absolute signed effect sum", "Spatial feature", p, 25),
                "Top recurrent spatial features by summed absolute signed effect.",
            ),
            (
                "fig_04_component_class_mechanism_atlas.png",
                lambda p: figure_bar(component_class_atlas, "component_class", "absolute_theme_effect_sum", "Treatment component classes by mechanism effect", "Absolute signed effect sum", "Component class", p, 20),
                "Treatment component classes ranked by summed absolute theme effect.",
            ),
            (
                "fig_05_sample_net_spatial_interpretation_score_distribution.png",
                lambda p: figure_hist(sample_scores, "net_signed_spatial_interpretation_score", "Distribution of sample-treatment spatial interpretation scores", "Net signed spatial interpretation score", p),
                "Distribution of sample-treatment signed spatial interpretation scores.",
            ),
            (
                "fig_06_treatment_theme_signed_effect_heatmap.png",
                lambda p: figure_heatmap(treatment_theme_matrix, "Treatment-theme signed effect matrix", p, max_rows=35),
                "Heatmap of signed biology-theme effects by validated treatment.",
            ),
            (
                "fig_07_treatment_theme_similarity_distribution.png",
                lambda p: figure_hist(treatment_similarity_edges, "cosine_similarity_signed_theme_profile", "Treatment similarity by signed theme profile", "Cosine similarity", p),
                "Distribution of treatment pair similarities based on signed theme profiles.",
            ),
        ]

        for filename, maker, caption in figure_specs:
            path = figure_dir / filename
            made = maker(path)
            if made:
                figure_rows.append({
                    "figure_id": filename.replace(".png", ""),
                    "figure_file": filename,
                    "figure_path": str(path),
                    "caption": caption,
                })

        figure_manifest = pd.DataFrame(figure_rows)
        write_tsv(manifest_dir / "final_figure_manifest.tsv", figure_manifest)

        caption_lines = ["PREDICTION INTERPRETATION MODEL FINAL FIGURE CAPTIONS", ""]
        for i, row in enumerate(figure_rows, start=1):
            caption_lines.append(f"Figure {i}. {row['caption']} File: {row['figure_file']}")
        write_text_report(report_dir / "final_figure_captions.txt", "\n".join(caption_lines))

        final_table_manifest = build_final_table_manifest(publication_dir)
        write_tsv(manifest_dir / "final_publication_table_manifest.tsv", final_table_manifest)

        sample_count = int(sample_scores["sample_id"].nunique()) if "sample_id" in sample_scores.columns else 0
        treatment_count = int(treatment_cards["drug_key"].nunique()) if "drug_key" in treatment_cards.columns else 0
        theme_count = int(theme_atlas["biological_theme"].nunique()) if "biological_theme" in theme_atlas.columns else 0
        feature_count = int(feature_dictionary["feature_name"].nunique()) if "feature_name" in feature_dictionary.columns else 0

        methods_lines = [
            "PREDICTION INTERPRETATION MODEL FINAL METHODS RESULTS DISCUSSION",
            "",
            "Methods summary",
            "The final interpretation layer consumed the completed spatial_prediction_model_V2 full-run outputs through Step 01 prepared inputs. It then built dictionaries, computed signed residual spatial effects, generated treatment interpretation cards, summarized sample-level spatial alignments, and assembled a cross-treatment mechanism atlas.",
            "",
            "Directional interpretation",
            "Signed treatment effects were estimated by combining V2 feature-importance evidence with empirical directionality against fused_residual_vs_prior. Positive signed effects indicate higher spatial feature values associated with above-prior residual response signal; negative signed effects indicate higher spatial feature values associated with below-prior residual response signal.",
            "",
            "Results summary",
            f"The final output includes {feature_count} strict spatial biology features, {theme_count} recurrent biology themes, {treatment_count} label-shuffle-validated treatments, and {sample_count} samples with validated-treatment interpretation scores.",
            f"The signed effect layer contains {len(signed_feature_effects)} treatment-feature effects and {len(signed_theme_effects)} treatment-theme effects.",
            f"The mechanism atlas contains {len(theme_atlas)} biology-theme rows and {len(feature_atlas)} feature-atlas rows.",
            "",
            "Sample coverage note",
            "The sample-level interpretation layer scores only sample-treatment rows belonging to label-shuffle-validated treatments. Therefore, sample coverage may be lower than the 102-sample V2 source population when some samples do not have eligible rows for validated treatment keys.",
            "",
            "Limitations",
            "These outputs summarize model-derived residual spatial associations. They do not prove causal mechanisms and are not clinical treatment recommendations.",
        ]
        write_text_report(report_dir / "final_methods_results_discussion.txt", "\n".join(methods_lines))

        final_report_lines = [
            "PREDICTION INTERPRETATION MODEL FINAL REPORT",
            "",
            f"status: pending_final_qc_step08",
            f"output_root: {output_root}",
            f"step_root: {step_root}",
            f"prepared_input_root: {prepared_root}",
            f"started: {started.isoformat(timespec='seconds')}",
            f"finished: {dt.datetime.now().isoformat(timespec='seconds')}",
            "",
            "Final output counts",
            f"strict_spatial_biology_features: {feature_count}",
            f"recurrent_biology_themes: {theme_count}",
            f"validated_treatments_with_cards: {treatment_count}",
            f"samples_with_validated_treatment_scores: {sample_count}",
            f"signed_treatment_feature_effect_rows: {len(signed_feature_effects)}",
            f"signed_treatment_theme_effect_rows: {len(signed_theme_effects)}",
            f"treatment_card_rows: {len(treatment_cards)}",
            f"theme_atlas_rows: {len(theme_atlas)}",
            f"feature_atlas_rows: {len(feature_atlas)}",
            f"final_figures: {len(figure_rows)}",
            "",
            "Main outputs",
            f"publication_tsv_folder: {publication_dir}",
            f"publication_workbook: {workbook_path}",
            f"final_figures_folder: {figure_dir}",
            f"final_figure_manifest: {manifest_dir / 'final_figure_manifest.tsv'}",
            f"final_methods_results_discussion: {report_dir / 'final_methods_results_discussion.txt'}",
            "",
            "Interpretation caveat",
            "This final report summarizes biological interpretation outputs from model-derived spatial residual associations. It is not causal proof and not a clinical recommendation.",
        ]
        write_text_report(report_dir / "prediction_interpretation_model_final_report.txt", "\n".join(final_report_lines))

        add_qc(qc, "final_treatment_cards_count", "pass" if treatment_count == 27 else "warn", treatment_count, 27, "Final treatment cards represented.")
        add_qc(qc, "final_strict_feature_count", "pass" if feature_count == 139 else "warn", feature_count, 139, "Strict spatial biology feature dictionary represented.")
        add_qc(qc, "final_theme_count", "pass" if theme_count == 11 else "warn", theme_count, 11, "Recurrent biology theme atlas represented.")
        add_qc(qc, "final_sample_score_rows", "pass" if len(sample_scores) > 0 else "fail", len(sample_scores), ">0", "Sample-treatment interpretation scores represented.")
        add_qc(qc, "final_sample_coverage", "pass" if sample_count >= 90 else "warn", sample_count, ">=90", "Sample coverage reflects validated-treatment eligible sample-treatment rows.")
        add_qc(qc, "final_publication_tables", "pass" if len(final_table_manifest) >= 10 else "fail", len(final_table_manifest), ">=10", "Final TSV publication tables created.")
        add_qc(qc, "final_workbook_exists", "pass" if workbook_path.exists() else "fail", workbook_path.exists(), True, "Final Excel workbook created.")
        add_qc(qc, "final_figure_count", "pass" if len(figure_rows) >= 5 else "warn", len(figure_rows), ">=5", "Final figure set created.")
        add_qc(qc, "treatment_card_txt_copied", "pass" if card_txt_count == 27 else "warn", card_txt_count, 27, "Treatment card text files copied to final supporting files.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        final_table_manifest = pd.DataFrame()
        figure_rows = []

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(manifest_dir / "step07_final_outputs_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "final_publication_tables": int(len(final_table_manifest)) if "final_table_manifest" in locals() else 0,
        "final_figures": len(figure_rows),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step07_summary.json", summary)
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 07 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"final_publication_tables: {summary['final_publication_tables']}")
    print(f"final_figures: {len(figure_rows)}")
    print(f"workbook: {workbook_dir / 'prediction_interpretation_model_final_publication_tables.xlsx'}")
    print(f"report: {report_dir / 'prediction_interpretation_model_final_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())
