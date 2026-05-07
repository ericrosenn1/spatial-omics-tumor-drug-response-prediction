#!/usr/bin/env python
"""
Script:
    06_build_mechanism_atlas.py

Description:
    Builds the cross-treatment mechanism atlas from signed treatment-feature
    effects, signed treatment-theme effects, treatment/component dictionaries,
    treatment cards, and sample-level interpretation scores.

Instructions:
    Run after Steps 02-05 pass. Use the resulting theme atlas, feature atlas,
    component atlas, treatment similarity tables, and sample mechanism summaries
    as the mechanism layer for final publication outputs.

Source-truth policy:
    This step summarizes associations already computed by prior interpretation
    steps. It does not rerun V2, select models, or make clinical recommendations.
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
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd

from _pim_utils import (
    add_qc,
    ensure_dir,
    load_prepared_index,
    open_folder,
    read_table,
    save_output_manifest,
    summarize_examples,
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


def safe_float_series(values: pd.Series) -> pd.Series:
    """Convert a Series to finite float values with missing values as zero.
    Used for atlas aggregation where absent effects should not crash summaries."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)


def split_semicolon_values(value: object) -> List[str]:
    """Split semicolon-delimited text into clean values.
    Used for treatment component and class expansion."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    text = str(value)
    if text.lower() == "nan":
        return []
    parts = [x.strip() for x in text.split(";")]
    return [x for x in parts if x]


def consensus_label(pos_count: int, neg_count: int, signed_sum: float) -> str:
    """Convert positive/negative counts and signed sum into a consensus label.
    Summarizes whether a theme is net sensitivity-associated or resistance-associated."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if pos_count > neg_count and signed_sum > 0:
        return "consensus_sensitivity_associated"
    if neg_count > pos_count and signed_sum < 0:
        return "consensus_resistance_associated"
    if signed_sum > 0:
        return "mixed_but_net_sensitivity_associated"
    if signed_sum < 0:
        return "mixed_but_net_resistance_associated"
    return "balanced_or_ambiguous"


def build_theme_atlas(theme_effects: pd.DataFrame, treatment_dict: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build cross-treatment biology-theme atlas tables.
    Returns summary atlas and treatment-theme matrices."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    df = theme_effects.copy()
    df["signed_theme_effect"] = safe_float_series(df["signed_theme_effect"])
    df["absolute_theme_effect"] = safe_float_series(df["absolute_theme_effect"]) if "absolute_theme_effect" in df.columns else df["signed_theme_effect"].abs()
    df["direction_for_count"] = np.where(df["signed_theme_effect"] > 0, "sensitivity_associated", np.where(df["signed_theme_effect"] < 0, "resistance_associated", "ambiguous"))

    keep = [c for c in ["drug_key", "treatment_components", "component_classes", "interpretation_tier", "label_shuffle_validation_status"] if c in treatment_dict.columns]
    if keep:
        df = df.merge(treatment_dict[keep].drop_duplicates("drug_key"), on="drug_key", how="left")

    rows: List[dict] = []
    for theme, sub in df.groupby("biological_theme", dropna=False):
        pos = int((sub["signed_theme_effect"] > 0).sum())
        neg = int((sub["signed_theme_effect"] < 0).sum())
        zero = int((sub["signed_theme_effect"] == 0).sum())
        signed_sum = float(sub["signed_theme_effect"].sum())
        abs_sum = float(sub["absolute_theme_effect"].sum())
        rows.append({
            "biological_theme": theme,
            "validated_treatment_count": int(sub["drug_key"].nunique()),
            "sensitivity_associated_treatment_count": pos,
            "resistance_associated_treatment_count": neg,
            "ambiguous_treatment_count": zero,
            "signed_theme_effect_sum": signed_sum,
            "signed_theme_effect_mean": float(sub["signed_theme_effect"].mean()),
            "absolute_theme_effect_sum": abs_sum,
            "absolute_theme_effect_mean": float(sub["absolute_theme_effect"].mean()),
            "consensus_direction": consensus_label(pos, neg, signed_sum),
            "top_sensitivity_treatments": summarize_examples(sub.sort_values("signed_theme_effect", ascending=False)["drug_key"], 8),
            "top_resistance_treatments": summarize_examples(sub.sort_values("signed_theme_effect", ascending=True)["drug_key"], 8),
        })

    atlas = pd.DataFrame(rows).sort_values(["absolute_theme_effect_sum", "biological_theme"], ascending=[False, True])
    matrix = df.pivot_table(index="drug_key", columns="biological_theme", values="signed_theme_effect", aggfunc="sum", fill_value=0.0).reset_index()
    abs_matrix = df.pivot_table(index="drug_key", columns="biological_theme", values="absolute_theme_effect", aggfunc="sum", fill_value=0.0).reset_index()
    return atlas, matrix, abs_matrix


def build_feature_atlas(feature_effects: pd.DataFrame) -> pd.DataFrame:
    """Build cross-treatment feature-effect atlas table.
    Aggregates signed feature evidence across validated treatments."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    df = feature_effects.copy()
    df["signed_effect"] = safe_float_series(df["signed_effect"])
    df["effect_weight"] = safe_float_series(df["effect_weight"]) if "effect_weight" in df.columns else df["signed_effect"].abs()

    rows: List[dict] = []
    for feature, sub in df.groupby("feature_name", dropna=False):
        pos = int((sub["signed_effect"] > 0).sum())
        neg = int((sub["signed_effect"] < 0).sum())
        signed_sum = float(sub["signed_effect"].sum())
        abs_sum = float(sub["effect_weight"].sum())
        rows.append({
            "feature_name": feature,
            "feature_label": sub.get("feature_label", pd.Series([""])).dropna().astype(str).iloc[0] if "feature_label" in sub.columns and sub["feature_label"].notna().any() else "",
            "feature_group": sub.get("feature_group", pd.Series([""])).dropna().astype(str).iloc[0] if "feature_group" in sub.columns and sub["feature_group"].notna().any() else "",
            "biological_theme": sub.get("biological_theme", pd.Series([""])).dropna().astype(str).iloc[0] if "biological_theme" in sub.columns and sub["biological_theme"].notna().any() else "",
            "validated_treatment_count": int(sub["drug_key"].nunique()),
            "sensitivity_associated_treatment_count": pos,
            "resistance_associated_treatment_count": neg,
            "signed_feature_effect_sum": signed_sum,
            "signed_feature_effect_mean": float(sub["signed_effect"].mean()),
            "absolute_feature_effect_sum": abs_sum,
            "absolute_feature_effect_mean": float(sub["effect_weight"].mean()),
            "consensus_direction": consensus_label(pos, neg, signed_sum),
            "top_sensitivity_treatments": summarize_examples(sub.sort_values("signed_effect", ascending=False)["drug_key"], 6),
            "top_resistance_treatments": summarize_examples(sub.sort_values("signed_effect", ascending=True)["drug_key"], 6),
        })

    return pd.DataFrame(rows).sort_values(["absolute_feature_effect_sum", "feature_name"], ascending=[False, True])


def cosine_similarity_matrix(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if matrix.empty or "drug_key" not in matrix.columns:
        return pd.DataFrame(), pd.DataFrame()

    labels = matrix["drug_key"].astype(str).tolist()
    cols = [c for c in matrix.columns if c != "drug_key"]
    X = matrix[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    norms = np.linalg.norm(X, axis=1)
    sim = np.zeros((len(labels), len(labels)), dtype=float)
    for i in range(len(labels)):
        for j in range(len(labels)):
            denom = norms[i] * norms[j]
            sim[i, j] = float(np.dot(X[i], X[j]) / denom) if denom > 0 else 0.0

    sim_df = pd.DataFrame(sim, columns=labels)
    sim_df.insert(0, "drug_key", labels)

    edges: List[dict] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            edges.append({
                "drug_key_a": a,
                "drug_key_b": b,
                "cosine_similarity_signed_theme_profile": sim[i, j],
                "absolute_similarity": abs(sim[i, j]),
            })

    edge_df = pd.DataFrame(edges).sort_values(["absolute_similarity", "cosine_similarity_signed_theme_profile"], ascending=[False, False])
    return sim_df, edge_df


def build_component_atlas(theme_effects: pd.DataFrame, treatment_dict: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build component and component-class mechanism atlas tables.
    Links treatment components to signed biological themes."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if treatment_dict.empty or "drug_key" not in treatment_dict.columns:
        return pd.DataFrame(), pd.DataFrame()

    td = treatment_dict.copy()
    if "component_classes" not in td.columns:
        td["component_classes"] = ""
    if "treatment_components" not in td.columns:
        td["treatment_components"] = ""

    class_rows: List[dict] = []
    component_rows: List[dict] = []
    for _, row in td.iterrows():
        drug_key = str(row["drug_key"])
        for cls in split_semicolon_values(row.get("component_classes", "")):
            class_rows.append({"drug_key": drug_key, "component_class": cls})
        for comp in split_semicolon_values(row.get("treatment_components", "")):
            component_rows.append({"drug_key": drug_key, "component_name": comp})

    class_df = pd.DataFrame(class_rows).drop_duplicates() if class_rows else pd.DataFrame()
    component_df = pd.DataFrame(component_rows).drop_duplicates() if component_rows else pd.DataFrame()

    effects = theme_effects.copy()
    effects["signed_theme_effect"] = safe_float_series(effects["signed_theme_effect"])
    effects["absolute_theme_effect"] = safe_float_series(effects["absolute_theme_effect"]) if "absolute_theme_effect" in effects.columns else effects["signed_theme_effect"].abs()

    out_class = pd.DataFrame()
    if not class_df.empty:
        merged = class_df.merge(effects, on="drug_key", how="inner")
        rows = []
        for (component_class, theme), sub in merged.groupby(["component_class", "biological_theme"], dropna=False):
            signed_sum = float(sub["signed_theme_effect"].sum())
            pos = int((sub["signed_theme_effect"] > 0).sum())
            neg = int((sub["signed_theme_effect"] < 0).sum())
            rows.append({
                "component_class": component_class,
                "biological_theme": theme,
                "treatment_count": int(sub["drug_key"].nunique()),
                "signed_theme_effect_sum": signed_sum,
                "absolute_theme_effect_sum": float(sub["absolute_theme_effect"].sum()),
                "mean_signed_theme_effect": float(sub["signed_theme_effect"].mean()),
                "sensitivity_associated_treatment_count": pos,
                "resistance_associated_treatment_count": neg,
                "consensus_direction": consensus_label(pos, neg, signed_sum),
                "example_treatments": summarize_examples(sub["drug_key"], 6),
            })
        out_class = pd.DataFrame(rows).sort_values(["absolute_theme_effect_sum", "component_class", "biological_theme"], ascending=[False, True, True])

    out_component = pd.DataFrame()
    if not component_df.empty:
        merged = component_df.merge(effects, on="drug_key", how="inner")
        rows = []
        for (component_name, theme), sub in merged.groupby(["component_name", "biological_theme"], dropna=False):
            signed_sum = float(sub["signed_theme_effect"].sum())
            pos = int((sub["signed_theme_effect"] > 0).sum())
            neg = int((sub["signed_theme_effect"] < 0).sum())
            rows.append({
                "component_name": component_name,
                "biological_theme": theme,
                "treatment_count": int(sub["drug_key"].nunique()),
                "signed_theme_effect_sum": signed_sum,
                "absolute_theme_effect_sum": float(sub["absolute_theme_effect"].sum()),
                "mean_signed_theme_effect": float(sub["signed_theme_effect"].mean()),
                "sensitivity_associated_treatment_count": pos,
                "resistance_associated_treatment_count": neg,
                "consensus_direction": consensus_label(pos, neg, signed_sum),
                "example_treatments": summarize_examples(sub["drug_key"], 6),
            })
        out_component = pd.DataFrame(rows).sort_values(["absolute_theme_effect_sum", "component_name", "biological_theme"], ascending=[False, True, True])

    return out_class, out_component


def build_sample_mechanism_tables(score_df: pd.DataFrame, theme_effects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper routine for this prediction_interpretation_model script.
    Keeps data movement, QC, or reporting behavior explicit and auditable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if score_df.empty or theme_effects.empty:
        return pd.DataFrame(), pd.DataFrame()

    score = score_df.copy()
    theme = theme_effects.copy()

    score["net_signed_spatial_interpretation_score"] = safe_float_series(score["net_signed_spatial_interpretation_score"])
    theme["signed_theme_effect"] = safe_float_series(theme["signed_theme_effect"])

    keep = [c for c in ["sample_id", "drug_key", "net_signed_spatial_interpretation_score", "sensitivity_alignment_score", "resistance_alignment_score", "sample_spatial_alignment_label"] if c in score.columns]
    merged = score[keep].merge(
        theme[["drug_key", "biological_theme", "signed_theme_effect", "absolute_theme_effect"]].copy(),
        on="drug_key",
        how="inner",
    )
    merged["sample_theme_alignment_score"] = merged["net_signed_spatial_interpretation_score"] * merged["signed_theme_effect"]
    merged["sample_theme_alignment_abs"] = merged["sample_theme_alignment_score"].abs()
    merged["sample_theme_alignment_direction"] = np.where(
        merged["sample_theme_alignment_score"] > 0,
        "sample_aligned_with_theme_sensitivity_pattern",
        np.where(merged["sample_theme_alignment_score"] < 0, "sample_aligned_with_theme_resistance_pattern", "balanced_or_ambiguous"),
    )

    summary_rows: List[dict] = []
    for sample_id, sub in merged.groupby("sample_id", dropna=False):
        summary_rows.append({
            "sample_id": sample_id,
            "n_validated_treatments_with_theme_scores": int(sub["drug_key"].nunique()),
            "n_biological_themes": int(sub["biological_theme"].nunique()),
            "mean_sample_theme_alignment_score": float(sub["sample_theme_alignment_score"].mean()),
            "sum_abs_sample_theme_alignment_score": float(sub["sample_theme_alignment_abs"].sum()),
            "top_sensitivity_aligned_themes": summarize_examples(sub.sort_values("sample_theme_alignment_score", ascending=False)["biological_theme"], 5),
            "top_resistance_aligned_themes": summarize_examples(sub.sort_values("sample_theme_alignment_score", ascending=True)["biological_theme"], 5),
            "interpretation_caveat": "Sample mechanism patterns summarize model-derived spatial alignments only; not clinical recommendations.",
        })

    summary = pd.DataFrame(summary_rows).sort_values(["sum_abs_sample_theme_alignment_score", "sample_id"], ascending=[False, True])
    return merged, summary


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

    treatment_dict_path = required_file(step02 / "02_treatment_dictionary" / "treatment_dictionary.tsv")
    feature_effect_path = required_file(step03 / "01_treatment_feature_effects" / "signed_treatment_feature_effects.tsv")
    theme_effect_path = required_file(step03 / "02_treatment_theme_effects" / "signed_treatment_theme_effects.tsv")
    cards_path = required_file(step04 / "02_cards_tsv" / "treatment_interpretation_cards.tsv")
    sample_scores_path = required_file(step05 / "01_sample_treatment_scores" / "sample_treatment_signed_interpretation_scores.tsv")

    treatment_dict = read_table(treatment_dict_path)
    feature_effects = read_table(feature_effect_path)
    theme_effects = read_table(theme_effect_path)
    cards = read_table(cards_path)
    sample_scores = read_table(sample_scores_path)

    step_root = output_root / "06_mechanism_atlas"
    theme_dir = step_root / "01_theme_atlas"
    feature_dir = step_root / "02_feature_atlas"
    similarity_dir = step_root / "03_treatment_similarity"
    component_dir = step_root / "04_component_atlas"
    sample_dir = step_root / "05_sample_mechanism_patterns"
    qc_dir = step_root / "06_qc"
    report_dir = step_root / "07_reports"

    for path in [theme_dir, feature_dir, similarity_dir, component_dir, sample_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        theme_atlas, theme_matrix, theme_abs_matrix = build_theme_atlas(theme_effects, treatment_dict)
        feature_atlas = build_feature_atlas(feature_effects)
        sim_matrix, sim_edges = cosine_similarity_matrix(theme_matrix)
        component_class_atlas, component_atlas = build_component_atlas(theme_effects, treatment_dict)
        sample_theme_scores, sample_theme_summary = build_sample_mechanism_tables(sample_scores, theme_effects)

        write_tsv(theme_dir / "cross_treatment_biology_theme_atlas.tsv", theme_atlas)
        write_tsv(theme_dir / "treatment_theme_signed_effect_matrix.tsv", theme_matrix)
        write_tsv(theme_dir / "treatment_theme_absolute_effect_matrix.tsv", theme_abs_matrix)
        write_tsv(feature_dir / "cross_treatment_feature_effect_atlas.tsv", feature_atlas)
        write_tsv(similarity_dir / "treatment_theme_similarity_matrix.tsv", sim_matrix)
        write_tsv(similarity_dir / "treatment_theme_similarity_edges.tsv", sim_edges)
        write_tsv(component_dir / "component_class_mechanism_atlas.tsv", component_class_atlas)
        write_tsv(component_dir / "component_mechanism_atlas.tsv", component_atlas)
        write_tsv(sample_dir / "sample_treatment_theme_alignment_scores.tsv", sample_theme_scores)
        write_tsv(sample_dir / "sample_mechanism_summary.tsv", sample_theme_summary)

        treatment_count = int(theme_effects["drug_key"].nunique()) if "drug_key" in theme_effects.columns else 0
        theme_count = int(theme_effects["biological_theme"].nunique()) if "biological_theme" in theme_effects.columns else 0
        card_count = int(cards["drug_key"].nunique()) if "drug_key" in cards.columns else 0
        sample_count = int(sample_scores["sample_id"].nunique()) if "sample_id" in sample_scores.columns else 0

        add_qc(qc, "mechanism_atlas_treatment_count", "pass" if treatment_count == 27 else "warn", treatment_count, 27, "Expected validated treatment count in signed theme effects.")
        add_qc(qc, "mechanism_atlas_theme_count", "pass" if theme_count == 11 else "warn", theme_count, 11, "Expected recurrent V2 biology themes represented in atlas.")
        add_qc(qc, "treatment_card_count_carried_forward", "pass" if card_count == 27 else "warn", card_count, 27, "Treatment card table carried forward.")
        add_qc(qc, "theme_atlas_rows", "pass" if len(theme_atlas) > 0 else "fail", len(theme_atlas), ">0", "Cross-treatment theme atlas created.")
        add_qc(qc, "feature_atlas_rows", "pass" if len(feature_atlas) > 0 else "fail", len(feature_atlas), ">0", "Cross-treatment feature atlas created.")
        add_qc(qc, "treatment_similarity_edges", "pass" if len(sim_edges) > 0 else "fail", len(sim_edges), ">0", "Treatment similarity edges created.")
        add_qc(qc, "component_class_atlas_rows", "pass" if len(component_class_atlas) > 0 else "warn", len(component_class_atlas), ">0", "Component-class mechanism atlas created.")
        add_qc(qc, "sample_mechanism_summary_rows", "pass" if len(sample_theme_summary) > 0 else "fail", len(sample_theme_summary), ">0", "Sample mechanism summary created.")
        add_qc(qc, "sample_mechanism_coverage", "pass" if sample_count >= 90 else "warn", sample_count, ">=90", "Sample coverage reflects validated-treatment eligible sample-treatment rows.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        theme_atlas = pd.DataFrame()
        feature_atlas = pd.DataFrame()
        sim_edges = pd.DataFrame()
        component_class_atlas = pd.DataFrame()
        sample_theme_summary = pd.DataFrame()

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step06_mechanism_atlas_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "theme_atlas_rows": len(theme_atlas),
        "feature_atlas_rows": len(feature_atlas),
        "treatment_similarity_edges": len(sim_edges),
        "component_class_atlas_rows": len(component_class_atlas),
        "sample_mechanism_summary_rows": len(sample_theme_summary),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step06_summary.json", summary)

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 06 REPORT",
        "",
        f"status: {status}",
        f"prepared_input_root: {prepared_root}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        f"cross_treatment_biology_theme_atlas: {theme_dir / 'cross_treatment_biology_theme_atlas.tsv'}",
        f"treatment_theme_signed_effect_matrix: {theme_dir / 'treatment_theme_signed_effect_matrix.tsv'}",
        f"cross_treatment_feature_effect_atlas: {feature_dir / 'cross_treatment_feature_effect_atlas.tsv'}",
        f"treatment_theme_similarity_edges: {similarity_dir / 'treatment_theme_similarity_edges.tsv'}",
        f"component_class_mechanism_atlas: {component_dir / 'component_class_mechanism_atlas.tsv'}",
        f"sample_mechanism_summary: {sample_dir / 'sample_mechanism_summary.tsv'}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Interpretation",
        "The mechanism atlas summarizes signed residual spatial effects across validated treatments, features, biological themes, treatment components, and sample-level spatial alignment scores.",
        "",
        "Caveat",
        "Mechanism labels are interpretation-layer summaries of model-derived associations. They are not causal mechanisms and are not clinical treatment recommendations.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step06_mechanism_atlas_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 06 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"theme_atlas_rows: {len(theme_atlas)}")
    print(f"feature_atlas_rows: {len(feature_atlas)}")
    print(f"treatment_similarity_edges: {len(sim_edges)}")
    print(f"component_class_atlas_rows: {len(component_class_atlas)}")
    print(f"sample_mechanism_summary_rows: {len(sample_theme_summary)}")
    print(f"report: {report_dir / 'step06_mechanism_atlas_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())
