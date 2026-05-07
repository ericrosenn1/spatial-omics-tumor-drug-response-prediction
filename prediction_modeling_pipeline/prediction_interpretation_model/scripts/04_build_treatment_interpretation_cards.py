#!/usr/bin/env python
"""
Script:
    04_build_treatment_interpretation_cards.py

Description:
    Creates one readable treatment interpretation card per label-shuffle-validated
    V2 treatment, plus compact card tables for final reporting.

Instructions:
    Run after Step 03 passes. Each card should contain model/validation evidence,
    sensitivity-associated spatial effects, resistance-associated effects, biology
    themes, and interpretation caveats.

Source-truth policy:
    Treatment cards summarize model-derived residual spatial associations. They do
    not establish causality and are not clinical treatment recommendations.
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
from typing import List

import numpy as np
import pandas as pd

from _pim_utils import (
    add_qc,
    ensure_dir,
    load_prepared_index,
    open_folder,
    read_table,
    safe_filename,
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
    parser.add_argument("--top-n-features", type=int, default=8)
    parser.add_argument("--top-n-themes", type=int, default=5)
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def fmt_num(value: object, digits: int = 4) -> str:
    """Format numeric values for readable reports.
    Returns blank text for missing or nonfinite values."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def top_feature_summary(sub: pd.DataFrame, direction: str, n: int) -> str:
    """Summarize top signed features for a card.
    Separates sensitivity-associated and resistance-associated directions."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if sub.empty:
        return "none"
    if direction == "positive":
        part = sub[pd.to_numeric(sub["signed_effect"], errors="coerce") > 0].copy()
        label = "sensitivity-associated"
        ascending = False
    else:
        part = sub[pd.to_numeric(sub["signed_effect"], errors="coerce") < 0].copy()
        label = "resistance-associated"
        ascending = True

    if part.empty:
        return "none"

    part["abs_effect"] = pd.to_numeric(part["signed_effect"], errors="coerce").abs()
    part = part.sort_values("abs_effect", ascending=False).head(n)
    pieces = []
    for _, row in part.iterrows():
        pieces.append(
            f"{row.get('feature_name', '')} ({label}; r={fmt_num(row.get('feature_target_pearson'))}; theme={row.get('biological_theme', '')})"
        )
    return "; ".join(pieces)


def top_theme_summary(sub: pd.DataFrame, direction: str, n: int) -> str:
    """Summarize top signed themes for a card.
    Keeps treatment cards compact and biologically readable."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if sub.empty:
        return "none"
    if direction == "positive":
        part = sub[pd.to_numeric(sub["signed_theme_effect"], errors="coerce") > 0].copy()
        label = "sensitivity-associated"
    else:
        part = sub[pd.to_numeric(sub["signed_theme_effect"], errors="coerce") < 0].copy()
        label = "resistance-associated"

    if part.empty:
        return "none"

    part["abs_effect"] = pd.to_numeric(part["signed_theme_effect"], errors="coerce").abs()
    part = part.sort_values("abs_effect", ascending=False).head(n)
    pieces = []
    for _, row in part.iterrows():
        pieces.append(
            f"{row.get('biological_theme', '')} ({label}; signed={fmt_num(row.get('signed_theme_effect'))}; n_features={row.get('n_features', '')})"
        )
    return "; ".join(pieces)


def build_card_text(row: pd.Series, features: pd.DataFrame, themes: pd.DataFrame, top_n_features: int, top_n_themes: int) -> str:
    """Build the text body for one treatment interpretation card.
    Includes evidence, spatial mechanisms, caveats, and non-clinical framing."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    drug_key = str(row.get("drug_key", ""))
    sensitivity_features = top_feature_summary(features, "positive", top_n_features)
    resistance_features = top_feature_summary(features, "negative", top_n_features)
    sensitivity_themes = top_theme_summary(themes, "positive", top_n_themes)
    resistance_themes = top_theme_summary(themes, "negative", top_n_themes)

    lines = [
        "PREDICTION INTERPRETATION MODEL TREATMENT CARD",
        "",
        f"Treatment key: {drug_key}",
        f"Components: {row.get('treatment_components', '')}",
        f"Component classes: {row.get('component_classes', '')}",
        f"Validation status: {row.get('label_shuffle_validation_status', row.get('integrated_interpretation_status', ''))}",
        f"Interpretation tier: {row.get('interpretation_tier', '')}",
        "",
        "Model and validation evidence",
        f"Observed test Pearson mean: {fmt_num(row.get('observed_test_pearson_mean', row.get('test_pearson_mean', '')))}",
        f"Observed test R2 mean: {fmt_num(row.get('observed_test_r2_mean', row.get('test_r2_mean', '')))}",
        f"Observed RMSE improvement versus baseline: {fmt_num(row.get('observed_rmse_improvement_vs_baseline_mean', row.get('rmse_improvement_vs_baseline_mean', '')))}",
        f"Empirical p-value, Pearson: {fmt_num(row.get('empirical_p_pearson', ''))}",
        f"FDR q-value, Pearson: {fmt_num(row.get('fdr_q_pearson', ''))}",
        f"Null shuffles: {row.get('n_null_shuffles', '')}",
        f"Samples available: {row.get('n_samples_total', row.get('n_samples', ''))}",
        "",
        "Sensitivity-associated spatial effects",
        sensitivity_features,
        "",
        "Resistance-associated spatial effects",
        resistance_features,
        "",
        "Sensitivity-associated biology themes",
        sensitivity_themes,
        "",
        "Resistance-associated biology themes",
        resistance_themes,
        "",
        "Interpretation",
        "This card summarizes V2 residual spatial biology for the treatment key above. Positive directionality means higher feature values are associated with above-prior fused response residuals for this treatment. Negative directionality means higher feature values are associated with below-prior fused response residuals.",
        "",
        "Caveats",
        "This is a biological interpretation of model-derived residual associations, not causal proof.",
        "This is not a clinical treatment recommendation.",
        "Treatment keys may represent multi-agent regimens as harmonized by the upstream teacher and V2 modeling layers.",
    ]
    return "\n".join(lines)


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

    step02_root = output_root / "02_feature_and_treatment_dictionary"
    step03_root = output_root / "03_signed_spatial_effects"

    treatment_dict_path = step02_root / "02_treatment_dictionary" / "treatment_dictionary.tsv"
    feature_effect_path = step03_root / "01_treatment_feature_effects" / "signed_treatment_feature_effects.tsv"
    theme_effect_path = step03_root / "02_treatment_theme_effects" / "signed_treatment_theme_effects.tsv"

    for path in [treatment_dict_path, feature_effect_path, theme_effect_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required upstream file missing: {path}")

    treatment_dict = read_table(treatment_dict_path)
    feature_effects = read_table(feature_effect_path)
    theme_effects = read_table(theme_effect_path)

    step_root = output_root / "04_treatment_interpretation_cards"
    card_txt_dir = step_root / "01_cards_txt"
    card_table_dir = step_root / "02_cards_tsv"
    qc_dir = step_root / "03_qc"
    report_dir = step_root / "04_reports"

    for path in [card_txt_dir, card_table_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []
    card_rows: List[dict] = []
    top_feature_rows: List[dict] = []
    top_theme_rows: List[dict] = []

    try:
        if "label_shuffle_validated" in treatment_dict.columns:
            validated = treatment_dict[treatment_dict["label_shuffle_validated"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
        else:
            validated = treatment_dict.copy()

        for _, row in validated.sort_values("drug_key").iterrows():
            drug_key = str(row["drug_key"])
            fsub = feature_effects[feature_effects["drug_key"].astype(str) == drug_key].copy()
            tsub = theme_effects[theme_effects["drug_key"].astype(str) == drug_key].copy()

            if not fsub.empty:
                fsub["abs_effect"] = pd.to_numeric(fsub["signed_effect"], errors="coerce").abs()
                topf = fsub.sort_values("abs_effect", ascending=False).head(max(args.top_n_features * 2, 10)).copy()
                topf["card_drug_key"] = drug_key
                top_feature_rows.extend(topf.to_dict("records"))

            if not tsub.empty:
                tsub["abs_effect"] = pd.to_numeric(tsub["signed_theme_effect"], errors="coerce").abs()
                topt = tsub.sort_values("abs_effect", ascending=False).head(max(args.top_n_themes * 2, 8)).copy()
                topt["card_drug_key"] = drug_key
                top_theme_rows.extend(topt.to_dict("records"))

            sensitivity_features = top_feature_summary(fsub, "positive", args.top_n_features)
            resistance_features = top_feature_summary(fsub, "negative", args.top_n_features)
            sensitivity_themes = top_theme_summary(tsub, "positive", args.top_n_themes)
            resistance_themes = top_theme_summary(tsub, "negative", args.top_n_themes)

            card_filename = safe_filename(drug_key) + ".txt"
            card_path = card_txt_dir / card_filename
            write_text_report(card_path, build_card_text(row, fsub, tsub, args.top_n_features, args.top_n_themes))

            card_rows.append({
                "drug_key": drug_key,
                "card_path": str(card_path),
                "treatment_components": row.get("treatment_components", ""),
                "component_classes": row.get("component_classes", ""),
                "interpretation_tier": row.get("interpretation_tier", ""),
                "label_shuffle_validation_status": row.get("label_shuffle_validation_status", row.get("integrated_interpretation_status", "")),
                "observed_test_pearson_mean": row.get("observed_test_pearson_mean", row.get("test_pearson_mean", "")),
                "observed_test_r2_mean": row.get("observed_test_r2_mean", row.get("test_r2_mean", "")),
                "observed_rmse_improvement_vs_baseline_mean": row.get("observed_rmse_improvement_vs_baseline_mean", row.get("rmse_improvement_vs_baseline_mean", "")),
                "empirical_p_pearson": row.get("empirical_p_pearson", ""),
                "fdr_q_pearson": row.get("fdr_q_pearson", ""),
                "n_samples_total": row.get("n_samples_total", row.get("n_samples", "")),
                "n_signed_features": int(fsub["feature_name"].nunique()) if not fsub.empty else 0,
                "n_signed_themes": int(tsub["biological_theme"].nunique()) if not tsub.empty else 0,
                "top_sensitivity_features": sensitivity_features,
                "top_resistance_features": resistance_features,
                "top_sensitivity_themes": sensitivity_themes,
                "top_resistance_themes": resistance_themes,
                "interpretation_caveat": "Model-derived spatial residual association; not causal proof and not a treatment recommendation.",
            })

        cards = pd.DataFrame(card_rows)
        top_features = pd.DataFrame(top_feature_rows)
        top_themes = pd.DataFrame(top_theme_rows)

        write_tsv(card_table_dir / "treatment_interpretation_cards.tsv", cards)
        write_tsv(card_table_dir / "treatment_card_top_features.tsv", top_features)
        write_tsv(card_table_dir / "treatment_card_top_themes.tsv", top_themes)

        add_qc(qc, "treatment_card_count", "pass" if len(cards) == 27 else "warn", len(cards), 27, "One card should be generated per label-shuffle-validated treatment.")
        add_qc(qc, "cards_with_signed_features", "pass" if (cards.get("n_signed_features", pd.Series(dtype=int)) > 0).all() else "fail", int((cards.get("n_signed_features", pd.Series(dtype=int)) > 0).sum()), len(cards), "Each card should have signed feature evidence.")
        add_qc(qc, "cards_with_signed_themes", "pass" if (cards.get("n_signed_themes", pd.Series(dtype=int)) > 0).all() else "fail", int((cards.get("n_signed_themes", pd.Series(dtype=int)) > 0).sum()), len(cards), "Each card should have signed theme evidence.")
        add_qc(qc, "txt_cards_written", "pass" if all(Path(p).exists() for p in cards.get("card_path", [])) else "fail", int(sum(Path(p).exists() for p in cards.get("card_path", []))), len(cards), "Every card path should exist.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        cards = pd.DataFrame()
        top_features = pd.DataFrame()
        top_themes = pd.DataFrame()

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step04_treatment_card_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "treatment_card_rows": len(cards),
        "top_feature_rows": len(top_features),
        "top_theme_rows": len(top_themes),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step04_summary.json", summary)

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 04 REPORT",
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
        f"card_text_folder: {card_txt_dir}",
        f"treatment_interpretation_cards: {card_table_dir / 'treatment_interpretation_cards.tsv'}",
        f"treatment_card_top_features: {card_table_dir / 'treatment_card_top_features.tsv'}",
        f"treatment_card_top_themes: {card_table_dir / 'treatment_card_top_themes.tsv'}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Caveat",
        "Treatment cards summarize spatial residual associations. They do not establish causality and are not clinical recommendations.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step04_treatment_interpretation_cards_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 04 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"treatment_cards: {len(cards)}")
    print(f"top_feature_rows: {len(top_features)}")
    print(f"top_theme_rows: {len(top_themes)}")
    print(f"report: {report_dir / 'step04_treatment_interpretation_cards_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())
