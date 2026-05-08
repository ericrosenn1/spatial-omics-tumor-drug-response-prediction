#!/usr/bin/env python
"""
Script:
    04_make_single_slide_prediction_table.py

Purpose:
    Convert transfer scores and feature/theme contributions into the
    single-slide drug-response interpretation table.

Output:
    single_slide_drug_response_interpretation_table.tsv
    single_slide_drug_response_interpretation_table.xlsx

Policy:
    Research-use spatial response alignment only.
    Not a clinical treatment recommendation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import traceback
from typing import List

import pandas as pd

from _stim_utils import (
    add_qc,
    ensure_dir,
    load_pim_treatment_cards,
    open_folder,
    read_table,
    report_status_from_qc,
    save_output_manifest,
    summarize_examples,
    write_json,
    write_text_report,
    write_tsv,
)


BARRIER_TERMS = [
    "barrier",
    "access",
    "penetration",
    "hypoxia",
    "hypoxic",
    "ecm",
    "stromal",
    "stroma",
    "vascular",
    "angiogenic",
    "distance",
    "core",
    "boundary",
    "myeloid",
    "macrophage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\Adv_Omics_Fenyo\project")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--pim-run-root", required=True)
    parser.add_argument("--spatial-feature-run-root", default="")
    parser.add_argument("--single-slide-feature-table", default="")
    parser.add_argument("--sample-id", default="TRANSFER_SAMPLE_001")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-n-features", type=int, default=5)
    parser.add_argument("--top-n-themes", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def compact_feature_phrase(row: dict) -> str:
    feature = str(row.get("feature_name", ""))
    theme = str(row.get("biological_theme", ""))
    contribution = fmt_float(row.get("transfer_feature_contribution", ""), 4)
    if theme and theme.lower() != "nan":
        return f"{feature} [{theme}; contribution={contribution}]"
    return f"{feature} [contribution={contribution}]"


def compact_theme_phrase(row: dict) -> str:
    theme = str(row.get("biological_theme", ""))
    contribution = fmt_float(row.get("transfer_theme_contribution_sum", ""), 4)
    return f"{theme} [contribution={contribution}]"


def choose_barrier_features(negative_features: pd.DataFrame, top_n: int) -> str:
    if negative_features.empty:
        return "none"

    tmp = negative_features.copy()
    tmp["barrier_like"] = tmp.apply(
        lambda r: any(term in str(r.get("feature_name", "")).lower() or term in str(r.get("biological_theme", "")).lower() for term in BARRIER_TERMS),
        axis=1,
    )
    tmp["abs_contribution"] = pd.to_numeric(tmp["transfer_feature_contribution"], errors="coerce").abs()
    barrier = tmp[tmp["barrier_like"]].sort_values("abs_contribution", ascending=False).head(top_n)
    if barrier.empty:
        barrier = tmp.sort_values("abs_contribution", ascending=False).head(top_n)
    return "; ".join(compact_feature_phrase(r) for r in barrier.to_dict("records")) if not barrier.empty else "none"


def make_explanation(row: dict, pos_features: str, neg_features: str, pos_themes: str, neg_themes: str, barrier_text: str) -> str:
    drug = str(row.get("drug_key", ""))
    label = str(row.get("spatial_response_alignment", ""))
    score = fmt_float(row.get("transfer_alignment_score", ""), 4)
    confidence = str(row.get("confidence_level", ""))

    if "sensitivity_aligned" in label:
        lead = f"For {drug}, the slide is spatially sensitivity-aligned in the frozen V2 residual-biology atlas."
        why = f"Main supporting features/themes: {pos_features}; {pos_themes}."
        caution = f"Potential resistance or penetration-barrier signals still present: {barrier_text if barrier_text != 'none' else neg_features}."
    elif "resistance" in label or "barrier" in label:
        lead = f"For {drug}, the slide is spatially resistance/barrier-aligned in the frozen V2 residual-biology atlas."
        why = f"Main resistance/barrier features/themes: {barrier_text if barrier_text != 'none' else neg_features}; {neg_themes}."
        caution = f"Sensitivity-supporting signals were weaker or counterbalanced: {pos_features}."
    else:
        lead = f"For {drug}, the slide has an indeterminate or balanced spatial response-alignment profile."
        why = f"Sensitivity-supporting and resistance/barrier-supporting signals are mixed. Positive signals: {pos_features}. Negative signals: {neg_features}."
        caution = f"Dominant themes should be reviewed rather than treated as a binary prediction."

    return f"{lead} Alignment score={score}; confidence={confidence}. {why} {caution} Research-use only; not a clinical treatment recommendation."


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    output_root = Path(args.output_root)
    pim_run_root = Path(args.pim_run_root)

    score_path = output_root / "03_transfer_scores" / "01_treatment_alignment_scores" / "single_slide_treatment_alignment_scores.tsv"
    feature_contrib_path = output_root / "03_transfer_scores" / "02_feature_contributions" / "single_slide_treatment_feature_contributions.tsv"
    theme_contrib_path = output_root / "03_transfer_scores" / "03_theme_contributions" / "single_slide_treatment_theme_contributions.tsv"

    for path in [score_path, feature_contrib_path, theme_contrib_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Step 03 output missing: {path}")

    step_root = output_root / "04_prediction_table"
    table_dir = step_root / "01_prediction_tables"
    report_dir = step_root / "02_reports"
    qc_dir = step_root / "03_qc"

    for path in [table_dir, report_dir, qc_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        scores = read_table(score_path)
        feature_contrib = read_table(feature_contrib_path)
        theme_contrib = read_table(theme_contrib_path)
        cards = load_pim_treatment_cards(pim_run_root)

        card_cols = [c for c in [
            "drug_key",
            "treatment_components",
            "component_classes",
            "interpretation_tier",
            "label_shuffle_validation_status",
            "observed_test_pearson_mean",
            "observed_test_r2_mean",
            "fdr_q_pearson",
            "card_path",
        ] if c in cards.columns]
        if card_cols:
            scores = scores.merge(cards[card_cols].drop_duplicates("drug_key"), on="drug_key", how="left")

        rows: List[dict] = []

        for _, score_row in scores.iterrows():
            sample_id = str(score_row["sample_id"])
            drug_key = str(score_row["drug_key"])

            fsub = feature_contrib[
                (feature_contrib["sample_id"].astype(str) == sample_id)
                & (feature_contrib["drug_key"].astype(str) == drug_key)
            ].copy()
            tsub = theme_contrib[
                (theme_contrib["sample_id"].astype(str) == sample_id)
                & (theme_contrib["drug_key"].astype(str) == drug_key)
            ].copy()

            fsub["transfer_feature_contribution"] = pd.to_numeric(fsub["transfer_feature_contribution"], errors="coerce")
            tsub["transfer_theme_contribution_sum"] = pd.to_numeric(tsub["transfer_theme_contribution_sum"], errors="coerce")

            pos_f = fsub[fsub["transfer_feature_contribution"] > 0].sort_values("transfer_feature_contribution", ascending=False).head(args.top_n_features)
            neg_f = fsub[fsub["transfer_feature_contribution"] < 0].assign(
                abs_contribution=lambda d: d["transfer_feature_contribution"].abs()
            ).sort_values("abs_contribution", ascending=False).head(args.top_n_features)

            pos_t = tsub[tsub["transfer_theme_contribution_sum"] > 0].sort_values("transfer_theme_contribution_sum", ascending=False).head(args.top_n_themes)
            neg_t = tsub[tsub["transfer_theme_contribution_sum"] < 0].assign(
                abs_contribution=lambda d: d["transfer_theme_contribution_sum"].abs()
            ).sort_values("abs_contribution", ascending=False).head(args.top_n_themes)

            pos_features = "; ".join(compact_feature_phrase(r) for r in pos_f.to_dict("records")) if not pos_f.empty else "none"
            neg_features = "; ".join(compact_feature_phrase(r) for r in neg_f.to_dict("records")) if not neg_f.empty else "none"
            pos_themes = "; ".join(compact_theme_phrase(r) for r in pos_t.to_dict("records")) if not pos_t.empty else "none"
            neg_themes = "; ".join(compact_theme_phrase(r) for r in neg_t.to_dict("records")) if not neg_t.empty else "none"
            barrier_text = choose_barrier_features(neg_f, args.top_n_features)

            d = score_row.to_dict()
            explanation = make_explanation(d, pos_features, neg_features, pos_themes, neg_themes, barrier_text)

            alignment = str(d.get("spatial_response_alignment", ""))
            if "sensitivity_aligned" in alignment:
                research_prediction = "favorable_spatial_profile"
                predicted_effective_research_label = "Yes_research_spatial_alignment"
            elif "resistance" in alignment or "barrier" in alignment:
                research_prediction = "unfavorable_spatial_barrier_profile"
                predicted_effective_research_label = "No_research_spatial_alignment"
            else:
                research_prediction = "indeterminate_spatial_profile"
                predicted_effective_research_label = "Indeterminate"

            rows.append({
                "sample_id": sample_id,
                "drug_key": drug_key,
                "research_prediction_label": research_prediction,
                "predicted_effective_research_label": predicted_effective_research_label,
                "probability_effective_research_not_calibrated": d.get("spatial_favorable_score_0_1_not_calibrated", ""),
                "spatial_alignment_score": d.get("transfer_alignment_score", ""),
                "confidence_level": d.get("confidence_level", ""),
                "feature_effect_coverage_fraction": d.get("feature_effect_coverage_fraction", ""),
                "spatial_response_alignment": alignment,
                "treatment_components": d.get("treatment_components", ""),
                "component_classes": d.get("component_classes", ""),
                "interpretation_tier": d.get("interpretation_tier", ""),
                "label_shuffle_validation_status": d.get("label_shuffle_validation_status", ""),
                "model_observed_test_pearson_mean": d.get("observed_test_pearson_mean", ""),
                "model_observed_test_r2_mean": d.get("observed_test_r2_mean", ""),
                "model_fdr_q_pearson": d.get("fdr_q_pearson", ""),
                "top_sensitivity_features": pos_features,
                "top_resistance_or_barrier_features": neg_features,
                "dominant_sensitivity_themes": pos_themes,
                "dominant_resistance_or_barrier_themes": neg_themes,
                "barrier_interpretation": barrier_text,
                "explanation": explanation,
                "research_use_caveat": "Research-use spatial transfer inference. Not a calibrated clinical drug-efficacy probability and not a treatment recommendation.",
            })

        table = pd.DataFrame(rows).sort_values(["sample_id", "spatial_alignment_score"], ascending=[True, False])

        table_path = table_dir / "single_slide_drug_response_interpretation_table.tsv"
        write_tsv(table_path, table)

        xlsx_path = table_dir / "single_slide_drug_response_interpretation_table.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                table.to_excel(writer, sheet_name="Prediction_Table", index=False)
                scores.to_excel(writer, sheet_name="Raw_Scores", index=False)
                feature_contrib.head(5000).to_excel(writer, sheet_name="Feature_Contrib_Top5000", index=False)
                theme_contrib.to_excel(writer, sheet_name="Theme_Contrib", index=False)
        except Exception as exc:
            warnings.append(f"Excel write failed: {exc}")

        add_qc(qc, "prediction_table_rows", "pass" if len(table) > 0 else "fail", len(table), ">0", "Prediction interpretation table generated.")
        add_qc(qc, "prediction_table_treatment_count", "pass" if table["drug_key"].nunique() == 27 else "warn", table["drug_key"].nunique() if not table.empty else 0, 27, "Expected 27 validated treatments.")
        add_qc(qc, "prediction_table_has_explanations", "pass" if table["explanation"].astype(str).str.len().gt(20).all() else "fail", int(table["explanation"].astype(str).str.len().gt(20).sum()) if not table.empty else 0, len(table), "Every prediction row should have a readable explanation.")
        add_qc(qc, "prediction_table_has_barrier_interpretations", "pass" if "barrier_interpretation" in table.columns else "fail", "barrier_interpretation" in table.columns, True, "Barrier interpretation column present.")
        add_qc(qc, "prediction_table_excel_exists", "pass" if xlsx_path.exists() else "warn", xlsx_path.exists(), True, "Excel version of prediction table created.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        table = pd.DataFrame()
        table_path = table_dir / "single_slide_drug_response_interpretation_table.tsv"
        xlsx_path = table_dir / "single_slide_drug_response_interpretation_table.xlsx"

    status = report_status_from_qc(qc, errors)
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step04_prediction_table_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prediction_table_rows": len(table),
        "prediction_table_path": str(table_path),
        "prediction_table_xlsx": str(xlsx_path),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "spatial_transfer_inference_model_step04_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL STEP 04 REPORT",
        "",
        f"status: {status}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        str(table_path),
        str(xlsx_path),
        "",
        "Prediction table note",
        "The probability_effective_research_not_calibrated column is a monotonic spatial-alignment score transformed to 0-1 scale. It is not a calibrated clinical efficacy probability.",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step04_make_single_slide_prediction_table_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL STEP 04 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"prediction_table_rows: {len(table)}")
    print(f"prediction_table: {table_path}")
    print(f"report: {report_dir / 'step04_make_single_slide_prediction_table_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())