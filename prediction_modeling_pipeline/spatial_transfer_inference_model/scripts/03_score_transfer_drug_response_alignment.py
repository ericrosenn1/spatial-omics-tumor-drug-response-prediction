#!/usr/bin/env python
"""
Script:
    03_score_transfer_drug_response_alignment.py

Purpose:
    Apply frozen PIM signed treatment-feature effects to one or more V2-scaled
    transfer feature vectors.

Interpretation:
    Positive transfer alignment score means the slide's spatial profile is more
    aligned with sensitivity-associated residual biology for that treatment.
    Negative score means the slide's spatial profile is more aligned with
    resistance/barrier-associated residual biology for that treatment.

Policy:
    Research-use interpretation only. Not a clinical treatment recommendation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import traceback
from typing import List

import numpy as np
import pandas as pd

from _stim_utils import (
    add_qc,
    ensure_dir,
    load_pim_signed_feature_effects,
    load_pim_signed_theme_effects,
    load_pim_treatment_cards,
    numeric_series,
    open_folder,
    read_table,
    report_status_from_qc,
    save_output_manifest,
    sigmoid,
    summarize_examples,
    write_json,
    write_text_report,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"D:\Adv_Omics_Fenyo\project")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--pim-run-root", required=True)
    parser.add_argument("--spatial-feature-run-root", default="")
    parser.add_argument("--single-slide-feature-table", default="")
    parser.add_argument("--sample-id", default="TRANSFER_SAMPLE_001")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def label_alignment(score: float, threshold: float) -> str:
    if score >= threshold:
        return "spatial_profile_sensitivity_aligned"
    if score <= -threshold:
        return "spatial_profile_resistance_or_barrier_aligned"
    return "spatial_profile_indeterminate_or_balanced"


def confidence_level(score: float, feature_coverage: float) -> str:
    a = abs(float(score))
    c = float(feature_coverage)
    if c >= 0.90 and a >= 0.50:
        return "high"
    if c >= 0.80 and a >= 0.25:
        return "moderate"
    if c >= 0.60 and a >= 0.15:
        return "low_moderate"
    return "low_or_indeterminate"


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    output_root = Path(args.output_root)
    pim_run_root = Path(args.pim_run_root)

    scaled_path = output_root / "02_aligned_features" / "01_aligned_feature_vectors" / "single_slide_v2_scaled_feature_vector.tsv"
    if not scaled_path.exists():
        raise FileNotFoundError(f"Step 02 scaled feature vector not found: {scaled_path}")

    step_root = output_root / "03_transfer_scores"
    score_dir = step_root / "01_treatment_alignment_scores"
    feature_dir = step_root / "02_feature_contributions"
    theme_dir = step_root / "03_theme_contributions"
    qc_dir = step_root / "04_qc"
    report_dir = step_root / "05_reports"

    for path in [score_dir, feature_dir, theme_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        scaled = read_table(scaled_path)
        effects = load_pim_signed_feature_effects(pim_run_root)
        theme_effects = load_pim_signed_theme_effects(pim_run_root)
        cards = load_pim_treatment_cards(pim_run_root)

        required_cols = {"drug_key", "feature_name", "signed_effect"}
        missing = required_cols - set(effects.columns)
        if missing:
            raise ValueError(f"PIM signed feature effects missing required columns: {missing}")

        effects = effects.copy()
        effects["feature_name"] = effects["feature_name"].astype(str)
        effects["drug_key"] = effects["drug_key"].astype(str)
        effects["signed_effect"] = numeric_series(effects["signed_effect"]).fillna(0.0)
        effects["effect_weight"] = numeric_series(effects["effect_weight"]) if "effect_weight" in effects.columns else effects["signed_effect"].abs()
        effects["effect_weight"] = effects["effect_weight"].fillna(effects["signed_effect"].abs())

        cards_drugs = sorted(cards["drug_key"].dropna().astype(str).unique()) if "drug_key" in cards.columns else sorted(effects["drug_key"].unique())
        effects = effects[effects["drug_key"].isin(cards_drugs)].copy()

        score_rows: List[dict] = []
        contribution_rows: List[dict] = []

        feature_columns = [c for c in scaled.columns if c != "sample_id"]

        for _, sample_row in scaled.iterrows():
            sample_id = str(sample_row["sample_id"])
            sample_values = {feature: float(pd.to_numeric(pd.Series([sample_row.get(feature, 0.0)]), errors="coerce").fillna(0.0).iloc[0]) for feature in feature_columns}

            for drug_key, sub in effects.groupby("drug_key", dropna=False):
                sub = sub.copy()
                present = sub["feature_name"].isin(feature_columns)
                used = sub[present].copy()

                if used.empty:
                    continue

                used["sample_feature_z"] = used["feature_name"].map(sample_values).fillna(0.0).astype(float)
                used["transfer_feature_contribution"] = used["sample_feature_z"] * used["signed_effect"]

                denom = float(used["signed_effect"].abs().sum())
                if denom <= 0 or not np.isfinite(denom):
                    denom = 1.0

                raw_sum = float(used["transfer_feature_contribution"].sum())
                alignment_score = raw_sum / denom
                favorable_score = sigmoid(alignment_score * 2.0)

                coverage = float(len(used) / max(len(sub), 1))
                pos_sum = float(used.loc[used["transfer_feature_contribution"] > 0, "transfer_feature_contribution"].sum())
                neg_sum = float(used.loc[used["transfer_feature_contribution"] < 0, "transfer_feature_contribution"].sum())

                score_rows.append({
                    "sample_id": sample_id,
                    "drug_key": drug_key,
                    "transfer_alignment_score": alignment_score,
                    "transfer_alignment_raw_sum": raw_sum,
                    "spatial_favorable_score_0_1_not_calibrated": favorable_score,
                    "spatial_response_alignment": label_alignment(alignment_score, args.score_threshold),
                    "confidence_level": confidence_level(alignment_score, coverage),
                    "feature_effects_available_for_drug": int(len(sub)),
                    "feature_effects_used_for_scoring": int(len(used)),
                    "feature_effect_coverage_fraction": coverage,
                    "positive_contribution_sum": pos_sum,
                    "negative_contribution_sum": neg_sum,
                    "absolute_contribution_sum": float(used["transfer_feature_contribution"].abs().sum()),
                    "interpretation_caveat": "Research-use spatial response alignment. This is not a calibrated clinical probability and not a treatment recommendation.",
                })

                used["sample_id"] = sample_id
                used["drug_key"] = drug_key
                contribution_rows.extend(used.to_dict("records"))

        scores = pd.DataFrame(score_rows).sort_values(["sample_id", "transfer_alignment_score"], ascending=[True, False])
        contributions = pd.DataFrame(contribution_rows)

        theme_rows: List[dict] = []
        if not contributions.empty:
            if "biological_theme" not in contributions.columns:
                contributions["biological_theme"] = "other interpretable spatial signal"
            for (sample_id, drug_key, theme), sub in contributions.groupby(["sample_id", "drug_key", "biological_theme"], dropna=False):
                contribution_sum = float(pd.to_numeric(sub["transfer_feature_contribution"], errors="coerce").sum())
                theme_rows.append({
                    "sample_id": sample_id,
                    "drug_key": drug_key,
                    "biological_theme": theme,
                    "transfer_theme_contribution_sum": contribution_sum,
                    "transfer_theme_abs_contribution_sum": float(pd.to_numeric(sub["transfer_feature_contribution"], errors="coerce").abs().sum()),
                    "n_features": int(sub["feature_name"].nunique()),
                    "theme_alignment_direction": "sensitivity_supporting" if contribution_sum > 0 else ("resistance_or_barrier_supporting" if contribution_sum < 0 else "balanced"),
                    "top_features_in_theme": summarize_examples(sub.assign(abs_c=pd.to_numeric(sub["transfer_feature_contribution"], errors="coerce").abs()).sort_values("abs_c", ascending=False)["feature_name"], 6),
                })

        themes = pd.DataFrame(theme_rows)
        if not themes.empty:
            themes = themes.sort_values(["sample_id", "drug_key", "transfer_theme_abs_contribution_sum"], ascending=[True, True, False])

        write_tsv(score_dir / "single_slide_treatment_alignment_scores.tsv", scores)
        write_tsv(feature_dir / "single_slide_treatment_feature_contributions.tsv", contributions)
        write_tsv(theme_dir / "single_slide_treatment_theme_contributions.tsv", themes)

        add_qc(qc, "samples_scored", "pass" if scores["sample_id"].nunique() >= 1 else "fail", scores["sample_id"].nunique() if not scores.empty else 0, ">=1", "At least one transfer sample scored.")
        add_qc(qc, "treatments_scored", "pass" if scores["drug_key"].nunique() == 27 else "warn", scores["drug_key"].nunique() if not scores.empty else 0, 27, "Expected 27 PIM validated treatment cards.")
        add_qc(qc, "feature_contribution_rows", "pass" if len(contributions) > 0 else "fail", len(contributions), ">0", "Feature contributions generated.")
        add_qc(qc, "theme_contribution_rows", "pass" if len(themes) > 0 else "fail", len(themes), ">0", "Theme contributions generated.")
        if not scores.empty:
            add_qc(qc, "min_feature_effect_coverage", "pass" if scores["feature_effect_coverage_fraction"].min() >= 0.80 else "warn", f"{scores['feature_effect_coverage_fraction'].min():.3f}", ">=0.80 preferred", "Feature effect coverage for scoring.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        scores = pd.DataFrame()
        contributions = pd.DataFrame()
        themes = pd.DataFrame()

    status = report_status_from_qc(qc, errors)
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step03_transfer_scoring_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "score_rows": len(scores),
        "feature_contribution_rows": len(contributions),
        "theme_contribution_rows": len(themes),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "spatial_transfer_inference_model_step03_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL STEP 03 REPORT",
        "",
        f"status: {status}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"score_threshold: {args.score_threshold}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        str(score_dir / "single_slide_treatment_alignment_scores.tsv"),
        str(feature_dir / "single_slide_treatment_feature_contributions.tsv"),
        str(theme_dir / "single_slide_treatment_theme_contributions.tsv"),
        "",
        "Interpretation",
        "Positive transfer_alignment_score means the slide's V2-scaled spatial profile is aligned with sensitivity-associated residual biology for the treatment.",
        "Negative transfer_alignment_score means the slide's profile is aligned with resistance/barrier-associated residual biology.",
        "The spatial_favorable_score_0_1_not_calibrated column is a monotonic score for ranking, not a calibrated clinical probability.",
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
    write_text_report(report_dir / "step03_score_transfer_drug_response_alignment_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL STEP 03 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"score_rows: {len(scores)}")
    print(f"feature_contribution_rows: {len(contributions)}")
    print(f"theme_contribution_rows: {len(themes)}")
    print(f"report: {report_dir / 'step03_score_transfer_drug_response_alignment_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())