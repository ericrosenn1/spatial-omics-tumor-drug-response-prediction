#!/usr/bin/env python
"""
Script:
    05_build_sample_level_interpretations.py

Description:
    Builds sample-treatment spatial interpretation scores by applying signed
    treatment feature effects to sample spatial feature profiles. Scores describe
    sensitivity-aligned versus resistance-aligned spatial biology.

Instructions:
    Run after Step 04. Interpret coverage through validated-treatment pair rows:
    not every V2 sample necessarily has eligible rows for every validated treatment.

Source-truth policy:
    Sample-level outputs are interpretation summaries of model-derived associations,
    not treatment recommendations or causal claims.
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
from typing import Dict, List

import numpy as np
import pandas as pd

from _pim_utils import (
    add_qc,
    choose_col,
    ensure_dir,
    load_prepared_index,
    numeric_series,
    open_folder,
    read_header,
    read_source_table,
    read_table,
    save_output_manifest,
    source_path,
    summarize_examples,
    write_json,
    write_text_report,
    write_tsv,
    zscore_frame,
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
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--v2-run-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-input-root", default="")
    parser.add_argument("--target-col", default="fused_residual_vs_prior")
    parser.add_argument("--top-n-features-per-treatment", type=int, default=20)
    parser.add_argument("--top-n-driver-features", type=int, default=3)
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def read_pair_minimal(index_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Read only required pair-level columns from the large V2 table.
    Keeps signed-effect and sample-level steps memory-conscious."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    pair_path = source_path(index_df, "v2_pair_level_residual_dataset", prefer_copied=False)
    columns = read_header(pair_path)

    sample_col = choose_col(columns, ["sample_id", "slide_id", "sample"], required=True, label="sample column")
    treatment_col = choose_col(columns, ["drug_key", "treatment_key", "drug", "treatment"], required=True, label="treatment column")

    optional = [
        target_col,
        "fused_prob_responder",
        "treatment_prior",
        "prior_prob_responder",
        "fused_confidence",
        "label_quality_flag",
        "modality_used",
    ]
    usecols = [sample_col, treatment_col] + [c for c in optional if c in columns]
    if target_col not in usecols:
        raise ValueError(f"Pair-level dataset does not contain required target column: {target_col}")

    pair = read_table(pair_path, usecols=usecols)
    if sample_col != "sample_id":
        pair = pair.rename(columns={sample_col: "sample_id"})
    if treatment_col != "drug_key":
        pair = pair.rename(columns={treatment_col: "drug_key"})

    return pair


def load_spatial_z(index_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Load and z-score spatial features for sample scoring.
    Z-scores are computed across available samples in the spatial feature table."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    spatial = read_source_table(index_df, "v2_spatial_features_broad_pool")
    sample_col = choose_col(spatial.columns, ["sample_id", "slide_id", "sample"], required=True, label="spatial sample column")
    if sample_col != "sample_id":
        spatial = spatial.rename(columns={sample_col: "sample_id"})
    present = [f for f in features if f in spatial.columns]
    spatial = spatial[["sample_id"] + present].copy()
    spatial_z = zscore_frame(spatial, present)
    return spatial_z


def driver_text(rows: pd.DataFrame, direction: str, n: int) -> str:
    """Create readable driver-feature text for sample scores.
    Reports the strongest positive or negative feature contributions."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if rows.empty:
        return "none"
    if direction == "positive":
        part = rows[pd.to_numeric(rows["contribution"], errors="coerce") > 0].copy()
        part = part.sort_values("contribution", ascending=False).head(n)
        label = "supports sensitivity-aligned score"
    else:
        part = rows[pd.to_numeric(rows["contribution"], errors="coerce") < 0].copy()
        part["abs_contribution"] = pd.to_numeric(part["contribution"], errors="coerce").abs()
        part = part.sort_values("abs_contribution", ascending=False).head(n)
        label = "supports resistance-aligned score"

    pieces = []
    for _, row in part.iterrows():
        pieces.append(f"{row.get('feature_name', '')} ({label}; contribution={float(row.get('contribution', 0.0)):.4f})")
    return "; ".join(pieces) if pieces else "none"


def build_scores_for_treatment(
    drug_key: str,
    pair_sub: pd.DataFrame,
    spatial_z: pd.DataFrame,
    effects: pd.DataFrame,
    target_col: str,
    top_n_driver_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sample-level scores for one validated treatment.
    Returns score rows and top feature-contribution rows."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    if pair_sub.empty or effects.empty:
        return pd.DataFrame(), pd.DataFrame()

    effects = effects.copy()
    effects["effect_weight"] = pd.to_numeric(effects["effect_weight"], errors="coerce").fillna(0.0)
    effects = effects.sort_values("effect_weight", ascending=False)

    features = [f for f in effects["feature_name"].astype(str).tolist() if f in spatial_z.columns]
    if not features:
        return pd.DataFrame(), pd.DataFrame()

    data = pair_sub.merge(spatial_z[["sample_id"] + features], on="sample_id", how="left")
    weights = effects.set_index("feature_name")["signed_effect"].astype(float).reindex(features).fillna(0.0)
    denominator = float(weights.abs().sum())
    if not np.isfinite(denominator) or denominator <= 0:
        denominator = 1.0

    X = data[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    contrib = X.mul(weights, axis=1)

    net = contrib.sum(axis=1) / denominator
    sens = contrib.clip(lower=0.0).sum(axis=1) / denominator
    resist = (-contrib.clip(upper=0.0)).sum(axis=1) / denominator

    score_rows: List[dict] = []
    contribution_rows: List[dict] = []

    effect_meta_cols = [
        "feature_name",
        "feature_label",
        "feature_group",
        "biological_theme",
        "direction_label",
        "feature_target_pearson",
        "signed_effect",
        "effect_weight",
        "evidence_grade",
    ]
    effect_meta = effects[[c for c in effect_meta_cols if c in effects.columns]].drop_duplicates("feature_name").set_index("feature_name")

    for idx, row in data.iterrows():
        sample_id = str(row["sample_id"])
        row_contrib = contrib.loc[idx]
        cdf = pd.DataFrame({
            "feature_name": features,
            "sample_spatial_zscore": X.loc[idx, features].values,
            "signed_effect": weights.values,
            "contribution": row_contrib.values,
        })
        cdf = cdf.merge(effect_meta.reset_index(), on="feature_name", how="left", suffixes=("", "_effect"))

        top_pos = cdf[pd.to_numeric(cdf["contribution"], errors="coerce") > 0].sort_values("contribution", ascending=False).head(top_n_driver_features)
        top_neg = cdf[pd.to_numeric(cdf["contribution"], errors="coerce") < 0].assign(
            abs_contribution=lambda d: pd.to_numeric(d["contribution"], errors="coerce").abs()
        ).sort_values("abs_contribution", ascending=False).head(top_n_driver_features)

        for contribution_direction, part in [("sensitivity_supporting", top_pos), ("resistance_supporting", top_neg)]:
            for rank, crow in enumerate(part.to_dict("records"), start=1):
                contribution_rows.append({
                    "sample_id": sample_id,
                    "drug_key": drug_key,
                    "contribution_direction": contribution_direction,
                    "rank_within_sample_treatment_direction": rank,
                    **crow,
                })

        net_score = float(net.loc[idx])
        if net_score > 0:
            label = "spatial_profile_sensitivity_aligned"
        elif net_score < 0:
            label = "spatial_profile_resistance_aligned"
        else:
            label = "spatial_profile_balanced_or_ambiguous"

        target_value = row.get(target_col, np.nan)
        try:
            target_float = float(target_value)
        except Exception:
            target_float = np.nan

        if np.isfinite(target_float):
            observed_label = "observed_above_prior_residual" if target_float > 0 else ("observed_below_prior_residual" if target_float < 0 else "observed_zero_residual")
        else:
            observed_label = "observed_residual_missing"

        score_rows.append({
            "sample_id": sample_id,
            "drug_key": drug_key,
            "n_features_used": len(features),
            "net_signed_spatial_interpretation_score": net_score,
            "sensitivity_alignment_score": float(sens.loc[idx]),
            "resistance_alignment_score": float(resist.loc[idx]),
            "sample_spatial_alignment_label": label,
            target_col: target_value,
            "observed_residual_label": observed_label,
            "fused_prob_responder": row.get("fused_prob_responder", np.nan),
            "treatment_prior": row.get("treatment_prior", row.get("prior_prob_responder", np.nan)),
            "fused_confidence": row.get("fused_confidence", np.nan),
            "top_sensitivity_supporting_features": driver_text(top_pos, "positive", top_n_driver_features),
            "top_resistance_supporting_features": driver_text(top_neg, "negative", top_n_driver_features),
            "interpretation_caveat": "Spatial alignment score is an interpretation of signed model associations, not a treatment recommendation.",
        })

    return pd.DataFrame(score_rows), pd.DataFrame(contribution_rows)


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

    prepared_root, index_df = load_prepared_index(output_root, Path(args.prepared_input_root) if args.prepared_input_root else None)

    step03_root = output_root / "03_signed_spatial_effects"
    feature_effect_path = step03_root / "01_treatment_feature_effects" / "signed_treatment_feature_effects.tsv"

    step04_root = output_root / "04_treatment_interpretation_cards"
    card_path = step04_root / "02_cards_tsv" / "treatment_interpretation_cards.tsv"

    if not feature_effect_path.exists():
        raise FileNotFoundError(f"Step 03 signed feature effects not found: {feature_effect_path}")
    if not card_path.exists():
        raise FileNotFoundError(f"Step 04 treatment card table not found: {card_path}")

    effects = read_table(feature_effect_path)
    cards = read_table(card_path)

    step_root = output_root / "05_sample_level_interpretations"
    scores_dir = step_root / "01_sample_treatment_scores"
    contrib_dir = step_root / "02_feature_contributions"
    sample_dir = step_root / "03_sample_summaries"
    ranking_dir = step_root / "04_treatment_sample_rankings"
    qc_dir = step_root / "05_qc"
    report_dir = step_root / "06_reports"

    for path in [scores_dir, contrib_dir, sample_dir, ranking_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        valid_drugs = sorted(cards["drug_key"].dropna().astype(str).unique())
        effects = effects[effects["drug_key"].astype(str).isin(valid_drugs)].copy()

        effects["effect_weight"] = pd.to_numeric(effects["effect_weight"], errors="coerce").fillna(0.0)
        top_effects = (
            effects.sort_values(["drug_key", "effect_weight"], ascending=[True, False])
            .groupby("drug_key", as_index=False)
            .head(args.top_n_features_per_treatment)
            .copy()
        )

        all_features = sorted(top_effects["feature_name"].dropna().astype(str).unique())
        pair = read_pair_minimal(index_df, args.target_col)
        pair = pair[pair["drug_key"].astype(str).isin(valid_drugs)].copy()

        spatial_z = load_spatial_z(index_df, all_features)

        score_parts: List[pd.DataFrame] = []
        contribution_parts: List[pd.DataFrame] = []

        for drug_key, drug_effects in top_effects.groupby("drug_key"):
            pair_sub = pair[pair["drug_key"].astype(str) == str(drug_key)].copy()
            scores, contribs = build_scores_for_treatment(
                drug_key=str(drug_key),
                pair_sub=pair_sub,
                spatial_z=spatial_z,
                effects=drug_effects,
                target_col=args.target_col,
                top_n_driver_features=args.top_n_driver_features,
            )
            if not scores.empty:
                score_parts.append(scores)
            if not contribs.empty:
                contribution_parts.append(contribs)

        score_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
        contribution_df = pd.concat(contribution_parts, ignore_index=True) if contribution_parts else pd.DataFrame()

        if not score_df.empty:
            score_df["rank_within_treatment_sensitivity_alignment"] = (
                score_df.groupby("drug_key")["net_signed_spatial_interpretation_score"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            score_df["rank_within_treatment_resistance_alignment"] = (
                score_df.groupby("drug_key")["net_signed_spatial_interpretation_score"]
                .rank(method="first", ascending=True)
                .astype(int)
            )

            card_keep = [c for c in [
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
            score_df = score_df.merge(cards[card_keep].drop_duplicates("drug_key"), on="drug_key", how="left")

        write_tsv(scores_dir / "sample_treatment_signed_interpretation_scores.tsv", score_df)
        write_tsv(contrib_dir / "sample_treatment_top_feature_contributions.tsv", contribution_df)

        if not score_df.empty:
            sample_summary = (
                score_df.groupby("sample_id", as_index=False)
                .agg(
                    n_validated_treatments_scored=("drug_key", "nunique"),
                    mean_net_signed_spatial_score=("net_signed_spatial_interpretation_score", "mean"),
                    median_net_signed_spatial_score=("net_signed_spatial_interpretation_score", "median"),
                    max_sensitivity_alignment_score=("sensitivity_alignment_score", "max"),
                    max_resistance_alignment_score=("resistance_alignment_score", "max"),
                    strongest_sensitivity_aligned_treatment=("drug_key", lambda s: score_df.loc[s.index].sort_values("net_signed_spatial_interpretation_score", ascending=False)["drug_key"].iloc[0]),
                    strongest_resistance_aligned_treatment=("drug_key", lambda s: score_df.loc[s.index].sort_values("net_signed_spatial_interpretation_score", ascending=True)["drug_key"].iloc[0]),
                )
            )
            sample_summary["interpretation_caveat"] = "Sample summary ranks spatial alignment patterns only; it is not a treatment recommendation."

            ranking_rows: List[pd.DataFrame] = []
            for drug_key, sub in score_df.groupby("drug_key"):
                top_sens = sub.sort_values("net_signed_spatial_interpretation_score", ascending=False).head(10).copy()
                top_sens["ranking_direction"] = "most_sensitivity_aligned_spatial_profiles"
                top_res = sub.sort_values("net_signed_spatial_interpretation_score", ascending=True).head(10).copy()
                top_res["ranking_direction"] = "most_resistance_aligned_spatial_profiles"
                ranking_rows.extend([top_sens, top_res])
            ranking_df = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()

        else:
            sample_summary = pd.DataFrame()
            ranking_df = pd.DataFrame()

        write_tsv(sample_dir / "sample_interpretation_summary.tsv", sample_summary)
        write_tsv(ranking_dir / "treatment_sample_interpretation_rankings.tsv", ranking_df)

        add_qc(qc, "validated_treatments_scored", "pass" if score_df.get("drug_key", pd.Series(dtype=str)).nunique() == 27 else "warn", score_df.get("drug_key", pd.Series(dtype=str)).nunique(), 27, "Expected sample-level scores for 27 validated treatments.")
        add_qc(qc, "sample_treatment_score_rows", "pass" if len(score_df) > 0 else "fail", len(score_df), ">0", "Sample-treatment interpretation score rows generated.")
        add_qc(qc, "unique_samples_scored", "pass" if score_df.get("sample_id", pd.Series(dtype=str)).nunique() == 102 else "warn", score_df.get("sample_id", pd.Series(dtype=str)).nunique(), 102, "Expected scoring coverage across available samples.")
        add_qc(qc, "feature_contribution_rows", "pass" if len(contribution_df) > 0 else "fail", len(contribution_df), ">0", "Top driver feature contributions generated.")
        add_qc(qc, "sample_summary_rows", "pass" if len(sample_summary) > 0 else "fail", len(sample_summary), ">0", "Sample-level summary generated.")
        add_qc(qc, "ranking_rows", "pass" if len(ranking_df) > 0 else "fail", len(ranking_df), ">0", "Treatment-specific sample rankings generated.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        score_df = pd.DataFrame()
        contribution_df = pd.DataFrame()
        sample_summary = pd.DataFrame()
        ranking_df = pd.DataFrame()

    status = "pass" if not errors and not any(row["status"] == "fail" for row in qc) else "fail"
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step05_sample_level_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "prepared_input_root": str(prepared_root),
        "sample_treatment_score_rows": len(score_df),
        "feature_contribution_rows": len(contribution_df),
        "sample_summary_rows": len(sample_summary),
        "ranking_rows": len(ranking_df),
        "top_n_features_per_treatment": args.top_n_features_per_treatment,
        "top_n_driver_features": args.top_n_driver_features,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "prediction_interpretation_model_step05_summary.json", summary)

    report_lines = [
        "PREDICTION INTERPRETATION MODEL STEP 05 REPORT",
        "",
        f"status: {status}",
        f"target_col: {args.target_col}",
        f"prepared_input_root: {prepared_root}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Interpretation rule",
        "A positive net signed spatial interpretation score means the sample's feature profile aligns with sensitivity-associated residual biology for that treatment.",
        "A negative score means the sample's feature profile aligns with resistance-associated residual biology for that treatment.",
        "",
        "Outputs",
        f"sample_treatment_scores: {scores_dir / 'sample_treatment_signed_interpretation_scores.tsv'}",
        f"top_feature_contributions: {contrib_dir / 'sample_treatment_top_feature_contributions.tsv'}",
        f"sample_summary: {sample_dir / 'sample_interpretation_summary.tsv'}",
        f"treatment_sample_rankings: {ranking_dir / 'treatment_sample_interpretation_rankings.tsv'}",
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Caveat",
        "These scores are interpretation-layer summaries of model-derived associations. They are not clinical recommendations and should not be used to choose treatment.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step05_sample_level_interpretations_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL STEP 05 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"step_root: {step_root}")
    print(f"sample_treatment_score_rows: {len(score_df)}")
    print(f"feature_contribution_rows: {len(contribution_df)}")
    print(f"sample_summary_rows: {len(sample_summary)}")
    print(f"ranking_rows: {len(ranking_df)}")
    print(f"report: {report_dir / 'step05_sample_level_interpretations_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())

