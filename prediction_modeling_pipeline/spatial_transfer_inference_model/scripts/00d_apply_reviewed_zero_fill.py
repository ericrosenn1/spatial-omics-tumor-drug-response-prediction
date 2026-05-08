#!/usr/bin/env python
"""
Script:
    00d_apply_reviewed_zero_fill.py

Purpose:
    Apply conservative reviewed zero-fill to strict transfer features that are
    missing but plausibly represent biological absence.

Input:
    A handoff directory from 00b and its missingness audit from 00c.

Output:
    A new handoff directory containing:
      - model_input_numeric.csv
      - feature_manifest.csv
      - transfer_feature_coverage_by_sample_feature.tsv
      - transfer_feature_coverage_by_sample.tsv
      - transfer_feature_coverage_by_sample_category.tsv
      - reviewed_zero_fill_decisions.tsv
      - reviewed_zero_fill_report.txt

Policy:
    This script does not change observed values.
    It only fills missing values as 0 when absence is conservative and biologically plausible.
    Risky continuous/distance/slope/score features remain NaN and are handled as neutral z=0 downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


def choose_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in [".tsv", ".tab"] else ","


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=choose_sep(path), low_memory=False)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("FILEPATH: " + str(path) + "\n\n" + "\n".join(lines), encoding="utf-8")


def is_safe_zero_fill_feature(feature: str, category: str) -> tuple[bool, str, str]:
    f = str(feature).lower()
    c = str(category).lower()

    # These features usually represent abundance, fraction, count, presence,
    # component size, or fragmentation. If the structure is absent, zero is
    # often the correct biological value.
    safe_tokens = [
        "fraction",
        "_frac",
        "proportion",
        "_count",
        "count_",
        "component_count",
        "largest_component",
        "fragmentation",
        "hotspot_fraction",
        "hotspot__",
        "motif_fraction",
        "motif__",
    ]

    # These are continuous measurements where missing does not safely mean zero.
    # Examples: distances, slopes, medians, means, q90, scores.
    risky_tokens = [
        "distance",
        "_dist_",
        "slope",
        "median",
        "mean",
        "_max",
        "_min",
        "_q90",
        "_q75",
        "_q25",
        "score",
        "centroid",
        "depth",
        "gradient",
    ]

    if any(t in f for t in risky_tokens):
        return False, "keep_missing_neutral_z0", "continuous_or_distance_like_missing_not_safely_zero"

    if any(t in f for t in safe_tokens):
        return True, "fill_zero", "absence_like_count_fraction_component_or_hotspot_feature"

    if c in ["spatial_architecture", "stromal_ecm_barrier"] and any(t in f for t in ["component", "fragment", "fraction"]):
        return True, "fill_zero", "category_and_name_support_absence_as_zero"

    return False, "keep_missing_neutral_z0", "not_enough_evidence_that_missing_means_zero"


def recompute_summaries(long: pd.DataFrame, output_dir: Path) -> None:
    sample_cov = (
        long
        .groupby(["sample_id", "original_internal_sample_id"], as_index=False)
        .agg(
            strict_features_total=("feature", "nunique"),
            strict_features_nonmissing=("nonmissing", "sum"),
            strict_features_observed_as_columns=("transfer_feature_status", lambda s: int((s != "not_found_in_single_or_batch_outputs").sum())),
            strict_features_zero_filled=("transfer_feature_status", lambda s: int(s.astype(str).str.startswith("zero_filled").sum())),
        )
    )
    sample_cov["strict_feature_nonmissing_fraction"] = (
        sample_cov["strict_features_nonmissing"] / sample_cov["strict_features_total"].clip(lower=1)
    )
    sample_cov["strict_feature_observed_column_fraction"] = (
        sample_cov["strict_features_observed_as_columns"] / sample_cov["strict_features_total"].clip(lower=1)
    )
    write_tsv(output_dir / "transfer_feature_coverage_by_sample.tsv", sample_cov)

    cat_cov = (
        long
        .groupby(["sample_id", "feature_category"], as_index=False)
        .agg(
            category_features_total=("feature", "nunique"),
            category_features_nonmissing=("nonmissing", "sum"),
        )
    )
    cat_cov["category_nonmissing_fraction"] = (
        cat_cov["category_features_nonmissing"] / cat_cov["category_features_total"].clip(lower=1)
    )
    write_tsv(output_dir / "transfer_feature_coverage_by_sample_category.tsv", cat_cov)

    feature_manifest = (
        long
        .groupby("feature", as_index=False)
        .agg(
            missing_fraction=("nonmissing", lambda s: float(1.0 - s.mean())),
            nonmissing_count=("nonmissing", "sum"),
            unique_values=("numeric_value", lambda s: int(pd.Series(s).dropna().nunique())),
            feature_statuses=("transfer_feature_status", lambda s: "; ".join(sorted(set(map(str, s))))),
            source_tables=("source_table", lambda s: "; ".join([x for x in sorted(set(map(str, s))) if x])[:1000]),
            feature_category=("feature_category", "first"),
            zero_like_if_absent=("zero_like_if_absent", "first"),
        )
    )
    feature_manifest["kept"] = True
    feature_manifest["filter_reason"] = "kept_for_transfer_reviewed_zero_fill"
    feature_manifest["std"] = ""
    feature_manifest["feature_group"] = feature_manifest["feature_category"]
    feature_manifest["feature_axis"] = "transfer_strict_spatial_biology"
    feature_manifest["pipeline_stage"] = "reviewed_zero_fill_transfer_handoff"

    feature_manifest = feature_manifest[
        [
            "feature",
            "kept",
            "filter_reason",
            "missing_fraction",
            "nonmissing_count",
            "unique_values",
            "std",
            "feature_group",
            "feature_axis",
            "pipeline_stage",
            "feature_statuses",
            "source_tables",
            "zero_like_if_absent",
        ]
    ]
    feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", required=True)
    parser.add_argument("--audit-dir", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    handoff_dir = Path(args.handoff_dir)
    audit_dir = Path(args.audit_dir) if args.audit_dir else handoff_dir / "strict_feature_missingness_audit"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = handoff_dir / "model_input_numeric.csv"
    long_path = handoff_dir / "transfer_feature_coverage_by_sample_feature.tsv"
    audit_path = audit_dir / "strict_feature_missingness_audit.tsv"

    for path in [model_path, long_path, audit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    model = read_table(model_path)
    long = read_table(long_path)
    audit = read_table(audit_path)

    # Correct duplicates defensively.
    priority = {
        "observed_nonmissing_primary_table": 0,
        "recovered_from_other_table": 1,
        "biologically_absent_zero_filled": 2,
        "biologically_absent_zero_like_candidate": 3,
        "observed_column_but_missing_value": 4,
        "unavailable": 5,
    }
    audit["_priority"] = audit["missingness_classification"].map(priority).fillna(9).astype(int)
    audit = (
        audit
        .sort_values(["sample_id", "feature", "_priority"])
        .drop_duplicates(["sample_id", "feature"], keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    long = (
        long
        .sort_values(["sample_id", "feature"])
        .drop_duplicates(["sample_id", "feature"], keep="first")
        .reset_index(drop=True)
    )

    decision_rows = []
    fill_keys = set()

    for _, row in audit.iterrows():
        feature = str(row["feature"])
        sample_id = str(row["sample_id"])
        category = str(row.get("feature_category", ""))
        classification = str(row.get("missingness_classification", ""))

        fill = False
        decision = "keep_observed_or_recovered"
        reason = "feature_already_observed_or_recovered"

        if classification == "biologically_absent_zero_like_candidate":
            fill, decision, reason = is_safe_zero_fill_feature(feature, category)

        elif classification in ["observed_column_but_missing_value", "unavailable"]:
            fill = False
            decision = "keep_missing_neutral_z0"
            reason = "missing_not_reviewed_as_safe_zero"

        if fill:
            fill_keys.add((sample_id, feature))

        decision_rows.append({
            "sample_id": sample_id,
            "feature": feature,
            "feature_category": category,
            "missingness_classification": classification,
            "reviewed_zero_fill": bool(fill),
            "recommended_fill": 0 if fill else "",
            "decision": decision,
            "reason": reason,
        })

    decisions = pd.DataFrame(decision_rows)
    write_tsv(output_dir / "reviewed_zero_fill_decisions.tsv", decisions)

    # Apply fills to model.
    model_out = model.copy()
    if "sample_id" not in model_out.columns:
        raise ValueError("model_input_numeric.csv missing sample_id column.")

    for sample_id, feature in fill_keys:
        mask = model_out["sample_id"].astype(str) == str(sample_id)
        if feature in model_out.columns:
            model_out.loc[mask, feature] = pd.to_numeric(model_out.loc[mask, feature], errors="coerce").fillna(0.0)

    model_out.to_csv(output_dir / "model_input_numeric.csv", index=False)

    # Apply fills to long.
    long_out = long.copy()
    key_series = list(zip(long_out["sample_id"].astype(str), long_out["feature"].astype(str)))
    fill_mask = pd.Series([k in fill_keys for k in key_series], index=long_out.index)

    long_out.loc[fill_mask, "numeric_value"] = 0.0
    long_out.loc[fill_mask, "nonmissing"] = True
    long_out.loc[fill_mask, "transfer_feature_status"] = "zero_filled_reviewed_biological_absence"
    long_out.loc[fill_mask, "fill_policy"] = "reviewed_zero_fill"

    write_tsv(output_dir / "transfer_feature_coverage_by_sample_feature.tsv", long_out)

    # Copy candidate scan if present.
    candidate = handoff_dir / "single_or_batch_handoff_candidate_source_tables.tsv"
    if candidate.exists():
        candidate_df = read_table(candidate)
        write_tsv(output_dir / "single_or_batch_handoff_candidate_source_tables.tsv", candidate_df)

    recompute_summaries(long_out, output_dir)

    sample_cov = read_table(output_dir / "transfer_feature_coverage_by_sample.tsv")
    n_filled = int(decisions["reviewed_zero_fill"].sum())

    summary = {
        "status": "pass",
        "input_handoff_dir": str(handoff_dir),
        "input_audit_dir": str(audit_dir),
        "output_dir": str(output_dir),
        "reviewed_zero_filled_features": n_filled,
        "sample_coverage": sample_cov.to_dict("records"),
    }
    (output_dir / "reviewed_zero_fill_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = output_dir / "reviewed_zero_fill_report.txt"
    lines = [
        "REVIEWED ZERO-FILL REPORT",
        "",
        "status: pass",
        f"input_handoff_dir: {handoff_dir}",
        f"input_audit_dir: {audit_dir}",
        f"output_dir: {output_dir}",
        f"reviewed_zero_filled_features: {n_filled}",
        "",
        "Sample coverage after reviewed zero-fill",
        sample_cov.to_string(index=False),
        "",
        "Policy",
        "Only conservative absence-like count/fraction/component/hotspot/motif features were zero-filled.",
        "Continuous distance/slope/score/mean/median/q90-like features were not zero-filled.",
        "Observed values were never overwritten.",
        "",
        "Outputs",
        str(output_dir / "model_input_numeric.csv"),
        str(output_dir / "transfer_feature_coverage_by_sample_feature.tsv"),
        str(output_dir / "transfer_feature_coverage_by_sample.tsv"),
        str(output_dir / "reviewed_zero_fill_decisions.tsv"),
        str(output_dir / "reviewed_zero_fill_summary.json"),
    ]
    write_report(report, lines)

    print("")
    print("=" * 72)
    print("REVIEWED ZERO-FILL SUMMARY")
    print("=" * 72)
    print(f"reviewed_zero_filled_features: {n_filled}")
    print(sample_cov.to_string(index=False))
    print("")
    print("Output:")
    print(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())