#!/usr/bin/env python
"""
Script:
    00c_audit_transfer_strict_feature_missingness.py

Purpose:
    Audit every frozen PIM/V2 strict spatial feature in a transfer handoff and
    classify why it is observed, recovered, missing, biologically absent-like,
    or unavailable.

Inputs:
    A handoff directory created by:
        00b_build_improved_transfer_handoff.py

Required files:
    model_input_numeric.csv
    transfer_feature_coverage_by_sample_feature.tsv
    single_or_batch_handoff_candidate_source_tables.tsv

Outputs:
    strict_feature_missingness_audit.tsv
    strict_feature_missingness_audit_summary_by_sample.tsv
    strict_feature_missingness_audit_summary_by_category.tsv
    strict_feature_missingness_audit_summary_by_status.tsv
    strict_feature_missingness_audit_report.txt

Interpretation:
    This audit does not change model scores. It improves transparency and
    confidence by explaining exactly which spatial features were observed,
    recovered from secondary tables, absent-like, or unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


PRIMARY_TABLE_MARKERS = [
    "output_09_build_motif_tables/slide_features_with_motif_tables.csv",
    "output_09_build_motif_tables\\slide_features_with_motif_tables.csv",
]


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


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


def feature_category(feature: str) -> str:
    s = str(feature).lower()

    if any(x in s for x in ["access", "penetration", "boundary", "core", "distance"]):
        return "access_boundary_penetration"
    if any(x in s for x in ["stromal", "ecm", "fibroblast", "barrier"]):
        return "stromal_ecm_barrier"
    if any(x in s for x in ["myeloid", "macrophage"]):
        return "myeloid_macrophage"
    if any(x in s for x in ["immune", "t_cell", "interferon", "inflamed", "lymphocyte"]):
        return "immune_tcell_inflammation"
    if any(x in s for x in ["vascular", "angiogenic", "endothelial"]):
        return "vascular_angiogenic"
    if any(x in s for x in ["hypoxia", "stress"]):
        return "hypoxia_stress"
    if any(x in s for x in ["metabolic", "tryptophan", "kynurenine"]):
        return "metabolic_immune_suppression"
    if any(x in s for x in ["hotspot", "motif", "pair", "gradient"]):
        return "spatial_architecture"
    return "other"


def zero_like_if_absent(feature: str) -> bool:
    s = str(feature).lower()

    tokens = [
        "_count",
        "count_",
        "_n_",
        "fraction",
        "_frac",
        "proportion",
        "hotspot__",
        "motif__",
        "component",
        "access_",
        "boundary",
        "pair_",
        "gradient_",
        "largest_component",
        "fragmentation",
    ]

    return any(token in s for token in tokens)


def is_primary_source(source_table: object) -> bool:
    text = str(source_table)
    return any(marker in text for marker in PRIMARY_TABLE_MARKERS)


def classify_row(row: pd.Series) -> str:
    status = str(row.get("transfer_feature_status", ""))
    nonmissing = bool(row.get("nonmissing", False))
    source_table = str(row.get("source_table", ""))
    feature = str(row.get("feature", ""))

    if status.startswith("zero_filled"):
        return "biologically_absent_zero_filled"

    if nonmissing:
        if is_primary_source(source_table):
            return "observed_nonmissing_primary_table"
        return "recovered_from_other_table"

    if status == "observed_column_but_missing_value":
        if zero_like_if_absent(feature):
            return "biologically_absent_zero_like_candidate"
        return "observed_column_but_missing_value"

    if status == "not_found_in_single_or_batch_outputs":
        if zero_like_if_absent(feature):
            return "biologically_absent_zero_like_candidate"
        return "unavailable"

    if zero_like_if_absent(feature):
        return "biologically_absent_zero_like_candidate"

    return "unavailable"


def confidence_impact(row: pd.Series) -> str:
    classification = str(row.get("missingness_classification", ""))
    category = str(row.get("feature_category", ""))

    if classification in ["observed_nonmissing_primary_table", "recovered_from_other_table"]:
        return "supports_confidence"

    if classification in ["biologically_absent_zero_filled", "biologically_absent_zero_like_candidate"]:
        if category in ["access_boundary_penetration", "spatial_architecture", "stromal_ecm_barrier"]:
            return "moderate_confidence_penalty_review_absence"
        return "minor_confidence_penalty_absence_may_be_biological"

    if classification == "observed_column_but_missing_value":
        return "confidence_penalty_missing_value"

    return "confidence_penalty_unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", required=True, help="Directory created by 00b_build_improved_transfer_handoff.py")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to handoff-dir/strict_feature_missingness_audit")
    args = parser.parse_args()

    handoff_dir = Path(args.handoff_dir)
    output_dir = Path(args.output_dir) if args.output_dir else handoff_dir / "strict_feature_missingness_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_input_path = handoff_dir / "model_input_numeric.csv"
    long_path = handoff_dir / "transfer_feature_coverage_by_sample_feature.tsv"
    candidate_path = handoff_dir / "single_or_batch_handoff_candidate_source_tables.tsv"
    sample_coverage_path = handoff_dir / "transfer_feature_coverage_by_sample.tsv"

    for path in [model_input_path, long_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required handoff file missing: {path}")

    model = read_table(model_input_path)
    long = read_table(long_path)
    candidate_tables = read_table(candidate_path) if candidate_path.exists() else pd.DataFrame()
    sample_coverage = read_table(sample_coverage_path) if sample_coverage_path.exists() else pd.DataFrame()

    if "feature_category" not in long.columns:
        long["feature_category"] = long["feature"].astype(str).map(feature_category)

    if "zero_like_if_absent" not in long.columns:
        long["zero_like_if_absent"] = long["feature"].astype(str).map(zero_like_if_absent)

    long["missingness_classification"] = long.apply(classify_row, axis=1)
    long["confidence_impact"] = long.apply(confidence_impact, axis=1)

    # Correct audit math by retaining one row per sample_id + feature.
    # Prefer the most evidence-rich row if duplicates are present.
    classification_priority = {
        "observed_nonmissing_primary_table": 0,
        "recovered_from_other_table": 1,
        "biologically_absent_zero_filled": 2,
        "biologically_absent_zero_like_candidate": 3,
        "observed_column_but_missing_value": 4,
        "unavailable": 5,
    }
    long["_dedup_priority"] = long["missingness_classification"].map(classification_priority).fillna(9).astype(int)
    long = (
        long
        .sort_values(["sample_id", "feature", "_dedup_priority"])
        .drop_duplicates(["sample_id", "feature"], keep="first")
        .drop(columns=["_dedup_priority"])
        .reset_index(drop=True)
    )

    long["is_observed_or_recovered"] = long["missingness_classification"].isin([
        "observed_nonmissing_primary_table",
        "recovered_from_other_table",
        "biologically_absent_zero_filled",
    ])

    long["is_missing_but_biologically_absent_like"] = long["missingness_classification"].isin([
        "biologically_absent_zero_like_candidate",
        "biologically_absent_zero_filled",
    ])

    long["is_unavailable_or_unexplained_missing"] = long["missingness_classification"].isin([
        "unavailable",
        "observed_column_but_missing_value",
    ])

    audit_path = output_dir / "strict_feature_missingness_audit.tsv"
    write_tsv(audit_path, long)

    by_sample = (
        long
        .groupby("sample_id", as_index=False)
        .agg(
            strict_features_total=("feature", "nunique"),
            observed_nonmissing_primary_table=("missingness_classification", lambda s: int((s == "observed_nonmissing_primary_table").sum())),
            recovered_from_other_table=("missingness_classification", lambda s: int((s == "recovered_from_other_table").sum())),
            biologically_absent_zero_like_candidate=("missingness_classification", lambda s: int((s == "biologically_absent_zero_like_candidate").sum())),
            biologically_absent_zero_filled=("missingness_classification", lambda s: int((s == "biologically_absent_zero_filled").sum())),
            observed_column_but_missing_value=("missingness_classification", lambda s: int((s == "observed_column_but_missing_value").sum())),
            unavailable=("missingness_classification", lambda s: int((s == "unavailable").sum())),
            observed_or_recovered_total=("is_observed_or_recovered", "sum"),
            biologically_absent_like_total=("is_missing_but_biologically_absent_like", "sum"),
            unavailable_or_unexplained_total=("is_unavailable_or_unexplained_missing", "sum"),
        )
    )

    by_sample["observed_or_recovered_fraction"] = (
        by_sample["observed_or_recovered_total"] / by_sample["strict_features_total"].clip(lower=1)
    )
    by_sample["biologically_explainable_or_observed_fraction"] = (
        (by_sample["observed_or_recovered_total"] + by_sample["biologically_absent_like_total"])
        / by_sample["strict_features_total"].clip(lower=1)
    )
    by_sample["unavailable_or_unexplained_fraction"] = (
        by_sample["unavailable_or_unexplained_total"] / by_sample["strict_features_total"].clip(lower=1)
    )

    by_sample_path = output_dir / "strict_feature_missingness_audit_summary_by_sample.tsv"
    write_tsv(by_sample_path, by_sample)

    by_category = (
        long
        .groupby(["sample_id", "feature_category"], as_index=False)
        .agg(
            category_features_total=("feature", "nunique"),
            observed_or_recovered_total=("is_observed_or_recovered", "sum"),
            biologically_absent_like_total=("is_missing_but_biologically_absent_like", "sum"),
            unavailable_or_unexplained_total=("is_unavailable_or_unexplained_missing", "sum"),
        )
    )
    by_category["observed_or_recovered_fraction"] = (
        by_category["observed_or_recovered_total"] / by_category["category_features_total"].clip(lower=1)
    )
    by_category["biologically_explainable_or_observed_fraction"] = (
        (by_category["observed_or_recovered_total"] + by_category["biologically_absent_like_total"])
        / by_category["category_features_total"].clip(lower=1)
    )

    by_category_path = output_dir / "strict_feature_missingness_audit_summary_by_category.tsv"
    write_tsv(by_category_path, by_category)

    by_status = (
        long
        .groupby(["sample_id", "missingness_classification"], as_index=False)
        .agg(
            feature_count=("feature", "nunique"),
            examples=("feature", lambda s: "; ".join(list(map(str, s))[:12])),
        )
    )
    by_status_path = output_dir / "strict_feature_missingness_audit_summary_by_status.tsv"
    write_tsv(by_status_path, by_status)

    missing_detail = long[~long["is_observed_or_recovered"]].copy()
    missing_detail_path = output_dir / "strict_features_missing_or_absent_detail.tsv"
    write_tsv(missing_detail_path, missing_detail)

    summary = {
        "status": "pass",
        "handoff_dir": str(handoff_dir),
        "model_input": str(model_input_path),
        "audit": str(audit_path),
        "summary_by_sample": str(by_sample_path),
        "summary_by_category": str(by_category_path),
        "summary_by_status": str(by_status_path),
        "missing_or_absent_detail": str(missing_detail_path),
        "sample_count": int(long["sample_id"].nunique()),
        "feature_count": int(long["feature"].nunique()),
        "sample_summaries": by_sample.to_dict("records"),
    }
    summary_path = output_dir / "strict_feature_missingness_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_path = output_dir / "strict_feature_missingness_audit_report.txt"
    lines = [
        "STRICT FEATURE MISSINGNESS AUDIT REPORT",
        "",
        f"status: pass",
        f"handoff_dir: {handoff_dir}",
        f"model_input: {model_input_path}",
        f"sample_count: {summary['sample_count']}",
        f"strict_feature_count: {summary['feature_count']}",
        "",
        "Sample summary",
        by_sample.to_string(index=False),
        "",
        "Interpretation",
        "observed_nonmissing_primary_table: feature was nonmissing in the preferred final slide-level table.",
        "recovered_from_other_table: feature was missing from the preferred source but recovered from another spatial pipeline table.",
        "biologically_absent_zero_like_candidate: feature is missing but its name suggests that absence may represent a true zero/absence state; review before explicit zero fill.",
        "observed_column_but_missing_value: the feature column existed but the sample value was missing.",
        "unavailable: the feature was not found and is not clearly zero-like.",
        "",
        "Outputs",
        str(audit_path),
        str(by_sample_path),
        str(by_category_path),
        str(by_status_path),
        str(missing_detail_path),
        str(summary_path),
    ]
    write_report(report_path, lines)

    print("")
    print("=" * 72)
    print("STRICT FEATURE MISSINGNESS AUDIT SUMMARY")
    print("=" * 72)
    print(by_sample.to_string(index=False))
    print("")
    print("Wrote:")
    print(audit_path)
    print(by_sample_path)
    print(by_category_path)
    print(by_status_path)
    print(missing_detail_path)
    print(report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())