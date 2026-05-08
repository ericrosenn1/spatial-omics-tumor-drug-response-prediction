#!/usr/bin/env python
"""
Script:
    00b_build_improved_transfer_handoff.py

Purpose:
    Build a transfer-ready model_input_numeric.csv from one or more samples that
    have already been processed through spatial_feature_identification_pipeline.

Why this exists:
    The canonical spatial_feature_identification_pipeline Step 10 is a cohort
    model-table builder. It applies cohort-style filtering, including variance
    and missingness filters, which are not appropriate for one-sample or very
    small transfer batches. This script replaces that behavior for transfer
    inference.

What it does:
    - Accepts a spatial feature pipeline output root containing one or more rows.
    - Harvests the frozen PIM/V2 strict spatial features from all slide-level
      output tables, preferring later/more complete pipeline outputs.
    - Preserves real numeric zero values.
    - Preserves unavailable values as NaN so transfer Step 02 can scale them as
      neutral z=0 while retaining missingness provenance.
    - Writes per-sample and per-feature coverage audits.
    - Supports an optional sample ID map so internal SAMPLE_0000-style IDs can
      be renamed to biologically meaningful slide IDs.

Research-use only:
    This handoff prepares spatial features for research transfer inference. It
    does not make clinical treatment recommendations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PREFERRED_SLIDE_LEVEL_TABLES = [
    "output_09_build_motif_tables/slide_features_with_motif_tables.csv",
    "output_08_context_alignment_and_metabolic_concordance/slide_features_with_metabolic_concordance.csv",
    "output_07_append_hotspot_metrics/slide_features_with_hotspot_metrics.csv",
    "output_06_build_accessibility_profiles/slide_features_with_accessibility.csv",
    "output_05_build_multi_axis_transcriptome_labels/slide_features_with_multi_axis_labels.csv",
    "output_04_score_and_label_slides/slide_features_scored_labeled.csv",
    "output_03_merge_slide_features/merged_slide_features.csv",
]

SKIP_PATH_TOKENS = [
    "__pycache__",
    "_repo_local_archive",
    "output_11_overlay",
    "output_12_data_analysis_and_visuals",
    "output_13_external_study_validation",
    "_external_study_validation",
    "per_sample_h5ad",
]


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def choose_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in [".tsv", ".tab"] else ","


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=choose_sep(path), nrows=nrows, low_memory=False)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("FILEPATH: " + str(path) + "\n\n" + "\n".join(lines), encoding="utf-8")


def find_col(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    norm_to_original = {norm(c): str(c) for c in columns}
    for alias in aliases:
        key = norm(alias)
        if key in norm_to_original:
            return norm_to_original[key]
    return None


def sample_col(columns: Iterable[str]) -> Optional[str]:
    return find_col(
        columns,
        [
            "sample_id",
            "slide_id",
            "sample",
            "library_id",
            "visium_sample_id",
            "source_sample_id",
        ],
    )


def load_strict_features(pim_run_root: Path, strict_feature_dict: Optional[Path]) -> Tuple[List[str], Path, pd.DataFrame]:
    candidates: List[Path] = []

    if strict_feature_dict:
        candidates.append(strict_feature_dict)

    candidates.extend([
        pim_run_root / "02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv",
        pim_run_root / "07_final_outputs/01_publication_tables_tsv/Final_Feature_Dictionary.tsv",
        pim_run_root / "01_prepared_inputs/02_copied_v2_tables/tables/v2_strict_biology_feature_registry.tsv",
    ])

    for path in candidates:
        if not path.exists():
            continue

        df = read_table(path)
        feature_col = find_col(
            df.columns,
            [
                "feature_name",
                "feature",
                "spatial_feature",
                "spatial_feature_name",
                "model_feature",
                "feature_id",
                "variable",
            ],
        )

        if feature_col is None:
            continue

        features = (
            df[feature_col]
            .dropna()
            .astype(str)
            .loc[lambda s: s.str.lower() != "nan"]
            .drop_duplicates()
            .tolist()
        )

        if features:
            if feature_col != "feature_name":
                df = df.rename(columns={feature_col: "feature_name"})
            df["feature_name"] = df["feature_name"].astype(str)
            return sorted(features), path, df

    raise FileNotFoundError("Could not load PIM/V2 strict feature dictionary.")


def load_sample_id_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}

    df = read_table(path)
    internal_col = find_col(
        df.columns,
        ["internal_sample_id", "source_sample_id", "original_sample_id", "sample_id", "slide_id"],
    )
    transfer_col = find_col(
        df.columns,
        ["transfer_sample_id", "output_sample_id", "display_sample_id", "new_sample_id"],
    )

    if internal_col is None or transfer_col is None:
        raise ValueError(
            f"Sample ID map must contain internal/source and transfer/output columns. Columns: {list(df.columns)}"
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        internal = str(row[internal_col])
        transfer = str(row[transfer_col])
        if internal and internal.lower() != "nan" and transfer and transfer.lower() != "nan":
            mapping[internal] = transfer

    return mapping


def candidate_tables(spatial_output_root: Path) -> List[Path]:
    out: List[Path] = []
    seen = set()

    for rel in PREFERRED_SLIDE_LEVEL_TABLES:
        path = spatial_output_root / rel
        if path.exists() and path.is_file():
            out.append(path)
            seen.add(str(path).lower())

    for path in spatial_output_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".csv", ".tsv", ".tab"]:
            continue

        low = str(path).lower()
        if any(tok.lower() in low for tok in SKIP_PATH_TOKENS):
            continue

        try:
            if path.stat().st_size > 100 * 1024 * 1024:
                continue
        except OSError:
            continue

        if str(path).lower() not in seen:
            out.append(path)
            seen.add(str(path).lower())

    return out


def infer_sample_ids_from_tables(paths: List[Path]) -> List[str]:
    sample_ids: List[str] = []
    seen = set()

    for path in paths:
        try:
            df = read_table(path, nrows=10000)
        except Exception:
            continue

        if df.empty:
            continue

        col = sample_col(df.columns)

        if col is not None:
            vals = df[col].dropna().astype(str).tolist()
        elif len(df) == 1:
            vals = ["SAMPLE_0000"]
        else:
            vals = []

        for value in vals:
            value = str(value)
            if value and value.lower() != "nan" and value not in seen:
                sample_ids.append(value)
                seen.add(value)

    return sample_ids


def feature_category(feature: str, theme: str = "") -> str:
    s = (str(feature) + " " + str(theme)).lower()

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
    f = feature.lower()
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
    ]
    return any(tok in f for tok in tokens)


def row_for_sample(df: pd.DataFrame, sample_id: str, path: Path) -> Optional[pd.Series]:
    col = sample_col(df.columns)

    if col is not None:
        hit = df[df[col].astype(str) == str(sample_id)]
        if not hit.empty:
            return hit.iloc[0]
        return None

    # If a table has only one row and no sample column, it can only be safely used
    # for a one-sample output root.
    if len(df) == 1:
        return df.iloc[0]

    return None


def build_handoff(
    spatial_output_root: Path,
    pim_run_root: Path,
    out_dir: Path,
    strict_feature_dict: Optional[Path] = None,
    sample_id_map_path: Optional[Path] = None,
    zero_fill_absent_zero_like: bool = False,
) -> Dict[str, object]:
    started = now()
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_features, strict_source, strict_df = load_strict_features(pim_run_root, strict_feature_dict)
    strict_norm_to_feature = {norm(f): f for f in strict_features}

    sample_map = load_sample_id_map(sample_id_map_path)
    paths = candidate_tables(spatial_output_root)

    if not paths:
        raise FileNotFoundError(f"No candidate slide-level tables found under: {spatial_output_root}")

    raw_sample_ids = infer_sample_ids_from_tables(paths)
    if not raw_sample_ids:
        raise ValueError("Could not infer any sample IDs from candidate tables.")

    # Deduplicate by final transfer sample ID. This prevents the same slide from
    # being emitted twice when both an internal SAMPLE_#### ID and an already
    # renamed/display sample ID are discovered in different source tables.
    #
    # Prefer explicitly mapped internal IDs over incidental literal display IDs.
    sample_ids = []
    seen_transfer_ids = set()
    ordered_raw_sample_ids = sorted(raw_sample_ids, key=lambda x: 0 if x in sample_map else 1)

    for internal_id in ordered_raw_sample_ids:
        transfer_id = sample_map.get(internal_id, internal_id)
        if transfer_id in seen_transfer_ids:
            continue
        sample_ids.append(internal_id)
        seen_transfer_ids.add(transfer_id)

    # If only one sample is present and the caller supplied a mapping from SAMPLE_0000,
    # preserve that mapping. Otherwise the internal ID is used as output ID.
    candidate_scan_rows = []
    loaded_tables: Dict[str, pd.DataFrame] = {}

    for path in paths:
        try:
            df = read_table(path)
            loaded_tables[str(path)] = df

            exact = [c for c in df.columns if str(c) in strict_features]
            normalized = [
                c for c in df.columns
                if norm(c) in strict_norm_to_feature
            ]

            candidate_scan_rows.append({
                "path": str(path),
                "status": "read_ok",
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "sample_column": sample_col(df.columns) or "",
                "strict_exact_overlap": len(set(exact)),
                "strict_normalized_overlap": len(set(normalized)),
                "example_exact_features": "; ".join(exact[:10]),
                "example_normalized_features": "; ".join(normalized[:10]),
            })

        except Exception as exc:
            candidate_scan_rows.append({
                "path": str(path),
                "status": "read_error",
                "error": str(exc),
                "n_rows": "",
                "n_columns": "",
                "sample_column": "",
                "strict_exact_overlap": 0,
                "strict_normalized_overlap": 0,
                "example_exact_features": "",
                "example_normalized_features": "",
            })

    write_tsv(out_dir / "single_or_batch_handoff_candidate_source_tables.tsv", pd.DataFrame(candidate_scan_rows))

    # Build feature values by sample and strict feature. For each sample-feature,
    # prefer the first nonmissing value from the preferred table order. If a column
    # is found but all candidate values are missing, record observed_missing.
    output_rows = []
    long_rows = []

    for internal_sample_id in sample_ids:
        transfer_sample_id = sample_map.get(internal_sample_id, internal_sample_id)

        out_row: Dict[str, object] = {
            "sample_id": transfer_sample_id,
            "original_internal_sample_id": internal_sample_id,
            "source_spatial_output_root": str(spatial_output_root),
            "transfer_handoff_mode": "improved_single_or_batch_transfer_handoff",
        }

        for feature in strict_features:
            chosen_value = np.nan
            chosen_source = ""
            chosen_column = ""
            chosen_match_type = ""
            status = "not_found_in_single_or_batch_outputs"
            raw_value = ""

            for path in paths:
                df = loaded_tables.get(str(path))
                if df is None or df.empty:
                    continue

                row = row_for_sample(df, internal_sample_id, path)
                if row is None:
                    continue

                source_col = None
                match_type = ""

                if feature in df.columns:
                    source_col = feature
                    match_type = "exact"
                else:
                    # normalized match
                    for col in df.columns:
                        if norm(col) == norm(feature):
                            source_col = col
                            match_type = "normalized"
                            break

                if source_col is None:
                    continue

                value = pd.to_numeric(pd.Series([row[source_col]]), errors="coerce").iloc[0]
                raw = row[source_col]

                if pd.notna(value):
                    chosen_value = value
                    chosen_source = str(path)
                    chosen_column = str(source_col)
                    chosen_match_type = match_type
                    raw_value = raw
                    status = "observed_nonmissing"
                    break

                if status == "not_found_in_single_or_batch_outputs":
                    chosen_value = np.nan
                    chosen_source = str(path)
                    chosen_column = str(source_col)
                    chosen_match_type = match_type
                    raw_value = raw
                    status = "observed_column_but_missing_value"

            if pd.isna(chosen_value) and zero_fill_absent_zero_like and zero_like_if_absent(feature):
                chosen_value = 0.0
                if status == "not_found_in_single_or_batch_outputs":
                    status = "zero_filled_absent_zero_like_feature"
                else:
                    status = "zero_filled_missing_zero_like_feature"

            out_row[feature] = chosen_value

            long_rows.append({
                "sample_id": transfer_sample_id,
                "original_internal_sample_id": internal_sample_id,
                "feature": feature,
                "numeric_value": chosen_value,
                "nonmissing": bool(pd.notna(chosen_value)),
                "transfer_feature_status": status,
                "fill_policy": (
                    "observed"
                    if status == "observed_nonmissing"
                    else (
                        "explicit_zero_fill"
                        if status.startswith("zero_filled")
                        else "neutral_z0_in_transfer_step02"
                    )
                ),
                "zero_like_if_absent": zero_like_if_absent(feature),
                "feature_category": feature_category(feature),
                "source_table": chosen_source,
                "source_column": chosen_column,
                "match_type": chosen_match_type,
                "raw_value": raw_value,
            })

        output_rows.append(out_row)

    model_input = pd.DataFrame(output_rows)
    long = pd.DataFrame(long_rows)

    model_input_path = out_dir / "model_input_numeric.csv"
    long_path = out_dir / "transfer_feature_coverage_by_sample_feature.tsv"
    sample_coverage_path = out_dir / "transfer_feature_coverage_by_sample.tsv"
    feature_manifest_path = out_dir / "feature_manifest.csv"
    category_coverage_path = out_dir / "transfer_feature_coverage_by_sample_category.tsv"

    model_input.to_csv(model_input_path, index=False)
    write_tsv(long_path, long)

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
    write_tsv(sample_coverage_path, sample_cov)

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
    write_tsv(category_coverage_path, cat_cov)

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
    feature_manifest["filter_reason"] = "kept_for_transfer_no_cohort_variance_filter"
    feature_manifest["std"] = [
        float(pd.to_numeric(model_input[f], errors="coerce").std(ddof=0))
        if pd.to_numeric(model_input[f], errors="coerce").notna().any()
        else ""
        for f in feature_manifest["feature"]
    ]
    feature_manifest["feature_group"] = feature_manifest["feature_category"]
    feature_manifest["feature_axis"] = "transfer_strict_spatial_biology"
    feature_manifest["pipeline_stage"] = "improved_transfer_handoff"

    # Reorder to resemble the canonical feature_manifest as much as possible.
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
    feature_manifest.to_csv(feature_manifest_path, index=False)

    summary = {
        "status": "pass",
        "started": started,
        "finished": now(),
        "spatial_output_root": str(spatial_output_root),
        "pim_run_root": str(pim_run_root),
        "strict_feature_source": str(strict_source),
        "strict_feature_count": len(strict_features),
        "samples": sample_cov.to_dict("records"),
        "model_input_numeric": str(model_input_path),
        "feature_manifest": str(feature_manifest_path),
        "sample_coverage": str(sample_coverage_path),
        "sample_feature_coverage_long": str(long_path),
        "category_coverage": str(category_coverage_path),
        "zero_fill_absent_zero_like": bool(zero_fill_absent_zero_like),
    }
    write_json(out_dir / "improved_transfer_handoff_summary.json", summary)

    report_path = out_dir / "improved_transfer_handoff_report.txt"
    report_lines = [
        "IMPROVED SINGLE/BATCH TRANSFER HANDOFF REPORT",
        "",
        f"status: pass",
        f"spatial_output_root: {spatial_output_root}",
        f"pim_run_root: {pim_run_root}",
        f"strict_feature_source: {strict_source}",
        f"strict_feature_count: {len(strict_features)}",
        f"sample_count: {len(sample_cov)}",
        f"zero_fill_absent_zero_like: {zero_fill_absent_zero_like}",
        "",
        "Sample coverage",
        sample_cov.to_string(index=False),
        "",
        "Output files",
        str(model_input_path),
        str(feature_manifest_path),
        str(sample_coverage_path),
        str(long_path),
        str(category_coverage_path),
        str(out_dir / "single_or_batch_handoff_candidate_source_tables.tsv"),
        str(out_dir / "improved_transfer_handoff_summary.json"),
        "",
        "Interpretation",
        "This handoff is intended for transfer inference, not cohort model training.",
        "It intentionally avoids cohort variance filtering.",
        "Real numeric zeros are preserved.",
        "Unavailable features are retained as NaN unless explicit zero-fill is requested.",
        "Transfer Step 02 converts unavailable features to neutral z=0 while retaining missingness provenance.",
        "Confidence should be interpreted with the per-sample and treatment-specific coverage reports.",
    ]
    write_report(report_path, report_lines)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-output-root", required=True, help="Output root from spatial_feature_identification_pipeline Steps 01-09.")
    parser.add_argument("--pim-run-root", required=True, help="Completed prediction_interpretation_model run root.")
    parser.add_argument("--output-dir", required=True, help="Directory where transfer handoff files will be written.")
    parser.add_argument("--strict-feature-dictionary", default="", help="Optional explicit strict feature dictionary path.")
    parser.add_argument("--sample-id-map", default="", help="Optional TSV/CSV mapping internal sample IDs to transfer sample IDs.")
    parser.add_argument("--zero-fill-absent-zero-like", action="store_true", help="Optional: set absent zero-like features to 0 instead of NaN.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = build_handoff(
        spatial_output_root=Path(args.spatial_output_root),
        pim_run_root=Path(args.pim_run_root),
        out_dir=Path(args.output_dir),
        strict_feature_dict=Path(args.strict_feature_dictionary) if args.strict_feature_dictionary else None,
        sample_id_map_path=Path(args.sample_id_map) if args.sample_id_map else None,
        zero_fill_absent_zero_like=bool(args.zero_fill_absent_zero_like),
    )

    print("")
    print("=" * 72)
    print("IMPROVED TRANSFER HANDOFF SUMMARY")
    print("=" * 72)
    print(f"status: {summary['status']}")
    print(f"strict_feature_count: {summary['strict_feature_count']}")
    print(f"sample_count: {len(summary['samples'])}")
    for row in summary["samples"]:
        print(
            f"{row['sample_id']}: "
            f"{row['strict_features_nonmissing']}/{row['strict_features_total']} nonmissing "
            f"({row['strict_feature_nonmissing_fraction']:.3f})"
        )
    print("")
    print("model_input_numeric:")
    print(summary["model_input_numeric"])
    print("feature_manifest:")
    print(summary["feature_manifest"])
    print("sample_coverage:")
    print(summary["sample_coverage"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())