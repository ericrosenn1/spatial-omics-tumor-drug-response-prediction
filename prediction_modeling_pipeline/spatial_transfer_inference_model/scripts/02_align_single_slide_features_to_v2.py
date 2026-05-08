#!/usr/bin/env python
"""
Script:
    02_align_single_slide_features_to_v2.py

Purpose:
    Align single-slide spatial features to the frozen PIM/V2 strict biology
    feature registry and scale them using the V2 spatial feature reference
    distribution.

Outputs:
    - raw strict feature vector
    - V2-scaled strict feature vector
    - long feature alignment table
    - V2 reference statistics
    - coverage QC
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
    choose_col,
    ensure_dir,
    load_pim_feature_dictionary,
    load_pim_spatial_feature_pool,
    numeric_series,
    open_folder,
    read_header,
    read_table,
    report_status_from_qc,
    save_output_manifest,
    write_json,
    write_text_report,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--pim-run-root", required=True)
    parser.add_argument("--spatial-feature-run-root", default="")
    parser.add_argument("--single-slide-feature-table", default="")
    parser.add_argument("--sample-id", default="TRANSFER_SAMPLE_001")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--open-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    output_root = Path(args.output_root)
    pim_run_root = Path(args.pim_run_root)

    step01_root = output_root / "01_prepared_transfer_inputs"
    feature_input_path = step01_root / "02_single_slide_feature_input" / "transfer_single_slide_feature_input.tsv"
    if not feature_input_path.exists():
        raise FileNotFoundError(f"Step 01 transfer feature input not found: {feature_input_path}")

    step_root = output_root / "02_aligned_features"
    vector_dir = step_root / "01_aligned_feature_vectors"
    ref_dir = step_root / "02_v2_reference_statistics"
    qc_dir = step_root / "03_qc"
    report_dir = step_root / "04_reports"

    for path in [vector_dir, ref_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []

    try:
        feature_dict = load_pim_feature_dictionary(pim_run_root)
        strict_features = sorted(feature_dict["feature_name"].dropna().astype(str).unique())

        input_df = read_table(feature_input_path)
        sample_col = choose_col(input_df.columns, ["sample_id", "slide_id", "sample"], required=True, label="transfer input sample column")
        if sample_col != "sample_id":
            input_df = input_df.rename(columns={sample_col: "sample_id"})

        v2_header = list(load_pim_spatial_feature_pool(pim_run_root, nrows=0).columns)
        v2_sample_col = choose_col(v2_header, ["sample_id", "slide_id", "sample"], required=True, label="V2 spatial pool sample column")
        v2_features = [f for f in strict_features if f in v2_header]
        v2 = load_pim_spatial_feature_pool(pim_run_root, usecols=[v2_sample_col] + v2_features)
        if v2_sample_col != "sample_id":
            v2 = v2.rename(columns={v2_sample_col: "sample_id"})

        ref_rows = []
        for feature in strict_features:
            if feature in v2.columns:
                vals = numeric_series(v2[feature])
                mean = float(vals.mean()) if vals.notna().any() else 0.0
                std = float(vals.std(ddof=0)) if vals.notna().any() else 0.0
                median = float(vals.median()) if vals.notna().any() else 0.0
                q25 = float(vals.quantile(0.25)) if vals.notna().any() else 0.0
                q75 = float(vals.quantile(0.75)) if vals.notna().any() else 0.0
                min_v = float(vals.min()) if vals.notna().any() else 0.0
                max_v = float(vals.max()) if vals.notna().any() else 0.0
                missing_rate = float(vals.isna().mean())
                present = True
            else:
                mean = std = median = q25 = q75 = min_v = max_v = 0.0
                missing_rate = 1.0
                present = False
            ref_rows.append({
                "feature_name": feature,
                "present_in_v2_reference": present,
                "v2_mean": mean,
                "v2_std": std,
                "v2_median": median,
                "v2_q25": q25,
                "v2_q75": q75,
                "v2_iqr": q75 - q25,
                "v2_min": min_v,
                "v2_max": max_v,
                "v2_missing_rate": missing_rate,
            })

        ref_df = pd.DataFrame(ref_rows)
        write_tsv(ref_dir / "v2_strict_feature_reference_statistics.tsv", ref_df)

        ref_by_feature = ref_df.set_index("feature_name").to_dict("index")

        raw_rows = []
        scaled_rows = []
        long_rows = []

        for _, sample_row in input_df.iterrows():
            sample_id = str(sample_row["sample_id"])
            raw_row = {"sample_id": sample_id}
            scaled_row = {"sample_id": sample_id}

            for feature in strict_features:
                present_col = feature in input_df.columns
                value = pd.to_numeric(pd.Series([sample_row.get(feature, np.nan)]), errors="coerce").iloc[0] if present_col else np.nan
                is_nonmissing = bool(pd.notna(value))

                ref = ref_by_feature.get(feature, {})
                mean = float(ref.get("v2_mean", 0.0) or 0.0)
                std = float(ref.get("v2_std", 0.0) or 0.0)
                v2_min = float(ref.get("v2_min", 0.0) or 0.0)
                v2_max = float(ref.get("v2_max", 0.0) or 0.0)

                if is_nonmissing and np.isfinite(value) and np.isfinite(std) and std > 0:
                    z = float((float(value) - mean) / std)
                else:
                    z = 0.0

                within_minmax = bool(is_nonmissing and float(value) >= v2_min and float(value) <= v2_max) if pd.notna(value) else False

                raw_row[feature] = value
                scaled_row[feature] = z

                long_rows.append({
                    "sample_id": sample_id,
                    "feature_name": feature,
                    "present_in_transfer_input": present_col,
                    "nonmissing_in_transfer_input": is_nonmissing,
                    "raw_value": value,
                    "v2_scaled_z": z,
                    "v2_mean": mean,
                    "v2_std": std,
                    "v2_min": v2_min,
                    "v2_max": v2_max,
                    "within_v2_minmax": within_minmax,
                })

            raw_rows.append(raw_row)
            scaled_rows.append(scaled_row)

        raw_df = pd.DataFrame(raw_rows)
        scaled_df = pd.DataFrame(scaled_rows)
        long_df = pd.DataFrame(long_rows)

        coverage = (
            long_df.groupby("sample_id", as_index=False)
            .agg(
                strict_features_total=("feature_name", "nunique"),
                strict_features_present=("present_in_transfer_input", "sum"),
                strict_features_nonmissing=("nonmissing_in_transfer_input", "sum"),
                features_within_v2_minmax=("within_v2_minmax", "sum"),
            )
        )
        coverage["present_fraction"] = coverage["strict_features_present"] / coverage["strict_features_total"]
        coverage["nonmissing_fraction"] = coverage["strict_features_nonmissing"] / coverage["strict_features_total"]
        coverage["within_v2_minmax_fraction"] = coverage["features_within_v2_minmax"] / coverage["strict_features_total"]

        write_tsv(vector_dir / "single_slide_raw_strict_feature_vector.tsv", raw_df)
        write_tsv(vector_dir / "single_slide_v2_scaled_feature_vector.tsv", scaled_df)
        write_tsv(vector_dir / "single_slide_feature_alignment_long.tsv", long_df)
        write_tsv(vector_dir / "single_slide_feature_coverage_summary.tsv", coverage)

        min_coverage = float(coverage["nonmissing_fraction"].min()) if not coverage.empty else 0.0
        min_present = float(coverage["present_fraction"].min()) if not coverage.empty else 0.0

        add_qc(qc, "strict_feature_count", "pass" if len(strict_features) == 139 else "warn", len(strict_features), 139, "Strict V2 biology features used for transfer alignment.")
        add_qc(qc, "transfer_samples_aligned", "pass" if len(raw_df) >= 1 else "fail", len(raw_df), ">=1", "At least one sample aligned.")
        add_qc(qc, "min_strict_feature_present_fraction", "pass" if min_present >= 0.80 else ("warn" if min_present >= 0.50 else "fail"), f"{min_present:.3f}", ">=0.80 preferred", "Fraction of strict features present as columns.")
        add_qc(qc, "min_strict_feature_nonmissing_fraction", "pass" if min_coverage >= 0.80 else "warn", f"{min_coverage:.3f}", ">=0.80 preferred; low coverage allowed for single-slide transfer with neutral missing-feature scaling", "Fraction of strict features with nonmissing values.")
        add_qc(qc, "v2_reference_features_available", "pass" if len(v2_features) >= 139 else "warn", len(v2_features), 139, "Strict features available in V2 reference pool.")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        raw_df = pd.DataFrame()
        scaled_df = pd.DataFrame()
        long_df = pd.DataFrame()
        coverage = pd.DataFrame()

    status = report_status_from_qc(qc, errors)
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step02_align_features_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "aligned_sample_rows": len(raw_df),
        "strict_feature_columns": len([c for c in raw_df.columns if c != "sample_id"]) if not raw_df.empty else 0,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "spatial_transfer_inference_model_step02_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL STEP 02 REPORT",
        "",
        f"status: {status}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        str(vector_dir / "single_slide_raw_strict_feature_vector.tsv"),
        str(vector_dir / "single_slide_v2_scaled_feature_vector.tsv"),
        str(vector_dir / "single_slide_feature_alignment_long.tsv"),
        str(vector_dir / "single_slide_feature_coverage_summary.tsv"),
        str(ref_dir / "v2_strict_feature_reference_statistics.tsv"),
        "",
        "QC checks",
        qc_df.to_string(index=False) if not qc_df.empty else "none",
        "",
        "Interpretation",
        "Feature values are aligned to the frozen PIM/V2 strict biology registry and converted to V2-reference z-scores for transfer scoring.",
        "",
        "Errors",
        "\n".join(errors) if errors else "none",
        "",
        "Warnings",
        "\n".join(warnings) if warnings else "none",
    ]
    write_text_report(report_dir / "step02_align_single_slide_features_to_v2_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL STEP 02 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"aligned_sample_rows: {len(raw_df)}")
    print(f"strict_feature_columns: {summary['strict_feature_columns']}")
    if not coverage.empty:
        print(f"min_nonmissing_fraction: {coverage['nonmissing_fraction'].min():.3f}")
    print(f"report: {report_dir / 'step02_align_single_slide_features_to_v2_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
