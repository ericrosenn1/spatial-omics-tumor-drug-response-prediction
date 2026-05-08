#!/usr/bin/env python
"""
Script:
    01_prepare_transfer_inputs.py

Purpose:
    Prepare one-slide or batch transfer inference inputs.

Inputs:
    - Completed prediction_interpretation_model run root
    - Optional explicit single-slide feature table
    - Optional spatial_feature_identification_pipeline output root

Smoke-test behavior:
    If --smoke-test is provided and no feature table is supplied, this step creates
    a V2-compatible synthetic transfer input by taking one sample row from the
    PIM-copied V2 spatial feature pool. This smoke test validates adapter/scorer
    logic, not raw Visium feature extraction.

Policy:
    Does not rerun V2.
    Does not retrain drug-response models.
    Does not make clinical treatment recommendations.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import shutil
import traceback
from typing import List, Optional

import pandas as pd

from _stim_utils import (
    add_qc,
    choose_col,
    ensure_dir,
    load_pim_feature_dictionary,
    load_pim_spatial_feature_pool,
    open_folder,
    read_header,
    read_table,
    report_status_from_qc,
    save_output_manifest,
    strict_feature_names,
    summarize_examples,
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


def candidate_score(path: Path, strict_features: List[str]) -> dict:
    try:
        header = read_header(path)
    except Exception as exc:
        return {
            "path": str(path),
            "readable": False,
            "overlap_count": 0,
            "overlap_fraction": 0.0,
            "score": -999,
            "error": str(exc),
        }

    header_set = set(map(str, header))
    strict_set = set(strict_features)
    overlap = sorted(header_set.intersection(strict_set))
    name = path.name.lower()
    keyword_score = 0
    for kw in ["model_ready", "feature_matrix", "slide_feature", "spatial_features", "sample_features", "strict", "final_features"]:
        if kw in name:
            keyword_score += 10

    return {
        "path": str(path),
        "readable": True,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / max(len(strict_features), 1),
        "score": len(overlap) + keyword_score,
        "error": "",
        "example_overlapping_features": summarize_examples(overlap, 10),
        "n_columns": len(header),
    }


def find_feature_table(root: Path, strict_features: List[str]) -> tuple[Optional[Path], pd.DataFrame]:
    if not root or not root.exists():
        return None, pd.DataFrame()

    skip_parts = {".git", "__pycache__", ".venv", "venv", "env", "logs", "backup", "deprecated"}
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".tsv", ".tab", ".csv"]:
            continue
        parts_lower = {p.lower() for p in path.parts}
        if parts_lower.intersection(skip_parts):
            continue
        candidates.append(candidate_score(path, strict_features))

    df = pd.DataFrame(candidates)
    if df.empty:
        return None, df

    df = df.sort_values(["score", "overlap_count"], ascending=[False, False])
    best = df.iloc[0]
    if int(best["overlap_count"]) <= 0:
        return None, df
    return Path(str(best["path"])), df


def standardize_feature_input(path: Path, sample_id: str, strict_features: List[str]) -> pd.DataFrame:
    """
    Standardize explicit transfer input.

    Batch-safe behavior:
      - If the input already has a sample_id/slide_id column, preserve all rows.
      - Do not filter to --sample-id unless it uniquely matches an existing row.
      - Do not collapse multi-row transfer handoffs to the first row.
      - --sample-id is treated as a run/batch label when the table already has sample IDs.
    """
    df = read_table(path)
    if df.empty:
        raise ValueError(f"Feature input table is empty: {path}")

    sample_col = choose_col(df.columns, ["sample_id", "slide_id", "sample", "library_id", "visium_sample_id"], required=False)

    if sample_col is None:
        if len(df) == 1:
            df.insert(0, "sample_id", sample_id if sample_id else "TRANSFER_SAMPLE_001")
        else:
            df.insert(0, "sample_id", [f"TRANSFER_SAMPLE_{i:04d}" for i in range(len(df))])
    elif sample_col != "sample_id":
        df = df.rename(columns={sample_col: "sample_id"})

    df["sample_id"] = df["sample_id"].astype(str)

    # Only subset if --sample-id exactly matches one or more rows.
    # For batch runs, --sample-id is usually a batch/run label and should not subset.
    if sample_id and sample_id in set(df["sample_id"].astype(str)):
        df = df[df["sample_id"].astype(str) == str(sample_id)].copy()

    # Drop accidental duplicate sample rows, keeping first complete row.
    feature_cols = [c for c in df.columns if c in strict_features]
    if feature_cols:
        df["_transfer_nonmissing_strict_count"] = df[feature_cols].notna().sum(axis=1)
        df = (
            df.sort_values(["sample_id", "_transfer_nonmissing_strict_count"], ascending=[True, False])
              .drop_duplicates("sample_id", keep="first")
              .drop(columns=["_transfer_nonmissing_strict_count"])
              .reset_index(drop=True)
        )
    else:
        df = df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    return df


def make_smoke_input(pim_run_root: Path, sample_id: str, strict_features: List[str]) -> pd.DataFrame:
    header_df = load_pim_spatial_feature_pool(pim_run_root, nrows=0)
    header = list(header_df.columns)

    source_sample_col = choose_col(header, ["sample_id", "slide_id", "sample"], required=True, label="PIM spatial pool sample column")
    available_features = [f for f in strict_features if f in header]
    usecols = [source_sample_col] + available_features

    df = load_pim_spatial_feature_pool(pim_run_root, usecols=usecols, nrows=1)
    if source_sample_col != "sample_id":
        df = df.rename(columns={source_sample_col: "sample_id"})

    original_sample_id = str(df["sample_id"].iloc[0])
    df["source_v2_sample_id_for_smoke_test"] = original_sample_id
    df["sample_id"] = sample_id
    return df


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    output_root = Path(args.output_root)
    pim_run_root = Path(args.pim_run_root)

    step_root = output_root / "01_prepared_transfer_inputs"
    manifest_dir = step_root / "01_input_manifests"
    feature_dir = step_root / "02_single_slide_feature_input"
    registry_dir = step_root / "03_reference_registry"
    qc_dir = step_root / "04_qc"
    report_dir = step_root / "05_reports"

    for path in [manifest_dir, feature_dir, registry_dir, qc_dir, report_dir]:
        ensure_dir(path)

    errors: List[str] = []
    warnings: List[str] = []
    qc: List[dict] = []
    candidate_df = pd.DataFrame()
    selected_feature_table = None
    input_mode = ""

    try:
        feature_dict = load_pim_feature_dictionary(pim_run_root)
        strict_features = sorted(feature_dict["feature_name"].dropna().astype(str).unique())
        write_tsv(registry_dir / "pim_strict_feature_registry_for_transfer.tsv", feature_dict)

        if args.single_slide_feature_table and Path(args.single_slide_feature_table).exists():
            selected_feature_table = Path(args.single_slide_feature_table)
            input_mode = "explicit_single_slide_feature_table"
            single_df = standardize_feature_input(selected_feature_table, args.sample_id, strict_features)

        elif args.spatial_feature_run_root and Path(args.spatial_feature_run_root).exists():
            selected_feature_table, candidate_df = find_feature_table(Path(args.spatial_feature_run_root), strict_features)
            if selected_feature_table is None:
                if args.smoke_test:
                    input_mode = "smoke_test_fallback_from_pim_v2_spatial_feature_pool"
                    single_df = make_smoke_input(pim_run_root, args.sample_id, strict_features)
                else:
                    raise FileNotFoundError("Could not identify a spatial feature table with overlap to the PIM strict feature registry.")
            else:
                input_mode = "auto_detected_spatial_feature_table"
                single_df = standardize_feature_input(selected_feature_table, args.sample_id, strict_features)

        elif args.smoke_test:
            input_mode = "smoke_test_from_pim_v2_spatial_feature_pool"
            single_df = make_smoke_input(pim_run_root, args.sample_id, strict_features)

        else:
            raise FileNotFoundError("Provide --single-slide-feature-table, --spatial-feature-run-root, or --smoke-test.")

        write_tsv(feature_dir / "transfer_single_slide_feature_input.tsv", single_df)

        if not candidate_df.empty:
            write_tsv(manifest_dir / "auto_detected_feature_table_candidates.tsv", candidate_df)

        input_features = set(single_df.columns.astype(str))
        strict_set = set(strict_features)
        overlap = sorted(input_features.intersection(strict_set))
        missing = sorted(strict_set - input_features)

        feature_coverage = pd.DataFrame([
            {
                "feature_name": f,
                "present_in_transfer_input": f in input_features,
                "input_column": f if f in input_features else "",
            }
            for f in strict_features
        ])
        write_tsv(manifest_dir / "transfer_input_strict_feature_coverage.tsv", feature_coverage)

        input_manifest = pd.DataFrame([
            {
                "input_id": "pim_run_root",
                "path": str(pim_run_root),
                "exists": pim_run_root.exists(),
                "role": "completed prediction_interpretation_model source",
            },
            {
                "input_id": "spatial_feature_run_root",
                "path": args.spatial_feature_run_root,
                "exists": Path(args.spatial_feature_run_root).exists() if args.spatial_feature_run_root else False,
                "role": "optional spatial_feature_identification_pipeline output root",
            },
            {
                "input_id": "selected_single_slide_feature_table",
                "path": str(selected_feature_table) if selected_feature_table else "",
                "exists": selected_feature_table.exists() if selected_feature_table else False,
                "role": "explicit or auto-detected transfer feature table",
            },
            {
                "input_id": "prepared_transfer_single_slide_feature_input",
                "path": str(feature_dir / "transfer_single_slide_feature_input.tsv"),
                "exists": (feature_dir / "transfer_single_slide_feature_input.tsv").exists(),
                "role": "standardized transfer input consumed by Step 02",
            },
        ])
        write_tsv(manifest_dir / "transfer_input_manifest.tsv", input_manifest)

        add_qc(qc, "pim_run_root_exists", "pass" if pim_run_root.exists() else "fail", pim_run_root.exists(), True, "Completed prediction_interpretation_model run root must exist.")
        add_qc(qc, "strict_feature_registry_loaded", "pass" if len(strict_features) == 139 else "warn", len(strict_features), 139, "PIM strict feature registry loaded.")
        add_qc(qc, "transfer_feature_input_rows", "pass" if len(single_df) >= 1 else "fail", len(single_df), ">=1", "Transfer feature input should have at least one sample row.")
        add_qc(qc, "strict_feature_overlap_count", "pass" if len(overlap) >= 100 else ("warn" if len(overlap) >= 50 else "fail"), len(overlap), ">=100 preferred", "Overlap between transfer input and PIM strict features.")
        add_qc(qc, "input_mode_recorded", "pass", input_mode, "recorded", "Input mode selected by Step 01.")

        if missing:
            warnings.append(f"Missing strict features in transfer input: {len(missing)}")

    except Exception as exc:
        errors.append("".join(traceback.format_exception(exc)))
        single_df = pd.DataFrame()
        strict_features = []

    status = report_status_from_qc(qc, errors)
    finished = dt.datetime.now()

    qc_df = pd.DataFrame(qc)
    write_tsv(qc_dir / "step01_prepare_transfer_inputs_qc_checks.tsv", qc_df)

    summary = {
        "status": status,
        "output_root": str(output_root),
        "step_root": str(step_root),
        "pim_run_root": str(pim_run_root),
        "spatial_feature_run_root": args.spatial_feature_run_root,
        "single_slide_feature_table": args.single_slide_feature_table,
        "selected_feature_table": str(selected_feature_table) if selected_feature_table else "",
        "sample_id": args.sample_id,
        "input_mode": input_mode,
        "smoke_test": bool(args.smoke_test),
        "strict_feature_count": len(strict_features),
        "transfer_input_rows": int(len(single_df)),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_root / "spatial_transfer_inference_model_step01_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL STEP 01 REPORT",
        "",
        f"status: {status}",
        f"pim_run_root: {pim_run_root}",
        f"spatial_feature_run_root: {args.spatial_feature_run_root}",
        f"single_slide_feature_table: {args.single_slide_feature_table}",
        f"selected_feature_table: {selected_feature_table if selected_feature_table else ''}",
        f"sample_id: {args.sample_id}",
        f"input_mode: {input_mode}",
        f"smoke_test: {args.smoke_test}",
        f"output_root: {output_root}",
        f"step_root: {step_root}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Outputs",
        str(feature_dir / "transfer_single_slide_feature_input.tsv"),
        str(registry_dir / "pim_strict_feature_registry_for_transfer.tsv"),
        str(manifest_dir / "transfer_input_manifest.tsv"),
        str(manifest_dir / "transfer_input_strict_feature_coverage.tsv"),
        "",
        "Smoke-test note",
        "If input_mode starts with smoke_test, this step used one V2-compatible feature row from the completed PIM run to test the transfer adapter. It did not run raw Visium feature extraction.",
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
    write_text_report(report_dir / "step01_prepare_transfer_inputs_report.txt", "\n".join(report_lines))
    save_output_manifest(output_root)

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL STEP 01 SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"input_mode: {input_mode}")
    print(f"sample_id: {args.sample_id}")
    print(f"strict_feature_count: {len(strict_features)}")
    print(f"transfer_input_rows: {len(single_df)}")
    print(f"report: {report_dir / 'step01_prepare_transfer_inputs_report.txt'}")

    if args.open_output:
        open_folder(step_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
