#!/usr/bin/env python
"""
Script:
    00_run_spatial_transfer_inference_model.py

Purpose:
    Orchestrate spatial_transfer_inference_model steps 01-05.

Role:
    This is the transfer inference layer that consumes:
      1. a single-slide or batch spatial feature table from spatial_feature_identification_pipeline
      2. the completed prediction_interpretation_model run

    It does not rerun spatial_prediction_model_V2.
    It does not train new drug-response models.
    It does not make clinical treatment recommendations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Dict, List

from _stim_utils import build_output_manifest, ensure_dir, open_folder, write_json, write_text_report, write_tsv


STEP_MAP: Dict[str, Dict[str, str]] = {
    "01": {"script": "01_prepare_transfer_inputs.py", "label": "prepare_transfer_inputs"},
    "02": {"script": "02_align_single_slide_features_to_v2.py", "label": "align_single_slide_features_to_v2"},
    "03": {"script": "03_score_transfer_drug_response_alignment.py", "label": "score_transfer_drug_response_alignment"},
    "04": {"script": "04_make_single_slide_prediction_table.py", "label": "make_single_slide_prediction_table"},
    "05": {"script": "05_qc_and_package_transfer_outputs.py", "label": "qc_and_package_transfer_outputs"},
}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_steps(value: str) -> List[str]:
    value = str(value).strip()
    if value.lower() == "all":
        return list(STEP_MAP.keys())
    out = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            part = part.zfill(2)
        if part not in STEP_MAP:
            raise ValueError(f"Unknown step: {part}")
        out.append(part)
    if not out:
        raise ValueError("No steps requested.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=r"D:\Adv_Omics_Fenyo\project")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--pim-run-root", required=True, help="Completed prediction_interpretation_model run root.")
    parser.add_argument("--spatial-feature-run-root", default="", help="Optional spatial_feature_identification_pipeline output root.")
    parser.add_argument("--single-slide-feature-table", default="", help="Optional explicit one-row or batch feature table.")
    parser.add_argument("--sample-id", default="TRANSFER_SAMPLE_001")
    parser.add_argument("--run-name", default=f"spatial_transfer_inference_{now_stamp()}")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--steps", default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--open-output", action="store_true")

    return parser.parse_args()


def run_step(
    *,
    step: str,
    python_exe: str,
    model_root: Path,
    project_root: Path,
    pim_run_root: Path,
    spatial_feature_run_root: str,
    single_slide_feature_table: str,
    sample_id: str,
    output_root: Path,
    smoke_test: bool,
    open_output: bool,
    logs_root: Path,
) -> dict:
    info = STEP_MAP[step]
    script_path = model_root / "scripts" / info["script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script for step {step}: {script_path}")

    stdout_path = logs_root / f"{step}_{info['label']}_stdout.log"
    stderr_path = logs_root / f"{step}_{info['label']}_stderr.log"

    cmd = [
        python_exe,
        str(script_path),
        "--project-root",
        str(project_root),
        "--model-root",
        str(model_root),
        "--pim-run-root",
        str(pim_run_root),
        "--output-root",
        str(output_root),
        "--sample-id",
        str(sample_id),
    ]

    if spatial_feature_run_root:
        cmd.extend(["--spatial-feature-run-root", spatial_feature_run_root])
    if single_slide_feature_table:
        cmd.extend(["--single-slide-feature-table", single_slide_feature_table])
    if smoke_test:
        cmd.append("--smoke-test")
    if open_output:
        cmd.append("--open-output")

    started = dt.datetime.now()

    print("")
    print("=" * 72)
    print(f"Running spatial_transfer_inference_model step {step}: {info['label']}")
    print("=" * 72)
    print("Command:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))

    proc = subprocess.run(cmd, text=True, capture_output=True)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print("")
        print("STDERR:")
        print(proc.stderr)

    finished = dt.datetime.now()

    row = {
        "step": step,
        "label": info["label"],
        "script": str(script_path),
        "return_code": proc.returncode,
        "status": "pass" if proc.returncode == 0 else "fail",
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": " ".join(cmd),
    }

    if proc.returncode != 0:
        raise RuntimeError(f"Step {step} failed with return code {proc.returncode}. See {stdout_path} and {stderr_path}")

    return row


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root)
    model_root = Path(args.model_root) if args.model_root else Path(__file__).resolve().parents[1]
    pim_run_root = Path(args.pim_run_root)

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = model_root / "outputs" / args.run_name

    logs_root = output_root / "pipeline_run_logs"
    ensure_dir(logs_root)

    steps = parse_steps(args.steps)
    started = dt.datetime.now()

    rows: List[dict] = []
    status = "pass"
    error_text = ""

    try:
        for step in steps:
            rows.append(
                run_step(
                    step=step,
                    python_exe=args.python,
                    model_root=model_root,
                    project_root=project_root,
                    pim_run_root=pim_run_root,
                    spatial_feature_run_root=args.spatial_feature_run_root,
                    single_slide_feature_table=args.single_slide_feature_table,
                    sample_id=args.sample_id,
                    output_root=output_root,
                    smoke_test=args.smoke_test,
                    open_output=args.open_output,
                    logs_root=logs_root,
                )
            )
    except Exception as exc:
        status = "fail"
        error_text = "".join(traceback.format_exception(exc))
        print(error_text)

    finished = dt.datetime.now()

    write_tsv(output_root / "spatial_transfer_inference_model_orchestrator_step_manifest.tsv", __import__("pandas").DataFrame(rows))
    write_tsv(output_root / "spatial_transfer_inference_model_output_manifest.tsv", build_output_manifest(output_root))

    summary = {
        "status": status,
        "run_name": args.run_name,
        "project_root": str(project_root),
        "model_root": str(model_root),
        "pim_run_root": str(pim_run_root),
        "spatial_feature_run_root": args.spatial_feature_run_root,
        "single_slide_feature_table": args.single_slide_feature_table,
        "sample_id": args.sample_id,
        "output_root": str(output_root),
        "requested_steps": steps,
        "smoke_test": bool(args.smoke_test),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "error": error_text,
    }
    write_json(output_root / "spatial_transfer_inference_model_run_summary.json", summary)

    report_lines = [
        "SPATIAL TRANSFER INFERENCE MODEL ORCHESTRATOR REPORT",
        "",
        f"status: {status}",
        f"run_name: {args.run_name}",
        f"model_root: {model_root}",
        f"pim_run_root: {pim_run_root}",
        f"spatial_feature_run_root: {args.spatial_feature_run_root}",
        f"single_slide_feature_table: {args.single_slide_feature_table}",
        f"sample_id: {args.sample_id}",
        f"output_root: {output_root}",
        f"smoke_test: {args.smoke_test}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Policy",
        "This transfer inference layer consumes frozen prediction_interpretation_model outputs and single-slide spatial features.",
        "It does not rerun spatial_prediction_model_V2.",
        "It does not train new drug-response models.",
        "It does not make clinical treatment recommendations.",
        "",
        "Error",
        error_text if error_text else "none",
    ]
    write_text_report(output_root / "spatial_transfer_inference_model_orchestrator_report.txt", "\n".join(report_lines))

    print("")
    print("=" * 72)
    print("SPATIAL TRANSFER INFERENCE MODEL ORCHESTRATOR SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"output_root: {output_root}")
    print(f"run_summary: {output_root / 'spatial_transfer_inference_model_run_summary.json'}")
    print(f"step_manifest: {output_root / 'spatial_transfer_inference_model_orchestrator_step_manifest.tsv'}")

    if args.open_output:
        open_folder(output_root)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())