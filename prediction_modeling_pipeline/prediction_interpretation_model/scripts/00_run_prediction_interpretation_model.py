#!/usr/bin/env python
"""
Script:
    00_run_prediction_interpretation_model.py

Description:
    Orchestrates the complete prediction_interpretation_model workflow.
    It dispatches Steps 01-08 against one frozen spatial_prediction_model_V2 run root
    and writes run-level logs, step manifests, output manifests, and summaries.

Instructions:
    Run this script for normal pipeline execution instead of calling step scripts
    manually. Use --steps 01,02,... for partial reruns and --steps all only when
    all upstream source contracts are already in place.

Source-truth policy:
    This script consumes spatial_prediction_model_V2 outputs as read-only inputs.
    It must not rerun V2, perform open model selection, modify V2 scripts, or use
    deprecated interpretation outputs as final source truth.
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
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Dict, List


# =============================================================================
# PIM_DOCS_SECTION: constants and source contracts
# =============================================================================
# Constants define expected files, output names, QC contracts, or reporting rules.

STEP_MAP: Dict[str, Dict[str, str]] = {
    "01": {
        "script": "01_prepare_interpretation_inputs.py",
        "label": "prepare_interpretation_inputs",
    },
    "02": {
        "script": "02_build_feature_and_treatment_dictionary.py",
        "label": "build_feature_and_treatment_dictionary",
    },
    "03": {
        "script": "03_compute_signed_spatial_effects.py",
        "label": "compute_signed_spatial_effects",
    },
    "04": {
        "script": "04_build_treatment_interpretation_cards.py",
        "label": "build_treatment_interpretation_cards",
    },
    "05": {
        "script": "05_build_sample_level_interpretations.py",
        "label": "build_sample_level_interpretations",
    },
    "06": {
        "script": "06_build_mechanism_atlas.py",
        "label": "build_mechanism_atlas",
    },
    "07": {
        "script": "07_make_final_outputs.py",
        "label": "make_final_outputs",
    },
    "08": {
        "script": "08_qc_and_package_final_outputs.py",
        "label": "qc_and_package_final_outputs",
    },
}


# =============================================================================
# PIM_DOCS_SECTION: functions
# =============================================================================
# Functions are intentionally small enough to support reruns, QC tracing, and
# clear failure messages when upstream source contracts are incomplete.

def now_stamp() -> str:
    """Return a filesystem-safe timestamp string.
    Used for run names, patch logs, and reproducible report folders."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_text_report(path: Path, body: str) -> None:
    """Write a text report with FILEPATH on the first line.
    This convention is required for all generated text reports."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    """Write structured metadata as formatted JSON.
    Creates parent folders and preserves readable provenance output."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_tsv(path: Path, rows: List[dict]) -> None:
    """Write a pandas DataFrame as a tab-separated table.
    Creates parent folders before writing the output artifact."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_steps(value: str) -> List[str]:
    """Parse a comma-separated step request.
    Validates requested steps against the orchestrator step map."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    value = str(value).strip()
    if value.lower() == "all":
        return list(STEP_MAP.keys())

    steps = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            part = part.zfill(2)
        if part not in STEP_MAP:
            raise ValueError(f"Unknown step requested: {part}")
        steps.append(part)

    if not steps:
        raise ValueError("No steps requested.")

    return steps


def open_folder(path: Path) -> None:
    """Open an output folder in the local operating system.
    Failures are intentionally nonfatal so batch runs can continue."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def build_output_manifest(root: Path) -> List[dict]:
    """Inventory files under an output root.
    Captures relative paths, absolute paths, file sizes, and suffixes."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            rows.append({
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "absolute_path": str(path),
                "size_bytes": size,
                "suffix": path.suffix.lower(),
            })
    return rows


def run_step(
    *,
    step: str,
    python_exe: str,
    model_root: Path,
    project_root: Path,
    v2_run_root: Path,
    output_root: Path,
    open_output: bool,
    logs_root: Path,
) -> dict:
    """Run one pipeline step as a subprocess.
    Captures stdout, stderr, return code, elapsed time, and status metadata."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
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
        "--v2-run-root",
        str(v2_run_root),
        "--output-root",
        str(output_root),
    ]

    if open_output:

        cmd.append("--open-output")

    started = dt.datetime.now()
    print("")
    print("=" * 72)
    print(f"Running prediction_interpretation_model step {step}: {info['label']}")
    print("=" * 72)
    print("Command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))

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
    elapsed = (finished - started).total_seconds()

    row = {
        "step": step,
        "label": info["label"],
        "script": str(script_path),
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "status": "pass" if proc.returncode == 0 else "fail",
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    if proc.returncode != 0:
        raise RuntimeError(f"Step {step} failed with return code {proc.returncode}. See {stdout_path} and {stderr_path}")

    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    Defaults preserve local project paths while allowing explicit overrides."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=None,
    )
    parser.add_argument(
        "--model-root",
        default="",
    )
    parser.add_argument(
        "--v2-run-root",
        required=True,
    )
    parser.add_argument(
        "--run-name",
        default=f"prediction_interpretation_model_full_{now_stamp()}",
    )
    parser.add_argument(
        "--output-root",
        default="",
    )
    parser.add_argument(
        "--steps",
        default="01",
        help="Comma-separated steps such as 01 or 01,02. Use all after all scripts are implemented.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument(
        "--open-output",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# PIM_DOCS_SECTION: main entry point
# =============================================================================
# The main function wires inputs, output folders, QC checks, reports, and terminal summaries.

def main() -> int:
    """Run the script's command-line workflow.
    Writes outputs, QC checks, summaries, and terminal status messages."""
    # PIM_DOCS: keep this block explicit so downstream QC and reports remain traceable.
    args = parse_args()

    project_root = Path(args.project_root)
    if args.model_root:
        model_root = Path(args.model_root)
    else:
        model_root = Path(__file__).resolve().parents[1]

    v2_run_root = Path(args.v2_run_root)

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = model_root / "outputs" / args.run_name

    logs_root = output_root / "pipeline_run_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    requested_steps = parse_steps(args.steps)

    status = "pass"
    error_text = ""
    step_rows: List[dict] = []

    started = dt.datetime.now()

    try:
        for step in requested_steps:
            step_row = run_step(
                step=step,
                python_exe=args.python,
                model_root=model_root,
                project_root=project_root,
                v2_run_root=v2_run_root,
                output_root=output_root,
                open_output=args.open_output,
                logs_root=logs_root,
            )
            step_rows.append(step_row)

    except Exception as exc:
        status = "fail"
        error_text = "".join(traceback.format_exception(exc))
        print(error_text)

    finished = dt.datetime.now()

    write_tsv(output_root / "prediction_interpretation_model_orchestrator_step_manifest.tsv", step_rows)

    manifest_rows = build_output_manifest(output_root)
    write_tsv(output_root / "prediction_interpretation_model_output_manifest.tsv", manifest_rows)

    summary = {
        "status": status,
        "run_name": args.run_name,
        "project_root": str(project_root),
        "model_root": str(model_root),
        "v2_run_root": str(v2_run_root),
        "output_root": str(output_root),
        "requested_steps": requested_steps,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "error": error_text,
        "step_manifest": str(output_root / "prediction_interpretation_model_orchestrator_step_manifest.tsv"),
        "output_manifest": str(output_root / "prediction_interpretation_model_output_manifest.tsv"),
    }
    write_json(output_root / "prediction_interpretation_model_run_summary.json", summary)

    report_body = "\n".join([
        "PREDICTION INTERPRETATION MODEL ORCHESTRATOR REPORT",
        "",
        f"status: {status}",
        f"run_name: {args.run_name}",
        f"project_root: {project_root}",
        f"model_root: {model_root}",
        f"v2_run_root: {v2_run_root}",
        f"output_root: {output_root}",
        f"requested_steps: {', '.join(requested_steps)}",
        f"started: {started.isoformat(timespec='seconds')}",
        f"finished: {finished.isoformat(timespec='seconds')}",
        f"elapsed_seconds: {(finished - started).total_seconds():.2f}",
        "",
        "Policy",
        "This orchestrator consumes spatial_prediction_model_V2 outputs.",
        "It does not rerun V2.",
        "It does not perform open model selection.",
        "It does not use deprecated pipeline outputs as source truth.",
        "",
        "Step manifest",
        str(output_root / "prediction_interpretation_model_orchestrator_step_manifest.tsv"),
        "",
        "Output manifest",
        str(output_root / "prediction_interpretation_model_output_manifest.tsv"),
        "",
        "Error",
        error_text if error_text else "none",
    ])
    write_text_report(output_root / "prediction_interpretation_model_orchestrator_report.txt", report_body)

    print("")
    print("=" * 72)
    print("PREDICTION INTERPRETATION MODEL ORCHESTRATOR SUMMARY")
    print("=" * 72)
    print(f"status: {status}")
    print(f"output_root: {output_root}")
    print(f"step_manifest: {output_root / 'prediction_interpretation_model_orchestrator_step_manifest.tsv'}")
    print(f"run_summary: {output_root / 'prediction_interpretation_model_run_summary.json'}")

    if args.open_output:
        open_folder(output_root)

    return 0 if status == "pass" else 1


# =============================================================================
# PIM_DOCS_SECTION: command-line guard
# =============================================================================
# Keep this guard so scripts can be imported for testing without executing the step.

if __name__ == "__main__":
    raise SystemExit(main())

