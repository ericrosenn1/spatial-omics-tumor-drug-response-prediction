"""
Script: 00_run_spatial_prediction_model_v2.py

Purpose:
    Orchestrate the official spatial_prediction_model_V2 workflow.

Pipeline role:
    This runner executes V2 Steps 01 through 12 in order, using the governed
    teacher-builder handoff as input and writing a single organized V2 run folder.

Scientific role:
    The orchestrator does not define new modeling logic. It enforces the
    validated V2 analysis order: input validation, governed dataset construction,
    probability baseline diagnostics, prior-adjusted residual evidence,
    strict-biology registry generation, broad residual modeling, per-treatment
    residual modeling, curation, label-shuffle validation, integrated
    interpretation packaging, publication table packaging, and final output QC.

Worker policy:
    Step 09 receives --max-workers.
    --max-workers 0 delegates automatic worker selection to Step 09.
    Step 09 parallelizes across Tier 1 treatments and keeps XGBoost n_jobs=1
    inside each worker to avoid nested CPU oversubscription.

Policy:
    V2 production runs do not modify V1 scripts.
    V2 production runs do not depend on V1 output files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = V2_ROOT.parents[1]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_report(path: Path | str, body: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")
    return path


def write_json(path: Path | str, obj: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def terminal_block(title: str, lines: list[str]) -> str:
    bar = "=" * 90
    return "\n".join([bar, title, bar] + lines)


def open_path(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
    except Exception:
        pass


def resolve_python(user_python: str) -> str:
    if user_python:
        return user_python

    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)

    return sys.executable


def required_script(name: str) -> Path:
    path = SCRIPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required V2 script missing: {path}")
    return path


def run_step(
    step_name: str,
    command: list[str],
    log_root: Path,
    continue_on_error: bool = False,
) -> dict:
    ensure_dir(log_root)

    safe_name = step_name.replace(" ", "_")
    stdout_path = log_root / f"{safe_name}_stdout.log"
    stderr_path = log_root / f"{safe_name}_stderr.log"

    start = time.time()

    print("")
    print("=" * 90)
    print(f"RUNNING {step_name}")
    print("=" * 90)
    print("Command:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
    print("STDOUT log:", stdout_path)
    print("STDERR log:", stderr_path)

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            cwd=str(V2_ROOT),
        )

    elapsed = time.time() - start

    print("")
    print(f"{step_name} stdout tail")
    print("=" * 90)
    if stdout_path.exists():
        try:
            lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-160:]:
                print(line)
        except Exception:
            print("Could not read stdout log.")

    print("")
    print(f"{step_name} stderr tail")
    print("=" * 90)
    if stderr_path.exists() and stderr_path.stat().st_size > 0:
        try:
            lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-160:]:
                print(line)
        except Exception:
            print("Could not read stderr log.")
    else:
        print("No stderr was written.")

    result = {
        "step_name": step_name,
        "command": command,
        "return_code": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "status": "pass" if proc.returncode == 0 else "fail",
    }

    if proc.returncode != 0 and not continue_on_error:
        raise RuntimeError(f"{step_name} failed with return code {proc.returncode}. See logs: {log_root}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default="")

    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Step 09 workers. Use 0 for automatic worker selection inside Step 09.",
    )

    parser.add_argument("--step04-max-samples", type=int, default=80)
    parser.add_argument("--step04-max-treatments", type=int, default=160)
    parser.add_argument("--step07-max-treatments", type=int, default=60)

    parser.add_argument("--step09-n-shuffles", type=int, default=25)
    parser.add_argument("--step09-n-repeats", type=int, default=3)

    parser.add_argument("--full-step09-n-shuffles", type=int, default=100)
    parser.add_argument("--full-step09-n-repeats", type=int, default=5)

    parser.add_argument("--open-output", action="store_true")
    args = parser.parse_args()

    py = resolve_python(args.python)
    handoff_root = Path(args.handoff_root)

    if not handoff_root.exists():
        raise FileNotFoundError(f"Handoff root not found: {handoff_root}")

    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.output_root:
        run_root = ensure_dir(args.output_root)
    else:
        run_root = ensure_dir(V2_ROOT / "outputs" / f"v2_{args.mode}_run_{stamp}")

    log_root = ensure_dir(V2_ROOT / "logs" / f"{run_root.name}_{stamp}")

    step01 = ensure_dir(run_root / "01_validate_inputs")
    step02 = ensure_dir(run_root / "02_build_modeling_dataset")
    step03 = ensure_dir(run_root / "03_probability_baseline")
    step04 = ensure_dir(run_root / "04_pair_level_residual_model")
    step05 = ensure_dir(run_root / "05_residual_biology_registry")
    step06 = ensure_dir(run_root / "06_broad_residual_model")
    step07 = ensure_dir(run_root / "07_filtered_per_treatment_residual_models")
    step08 = ensure_dir(run_root / "08_curated_per_treatment_residual_models")
    step09 = ensure_dir(run_root / "09_tier1_label_shuffle_validation")
    step10 = ensure_dir(run_root / "10_integrated_interpretation_package")
    step11 = ensure_dir(run_root / "11_publication_tables")
    step12 = ensure_dir(run_root / "12_v2_output_qc")

    n_shuffles = args.step09_n_shuffles if args.mode == "smoke" else args.full_step09_n_shuffles
    n_repeats_09 = args.step09_n_repeats if args.mode == "smoke" else args.full_step09_n_repeats

    if args.mode == "smoke":
        step04_mode_args = [
            "--mode", "smoke",
            "--max-samples", str(args.step04_max_samples),
            "--max-treatments", str(args.step04_max_treatments),
            "--n-repeats", "3",
            "--n-estimators", "150",
            "--run-shap",
        ]
        step06_mode_args = [
            "--mode", "smoke",
            "--n-repeats", "10",
            "--n-estimators", "100",
            "--run-shap",
        ]
        step07_mode_args = [
            "--mode", "smoke",
            "--max-treatments", str(args.step07_max_treatments),
            "--n-repeats", "5",
            "--n-estimators", "80",]
    else:
        step04_mode_args = [
            "--mode", "full",
            "--n-repeats", "5",
            "--n-estimators", "200",
            "--run-shap",
        ]
        step06_mode_args = [
            "--mode", "full",
            "--n-repeats", "20",
            "--n-estimators", "150",
            "--run-shap",
        ]
        step07_mode_args = [
            "--mode", "full",
            "--n-repeats", "10",
            "--n-estimators", "120",]

    scripts = {
        "01": required_script("01_validate_inputs.py"),
        "02": required_script("02_build_modeling_dataset.py"),
        "03": required_script("03_train_probability_baseline.py"),
        "04": required_script("04_train_pair_level_residual_model.py"),
        "05": required_script("05_build_residual_biology_registry.py"),
        "06": required_script("06_train_broad_residual_model.py"),
        "07": required_script("07_train_filtered_per_treatment_residual_models.py"),
        "08": required_script("08_curate_per_treatment_residual_models.py"),
        "09": required_script("09_label_shuffle_validate_tier1.py"),
        "10": required_script("10_build_integrated_interpretation_package.py"),
        "11": required_script("11_make_publication_tables.py"),
        "12": required_script("12_qc_v2_outputs.py"),
    }

    results = []

    results.append(run_step("01_validate_inputs", [
        py, str(scripts["01"]),
        "--handoff-root", str(handoff_root),
        "--output-root", str(step01),
    ], log_root))

    results.append(run_step("02_build_modeling_dataset", [
        py, str(scripts["02"]),
        "--handoff-root", str(handoff_root),
        "--output-root", str(step02),
        "--residual-col", "fused_residual_vs_prior",
    ], log_root))

    results.append(run_step("03_train_probability_baseline", [
        py, str(scripts["03"]),
        "--dataset-root", str(step02),
        "--output-root", str(step03),
        "--mode", args.mode,
        "--target-col", "fused_prob_responder",
        "--n-repeats", "3" if args.mode == "smoke" else "5",
        "--n-estimators", "150",
    ], log_root))

    results.append(run_step("04_train_pair_level_residual_model", [
        py, str(scripts["04"]),
        "--dataset-root", str(step02),
        "--output-root", str(step04),
        "--target-col", "fused_residual_vs_prior",
        *step04_mode_args,
    ], log_root))

    results.append(run_step("05_build_residual_biology_registry", [
        py, str(scripts["05"]),
        "--run-root", str(run_root),
        "--dataset-root", str(step02),
        "--step04-root", str(step04),
        "--output-root", str(step05),
    ], log_root))

    results.append(run_step("06_train_broad_residual_model", [
        py, str(scripts["06"]),
        "--run-root", str(run_root),
        "--dataset-root", str(step02),
        "--step05-root", str(step05),
        "--output-root", str(step06),
        *step06_mode_args,
    ], log_root))

    results.append(run_step("07_train_filtered_per_treatment_residual_models", [
        py, str(scripts["07"]),
        "--run-root", str(run_root),
        "--dataset-root", str(step02),
        "--step05-root", str(step05),
        "--step06-root", str(step06),
        "--output-root", str(step07),
        "--target-col", "fused_residual_vs_prior",
        *step07_mode_args,
    ], log_root))

    results.append(run_step("08_curate_per_treatment_residual_models", [
        py, str(scripts["08"]),
        "--run-root", str(run_root),
        "--dataset-root", str(step02),
        "--step05-root", str(step05),
        "--step06-root", str(step06),
        "--step07-root", str(step07),
        "--output-root", str(step08),
    ], log_root))

    results.append(run_step("09_label_shuffle_validate_tier1", [
        py, str(scripts["09"]),
        "--run-root", str(run_root),
        "--dataset-root", str(step02),
        "--step05-root", str(step05),
        "--step08-root", str(step08),
        "--output-root", str(step09),
        "--target-col", "fused_residual_vs_prior",
        "--n-shuffles", str(n_shuffles),
        "--n-repeats", str(n_repeats_09),
        "--max-workers", str(args.max_workers),
    ], log_root))

    results.append(run_step("10_build_integrated_interpretation_package", [
        py, str(scripts["10"]),
        "--run-root", str(run_root),
        "--step02-root", str(step02),
        "--step03-root", str(step03),
        "--step04-root", str(step04),
        "--step05-root", str(step05),
        "--step06-root", str(step06),
        "--step07-root", str(step07),
        "--step08-root", str(step08),
        "--step09-root", str(step09),
        "--output-root", str(step10),
        "--open-output",
    ], log_root))

    results.append(run_step("11_make_publication_tables", [
        py, str(scripts["11"]),
        "--step10-root", str(step10),
        "--output-root", str(step11),
        "--open-output",
    ], log_root))

    results.append(run_step("12_qc_v2_outputs", [
        py, str(scripts["12"]),
        "--run-root", str(run_root),
        "--output-root", str(step12),
        "--open-output",
    ], log_root))

    summary = {
        "status": "pass" if all(r["return_code"] == 0 for r in results) else "fail",
        "mode": args.mode,
        "run_root": str(run_root),
        "handoff_root": str(handoff_root),
        "log_root": str(log_root),
        "python": py,
        "max_workers_requested": int(args.max_workers),
        "max_workers_policy": "step09_auto" if int(args.max_workers) <= 0 else "step09_user_requested",
        "step09_n_shuffles": int(n_shuffles),
        "step09_n_repeats": int(n_repeats_09),
        "production_dependency_on_v1_outputs": "no",
        "canonical_v1_scripts_modified": "no",
        "steps": results,
    }

    write_json(run_root / "v2_orchestrator_run_summary.json", summary)

    rows = []
    for r in results:
        rows.append(
            "\t".join([
                r["step_name"],
                r["status"],
                str(r["return_code"]),
                f"{r['elapsed_seconds']:.2f}",
                r["stdout_log"],
                r["stderr_log"],
            ])
        )

    manifest_body = [
        "step_name\tstatus\treturn_code\telapsed_seconds\tstdout_log\tstderr_log",
        *rows,
    ]
    (run_root / "v2_orchestrator_step_manifest.tsv").write_text("\n".join(manifest_body) + "\n", encoding="utf-8")

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 ORCHESTRATOR REPORT")
    report_lines.append("")
    for key in [
        "status",
        "mode",
        "run_root",
        "handoff_root",
        "log_root",
        "python",
        "max_workers_requested",
        "max_workers_policy",
        "step09_n_shuffles",
        "step09_n_repeats",
        "production_dependency_on_v1_outputs",
        "canonical_v1_scripts_modified",
    ]:
        report_lines.append(f"{key}: {summary[key]}")
    report_lines.append("")
    report_lines.append("Step results")
    report_lines.extend(manifest_body)

    report_path = write_text_report(run_root / "v2_orchestrator_report.txt", "\n".join(report_lines))

    print("")
    print(terminal_block("SPATIAL PREDICTION MODEL V2 ORCHESTRATOR COMPLETE", [
        f"Status: {summary['status']}",
        f"Mode: {args.mode}",
        f"Run root: {run_root}",
        f"Log root: {log_root}",
        f"Report: {report_path}",
        f"Step 10 package: {step10}",
        f"Step 11 publication package: {step11}",
        f"Step 12 QC package: {step12}",
        f"Step 09 workers requested: {args.max_workers}",
        f"Step 09 worker policy: {summary['max_workers_policy']}",
        "Production dependency on V1 outputs: no",
        "Canonical V1 scripts modified: no",
    ]))
    print("")

    if args.open_output:
        open_path(run_root)
        open_path(step10)
        open_path(step11)
        open_path(step12)
        open_path(log_root)

    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

