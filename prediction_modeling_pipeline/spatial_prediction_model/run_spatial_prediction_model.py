"""
Script:
    run_spatial_prediction_model.py

Purpose:
    Run spatial_prediction_model steps in order.

Notes:
    YAML-driven runner.
    Logs every step under output_root/pipeline_run_logs/run_<timestamp>.
    Step scripts are scaffold stubs until implemented.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import subprocess
import sys
from typing import Iterable

import yaml


STEPS = [
    ("01", "01_validate_prediction_inputs", "scripts/01_validate_prediction_inputs.py"),
    ("02", "02_build_spatial_modeling_dataset", "scripts/02_build_spatial_modeling_dataset.py"),
    ("03", "03_train_global_spatial_response_model", "scripts/03_train_global_spatial_response_model.py"),
    ("04", "04_train_per_treatment_models", "scripts/04_train_per_treatment_models.py"),
    ("05", "05_explain_spatial_response_model", "scripts/05_explain_spatial_response_model.py"),
    ("06", "06_predict_all_sample_treatment_pairs", "scripts/06_predict_all_sample_treatment_pairs.py"),
    ("07", "07_qc_spatial_prediction_outputs", "scripts/07_qc_spatial_prediction_outputs.py"),
]


def parse_args() -> argparse.Namespace:
    """parse CLI args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spatial_prediction_model.yaml")
    parser.add_argument(
        "--steps",
        default="01,02,03,04,05,06,07",
        help="Comma-separated step numbers, e.g. 01,02,03",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """load YAML config"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_steps(step_text: str) -> set[str]:
    """normalize selected step numbers"""
    return {part.strip().zfill(2) for part in step_text.split(",") if part.strip()}


def should_run_step(cfg: dict, step_name: str) -> bool:
    """check run_steps config flag"""
    toggles = cfg.get("run_steps", {}) or {}
    return bool(toggles.get(step_name, True))


def run_step(root: Path, config_path: Path, log_dir: Path, step_num: str, step_name: str, script_rel: str) -> None:
    """run one pipeline step and save log"""
    script_path = root / script_rel
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    log_path = log_dir / f"step_{step_num}_{step_name}.log"
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]

    print(f"\n[{step_num}] {step_name}")
    print(" ".join(cmd))
    print(f"log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {step_num} {step_name}; see {log_path}")


def main() -> None:
    """run selected steps"""
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_config(config_path)
    output_root = Path(cfg["output_root"])
    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_root / "pipeline_run_logs" / f"run_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = selected_steps(args.steps)

    print("spatial_prediction_model runner")
    print(f"root: {root}")
    print(f"config: {config_path}")
    print(f"output_root: {output_root}")
    print(f"steps: {sorted(wanted)}")

    for step_num, step_name, script_rel in STEPS:
        if step_num not in wanted:
            print(f"skip {step_num}: not selected")
            continue

        if not should_run_step(cfg, step_name):
            print(f"skip {step_num}: disabled in YAML run_steps.{step_name}")
            continue

        run_step(root, config_path, log_dir, step_num, step_name, script_rel)

    print("\nDONE")
    print(f"logs: {log_dir}")


if __name__ == "__main__":
    main()