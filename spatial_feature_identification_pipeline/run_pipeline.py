"""
Main command line entry point for the spatial feature identification pipeline.

This orchestrator can dry run or execute selected numbered scripts. It does not
change scientific logic inside the step scripts.
"""


# =========================
# Imports
# =========================

from pathlib import Path
import argparse
import os
import subprocess
import sys


# =========================
# Project path setup
# =========================

CODE_ROOT = Path(__file__).resolve().parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lib.config import load_config, validate_config



# =========================
# Pipeline step registry
# =========================

STEPS = [
    ("01", "01_validate_inputs.py"),
    ("02", "02_process_samples.py"),
    ("03", "03_merge_slide_features.py"),
    ("04", "04_score_and_label_slides.py"),
    ("05", "05_build_multi_axis_transcriptome_labels.py"),
    ("06", "06_build_accessibility_profiles.py"),
    ("07", "07_append_hotspot_metrics.py"),
    ("08", "08_add_context_alignment.py"),
    ("09", "09_build_motif_tables.py"),
    ("10", "10_build_model_ready_table.py"),
    ("11", "11_overlay.py"),
    ("12", "12_data_analysis_and_visuals.py"),
]



# =========================
# Argument parsing
# =========================

def parse_args() -> argparse.Namespace:
    """Parse orchestrator command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="01")
    parser.add_argument("--end", default="12")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()



# =========================
# Step selection
# =========================

def selected_steps(start: str, end: str) -> list[tuple[str, str]]:
    """Return inclusive step range."""
    labels = [s for s, _ in STEPS]
    if start not in labels or end not in labels:
        raise ValueError("Start and end must be between 01 and 12.")
    i = labels.index(start)
    j = labels.index(end)
    if i > j:
        raise ValueError("Start step must be before or equal to end step.")
    return STEPS[i:j + 1]



# =========================
# Command line workflow
# =========================

def main() -> None:
    """Run or preview selected pipeline steps."""
    args = parse_args()
    cfg = validate_config(load_config(args.config))

    output_root = Path(cfg["output_root"])
    chosen = selected_steps(args.start, args.end)

    print("")
    print("============================================================")
    print("Spatial feature identification pipeline runner")
    print("============================================================")
    print("Config:", cfg["config_path"])
    print("Output root:", output_root)
    print("Dry run:", args.dry_run)
    print("Selected steps:", ", ".join(s for s, _ in chosen))
    print("")

    results = []

    for step, script_name in chosen:
        script = CODE_ROOT / script_name
        cmd = [sys.executable, str(script), "--config", str(cfg["config_path"])]
        print(f"Step {step}: {script_name}")
        print("  command:", " ".join(cmd))

        if args.dry_run:
            results.append((step, script_name, "dry_run"))
            continue

        completed = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
        status = "ok" if completed.returncode == 0 else f"failed_returncode_{completed.returncode}"
        results.append((step, script_name, status))

        if completed.returncode != 0:
            break

    print("")
    print("============================================================")
    print("Run summary")
    print("============================================================")
    for step, script_name, status in results:
        print(f"{step} {script_name}: {status}")

    print("")
    print("Output root:", output_root)
    print("DONE")

    if args.open and os.name == "nt":
        os.startfile(str(output_root))


if __name__ == "__main__":
    main()
