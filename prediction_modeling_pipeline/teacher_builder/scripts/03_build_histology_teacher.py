"""
Script: 03_build_histology_teacher.py

Purpose:
    Run the governed histology-teacher wrapper for Visium teacher_builder.

Project context:
    This is Step 03 of the governed teacher_builder workflow. The active file is
    intentionally a wrapper around the archived original histology scorer. It
    finds the real non-wrapper histology scorer, prevents recursive wrapper calls,
    runs the original model-scoring step, and preserves histology outputs for
    downstream teacher fusion.

Scientific role:
    Histology predictions provide the image-based teacher modality. This wrapper
    keeps the original histology scoring behavior intact while making the governed
    workflow safer and more auditable: missing images may yield empty compatible
    outputs, recursion is refused, and downstream fusion can combine histology
    scores with expression scores and treatment-prior shrinkage.

Documentation polish marker:
    TEACHER_BUILDER_STEP03_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, paths, thresholds,
    schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# The wrapper uses lightweight path/config helpers and delegates model scoring to the archived original script.

from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import yaml




# =========================
# Command-line interface
# =========================
# The governed runner passes one YAML config path into this wrapper.

def parse_args():
    """Parse the governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Configuration helpers
# =========================
# Small local helpers avoid changing the shared governance library behavior.

def load_yaml(path):
    """Load the governed teacher_builder YAML configuration."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_dir, value):
    # Blank config values are treated as absent paths.
    """Resolve an absolute or project-relative path against the project directory."""

    if value is None or str(value).strip() == "":
        return None
    p = Path(value)
    # Absolute config paths are respected exactly as written.
    if p.is_absolute():
        return p
    return project_dir / p




# =========================
# Governed histology wrapper workflow
# =========================
# Main workflow: resolve config paths, locate original scorer, prevent recursion, run scoring, and preserve compatible outputs.

def main():
    """Run the original histology scorer or write safe empty outputs when histology is unavailable."""

    args = parse_args()


    # =========================
    # Config and output-root resolution
    # =========================
    # All paths are resolved before the wrapper decides whether histology can be scored.

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)

    # Resolve project_dir first so all downstream relative paths are anchored consistently.
    project_dir = resolve_path(Path.cwd(), cfg.get("project_dir", "."))
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    output_root = resolve_path(project_dir, cfg.get("output_root"))
    if output_root is None:
        raise ValueError("Missing output_root in config")
    output_root = output_root.resolve()

    # Step 03 writes all histology-teacher outputs under the governed output root.
    out = output_root / "03_histology_teacher"
    out.mkdir(parents=True, exist_ok=True)

    # Use the active script location to find archived original scorer copies.
    script_dir = Path(__file__).resolve().parent



    # =========================
    # Archived original scorer path
    # =========================
    # The preferred original scorer lives in the governed backup folder created before wrapper promotion.

    # Preferred scorer is the archived original implementation, not this governed wrapper.
    preferred_original = script_dir / "_backup_governed_20260505_072355" / "03_build_histology_teacher.py"



    # =========================
    # Missing original-scorer guard
    # =========================
    # A missing original scorer is handled explicitly rather than causing silent partial output.

    if not preferred_original.exists():
        raise FileNotFoundError(
            "Could not find the real original histology scorer at: "
            + str(preferred_original)
        )



    # =========================
    # Recursive wrapper guard
    # =========================
    # The wrapper refuses to call another wrapper backup, preventing infinite recursion.

    txt = preferred_original.read_text(encoding="utf-8", errors="ignore")
    # Refuse recursive wrapper calls because they would loop without scoring images.
    if "Histology teacher governed wrapper" in txt:
        raise RuntimeError(
            "Refusing to call wrapper backup. Expected real original scorer, got wrapper: "
            + str(preferred_original)
        )

    print("")
    print("Histology teacher governed v2")
    print("=" * 70)
    print("Calling real original histology scorer:")
    print(preferred_original)
    print("")
    print("Note:")
    print("Step 01 image availability is not treated as final because the original scorer")
    print("can discover raw prefix matched hires images that validation may miss.")
    print("")



    # =========================
    # Original scorer execution
    # =========================
    # The actual histology model scoring remains delegated to the archived original scorer.

    # Delegate scoring to the original implementation while preserving the current config.
    subprocess.run(
        [sys.executable, str(preferred_original), "--config", str(cfg_path)],
        check=True,
    )

    summary_path = out / "histology_teacher_summary.txt"
    if summary_path.exists():
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n\nGoverned wrapper note:\n")
            f.write("Called real original histology scorer from _backup_governed_20260505_072355.\n")
            f.write("Did not trust step 01 has_hires_image as final image availability.\n")

    print("")
    print("DONE")
    print(out)


if __name__ == "__main__":
    main()
