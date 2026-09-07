"""
Script: 09B_manuscript_1000_shuffle_validation.py

Purpose:
    Run a manuscript-specific robustness validation using 1,000 within-treatment
    label shuffles and 5 repeated train/test splits for the Step 08 Tier 1
    treatment-specific residual models.

Pipeline role:
    This is NOT a canonical production-pipeline stage. It is a manuscript
    robustness analysis that reuses the frozen scientific logic in
    09_label_shuffle_validate_tier1.py without modifying Step 09, Step 08
    curation, model definitions, feature selection, thresholds, or downstream
    production outputs.

Manuscript rationale:
    The canonical full V2 workflow uses 100 label shuffles. That provides a
    minimum empirical p-value of 1 / 101 = 0.00990099. This supplemental run
    increases the null distribution to 1,000 shuffles, giving a minimum
    empirical p-value of 1 / 1001 = 0.000999001 and a more stable estimate of
    the treatment-specific null distribution.

Expected inputs:
    An already completed V2 run containing:
      02_build_modeling_dataset/
      05_residual_biology_registry/
      08_curated_per_treatment_residual_models/

Default output:
    <run-root>/09B_manuscript_1000_shuffle_validation/

The script validates its inputs, invokes the canonical Step 09 implementation,
checks that 1,000 null shuffles were actually produced for every candidate,
compares validation membership with the canonical 100-shuffle Step 09 output
when available, and writes a manuscript-specific metadata JSON and comparison
TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


N_SHUFFLES = 1000
N_REPEATS = 5
CANONICAL_STEP09_DIRNAME = "09_tier1_label_shuffle_validation"
MANUSCRIPT_STEP09B_DIRNAME = "09B_manuscript_1000_shuffle_validation"

SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_STEP09_SCRIPT = SCRIPT_DIR / "09_label_shuffle_validate_tier1.py"


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    require_path(path, "TSV file")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def open_path(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"Could not open output automatically: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run manuscript-specific 1,000-shuffle robustness validation using "
            "the canonical V2 Step 09 implementation."
        )
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Existing completed V2 run root containing Steps 02, 05, and 08.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional isolated output directory. Default: "
            "<run-root>/09B_manuscript_1000_shuffle_validation"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Maximum parallel treatment workers. Use 0 for Step 09 automatic selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing 09B output directory if present.",
    )
    parser.add_argument(
        "--open-output",
        action="store_true",
        help="Open the 09B output directory after successful validation.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    require_path(run_root, "V2 run root")
    require_path(CANONICAL_STEP09_SCRIPT, "canonical Step 09 script")

    dataset_root = require_path(run_root / "02_build_modeling_dataset", "Step 02 output")
    step05_root = require_path(run_root / "05_residual_biology_registry", "Step 05 output")
    step08_root = require_path(run_root / "08_curated_per_treatment_residual_models", "Step 08 output")

    candidate_path = require_path(
        step08_root / "07_label_shuffle_handoff" / "tier1_label_shuffle_candidates.tsv",
        "Step 08 Tier 1 candidate table",
    )
    candidate_rows = read_tsv(candidate_path)
    if not candidate_rows:
        raise RuntimeError(f"Tier 1 candidate table is empty: {candidate_path}")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else run_root / MANUSCRIPT_STEP09B_DIRNAME
    )

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"09B output already exists and is not empty: {output_root}\n"
                "Use --overwrite only if replacement is intentional."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(CANONICAL_STEP09_SCRIPT),
        "--run-root",
        str(run_root),
        "--dataset-root",
        str(dataset_root),
        "--step05-root",
        str(step05_root),
        "--step08-root",
        str(step08_root),
        "--output-root",
        str(output_root),
        "--target-col",
        "fused_residual_vs_prior",
        "--n-shuffles",
        str(N_SHUFFLES),
        "--n-repeats",
        str(N_REPEATS),
        "--max-workers",
        str(args.max_workers),
    ]

    print("=" * 72)
    print("V2 MANUSCRIPT ROBUSTNESS VALIDATION: STEP 09B")
    print("=" * 72)
    print(f"Run root:           {run_root}")
    print(f"Tier 1 candidates:  {len(candidate_rows)}")
    print(f"Label shuffles:     {N_SHUFFLES}")
    print(f"Repeated splits:    {N_REPEATS}")
    print(f"Output root:        {output_root}")
    print("Canonical Step 09 scientific logic is reused unchanged.")
    print("")

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Canonical Step 09 failed with exit code {completed.returncode}")

    results_path = require_path(
        output_root / "04_validation_results" / "tier1_label_shuffle_validation_results.tsv",
        "09B validation results",
    )
    new_rows = read_tsv(results_path)

    if len(new_rows) != len(candidate_rows):
        raise RuntimeError(
            f"Candidate/result count mismatch: {len(candidate_rows)} candidates, "
            f"{len(new_rows)} validation rows."
        )

    shuffle_counts = {
        int(float(row["n_null_shuffles"]))
        for row in new_rows
        if row.get("n_null_shuffles", "") != ""
    }
    if shuffle_counts != {N_SHUFFLES}:
        raise RuntimeError(
            f"Expected exactly {N_SHUFFLES} null shuffles for every treatment, "
            f"observed counts: {sorted(shuffle_counts)}"
        )

    validated_new = [row for row in new_rows if as_bool(row.get("validated_for_step10"))]

    comparison_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []
    canonical_results_path = (
        run_root
        / CANONICAL_STEP09_DIRNAME
        / "04_validation_results"
        / "tier1_label_shuffle_validation_results.tsv"
    )

    old_validated_count: int | None = None
    if canonical_results_path.exists():
        old_rows = read_tsv(canonical_results_path)
        old_by_drug = {str(row.get("drug_key", "")): row for row in old_rows}
        old_validated_count = sum(as_bool(row.get("validated_for_step10")) for row in old_rows)

        for new in new_rows:
            drug_key = str(new.get("drug_key", ""))
            old = old_by_drug.get(drug_key, {})
            old_validated = as_bool(old.get("validated_for_step10"))
            new_validated = as_bool(new.get("validated_for_step10"))

            row = {
                "drug_key": drug_key,
                "canonical_validated": old_validated,
                "manuscript_1000_validated": new_validated,
                "canonical_empirical_p_pearson": old.get("empirical_p_pearson", ""),
                "manuscript_1000_empirical_p_pearson": new.get("empirical_p_pearson", ""),
                "canonical_fdr_q_pearson": old.get("fdr_q_pearson", ""),
                "manuscript_1000_fdr_q_pearson": new.get("fdr_q_pearson", ""),
                "observed_test_pearson_mean": new.get("observed_test_pearson_mean", ""),
                "manuscript_1000_null_q95": new.get("null_test_pearson_mean_q95", ""),
                "n_null_shuffles": new.get("n_null_shuffles", ""),
            }
            comparison_rows.append(row)
            if old_validated != new_validated:
                changed_rows.append(row)

        write_tsv(output_root / "canonical_100_vs_manuscript_1000_comparison.tsv", comparison_rows)

    p_values = [
        float(row["empirical_p_pearson"])
        for row in new_rows
        if row.get("empirical_p_pearson", "") not in {"", None}
    ]
    q_values_validated = [
        float(row["fdr_q_pearson"])
        for row in validated_new
        if row.get("fdr_q_pearson", "") not in {"", None}
    ]

    metadata = {
        "analysis_role": "manuscript_specific_robustness_validation",
        "canonical_pipeline_stage_modified": False,
        "canonical_step09_script_reused": str(CANONICAL_STEP09_SCRIPT),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "target_col": "fused_residual_vs_prior",
        "n_tier1_candidates": len(candidate_rows),
        "n_shuffles": N_SHUFFLES,
        "n_repeats": N_REPEATS,
        "minimum_attainable_empirical_p": 1.0 / (N_SHUFFLES + 1),
        "n_validated_treatments_1000": len(validated_new),
        "n_validated_treatments_canonical": old_validated_count,
        "n_validation_membership_changes": len(changed_rows),
        "minimum_observed_empirical_p_pearson": min(p_values) if p_values else None,
        "maximum_fdr_q_pearson_among_validated": max(q_values_validated) if q_values_validated else None,
        "comparison_file": (
            str(output_root / "canonical_100_vs_manuscript_1000_comparison.tsv")
            if comparison_rows
            else None
        ),
    }

    metadata_path = output_root / "manuscript_1000_shuffle_validation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("")
    print("=" * 72)
    print("STEP 09B MANUSCRIPT VALIDATION: SUCCESS")
    print("=" * 72)
    print(f"Tier 1 candidates tested:       {len(candidate_rows)}")
    print(f"Null shuffles per treatment:    {N_SHUFFLES}")
    print(f"Repeated splits per shuffle:    {N_REPEATS}")
    print(f"Validated at 1,000 shuffles:    {len(validated_new)}")
    if old_validated_count is not None:
        print(f"Validated in canonical Step 09: {old_validated_count}")
        print(f"Membership changes:             {len(changed_rows)}")
    if p_values:
        print(f"Minimum empirical p, Pearson:   {min(p_values):.6g}")
    if q_values_validated:
        print(f"Largest q among validated:      {max(q_values_validated):.6g}")
    print(f"Metadata:                       {metadata_path}")
    print(f"Output root:                    {output_root}")
    print("=" * 72)

    if changed_rows:
        print("")
        print("Validation membership changed for:")
        for row in changed_rows:
            print(
                f"  {row['drug_key']}: "
                f"canonical={row['canonical_validated']} "
                f"1000={row['manuscript_1000_validated']}"
            )

    if args.open_output:
        open_path(output_root)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("")
        print("STEP 09B MANUSCRIPT VALIDATION: FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
