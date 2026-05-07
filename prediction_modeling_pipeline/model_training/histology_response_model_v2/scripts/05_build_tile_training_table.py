"""
Script: 05_build_tile_training_table.py

Purpose:
    Build the artifact-filtered tile-level training table.

Pipeline role:
    Step 05 of histology_response_model_v2. This step joins tile rows to strict
    case labels, confirms tile files exist, computes tile artifact metrics,
    excludes flagged tiles, and writes tile- and patient-level model training
    tables.

Scientific context:
    This step prevents obvious read failures, blank tiles, dark artifacts, and
    low-information image regions from entering the histology response model.
    Excluded artifact rows and slide summaries are retained as audit artifacts
    rather than silently discarded.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

from histology_model_v2_lib import load_yaml, output_root, ensure_dir, read_table



# =============================================================================
# Artifact metric helpers
# =============================================================================

def metric_one(path):
    """Compute lightweight artifact metrics for one tile image."""
    p = Path(str(path))

    try:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        rgb = arr[:, :, :3]
        gray = rgb.mean(axis=2)

        nonwhite = np.any(rgb < 220, axis=2)
        tissue_fraction = float(nonwhite.mean())
        brightness = float(gray.mean())

        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)

        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(mx > 0, (mx - mn) / mx, 0.0)

        saturation = float(np.mean(sat))

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        sharpness = float(np.var(gx) + np.var(gy))

        red = float(rgb[:, :, 0].mean())
        green = float(rgb[:, :, 1].mean())
        blue = float(rgb[:, :, 2].mean())
        color_range = float(max(red, green, blue) - min(red, green, blue))

        return [
            str(p),
            True,
            tissue_fraction,
            brightness,
            saturation,
            sharpness,
            color_range,
            "",
        ]

    except Exception as e:
        return [
            str(p),
            False,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            str(e),
        ]


def compute_or_load_artifact_metrics(tile_paths, qc_dir, workers):
    """Load cached artifact metrics when valid, otherwise compute them in parallel."""
    metrics_path = qc_dir / "all_tile_artifact_metrics.tsv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, sep="\t", dtype={"tile_path": str}, low_memory=False)
        existing = set(metrics["tile_path"].astype(str))
        needed = set(map(str, tile_paths))

        if existing == needed:
            print("Using existing artifact metrics:")
            print(metrics_path)
            return metrics

        print("Existing artifact metrics do not match current tile list. Recomputing.")

    tile_paths = list(map(str, tile_paths))

    rows = []
    done = 0

    print("")
    print("Computing tile artifact metrics")
    print("unique_tile_files:", len(tile_paths))
    print("workers:", workers)

    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(metric_one, tile_paths, chunksize=500):
            rows.append(result)
            done += 1

            if done % 50000 == 0:
                print("processed:", done, flush=True)

    metrics = pd.DataFrame(
        rows,
        columns=[
            "tile_path",
            "read_ok",
            "tissue_fraction",
            "brightness",
            "saturation",
            "sharpness",
            "color_range",
            "error",
        ],
    )

    metrics.to_csv(metrics_path, sep="\t", index=False)

    return metrics


def add_artifact_flags(metrics):
    """Add artifact exclusion flags from color, tissue, brightness, and sharpness metrics."""
    metrics = metrics.copy()

    # Artifact flags are intentionally simple, auditable image statistics rather than learned filters.
    metrics["flag_read_fail"] = ~metrics["read_ok"].astype(bool)

    metrics["flag_blank_or_nearly_blank"] = (
        (metrics["tissue_fraction"] < 0.20)
        | (metrics["brightness"] > 245)
    )

    metrics["flag_dark_gray_artifact"] = (
        (metrics["saturation"] < 0.035)
        & (metrics["color_range"] < 18)
        & (metrics["brightness"].between(40, 235))
    )

    metrics["flag_low_information"] = (
        (metrics["saturation"] < 0.025)
        & (metrics["sharpness"] < 35)
    )

    metrics["flag_extreme_dark"] = metrics["brightness"] < 25

    flag_cols = [
        "flag_read_fail",
        "flag_blank_or_nearly_blank",
        "flag_dark_gray_artifact",
        "flag_low_information",
        "flag_extreme_dark",
    ]

    metrics["artifact_flag_count"] = metrics[flag_cols].sum(axis=1)
    metrics["artifact_recommendation"] = np.where(metrics["artifact_flag_count"] > 0, "exclude", "keep")

    return metrics, flag_cols



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    out = ensure_dir(output_root(cfg) / "05_training_table")
    qc_dir = ensure_dir(output_root(cfg) / "05_training_table" / "artifact_qc")

    labels = read_table(output_root(cfg) / "02_case_labels" / "case_label_table_strict.tsv", sep="\t")
    tiles = read_table(output_root(cfg) / "04_tiles" / "tile_manifest.tsv", sep="\t")

    print("")
    print("Step 05 build tile training table")
    print("=" * 70)
    print("tile_manifest_rows:", len(tiles))
    print("label_rows:", len(labels))

    # Tile rows are joined to strict labels before model training table construction.
    merged = tiles.merge(labels, on=["patient_id", "case_id"], how="left", suffixes=("", "_label"))

    # Tile existence is checked explicitly so stale manifest rows cannot enter training.
    merged["tile_exists"] = merged["tile_path"].astype(str).apply(lambda p: Path(p).exists())
    merged = merged[merged["tile_exists"]].copy()

    merged = merged[merged["binary_response_label"].isin(["RESPONDER", "NON_RESPONDER"])].copy()
    merged = merged[merged["canonical_treatment_key"].astype(str).str.len().gt(0)].copy()

    print("rows_after_label_and_file_filter:", len(merged))

    workers = int(cfg.get("artifact_qc", {}).get("workers", min(16, max(1, os.cpu_count() or 1))))
    apply_artifact_filter = bool(cfg.get("artifact_qc", {}).get("enabled", True))

    if apply_artifact_filter:
        unique_tile_paths = merged["tile_path"].drop_duplicates().astype(str).tolist()

        # Artifact metrics are cached because this operation can be expensive on large tile banks.
        metrics = compute_or_load_artifact_metrics(unique_tile_paths, qc_dir, workers)
        metrics, flag_cols = add_artifact_flags(metrics)

        metrics_path = qc_dir / "all_tile_artifact_metrics.tsv"
        metrics.to_csv(metrics_path, sep="\t", index=False)

        merged = merged.merge(
            metrics[["tile_path", "artifact_recommendation", "artifact_flag_count"] + flag_cols],
            on="tile_path",
            how="left",
        )

        merged["artifact_recommendation"] = merged["artifact_recommendation"].fillna("exclude")

        excluded_artifacts = merged[merged["artifact_recommendation"] == "exclude"].copy()
        included = merged[merged["artifact_recommendation"] == "keep"].copy()

        # Excluded artifact rows are retained as an audit table rather than silently discarded.
        excluded_artifacts.to_csv(qc_dir / "excluded_artifact_tiles.tsv", sep="\t", index=False)

        slide_artifact_summary = (
            merged
            .groupby(["patient_id", "slide_id"], as_index=False)
            .agg(
                tiles=("tile_path", "size"),
                excluded_tiles=("artifact_recommendation", lambda s: int((s == "exclude").sum())),
            )
        )

        slide_artifact_summary["excluded_fraction"] = slide_artifact_summary["excluded_tiles"] / slide_artifact_summary["tiles"]
        slide_artifact_summary.to_csv(qc_dir / "artifact_filter_slide_summary.tsv", sep="\t", index=False)

        merged = included.copy()

        print("")
        print("Artifact filter summary")
        print("kept_tile_rows:", len(included))
        print("excluded_artifact_tile_rows:", len(excluded_artifacts))
        print("patients_with_any_artifact_removed:", excluded_artifacts["patient_id"].nunique())
        print("slides_with_any_artifact_removed:", excluded_artifacts["slide_id"].nunique())
        print("")
        print("artifact recommendation counts:")
        print(metrics["artifact_recommendation"].value_counts(dropna=False).to_string())
        print("")
        print("artifact flag counts:")
        for c in flag_cols:
            print(f"{c}: {int(metrics[c].sum())}")
        print("")
        print("slides with highest excluded fraction:")
        print(
            slide_artifact_summary
            .sort_values(["excluded_fraction", "excluded_tiles"], ascending=False)
            .head(20)
            .to_string(index=False)
        )
    else:
        excluded_artifacts = pd.DataFrame()
        print("Artifact filter disabled")

    merged.to_csv(out / "tile_training_table.tsv", sep="\t", index=False)

    # The patient-level table provides a compact patient manifest for downstream split and QC review.
    patient = merged.drop_duplicates("patient_id").copy()
    patient.to_csv(out / "patient_training_table.tsv", sep="\t", index=False)

    summary = [
        "Tile training table summary",
        f"tile_rows: {len(merged)}",
        f"patients: {merged['patient_id'].nunique()}",
        f"slides: {merged['slide_id'].nunique()}",
        f"treatments: {merged['canonical_treatment_key'].nunique()}",
        f"excluded_artifact_tile_rows: {len(excluded_artifacts)}",
        "response_counts:",
        merged.drop_duplicates("patient_id")["binary_response_label"].value_counts(dropna=False).to_string(),
        "top_treatments_by_patient:",
        merged.drop_duplicates("patient_id")["canonical_treatment_key"].value_counts().head(30).to_string(),
    ]

    summary_text = "\n".join(summary)
    (out / "tile_training_table_summary.txt").write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print("DONE")
    print(out)


if __name__ == "__main__":
    main()
