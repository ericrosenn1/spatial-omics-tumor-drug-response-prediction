"""
Script: 04_tile_slides.py

Purpose:
    Tile locally available SVS whole-slide images into tissue-containing image tiles.

Pipeline role:
    Step 04 of histology_response_model_v2. This step opens SVS files, samples
    tile coordinates at the configured pyramid level, filters non-tissue tiles,
    writes tile image files, and records tile/status manifests for downstream QC
    and model training.

Scientific context:
    Tiling converts patient-linked whole-slide images into model-readable image
    patches while retaining patient, case, slide, and coordinate provenance. The
    tissue filter is intentionally simple and auditable so failed or low-content
    tiles can be traced in later QC steps.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

from histology_model_v2_lib import load_yaml, output_root, ensure_dir, resolve_path, read_table



# =============================================================================
# Tile filtering and writing helpers
# =============================================================================

def is_tissue_tile(pil_img, white_threshold, min_tissue_fraction):
    """Return whether an RGB tile contains enough non-white tissue signal."""
    arr = np.array(pil_img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False

    rgb = arr[:, :, :3]
    non_white = np.any(rgb < white_threshold, axis=2)
    return float(non_white.mean()) >= float(min_tissue_fraction)


def save_tile(tile, out_path, image_format, jpg_quality):
    """Write a tile image using the configured image format and JPEG quality."""
    tile = tile.convert("RGB")
    if image_format.lower() in ["jpg", "jpeg"]:
        tile.save(out_path, quality=int(jpg_quality))
    else:
        tile.save(out_path)



# =============================================================================
# Per-slide tiling worker
# =============================================================================

def tile_one_slide(task):
    """Tile one SVS slide and return manifest rows plus tiling status."""
    import openslide

    row = task["row"]
    tile_cfg = task["tile_cfg"]
    tile_root = Path(task["tile_root"])
    slide_number = task["slide_number"]
    total_slides = task["total_slides"]

    svs_path = Path(str(row["svs_path"]))
    patient_id = str(row["patient_id"])
    slide_id = str(row["slide_id"])
    case_id = str(row.get("case_id", ""))

    ext = str(tile_cfg.get("image_format", "jpg")).lower()
    slide_out = tile_root / patient_id / slide_id
    slide_out.mkdir(parents=True, exist_ok=True)

    # Existing tile files allow interrupted tiling runs to resume without regenerating completed slide folders.
    existing = list(slide_out.glob(f"*.{ext}"))

    if bool(tile_cfg.get("skip_if_done", True)) and existing:
        manifest_rows = []
        for tile_path in existing:
            manifest_rows.append([
                patient_id,
                case_id,
                slide_id,
                str(svs_path),
                str(tile_path),
                "",
                "",
                tile_cfg["level"],
                tile_cfg["tile_size"],
            ])

        return {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "status": "skipped_existing",
            "tiles_saved": len(existing),
            "tiles_checked": 0,
            "manifest_rows": manifest_rows,
            "message": f"{slide_number}/{total_slides} {patient_id} {slide_id} skipped_existing={len(existing)}",
        }

    try:
        # Per-slide seeded shuffling makes tile sampling reproducible while avoiding systematic coordinate order.
        random.seed(int(tile_cfg.get("random_seed", 42)) + slide_number)

        slide = openslide.OpenSlide(str(svs_path))

        level = int(tile_cfg["level"])
        tile_size = int(tile_cfg["tile_size"])
        stride = int(tile_cfg["stride"])
        width, height = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])

        coords = [
            (x, y)
            for y in range(0, height - tile_size + 1, stride)
            for x in range(0, width - tile_size + 1, stride)
        ]

        random.shuffle(coords)

        saved = 0
        checked = 0
        manifest_rows = []

        max_tiles = tile_cfg.get("max_tiles_per_slide")
        max_tiles = int(max_tiles) if max_tiles is not None else None

        for x, y in coords:
            if max_tiles is not None and saved >= max_tiles:
                break

            checked += 1

            # OpenSlide reads level-0 coordinates, so downsampled tile coordinates are converted back here.
            x0 = int(round(x * downsample))
            y0 = int(round(y * downsample))

            tile = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")

            # The tissue filter removes low-content tiles before disk write and manifest inclusion.
            if not is_tissue_tile(tile, tile_cfg["white_threshold"], tile_cfg["min_tissue_fraction"]):
                continue

            tile_name = f"{slide_id}_x{x}_y{y}.{ext}"
            tile_path = slide_out / tile_name

            save_tile(tile, tile_path, ext, tile_cfg.get("jpg_quality", 95))

            manifest_rows.append([
                patient_id,
                case_id,
                slide_id,
                str(svs_path),
                str(tile_path),
                x,
                y,
                level,
                tile_size,
            ])

            saved += 1

        slide.close()

        return {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "status": "ok",
            "tiles_saved": saved,
            "tiles_checked": checked,
            "manifest_rows": manifest_rows,
            "message": f"{slide_number}/{total_slides} {patient_id} {slide_id} saved={saved} checked={checked}",
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "status": f"failed: {e}",
            "tiles_saved": 0,
            "tiles_checked": 0,
            "manifest_rows": [],
            "message": f"FAILED {slide_number}/{total_slides} {patient_id} {slide_id}: {e}",
        }



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    try:
        import openslide
    except Exception as e:
        raise RuntimeError(f"openslide is not installed or could not be imported: {e}")

    out = ensure_dir(output_root(cfg) / "04_tiles")
    tile_cfg = dict(cfg["Tiling"])
    tile_root = ensure_dir(resolve_path(cfg, cfg["paths"]["tile_dir"]))

    slide_manifest = read_table(output_root(cfg) / "03_slide_manifest" / "slide_manifest_with_labels.tsv", sep="\t")

    if slide_manifest.empty:
        raise RuntimeError("slide_manifest_with_labels.tsv is empty")

    workers = int(os.environ.get("HISTOLOGY_TILE_WORKERS", tile_cfg.get("workers", 1)))
    workers = max(1, workers)

    manifest_path = out / "tile_manifest.tsv"
    status_path = out / "tile_status.tsv"

    print(f"Slides: {len(slide_manifest)}")
    print(f"Tile root: {tile_root}")
    print(f"Workers: {workers}")
    print(f"Tile size: {tile_cfg['tile_size']}")
    print(f"Level: {tile_cfg['level']}")
    print(f"Min tissue fraction: {tile_cfg['min_tissue_fraction']}")
    print(f"Max tiles per slide: {tile_cfg.get('max_tiles_per_slide')}")

    tasks = []
    total_slides = len(slide_manifest)

    for i, (_, row) in enumerate(slide_manifest.iterrows(), start=1):
        tasks.append({
            "row": row.to_dict(),
            "tile_cfg": tile_cfg,
            "tile_root": str(tile_root),
            "slide_number": i,
            "total_slides": total_slides,
        })

    all_manifest_rows = []
    status_rows = []

    start = time.time()

    if workers == 1:
        for task in tasks:
            result = tile_one_slide(task)
            print(result["message"], flush=True)
            status_rows.append({
                "patient_id": result["patient_id"],
                "slide_id": result["slide_id"],
                "status": result["status"],
                "tiles_saved": result["tiles_saved"],
                "tiles_checked": result["tiles_checked"],
            })
            all_manifest_rows.extend(result["manifest_rows"])
    else:
        # Parallel tiling is optional and controlled by the configured or environment-provided worker count.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(tile_one_slide, task) for task in tasks]

            done = 0

            for fut in as_completed(futures):
                result = fut.result()
                done += 1

                print(f"[{done}/{total_slides}] {result['message']}", flush=True)

                status_rows.append({
                    "patient_id": result["patient_id"],
                    "slide_id": result["slide_id"],
                    "status": result["status"],
                    "tiles_saved": result["tiles_saved"],
                    "tiles_checked": result["tiles_checked"],
                })

                all_manifest_rows.extend(result["manifest_rows"])

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "patient_id",
            "case_id",
            "slide_id",
            "svs_path",
            "tile_path",
            "x",
            "y",
            "level",
            "tile_size",
        ])
        writer.writerows(all_manifest_rows)

    status = pd.DataFrame(status_rows)
    status.to_csv(status_path, sep="\t", index=False)

    elapsed = time.time() - start

    summary = [
        "Parallel tile slides summary",
        f"slides_requested: {len(slide_manifest)}",
        f"workers: {workers}",
        f"tile_root: {tile_root}",
        f"tile_manifest: {manifest_path}",
        f"tile_status: {status_path}",
        f"tile_rows: {len(all_manifest_rows)}",
        f"elapsed_minutes: {elapsed / 60:.2f}",
        "",
        "status_counts:",
        status["status"].value_counts(dropna=False).to_string(),
        "",
        "tiles_saved_summary:",
        status["tiles_saved"].describe().to_string(),
    ]

    (out / "tile_slides_parallel_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("DONE")
    print(manifest_path)
    print(status_path)
    print(f"Elapsed minutes: {elapsed / 60:.2f}")


if __name__ == "__main__":
    main()
