from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def download_file(url: str, path: Path, retries: int = 3) -> bool:
    ensure_dir(path.parent)

    if path.exists() and path.stat().st_size > 0:
        print(f"SKIP existing: {path}")
        return True

    if not url:
        print(f"NO URL: {path}")
        return False

    for attempt in range(1, retries + 1):
        try:
            print(f"DOWNLOAD attempt {attempt}: {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as response, path.open("wb") as out:
                shutil.copyfileobj(response, out)
            print(f"OK: {path}")
            return True
        except Exception as exc:
            print(f"FAILED attempt {attempt}: {url}")
            print(exc)
            time.sleep(2 * attempt)

    return False


def maybe_gunzip(src: Path, dst: Path) -> bool:
    ensure_dir(dst.parent)

    if dst.exists() and dst.stat().st_size > 0:
        return True

    if src.suffix.lower() != ".gz":
        return False

    try:
        with gzip.open(src, "rb") as inp, dst.open("wb") as out:
            shutil.copyfileobj(inp, out)
        return True
    except Exception as exc:
        print(f"GUNZIP FAILED: {src} -> {dst}")
        print(exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--visium-root", required=True)
    parser.add_argument("--source-manifest", default="data_manifest/visium_public_source_reconstruction_manifest.tsv")
    parser.add_argument("--expected-manifest", default="data_manifest/visium_expected_cohort_files.tsv")
    parser.add_argument("--download-raw", action="store_true")
    parser.add_argument("--reconstruct-cohort", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    visium_root = Path(args.visium_root)
    source_manifest = repo_root / args.source_manifest
    expected_manifest = repo_root / args.expected_manifest

    if not source_manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")
    if not expected_manifest.exists():
        raise FileNotFoundError(f"Expected manifest not found: {expected_manifest}")

    source_rows = read_tsv(source_manifest)
    expected_rows = read_tsv(expected_manifest)

    print("")
    print("PUBLIC VISIUM SOURCE RECONSTRUCTION")
    print("=" * 80)
    print(f"Repo root: {repo_root}")
    print(f"Visium root: {visium_root}")
    print(f"Source manifest: {source_manifest}")
    print(f"Expected manifest: {expected_manifest}")
    print(f"Raw source rows: {len(source_rows)}")
    print(f"Expected cohort rows: {len(expected_rows)}")
    print(f"Dry run: {args.dry_run}")
    print("")

    downloaded = 0
    download_failed = 0

    if args.download_raw:
        for row in source_rows:
            rel = row.get("local_raw_relative_path", "")
            url = row.get("source_url", "")
            if not rel:
                continue

            dst = visium_root / rel

            if args.dry_run:
                print(f"DRY DOWNLOAD: {url} -> {dst}")
                continue

            ok = download_file(url, dst)
            if ok:
                downloaded += 1
            else:
                download_failed += 1

    reconstructed = 0
    reconstruction_skipped = 0
    reconstruction_failed = 0

    if args.reconstruct_cohort:
        for row in expected_rows:
            method = row.get("unpack_method", "")
            dst_rel = row.get("destination_relative_path", "")
            raw_rel = row.get("local_raw_relative_path", "")

            if not dst_rel:
                continue

            dst = visium_root / dst_rel

            if method == "copy_from_downloaded_raw" and raw_rel:
                src = visium_root / raw_rel

                if args.dry_run:
                    print(f"DRY COPY: {src} -> {dst}")
                    continue

                if not src.exists():
                    print(f"MISSING RAW SOURCE: {src}")
                    reconstruction_failed += 1
                    continue

                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
                reconstructed += 1

            elif method == "gunzip_from_downloaded_raw" and raw_rel:
                src = visium_root / raw_rel

                if args.dry_run:
                    print(f"DRY GUNZIP: {src} -> {dst}")
                    continue

                if maybe_gunzip(src, dst):
                    reconstructed += 1
                else:
                    reconstruction_failed += 1

            elif method == "direct_download_to_expected_path":
                url = row.get("source_url", "")

                if args.dry_run:
                    print(f"DRY DIRECT DOWNLOAD: {url} -> {dst}")
                    continue

                ok = download_file(url, dst)
                if ok:
                    reconstructed += 1
                else:
                    reconstruction_failed += 1

            else:
                reconstruction_skipped += 1

    print("")
    print("SUMMARY")
    print("=" * 80)
    print(f"downloaded: {downloaded}")
    print(f"download_failed: {download_failed}")
    print(f"reconstructed: {reconstructed}")
    print(f"reconstruction_skipped_needs_manual_mapping: {reconstruction_skipped}")
    print(f"reconstruction_failed: {reconstruction_failed}")

    if download_failed or reconstruction_failed:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

