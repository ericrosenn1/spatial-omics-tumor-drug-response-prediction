from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MANIFEST = "data_manifest/public_visium_cohort_staging_manifest.tsv"
SCRIPT_NAME = "scripts/download_and_reconstruct_public_visium_sources.py"
ZENODO_RECORD_API = "https://zenodo.org/api/records/14620362"
TLS_SOURCE_TYPE = "zenodo_tls_archive"
TLS_ZIP_NAME = "TLS_VISIUM_USZ.zip"
TLS_ZIP_RELATIVE_PATH = Path("raw_visium_new") / "TLS_VISIUM_USZ" / TLS_ZIP_NAME
TLS_EXTRACT_RELATIVE_ROOT = Path("raw_visium_new") / "TLS_VISIUM_USZ"

MANIFEST_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_accession_gse",
    "source_accession_gsm",
    "source_type",
    "source_sample_label",
    "file_role",
    "source_url",
    "source_download_url",
    "source_file_name",
    "source_archive_name",
    "source_archive_type",
    "raw_cache_relative_path",
    "cohort_relative_path",
    "archive_member_relative_path",
    "required_for_step01",
    "notes",
]

INVENTORY_COLUMNS = [
    "sample_id",
    "file_role",
    "source_type",
    "source_url",
    "raw_cache_path",
    "cohort_path",
    "status",
    "bytes_written_or_existing",
    "checksum_status",
    "notes",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def public_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def manifest_path(root: Path, relative: str) -> Path:
    return root / Path(str(relative).replace("\\", "/"))


def source_url(row: dict[str, str]) -> str:
    return row.get("source_download_url") or row.get("source_url", "")


def download_file(url: str, path: Path, overwrite: bool = False, retries: int = 3) -> tuple[bool, str]:
    ensure_dir(path.parent)

    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return True, "existing"

    if not url:
        return False, "missing_source_url"

    part_path = path.with_name(path.name + ".part")
    if part_path.exists():
        part_path.unlink()

    for attempt in range(1, retries + 1):
        try:
            print(f"DOWNLOAD attempt {attempt}: {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as response, part_path.open("wb") as out:
                shutil.copyfileobj(response, out)
            if part_path.stat().st_size <= 0:
                part_path.unlink(missing_ok=True)
                return False, "download_empty"
            part_path.replace(path)
            return True, "downloaded"
        except Exception as exc:
            print(f"FAILED attempt {attempt}: {url}")
            print(exc)
            part_path.unlink(missing_ok=True)
            time.sleep(2 * attempt)

    return False, "download_failed"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_zenodo_file_metadata() -> dict[str, str]:
    req = Request(ZENODO_RECORD_API, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    for file_info in payload.get("files", []):
        key = file_info.get("key", "")
        if key == TLS_ZIP_NAME or key.endswith("/" + TLS_ZIP_NAME):
            links = file_info.get("links", {})
            return {
                "url": links.get("self") or links.get("download") or links.get("content") or "",
                "checksum": file_info.get("checksum", ""),
                "size": str(file_info.get("size", "")),
            }

    raise RuntimeError(f"Could not find {TLS_ZIP_NAME} in Zenodo record metadata")


def verify_md5_if_available(path: Path, checksum: str) -> str:
    if not checksum:
        return "checksum_not_available"
    if not checksum.lower().startswith("md5:"):
        return "checksum_type_not_supported"
    expected = checksum.split(":", 1)[1].strip().lower()
    return "md5_ok" if file_md5(path).lower() == expected else "md5_mismatch"


def prepare_tls_archive(visium_root: Path, tls_rows: list[dict[str, str]], download: bool, dry_run: bool, overwrite: bool, skip_zenodo: bool) -> dict[str, str]:
    if not tls_rows:
        return {"status": "not_needed", "checksum_status": "not_applicable"}
    if skip_zenodo:
        return {"status": "skipped_zenodo", "checksum_status": "skipped_zenodo"}

    zip_path = visium_root / TLS_ZIP_RELATIVE_PATH
    extract_root = visium_root / TLS_EXTRACT_RELATIVE_ROOT

    if dry_run:
        if download:
            print(f"DRY ZENODO API: {ZENODO_RECORD_API}")
            print(f"DRY DOWNLOAD ZENODO ZIP: {zip_path}")
        print(f"DRY EXTRACT ZENODO ZIP: {zip_path} -> {extract_root}")
        return {"status": "dry_run", "checksum_status": "dry_run"}

    checksum_status = "not_checked"
    if download:
        metadata = fetch_zenodo_file_metadata()
        ok, status = download_file(metadata["url"], zip_path, overwrite=overwrite)
        if not ok:
            return {"status": status, "checksum_status": "not_checked"}
        checksum_status = verify_md5_if_available(zip_path, metadata.get("checksum", ""))
        if checksum_status == "md5_mismatch":
            return {"status": "zenodo_checksum_failed", "checksum_status": checksum_status}
    elif not zip_path.exists():
        return {"status": "missing_zenodo_zip", "checksum_status": "not_checked"}

    missing_members = [row for row in tls_rows if not manifest_path(visium_root, row["raw_cache_relative_path"]).exists()]
    if not missing_members and not overwrite:
        return {"status": "extracted_existing", "checksum_status": checksum_status}

    ensure_dir(extract_root)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_root)
    except Exception as exc:
        print(f"ZENODO ZIP EXTRACT FAILED: {zip_path}")
        print(exc)
        return {"status": "zenodo_extract_failed", "checksum_status": checksum_status}
    return {"status": "zenodo_ready", "checksum_status": checksum_status}


def extract_tar_if_needed(tar_path: Path, extract_root: Path, overwrite: bool) -> tuple[bool, str]:
    if extract_root.exists() and any(extract_root.iterdir()) and not overwrite:
        return True, "extracted_existing"
    ensure_dir(extract_root)
    try:
        with tarfile.open(tar_path) as archive:
            archive.extractall(extract_root)
        return True, "extracted"
    except Exception as exc:
        print(f"TAR EXTRACT FAILED: {tar_path}")
        print(exc)
        return False, "tar_extract_failed"


def row_status_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def staged_source_path(visium_root: Path, row: dict[str, str]) -> Path:
    archive_type = row.get("source_archive_type", "direct_file")
    raw_path = manifest_path(visium_root, row["raw_cache_relative_path"])
    if archive_type == "geo_tar":
        extract_root = raw_path.parent / (raw_path.name + "_extracted")
        return manifest_path(extract_root, row.get("archive_member_relative_path", ""))
    return raw_path


def stage_file(visium_root: Path, row: dict[str, str], dry_run: bool, overwrite: bool, checksum_status: str) -> dict[str, object]:
    raw_path = staged_source_path(visium_root, row)
    cohort_path = manifest_path(visium_root, row["cohort_relative_path"])
    archive_type = row.get("source_archive_type", "direct_file")

    if dry_run:
        status = "dry_run"
        byte_count = row_status_bytes(cohort_path)
    elif not raw_path.exists():
        status = "missing_raw_source"
        byte_count = 0
    elif cohort_path.exists() and cohort_path.stat().st_size > 0 and not overwrite:
        status = "existing"
        byte_count = cohort_path.stat().st_size
    else:
        ensure_dir(cohort_path.parent)
        if archive_type == "gzip":
            with gzip.open(raw_path, "rb") as inp, cohort_path.open("wb") as out:
                shutil.copyfileobj(inp, out)
        else:
            shutil.copy2(raw_path, cohort_path)
        status = "staged"
        byte_count = cohort_path.stat().st_size

    return {
        "sample_id": row["sample_id"],
        "file_role": row["file_role"],
        "source_type": row["source_type"],
        "source_url": source_url(row),
        "raw_cache_path": row["raw_cache_relative_path"],
        "cohort_path": row["cohort_relative_path"],
        "status": status,
        "bytes_written_or_existing": byte_count,
        "checksum_status": checksum_status,
        "notes": row.get("notes", ""),
    }


def build_sample_metadata(sample_id: str, rows: list[dict[str, str]]) -> dict[str, object]:
    def unique(field: str) -> list[str]:
        return sorted({row.get(field, "") for row in rows if row.get(field, "")})
    return {
        "sample_id": sample_id,
        "source_dataset": unique("source_dataset"),
        "source_accession_gse": unique("source_accession_gse"),
        "source_accession_gsm": unique("source_accession_gsm"),
        "source_sample_label": unique("source_sample_label"),
        "source_type": unique("source_type"),
        "staged_by_script": SCRIPT_NAME,
        "notes": unique("notes"),
    }


def write_sample_metadata(visium_root: Path, rows_by_sample: dict[str, list[dict[str, str]]], staged_samples: set[str], dry_run: bool, overwrite: bool) -> list[dict[str, object]]:
    inventory_rows: list[dict[str, object]] = []
    for sample_id in sorted(staged_samples):
        metadata_path = visium_root / "visium_cohort_clean" / sample_id / "metadata.json"
        metadata = build_sample_metadata(sample_id, rows_by_sample[sample_id])
        if dry_run:
            status = "dry_run"
            byte_count = row_status_bytes(metadata_path)
        elif metadata_path.exists() and metadata_path.stat().st_size > 0 and not overwrite:
            status = "existing"
            byte_count = metadata_path.stat().st_size
        else:
            ensure_dir(metadata_path.parent)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            status = "metadata_written"
            byte_count = metadata_path.stat().st_size
        inventory_rows.append({
            "sample_id": sample_id,
            "file_role": "metadata",
            "source_type": "manifest_provenance",
            "source_url": "",
            "raw_cache_path": "",
            "cohort_path": public_path(metadata_path.relative_to(visium_root)),
            "status": status,
            "bytes_written_or_existing": byte_count,
            "checksum_status": "not_applicable",
            "notes": "Minimal factual provenance metadata generated from the staging manifest.",
        })
    return inventory_rows


def write_summary(path: Path, manifest_rows: list[dict[str, str]], inventory_rows: list[dict[str, object]], dry_run: bool) -> None:
    status_counts = Counter(str(row["status"]) for row in inventory_rows)
    source_counts = Counter(row["source_dataset"] for row in manifest_rows)
    role_counts = Counter(row["file_role"] for row in manifest_rows)
    sample_counts = Counter(row["sample_id"] for row in manifest_rows)
    lines = ["Public Visium staging summary", "", f"Dry run: {dry_run}", f"Manifest rows: {len(manifest_rows)}", f"Samples represented: {len(sample_counts)}", f"Inventory rows: {len(inventory_rows)}", "", "Status counts:"]
    lines.extend(f"  {key}: {value}" for key, value in sorted(status_counts.items()))
    lines += ["", "Source dataset counts:"]
    lines.extend(f"  {key}: {value}" for key, value in sorted(source_counts.items()))
    lines += ["", "File role counts:"]
    lines.extend(f"  {key}: {value}" for key, value in sorted(role_counts.items()))
    lines += ["", "Rows by sample:"]
    lines.extend(f"  {key}: {value}" for key, value in sorted(sample_counts.items()))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_manifest(rows: list[dict[str, str]], manifest: Path) -> None:
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest}")
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")
    missing_paths = [row for row in rows if not row.get("raw_cache_relative_path") or not row.get("cohort_relative_path")]
    if missing_paths:
        raise ValueError(f"Manifest rows missing staging paths: {len(missing_paths)}")


def url_check(url: str) -> tuple[bool, str]:
    if not url:
        return False, "missing_url"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = Request(url, method="HEAD", headers=headers)
        with urlopen(req, timeout=30) as response:
            if 200 <= response.status < 400:
                return True, f"HEAD_{response.status}"
            return False, f"HEAD_{response.status}"
    except Exception:
        pass
    try:
        req = Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urlopen(req, timeout=30) as response:
            if 200 <= response.status < 400:
                return True, f"GET_{response.status}"
            return False, f"GET_{response.status}"
    except HTTPError as exc:
        return False, f"HTTP_{exc.code}"
    except URLError as exc:
        return False, f"URL_ERROR_{exc.reason}"
    except Exception as exc:
        return False, f"ERROR_{type(exc).__name__}"


def validate_urls(rows: list[dict[str, str]], max_checks: int, skip_zenodo: bool) -> int:
    seen: set[str] = set()
    counts = Counter()
    checked = 0
    for row in rows:
        if row.get("source_type") == TLS_SOURCE_TYPE and skip_zenodo:
            counts["skipped_zenodo"] += 1
            continue
        url = source_url(row)
        if not url:
            counts["missing_url"] += 1
            continue
        if url in seen:
            continue
        seen.add(url)
        if max_checks and checked >= max_checks:
            counts["not_checked_limit"] += 1
            continue
        ok, detail = url_check(url)
        checked += 1
        status = "ok" if ok else "failed"
        counts[status] += 1
        print(f"URL {checked}: {status} {detail} {url}")
    print("")
    print("URL VALIDATION SUMMARY")
    print("=" * 80)
    for key in ["ok", "failed", "skipped_zenodo", "missing_url", "not_checked_limit"]:
        print(f"{key}: {counts[key]}")
    return 2 if counts["failed"] or counts["missing_url"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and stage public Visium source files into a stable SAMPLE_#### cohort layout.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--visium-root", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-manifest", default=None, help="Deprecated compatibility alias; use --manifest.")
    parser.add_argument("--expected-manifest", default=None, help="Deprecated compatibility alias; use --manifest.")
    parser.add_argument("--download", action="store_true", help="Download/cache public source files before staging.")
    parser.add_argument("--stage", action="store_true", help="Stage cached public files into visium_cohort_clean/SAMPLE_#### folders.")
    parser.add_argument("--download-raw", action="store_true", help="Deprecated alias for --download.")
    parser.add_argument("--reconstruct-cohort", action="store_true", help="Deprecated alias for --stage.")
    parser.add_argument("--skip-zenodo", action="store_true", help="Skip the large TLS_VISIUM_USZ Zenodo archive and report skipped TLS rows.")
    parser.add_argument("--overwrite", "--force", dest="overwrite", action="store_true", help="Overwrite existing cached or staged files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-urls", action="store_true", help="Validate source acquisition URLs without downloading full files.")
    parser.add_argument("--max-url-checks", type=int, default=0, help="Maximum number of unique source URLs to check; 0 checks all.")
    parser.add_argument("--sample-id", default="", help="Optional SAMPLE_#### filter for staging or URL validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_manifest or args.expected_manifest:
        print("WARNING: --source-manifest and --expected-manifest are deprecated and ignored; use --manifest.", file=sys.stderr)
    if args.download_raw:
        args.download = True
    if args.reconstruct_cohort:
        args.stage = True
    if not args.download and not args.stage and not args.validate_urls:
        args.download = True
        args.stage = True

    repo_root = Path(args.repo_root).resolve()
    visium_root = Path(args.visium_root)
    if not visium_root.is_absolute():
        visium_root = (repo_root / visium_root).resolve()
    manifest = manifest_path(repo_root, args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = read_tsv(manifest)
    validate_manifest(rows, manifest)
    if args.sample_id:
        rows = [row for row in rows if row.get("sample_id") == args.sample_id]
        if not rows:
            raise ValueError(f"No manifest rows matched --sample-id {args.sample_id}")

    if args.validate_urls:
        return validate_urls(rows, args.max_url_checks, args.skip_zenodo)

    print("")
    print("PUBLIC VISIUM DATA STAGING")
    print("=" * 80)
    print(f"Repo root: {repo_root}")
    print(f"Visium root: {visium_root}")
    print(f"Manifest: {manifest}")
    print(f"Manifest rows: {len(rows)}")
    print(f"Samples represented: {len({row['sample_id'] for row in rows})}")
    print(f"Download: {args.download}")
    print(f"Stage: {args.stage}")
    print(f"Skip Zenodo: {args.skip_zenodo}")
    print(f"Dry run: {args.dry_run}")
    print("")

    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_sample[row["sample_id"]].append(row)

    inventory_rows: list[dict[str, object]] = []
    non_tls_rows = [row for row in rows if row.get("source_type") != TLS_SOURCE_TYPE]
    tls_rows = [row for row in rows if row.get("source_type") == TLS_SOURCE_TYPE]

    if args.download:
        seen_raw: set[str] = set()
        for row in non_tls_rows:
            raw_rel = row["raw_cache_relative_path"]
            if raw_rel in seen_raw:
                continue
            seen_raw.add(raw_rel)
            raw_path = manifest_path(visium_root, raw_rel)
            if args.dry_run:
                continue
            ok, status = download_file(source_url(row), raw_path, overwrite=args.overwrite)
            if ok and row.get("source_archive_type") == "geo_tar":
                extract_tar_if_needed(raw_path, raw_path.parent / (raw_path.name + "_extracted"), args.overwrite)
            elif not ok:
                print(f"DOWNLOAD PROBLEM: {status} {source_url(row)}")

    tls_status = prepare_tls_archive(visium_root, tls_rows, args.download, args.dry_run, args.overwrite, args.skip_zenodo)

    staged_samples: set[str] = set()
    if args.stage:
        for row in rows:
            if row.get("source_type") == TLS_SOURCE_TYPE and args.skip_zenodo:
                inventory = {
                    "sample_id": row["sample_id"], "file_role": row["file_role"], "source_type": row["source_type"],
                    "source_url": source_url(row), "raw_cache_path": row["raw_cache_relative_path"], "cohort_path": row["cohort_relative_path"],
                    "status": "skipped_zenodo", "bytes_written_or_existing": 0, "checksum_status": "skipped_zenodo", "notes": row.get("notes", ""),
                }
            else:
                checksum_status = tls_status.get("checksum_status", "not_available") if row.get("source_type") == TLS_SOURCE_TYPE else "not_available"
                inventory = stage_file(visium_root, row, args.dry_run, args.overwrite, checksum_status)
                if inventory["status"] in {"dry_run", "existing", "staged"}:
                    staged_samples.add(row["sample_id"])
            inventory_rows.append(inventory)
        inventory_rows.extend(write_sample_metadata(visium_root, rows_by_sample, staged_samples, args.dry_run, args.overwrite))

    if not args.dry_run:
        write_tsv(visium_root / "public_visium_staging_inventory.tsv", inventory_rows, INVENTORY_COLUMNS)
        write_summary(visium_root / "public_visium_staging_summary.txt", rows, inventory_rows, args.dry_run)
    else:
        print("DRY RUN: inventory and summary files were not written.")

    status_counts = Counter(str(row["status"]) for row in inventory_rows)
    source_counts = Counter(row["source_dataset"] for row in rows)
    role_counts = Counter(row["file_role"] for row in rows)
    print("")
    print("SUMMARY")
    print("=" * 80)
    print(f"manifest_rows: {len(rows)}")
    print(f"sample_count: {len(rows_by_sample)}")
    print(f"geo_or_direct_rows: {len(non_tls_rows)}")
    print(f"zenodo_tls_rows: {len(tls_rows)}")
    print(f"tls_archive_status: {tls_status.get('status', 'not_needed')}")
    print(f"tls_checksum_status: {tls_status.get('checksum_status', 'not_applicable')}")
    print("status_counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("source_dataset_counts:")
    for source_dataset, count in sorted(source_counts.items()):
        print(f"  {source_dataset}: {count}")
    print("file_role_counts:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")

    failure_statuses = {"download_failed", "missing_source_url", "missing_raw_source", "zenodo_checksum_failed", "zenodo_extract_failed", "missing_zenodo_zip", "download_empty", "tar_extract_failed"}
    if any(status in failure_statuses for status in status_counts) or tls_status.get("status") in failure_statuses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())