"""Offline staging checks use tiny archives; no public datasets are downloaded."""
import gzip
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staging", ROOT / "scripts/download_and_reconstruct_public_visium_sources.py")
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


@pytest.mark.parametrize("member", ["../outside", "/outside", "X:/outside", "..\\outside"])
def test_manifest_paths_cannot_escape(tmp_path, member):
    with pytest.raises(ValueError, match="escapes"):
        staging.manifest_path(tmp_path, member)


def tar_fixture(path, name="data/matrix.mtx", content=b"matrix fixture"):
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def test_tar_partial_extraction_is_repaired_and_verified_resume_is_reused(tmp_path):
    source, output = tmp_path / "input.tar", tmp_path / "extracted"
    tar_fixture(source)
    output.mkdir()
    (output / "partial.txt").write_text("interrupted")
    assert staging.extract_archive(source, output, False) == "extracted"
    assert staging.extract_archive(source, output, False) == "extracted_existing"
    (output / "data/matrix.mtx").write_bytes(b"corrupt")
    assert staging.extract_archive(source, output, False) == "extracted"
    assert (output / "data/matrix.mtx").read_bytes() == b"matrix fixture"


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_archive_traversal_is_rejected_before_writing(tmp_path, kind):
    source = tmp_path / ("input." + kind)
    if kind == "tar":
        tar_fixture(source, "../escaped")
    else:
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("../escaped", "bad")
    with pytest.raises(ValueError, match="escapes"):
        staging.extract_archive(source, tmp_path / "output", False)
    assert not (tmp_path / "escaped").exists()


def test_gzip_staging_failure_does_not_publish_partial_file(tmp_path):
    row = dict(sample_id="SAMPLE_0001", file_role="matrix", source_type="geo", source_url="",
               raw_cache_relative_path="bad.gz", cohort_relative_path="cohort/matrix.mtx", source_archive_type="gzip")
    (tmp_path / "bad.gz").write_bytes(gzip.compress(b"hello")[:-5])
    result = staging.stage_file(tmp_path, row, False, False, "not_available")
    assert result["status"] == "stage_failed"
    assert not (tmp_path / "cohort/matrix.mtx").exists()
    (tmp_path / "bad.gz").write_bytes(gzip.compress(b"hello"))
    assert staging.stage_file(tmp_path, row, False, False, "not_available")["status"] == "staged"
    assert (tmp_path / "cohort/matrix.mtx").read_bytes() == b"hello"
    assert staging.stage_file(tmp_path, row, False, False, "not_available")["status"] == "existing"
    (tmp_path / "cohort/matrix.mtx").write_bytes(b"corrupted output")
    assert staging.stage_file(tmp_path, row, False, False, "not_available")["status"] == "stage_failed"
    assert (tmp_path / "cohort/matrix.mtx").read_bytes() == b"corrupted output"


def test_download_only_failure_is_nonzero(tmp_path, monkeypatch):
    row = {column: "" for column in staging.MANIFEST_COLUMNS}
    row.update(sample_id="SAMPLE_0001", raw_cache_relative_path="raw/file", cohort_relative_path="visium_cohort_clean/SAMPLE_0001/file")
    manifest = tmp_path / "manifest.tsv"
    staging.write_tsv(manifest, [row], staging.MANIFEST_COLUMNS)
    monkeypatch.setattr(staging, "download_file", lambda *a, **k: (False, "download_failed"))
    monkeypatch.setattr(sys, "argv", ["staging", "--manifest", str(manifest), "--visium-root", str(tmp_path / "data"), "--download"])
    assert staging.main() == 2


def test_public_manifest_is_valid():
    manifest = ROOT / staging.DEFAULT_MANIFEST
    staging.validate_manifest(staging.read_tsv(manifest), manifest)


@pytest.mark.parametrize("failed_source", ["direct", "tls"])
def test_failed_acquisition_preserves_independent_staging_and_full_inventory(tmp_path, monkeypatch, failed_source):
    rows = []
    for index in (1, 2):
        row = {column: "" for column in staging.MANIFEST_COLUMNS}
        row.update(sample_id=f"SAMPLE_{index:04d}", file_role="matrix", source_type="geo",
                   source_dataset="fixture", source_archive_type="direct_file",
                   source_url=f"https://example.org/{index}", raw_cache_relative_path=f"raw/{index}",
                   cohort_relative_path=f"visium_cohort_clean/SAMPLE_{index:04d}/matrix.mtx")
        rows.append(row)
    if failed_source == "tls":
        rows[0]["source_type"] = staging.TLS_SOURCE_TYPE
        monkeypatch.setattr(staging, "prepare_tls_archive", lambda *a, **k: {
            "status": "zenodo_checksum_failed", "checksum_status": "md5_mismatch"
        })
    manifest = tmp_path / "manifest.tsv"
    staging.write_tsv(manifest, rows, staging.MANIFEST_COLUMNS)
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw/1").write_bytes(b"stale failed source")

    def download(url, path, **kwargs):
        if url.endswith("/1"):
            return False, "download_failed"
        path.write_bytes(b"verified independent source")
        return True, "downloaded"

    monkeypatch.setattr(staging, "download_file", download)
    monkeypatch.setattr(sys, "argv", ["staging", "--manifest", str(manifest), "--visium-root", str(data), "--download", "--stage"])
    assert staging.main() == 2
    assert not (data / rows[0]["cohort_relative_path"]).exists()
    assert (data / rows[1]["cohort_relative_path"]).read_bytes() == b"verified independent source"
    inventory = staging.read_tsv(data / "public_visium_staging_inventory.tsv")
    matrices = [row for row in inventory if row["file_role"] == "matrix"]
    assert len(matrices) == 2
    assert matrices[0]["status"] == ("download_failed" if failed_source == "direct" else "zenodo_checksum_failed")
    assert matrices[1]["status"] == "staged"
