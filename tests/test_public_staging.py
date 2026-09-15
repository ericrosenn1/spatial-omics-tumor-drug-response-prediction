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
