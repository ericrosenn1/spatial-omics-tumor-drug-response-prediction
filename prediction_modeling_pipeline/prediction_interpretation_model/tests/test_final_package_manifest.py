"""Archived manifests describe payload bytes on first build and on refresh."""
import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile

import pandas as pd
import pytest


@pytest.mark.parametrize("existing_package", [False, True])
def test_refreshed_package_manifest_matches_archived_payloads(tmp_path, monkeypatch, existing_package):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pim_final_packaging", scripts / "08_qc_and_package_final_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = tmp_path / "07_final_outputs/tables/results.tsv"
    payload.parent.mkdir(parents=True)
    expected = b"treatment\tvalue\nexample\t0.25\n"
    payload.write_bytes(expected)
    manifest = tmp_path / "08_qc_and_final_package/02_manifests/final_package_file_manifest.tsv"
    archive = tmp_path / "08_qc_and_final_package/03_final_zip/prediction_interpretation_model_final_package.zip"
    manifest.parent.mkdir(parents=True)
    archive.parent.mkdir(parents=True)
    if existing_package:
        manifest.write_text("stale manifest", encoding="utf-8")
        archive.write_bytes(b"previous archive")

    for _ in range(2):
        candidates = [p for p in module.all_files_under(tmp_path) if module.should_package(p, tmp_path)]
        module.write_tsv(manifest, module.build_package_file_manifest(candidates, tmp_path))
        candidates = [p for p in module.all_files_under(tmp_path) if module.should_package(p, tmp_path)]
        module.zip_files(archive, candidates, tmp_path)
        with zipfile.ZipFile(archive) as saved:
            assert saved.testzip() is None
            members = saved.namelist()
            manifest_name = manifest.relative_to(tmp_path).as_posix()
            assert len(members) == len(set(members)) == len(candidates)
            assert archive.relative_to(tmp_path).as_posix() not in members
            table = pd.read_csv(io.BytesIO(saved.read(manifest_name)), sep="\t")
            assert set(table.relative_path) == set(members) - {manifest_name}
            for row in table.itertuples():
                data = saved.read(row.relative_path)
                assert len(data) == row.size_bytes
                assert hashlib.sha256(data).hexdigest() == row.sha256
            assert saved.read(payload.relative_to(tmp_path).as_posix()) == expected
