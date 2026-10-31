"""Final transfer archives retain current reports and verifiable payload hashes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest


def load_packager(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("final_transfer_packager", scripts / "05_qc_and_package_transfer_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_outputs(module, root, monkeypatch):
    pim = root / "atlas"
    cards = pim / "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv"
    cards.parent.mkdir(parents=True)
    pd.DataFrame({"drug_key": ["drug"]}).to_csv(cards, sep="\t", index=False)
    output = root / "transfer"
    for name in module.EXPECTED_CORE_FILES:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": ["sample"], "drug_key": ["drug"],
                      "explanation": ["A recorded research alignment with explicit coverage."],
                      "rescaled_alignment_score_0_1": [.6]}).to_csv(path, sep="\t", index=False)
    for name in module.EXPECTED_REPORTS:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FILEPATH: " + str(path) + "\n", encoding="utf-8")
    for step in range(1, 5):
        (output / f"spatial_transfer_inference_model_step{step:02}_summary.json").write_text('{"status":"pass"}', encoding="utf-8")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
        output_root=str(output), pim_run_root=str(pim), sample_id="sample", smoke_test=True, open_output=False))
    return output


def test_final_archive_manifest_matches_payload_on_first_run_and_restart(tmp_path, monkeypatch):
    module = load_packager(monkeypatch)
    output = prepare_outputs(module, tmp_path, monkeypatch)
    manifest_name = "05_qc_and_transfer_package/02_manifests/transfer_package_file_manifest.tsv"
    archive = output / "05_qc_and_transfer_package/03_transfer_zip/spatial_transfer_inference_package.zip"
    summary_name = "spatial_transfer_inference_model_step05_summary.json"
    report_name = "05_qc_and_transfer_package/04_reports/spatial_transfer_inference_final_qc_and_package_report.txt"
    for _ in range(2):
        assert module.main() == 0
        with zipfile.ZipFile(archive) as saved:
            assert saved.testzip() is None
            names = saved.namelist()
            assert len(names) == len(set(names))
            manifest = pd.read_csv(io.BytesIO(saved.read(manifest_name)), sep="\t")
            assert set(manifest.relative_path) == set(names) - {manifest_name}
            for row in manifest.itertuples():
                data = saved.read(row.relative_path)
                assert len(data) == row.size_bytes
                assert hashlib.sha256(data).hexdigest() == row.sha256
            for name in [summary_name, report_name, "spatial_transfer_inference_model_output_manifest.tsv"]:
                assert saved.read(name) == (output / name).read_bytes()
            summary = json.loads(saved.read(summary_name))
            assert summary["package_file_count"] == len(names)
            assert f"package_file_count: {len(names)}" in saved.read(report_name).decode()
        inventory = pd.read_csv(output / "spatial_transfer_inference_model_output_manifest.tsv", sep="\t")
        for row in inventory.itertuples():
            assert (output / row.relative_path).stat().st_size == row.size_bytes


def test_final_archive_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    module = load_packager(monkeypatch)
    output = prepare_outputs(module, tmp_path, monkeypatch)
    original = module.package_zip
    calls = 0

    def fail_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("archive write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "package_zip", fail_refresh)
    assert module.main() == 1
    summary = json.loads((output / "spatial_transfer_inference_model_step05_summary.json").read_text())
    assert summary["status"] == "fail"
    assert any("archive write failed" in error for error in summary["errors"])


def test_failed_archive_write_preserves_existing_archive(tmp_path, monkeypatch):
    module = load_packager(monkeypatch)
    import _stim_utils

    root = tmp_path / "transfer"
    root.mkdir()
    (root / "data.tsv").write_text("value\n1\n")
    archive = root / "05_qc_and_transfer_package/03_transfer_zip/spatial_transfer_inference_package.zip"
    _stim_utils.package_zip(archive, root)
    before = archive.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("interrupted write")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)
    with pytest.raises(OSError, match="interrupted write"):
        _stim_utils.package_zip(archive, root)
    assert archive.read_bytes() == before
    assert not archive.with_suffix(".zip.tmp").exists()


def test_failed_final_publication_preserves_previous_complete_package(tmp_path, monkeypatch):
    module = load_packager(monkeypatch)
    output = prepare_outputs(module, tmp_path, monkeypatch)
    assert module.main() == 0
    archive = output / "05_qc_and_transfer_package/03_transfer_zip/spatial_transfer_inference_package.zip"
    before = archive.read_bytes()
    original = module.package_zip

    def fail_final_publication(path, *args, **kwargs):
        if path == archive:
            raise OSError("final publication interrupted")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module, "package_zip", fail_final_publication)
    assert module.main() == 1
    assert archive.read_bytes() == before
    assert not archive.with_name("spatial_transfer_inference_package.validation.zip").exists()
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        old_summary = json.loads(saved.read("spatial_transfer_inference_model_step05_summary.json"))
        assert old_summary["status"] == "pass"
    failed_summary = json.loads((output / "spatial_transfer_inference_model_step05_summary.json").read_text())
    assert failed_summary["status"] == "fail"
