"""Completed caches survive restart; unsuccessful or invalid caches do not."""
import importlib.util
from pathlib import Path
import sys

import anndata
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
spec = importlib.util.spec_from_file_location("cached_merge", ROOT / "code/03_merge_slide_features.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def cached_sample(output_root, sample_id="SAMPLE_0001"):
    root = output_root / "output_02_01_process_samples_data" / sample_id
    h5ad = root / "adata/02_processed.h5ad"
    features = root / "tables/slide_level_feature_row.csv"
    h5ad.parent.mkdir(parents=True)
    features.parent.mkdir(parents=True)
    data = anndata.AnnData(np.ones((3, 2)))
    data.obs["sample_id"] = sample_id
    data.obs["leiden"] = pd.Categorical(["0", "0", "1"])
    data.write_h5ad(h5ad)
    pd.DataFrame([{"sample_id": sample_id, "n_spots": 3, "n_genes": 2, "n_clusters": 2}]).to_csv(features, index=False)
    return {"sample_id": sample_id, "status": "SKIPPED", "reason": "outputs_exist",
            "processed_h5ad": str(h5ad), "slide_feature_row": str(features), "error": ""}


@pytest.mark.parametrize("status,reason,accepted", [
    ("OK", "", True),
    ("SKIPPED", "outputs_exist", True),
    ("ERROR", "outputs_exist", False),
    ("SKIPPED", "empty_after_filtering", False),
    ("SKIPPED", "invalid", False),
    ("SKIPPED", "", False),
])
def test_only_success_or_verified_cache_is_accepted(tmp_path, status, reason, accepted):
    row = cached_sample(tmp_path)
    row.update(status=status, reason=reason)
    assert bool(merge.is_success_or_verified_cached_output(row, tmp_path)) is accepted


@pytest.mark.parametrize("damage", ["missing_h5ad", "missing_csv", "corrupt_h5ad", "wrong_id", "wrong_shape", "no_leiden", "empty"])
def test_invalid_completed_files_are_rejected(tmp_path, damage):
    row = cached_sample(tmp_path)
    h5ad, features = Path(row["processed_h5ad"]), Path(row["slide_feature_row"])
    if damage == "missing_h5ad":
        h5ad.unlink()
    elif damage == "missing_csv":
        features.unlink()
    elif damage == "corrupt_h5ad":
        h5ad.write_text("not an H5AD")
    elif damage in {"wrong_id", "wrong_shape"}:
        frame = pd.read_csv(features)
        frame.loc[0, "sample_id" if damage == "wrong_id" else "n_spots"] = "OTHER" if damage == "wrong_id" else 99
        frame.to_csv(features, index=False)
    else:
        data = anndata.read_h5ad(h5ad)
        if damage == "no_leiden":
            del data.obs["leiden"]
        else:
            data = data[:0].copy()
        data.write_h5ad(h5ad)
    assert not merge.is_success_or_verified_cached_output(row, tmp_path)


def test_all_cached_restart_merges_current_files(tmp_path, monkeypatch):
    rows = [cached_sample(tmp_path, sample_id) for sample_id in ["SAMPLE_0001", "SAMPLE_0002"]]
    # Reports can contain historical machine paths; current expected files govern reuse.
    for row in rows:
        row.update(processed_h5ad="Z:/obsolete/cache.h5ad", slide_feature_row="Z:/obsolete/row.csv")
    report = tmp_path / "output_02_process_samples_reports/processing_report.csv"
    report.parent.mkdir()
    pd.DataFrame(rows).to_csv(report, index=False)
    monkeypatch.setattr(merge, "load_config", lambda _: {"output_root": tmp_path})
    monkeypatch.setattr(merge, "validate_config", lambda cfg: cfg)
    monkeypatch.setattr(sys, "argv", ["03_merge_slide_features.py", "--config", "unused"])
    merge.main()
    saved = pd.read_csv(tmp_path / "output_03_merge_slide_features/merged_slide_features.csv")
    assert saved["sample_id"].tolist() == ["SAMPLE_0001", "SAMPLE_0002"]
    assert not saved["sample_id"].duplicated().any()


@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_sparse_cache_requires_complete_matrix_structure(tmp_path, sparse_format):
    import h5py
    from scipy import sparse

    row = cached_sample(tmp_path)
    h5ad = Path(row["processed_h5ad"])
    data = anndata.read_h5ad(h5ad)
    data.X = getattr(sparse, sparse_format + "_matrix")(data.X)
    data.write_h5ad(h5ad)
    assert merge.is_success_or_verified_cached_output(row, tmp_path)
    with h5py.File(h5ad, "r+") as handle:
        del handle["X/data"]
    assert not merge.is_success_or_verified_cached_output(row, tmp_path)


def test_invalid_restart_preserves_previous_merged_table(tmp_path, monkeypatch):
    row = cached_sample(tmp_path)
    Path(row["processed_h5ad"]).write_text("incomplete")
    report = tmp_path / "output_02_process_samples_reports/processing_report.csv"
    report.parent.mkdir()
    pd.DataFrame([row]).to_csv(report, index=False)
    merged = tmp_path / "output_03_merge_slide_features/merged_slide_features.csv"
    merged.parent.mkdir()
    previous = b"sample_id,n_spots\nSAMPLE_PREVIOUS,500\n"
    merged.write_bytes(previous)
    monkeypatch.setattr(merge, "load_config", lambda _: {"output_root": tmp_path})
    monkeypatch.setattr(merge, "validate_config", lambda cfg: cfg)
    monkeypatch.setattr(sys, "argv", ["03_merge_slide_features.py", "--config", "unused"])
    with pytest.raises(RuntimeError, match="failed or produced no successful samples"):
        merge.main()
    assert merged.read_bytes() == previous
