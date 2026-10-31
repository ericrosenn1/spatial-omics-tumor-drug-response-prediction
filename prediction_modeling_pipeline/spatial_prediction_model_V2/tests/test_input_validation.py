"""Invalid handoff identities must fail before modeling joins change row counts."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spm_v2.data_discovery import handoff_sample_coverage, handoff_validation_errors
from spm_v2.dataset_builder import build_dataset_bundle


def handoff():
    teacher = pd.DataFrame({
        "sample_id": ["S1", "S2"], "drug_key": ["D1", "D1"],
        "fused_residual_vs_prior": [.1, -.1],
    })
    spatial = pd.DataFrame({"sample_id": ["S1", "S2"], "feature_a": [1., np.nan]})
    manifest = pd.DataFrame({"feature": ["feature_a"]})
    return teacher, spatial, manifest


def test_valid_handoff_and_explicit_missing_measurements():
    teacher, spatial, manifest = handoff()
    teacher.loc[0, "fused_residual_vs_prior"] = np.nan
    assert handoff_validation_errors(teacher, spatial, manifest) == []


@pytest.mark.parametrize("table,column", [("teacher", "sample_id"), ("teacher", "drug_key"), ("spatial", "sample_id")])
@pytest.mark.parametrize("bad", [None, "  "])
def test_missing_identities_rejected(table, column, bad):
    teacher, spatial, manifest = handoff()
    target = teacher if table == "teacher" else spatial
    target.loc[0, column] = bad
    assert any(f"missing or blank {column}" in e for e in handoff_validation_errors(teacher, spatial, manifest))


def test_duplicate_rows_cannot_multiply_training_pairs():
    teacher, spatial, manifest = handoff()
    duplicate_spatial = pd.concat([spatial, spatial.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_dataset_bundle(teacher, duplicate_spatial, manifest)
    duplicate_teacher = pd.concat([teacher, teacher.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sample_id/drug_key"):
        build_dataset_bundle(duplicate_teacher, spatial, manifest)


def test_no_matching_sample_ids_rejected():
    teacher, spatial, manifest = handoff()
    spatial["sample_id"] = ["X1", "X2"]
    with pytest.raises(ValueError, match="no matching sample_id"):
        build_dataset_bundle(teacher, spatial, manifest)


def test_partial_overlap_keeps_existing_inner_join_and_exposes_coverage():
    teacher, spatial, _ = handoff()
    spatial = spatial.iloc[[0]].copy()
    features = [f"feature_{i}" for i in range(10)]
    for feature in features:
        spatial[feature] = 1.
    manifest = pd.DataFrame({"feature": features})
    assert handoff_validation_errors(teacher, spatial, manifest) == []
    assert handoff_sample_coverage(teacher, spatial) == {
        "matched_sample_count": 1, "teacher_only_sample_count": 1, "spatial_only_sample_count": 0,
    }
    bundle = build_dataset_bundle(teacher, spatial, manifest)
    assert bundle["pair_level_residual_dataset"].sample_id.tolist() == ["S1"]


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), "invalid"])
def test_nonfinite_or_malformed_targets_rejected(bad):
    teacher, spatial, manifest = handoff()
    teacher["fused_residual_vs_prior"] = teacher.fused_residual_vs_prior.astype(object)
    teacher.loc[0, "fused_residual_vs_prior"] = bad
    assert any("nonnumeric or infinite" in e for e in handoff_validation_errors(teacher, spatial, manifest))


def test_cli_reports_failure_for_duplicate_spatial_identity(tmp_path, monkeypatch):
    teacher, spatial, manifest = handoff()
    teacher.to_csv(tmp_path / "visium_fused_teacher_table.tsv", sep="\t", index=False)
    pd.concat([spatial, spatial.iloc[[0]]]).to_csv(tmp_path / "model_input_numeric.csv", index=False)
    manifest.to_csv(tmp_path / "feature_manifest.csv", index=False)
    output = tmp_path / "validation"
    spec = importlib.util.spec_from_file_location("validate_inputs", ROOT / "scripts/01_validate_inputs.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(sys, "argv", ["validate_inputs", "--handoff-root", str(tmp_path), "--output-root", str(output)])
    assert entry.main() == 2
    report = json.loads((output / "v2_input_validation_summary.json").read_text())
    assert report["status"] == "fail"
    assert "spatial_numeric: duplicate sample_id values" in report["validation_errors"]
