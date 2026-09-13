import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from _alignment_contract import fit_reference, transform, score, save_reference, load_reference
from _stim_utils import load_pim_feature_dictionary, load_pim_treatment_cards, read_table
import alignment_bundle


def test_final_package_retains_upstream_coverage_warning(tmp_path):
    spec = importlib.util.spec_from_file_location("transfer_package", SCRIPTS / "05_qc_and_package_transfer_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "02_aligned_features/03_qc/step02_align_features_qc_checks.tsv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"check_id": "observed_fraction", "status": "warn", "observed": .5,
                   "expected": ">=.8 preferred", "detail": "Observed feature coverage is low."}]).to_csv(path, sep="\t", index=False)
    findings = module.upstream_qc_findings(tmp_path)
    assert len(findings) == 1 and findings[0]["status"] == "warn"
    assert "step02_align_features_qc_checks.tsv" in findings[0]["detail"]


def test_duplicate_raw_headers_and_in_memory_columns_fail(tmp_path, fixture):
    path = tmp_path / "duplicated.csv"
    path.write_text("sample_id,distance,distance,fraction\nGSM_A,1,2,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate raw table headers"):
        read_table(path)
    reference, table, _ = fixture
    duplicate = pd.concat([table, table[["distance"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        transform(reference, duplicate)


def test_raw_alignment_adapter_requires_source_bound_missingness_evidence(tmp_path, fixture, monkeypatch):
    reference, table, _ = fixture
    bundles = tmp_path / "predictors"
    bundles.mkdir()
    manifest = bundles / "bundle_manifest.tsv"
    manifest.write_text("verified predictor manifest", encoding="utf-8")
    reference["upstream_raw_feature_reference"] = {"bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    atlas = tmp_path / "atlas.json"
    save_reference(reference, atlas)
    raw = tmp_path / "raw.csv"
    table.drop(columns="fraction").to_csv(raw, index=False)
    class FixedReference:
        def transform(self, frame):
            return frame.copy()
    monkeypatch.setattr(alignment_bundle, "load_predictor_reference", lambda _: FixedReference())
    output = tmp_path / "normalized.tsv"
    with pytest.raises(ValueError, match="Unreviewed missing"):
        alignment_bundle.prepare_raw_features(atlas, bundles, raw, output)
    evidence = pd.DataFrame({"feature_name": ["fraction"], "reason": ["measurement unavailable"],
                             "source_evidence": ["verified extraction gate"], "adapter_value": ["NA"],
                             "raw_input_sha256": [hashlib.sha256(raw.read_bytes()).hexdigest()]})
    review = tmp_path / "review.tsv"
    evidence.to_csv(review, sep="\t", index=False)
    result = alignment_bundle.prepare_raw_features(atlas, bundles, raw, output, review)
    assert result.fraction.isna().all() and result.distance.equals(table.distance)
    evidence["raw_input_sha256"] = "0" * 64
    evidence.to_csv(review, sep="\t", index=False)
    with pytest.raises(ValueError, match="exact input hash"):
        alignment_bundle.prepare_raw_features(atlas, bundles, raw, output, review)


@pytest.fixture
def fixture():
    train = pd.DataFrame({"sample_id": ["train_a", "train_b", "train_c"], "distance": [1., 2., 6.], "fraction": [0., .5, 1.]})
    transfer = pd.DataFrame({"sample_id": ["GSM_A", "GSM_B"], "distance": [2., np.nan], "fraction": [.1, .7]})
    effects = pd.DataFrame({"drug_key": ["FULL|RECORDED|PROFILE", "FULL|RECORDED|PROFILE"], "feature_name": ["distance", "fraction"], "signed_effect": [1., -.5]})
    return fit_reference(train, ["distance", "fraction"]), transfer, effects


def ordered(frame):
    return frame.sort_values(["sample_id", "drug_key"]).reset_index(drop=True)


def test_reload_single_batch_reorder_and_rerun_equivalence(tmp_path, fixture):
    reference, table, effects = fixture
    path = tmp_path / "reference.json"
    save_reference(reference, path)
    loaded = load_reference(path)
    expected = score(reference, table, effects)
    pd.testing.assert_frame_equal(expected, score(loaded, table, effects))
    pd.testing.assert_frame_equal(ordered(expected), ordered(pd.concat([score(loaded, table.iloc[[i]], effects) for i in range(len(table))])))
    pd.testing.assert_frame_equal(ordered(expected), ordered(score(loaded, table.iloc[::-1, ::-1], effects.iloc[::-1])))
    unrelated = pd.concat([table, pd.DataFrame({"sample_id": ["unrelated"], "distance": [1000.], "fraction": [-5.]})], ignore_index=True)
    pd.testing.assert_frame_equal(expected, score(loaded, unrelated, effects).iloc[:2].reset_index(drop=True))


def test_missing_values_are_explicit_neutral_reference_imputation(fixture):
    reference, table, effects = fixture
    z, observed = transform(reference, table)
    assert z.loc[1, "distance"] == 0
    assert not observed.loc[1, "distance"]
    assert np.isnan(table.loc[1, "distance"])
    assert score(reference, table, effects).loc[1, "observed_weight_fraction"] == pytest.approx(1/3)
    with pytest.raises(ValueError, match="Missing required feature"):
        transform(reference, table.drop(columns="distance"))


def test_duplicate_keys_and_train_transfer_collisions_fail(fixture):
    reference, table, effects = fixture
    with pytest.raises(ValueError, match="duplicate identity"):
        transform(reference, pd.concat([table, table]))
    with pytest.raises(ValueError, match="collision"):
        transform(reference, table.assign(sample_id=["train_a", "GSM_B"]))
    with pytest.raises(ValueError, match="duplicate identity"):
        score(reference, table, pd.concat([effects, effects]))


def test_source_bound_loaders_never_use_other_atlas(tmp_path):
    for root, feature in [(tmp_path / "corrected", "corrected_feature"), (tmp_path / "old", "old_feature")]:
        path = root / "02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"feature_name": [feature]}).to_csv(path, sep="\t", index=False)
    assert load_pim_feature_dictionary(tmp_path / "corrected").feature_name.tolist() == ["corrected_feature"]
    with pytest.raises(FileNotFoundError):
        load_pim_feature_dictionary(tmp_path / "missing")


def test_duplicate_atlas_keys_fail(tmp_path):
    path = tmp_path / "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"drug_key": ["A", "A"]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate identity"):
        load_pim_treatment_cards(tmp_path)


def test_standardizer_preserves_batch_and_rejects_duplicate_identity(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare", SCRIPTS / "01_prepare_transfer_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "input.csv"
    pd.DataFrame({"sample_id": ["A", "B"], "fraction": [1., 2.]}).to_csv(path, index=False)
    assert module.standardize_feature_input(path, "batch", ["fraction"]).sample_id.tolist() == ["A", "B"]
    pd.DataFrame({"sample_id": ["A", "A"], "fraction": [1., 2.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate identity"):
        module.standardize_feature_input(path, "batch", ["fraction"])


def test_unsupported_all_missing_does_not_create_neutral_prediction(fixture):
    reference, table, effects = fixture
    result = score(reference, table.assign(distance=np.nan, fraction=np.nan), effects)
    assert result.signed_spatial_alignment.isna().all()
    assert result.rescaled_alignment_score_0_1.isna().all()
    assert result.score_status.eq("UNSUPPORTED_NO_OBSERVED_EFFECT_WEIGHT").all()


def test_adapter_does_not_reingest_own_zero_filled_derivative(tmp_path):
    spec = importlib.util.spec_from_file_location("adapter", SCRIPTS / "00b_build_improved_transfer_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    primary = source / "output_09_build_motif_tables/slide_features_with_motif_tables.csv"
    primary.parent.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["SOURCE"], "hotspot__fraction": [np.nan]}).to_csv(primary, index=False)
    stale = source / "output_10_build_model_ready_table_transfer_reviewed_zero_fill/model_input_numeric.csv"
    stale.parent.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["SOURCE"], "hotspot__fraction": [0.]}).to_csv(stale, index=False)
    dictionary = tmp_path / "features.tsv"
    pd.DataFrame({"feature_name": ["hotspot__fraction"]}).to_csv(dictionary, sep="\t", index=False)
    module.build_handoff(source, tmp_path / "unused", tmp_path / "result", strict_feature_dict=dictionary)
    result = pd.read_csv(tmp_path / "result/model_input_numeric.csv")
    assert result["hotspot__fraction"].isna().all()
    with pytest.raises(ValueError, match="Name-based zero-fill"):
        module.build_handoff(source, tmp_path / "unused", tmp_path / "rejected", strict_feature_dict=dictionary, zero_fill_absent_zero_like=True)
