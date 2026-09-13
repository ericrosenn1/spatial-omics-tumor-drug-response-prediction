import importlib.util
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


effects_module = script("03_compute_signed_spatial_effects")
cards_module = script("04_build_treatment_interpretation_cards")
prepare_module = script("01_prepare_interpretation_inputs")
sample_module = script("05_build_sample_level_interpretations")
atlas_module = script("06_build_mechanism_atlas")


def test_signed_effect_uses_full_correlation_and_preserves_profile_key():
    key = "bevacizumab|carboplatin|docetaxel|topotecan"
    joined = pd.DataFrame({"sample_id": list("abcdef"), "drug_key": [key] * 6,
                           "x": [1., 2., 3., 4., 5., 6.], "y": [4., 1., 3., 2., 6., 5.],
                           "target": [0., .1, .2, .7, .4, .5]})
    evidence = pd.DataFrame({"drug_key": [key, key], "feature_name": ["x", "y"],
                             "mean_abs_shap": [3., 1.]})
    dictionary = pd.DataFrame({"feature_name": ["x", "y"], "biological_theme": ["theme", "theme"]})
    result = effects_module.compute_treatment_feature_effects(joined, evidence, dictionary, "target").set_index("feature_name")
    assert set(result.drug_key) == {key}
    assert result.loc["x", "signed_effect"] == pytest.approx(.75 * np.corrcoef(joined.x, joined.target)[0, 1])
    assert result.loc["y", "signed_effect"] == pytest.approx(.25 * np.corrcoef(joined.y, joined.target)[0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        effects_module.compute_treatment_feature_effects(pd.concat([joined, joined.iloc[[0]]]), evidence, dictionary, "target")
    with pytest.raises(ValueError, match="duplicate"):
        effects_module.compute_treatment_feature_effects(joined, pd.concat([evidence, evidence.iloc[[0]]]), dictionary, "target")


def test_card_does_not_infer_named_regimen_or_alignment_performance():
    key = "bevacizumab|carboplatin|docetaxel|topotecan"
    row = pd.Series({"drug_key": key, "treatment_components": key.replace("|", "; ")})
    text = cards_module.build_card_text(row, pd.DataFrame(), pd.DataFrame(), 5, 5)
    assert f"Treatment key: {key}" in text
    assert "simultaneous administration are not established" in text
    assert "Predictive performance has not been evaluated for the exact signed-effect alignment rule" in text
    assert "may represent multi-agent regimens" not in text


def test_spatial_override_binds_hash_full_schema_and_canonical_identity(tmp_path):
    original = tmp_path / "original.tsv"
    corrected = tmp_path / "corrected.tsv"
    frame = pd.DataFrame({"sample_id": ["s1", "s2"], "label": [0., 1.]})
    frame.to_csv(original, sep="\t", index=False)
    frame.loc[0, "label"] = np.nan
    frame.to_csv(corrected, sep="\t", index=False)
    expected = hashlib.sha256(corrected.read_bytes()).hexdigest()
    result = prepare_module.validate_spatial_override(original, corrected, expected)
    assert result["samples"] == 2 and result["override_sha256"] == expected
    with pytest.raises(ValueError, match="requires its exact SHA256"):
        prepare_module.validate_spatial_override(original, corrected, "")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_module.validate_spatial_override(original, corrected, "0" * 64)
    frame.loc[0, "sample_id"] = "other"
    frame.to_csv(corrected, sep="\t", index=False)
    with pytest.raises(ValueError, match="identity set"):
        prepare_module.validate_spatial_override(original, corrected, hashlib.sha256(corrected.read_bytes()).hexdigest())
    frame = frame[["sample_id"]]
    frame.to_csv(corrected, sep="\t", index=False)
    with pytest.raises(ValueError, match="full original ordered column schema"):
        prepare_module.validate_spatial_override(original, corrected, hashlib.sha256(corrected.read_bytes()).hexdigest())


def test_sample_alignment_retains_missingness_and_marks_unsupported():
    pair = pd.DataFrame({"sample_id": ["s1", "s2"], "drug_key": ["FULL|PROFILE"] * 2, "target": [.1, -.1]})
    spatial = pd.DataFrame({"sample_id": ["s1", "s2"], "x": [np.nan, 1.]})
    effects = pd.DataFrame({"feature_name": ["x"], "signed_effect": [.5], "effect_weight": [.5]})
    scores, _ = sample_module.build_scores_for_treatment("FULL|PROFILE", pair, spatial, effects, "target", 3)
    assert np.isnan(scores.loc[0, "net_signed_spatial_interpretation_score"])
    assert scores.loc[0, "score_status"] == "UNSUPPORTED_NO_OBSERVED_EFFECT_WEIGHT"
    assert scores.loc[1, "net_signed_spatial_interpretation_score"] == 1.
    assert scores.loc[1, "observed_effect_weight_fraction"] == 1.
    from _pim_utils import zscore_frame
    z = zscore_frame(spatial, ["x"])
    assert np.isnan(z.loc[0, "x"]) and z.loc[1, "x"] == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_atlas_and_cards_direction_lists_require_actual_effect_sign(direction):
    keys = ["case profile A", "case profile B"]
    features = pd.DataFrame({"drug_key": keys, "feature_name": ["x", "x"],
                             "biological_theme": ["theme", "theme"],
                             "signed_effect": np.array([.1, .2]) * direction,
                             "feature_target_pearson": np.array([.2, .4]) * direction,
                             "effect_weight": [.1, .2]})
    original = features.copy(deep=True)
    themes = effects_module.compute_theme_effects(features)
    feature_atlas = atlas_module.build_feature_atlas(features)
    theme_atlas, _, _ = atlas_module.build_theme_atlas(themes, pd.DataFrame())
    present = "higher" if direction > 0 else "lower"
    absent = "lower" if direction > 0 else "higher"
    for atlas in [feature_atlas, theme_atlas]:
        assert atlas.iloc[0][f"{present}_teacher_residual_association_treatment_count"] == 2
        assert atlas.iloc[0][f"{absent}_teacher_residual_association_treatment_count"] == 0
        assert atlas.iloc[0][f"top_{absent}_teacher_residual_treatments"] in ["", "none"]
        assert all(key in atlas.iloc[0][f"top_{present}_teacher_residual_treatments"] for key in keys)
    opposite = "negative" if direction > 0 else "positive"
    assert cards_module.top_feature_summary(features, opposite, 5) == "none"
    assert cards_module.top_theme_summary(themes, opposite, 5) == "none"
    expected = f"{present}_teacher_residual_association"
    assert set(themes.dominant_direction_label) == {expected}
    pd.testing.assert_frame_equal(features, original)
