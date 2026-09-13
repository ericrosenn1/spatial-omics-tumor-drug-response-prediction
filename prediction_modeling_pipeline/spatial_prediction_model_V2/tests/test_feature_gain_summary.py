import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
from spm_v2.feature_gain_summary import summarize_normalized_gain


def evidence():
    return pd.DataFrame(dict(feature_name=['treatment_identity_A', 'spatial_A', 'spatial_B'],
                             feature_type=['treatment_identity', 'spatial_candidate', 'spatial_candidate'],
                             selection_count=[2, 1, 1], selection_frequency=[1., .5, .5],
                             mean_gain_importance=[.65, .2, .5]))


def test_nonselection_zero_weight_is_essential():
    # Actual fits: identity .8 / spatial_A .2; identity .5 / spatial_B .5.
    result = summarize_normalized_gain(evidence(), 2, 'fixture')
    assert result['mean_total_gain'] == pytest.approx(1.)
    assert result['spatial_feature_fraction'] == pytest.approx(.35)
    assert result['treatment_identity_fraction'] == pytest.approx(.65)
    assert result['spatial_gain_total'] == result['spatial_score_total']
    naive = (.2 + .5) / (.65 + .2 + .5)
    assert abs(result['spatial_feature_fraction'] - naive) > .15


def test_actual_step03_selected_row_semantics_reconstruct_across_fit_gain():
    source = Path(__file__).resolve().parents[1] / 'scripts/03_train_probability_baseline.py'
    spec = importlib.util.spec_from_file_location('step03_gain_review', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = [dict(repeat=0, feature_name='treatment_identity_A', feature_type='treatment_identity', gain_importance=.8, selected=True),
            dict(repeat=0, feature_name='spatial_A', feature_type='spatial_candidate', gain_importance=.2, selected=True),
            dict(repeat=1, feature_name='treatment_identity_A', feature_type='treatment_identity', gain_importance=.5, selected=True),
            dict(repeat=1, feature_name='spatial_B', feature_type='spatial_candidate', gain_importance=.5, selected=True)]
    result = summarize_normalized_gain(module.summarize_feature_evidence(rows), 2, 'probability_baseline')
    assert result['spatial_score_total'] == pytest.approx((.2 + .5) / 2)


def test_rejects_frequency_denominator_mismatch_and_duplicate_features():
    frame = evidence(); frame.loc[1, 'selection_frequency'] = 1.
    with pytest.raises(ValueError, match='frequency'):
        summarize_normalized_gain(frame, 2, 'fixture')
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_normalized_gain(pd.concat([evidence(), evidence().iloc[:1]]), 2, 'fixture')


def test_zero_gain_does_not_invent_a_fraction():
    result = summarize_normalized_gain(evidence().assign(mean_gain_importance=0.), 2, 'constant model')
    assert result['mean_total_gain'] == 0
    assert np.isnan(result['spatial_feature_fraction'])


def test_reordering_does_not_change_group_allocation():
    a = summarize_normalized_gain(evidence(), 2, 'fixture')
    b = summarize_normalized_gain(evidence().iloc[::-1], 2, 'fixture')
    for key in ['spatial_score_total', 'treatment_identity_score_total', 'spatial_feature_fraction']:
        assert a[key] == pytest.approx(b[key])
