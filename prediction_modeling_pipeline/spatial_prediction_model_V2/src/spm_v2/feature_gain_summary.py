"""Compare normalized feature gain on the same across-fit aggregation scale."""
import numpy as np
import pandas as pd


def summarize_normalized_gain(summary, n_repeats, model_family):
    """Restore zero gain for fits in which a feature was not selected.

The saved mean_gain_importance is conditional on feature selection. Multiplying
by selection_count / n_repeats recovers the mean normalized gain across all fits.
Gain is a model-importance allocation, not explained variance or clinical effect.
"""
    columns = {'feature_name', 'feature_type', 'selection_count', 'selection_frequency', 'mean_gain_importance'}
    if not columns.issubset(summary.columns):
        raise ValueError('Gain summary is missing required selection/gain fields')
    if int(n_repeats) != n_repeats or n_repeats < 1:
        raise ValueError('Positive integer repeated-fit count required')
    if summary.feature_name.isna().any() or summary.feature_name.duplicated().any():
        raise ValueError('Duplicate or missing feature identities in gain summary')
    if not summary.feature_type.isin(['spatial_candidate', 'treatment_identity']).all():
        raise ValueError('Unknown gain-summary feature type')
    count = pd.to_numeric(summary.selection_count, errors='raise').to_numpy(float)
    frequency = pd.to_numeric(summary.selection_frequency, errors='raise').to_numpy(float)
    mean = pd.to_numeric(summary.mean_gain_importance, errors='raise').to_numpy(float)
    if not np.isfinite(count).all() or not np.array_equal(count, np.floor(count)) or ((count < 1) | (count > n_repeats)).any():
        raise ValueError('Selection counts must record one row per selected repeated fit')
    if not np.isfinite(frequency).all() or not np.allclose(frequency, count / n_repeats, rtol=0, atol=1e-12):
        raise ValueError('Selection frequency does not match selection count / repeated fits')
    if not np.isfinite(mean).all() or ((mean < 0) | (mean > 1 + 1e-6)).any():
        raise ValueError('Expected finite nonnegative normalized per-feature gain')
    weighted = mean * frequency
    spatial = float(weighted[summary.feature_type.eq('spatial_candidate')].sum())
    treatment = float(weighted[summary.feature_type.eq('treatment_identity')].sum())
    total = spatial + treatment
    if total > 1 + 1e-6:
        raise ValueError('Across-fit normalized gain exceeds one; check summary aggregation')
    return dict(model_family=str(model_family), score_col='mean_normalized_gain_with_nonselection_zero',
                n_repeats=int(n_repeats), mean_total_gain=total,
                spatial_score_total=spatial, treatment_identity_score_total=treatment,
                spatial_gain_total=spatial, treatment_identity_gain_total=treatment,
                spatial_feature_fraction=spatial / total if total > 0 else np.nan,
                treatment_identity_fraction=treatment / total if total > 0 else np.nan,
                aggregation_definition='mean_gain_importance * selection_frequency, summed by feature_type; nonselected fits contribute zero',
                importance_interpretation='Normalized gain allocation among model inputs; not explained variance or an independently validated spatial effect')
