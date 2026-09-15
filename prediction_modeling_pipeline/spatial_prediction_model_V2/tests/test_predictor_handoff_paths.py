"""The public flat precomputed handoff reaches the maintained bundle exporter."""
import importlib.util
from pathlib import Path
import sys

import pytest
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/14_fit_predictor_bundles.py'
spec = importlib.util.spec_from_file_location('predictor_export_handoff_paths', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('subdir', ['', '05_prediction_ready_teacher'])
def test_actual_cli_routes_flat_and_original_handoff_without_fitting(tmp_path, monkeypatch, subdir):
    root = tmp_path / 'handoff'
    path = root / subdir / 'model_input_numeric.csv'
    path.parent.mkdir(parents=True)
    path.write_text('sample_id,x\nKNOWN_SECTION,1\n')
    called = []
    monkeypatch.setattr(module, 'attach_reference', lambda existing, raw, numeric, output: called.append(numeric))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--run-root', str(tmp_path / 'declared_run'),
        '--teacher-root', str(root), '--existing-bundles', str(tmp_path / 'saved_bundles'),
        '--raw-feature-table', str(tmp_path / 'raw.csv'), '--output', str(tmp_path / 'output')])
    module.main()
    assert called == [path]


def test_ambiguous_or_missing_handoff_fails_before_fitting(tmp_path):
    with pytest.raises(FileNotFoundError, match='No numeric feature handoff'):
        module.numeric_handoff_path(tmp_path)
    (tmp_path / 'model_input_numeric.csv').write_text('sample_id,x\nA,1\n')
    nested = tmp_path / '05_prediction_ready_teacher'; nested.mkdir()
    (nested / 'model_input_numeric.csv').write_text('sample_id,x\nA,1\n')
    with pytest.raises(ValueError, match='Ambiguous numeric handoff'):
        module.numeric_handoff_path(tmp_path)


class SavedMedianImputer:
    def __init__(self, median):
        self.median = median

    def transform(self, table):
        return table.fillna(self.median).to_numpy()


@pytest.mark.parametrize('label', ['label__endothelial_high', 'label__vascular_endothelial', 'label__myeloid_low'])
def test_historical_label_na_requires_unchanged_fitted_preprocessing(label):
    actual = pd.DataFrame({label: [np.nan, 1., 0.]})
    expected = pd.DataFrame({label: [0., 1., 0.]})
    original = actual.copy(deep=True)
    assert module.validate_reference_equivalence(actual, expected, SavedMedianImputer(0.)) == 1
    pd.testing.assert_frame_equal(actual, original)
    with pytest.raises(ValueError, match='Saved preprocessing changes'):
        module.validate_reference_equivalence(actual, expected, SavedMedianImputer(1.))


@pytest.mark.parametrize('feature,actual,saved,message', [
    ('distance', [np.nan, 1.], [0., 1.], 'missingness mismatch'),
    ('label__myeloid_low', [np.nan, 1.], [1., 1.], 'missingness mismatch'),
    ('label__myeloid_low', [0., 1.], [np.nan, 1.], 'missingness mismatch'),
    ('label__myeloid_low', [0., .5], [0., 1.], 'value mismatch'),
])
def test_reference_rejects_unexplained_missingness_or_changed_values(feature, actual, saved, message):
    with pytest.raises(ValueError, match=message):
        module.validate_reference_equivalence(pd.DataFrame({feature: actual}), pd.DataFrame({feature: saved}))
