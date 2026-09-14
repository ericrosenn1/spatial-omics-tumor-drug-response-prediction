"""The public flat precomputed handoff reaches the maintained bundle exporter."""
import importlib.util
from pathlib import Path
import sys

import pytest

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
