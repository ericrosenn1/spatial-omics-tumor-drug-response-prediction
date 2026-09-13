"""Exercise the real audit filter block with controlled batch-rounding fixtures."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
import pytest
from spm_v2.leakage_audit_consistency import audit_relational_consistency


class BatchRoundingReference:
    def __init__(self, mode):
        self.mode = mode
        self.training_ids = list('abcd')

    def transform(self, raw):
        result = raw.copy()
        result['signal'] = result.signal.astype(float)
        if len(raw) == 4:
            position = result.columns.get_loc('signal')
            if self.mode in {'one_ulp', 'eligibility_flip'}:
                result.iloc[0, position] = np.nextafter(1., np.inf)
            elif self.mode == 'missingness_change':
                result.iloc[0, position] = np.nan
            elif self.mode == 'material_change':
                result.iloc[0, position] = 1.001
        return result


class ReachedPostFilterAudit(Exception):
    pass


def fixture(tmp_path, mode, corrupt_count=False):
    root = tmp_path / 'run'; root.mkdir()
    metadata = tmp_path / 'metadata.tsv'
    teacher = tmp_path / 'teacher.tsv'
    raw = tmp_path / 'raw.csv'
    pd.DataFrame(dict(sample_id=list('abcdef'), evaluation_group=list('abcdef'))).to_csv(metadata, sep='\t', index=False)
    pd.DataFrame(dict(sample_id=list('abcdef'), drug_key=['full|key'] * 6,
                      fused_residual_vs_prior=np.arange(6.) / 10)).to_csv(teacher, sep='\t', index=False)
    values = [1.] * 6 if mode == 'eligibility_flip' else [1., 1., 2., 3., 4., 5.]
    pd.DataFrame(dict(sample_id=list('abcdef'), signal=values)).to_csv(raw, index=False)
    config = dict(inputs={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for k, p in [('metadata', metadata), ('teacher', teacher), ('raw_features', raw)]})
    (root / 'configuration.json').write_text(json.dumps(config))
    pd.DataFrame(dict(feature_name=['signal'], feature_class=['composition'])).to_csv(root / 'feature_classes.tsv', sep='\t', index=False)
    rows = []
    for i in range(5):
        for j, sample in enumerate('abcdef'):
            rows.append(dict(evaluation=f'outer_{i}', sample_id=sample,
                             partition='test' if j % 5 == i else 'train'))
    rows += [dict(evaluation='independent_selection', sample_id=s,
                  partition='train' if s in 'abcd' else 'test') for s in 'abcdef']
    pd.DataFrame(rows).to_csv(root / 'partitions.tsv', sep='\t', index=False)
    partition = root / 'independent_selection'; partition.mkdir()
    (partition / 'reference_dependencies.json').write_text(json.dumps(dict(training_ids=list('abcd'))))
    joblib.dump(BatchRoundingReference(mode), partition / 'feature_reference.joblib')
    count = 1 if mode == 'eligibility_flip' else 3
    pd.DataFrame(dict(feature_name=['signal'], train_nonmissing_fraction=[1.],
                      train_unique=[99 if corrupt_count else count],
                      available_training_feature=[count > 1])).to_csv(partition / 'training_feature_filter.tsv', sep='\t', index=False)
    return root, metadata, teacher


def run_through_actual_filter(root, metadata, teacher):
    original_read_csv = pd.read_csv

    def guarded_read(path, *args, **kwargs):
        if Path(path).name == 'training_eligibility.tsv':
            raise ReachedPostFilterAudit('Actual reference and saved filter checks completed')
        return original_read_csv(path, *args, **kwargs)

    with patch('spm_v2.leakage_audit_consistency.pd.read_csv', side_effect=guarded_read):
        audit_relational_consistency(root, metadata, teacher)


def test_one_ulp_unique_count_difference_passes_only_recorded_layout_count(tmp_path):
    paths = fixture(tmp_path, 'one_ulp')
    # Whole recorded transform has three distinct values; train-only has four.
    # Both have finite, varying features, so the scientific keep rule is stable.
    with pytest.raises(ReachedPostFilterAudit):
        run_through_actual_filter(*paths)


@pytest.mark.parametrize('mode,error', [
    ('missingness_change', 'missingness depends on batch'),
    ('material_change', 'values depend on batch beyond numerical tolerance'),
    ('eligibility_flip', 'training feature filter depends on transform batch'),
])
def test_batch_changes_cannot_hide_behind_rounding_allowance(tmp_path, mode, error):
    paths = fixture(tmp_path, mode)
    with pytest.raises(AssertionError, match=error):
        run_through_actual_filter(*paths)


def test_saved_unique_count_still_has_to_match_recorded_computation(tmp_path):
    paths = fixture(tmp_path, 'one_ulp', corrupt_count=True)
    with pytest.raises(AssertionError, match='raw-to-training feature filter'):
        run_through_actual_filter(*paths)
