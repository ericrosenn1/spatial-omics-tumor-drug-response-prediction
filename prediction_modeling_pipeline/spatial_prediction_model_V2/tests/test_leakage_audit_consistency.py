import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from spm_v2.leakage_audit import recompute
from spm_v2.leakage_audit_consistency import (
    audit_null_structure, audit_partition_keys, audit_screen_keys_targets,
    bind_input_arguments, expected_block_predictions, expected_matched,
    expected_source_influence, same_table, verify_prediction_metadata)


class RelationalAuditTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame(dict(sample_id=['a', 'b', 'c', 'd', 'e', 'f'],
                                     evaluation_group=['p1', 'p1', 'p2', 'p3', 'p4', 'p5'],
                                     metadata_dataset_id=['s1', 's1', 's1', 's2', 's2', 's2']))
        self.pred = pd.DataFrame(dict(sample_id=list('abcdef'), drug_key=['full|key'] * 6,
                                     target=[1., 3., 2., 4., 5., np.nan],
                                     prediction=[2., 4., 1., 4., 6., 9.],
                                     baseline=[2.] * 6,
                                     target_available=[True] * 5 + [False]))

    def test_loaded_input_arguments_must_match_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / 'original'; original.write_bytes(b'correct')
            other = Path(folder) / 'other'; other.write_bytes(b'wrong')
            record = dict(path=str(original), sha256=hashlib.sha256(b'correct').hexdigest())
            config = dict(inputs=dict(metadata=record, teacher=record))
            bind_input_arguments(config, original, original)
            with self.assertRaisesRegex(AssertionError, 'loaded argument'):
                bind_input_arguments(config, other, original)

    def test_duplicate_partition_record_is_rejected(self):
        rows = []
        for i in range(5):
            for j, sample in enumerate('abcdef'):
                rows.append(dict(evaluation=f'outer_{i}', sample_id=sample,
                                 partition='test' if j % 5 == i else 'train'))
        rows += [dict(evaluation='independent_selection', sample_id=s,
                      partition='test' if s == 'a' else 'train') for s in 'abcdef']
        parts = pd.DataFrame(rows)
        audit_partition_keys(parts, self.meta)
        with self.assertRaisesRegex(AssertionError, 'duplicate'):
            audit_partition_keys(pd.concat([parts, parts.iloc[[0]]]), self.meta)

    def test_screen_universe_and_teacher_targets_are_authoritative(self):
        teacher = self.pred[self.pred.target_available][['sample_id', 'drug_key', 'target']].rename(columns={'target': 'fused_residual_vs_prior'})
        sp = pd.concat([self.pred.iloc[:2].assign(repeat=r) for r in range(2)])
        sm = pd.DataFrame(dict(drug_key=['full|key'] * 2, repeat=[0, 1]))
        ss = pd.DataFrame(dict(drug_key=['full|key']))
        audit_screen_keys_targets(sm, sp, ss, teacher, {'full|key'}, 2)
        for bad in [pd.concat([sm, sm.iloc[[0]]]), sm.iloc[:1]]:
            with self.assertRaises(AssertionError):
                audit_screen_keys_targets(bad, sp, ss, teacher, {'full|key'}, 2)
        changed = sp.copy(); changed['target'] += 10
        with self.assertRaisesRegex(AssertionError, 'differs from corrected teacher'):
            audit_screen_keys_targets(sm, changed, ss, teacher, {'full|key'}, 2)

    def test_group_metadata_cannot_be_relabelled(self):
        p = self.pred.merge(self.meta, on='sample_id')
        verify_prediction_metadata(p, self.meta)
        p.loc[0, 'evaluation_group'] = 'invented-independent-patient'
        with self.assertRaisesRegex(AssertionError, 'verified metadata'):
            verify_prediction_metadata(p, self.meta)

    def test_independent_block_values_reconstructed_including_missingness(self):
        grouped = expected_block_predictions(self.pred, self.meta, ['full|key'])
        self.assertEqual(len(grouped), 5)
        p1 = grouped.set_index('evaluation_group').loc['p1']
        self.assertEqual(p1.target, 2.)
        self.assertEqual(p1.prediction, 3.)
        self.assertEqual(p1.sections, 2)
        self.assertTrue(np.isnan(grouped.set_index('evaluation_group').loc['p5'].target))
        corrupt = grouped.copy(); corrupt['prediction'] += 1
        with self.assertRaisesRegex(AssertionError, 'source-to-derived'):
            same_table(corrupt, grouped, ['evaluation_group', 'drug_key'], 'independent')
        with self.assertRaises(AssertionError):
            same_table(grouped.dropna(), grouped, ['evaluation_group', 'drug_key'], 'missing block')

    def test_matched_values_not_only_self_consistent_deltas(self):
        observed = self.pred[self.pred.target_available]
        records = []
        for arm, offset in [('composition', 0), ('composition_plus_spatial', .1)]:
            records.append(dict(arm=arm, drug_key='full|key', n_folds=5,
                                **recompute(observed.target, observed.prediction + offset, observed.baseline)))
        metrics = pd.DataFrame(records); expected = expected_matched(metrics)
        actual = expected.copy()
        actual['pearson_composition'] += .3; actual['pearson_spatial'] += .3
        self.assertAlmostEqual(actual.delta_pearson.iloc[0],
                               actual.pearson_spatial.iloc[0] - actual.pearson_composition.iloc[0])
        with self.assertRaisesRegex(AssertionError, 'source-to-derived'):
            same_table(actual, expected, ['drug_key'], 'matched')

    def test_source_influence_is_recomputed_from_retained_rows(self):
        p = self.pred.merge(self.meta, on='sample_id').assign(arm='composition')
        result = expected_source_influence(p)
        self.assertEqual(set(result.removed_source), {'s1', 's2'})
        self.assertEqual(result.set_index('removed_source').loc['s1', 'n'], 2)
        corrupt = result.copy(); corrupt['rmse'] += .25
        with self.assertRaises(AssertionError):
            same_table(corrupt, result, ['arm', 'drug_key', 'removed_source'], 'source influence')

    def test_null_movable_groups_derived_from_source_and_missingness(self):
        grouped = expected_block_predictions(self.pred, self.meta, ['full|key'])
        groups = sorted(grouped.evaluation_group)
        maps = pd.DataFrame([dict(permutation_id=i, evaluation_group=g, source_group=g)
                             for i in range(1000) for g in groups])
        summary = dict(tested_family=1, permutation_ids=1000, effective_unique_mappings=1,
                       movable_groups=4, null_denominator=1000, stratum_sizes=[2, 2, 1])
        audit_null_structure(grouped, maps, summary)
        with self.assertRaisesRegex(AssertionError, 'movable_groups'):
            audit_null_structure(grouped, maps, dict(summary, movable_groups=5))
        with self.assertRaisesRegex(AssertionError, 'bijection'):
            audit_null_structure(grouped, maps.iloc[1:], summary)


if __name__ == '__main__':
    unittest.main()
def test_reviewed_feature_class_cannot_be_relabeled():
    import pandas as pd
    import pytest
    from spm_v2.leakage_audit_consistency import bind_reviewed_feature_classes
    source=pd.DataFrame({'feature_name':['sample_id','a','b'],'strict_class':['excluded','composition','spatial'],'reason':['identity','raw expression summary','physical smoothing']})
    saved=source[source.feature_name.ne('sample_id')].rename(columns={'strict_class':'feature_class'}).copy()
    bind_reviewed_feature_classes(saved,source,['sample_id','a','b'])
    saved.loc[saved.feature_name.eq('b'),'feature_class']='composition'
    with pytest.raises(AssertionError,match='reviewed feature classes'):bind_reviewed_feature_classes(saved,source,['sample_id','a','b'])
