import copy
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from spm_v2.accepted_predictor_export import accepted_keys, verify_frozen_discovery, verify_bundle_reload
from spm_v2.model_training import make_xgb_pipeline
from spm_v2.predictor_bundle import SpatialPredictorBundle


class IdentityReference:
    def transform(self, frame):
        return frame.copy()


class AcceptedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = pd.DataFrame(dict(f1=np.arange(8.), f2=[1., 2., np.nan, 1., 3., 1., 5., 3.]))
        cls.pipeline = make_xgb_pipeline(n_estimators=8, max_depth=2, learning_rate=.03, n_jobs=1, random_state=17)
        cls.pipeline.fit(cls.x, np.array([-.2, -.1, .1, .3, .2, .3, .4, .2]))
        cls.selected = pd.DataFrame(dict(feature_name=['f1', 'f2'], selection_order=[0, 1], seed=[17, 17]))
        cls.source = dict(features=['f1', 'f2'], training_samples=['TRAIN'], seed=17,
                          target='fused_residual_vs_prior', estimator=cls.pipeline)
        cls.settings = dict(estimators=8, max_depth=2, learning_rate=.03)

    def test_only_explicit_accepted_full_keys_export(self):
        summary = pd.DataFrame(dict(drug_key=['drug|history', 'single'], accepted=[True, False]))
        self.assertEqual(accepted_keys(summary), ['drug|history'])
        self.assertEqual(accepted_keys(summary.assign(accepted=False)), [])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            accepted_keys(pd.concat([summary, summary.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            accepted_keys(summary.assign(accepted='False'))

    def test_frozen_model_requires_exact_selected_order(self):
        self.assertEqual(verify_frozen_discovery(self.source, self.selected, ['TRAIN'], self.settings), ['f1', 'f2'])
        bad = dict(self.source, features=['f2', 'f1'])
        with self.assertRaisesRegex(ValueError, 'feature order'):
            verify_frozen_discovery(bad, self.selected, ['TRAIN'], self.settings)

    def test_frozen_model_rejects_wrong_training_partition_or_settings(self):
        with self.assertRaisesRegex(ValueError, 'training samples'):
            verify_frozen_discovery(self.source, self.selected, ['HOLDOUT'], self.settings)
        with self.assertRaisesRegex(ValueError, 'setting'):
            verify_frozen_discovery(self.source, self.selected, ['TRAIN'], dict(self.settings, estimators=9))

    def make_bundle(self):
        return SpatialPredictorBundle('exact|history', ['f1', 'f2'], self.pipeline, .6,
                                      ['TRAIN'], IdentityReference(),
                                      dict(evaluation_holdout_sample_ids=['TEST1', 'TEST2'],
                                           reserved_cohort_sample_ids=['TRAIN', 'TEST1', 'TEST2'],
                                           training_prediction_partition='DISCOVERY_FITTED_TRAINING_REPRODUCTION'),
                                      support_status='INDEPENDENTLY_SUPPORTED_EXACT_DISCOVERY_PREDICTOR')

    def test_reload_matches_actual_heldout_prediction_and_labels(self):
        bundle = self.make_bundle()
        raw = self.x.iloc[:2].copy(); raw.insert(0, 'sample_id', ['TEST1', 'TEST2'])
        expected = pd.DataFrame(dict(sample_id=raw.sample_id, prediction=self.pipeline.predict(raw[['f1', 'f2']])))
        with tempfile.TemporaryDirectory() as folder:
            result = verify_bundle_reload(bundle, Path(folder) / 'bundle.joblib', raw, expected)
            self.assertTrue(result.prediction_partition.eq('HELDOUT_EVALUATION_REPRODUCTION').all())
            corrupt = expected.copy(); corrupt.prediction += .1
            with self.assertRaisesRegex(ValueError, 'saved held-out'):
                verify_bundle_reload(bundle, Path(folder) / 'corrupt.joblib', raw, corrupt)

    def test_evaluation_and_training_modes_cannot_be_confused(self):
        bundle = self.make_bundle()
        raw = self.x.iloc[:1].copy(); raw.insert(0, 'sample_id', ['TEST1'])
        with self.assertRaisesRegex(ValueError, 'identifier collision'):
            bundle.predict(raw, 'raw_reference', 'external')
        with self.assertRaisesRegex(ValueError, 'recorded held-out'):
            bundle.predict(raw.assign(sample_id='NEW'), 'raw_reference', 'heldout_evaluation_reproduction')
        train = bundle.predict(raw.assign(sample_id='TRAIN'), 'raw_reference', 'training_reproduction')
        self.assertTrue(train.prediction_partition.eq('DISCOVERY_FITTED_TRAINING_REPRODUCTION').all())


if __name__ == '__main__':
    unittest.main()
