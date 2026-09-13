import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import joblib
import tempfile
import numpy as np
import pandas as pd
from spm_v2.feature_reference import FeatureReference,rank_reference,apply_rank,feature_class

class ReferenceTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame({'sample_id':[f's{i}' for i in range(8)],'dataset_id':['a']*4+['b']*4,'mean__immune_general_score':[1,2,2,4,5,6,np.nan,8],'access_boundary_accessibility_mean':np.arange(8.)})
    def test_rank_reproduces_training_ties_and_na(self):
        x=pd.Series([1,2,2,np.nan,8]);np.testing.assert_allclose(apply_rank(x,rank_reference(x)),x.rank(pct=True),equal_nan=True)
        constant=pd.Series([2,2,2,2,2,np.nan]);np.testing.assert_allclose(apply_rank(constant,rank_reference(constant)),constant.rank(pct=True),equal_nan=True)
    def test_frozen_reloaded_and_batch_invariant(self):
        x=self.raw();ref=FeatureReference().fit(x)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ref.joblib';joblib.dump(ref,p);r=joblib.load(p)
            a=r.transform(x); b=pd.concat([r.transform(x.iloc[[i]]) for i in range(len(x))]);pd.testing.assert_frame_equal(a,b)
            pd.testing.assert_frame_equal(a.sort_index(axis=1),r.transform(x.iloc[::-1,::-1]).sort_index().sort_index(axis=1))
            c=x.copy();c.loc[0,'mean__immune_general_score']=1e8
            pd.testing.assert_frame_equal(a.iloc[1:],r.transform(c).iloc[1:])
    def test_missing_not_absent_and_duplicates_fail(self):
        x=self.raw();r=FeatureReference().fit(x)
        self.assertTrue(np.isnan(r.transform(x).loc[6,'label__immune_inflamed']))
        with self.assertRaises(ValueError):r.transform(x.drop(columns='mean__immune_general_score'))
        with self.assertRaises(ValueError):r.transform(pd.concat([x,x.iloc[[0]]]))
    def test_measurement_classes(self):
        self.assertEqual(feature_class('hotspot__t_cell_fraction')[0],'composition')
        for c in ['pair_a_b_centroid_distance','hotspot__t_cell_fragmentation_index','access_immune_depth_slope','context_module__immune_context']:
            self.assertEqual(feature_class(c)[0],'spatial')
        self.assertEqual(feature_class('n_genes')[0],'excluded')

if __name__=='__main__':unittest.main()
