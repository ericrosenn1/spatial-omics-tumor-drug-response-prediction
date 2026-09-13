import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from spm_v2.leakage_evaluation import metrics,assert_disjoint,make_partitions

class EvaluationTests(unittest.TestCase):
    def test_true_r2_negative_and_constant(self):
        m=metrics([1,2,3],[11,12,13],[2,2,2]);self.assertAlmostEqual(m['pearson'],1);self.assertLess(m['r2'],-100)
        self.assertTrue(np.isnan(metrics([1,1,1],[1,2,3])['r2']))
        with self.assertRaises(ValueError):metrics([1,np.nan],[1,2])
    def test_patient_crossing_rejected(self):
        m=pd.DataFrame({'sample_id':['a','b'],'evaluation_group':['patient','patient']})
        with self.assertRaises(ValueError):assert_disjoint(['a'],['b'],m)
    def test_partition_stability_and_groups(self):
        m=pd.DataFrame({'sample_id':[f's{i:03d}' for i in range(102)],'evaluation_group':[f'g{i//2}' for i in range(102)],'patient_id_verified':[i<20 for i in range(102)]})
        a=make_partitions(m);self.assertEqual(a,make_partitions(m.iloc[::-1]))
        tests=[]
        for name,tr,te in a:
            assert_disjoint(tr,te,m)
            if name.startswith('outer'):tests+=te
        self.assertEqual(len(tests),102);self.assertEqual(len(set(tests)),102)
if __name__=='__main__':unittest.main()
