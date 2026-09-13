import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from spm_v2.leakage_audit import compare,fdr,recompute

class IndependentAuditTests(unittest.TestCase):
    def test_missing_metric_cannot_pass(self):
        with self.assertRaises(AssertionError):compare({'r2':.4},{'pearson':.9},'missing column')
    def test_wrong_metric_cannot_pass(self):
        with self.assertRaises(AssertionError):compare({'r2':-.4},{'r2':.4},'squared correlation')
    def test_fdr_and_negative_r2(self):
        np.testing.assert_allclose(fdr([.01,.04,.03]),[.03,.04,.04])
        m=recompute([1,2,3],[11,12,13],[2,2,2]);self.assertAlmostEqual(m['pearson'],1);self.assertLess(m['r2'],0)

if __name__=='__main__':unittest.main()
