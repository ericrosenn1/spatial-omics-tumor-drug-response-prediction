import importlib.util,sys,unittest
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
spec=importlib.util.spec_from_file_location('step10_report',root/'scripts/10_build_integrated_interpretation_package.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReportingAcceptanceTests(unittest.TestCase):
    def test_negative_family_stays_empty(self):
        x=pd.DataFrame({'drug_key':['a','b'],'validated_for_step10':[False,False]})
        y=module.validated_only(x);self.assertEqual(len(y),0);self.assertEqual(y.columns.tolist(),x.columns.tolist())
    def test_missing_decision_and_duplicates_fail(self):
        with self.assertRaises(ValueError):module.validated_only(pd.DataFrame({'drug_key':['a']}))
        with self.assertRaises(ValueError):module.validated_only(pd.DataFrame({'drug_key':['a','a'],'validated_for_step10':[True,True]}))
    def test_only_explicit_acceptance(self):
        x=pd.DataFrame({'drug_key':['a','b','c'],'validated_for_step10':['true','false','UNKNOWN']})
        self.assertEqual(module.validated_only(x).drug_key.tolist(),['a'])
if __name__=='__main__':unittest.main()
