from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spm_v2.model_training import make_xgb_pipeline
from spm_v2.predictor_bundle import SpatialPredictorBundle,save_bundle,load_bundle,read_feature_table


class PredictorBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X=pd.DataFrame({"distance":[1.,2.,3.,4.,5.,6.,7.,8.],"fraction":[.1,.7,.2,np.nan,.3,.8,.4,.9]})
        pipe=make_xgb_pipeline(n_estimators=10,n_jobs=1,random_state=52)
        pipe.fit(X,np.array([-.2,-.1,.1,.2,.1,.3,.2,.4]))
        cls.bundle=SpatialPredictorBundle("exact full | treatment key",list(X),pipe,.5,["TRAIN_1"])
        cls.table=pd.DataFrame({"sample_id":["GSM_EXTERNAL_1","GSM_EXTERNAL_2"],"fraction":[.3,np.nan],"distance":[2.5,6.5]})

    def test_saved_reload_reordering_and_batch_equivalence(self):
        b=self.bundle;t=self.table;expected=b.predict(t)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bundle.joblib";save_bundle(b,p);reloaded=load_bundle(p)
            self.assertTrue(expected.equals(reloaded.predict(t)))
            actual=reloaded.predict(t.iloc[::-1,::-1]).iloc[::-1].reset_index(drop=True)
            self.assertTrue(expected.equals(actual))
            singles=pd.concat([reloaded.predict(t.iloc[[i]]) for i in range(len(t))],ignore_index=True)
            self.assertTrue(expected.equals(singles))

    def test_missing_column_fails_but_present_na_uses_saved_imputer(self):
        with self.assertRaisesRegex(ValueError,"Missing required spatial feature"):
            self.bundle.predict(self.table.drop(columns="distance"))
        result=self.bundle.predict(self.table)
        self.assertEqual(result.n_training_median_imputed_features.tolist(),[0,1])
        with self.assertRaisesRegex(ValueError,"All required features"):
            self.bundle.predict(self.table.assign(distance=np.nan,fraction=np.nan))

    def test_train_transfer_collision_fails(self):
        with self.assertRaisesRegex(ValueError,"identifier collision"):
            self.bundle.predict(self.table.iloc[[0]].assign(sample_id="TRAIN_1"))

    def test_duplicate_identities_and_columns_fail(self):
        with self.assertRaisesRegex(ValueError,"Unique nonmissing"):
            self.bundle.predict(pd.concat([self.table,self.table],ignore_index=True))
        with self.assertRaisesRegex(ValueError,"Duplicate input"):
            self.bundle.predict(pd.concat([self.table,self.table[["distance"]]],axis=1))

    def test_prior_reconstruction_and_contributions(self):
        pred=self.bundle.predict(self.table)
        np.testing.assert_allclose(pred.predicted_fused_teacher_response_unclipped,pred.predicted_residual_vs_prior+.5)
        c=self.bundle.contributions(self.table)
        summed=c.groupby('sample_id').residual_contribution.sum()
        np.testing.assert_allclose(summed.loc[pred.sample_id],pred.predicted_residual_vs_prior,atol=1e-7)
        self.assertTrue(c.drug_key.eq("exact full | treatment key").all())

    def test_undefined_prior_stays_undefined(self):
        b=SpatialPredictorBundle("drug",self.bundle.ordered_features,self.bundle.pipeline,None,["TRAIN_1"])
        result=b.predict(self.table)
        self.assertTrue(result.predicted_fused_teacher_response_unclipped.isna().all())
        self.assertTrue((~result.response_reconstruction_defined).all())

    def test_raw_csv_and_tsv_duplicate_headers_rejected_before_pandas(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix,sep in [(".csv",","),(".tsv","\t")]:
                p=Path(tmp)/("features"+suffix)
                p.write_text(sep.join(["sample_id","distance","distance"])+"\n"+sep.join(["external","1","2"]),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"Duplicate or missing raw"):
                    read_feature_table(p)


if __name__=="__main__":unittest.main()
