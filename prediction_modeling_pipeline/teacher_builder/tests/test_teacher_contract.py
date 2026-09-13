"""Regression checks for corrected reusable expression scoring and identity fusion."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from teacher_expression_contract import (
    raw_counts_to_pseudobulk, select_gene_ids, score_expression_artifact,
    apply_saved_calibrator,
)

spec = importlib.util.spec_from_file_location("fusion", SCRIPTS / "04_fuse_teacher_tables.py")
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)
expr_spec=importlib.util.spec_from_file_location("expression_step",SCRIPTS/"02_build_expression_teacher.py")
expression_step=importlib.util.module_from_spec(expr_spec)
expr_spec.loader.exec_module(expression_step)


class ToyEstimator:
    classes_ = np.array([1, 0])
    feature_names_in_ = np.array(["ENSG2", "ENSG1", "ENSG3"])

    def predict_proba(self, X):
        self.last_input = X.copy()
        # Genuine NA reaches the fitted preprocessing; this toy imitates its
        # trained median of 2, whereas missing feature ENSG3 is explicit zero.
        p = (X.fillna(2).to_numpy() @ np.array([.01, .02, .03]))
        return np.column_stack([p, 1-p])


class ToyCalibrator:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        self.last_input = np.asarray(X).copy()
        p = 1 / (1 + np.exp(-np.asarray(X).ravel() * .5))
        return np.column_stack([p, 1-p])


class TeacherContractTests(unittest.TestCase):
    def test_duplicate_gene_counts_are_aggregated_before_log(self):
        out = raw_counts_to_pseudobulk(np.array([[2, 3, 5], [1, 4, 5]]), ["ENSG1.1", "ENSG1.2", "ENSG2"], ["ENSG2", "ENSG1", "ENSG_missing"])
        self.assertEqual(list(out), ["ENSG2", "ENSG1", "ENSG_missing"])
        self.assertAlmostEqual(out["ENSG1"], np.log1p(500_000))
        self.assertEqual(out["ENSG_missing"], 0)

    def test_negative_raw_measurements_not_hidden_by_positive_gene_sum(self):
        with self.assertRaisesRegex(ValueError,"negative or nonfinite"):
            raw_counts_to_pseudobulk(np.array([[3.,2.],[-1.,4.]]),["ENSG1","ENSG2"],["ENSG1","ENSG2"])

    def test_gene_order_invariant_and_missing_genes_legitimate(self):
        X = np.array([[3, 7], [2, 8]])
        a = raw_counts_to_pseudobulk(X, ["ENSG1", "ENSG2"], ["ENSG2", "ENSG1", "ENSG3"])
        b = raw_counts_to_pseudobulk(X[:, ::-1], ["ENSG2", "ENSG1"], ["ENSG2", "ENSG1", "ENSG3"])
        self.assertEqual(a, b)

    def test_symbols_do_not_replace_ensembl_metadata(self):
        ids = select_gene_ids(pd.DataFrame({"gene_ids":["ENSG1.2", "ENSG2"]}), ["TP53", "KRAS"])
        self.assertEqual(ids.tolist(), ["ENSG1", "ENSG2"])
        with self.assertRaisesRegex(ValueError, "No Ensembl"):
            select_gene_ids(pd.DataFrame(), ["TP53", "KRAS"])

    def test_par_y_trained_identity_not_collapsed_into_x(self):
        out=raw_counts_to_pseudobulk(np.array([[4,6]]),["ENSG1","ENSG2"],["ENSG1.2","ENSG1.2_PAR_Y","ENSG2.1"])
        self.assertAlmostEqual(out["ENSG1.2"],np.log1p(400000))
        self.assertEqual(out["ENSG1.2_PAR_Y"],0.0)

    def test_artifact_class_order_calibration_missingness_and_columns(self):
        model, cal = ToyEstimator(), ToyCalibrator()
        artifact = {"base_model":model, "gene_columns":list(model.feature_names_in_), "calibration_method":"sigmoid", "calibrator":cal}
        X = pd.DataFrame({"ENSG1":[2., np.nan], "ENSG2":[4., 8.]})
        raw, probability, cov = score_expression_artifact(artifact, X)
        np.testing.assert_allclose(raw, [.08, .12])
        np.testing.assert_allclose(cal.last_input.ravel(), np.log(raw/(1-raw)))
        self.assertTrue(np.isnan(model.last_input.loc[1,"ENSG1"]))
        self.assertEqual(model.last_input.ENSG3.tolist(), [0,0])
        self.assertEqual(cov["missing_gene_count"], 1)
        raw2, prob2, _ = score_expression_artifact(artifact, X.iloc[::-1, ::-1])
        np.testing.assert_array_equal(raw, raw2[::-1])
        np.testing.assert_array_equal(probability, prob2[::-1])

    def test_missing_calibrator_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing saved calibrator"):
            apply_saved_calibrator([.2], "sigmoid", None)

    def test_legacy_step_helpers_preserve_saved_calibrator(self):
        model,cal=ToyEstimator(),ToyCalibrator()
        artifact={"base_model":model,"gene_columns":list(model.feature_names_in_),"calibration_method":"sigmoid","calibrator":cal}
        X=pd.DataFrame({"ENSG1":[2.],"ENSG2":[4.]})
        wrapper=expression_step.unwrap_predictor(artifact)
        self.assertIs(wrapper,artifact)
        np.testing.assert_array_equal(expression_step.predict_probability(wrapper,X),score_expression_artifact(artifact,X)[1])

    def test_capitalization_invariant_identity_fusion(self):
        e = pd.DataFrame({"sample_id":["sample"],"drug_key":["Cisplatin"],"drug":["CISPLATIN"],"expression_prob_calibrated":[.7]})
        h = pd.DataFrame({"sample_id":["sample"],"drug_key":["cisplatin"],"drug":["Cisplatin"],"histology_prob_calibrated":[.6]})
        joined = fusion.merge_teacher_modalities(fusion.standardize_expression(e), fusion.standardize_histology(h))
        self.assertEqual(len(joined), 1)
        self.assertTrue(joined.expression_available.iloc[0] and joined.histology_available.iloc[0])

    def test_duplicate_normalized_identities_rejected(self):
        for standardize, col in [(fusion.standardize_expression,"expression"),(fusion.standardize_histology,"histology")]:
            t=pd.DataFrame({"sample_id":["s","s"],"drug_key":["Cisplatin","cisplatin"],col+"_prob_calibrated":[.5,.7]})
            with self.assertRaisesRegex(ValueError,"duplicate sample-treatment"):
                standardize(t)

    def test_prior_only_is_not_expression_inference(self):
        e=pd.DataFrame({"sample_id":["s"],"drug_key":["drug"],"expression_prob_calibrated":[.6],"expression_teacher_mode":["model_artifact_missing_prior_only"]})
        row=fusion.standardize_expression(e).iloc[0]
        self.assertFalse(row.expression_available)
        self.assertEqual(row.expression_sample_confidence, 0)
        self.assertEqual(row.expression_reliability_weight, 0)

    def test_real_651_restored_overlaps_when_local_data_available(self):
        r=SCRIPTS.parent/"outputs_corrected_20260907_220451"
        if not r.exists():
            self.skipTest("Local corrected data not distributed with source")
        e=fusion.standardize_expression(pd.read_csv(r/"02_expression_teacher/expression_teacher_scores.tsv",sep="\t"))
        h=fusion.standardize_histology(pd.read_csv(r/"03_histology_teacher/histology_teacher_scores.tsv",sep="\t"))
        joined=fusion.merge_teacher_modalities(e,h)
        self.assertEqual(int((joined.expression_available & joined.histology_available).sum()),651)
        self.assertEqual(len(joined),34881)
        self.assertFalse(joined.duplicated(["sample_id","drug_key"]).any())


if __name__ == "__main__":
    unittest.main()
