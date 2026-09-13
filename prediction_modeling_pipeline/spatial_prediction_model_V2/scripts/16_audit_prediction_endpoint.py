"""Independently exercise saved bundles on actual external spatial feature rows."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spm_v2.predictor_bundle import load_bundle_directory,read_feature_table


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundles",type=Path,required=True)
    p.add_argument("--prepared-features",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--contributions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=read_feature_table(a.prepared_features)
    saved=pd.read_csv(a.predictions,sep="\t")
    contrib=pd.read_csv(a.contributions,sep="\t")
    bundles=load_bundle_directory(a.bundles)
    expected_keys={(s,b.drug_key) for s in raw.sample_id.astype(str) for b in bundles}
    if set(map(tuple,saved[["sample_id","drug_key"]].values))!=expected_keys or saved.duplicated(["sample_id","drug_key"]).any():
        raise ValueError("Saved external prediction keys mismatch")
    checks=[];coverage=[]
    for b in bundles:
        p1=b.predict(raw,"raw_reference")
        p2=b.predict(raw,"raw_reference")
        reverse=b.predict(raw.iloc[::-1,::-1],"raw_reference").iloc[::-1].reset_index(drop=True)
        single=pd.concat([b.predict(raw.iloc[[i]],"raw_reference") for i in range(len(raw))],ignore_index=True)
        unrelated=raw.iloc[[0]].copy().assign(sample_id="AUDIT_UNRELATED_EXTERNAL_SAMPLE")
        extended=b.predict(pd.concat([raw,unrelated],ignore_index=True),"raw_reference").iloc[:len(raw)].reset_index(drop=True)
        for label,actual in [("deterministic",p2),("reordering",reverse),("single_batch",single),("unrelated_batch_extension",extended)]:
            if not p1.equals(actual):raise ValueError(f"{label} failure: {b.drug_key}")
        old=saved[saved.drug_key==b.drug_key].set_index("sample_id").loc[p1.sample_id]
        if not np.allclose(old.predicted_residual_vs_prior,p1.predicted_residual_vs_prior,rtol=0,atol=1e-15):
            raise ValueError("Reloaded external predictions differ from saved table")
        # Strict numeric model API still rejects columns absent from the schema.
        transformed=b.feature_reference.transform(raw)
        try:b.predict(transformed.drop(columns=b.ordered_features[0]))
        except ValueError as e:
            if "Missing required spatial feature" not in str(e):raise
        else:raise ValueError("Missing required model feature did not fail")
        try:b.predict(raw.iloc[[0]].assign(sample_id=b.training_sample_ids[0]),"raw_reference")
        except ValueError as e:
            if "identifier collision" not in str(e):raise
        else:raise ValueError("Train/transfer identifier collision did not fail")
        c=contrib[contrib.drug_key==b.drug_key]
        if c.duplicated(["sample_id","feature_name"]).any() or len(c)!=len(raw)*(len(b.ordered_features)+1):
            raise ValueError("Contribution feature/sample key mismatch")
        summed=c.groupby("sample_id").residual_contribution.sum().loc[p1.sample_id].to_numpy()
        contribution_error=float(np.max(np.abs(summed-p1.predicted_residual_vs_prior.to_numpy())))
        if not np.allclose(summed,p1.predicted_residual_vs_prior,rtol=1e-5,atol=1e-7):
            raise ValueError("Contributions do not add to residual prediction")
        X=b.feature_matrix(raw,"raw_reference")
        for sample_i,sid in enumerate(raw.sample_id):
            for feature in b.ordered_features:
                v=X.iloc[sample_i][feature]
                coverage.append({"sample_id":sid,"drug_key":b.drug_key,"feature_name":feature,"feature_value":v,"status":"OBSERVED_OR_DERIVED_FROM_AVAILABLE_INPUTS" if np.isfinite(v) else "UNKNOWN_FITTED_TRAINING_MEDIAN_IMPUTATION","imputed_value":b.pipeline.named_steps["imputer"].statistics_[b.ordered_features.index(feature)] if not np.isfinite(v) else np.nan})
        checks.append({"drug_key":b.drug_key,"samples":len(raw),"reload_equal":True,"single_batch_equal":True,"row_column_reordering_equal":True,"unrelated_batch_extension_equal":True,"deterministic_equal":True,"missing_required_feature_rejected":True,"train_transfer_collision_rejected":True,"contribution_key_count":len(c),"contribution_sum_max_abs_error":contribution_error,"status":"PASS"})
        print("PASS",b.drug_key,flush=True)
    pd.DataFrame(checks).to_csv(a.output/"external_prediction_endpoint_checks.tsv",sep="\t",index=False)
    pd.DataFrame(coverage).to_csv(a.output/"per_model_feature_coverage.tsv",sep="\t",index=False)
    result={"status":"PASS","samples":len(raw),"model_entries":len(bundles),"predictions":len(saved),"sample_treatment_keys":len(expected_keys),"feature_contribution_rows":len(contrib),"all_tests":"All actual external samples tested alone and in batch for every bundle","source_hashes":{str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.prepared_features,a.predictions,a.contributions,a.bundles/"bundle_manifest.tsv"]},"interpretation":"Engineering inference route tested; development models and cross-domain teacher targets do not establish external drug response accuracy"}
    (a.output/"external_prediction_endpoint_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":main()
