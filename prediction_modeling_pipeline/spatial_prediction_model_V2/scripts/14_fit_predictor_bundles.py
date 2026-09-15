"""Fit only the missing deployable Step07 estimators; retain development status."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spm_v2.model_training import make_xgb_pipeline,select_features_training_only
from spm_v2.predictor_bundle import SpatialPredictorBundle,save_bundle,load_bundle,load_bundle_directory


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reference_equivalence(actual, expected, imputer=None):
    """Reuse the historical attachment allowance without changing either table."""
    allowed_labels={"label__endothelial_high","label__vascular_endothelial","label__myeloid_low"}
    corrections=0
    for c in actual.columns:
        a=actual[c].to_numpy(float);e=expected[c].to_numpy(float)
        different_missing=np.isnan(a)!=np.isnan(e)
        permitted=(c in allowed_labels) & np.isnan(a) & (e==0)
        if np.any(different_missing & ~permitted):
            raise ValueError(f"Unexplained reference missingness mismatch for {c}")
        if not np.allclose(a[~different_missing],e[~different_missing],rtol=0,atol=1e-10,equal_nan=True):
            raise ValueError(f"Raw reference value mismatch for {c}")
        corrections+=int(different_missing.sum())
    if imputer is not None and not np.allclose(imputer.transform(actual),imputer.transform(expected),rtol=0,atol=1e-10):
        raise ValueError("Saved preprocessing changes after reference attachment")
    return corrections


def numeric_handoff_path(root):
    """Resolve the flat precomputed handoff or original teacher-stage layout.

    The public precomputed command writes a flat three-file directory. Original
    teacher-builder runs place those files under 05_prediction_ready_teacher.
    Two matching locations are ambiguous and require an explicit narrower root.
    """
    root = Path(root)
    candidates = [root / "model_input_numeric.csv",
                  root / "05_prediction_ready_teacher/model_input_numeric.csv"]
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError("Ambiguous numeric handoff: both flat and teacher-stage files exist; supply the intended handoff directory")
    if not found:
        raise FileNotFoundError(f"No numeric feature handoff found under {root}")
    return found[0]


def attach_reference(existing, raw_path, numeric_path, output):
    """Attach the verified reference without repeating any final estimator fit."""
    from spm_v2.feature_reference import FeatureReference
    raw=pd.read_csv(raw_path);numeric=pd.read_csv(numeric_path)
    raw=raw[raw.sample_id.astype(str).isin(numeric.sample_id.astype(str))]
    if set(raw.sample_id.astype(str))!=set(numeric.sample_id.astype(str)):
        raise ValueError("Raw feature reference sample set mismatch")
    raw=raw.set_index("sample_id").loc[numeric.sample_id].reset_index()
    reference=FeatureReference().fit(raw)
    transformed=reference.transform(raw)
    manifests=[];checks=[];features=[];predictions=[]
    bundles=load_bundle_directory(existing)
    for i,bundle in enumerate(bundles):
        expected=bundle.feature_matrix(numeric,mode="training_reproduction")
        actual=transformed[bundle.ordered_features].apply(pd.to_numeric,errors="raise").replace([np.inf,-np.inf],np.nan)
        corrections=validate_reference_equivalence(actual,expected,bundle.pipeline.named_steps["imputer"])
        before=bundle.predict(numeric,mode="training_reproduction")
        bundle.feature_reference=reference
        bundle.provenance["feature_reference"]={"status":"PASS_FITTED_PREPROCESSING_AND_PREDICTION_EQUIVALENCE","raw_source":str(raw_path.resolve()),"raw_source_sha256":sha(raw_path),"source_bundle_directory":str(existing.resolve()),"original_false_labels_now_explicit_na_cells_for_this_model":corrections,"missingness_rule":"Keep NA; original saved median imputation reproduces original False encoding. No biological absence is inferred."}
        after=bundle.predict(raw,"raw_reference","training_reproduction")
        if not np.array_equal(before.predicted_residual_vs_prior,after.predicted_residual_vs_prior):
            raise ValueError("Attached reference changes original fitted predictions")
        filename=f"treatment_{i+1:02d}_{hashlib.sha256(bundle.drug_key.encode()).hexdigest()[:12]}.joblib"
        save_bundle(bundle,output/filename)
        loaded=load_bundle(output/filename)
        reloaded=loaded.predict(raw,"raw_reference","training_reproduction")
        reverse=loaded.predict(raw.iloc[::-1,::-1],"raw_reference","training_reproduction").iloc[::-1].reset_index(drop=True)
        # A deterministic representative set spans the full reference range;
        # external-sample tests exercise every available transfer row separately.
        indices=sorted(set([0,len(raw)//4,len(raw)//2,3*len(raw)//4,len(raw)-1]))
        singles=pd.concat([loaded.predict(raw.iloc[[j]],"raw_reference","training_reproduction") for j in indices],ignore_index=True)
        if not after.equals(reloaded) or not after.equals(reverse) or not after.iloc[indices].reset_index(drop=True).equals(singles):
            raise ValueError("Raw-reference reload/reorder/single-batch invariance failure")
        predictions.append(after)
        checks.append({"drug_key":bundle.drug_key,"original_prediction_equal":True,"saved_imputed_features_equal":True,"reload_equal":True,"reordering_equal":True,"single_batch_equal":True,"representative_single_samples":len(indices),"explicit_na_label_cells":corrections})
        for position,c in enumerate(bundle.ordered_features):features.append({"drug_key":bundle.drug_key,"feature_position":position,"feature_name":c})
        manifests.append({"drug_key":bundle.drug_key,"final_model_rank":bundle.provenance["final_model_rank"],"bundle_file":filename,"sha256":sha(output/filename),"n_features":len(bundle.ordered_features),"treatment_prior":bundle.treatment_prior,"support_status":bundle.support_status})
        print(f"[{i+1}/{len(bundles)}] attached reference and verified {bundle.drug_key}",flush=True)
    pd.DataFrame(manifests).to_csv(output/"bundle_manifest.tsv",sep="\t",index=False)
    pd.DataFrame(features).to_csv(output/"ordered_feature_manifest.tsv",sep="\t",index=False)
    pd.DataFrame(checks).to_csv(output/"raw_reference_reproducibility_checks.tsv",sep="\t",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/"all_data_fitted_raw_reference_predictions.tsv",sep="\t",index=False)
    (output/"bundle_summary.json").write_text(json.dumps({"status":"PASS","bundles":len(manifests),"fitted_estimator_refits":0,"source_bundles":str(existing.resolve()),"raw_source":str(raw_path.resolve()),"raw_sha256":sha(raw_path),"reference_fit_samples":len(raw),"reference_feature_module_sha256":sha(Path(__file__).resolve().parents[1]/"src/spm_v2/feature_reference.py"),"independent_validation_status":"UNKNOWN; development models","missing_label_policy":"Explicit NA with original fitted imputer; all original predictions identical"},indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True,type=Path)
    p.add_argument("--teacher-root",required=True,type=Path,help="Flat three-file handoff, or a teacher output root containing 05_prediction_ready_teacher")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--raw-feature-table",type=Path)
    p.add_argument("--existing-bundles",type=Path,help="Attach a verified raw reference to existing estimators; do not refit")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/"bundle_manifest.tsv").exists():
        raise ValueError("Bundle output already exists; select a new versioned destination")
    numeric_path = numeric_handoff_path(a.teacher_root)
    if a.existing_bundles:
        if not a.raw_feature_table:raise ValueError("Existing-bundle attachment requires --raw-feature-table")
        attach_reference(a.existing_bundles,a.raw_feature_table,numeric_path,a.output)
        return
    step=a.run_root/"07_filtered_per_treatment_residual_models"
    manifest_path=step/"03_final_models/final_model_manifest.tsv"
    evidence_path=step/"04_shap_feature_evidence/per_treatment_final_shap_feature_long.tsv"
    registry_path=a.run_root/"05_residual_biology_registry/03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv"
    pair_path=a.run_root/"02_build_modeling_dataset/03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"
    manifest=pd.read_csv(manifest_path,sep="\t").sort_values("final_model_rank")
    evidence=pd.read_csv(evidence_path,sep="\t")
    features=pd.read_csv(registry_path,sep="\t").feature_name.astype(str).tolist()
    pairs=pd.read_csv(pair_path,sep="\t",usecols=["sample_id","drug_key","fused_residual_vs_prior","treatment_prior"]+features)
    numeric=pd.read_csv(numeric_path)
    hashes={str(f.resolve()):sha(f) for f in [manifest_path,evidence_path,registry_path,pair_path,numeric_path]}
    reference=None;reference_report={"status":"NOT_REQUESTED"}
    if a.raw_feature_table:
        from spm_v2.feature_reference import FeatureReference
        raw=pd.read_csv(a.raw_feature_table)
        raw=raw[raw.sample_id.astype(str).isin(numeric.sample_id.astype(str))].copy()
        if set(raw.sample_id.astype(str))!=set(numeric.sample_id.astype(str)):
            raise ValueError("Raw feature reference sample set differs from corrected numeric cohort")
        reference=FeatureReference().fit(raw)
        transformed=reference.transform(raw).set_index("sample_id")
        expected=numeric.set_index("sample_id").loc[transformed.index]
        used=evidence.feature_name.unique().tolist()
        validate_reference_equivalence(transformed[used],expected[used])
        diffs=[]
        for col in used:
            actual=pd.to_numeric(transformed[col],errors="raise").to_numpy(float)
            saved=pd.to_numeric(expected[col],errors="raise").to_numpy(float)
            equal=np.allclose(actual,saved,rtol=0,atol=1e-10,equal_nan=True)
            diffs.append({"feature_name":col,"equal":bool(equal),"max_abs_error":float(np.nanmax(np.abs(actual-saved)))})
        pd.DataFrame(diffs).to_csv(a.output/"feature_reference_equivalence.tsv",sep="\t",index=False)
        hashes[str(a.raw_feature_table.resolve())]=sha(a.raw_feature_table)
        reference_report={"status":"PENDING_FITTED_PREPROCESSING_EQUIVALENCE","training_samples":len(raw),"selected_union_features":len(used)}
    environment={"python":platform.python_version(),"packages":{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","xgboost","joblib"]}}
    rows=[];selected_rows=[];predictions=[];checks=[]
    for row in manifest.itertuples():
        drug=str(row.drug_key); rank=int(row.final_model_rank)
        sub=pairs[pairs.drug_key.astype(str)==drug].copy()
        y=pd.to_numeric(sub.fused_residual_vs_prior,errors="raise")
        sub=sub[y.notna()].reset_index(drop=True);y=sub.fused_residual_vs_prior.astype(float)
        if sub.sample_id.duplicated().any():raise ValueError("Duplicate sample/treatment identity")
        X=sub[features].apply(pd.to_numeric,errors="raise").replace([np.inf,-np.inf],np.nan)
        selected=select_features_training_only(X,y,features,max_features=int(row.n_features_final_model),min_variance=1e-12)
        saved=evidence[evidence.drug_key.astype(str)==drug].set_index("feature_name")
        if set(selected)!=set(saved.index):raise ValueError(f"Reconstructed final selected set mismatch: {drug}")
        seed=42+(rank-1)+10000
        pipe=make_xgb_pipeline(random_state=seed,n_estimators=120,max_depth=2,learning_rate=.03,tree_method="hist",n_jobs=1)
        pipe.fit(X[selected],y)
        gain=pipe.named_steps["model"].feature_importances_
        gain_error=float(np.max(np.abs(gain-saved.loc[selected,"gain_importance"].to_numpy())))
        if gain_error>1e-7:raise ValueError(f"Refitted estimator gain evidence mismatch {drug}: {gain_error}")
        priors=pd.to_numeric(sub.treatment_prior,errors="raise").dropna().unique()
        if len(priors)!=1:raise ValueError(f"Prior-anchored reconstruction is ambiguous for {drug}")
        if X[selected].isna().all(axis=0).any():raise ValueError("Selected all-missing feature would be dropped by imputer")
        if reference is not None:
            actual_reference=transformed[selected].apply(pd.to_numeric,errors="raise").replace([np.inf,-np.inf],np.nan)
            saved_reference=expected[selected].apply(pd.to_numeric,errors="raise").replace([np.inf,-np.inf],np.nan)
            corrections=validate_reference_equivalence(actual_reference,saved_reference,pipe.named_steps["imputer"])
            if not np.array_equal(pipe.predict(actual_reference),pipe.predict(saved_reference)):
                raise ValueError("Raw reference changes fitted predictions")
            reference_report={**reference_report,"status":"PASS_FITTED_PREPROCESSING_AND_PREDICTION_EQUIVALENCE","original_false_labels_now_explicit_na_cells_for_this_model":corrections}
        bundle=SpatialPredictorBundle(drug,selected,pipe,float(priors[0]),numeric.sample_id.astype(str).tolist(),reference,{"input_hashes":hashes,"environment":environment,"final_model_rank":rank,"random_seed":seed,"settings":pipe.named_steps["model"].get_params(),"source_run":str(a.run_root.resolve()),"feature_reference":reference_report,"deployment_fit":"all available corrected teacher rows; held-out performance estimated separately"})
        filename=f"treatment_{rank:02d}_{hashlib.sha256(drug.encode()).hexdigest()[:12]}.joblib"
        save_bundle(bundle,a.output/filename)
        loaded=load_bundle(a.output/filename)
        original=bundle.predict(numeric,mode="training_reproduction")
        reloaded=loaded.predict(numeric,mode="training_reproduction")
        if not original.equals(reloaded):raise ValueError("Saved/reloaded predictions differ")
        reverse=loaded.predict(numeric.iloc[::-1,::-1],mode="training_reproduction").iloc[::-1].reset_index(drop=True)
        if not original.reset_index(drop=True).equals(reverse):raise ValueError("Row/column reorder changes predictions")
        singles=pd.concat([loaded.predict(numeric.iloc[[i]],mode="training_reproduction") for i in range(len(numeric))],ignore_index=True)
        if not original.reset_index(drop=True).equals(singles):raise ValueError("Single/batch predictions differ")
        predictions.append(original)
        checks.append({"drug_key":drug,"gain_max_abs_error":gain_error,"reload_equal":True,"reordering_equal":True,"single_batch_equal":True,"deterministic_rerun_equal":original.equals(loaded.predict(numeric,mode="training_reproduction"))})
        for position,(feature,g) in enumerate(zip(selected,gain)):
            selected_rows.append({"drug_key":drug,"feature_position":position,"feature_name":feature,"gain_importance":float(g)})
        rows.append({"drug_key":drug,"final_model_rank":rank,"bundle_file":filename,"sha256":sha(a.output/filename),"n_features":len(selected),"n_fitted_rows":len(sub),"treatment_prior":float(priors[0]),"random_seed":seed,"support_status":bundle.support_status})
        print(f"[{rank}/{len(manifest)}] saved and verified {drug}",flush=True)
    pd.DataFrame(rows).to_csv(a.output/"bundle_manifest.tsv",sep="\t",index=False)
    pd.DataFrame(selected_rows).to_csv(a.output/"ordered_feature_manifest.tsv",sep="\t",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(a.output/"all_data_fitted_predictions.tsv",sep="\t",index=False)
    pd.DataFrame(checks).to_csv(a.output/"bundle_reproducibility_checks.tsv",sep="\t",index=False)
    (a.output/"bundle_summary.json").write_text(json.dumps({"status":"PASS","bundles":len(rows),"source_model_set":"corrected Step07 final development entries","independent_validation_status":"UNKNOWN; do not inherit conditional or old validation labels","predictions":sum(len(x) for x in predictions),"environment":environment,"input_hashes":hashes,"feature_reference":reference_report},indent=2),encoding="utf-8")


if __name__=="__main__":main()
