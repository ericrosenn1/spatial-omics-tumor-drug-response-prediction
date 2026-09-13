"""Read-only numerical dependency audit for a versioned corrected teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from teacher_expression_contract import score_expression_artifact
from teacher_governance_lib import normalize_key


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit(teacher_root, v2_root, output, old_teacher=None):
    output.mkdir(parents=True, exist_ok=True)
    handoff = teacher_root / "05_prediction_ready_teacher"
    teacher_path = handoff / "visium_fused_teacher_table.tsv"
    teacher = pd.read_csv(teacher_path, sep="\t", low_memory=False)
    expr = pd.read_csv(teacher_root / "02_expression_teacher/expression_teacher_scores.tsv", sep="\t")
    hist = pd.read_csv(teacher_root / "03_histology_teacher/histology_teacher_scores.tsv", sep="\t", low_memory=False)
    for table in (teacher, expr, hist):
        table["drug_key"] = table["drug_key"].map(normalize_key)
    num = pd.read_csv(handoff / "model_input_numeric.csv")
    manifest = pd.read_csv(handoff / "feature_manifest.csv")
    checks = []

    def check(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": str(evidence)})
        print(name, "PASS" if passed else "FAIL", str(evidence), flush=True)

    for name, table in [("teacher", teacher), ("expression", expr), ("histology", hist)]:
        check(name + "_unique_keys", not table.duplicated(["sample_id", "drug_key"]).any(), len(table))
    expr_keys = set(map(tuple, expr[["sample_id", "drug_key"]].values))
    hist_keys = set(map(tuple, hist[["sample_id", "drug_key"]].values))
    teacher_keys = set(map(tuple, teacher[["sample_id", "drug_key"]].values))
    both_keys = set(map(tuple, teacher.loc[teacher.modality_used == "both", ["sample_id", "drug_key"]].values))
    check("restored_651_overlap", both_keys == expr_keys & hist_keys and len(both_keys) == 651, len(both_keys))
    check("union_keys", teacher_keys == expr_keys | hist_keys, len(teacher_keys))
    check("numeric_teacher_samples", set(num.sample_id) == set(teacher.sample_id) and num.sample_id.is_unique, num.sample_id.nunique())
    check("feature_manifest", set(manifest.loc[manifest.included, "feature"]) == set(num.columns) - {"sample_id"}, len(num.columns)-1)
    residual_error = np.max(np.abs(teacher.fused_residual_vs_prior - (teacher.fused_prob_responder - teacher.treatment_prior)))
    probability_error = np.max(np.abs(teacher.fused_prob_responder - np.clip(teacher.treatment_prior + teacher.expression_delta_vs_prior + teacher.histology_delta_vs_prior, .01, .99)))
    check("residual_arithmetic", residual_error < 1e-12, residual_error)
    check("fusion_arithmetic", probability_error < 1e-12, probability_error)
    check("expression_no_prior_fallback", ~expr.expression_teacher_mode.str.contains("prior_only").any(), expr.expression_teacher_mode.unique())
    dataset_root = v2_root / "02_build_modeling_dataset/03_modeling_datasets"
    pairs_path = dataset_root / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"
    header = pd.read_csv(pairs_path, sep="\t", nrows=0).columns
    selected = [c for c in ["sample_id", "drug_key", "fused_prob_responder", "fused_residual_vs_prior", "treatment_prior"] if c in header]
    pairs = pd.read_csv(pairs_path, sep="\t", usecols=selected)
    check("v2_teacher_key_set", set(map(tuple, pairs[["sample_id", "drug_key"]].values)) == teacher_keys and not pairs.duplicated(["sample_id", "drug_key"]).any(), len(pairs))
    joined = teacher[selected].merge(pairs, on=["sample_id", "drug_key"], suffixes=("_teacher", "_v2"), validate="one_to_one")
    for c in selected[2:]:
        err = float(np.nanmax(np.abs(joined[c+"_teacher"] - joined[c+"_v2"])))
        check("v2_"+c, err < 1e-12, err)
    features_path = dataset_root / "v2_spatial_features_broad_governed_candidate_pool.tsv"
    v2num = pd.read_csv(features_path, sep="\t").set_index("sample_id").sort_index()
    source_num = num.set_index("sample_id").reindex(index=v2num.index, columns=v2num.columns)
    check("v2_numeric_feature_values", np.allclose(source_num.to_numpy(dtype=float), v2num.to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True), list(v2num.shape))
    pseudo = pd.read_csv(teacher_root / "02_expression_teacher/visium_pseudobulk_expression_CANONICAL_LOG1P_CPM.tsv.gz", sep="\t").set_index("sample_id")
    # The canonical builder scored its float32 M matrix before TSV export.
    # Restore that recorded dtype rather than changing pipeline arithmetic.
    gene_cols = [c for c in pseudo if c.startswith("ENSG")]
    pseudo[gene_cols] = pseudo[gene_cols].astype(np.float32)
    scoring = []
    for drug, group in expr.groupby("drug_key", sort=True):
        paths = group.expression_model_source.unique()
        if len(paths) != 1:
            raise ValueError("Ambiguous model source")
        artifact = joblib.load(paths[0])
        # The original float32 PCA transform used multithread BLAS. One-thread
        # BLAS follows another floating-point reduction path (up to 4.4e-7 in
        # raw probabilities); two threads reproduces the saved output exactly.
        with warnings.catch_warnings(), threadpool_limits(limits=2):
            warnings.simplefilter("ignore")
            raw, cal, coverage = score_expression_artifact(artifact, pseudo.loc[group.sample_id])
        raw_error = float(np.max(np.abs(raw-group.expression_prob_raw.to_numpy())))
        cal_error = float(np.max(np.abs(cal-group.expression_prob_calibrated.to_numpy())))
        check("artifact_scoring:"+drug, raw_error < 1e-10 and cal_error < 1e-10, {"raw_error":raw_error,"calibrated_error":cal_error})
        scoring.append({"drug_key": drug, "artifact":paths[0], "sha256":sha(paths[0]), "raw_max_abs_error":raw_error, "calibrated_max_abs_error":cal_error, "calibration_method":artifact.get("calibration_method"), "base_classes":str(artifact["base_model"].classes_), "calibrator_classes":str(getattr(artifact.get("calibrator"), "classes_", None)), "fitted_pipeline_steps":str(list(artifact["base_model"].named_steps)), **coverage})
    pd.DataFrame(scoring).to_csv(output/"teacher_artifact_scoring_audit.tsv", sep="\t", index=False)
    if old_teacher:
        old = pd.read_csv(old_teacher,sep="\t",low_memory=False)
        compare = old[["sample_id","drug_key","fused_prob_responder","fused_residual_vs_prior","modality_used"]].merge(teacher[["sample_id","drug_key","fused_prob_responder","fused_residual_vs_prior","modality_used"]], on=["sample_id","drug_key"],how="outer",suffixes=("_old","_corrected"),validate="one_to_one",indicator=True)
        compare["probability_change"] = compare.fused_prob_responder_corrected - compare.fused_prob_responder_old
        compare.to_csv(output/"teacher_old_vs_corrected_rows.tsv.gz",sep="\t",index=False)
        compare.groupby("drug_key").agg(n_rows=("sample_id","size"),max_abs_probability_change=("probability_change", lambda x:x.abs().max()),mean_probability_change=("probability_change","mean"),changed_rows=("probability_change",lambda x:int((x.abs()>1e-12).sum()))).to_csv(output/"teacher_old_vs_corrected_treatments.tsv",sep="\t")
    protected = [teacher_path, handoff/"model_input_numeric.csv",handoff/"feature_manifest.csv",pairs_path,features_path]
    pd.DataFrame([{"path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in protected]).to_csv(output/"teacher_dependency_hashes.tsv",sep="\t",index=False)
    pd.DataFrame(checks).to_csv(output/"teacher_numerical_checks.tsv",sep="\t",index=False)
    result = {"status":"PASS" if all(c["passed"] for c in checks) else "FAIL", "teacher_rows":len(teacher), "samples":teacher.sample_id.nunique(),"treatment_keys":teacher.drug_key.nunique(),"modality_counts":teacher.modality_used.value_counts().to_dict(),"checks":len(checks),"failures":[c for c in checks if not c["passed"]],"scope":"Numerical teacher and V2 input dependency audit; not independent clinical validation or patient grouping verification"}
    (output/"teacher_numerical_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--teacher-root",type=Path,required=True)
    p.add_argument("--v2-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--old-teacher",type=Path)
    a=p.parse_args()
    audit(a.teacher_root,a.v2_root,a.output,a.old_teacher)
