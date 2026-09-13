"""Score verified spatial feature rows with saved V2 residual model bundles."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from spm_v2.predictor_bundle import load_bundle_directory,read_feature_table


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundles",required=True,type=Path)
    p.add_argument("--features",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--representation",choices=["saved_numeric","raw_reference"],default="saved_numeric")
    p.add_argument("--mode",choices=["external","training_reproduction","heldout_evaluation_reproduction"],default="external")
    p.add_argument("--contributions",type=Path)
    p.add_argument("--raw-missingness-manifest",type=Path,help="Reviewed TSV feature_name/reason/source_evidence permitting absent raw extraction fields to remain explicit NA")
    p.add_argument("--treatment-keys",type=Path,help="Optional TSV with exact drug_key values; reject unsupported keys")
    a=p.parse_args()
    table=read_feature_table(a.features)
    bundles=load_bundle_directory(a.bundles)
    if a.raw_missingness_manifest:
        if a.representation!="raw_reference":raise ValueError("Raw missingness manifest requires raw_reference representation")
        missingness=pd.read_csv(a.raw_missingness_manifest,sep="\t")
        required={"feature_name","reason","source_evidence"}
        if not required.issubset(missingness) or missingness.feature_name.duplicated().any():
            raise ValueError("Invalid reviewed raw missingness manifest")
        if missingness[list(required)].isna().any().any():raise ValueError("Raw missingness justification must be explicit")
        absent=[c for c in missingness.feature_name.astype(str) if c not in table]
        table=pd.concat([table,pd.DataFrame(np.nan,index=table.index,columns=absent)],axis=1)
    if a.treatment_keys:
        wanted=set(pd.read_csv(a.treatment_keys,sep="\t").drug_key.astype(str))
        missing=wanted-{b.drug_key for b in bundles}
        if missing:raise ValueError(f"Unsupported treatment keys: {sorted(missing)}")
        bundles=[b for b in bundles if b.drug_key in wanted]
    if not bundles:raise ValueError("No requested model bundles")
    results=pd.concat([b.predict(table,a.representation,a.mode) for b in bundles],ignore_index=True)
    expected=len(table)*len(bundles)
    if len(results)!=expected or results.duplicated(["sample_id","drug_key"]).any():raise ValueError("Unexpected output identities")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix(a.output.suffix+".tmp")
    results.to_csv(temporary,sep="\t",index=False);temporary.replace(a.output)
    if a.contributions:
        contrib=pd.concat([b.contributions(table,a.representation,a.mode) for b in bundles],ignore_index=True)
        a.contributions.parent.mkdir(parents=True,exist_ok=True);contrib.to_csv(a.contributions,sep="\t",index=False)
    print(json.dumps({"status":"PASS","samples":len(table),"treatments":len(bundles),"rows":expected,"output":str(a.output),"quantity":"fitted residual and prior-anchored teacher estimate, not clinical efficacy probability"},indent=2))


if __name__=="__main__":main()
