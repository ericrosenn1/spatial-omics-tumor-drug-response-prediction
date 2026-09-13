"""Rebuild raw-count pseudobulks read-only and compare every saved float32 value."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from teacher_expression_contract import read_canonical_pseudobulk


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher-root",type=Path,required=True)
    p.add_argument("--processed-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    saved=pd.read_csv(a.teacher_root/"02_expression_teacher/visium_pseudobulk_expression_CANONICAL_LOG1P_CPM.tsv.gz",sep="\t").set_index("sample_id")
    genes=[c for c in saved if c.startswith("ENSG")]
    rows=[]
    for i,(sid,row) in enumerate(saved.iterrows()):
        reconstructed=read_canonical_pseudobulk(a.processed_root/sid/"adata/02_processed.h5ad",genes)
        actual=np.asarray([reconstructed[g] for g in genes],dtype=np.float32)
        expected=row[genes].to_numpy(dtype=np.float32)
        mismatch=int(np.count_nonzero(actual!=expected))
        result={"sample_id":sid,"n_genes":len(genes),"unequal_float32_values":mismatch,"max_abs_difference":float(np.max(np.abs(actual-expected))),"status":"PASS" if mismatch==0 else "FAIL"}
        rows.append(result)
        temporary=a.output/"teacher_raw_pseudobulk_reconstruction.tmp.tsv"
        pd.DataFrame(rows).to_csv(temporary,sep="\t",index=False)
        temporary.replace(a.output/"teacher_raw_pseudobulk_reconstruction.tsv")
        print(f"[{i+1}/{len(saved)}] {sid} {result['status']} mismatches={mismatch}",flush=True)
    result={"status":"PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL","samples":len(rows),"genes":len(genes),"method":"Retained raw spot counts; duplicate normalized Ensembl IDs summed before CPM and log1p; cast to recorded float32; exact comparison"}
    (a.output/"teacher_raw_pseudobulk_reconstruction.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if result["status"] != "PASS":raise SystemExit(1)


if __name__ == "__main__":main()
