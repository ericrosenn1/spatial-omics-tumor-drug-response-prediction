"""Verified extraction export cannot mix repeated internal sample identifiers."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_verified_spatial_feature_handoff import export, sha


def fixture(tmp_path):
    matrix = tmp_path / 'matrix.mtx'; matrix.write_bytes(b'seeded test matrix authority')
    raw = tmp_path / 'step09.csv'; pd.DataFrame({'sample_id':['SAMPLE_0000'], 'distance':[float('nan')], 'fraction':[0.25]}).to_csv(raw,index=False)
    mapping = tmp_path / 'map.tsv'; pd.DataFrame({'internal_sample_id':['SAMPLE_0000'],'transfer_sample_id':['GSM_TEST']}).to_csv(mapping,sep='\t',index=False)
    identity = tmp_path / 'identity.tsv'; pd.DataFrame({'sample_id':['GSM_TEST'],'source_matrix_path':['matrix.mtx'],'source_matrix_sha256':[sha(matrix)],'retained_barcode_subset':[True],'duplicate_retained_barcodes':[0]}).to_csv(identity,sep='\t',index=False)
    group={'mapping_path':'map.tsv','mapping_sha256':sha(mapping),'source07_path':'step09.csv','source07_sha256':sha(raw),'source09_path':'step09.csv','source09_sha256':sha(raw),'sample_ids':['GSM_TEST']}
    manifest=tmp_path/'sources.json';manifest.write_text(json.dumps([group]));return manifest,identity,tmp_path/'export.csv',raw,mapping,group


def test_relative_authority_paths_and_missing_values_preserved(tmp_path):
    manifest,identity,out,*_=fixture(tmp_path)
    result=export(manifest,identity,out)
    assert result.sample_id.tolist()==['GSM_TEST'] and result.original_internal_sample_id.tolist()==['SAMPLE_0000']
    assert pd.isna(result.loc[0,'distance']) and result.loc[0,'fraction']==0.25
    with pytest.raises(FileExistsError):export(manifest,identity,out)


@pytest.mark.parametrize('defect',['matrix_hash','duplicate_map','false_barcode_audit','duplicate_header','group_collision'])
def test_identity_failures_never_create_output(tmp_path,defect):
    manifest,identity,out,raw,mapping,group=fixture(tmp_path)
    if defect=='matrix_hash':(tmp_path/'matrix.mtx').write_bytes(b'changed authority')
    if defect=='duplicate_map':
        frame=pd.read_csv(mapping,sep='\t');pd.concat([frame,frame]).to_csv(mapping,sep='\t',index=False);group['mapping_sha256']=sha(mapping)
    if defect=='false_barcode_audit':
        frame=pd.read_csv(identity,sep='\t');frame['retained_barcode_subset']='False';frame.to_csv(identity,sep='\t',index=False)
    if defect=='duplicate_header':
        raw.write_text('sample_id,x,x\nSAMPLE_0000,1,2\n');group['source07_sha256']=group['source09_sha256']=sha(raw)
    manifest.write_text(json.dumps([group,group] if defect=='group_collision' else [group]))
    with pytest.raises(ValueError):export(manifest,identity,out)
    assert not out.exists()
