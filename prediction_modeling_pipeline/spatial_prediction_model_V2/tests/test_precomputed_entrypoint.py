import importlib.util
from pathlib import Path
import pandas as pd
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/16_prepare_precomputed_teacher.py'
spec=importlib.util.spec_from_file_location('precomputed_entry',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def inputs(tmp_path):
    t=tmp_path/'teacher.tsv';x=tmp_path/'features.csv';m=tmp_path/'manifest.csv'
    pd.DataFrame({'sample_id':['S1','S2'],'drug_key':['full profile','full profile'],'fused_prob_responder':[.4,.6],'treatment_prior':[.5,.5],'fused_residual_vs_prior':[-.1,.1]}).to_csv(t,sep='\t',index=False)
    pd.DataFrame({'sample_id':['S2','S1'],'feature_a':[1,None]}).to_csv(x,index=False)
    pd.DataFrame({'feature':['feature_a']}).to_csv(m,index=False)
    return t,x,m

def test_preserves_teacher_missingness_and_source_bytes(tmp_path):
    t,x,m=inputs(tmp_path);out=tmp_path/'new'
    r=mod.prepare(t,x,m,out,mod.digest(t),mod.digest(x),mod.digest(m));assert r['rows']==2
    assert (out/'model_input_numeric.csv').read_bytes()==x.read_bytes()
    assert pd.isna(pd.read_csv(out/'model_input_numeric.csv').feature_a.iloc[1])

def test_rejects_wrong_authority_hash(tmp_path):
    t,x,m=inputs(tmp_path)
    with pytest.raises(ValueError,match='SHA256'):mod.prepare(t,x,m,tmp_path/'new','0'*64,mod.digest(x),mod.digest(m))
    assert not (tmp_path/'new').exists()

def test_rejects_inconsistent_residual(tmp_path):
    t,x,m=inputs(tmp_path);d=pd.read_csv(t,sep='\t');d.loc[0,'fused_residual_vs_prior']=.25;d.to_csv(t,sep='\t',index=False)
    with pytest.raises(ValueError,match='residual'):mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),mod.digest(x),mod.digest(m))

def test_rejects_duplicate_raw_header(tmp_path):
    t,x,m=inputs(tmp_path);x.write_text('sample_id,feature_a,feature_a\nS1,1,2\nS2,3,4\n')
    with pytest.raises(ValueError,match='raw table headers'):mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),mod.digest(x),mod.digest(m))

def test_rejects_duplicate_teacher_identity(tmp_path):
    t,x,m=inputs(tmp_path);d=pd.read_csv(t,sep='\t');pd.concat([d,d.iloc[:1]]).to_csv(t,sep='\t',index=False)
    with pytest.raises(ValueError,match='identities'):mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),mod.digest(x),mod.digest(m))

@pytest.mark.parametrize('which',['spatial_features','feature_manifest'])
def test_rejects_same_identity_inputs_with_wrong_hash(tmp_path,which):
    t,x,m=inputs(tmp_path);xh=mod.digest(x);mh=mod.digest(m)
    with pytest.raises(ValueError,match=which+' SHA256'):
        mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),'0'*64 if which=='spatial_features' else xh,'0'*64 if which=='feature_manifest' else mh)
    assert not (tmp_path/'new').exists()

@pytest.mark.parametrize('bad',['not_numeric',float('inf')])
def test_rejects_nonnumeric_or_infinite_features(tmp_path,bad):
    t,x,m=inputs(tmp_path);d=pd.read_csv(x);d['feature_a']=d.feature_a.astype(object);d.loc[0,'feature_a']=bad;d.to_csv(x,index=False)
    with pytest.raises(ValueError,match='numeric|Infinite'):
        mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),mod.digest(x),mod.digest(m))
    assert not (tmp_path/'new').exists()

def test_rejects_wrong_manifest_order(tmp_path):
    t,x,m=inputs(tmp_path);d=pd.read_csv(x);d['feature_b']=[2.,3.];d.to_csv(x,index=False)
    pd.DataFrame({'feature':['feature_b','feature_a']}).to_csv(m,index=False)
    with pytest.raises(ValueError,match='ordered'):
        mod.prepare(t,x,m,tmp_path/'new',mod.digest(t),mod.digest(x),mod.digest(m))

def test_gzip_authority_preserves_decompressed_teacher_bytes(tmp_path):
    import gzip
    t,x,m=inputs(tmp_path);z=tmp_path/'teacher.tsv.gz';z.write_bytes(gzip.compress(t.read_bytes(),mtime=0))
    out=tmp_path/'new';r=mod.prepare(z,x,m,out,mod.digest(z),mod.digest(x),mod.digest(m))
    assert (out/'visium_fused_teacher_table.tsv').read_bytes()==t.read_bytes()
    assert set(r['authority'])=={'teacher','spatial_features','feature_manifest'}
