"""Create a validated, versioned handoff from an explicit precomputed teacher.

This is a training/reproduction entry point. It never manufactures teacher
labels for an unseen Visium sample and never substitutes a default public file.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,shutil
import numpy as np
import pandas as pd

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(teacher,features,feature_manifest,output,expected_teacher_sha256,
            expected_features_sha256,expected_manifest_sha256):
    teacher=Path(teacher);features=Path(features);feature_manifest=Path(feature_manifest);output=Path(output)
    authorities={
        'teacher':(teacher,expected_teacher_sha256),
        'spatial_features':(features,expected_features_sha256),
        'feature_manifest':(feature_manifest,expected_manifest_sha256),
    }
    for label,(path,expected) in authorities.items():
        if digest(path).lower()!=expected.lower():raise ValueError(f'Precomputed {label} SHA256 does not match explicitly requested authority')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output must be a new or empty versioned handoff directory')
    for path,delimiter in [(teacher,'\t'),(features,','),(feature_manifest,',')]:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8-sig',newline='') as handle:
            header=next(csv.reader(handle,delimiter=delimiter))
        if len(header)!=len(set(header)):raise ValueError('Duplicate raw table headers: '+str(path))
    t=pd.read_csv(teacher,sep='\t');x=pd.read_csv(features);m=pd.read_csv(feature_manifest)
    required=['sample_id','drug_key','fused_prob_responder','fused_residual_vs_prior','treatment_prior']
    if any(c not in t for c in required):raise ValueError('Teacher is missing required response contract columns')
    if t.duplicated(['sample_id','drug_key']).any() or t[required[:2]].isna().any().any() or any(t[c].astype(str).str.strip().eq('').any() for c in required[:2]):raise ValueError('Duplicate or missing teacher identities')
    if 'sample_id' not in x or x.columns.duplicated().any() or x.sample_id.isna().any() or x.sample_id.duplicated().any() or x.sample_id.astype(str).str.strip().eq('').any():raise ValueError('Duplicate or missing spatial identities/features')
    if set(t.sample_id)!=set(x.sample_id):raise ValueError('Teacher and spatial feature sample sets differ')
    feature_order=[c for c in x if c!='sample_id']
    if 'feature' not in m or m.feature.isna().any() or m.feature.duplicated().any() or m.feature.tolist()!=feature_order:raise ValueError('Feature manifest does not match ordered numeric handoff')
    try:numeric=x[feature_order].apply(pd.to_numeric,errors='raise')
    except (TypeError,ValueError) as exc:raise ValueError('Spatial features must be numeric or explicitly missing') from exc
    if np.isinf(numeric.to_numpy(dtype=float)).any():raise ValueError('Infinite spatial feature values are not explicit missingness')
    vals=t[required[2:]].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(vals.to_numpy()).all():raise ValueError('Nonfinite teacher probability/prior/residual')
    if not np.allclose(vals.fused_residual_vs_prior,vals.fused_prob_responder-vals.treatment_prior,rtol=0,atol=1e-12):raise ValueError('Teacher residual does not equal probability minus recorded prior')
    if not vals[['fused_prob_responder','treatment_prior']].ge(0).all().all() or not vals[['fused_prob_responder','treatment_prior']].le(1).all().all():raise ValueError('Teacher probability/prior outside [0,1]')
    output.mkdir(parents=True,exist_ok=True)
    if teacher.suffix=='.gz':
        with gzip.open(teacher,'rb') as src,(output/'visium_fused_teacher_table.tsv').open('wb') as dst:shutil.copyfileobj(src,dst)
    else:shutil.copy2(teacher,output/'visium_fused_teacher_table.tsv')
    shutil.copy2(features,output/'model_input_numeric.csv');shutil.copy2(feature_manifest,output/'feature_manifest.csv')
    record=dict(status='VALIDATED_PRECOMPUTED_TRAINING_HANDOFF',rows=len(t),samples=x.sample_id.nunique(),treatment_keys=t.drug_key.nunique(),features=len(m),inputs={str(p.resolve()):digest(p) for p in [teacher,features,feature_manifest]},authority={label:dict(path=str(path.resolve()),sha256=expected.lower()) for label,(path,expected) in authorities.items()},outputs={p.name:digest(p) for p in output.iterdir() if p.is_file()},target='fused_prob_responder - treatment_prior',feature_contract='exact ordered manifest; numeric or explicit NA; source bytes and missingness preserved',unseen_sample_labels='NOT_PROVIDED; use fitted spatial predictor for external inference')
    (output/'precomputed_handoff_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--spatial-features',required=True);p.add_argument('--feature-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--teacher-sha256',required=True);p.add_argument('--spatial-features-sha256',required=True);p.add_argument('--feature-manifest-sha256',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.teacher,a.spatial_features,a.feature_manifest,a.output,a.teacher_sha256,a.spatial_features_sha256,a.feature_manifest_sha256),indent=2))
