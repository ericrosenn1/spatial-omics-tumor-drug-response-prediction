"""Independent numerical audit of saved grouped evaluation predictions.

Does not fit a model, run SHAP, invoke feature selection, or consume reported
metrics as numerical truth. Scikit-learn/SciPy metrics are reconstructed from
the stored held-out values and compared with all reported summaries.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def recompute(y,p,b):
    y=np.asarray(y,float);p=np.asarray(p,float);b=np.asarray(b,float)
    if not (np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(b).all()):raise AssertionError('Nonfinite metric input')
    out=dict(n=len(y),pearson=np.nan,spearman=np.nan,r2=np.nan,mae=np.nan,rmse=np.nan,baseline_rmse=np.nan,rmse_improvement=np.nan,mae_improvement=np.nan)
    if not len(y):return out
    out.update(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),baseline_rmse=float(np.sqrt(mean_squared_error(y,b))))
    out['rmse_improvement']=out['baseline_rmse']-out['rmse'];out['mae_improvement']=float(mean_absolute_error(y,b))-out['mae']
    if len(y)>=3 and np.var(y)>0:
        out['r2']=float(r2_score(y,p,force_finite=False))
        if np.var(p)>0:out['pearson']=float(pearsonr(y,p).statistic);out['spearman']=float(spearmanr(y,p).statistic)
    return out

def compare(a,b,context):
    for key,value in a.items():
        if key not in b:raise AssertionError(f'{context}: missing reported metric {key}')
        if not np.isclose(float(value),float(b[key]),rtol=1e-10,atol=1e-12,equal_nan=True):raise AssertionError(f'{context}: {key}: recomputed={value}, reported={b[key]}')

def fdr(p):
    p=np.asarray(p,float);order=np.argsort(p);vals=p[order]*len(p)/np.arange(1,len(p)+1)
    q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate(vals[::-1])[::-1]);return q

def audit(root,metadata,teacher):
    from spm_v2.leakage_audit_consistency import audit_relational_consistency
    audit_relational_consistency(root,metadata,teacher)
    root=Path(root);meta=pd.read_csv(metadata,sep='\t');t=pd.read_csv(teacher,sep='\t',usecols=['sample_id','drug_key','fused_residual_vs_prior'])
    config=json.loads((root/'configuration.json').read_text());parts=pd.read_csv(root/'partitions.tsv',sep='\t');checks=[]
    for k,src in config['inputs'].items():assert sha(src['path'])==src['sha256'],k
    for name,h in config['code_hashes'].items():assert sha(root/'code_snapshot/spm_v2'/name)==h,name
    all_partition_predictions=[]
    classes=pd.read_csv(root/'feature_classes.tsv',sep='\t')
    if classes.feature_name.duplicated().any():raise AssertionError('Duplicate feature class keys')
    import joblib
    for name,p in parts.groupby('evaluation'):
        tr=set(p.loc[p.partition=='train','sample_id']);te=set(p.loc[p.partition=='test','sample_id'])
        assert not tr&te and tr|te==set(meta.sample_id)
        mi=meta.set_index('sample_id');assert not set(mi.loc[list(tr)].evaluation_group)&set(mi.loc[list(te)].evaluation_group)
        complete=json.loads((root/name/'COMPLETE.json').read_text());assert complete['identity']==config['identity']
        for rel,h in complete['files'].items():assert sha(root/name/rel)==h,f'{name}/{rel}'
        eligibility=pd.read_csv(root/name/'training_eligibility.tsv',sep='\t');eligible=set(eligibility.loc[eligibility.eligible,'drug_key'])
        reference=joblib.load(root/name/'feature_reference.joblib');assert set(reference.training_ids)==tr
        filt=pd.read_csv(root/name/'training_feature_filter.tsv',sep='\t')
        assert not filt.feature_name.duplicated().any()
        available=set(filt.loc[filt.available_training_feature,'feature_name'])
        assert set(filt.loc[(filt.train_nonmissing_fraction>=.2)&(filt.train_unique>1),'feature_name'])==available
        for drug,s in t[t.sample_id.isin(tr)].groupby('drug_key'):
            y=s.fused_residual_vs_prior
            expect=len(s)>=60 and y.std()>=.02 and y.max()-y.min()>=.08 and y.nunique()>=10
            assert (drug in eligible)==expect,('eligibility',name,drug)
        for arm in ['composition','composition_plus_spatial']:
            ad=root/name/arm;pr=pd.read_parquet(ad/'heldout_predictions.parquet');assert set(pr.drug_key)==eligible
            assert not pr.duplicated(['sample_id','drug_key']).any()
            assert np.isfinite(pr.prediction).all() and np.isfinite(pr.baseline).all()
            merged=pr.merge(t,on=['sample_id','drug_key'],how='left',validate='one_to_one')
            assert np.allclose(merged.target,merged.fused_residual_vs_prior,equal_nan=True,rtol=0,atol=1e-12)
            assert np.array_equal(merged.target_available,np.isfinite(merged.fused_residual_vs_prior))
            if name.startswith('outer_'):
                saved=pr.copy();saved['fold']=int(name.split('_')[1]);saved['arm']=arm;all_partition_predictions.append(saved)
            for drug,s in pr.groupby('drug_key'):
                assert set(s.sample_id)==te
                baseline=float(t[(t.drug_key==drug)&t.sample_id.isin(tr)].fused_residual_vs_prior.mean())
                assert np.allclose(s.baseline,baseline,rtol=0,atol=1e-12)
            sm=pd.read_parquet(ad/'screen_metrics.parquet');sp=pd.read_parquet(ad/'screen_predictions.parquet');ss=pd.read_csv(ad/'training_screen_summary.tsv',sep='\t')
            for (drug,repeat),s in sp.groupby(['drug_key','repeat']):
                assert not s.sample_id.duplicated().any() and set(s.sample_id)<=tr
                train_part=t[(t.drug_key==drug)&t.sample_id.isin(tr-set(s.sample_id))]
                ytrain=train_part.fused_residual_vs_prior
                assert np.allclose(s.baseline,float(ytrain.mean()),rtol=0,atol=1e-12)
                evalgroups=set(mi.loc[s.sample_id].evaluation_group);assert not evalgroups & set(mi.loc[train_part.sample_id].evaluation_group)
                report=sm[(sm.drug_key==drug)&(sm.repeat==repeat)].iloc[0]
                compare(recompute(s.target,s.prediction,s.baseline),report,f'{name}/{arm}/screen/{drug}/{repeat}')
            for drug,s in sm.groupby('drug_key'):
                assert set(s.repeat)==set(range(10))
                row=ss[ss.drug_key==drug].iloc[0]
                compare(dict(pearson_mean=np.mean(s.pearson.to_numpy(float)),r2_mean=np.mean(s.r2.to_numpy(float)),rmse_improvement_mean=np.mean(s.rmse_improvement.to_numpy(float)),pearson_positive_fraction=float((s.pearson>0).mean())),row,drug)
            top=ss[(ss.pearson_mean>=.2)&(ss.pearson_positive_fraction>=.6)&(ss.rmse_improvement_mean>0)].sort_values(['pearson_mean','r2_mean','rmse_improvement_mean','drug_key'],ascending=[False,False,False,True]).head(30)
            expected_top=set(top.drug_key);assert set(ss.loc[ss.selected_top30,'drug_key'])==expected_top
            expected_tier=set(ss.loc[ss.drug_key.isin(expected_top)&(ss.pearson_mean>=.6)&(ss.pearson_positive_fraction>=.875)&(ss.rmse_improvement_mean>0)&(ss.r2_mean>0),'drug_key'])
            assert set(ss.loc[ss.selected_tier1,'drug_key'])==expected_tier
            if name=='independent_selection' and arm=='composition_plus_spatial':independent_expected_family=expected_tier
            pool=pd.read_csv(ad/'permissible_pool.tsv',sep='\t').feature_name.tolist()
            expected_pool=set(classes.loc[classes.feature_class.isin(['composition'] if arm=='composition' else ['composition','spatial']),'feature_name'])&available
            assert set(pool)==expected_pool and len(pool)==len(set(pool))
            ev=pd.read_parquet(ad/'pooled_feature_evidence.parquet')
            score=ev[ev.feature_name.isin(pool)].groupby('feature_name',as_index=False).agg(mean_abs_shap=('mean_abs_shap','mean'),mean_gain=('gain_importance','mean'),selection_count=('repeat','count'))
            score=score.sort_values(['mean_abs_shap','mean_gain','feature_name'],ascending=[False,False,True]).head(150)
            saved_registry=pd.read_csv(ad/'registry.tsv',sep='\t');assert score.feature_name.tolist()==saved_registry.feature_name.tolist()
            for c in ['mean_abs_shap','mean_gain','selection_count']:assert np.allclose(score[c],saved_registry[c],rtol=1e-10,atol=1e-12)
            ps=pd.read_csv(ad/'pooled_splits.tsv',sep='\t')
            assert set(ps.repeat)==set(range(5))
            for _,sparts in ps.groupby('repeat'):
                pa=set(sparts.loc[sparts.partition=='train','sample_id']);pb=set(sparts.loc[sparts.partition=='validation','sample_id'])
                assert not pa&pb and pa|pb==tr
                assert not set(mi.loc[list(pa)].evaluation_group)&set(mi.loc[list(pb)].evaluation_group)
            reg=set(pd.read_csv(ad/'registry.tsv',sep='\t').feature_name)
            sel=pd.read_csv(ad/'final_selected_features.tsv',sep='\t');assert set(sel.feature_name)<=reg
            assert not sel.duplicated(['drug_key','feature_name']).any()
            checks.append(dict(check='partition_source_keys_predictions_screen_metrics',evaluation=name,arm=arm,status='PASS',treatments=len(eligible),heldout_rows=len(pr)))
    allpred=pd.read_parquet(root/'outer_heldout_predictions.parquet');fold_metrics=pd.read_csv(root/'outer_fold_metrics.tsv',sep='\t');pooled=pd.read_csv(root/'outer_pooled_metrics.tsv',sep='\t');block=pd.read_csv(root/'outer_block_metrics.tsv',sep='\t')
    expected_predictions=pd.concat(all_partition_predictions,ignore_index=True)
    keycols=['fold','arm','sample_id','drug_key']
    actual_base=allpred[expected_predictions.columns].sort_values(keycols).reset_index(drop=True)
    expected_predictions=expected_predictions.sort_values(keycols).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_base,expected_predictions,check_exact=True)
    assert not allpred.duplicated(['arm','sample_id','drug_key']).any()
    for (fold,arm,drug),s in allpred[allpred.target_available].groupby(['fold','arm','drug_key']):
        row=fold_metrics[(fold_metrics.fold==fold)&(fold_metrics.arm==arm)&(fold_metrics.drug_key==drug)].iloc[0];compare(recompute(s.target,s.prediction,s.baseline),row,'outer fold')
    for (arm,drug),s in allpred[allpred.target_available].groupby(['arm','drug_key']):
        compare(recompute(s.target,s.prediction,s.baseline),pooled[(pooled.arm==arm)&(pooled.drug_key==drug)].iloc[0],'outer pooled')
        b=s.groupby('evaluation_group').agg(target=('target','mean'),prediction=('prediction','mean'),baseline=('baseline','mean'))
        compare(recompute(b.target,b.prediction,b.baseline),block[(block.arm==arm)&(block.drug_key==drug)].iloc[0],'outer block')
    for fn in ['matched_spatial_comparison.tsv','matched_spatial_comparison_block.tsv']:
        a=pd.read_csv(root/fn,sep='\t')
        for c in ['pearson','spearman','r2','mae','rmse','rmse_improvement']:assert np.allclose(a['delta_'+c],a[c+'_spatial']-a[c+'_composition'],equal_nan=True,rtol=1e-10,atol=1e-12)
    checks.append(dict(check='outer_metrics_and_matched_differences',status='PASS',heldout_rows=len(allpred)))
    family=pd.read_csv(root/'independent_tested_family.tsv',sep='\t').drug_key.tolist();report=pd.read_csv(root/'independent_validation_summary.tsv',sep='\t');null_audit=json.loads((root/'independent_null_audit.json').read_text())
    assert len(family)==len(set(family)) and set(family)==independent_expected_family
    assert set(report.drug_key)==set(family)
    if family:
        gp=pd.read_parquet(root/'independent_block_predictions.parquet');maps=pd.read_parquet(root/'independent_null_mappings.parquet');null=pd.read_parquet(root/'independent_null_metrics.parquet')
        assert set(maps.permutation_id)==set(range(1000)) and not maps.duplicated(['permutation_id','evaluation_group']).any()
        Y=gp.pivot(index='evaluation_group',columns='drug_key',values='target').reindex(columns=family);P=gp.pivot(index='evaluation_group',columns='drug_key',values='prediction').reindex(columns=family);B=gp.pivot(index='evaluation_group',columns='drug_key',values='baseline').reindex(columns=family)
        source=gp.drop_duplicates('evaluation_group').set_index('evaluation_group').metadata_dataset_id
        for pid,m in maps.groupby('permutation_id'):
            mapping=m.set_index('evaluation_group').source_group.reindex(Y.index)
            assert set(mapping)==set(Y.index)
            assert np.array_equal(source.reindex(Y.index).values,source.reindex(mapping).values)
            yp=Y.loc[mapping].to_numpy(float);assert np.array_equal(np.isfinite(Y),np.isfinite(yp))
            for j,drug in enumerate(family):
                valid=np.isfinite(Y[drug]);row=null[(null.drug_key==drug)&(null.permutation_id==pid)].iloc[0]
                compare(recompute(yp[valid,j],P[drug][valid],B[drug][valid]),row,'independent null')
        assert not null.duplicated(['permutation_id','drug_key']).any() and len(null)==1000*len(family)
        recomputed_p=[]
        for drug in family:
            v=Y[drug].notna();obs=recompute(Y[drug][v],P[drug][v],B[drug][v]);row=report[report.drug_key==drug].iloc[0];compare(obs,row,'independent observed')
            n=null[null.drug_key==drug].pearson.to_numpy(float);finite=np.isfinite(n)
            p=(1+np.sum((~finite)|(n>=obs['pearson'])))/1001 if np.isfinite(obs['pearson']) else 1.
            q95=float(np.quantile(np.where(finite,n,np.inf),.95));compare(dict(p_pearson=p,null_pearson_q95=q95),row,'independent P/null95');recomputed_p.append(p)
        q=fdr(recomputed_p)
        for drug,qv in zip(family,q):
            row=report[report.drug_key==drug].iloc[0];assert np.isclose(row.q_pearson,qv)
            accept=qv<=.1 and row.pearson>row.null_pearson_q95 and row.rmse_improvement>0 and null_audit['movable_groups']>0
            assert bool(row.accepted)==bool(accept)
    else:assert report.empty and null_audit['tested_family']==0
    checks.append(dict(check='independent_source_stratified_null_P_BH_decisions',status='PASS',tested_family=len(family)))
    result=dict(status='PASS',checks=checks,input_identity=config['identity'],accepted=int(report.accepted.sum()) if len(report) else 0,negative_outer_r2=int((pooled.r2<0).sum()),undefined_outer_r2=int(pooled.r2.isna().sum()),scope='numerical_and_dependency_audit; teacher response is not clinical outcome validation')
    tmp=root/'independent_numerical_audit.json.tmp';tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(root/'independent_numerical_audit.json')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--metadata',required=True);p.add_argument('--teacher',required=True);a=p.parse_args();audit(a.root,a.metadata,a.teacher)
