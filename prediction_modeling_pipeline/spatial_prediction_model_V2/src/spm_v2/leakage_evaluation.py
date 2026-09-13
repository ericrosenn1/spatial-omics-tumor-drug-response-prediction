"""Separate grouped, training-only evaluation of the established V2 procedure.

This module never reads a global feature registry or globally selected model
list. Results are not interchangeable with production conditional Step 09.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from .feature_reference import FeatureReference, feature_class, numeric
from .model_training import make_xgb_pipeline, select_features_training_only
from .target_building import treatment_eligibility
from .validation import bh_fdr
from .dependency_reuse import reuse_spatial_partition

TARGET='fused_residual_vs_prior'
SETTINGS=dict(pooled_repeats=5,pooled_estimators=200,pooled_max_features=120,registry_max=150,screen_repeats=10,estimators=120,max_features=60,max_depth=2,learning_rate=.03,test_size=.20,seed=20260913,min_samples=60,min_std=.02,min_range=.08,min_unique=10)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atom_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');tmp.replace(p)

def table(df,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    if p.suffix=='.parquet':df.to_parquet(tmp,index=False)
    else:df.to_csv(tmp,sep='\t',index=False)
    tmp.replace(p)

def metrics(y,p,b=None):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if len(y)!=len(p) or not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError('Metrics require explicitly matched finite predictions and targets')
    n=len(y); out=dict(n=n,pearson=np.nan,spearman=np.nan,r2=np.nan,mae=np.nan,rmse=np.nan,baseline_rmse=np.nan,rmse_improvement=np.nan,mae_improvement=np.nan,constant_target=True)
    if not n:return out
    e=y-p;out.update(mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))))
    sst=float(np.sum((y-y.mean())**2));out['constant_target']=bool(sst<=0)
    if n>=3 and sst>0:
        out['r2']=float(1-np.sum(e*e)/sst)
        if np.std(p)>0:out.update(pearson=float(pearsonr(y,p).statistic),spearman=float(spearmanr(y,p).statistic))
    if b is not None:
        b=np.broadcast_to(np.asarray(b,float),y.shape)
        if not np.isfinite(b).all():raise ValueError('Nonfinite baseline')
        out['baseline_rmse']=float(np.sqrt(np.mean((y-b)**2)))
        out['rmse_improvement']=out['baseline_rmse']-out['rmse']
        out['mae_improvement']=float(np.abs(y-b).mean())-out['mae']
    return out

def assert_disjoint(train_ids,test_ids,meta):
    a=set(train_ids);b=set(test_ids)
    if a&b:raise ValueError('Sample leakage')
    m=meta.set_index('sample_id')
    if set(m.loc[list(a),'evaluation_group']) & set(m.loc[list(b),'evaluation_group']):raise ValueError('Biological/conservative group leakage')

def make_partitions(meta):
    meta=meta.sort_values('sample_id').reset_index(drop=True)
    groups=meta.evaluation_group.astype(str)
    result=[]
    for i,(tr,te) in enumerate(GroupKFold(n_splits=5).split(meta,groups=groups)):
        result.append((f'outer_{i}',meta.iloc[tr].sample_id.tolist(),meta.iloc[te].sample_id.tolist()))
    # Independent significance needs auditable, exchangeable patient blocks.
    # Prefer all verified patients for evaluation; source blocks with unknown
    # donor maps remain conservative units. Allocation never reads outcomes.
    if 'patient_id_verified' not in meta:raise ValueError('Source-verified patient grouping flag required')
    verified=meta.patient_id_verified.astype(str).str.lower().isin(['true','1'])
    test_groups=set(groups[verified])
    remaining=meta[~groups.isin(test_groups)].groupby('evaluation_group').size()
    base_test=int(groups.isin(test_groups).sum())
    choices=[]
    # With the 102-sample cohort this adds the nine-sample source block:
    # 21 verified-patient sections +9 unresolved-source sections =30.
    for g,n in remaining.items():
        nt=base_test+int(n)
        if len(meta)-nt>=70 and nt>=20 and len(test_groups)+1>=5:
            choices.append((abs((len(meta)-nt)-72),str(g)))
    if not choices:raise ValueError('No label-blind allocation supports independent design')
    test_groups.add(sorted(choices)[0][1])
    te=np.flatnonzero(groups.isin(test_groups));tr=np.flatnonzero(~groups.isin(test_groups))
    result.append(('independent_selection',meta.iloc[tr].sample_id.tolist(),meta.iloc[te].sample_id.tolist()))
    for _,tr,te in result:assert_disjoint(tr,te,meta)
    return result

def prepare(raw,meta,train_ids,test_ids,classes,out):
    raw=raw.set_index('sample_id',drop=False)
    reference=FeatureReference().fit(raw.loc[train_ids])
    joblib.dump(reference,out/'feature_reference.joblib')
    transformed=reference.transform(raw.loc[train_ids+test_ids]).set_index('sample_id',drop=False)
    cols=classes.loc[classes.feature_class!='excluded','feature_name'].tolist()
    x=transformed[cols].apply(numeric)
    train=x.loc[train_ids]
    keep=(train.notna().mean()>=.20)&(train.nunique(dropna=True)>1)
    table(pd.DataFrame({'feature_name':cols,'train_nonmissing_fraction':train.notna().mean(),'train_unique':train.nunique(),'available_training_feature':keep}),out/'training_feature_filter.tsv')
    atom_json(out/'reference_dependencies.json',dict(program_inputs=list(reference.program_refs),modules=reference.module_columns,training_ids=train_ids))
    return x.loc[:,keep.values]

def split_samples(ids,meta,n,seed):
    m=meta.set_index('sample_id').loc[list(ids)].reset_index()
    gss=GroupShuffleSplit(n_splits=n,test_size=.2,random_state=seed)
    for i,(tr,te) in enumerate(gss.split(m,groups=m.evaluation_group)):
        yield i,m.iloc[tr].sample_id.tolist(),m.iloc[te].sample_id.tolist()

def pooled_registry(x,teacher,meta,train_ids,pool,out,seed,cfg):
    pair=teacher[teacher.sample_id.isin(train_ids)].sort_values(['sample_id','drug_key']).reset_index(drop=True)
    # The dictionary is learned only from outer/discovery training pairs.
    keys=pair.drug_key.value_counts().head(250).index.tolist()
    dummy=pd.get_dummies(pair.drug_key.where(pair.drug_key.isin(keys),'__OTHER__'),prefix='treatment_identity',dtype=float)
    xx=x.loc[pair.sample_id,pool].reset_index(drop=True)
    xx=pd.concat([xx,dummy.reset_index(drop=True)],axis=1)
    y=pair[TARGET].to_numpy(float)
    evidence=[];split_rows=[]
    for repeat,tr,te in split_samples(train_ids,meta,cfg['pooled_repeats'],seed):
        a=np.flatnonzero(pair.sample_id.isin(tr));b=np.flatnonzero(pair.sample_id.isin(te))
        # Refit percentile/preprocessing reference on each internal training set
        # is not needed for unbiased outer estimation; all these rows belong
        # to outer training. Label selection remains internal-training only.
        valid=[c for c in xx if xx.iloc[a][c].notna().any() and xx.iloc[a][c].nunique()>1]
        selected=select_features_training_only(xx.iloc[a],y[a],valid,max_features=cfg['pooled_max_features'])
        pipe=make_xgb_pipeline(random_state=seed+repeat,n_estimators=cfg['pooled_estimators'],max_depth=2,learning_rate=.03,n_jobs=1)
        pipe.fit(xx.iloc[a][selected],y[a])
        rng=np.random.default_rng(seed+repeat);bi=rng.choice(b,2500,replace=False) if len(b)>2500 else b
        # Exact TreeSHAP provided by the fitted XGBoost booster.
        import xgboost as xgb
        imp=pipe.named_steps['imputer'].transform(xx.iloc[bi][selected])
        sv=pipe.named_steps['model'].get_booster().predict(xgb.DMatrix(imp),pred_contribs=True)[:,:-1]
        for f,v,g in zip(selected,np.abs(sv).mean(axis=0),pipe.named_steps['model'].feature_importances_):
            evidence.append(dict(repeat=repeat,feature_name=f,mean_abs_shap=float(v),gain_importance=float(g),seed=seed+repeat))
        split_rows += [dict(repeat=repeat,sample_id=s,partition='train') for s in tr]+[dict(repeat=repeat,sample_id=s,partition='validation') for s in te]
    ev=pd.DataFrame(evidence);table(ev,out/'pooled_feature_evidence.parquet');table(pd.DataFrame(split_rows),out/'pooled_splits.tsv')
    score=ev[ev.feature_name.isin(pool)].groupby('feature_name',as_index=False).agg(mean_abs_shap=('mean_abs_shap','mean'),mean_gain=('gain_importance','mean'),selection_count=('repeat','count'))
    score=score.sort_values(['mean_abs_shap','mean_gain','feature_name'],ascending=[False,False,True]).head(cfg['registry_max'])
    if score.empty:raise ValueError('No training-derived registry')
    table(score,out/'registry.tsv');return score.feature_name.tolist()

def fit_one(x,y,cols,seed,cfg):
    cols=[c for c in cols if x[c].notna().mean()>=.20 and x[c].nunique()>1]
    if not cols:raise ValueError('No available variable training features')
    selected=select_features_training_only(x,y,cols,max_features=cfg['max_features'])
    pipe=make_xgb_pipeline(random_state=seed,n_estimators=cfg['estimators'],max_depth=2,learning_rate=.03,n_jobs=1)
    pipe.fit(x[selected],y)
    return pipe,selected

def screen_treatment(x,sub,meta,registry,seed,cfg):
    records=[];predictions=[];selected_rows=[]
    sub=sub.set_index('sample_id',drop=False)
    for repeat,tr,te in split_samples(sub.index,meta,cfg['screen_repeats'],seed):
        y=sub.loc[tr,TARGET].to_numpy(float);yt=sub.loc[te,TARGET].to_numpy(float)
        pipe,selected=fit_one(x.loc[tr],y,registry,seed+repeat,cfg)
        p=pipe.predict(x.loc[te,selected]);base=float(y.mean())
        records.append(dict(repeat=repeat,seed=seed+repeat,**metrics(yt,p,base)))
        predictions += [dict(repeat=repeat,sample_id=s,target=float(t),prediction=float(v),baseline=base) for s,t,v in zip(te,yt,p)]
        selected_rows += [dict(repeat=repeat,feature_name=f) for f in selected]
    return pd.DataFrame(records),pd.DataFrame(predictions),pd.DataFrame(selected_rows)

def partition_run(name,train_ids,test_ids,raw,teacher,meta,classes,root,cfg,identity,configuration=None,reuse_spatial_from=None):
    out=root/name;out.mkdir(exist_ok=True)
    done=out/'COMPLETE.json'
    if done.exists():
        d=json.loads(done.read_text())
        if d['identity']!=identity:raise ValueError('Incompatible partition checkpoint')
        for rel,h in d['files'].items():
            if sha(out/rel)!=h:raise ValueError('Corrupt partition checkpoint: '+rel)
        print('REUSED',name,flush=True);return
    atom_json(out/'partition.json',dict(name=name,train_ids=train_ids,test_ids=test_ids,identity=identity))
    x=prepare(raw,meta,train_ids,test_ids,classes,out)
    eligible=treatment_eligibility(teacher[teacher.sample_id.isin(train_ids)],min_samples=cfg['min_samples'],min_target_std=cfg['min_std'],min_target_range=cfg['min_range'],min_unique_targets=cfg['min_unique'])
    table(eligible,out/'training_eligibility.tsv')
    drugs=sorted(eligible.loc[eligible.eligible,'drug_key'])
    if cfg.get('benchmark_drug_limit'):drugs=drugs[:cfg['benchmark_drug_limit']]
    print(name,'train',len(train_ids),'test',len(test_ids),'eligible',len(drugs),flush=True)
    for arm in ['composition','composition_plus_spatial']:
        ad=out/arm;ad.mkdir(exist_ok=True)
        pool=classes.loc[classes.feature_class.isin(['composition'] if arm=='composition' else ['composition','spatial']),'feature_name'].tolist();pool=[f for f in pool if f in x]
        if arm=='composition_plus_spatial' and reuse_spatial_from and reuse_spatial_partition(reuse_spatial_from,root,name,configuration,pool):
            print('REUSED verified unchanged spatial arm',name,flush=True)
            continue
        registry=pooled_registry(x,teacher,meta,train_ids,pool,ad,cfg['seed'],cfg)
        table(pd.DataFrame({'feature_name':pool}),ad/'permissible_pool.tsv')
        summaries=[];pred_rows=[];feature_rows=[];screen_metrics=[];screen_preds=[];screen_features=[]
        for i,drug in enumerate(drugs):
            sub=teacher[(teacher.drug_key==drug)&teacher.sample_id.isin(train_ids)].sort_values('sample_id')
            ids=sub.sample_id.tolist();y=sub[TARGET].to_numpy(float)
            sm,sp,sf=screen_treatment(x,sub,meta,registry,cfg['seed']+i*100,cfg)
            for d in [sm,sp,sf]:d.insert(0,'drug_key',drug)
            screen_metrics.append(sm);screen_preds.append(sp);screen_features.append(sf)
            summary=dict(drug_key=drug,n_train=len(ids),pearson_mean=float(np.mean(sm.pearson.to_numpy(float))),r2_mean=float(np.mean(sm.r2.to_numpy(float))),rmse_improvement_mean=float(np.mean(sm.rmse_improvement.to_numpy(float))),pearson_positive_fraction=float((sm.pearson>0).mean()),undefined_pearson_folds=int(sm.pearson.isna().sum()),undefined_r2_folds=int(sm.r2.isna().sum()))
            summaries.append(summary)
            pipe,selected=fit_one(x.loc[ids],y,registry,cfg['seed']+i+10000,cfg)
            # All evaluation samples receive model outputs; absent targets stay
            # absent and are excluded explicitly from metric rows later.
            pred=pipe.predict(x.loc[test_ids,selected]);base=float(y.mean())
            target=teacher[teacher.drug_key==drug].set_index('sample_id')[TARGET].reindex(test_ids)
            for s,t,p in zip(test_ids,target,pred):pred_rows.append(dict(sample_id=s,drug_key=drug,target=t,prediction=float(p),baseline=base,target_available=bool(pd.notna(t))))
            feature_rows += [dict(drug_key=drug,feature_name=f,selection_order=j,seed=cfg['seed']+i+10000) for j,f in enumerate(selected)]
            if name=='independent_selection':
                bd=ad/'discovery_models';bd.mkdir(exist_ok=True)
                joblib.dump(dict(estimator=pipe,features=selected,drug_key=drug,target=TARGET,training_samples=ids,seed=cfg['seed']+i+10000,reference='../feature_reference.joblib'),bd/f'{i:04d}.joblib')
            if (i+1)%25==0:print(name,arm,'completed',i+1,'/',len(drugs),flush=True)
        summary=pd.DataFrame(summaries,columns=['drug_key','n_train','pearson_mean','r2_mean','rmse_improvement_mean','pearson_positive_fraction','undefined_pearson_folds','undefined_r2_folds'])
        selected=summary[(summary.pearson_mean>=.2)&(summary.pearson_positive_fraction>=.6)&(summary.rmse_improvement_mean>0)].sort_values(['pearson_mean','r2_mean','rmse_improvement_mean','drug_key'],ascending=[False,False,False,True]).head(30)
        summary['selected_top30']=summary.drug_key.isin(selected.drug_key)
        summary['selected_tier1']=summary.selected_top30&(summary.pearson_mean>=.6)&(summary.pearson_positive_fraction>=.875)&(summary.rmse_improvement_mean>0)&(summary.r2_mean>0)
        table(summary,ad/'training_screen_summary.tsv')
        for rows,fn in [(screen_metrics,'screen_metrics'),(screen_preds,'screen_predictions'),(screen_features,'screen_features')]:
            table(pd.concat(rows,ignore_index=True) if rows else pd.DataFrame({'drug_key':pd.Series(dtype=str)}),ad/(fn+'.parquet'))
        table(pd.DataFrame(pred_rows,columns=['sample_id','drug_key','target','prediction','baseline','target_available']),ad/'heldout_predictions.parquet')
        table(pd.DataFrame(feature_rows),ad/'final_selected_features.tsv')
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    atom_json(done,dict(identity=identity,files=files,completed=time.time()))

def independent_null(root,meta,n_permutations=1000):
    out=root/'independent_selection/composition_plus_spatial'
    screen=pd.read_csv(out/'training_screen_summary.tsv',sep='\t')
    family=sorted(screen.loc[screen.selected_tier1,'drug_key']);table(pd.DataFrame({'drug_key':family}),root/'independent_tested_family.tsv')
    all_pred=pd.read_parquet(out/'heldout_predictions.parquet')
    pred=all_pred[all_pred.drug_key.isin(family)].merge(meta[['sample_id','evaluation_group','metadata_dataset_id']],on='sample_id',validate='many_to_one')
    if not family:
        table(pd.DataFrame(columns=['drug_key','accepted']),root/'independent_validation_summary.tsv')
        atom_json(root/'independent_null_audit.json',dict(status='COMPLETE_EMPTY_DISCOVERY_FAMILY',tested_family=0,permutations=0,accepted=0));return
    group_universe=pred[['evaluation_group','metadata_dataset_id','drug_key']].drop_duplicates()
    grouped=pred[pred.target_available].groupby(['evaluation_group','metadata_dataset_id','drug_key'],as_index=False).agg(target=('target','mean'),prediction=('prediction','mean'),baseline=('baseline','mean'),sections=('sample_id','count'))
    grouped=group_universe.merge(grouped,on=['evaluation_group','metadata_dataset_id','drug_key'],how='left',validate='one_to_one')
    table(grouped,root/'independent_block_predictions.parquet')
    groups=sorted(grouped.evaluation_group.unique());Y=grouped.pivot(index='evaluation_group',columns='drug_key',values='target').reindex(index=groups,columns=family).to_numpy(float)
    P=grouped.pivot(index='evaluation_group',columns='drug_key',values='prediction').reindex(index=groups,columns=family).to_numpy(float)
    B=grouped.pivot(index='evaluation_group',columns='drug_key',values='baseline').reindex(index=groups,columns=family).to_numpy(float)
    source=grouped.drop_duplicates('evaluation_group').set_index('evaluation_group').metadata_dataset_id.reindex(groups).tolist()
    strata={}
    for i,(s,m) in enumerate(zip(source,np.isfinite(Y))):strata.setdefault((s,tuple(m)),[]).append(i)
    observed=[];null=[];maps=[];seen=set();rng=np.random.default_rng(20260913+990000)
    for j,d in enumerate(family):
        valid=np.isfinite(Y[:,j]);observed.append(dict(drug_key=d,**metrics(Y[valid,j],P[valid,j],B[valid,j])))
    for k in range(n_permutations):
        perm=np.arange(len(groups))
        for ids in strata.values():perm[ids]=rng.permutation(ids)
        seen.add(tuple(perm));maps += [dict(permutation_id=k,evaluation_group=groups[i],source_group=groups[j]) for i,j in enumerate(perm)]
        for j,d in enumerate(family):
            valid=np.isfinite(Y[:,j]);null.append(dict(permutation_id=k,drug_key=d,**metrics(Y[perm[valid],j],P[valid,j],B[valid,j])))
    obs=pd.DataFrame(observed);nd=pd.DataFrame(null);table(nd,root/'independent_null_metrics.parquet');table(pd.DataFrame(maps),root/'independent_null_mappings.parquet')
    ps=[];q95=[];undef=[]
    for _,row in obs.iterrows():
        v=nd.loc[nd.drug_key==row.drug_key,'pearson'].to_numpy(float);good=np.isfinite(v);undef.append(int((~good).sum()))
        ps.append((1+np.sum((~good)|(v>=row.pearson)))/(n_permutations+1) if np.isfinite(row.pearson) else 1.)
        q95.append(float(np.quantile(np.where(good,v,np.inf),.95)))
    obs['p_pearson']=ps;obs['q_pearson']=bh_fdr(ps);obs['null_pearson_q95']=q95;obs['undefined_null_pearson']=undef
    movable=sum(len(x) for x in strata.values() if len(x)>1)
    obs['accepted']=(obs.q_pearson<=.10)&(obs.pearson>obs.null_pearson_q95)&(obs.rmse_improvement>0)&(movable>0)
    obs['test_family']='independently_discovered_tier1';table(obs,root/'independent_validation_summary.tsv')
    atom_json(root/'independent_null_audit.json',dict(status='COMPLETE' if movable else 'COMPLETE_UNINFORMATIVE_NO_EXCHANGEABLE_BLOCKS',tested_family=len(family),permutation_ids=n_permutations,effective_unique_mappings=len(seen),movable_groups=movable,stratum_sizes=[len(s) for s in strata.values()],accepted=int(obs.accepted.sum()),null_denominator=n_permutations,nonfinite_null_rule='count_as_exceedance',aggregation='equal-weight mean over sections within each evaluation block, then equal weight per block'))

def summarize_outer(root,meta):
    rows=[];preds=[]
    for fold in range(5):
        for arm in ['composition','composition_plus_spatial']:
            p=root/f'outer_{fold}'/arm/'heldout_predictions.parquet';d=pd.read_parquet(p);d['fold']=fold;d['arm']=arm
            d=d.merge(meta[['sample_id','evaluation_group','metadata_dataset_id']],on='sample_id',validate='many_to_one');preds.append(d)
            for drug,s in d[d.target_available].groupby('drug_key'):
                rows.append(dict(fold=fold,arm=arm,drug_key=drug,**metrics(s.target,s.prediction,s.baseline)))
    pred=pd.concat(preds,ignore_index=True);table(pred,root/'outer_heldout_predictions.parquet');table(pd.DataFrame(rows),root/'outer_fold_metrics.tsv')
    pooled=[];influence=[];block_metrics=[];block_rows=[]
    for (arm,drug),s in pred[pred.target_available].groupby(['arm','drug_key']):
        pooled.append(dict(arm=arm,drug_key=drug,n_folds=int(s.fold.nunique()),**metrics(s.target,s.prediction,s.baseline)))
        gb=s.groupby(['evaluation_group','metadata_dataset_id'],as_index=False).agg(target=('target','mean'),prediction=('prediction','mean'),baseline=('baseline','mean'),sections=('sample_id','count'))
        block_metrics.append(dict(arm=arm,drug_key=drug,n_folds=int(s.fold.nunique()),**metrics(gb.target,gb.prediction,gb.baseline)))
        gb['arm']=arm;gb['drug_key']=drug;block_rows.append(gb)
        for source in sorted(s.metadata_dataset_id.unique()):
            t=s[s.metadata_dataset_id!=source];influence.append(dict(arm=arm,drug_key=drug,removed_source=source,**metrics(t.target,t.prediction,t.baseline)))
    pooled=pd.DataFrame(pooled);table(pooled,root/'outer_pooled_metrics.tsv');table(pd.DataFrame(influence),root/'source_influence_metrics.tsv')
    bm=pd.DataFrame(block_metrics);table(bm,root/'outer_block_metrics.tsv');table(pd.concat(block_rows,ignore_index=True),root/'outer_block_predictions.parquet')
    bp=bm[bm.arm=='composition'].merge(bm[bm.arm=='composition_plus_spatial'],on='drug_key',suffixes=('_composition','_spatial'),validate='one_to_one')
    for c in ['pearson','spearman','r2','mae','rmse','rmse_improvement']:bp['delta_'+c]=bp[c+'_spatial']-bp[c+'_composition']
    table(bp,root/'matched_spatial_comparison_block.tsv')
    a=pooled[pooled.arm=='composition'];b=pooled[pooled.arm=='composition_plus_spatial'];pair=a.merge(b,on='drug_key',suffixes=('_composition','_spatial'),validate='one_to_one')
    for c in ['pearson','spearman','r2','mae','rmse','rmse_improvement']:pair['delta_'+c]=pair[c+'_spatial']-pair[c+'_composition']
    if not (pair.n_composition==pair.n_spatial).all():raise ValueError('Unmatched ablation sample count')
    table(pair,root/'matched_spatial_comparison.tsv')

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths={k:Path(getattr(args,k)) for k in ['teacher','raw_features','metadata','feature_classes']}
    inputs={k:dict(path=str(p.resolve()),sha256=sha(p)) for k,p in paths.items()}
    cfg=SETTINGS.copy()
    if args.benchmark:cfg.update(pooled_repeats=2,screen_repeats=2,benchmark_drug_limit=3)
    import importlib.metadata as md
    code_hashes={name:sha(Path(__file__).with_name(name)) for name in ['leakage_evaluation.py','feature_reference.py','model_training.py','target_building.py','validation.py','dependency_reuse.py']}
    environment={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','pyarrow']}
    identity=hashlib.sha256(json.dumps(dict(inputs=inputs,settings=cfg,code=code_hashes,environment=environment),sort_keys=True).encode()).hexdigest()
    existing=out/'configuration.json'
    if existing.exists() and json.loads(existing.read_text())['identity']!=identity:raise ValueError('Incompatible output identity; use versioned sibling')
    configuration=dict(identity=identity,inputs=inputs,settings=cfg,code_hashes=code_hashes,environment=environment,benchmark=bool(args.benchmark),reuse_spatial_from=str(Path(args.reuse_spatial_from).resolve()) if args.reuse_spatial_from else None)
    atom_json(existing,configuration)
    import shutil
    snapshot=out/'code_snapshot/spm_v2';snapshot.mkdir(parents=True,exist_ok=True)
    for name in code_hashes:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    teacher=pd.read_csv(paths['teacher'],sep='\t',usecols=['sample_id','drug_key',TARGET,'treatment_prior']);raw=pd.read_csv(paths['raw_features']);meta=pd.read_csv(paths['metadata'],sep='\t')
    if teacher.duplicated(['sample_id','drug_key']).any() or raw.sample_id.duplicated().any() or meta.sample_id.duplicated().any():raise ValueError('Duplicate input keys')
    if set(raw.sample_id)!=set(teacher.sample_id) or set(meta.sample_id)!=set(teacher.sample_id):raise ValueError('Input sample sets differ')
    if not np.isfinite(teacher[TARGET]).all():raise ValueError('Nonfinite teacher residual')
    classes=pd.read_csv(paths['feature_classes'],sep='\t').rename(columns={'strict_class':'feature_class'})
    required={'feature_name','feature_class','reason'}
    if not required.issubset(classes) or classes.feature_name.duplicated().any():raise ValueError('Invalid reviewed feature-class manifest')
    classes=classes[classes.feature_name!='sample_id'].set_index('feature_name')
    if set(classes.index)!=set(raw.columns)-{'sample_id'} or not classes.feature_class.isin(['composition','spatial','excluded']).all():raise ValueError('Reviewed classes do not cover the exact raw feature pool')
    classes=classes.reindex([c for c in raw if c!='sample_id']).reset_index()
    table(classes,out/'feature_classes.tsv')
    partitions=make_partitions(meta)
    table(pd.DataFrame([dict(evaluation=name,partition=part,sample_id=s) for name,tr,te in partitions for part,ids in [('train',tr),('test',te)] for s in ids]),out/'partitions.tsv')
    if args.benchmark:partitions=partitions[:1]
    for name,tr,te in partitions:
        partition_run(name,tr,te,raw,teacher,meta,classes,out,cfg,identity,configuration,args.reuse_spatial_from)
    if not args.benchmark:
        summarize_outer(out,meta);independent_null(out,meta)
    atom_json(out/'RUN_COMPLETE.json',dict(identity=identity,status='BENCHMARK_COMPLETE' if args.benchmark else 'COMPUTATION_COMPLETE_AUDIT_PENDING',completed=time.time()))
    print('Completed',out,flush=True)
