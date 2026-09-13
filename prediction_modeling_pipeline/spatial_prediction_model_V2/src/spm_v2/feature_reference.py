"""Frozen references for the existing slide-program and context rank features.

The archived feature pipeline ranked the whole submitted cohort. This adapter
fits those references on training samples and interpolates new values against
the saved ranks. Within-sample architecture measurements are left unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

PROGRAMS = ('tumor_epithelial','immune_general','myeloid','fibroblast_stroma','endothelial','hypoxia','proliferation')
METABOLIC = {
 'glycolysis':['glycolysis','SLC2A1','LDHA','HK2'],
 'oxidative_phosphorylation':['oxidative_phosphorylation','oxphos','respiratory','mitochondrial'],
 'fatty_acid_metabolism':['fatty_acid','fatty_acid_oxidation','fatty_acid_synthesis','FASN','CPT1A'],
 'glutamine_metabolism':['glutamine','GLS','SLC1A5'],
 'nucleotide_synthesis':['nucleotide','RRM2','TYMS'],
 'tryptophan_kynurenine':['tryptophan','kynurenine','IDO1','TDO2'],
 'proline_collagen_support':['proline','collagen_support','P4HA','PLOD'],
}
CONTEXT = {
 'hypoxia_context':['hypoxia','hypoxic','hypoxic_stress'],
 'vascular_context':['vascular','angiogenic','endothelial','oxygen'],
 'stromal_ecm_context':['stromal','stroma','ecm','collagen','fibroblast'],
 'immune_context':['immune','t_cell','interferon','myeloid','b_plasma'],
 'checkpoint_context':['checkpoint','exhaustion','PDCD1','CD274','CTLA4','TIGIT'],
 'proliferation_context':['proliferation','proliferative','cell_cycle','G2M','E2F','MKI67'],
 'accessibility_context':['accessibility','accessible','vascular_access'],
 'barrier_context':['barrier','impermeable','stromal_barrier'],
}
CONCORDANCE = {
 'glycolysis_hypoxia_concordance':('glycolysis','hypoxia_context',1),
 'oxphos_vascular_concordance':('oxidative_phosphorylation','vascular_context',1),
 'fatty_acid_stromal_concordance':('fatty_acid_metabolism','stromal_ecm_context',1),
 'glutamine_proliferation_concordance':('glutamine_metabolism','proliferation_context',1),
 'nucleotide_proliferation_concordance':('nucleotide_synthesis','proliferation_context',1),
 'tryptophan_checkpoint_concordance':('tryptophan_kynurenine','checkpoint_context',1),
 'proline_stromal_concordance':('proline_collagen_support','stromal_ecm_context',1),
 'glycolysis_accessibility_inverse':('glycolysis','accessibility_context',-1),
 'barrier_metabolic_stress_concordance':('glycolysis','barrier_context',1),
}

def numeric(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)

def rank_reference(values, minimum=2):
    s=numeric(pd.Series(values)).dropna()
    if len(s)<minimum: return (np.array([]),np.array([]))
    r=pd.DataFrame({'value':s,'rank':s.rank(pct=True)})
    r=r.groupby('value',sort=True)['rank'].first()
    return r.index.to_numpy(float),r.to_numpy(float)

def apply_rank(values, ref):
    a=numeric(pd.Series(values)).to_numpy(float)
    x,y=ref
    if not len(x): return np.full(len(a),np.nan)
    result=np.interp(a,x,y,left=0.,right=1.)
    # np.interp with a one-point reference returns its sole value even for a
    # NaN query. Missing measurements must never become biological signals.
    result[~np.isfinite(a)]=np.nan
    return result

def is_module_signal(col):
    c=str(col).lower()
    if any(t in c for t in ['sample_id','path','status','label','used','summary','notes','rule_hits','metadata','h5ad']): return False
    return any(t in c for t in ['score','simple__','ucell__','gsva','hotspot__','access_','fraction','mean','median','q75','q90'])

@dataclass
class FeatureReference:
    program_refs: dict=field(default_factory=dict)
    module_refs: dict=field(default_factory=dict)
    module_columns: dict=field(default_factory=dict)
    training_ids: list=field(default_factory=list)
    raw_columns: list=field(default_factory=list)
    version: str='training-reference-v1'

    def fit(self, raw:pd.DataFrame):
        self._check(raw)
        self.training_ids=raw.sample_id.astype(str).tolist()
        self.raw_columns=raw.columns.tolist()
        groups=raw.get('dataset_id',pd.Series('__unknown__',index=raw.index)).fillna('__unknown__').astype(str)
        for p in PROGRAMS:
            for c in [f'mean__{p}_score',f'median__{p}_score',f'spot_fraction__{p}']:
                if c not in raw: continue
                v=numeric(raw[c]); grefs={}
                for g in sorted(groups.unique()):
                    gv=v[groups==g]; grefs[g]=(int(gv.notna().sum()),rank_reference(gv,minimum=1))
                self.program_refs[c]=(rank_reference(v),grefs)
        work=self._programs(raw)
        # The original Step08 selects inputs from its Step07 base, before motif
        # and module columns exist. Never allow derived columns to recurse.
        base_cols=[c for c in raw if not c.startswith(('metabolic_module__','context_module__','concordance__','motif_','pair_'))]
        for prefix,defs in [('metabolic_module__',METABOLIC),('context_module__',CONTEXT)]:
            for name,keywords in defs.items():
                key=prefix+name
                cols=[c for c in base_cols if is_module_signal(c) and any(k.lower() in c.lower() for k in keywords) and numeric(work[c]).notna().sum()>=5]
                self.module_columns[key]=cols
                for c in cols:
                    if c not in self.module_refs: self.module_refs[c]=rank_reference(work[c],minimum=5)
        return self

    @staticmethod
    def _check(raw):
        if 'sample_id' not in raw or raw.sample_id.isna().any() or raw.sample_id.astype(str).duplicated().any(): raise ValueError('Unique nonmissing sample_id required')
        if raw.columns.duplicated().any(): raise ValueError('Duplicate feature columns')

    def _programs(self, raw):
        out=raw.copy()
        groups=raw.get('dataset_id',pd.Series('__unknown__',index=raw.index)).fillna('__unknown__').astype(str)
        scores={}
        for p in PROGRAMS:
            parts=[]
            for c in [f'mean__{p}_score',f'median__{p}_score',f'spot_fraction__{p}']:
                if c not in self.program_refs: continue
                if c not in raw: raise ValueError(f'Missing required raw reference input: {c}')
                ref,grefs=self.program_refs[c]; global_rank=apply_rank(raw[c],ref); ranked=global_rank.copy()
                for g in groups.unique():
                    ix=np.flatnonzero((groups==g).to_numpy())
                    n,gref=grefs.get(g,(0,(np.array([]),np.array([]))))
                    if n and len(ref[0]):
                        w=n/(n+10.); ranked[ix]=w*apply_rank(raw[c].iloc[ix],gref)+(1-w)*global_rank[ix]
                parts.append(ranked)
            if parts:
                block=pd.DataFrame(np.array(parts).T,index=raw.index)
                val=block.mean(axis=1,skipna=True)
            else: val=pd.Series(np.nan,index=raw.index)
            out[f'score__{p}']=val; scores[p]=val
            for flag,test in [('high',val>=.67),('low',val<=.33)]:
                out[f'label__{p}_{flag}']=test.astype(float).where(val.notna())
        simple={'tumor_epithelial_enriched':('tumor_epithelial',True),'immune_inflamed':('immune_general',True),'immune_desert':('immune_general',False),'myeloid_enriched':('myeloid',True),'stromal_barrier':('fibroblast_stroma',True),'vascular_endothelial':('endothelial',True),'hypoxic':('hypoxia',True),'proliferative':('proliferation',True)}
        for label,(p,high) in simple.items():
            v=scores[p];out['label__'+label]=((v>=.67) if high else (v<=.33)).astype(float).where(v.notna())
        immune=scores['immune_general']; barrier=pd.concat([scores[x] for x in ['fibroblast_stroma','myeloid','hypoxia']],axis=1)
        barrier_high=(barrier>=.67).any(axis=1)
        known=immune.notna() & (barrier_high | barrier.notna().all(axis=1))
        out['label__immune_excluded_proxy']=((immune>=.33)&barrier_high).astype(float).where(known)
        out['label__cold_barriered_proxy']=((immune<=.33)&barrier_high).astype(float).where(known)
        return out

    def transform(self,raw):
        self._check(raw)
        out=self._programs(raw)
        for key,cols in self.module_columns.items():
            missing=[c for c in cols if c not in out]
            if missing: raise ValueError(f'Missing required raw reference inputs: {missing[:5]}')
            vals=[apply_rank(out[c],self.module_refs[c]) for c in cols]
            out[key]=pd.DataFrame(np.array(vals).T,index=out.index).mean(axis=1) if vals else np.nan
        for name,(m,c,sign) in CONCORDANCE.items():
            a=numeric(out['metabolic_module__'+m]); b=numeric(out['context_module__'+c])
            out['concordance__'+name]=(a+(b if sign==1 else 1-b))/2
        cols=['concordance__'+n for n in CONCORDANCE]
        out['concordance__overall_mean']=out[cols].mean(axis=1)
        out['concordance__overall_std']=out[cols].std(axis=1)
        return out

def feature_class(name):
    """Predeclared measurement classes; no response labels enter this rule."""
    c=str(name).lower()
    if any(t in c for t in ['sample_id','dataset_id','patient','cancer_type','path','file','h5ad','status','error','notes','summary','rule_hits','profile_label','dominant_program','annotation_column','column_used','score_column_used','metadata','format','timepoint','coord','barcode','filtering__','n_genes','total_counts','pct_counts','confidence','concordance_score','method_','_n_columns','support_n','support_cols','n_supported','_available','raw_fraction','region_consensus']):
        return 'excluded','identifier, technical support, confidence or governed caution'
    if c in ['n_spots','n_clusters'] or c.endswith('_spots') or 'spot_count' in c or 'largest_component_spots' in c or 'fraction_of_slide' in c:
        return 'excluded','raw sample size/count or governed size-associated feature'
    if c.startswith(('metabolic_module__','context_module__','concordance__')):
        return 'spatial','mixed module can include spatial measurements; conservative augmented-arm assignment'
    if c.startswith(('access_immune_score_tumor_','access_stromal_score_tumor_','access_hypoxia_score_tumor_','access_angiogenic_score_tumor_','access_barrier_score_tumor_')):
        return 'composition','marginal program summary inside the tumor mask; no explicit relationship statistic'
    if c.startswith(('pair_','access_')) or any(t in c for t in ['distance','dist_to_','contact_','boundary','penetration','fragmentation','n_components','largest_component','gradient','depth_','tumor_core']):
        return 'spatial','explicit relationship, interface, gradient, connected region or organization'
    if c.startswith(('mean__','median__','spot_fraction__','score__','label__','structure_fraction__','structure_region_fraction__','structure_score__','function_score__','metabolic_score__','fraction__')):
        return 'composition','marginal program state, composition or per-spot activation fraction'
    if c.startswith('hotspot__') and c.endswith(('_fraction','_threshold')):
        return 'composition','per-slide quantile threshold or activated-spot fraction without neighbor information'
    if c.startswith('motif_') and c.endswith('_fraction'):
        return 'composition','marginal biological-mask fraction; boundary/core fractions assigned spatial first'
    return 'excluded','not a reviewed numeric biological measurement'
