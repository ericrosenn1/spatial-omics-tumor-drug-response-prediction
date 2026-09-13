"""Verify invariant spatial-arm dependencies before derivative-only reuse."""
from pathlib import Path
import ast, hashlib, json, shutil
import copy
import pandas as pd

INVARIANT_FUNCTIONS=('metrics','assert_disjoint','make_partitions','prepare','split_samples','pooled_registry','fit_one','screen_treatment','independent_null','summarize_outer')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def function_signatures(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def verify_invariant_code(old_code,new_code):
    old_code=Path(old_code);new_code=Path(new_code)
    for name in ['feature_reference.py','model_training.py','target_building.py','validation.py']:
        if sha(old_code/name)!=sha(new_code/name):raise ValueError('Changed spatial computational dependency: '+name)
    a=function_signatures(old_code/'leakage_evaluation.py');b=function_signatures(new_code/'leakage_evaluation.py')
    for name in INVARIANT_FUNCTIONS:
        if name not in a or a.get(name)!=b.get(name):raise ValueError('Changed spatial computation: '+name)
    trees=[ast.parse((p/'leakage_evaluation.py').read_text(encoding='utf-8-sig')) for p in [old_code,new_code]]
    funcs=[{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)} for tree in trees]
    if 'partition_run' in funcs[0]:
        original=funcs[0]['partition_run'];updated=copy.deepcopy(funcs[1]['partition_run'])
        expected_args=copy.deepcopy(original.args)
        expected_args.args.extend([ast.arg(arg='configuration'),ast.arg(arg='reuse_spatial_from')])
        expected_args.defaults.extend([ast.Constant(value=None),ast.Constant(value=None)])
        if ast.dump(updated.args,include_attributes=False)!=ast.dump(expected_args,include_attributes=False):
            raise ValueError('Unexpected spatial reuse hook arguments')
        updated.args=copy.deepcopy(original.args)
        class RemoveReuseHook(ast.NodeTransformer):
            count=0
            def visit_If(self,node):
                if any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='reuse_spatial_partition' for x in ast.walk(node.test)):
                    self.count+=1
                    guard=ast.parse("arm=='composition_plus_spatial' and reuse_spatial_from and reuse_spatial_partition(reuse_spatial_from,root,name,configuration,pool)",mode='eval').body
                    body=ast.parse("print('REUSED verified unchanged spatial arm',name,flush=True)\ncontinue").body
                    if (self.count>1 or node.orelse or
                        ast.dump(node.test,include_attributes=False)!=ast.dump(guard,include_attributes=False) or
                        [ast.dump(n,include_attributes=False) for n in node.body]!=[ast.dump(n,include_attributes=False) for n in body]):
                        raise ValueError('Unexpected spatial reuse hook guard, body, else, or count')
                    return None
                return self.generic_visit(node)
        remover=RemoveReuseHook();updated=remover.visit(updated)
        if remover.count!=1:raise ValueError('Expected exactly one verified spatial reuse hook')
        if ast.dump(original,include_attributes=False)!=ast.dump(updated,include_attributes=False):raise ValueError('Changed spatial partition computation outside reuse hook')
    for name in ['TARGET','SETTINGS']:
        values=[]
        for tree in trees:
            values.append([ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)])
        if values[0]!=values[1]:raise ValueError('Changed spatial module constant: '+name)

def reuse_spatial_partition(source_root,root,name,configuration,pool):
    """Copy one complete B arm only after exact content/dependency checks.

    Reclassification between A and B can change A while leaving their ordered
    union, training reference, settings and complete B computations invariant.
    This function never treats code-hash mismatch as generally compatible.
    """
    source_root=Path(source_root);root=Path(root);old=source_root/name;new=root/name
    if not (old/'COMPLETE.json').is_file():return False
    previous=json.loads((source_root/'configuration.json').read_text())
    for key in ['teacher','raw_features','metadata']:
        if previous['inputs'][key]!=configuration['inputs'][key]:raise ValueError('Changed input: '+key)
    for key in ['settings','environment']:
        if previous[key]!=configuration[key]:raise ValueError('Changed reuse configuration: '+key)
    verify_invariant_code(source_root/'code_snapshot/spm_v2',root/'code_snapshot/spm_v2')
    complete=json.loads((old/'COMPLETE.json').read_text())
    if complete['identity']!=previous['identity']:raise ValueError('Incompatible source partition identity')
    for rel,h in complete['files'].items():
        source_path=(old/rel).resolve()
        if not source_path.is_relative_to(old.resolve()):raise ValueError('Source checkpoint path escapes partition: '+rel)
        if sha(source_path)!=h:raise ValueError('Source checkpoint corruption: '+rel)
    a=json.loads((old/'partition.json').read_text());b=json.loads((new/'partition.json').read_text())
    for key in ['name','train_ids','test_ids']:
        if a[key]!=b[key]:raise ValueError('Changed partition: '+key)
    for name2 in ['feature_reference.joblib','training_feature_filter.tsv','reference_dependencies.json','training_eligibility.tsv']:
        if sha(old/name2)!=sha(new/name2):raise ValueError('Changed fitted training dependency: '+name2)
    source_arm=old/'composition_plus_spatial';target_arm=new/'composition_plus_spatial'
    old_pool=pd.read_csv(source_arm/'permissible_pool.tsv',sep='\t').feature_name.tolist()
    if old_pool!=list(pool):raise ValueError('Changed ordered spatial-arm feature pool')
    listed={Path(rel).relative_to('composition_plus_spatial') for rel in complete['files']
            if Path(rel).parts[0]=='composition_plus_spatial'}
    actual={p.relative_to(source_arm) for p in source_arm.rglob('*') if p.is_file()}
    if actual!=listed:raise ValueError('Unmanifested or missing source spatial-arm files')
    target_arm.mkdir(exist_ok=True)
    files={}
    for rel in sorted(listed):
        p=source_arm/rel;dest=target_arm/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=sha(p):raise ValueError('Conflicting derivative destination: '+str(dest))
        if not dest.exists():shutil.copy2(p,dest)
        if sha(dest)!=sha(p):raise ValueError('Copy hash mismatch')
        files[str(rel)]=sha(dest)
    evidence=dict(status='VERIFIED_UNCHANGED_SPATIAL_ARM_REUSED',source_root=str(source_root.resolve()),source_identity=previous['identity'],derivative_identity=configuration['identity'],checks=['input_hashes','settings','environment','computational_function_AST','exact_single_reuse_hook','supporting_code_hashes','source_checkpoint_hashes','ordered_train_test_ids','fitted_reference_hash','training_filter_hash','eligibility_hash','ordered_permissible_pool','source_arm_file_set_equals_checkpoint','copied_file_hashes'],files=files)
    p=target_arm/'REUSED_FROM.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(evidence,indent=2),encoding='utf-8');tmp.replace(p)
    return True
