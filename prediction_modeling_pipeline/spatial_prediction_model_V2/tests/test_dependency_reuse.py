from pathlib import Path
import pytest
from spm_v2.dependency_reuse import INVARIANT_FUNCTIONS,verify_invariant_code

def pair(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    text='\n'.join(f'def {name}():\n    return 1\n' for name in INVARIANT_FUNCTIONS)
    for p in [a,b]:
        for name in ['feature_reference.py','model_training.py','target_building.py','validation.py']:(p/name).write_text('UNCHANGED',encoding='utf-8')
        (p/'leakage_evaluation.py').write_text(text,encoding='utf-8')
    return a,b

def test_allows_only_orchestration_extension(tmp_path):
    a,b=pair(tmp_path)
    with (b/'leakage_evaluation.py').open('a') as f:f.write('\ndef run():\n    return "explicit derivative orchestration"\n')
    verify_invariant_code(a,b)

def test_rejects_model_computation_change(tmp_path):
    a,b=pair(tmp_path);p=b/'leakage_evaluation.py';p.write_text(p.read_text().replace('def fit_one():\n    return 1','def fit_one():\n    return 2'))
    with pytest.raises(ValueError,match='fit_one'):verify_invariant_code(a,b)

def test_rejects_reference_change(tmp_path):
    a,b=pair(tmp_path);(b/'feature_reference.py').write_text('CHANGED')
    with pytest.raises(ValueError,match='feature_reference'):verify_invariant_code(a,b)
