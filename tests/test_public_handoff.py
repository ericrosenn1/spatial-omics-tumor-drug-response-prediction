"""Validate the redistributed three-file authority, including real NA masks."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'prediction_modeling_pipeline/teacher_builder'
spec = importlib.util.spec_from_file_location('prepare_public', ROOT / 'prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/16_prepare_precomputed_teacher.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_public_authority_and_real_identity_contract(tmp_path):
    manifest = json.loads((HANDOFF / 'precomputed_handoff_manifest.json').read_text())
    result = module.prepare_manifest(HANDOFF / 'precomputed_handoff_manifest.json', tmp_path / 'handoff')
    assert (result['rows'], result['samples'], result['treatment_keys'], result['features']) == (34881, 102, 374, 661)
    teacher = pd.read_csv(tmp_path / 'handoff/visium_fused_teacher_table.tsv', sep='\t')
    assert teacher.columns.tolist() == manifest['teacher_columns']
    assert teacher.modality_used.value_counts().to_dict() == manifest['modality_counts']
    assert module.digest(tmp_path / 'handoff/visium_fused_teacher_table.tsv') == manifest['files']['teacher']['decompressed_sha256']
    for name in ('model_input_numeric.csv', 'feature_manifest.csv'):
        assert (tmp_path / 'handoff' / name).read_bytes() == (HANDOFF / name).read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare_manifest(HANDOFF / 'precomputed_handoff_manifest.json', tmp_path / 'handoff')


def test_manifest_rejects_escape_and_wrong_role(tmp_path):
    source = json.loads((HANDOFF / 'precomputed_handoff_manifest.json').read_text())
    source['files']['teacher']['path'] = '../outside.tsv.gz'
    path = tmp_path / 'authority.json'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='escapes'):
        module.prepare_manifest(path, tmp_path / 'out')
    source['artifact_role'] = 'unverified'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='role'):
        module.prepare_manifest(path, tmp_path / 'out')
