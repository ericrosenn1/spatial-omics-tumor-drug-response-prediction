"""Failed stages retain reports and return failure to the invoking process."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_returns_child_exit_and_does_not_start_later_step(tmp_path, monkeypatch):
    runner = load('review_runner', ROOT / 'run_pipeline.py')
    (tmp_path / 'failure.py').write_text('raise SystemExit(7)\n')
    (tmp_path / 'later.py').write_text("raise RuntimeError('must not run')\n")
    monkeypatch.setattr(runner, 'CODE_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'STEPS', [('01', 'failure.py'), ('02', 'later.py')])
    monkeypatch.setattr(runner, 'parse_args', lambda: SimpleNamespace(config='unused', start='01', end='02', dry_run=False, open=False))
    monkeypatch.setattr(runner, 'load_config', lambda _: {'config_path': tmp_path / 'dummy.yaml', 'output_root': tmp_path})
    monkeypatch.setattr(runner, 'validate_config', lambda x: x)
    assert runner.main() == 7


def test_runner_preserves_caller_relative_paths(tmp_path, monkeypatch):
    runner = load('relative_runner', ROOT / 'run_pipeline.py')
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'inputs').mkdir()
    config = tmp_path / 'local.yaml'
    config.write_text('input_root: inputs\noutput_root: outputs\nsample_glob: SAMPLE_*\n')
    script = tmp_path / 'verify_paths.py'
    script.write_text("from pathlib import Path\nimport sys\nassert Path.cwd() == Path(sys.argv[-1]).parent\nassert Path('inputs').is_dir()\nassert Path('outputs').is_dir()\n")
    monkeypatch.setattr(runner, 'CODE_ROOT', tmp_path)
    monkeypatch.setattr(runner, 'STEPS', [('01', script.name)])
    monkeypatch.setattr(runner, 'parse_args', lambda: SimpleNamespace(config='local.yaml', start='01', end='01', dry_run=False, open=False))
    assert runner.main() == 0


@pytest.mark.parametrize('number,script,report', [
    ('01', '01_validate_inputs.py', 'output_01_validate_inputs/input_validation_report.csv'),
    ('02', '02_process_samples.py', 'output_02_process_samples_reports/processing_report.csv'),
])
def test_real_invalid_input_cli_main_retains_error_report(number, script, report, tmp_path, monkeypatch):
    module = load('invalid_stage_' + number, ROOT / 'code' / script)
    raw = tmp_path / 'raw'; (raw / 'SAMPLE_0000').mkdir(parents=True)
    output = tmp_path / 'out'
    config = tmp_path / 'config.yaml'
    config.write_text(f'input_root: "{raw.as_posix()}"\noutput_root: "{output.as_posix()}"\nsample_glob: "SAMPLE_*"\n')
    argv = [script, '--config', str(config)]
    if number == '01': argv.append('--no-extract-gz')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='failed; inspect'):
        module.main()
    saved = pd.read_csv(output / report)
    assert len(saved) == 1 and saved.iloc[0]['status'] == 'ERROR'
    assert saved.iloc[0]['sample_id'] == 'SAMPLE_0000'


def test_explicit_recorded_fallback_preserves_genes_without_network(monkeypatch):
    module = load('explicit_fallback_step05', ROOT / 'code/05_build_multi_axis_transcriptome_labels.py')
    def network_forbidden():
        raise AssertionError('An explicitly selected embedded source must not contact MSigDB')
    monkeypatch.setattr(module, 'HAS_GSEAPY', True)
    monkeypatch.setattr(module, 'Msigdb', network_forbidden)
    selected = module.load_external_libraries(75, 'fallback_curated_lite')
    assert selected.hallmark == module.uppercase_gene_sets(module.FALLBACK_HALLMARK_GENESETS)
    assert selected.reactome == module.cap_gene_sets(module.uppercase_gene_sets(module.FALLBACK_REACTOME_GENESETS), 75)
    assert len(selected.hallmark) == 8 and len(selected.reactome) == 13
    assert selected.source == 'fallback_curated_lite'


def test_failed_required_clustering_cannot_create_single_cluster(monkeypatch):
    import anndata
    import numpy as np
    module = load('clustering_failure_step02', ROOT / 'code/02_process_samples.py')
    data = anndata.AnnData(np.ones((5, 4), dtype=float))
    for name in ['normalize_total', 'log1p', 'highly_variable_genes', 'scale', 'neighbors']:
        monkeypatch.setattr(module.sc.pp, name, lambda *args, **kwargs: None)
    for name in ['pca', 'umap']:
        monkeypatch.setattr(module.sc.tl, name, lambda *args, **kwargs: None)
    def missing_dependency(*args, **kwargs):
        raise ImportError('leidenalg unavailable')
    monkeypatch.setattr(module.sc.tl, 'leiden', missing_dependency)
    with pytest.raises(RuntimeError, match='Leiden clustering failed'):
        module.preprocess_adata(data)
    assert 'leiden' not in data.obs
