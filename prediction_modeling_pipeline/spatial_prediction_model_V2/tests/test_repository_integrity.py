from pathlib import Path
import py_compile

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_spm_v2_imports():
    import spm_v2
    assert spm_v2 is not None

def test_pipeline_scripts_compile():
    expected = {
        '00_run_spatial_prediction_model_v2.py',
        '01_validate_inputs.py',
        '02_build_modeling_dataset.py',
        '03_train_probability_baseline.py',
        '04_train_pair_level_residual_model.py',
        '05_build_residual_biology_registry.py',
        '06_train_broad_residual_model.py',
        '07_train_filtered_per_treatment_residual_models.py',
        '08_curate_per_treatment_residual_models.py',
        '09_label_shuffle_validate_tier1.py',
        '10_build_integrated_interpretation_package.py',
        '11_make_publication_tables.py',
        '12_qc_v2_outputs.py',
    }
    scripts = {p.name for p in (REPO_ROOT / 'scripts').glob('*.py')}
    assert expected.issubset(scripts)
    for path in (REPO_ROOT / 'scripts').glob('*.py'):
        py_compile.compile(str(path), doraise=True)
