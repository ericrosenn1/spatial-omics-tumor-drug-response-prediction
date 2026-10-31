"""Exercise optional training entry points with tiny tables, without fitting models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image
import pytest

TRAINING = Path(__file__).resolve().parents[1] / "prediction_modeling_pipeline/model_training"


def run_step(tmp_path, module, script, config, cwd=None):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(TRAINING / module / "scripts" / script), "--config", str(config_path)],
        cwd=cwd or tmp_path, capture_output=True, text=True, timeout=30,
    )


def histology_config(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    return {
        "project_root": str(project), "output_root": "outputs/run",
        "paths": {"cptac_data_dir": "clinical", "tcia_download_dir": "slides", "tile_dir": "tiles"},
        "clinical_files": {"clinical": "clinical.tsv"},
        "teacher_export": {
            "approve_if_image_treatment_beats_treatment_only_by_auc": .03,
            "approve_if_patient_auc_at_least": .6,
        },
    }


@pytest.mark.parametrize("inputs_present", [False, True])
def test_histology_preflight_exit_matches_report(tmp_path, inputs_present):
    config = histology_config(tmp_path)
    project = Path(config["project_root"])
    if inputs_present:
        (project / "clinical").mkdir()
        (project / "clinical/clinical.tsv").write_text("patient_id\nP1\n")
        (project / "slides").mkdir()
        (project / "slides/one.svs").touch()
    result = run_step(tmp_path, "histology_response_model_v2", "01_validate_inputs.py", config)
    audit = project / "outputs/run/01_input_audit"
    report = (audit / "input_audit_summary.txt").read_text()
    assert (audit / "input_path_report.tsv").is_file()
    if inputs_present:
        assert result.returncode == 0, result.stderr
        assert "No critical issues detected" in report
    else:
        assert result.returncode != 0
        assert "missing cptac_data_dir" in report
        assert "missing tcia_download_dir" in report
        assert "DONE" not in result.stdout


def test_histology_slide_handoff_uses_project_root_and_excludes_unmatched(tmp_path, monkeypatch):
    config = histology_config(tmp_path)
    config["Tiling"] = {
        "level": 0, "tile_size": 2, "stride": 2, "white_threshold": 230,
        "min_tissue_fraction": .01, "image_format": "png", "workers": 1,
    }
    project = Path(config["project_root"])
    caller = tmp_path / "caller"
    caller.mkdir()
    labels = project / "outputs/run/02_case_labels"
    labels.mkdir(parents=True)
    pd.DataFrame({"patient_id": ["P1", "P2"], "case_id": ["C1", "C2"]}).to_csv(
        labels / "case_label_table.tsv", sep="\t", index=False,
    )
    (project / "slides/P1").mkdir(parents=True)
    (project / "slides/P1/one.svs").touch()
    result = run_step(tmp_path, "histology_response_model_v2", "03_build_slide_manifest.py", config, cwd=caller)
    assert result.returncode == 0, result.stderr
    output = project / "outputs/run/03_slide_manifest"
    manifest = pd.read_csv(output / "slide_manifest.tsv", sep="\t")
    assert manifest.set_index("patient_id").has_slide.to_dict() == {"P1": True, "P2": False}
    matched = pd.read_csv(output / "slide_manifest_matched.tsv", sep="\t")
    handoff = pd.read_csv(output / "slide_manifest_with_labels.tsv", sep="\t")
    pd.testing.assert_frame_equal(matched, handoff)
    assert handoff.patient_id.tolist() == ["P1"]
    assert Path(handoff.svs_path.iloc[0]) == project / "slides/P1/one.svs"
    assert not (caller / "outputs").exists()

    # Execute the actual Step 04 consumer with a 2x2 fake slide. This checks
    # the file contract without an OpenSlide installation or real WSI data.
    class FakeSlide:
        level_dimensions = [(2, 2)]
        level_downsamples = [1.]

        def __init__(self, path):
            assert Path(path) == project / "slides/P1/one.svs"

        def read_region(self, location, level, size):
            return Image.new("RGB", size, color=(50, 50, 50))

        def close(self):
            pass

    scripts = TRAINING / "histology_response_model_v2/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeSlide))
    monkeypatch.setenv("HISTOLOGY_TILE_WORKERS", "1")
    monkeypatch.chdir(caller)
    monkeypatch.setattr(sys, "argv", ["04_tile_slides.py", "--config", str(tmp_path / "config.json")])
    spec = importlib.util.spec_from_file_location("review_histology_tile_step", scripts / "04_tile_slides.py")
    step = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step)
    step.main()
    tiles = pd.read_csv(project / "outputs/run/04_tiles/tile_manifest.tsv", sep="\t")
    assert tiles.patient_id.tolist() == ["P1"]
    assert Path(tiles.tile_path.iloc[0]).is_file()


@pytest.mark.parametrize("image_auc,baseline_auc,approved", [
    (.8, None, False), (.8, np.nan, False), (.8, np.inf, False),
    (.8, .79, False), (.8, .7, True), (.5, .2, False),
])
def test_histology_teacher_approval_requires_observed_baseline_improvement(tmp_path, image_auc, baseline_auc, approved):
    config = histology_config(tmp_path)
    project = Path(config["project_root"])
    models = project / "outputs/run/07_models"
    models.mkdir(parents=True)
    rows = [{"split": "test", "model_type": "image_treatment", "auc": image_auc}]
    if baseline_auc is not None:
        rows.append({"split": "test", "model_type": "treatment_only", "auc": baseline_auc})
    pd.DataFrame(rows).to_csv(models / "model_comparison.tsv", sep="\t", index=False)
    result = run_step(tmp_path, "histology_response_model_v2", "09_audit_histology_model.py", config)
    assert result.returncode == 0, result.stderr
    index = pd.read_csv(project / "outputs/run/09_audit/histology_model_index.tsv", sep="\t")
    assert bool(index.approved_for_teacher.iloc[0]) is approved


@pytest.mark.parametrize("n_models", [0, 1])
def test_expression_audit_fails_only_for_existing_failure_checks(tmp_path, n_models):
    out = tmp_path / "expression_output"
    out.mkdir()
    pd.DataFrame({"drug": ["D1"] * n_models, "approved_for_teacher": [False] * n_models}).to_csv(
        out / "model_index.tsv", sep="\t", index=False,
    )
    config = {"project_root": str(tmp_path), "output_root": str(out)}
    result = run_step(tmp_path, "expression_response_model_v2", "04_audit_deployable_models.py", config)
    checks = pd.read_csv(out / "audit/deployable_model_audit_checks.tsv", sep="\t")
    assert (out / "audit/deployable_model_audit_summary.txt").is_file()
    if n_models:
        assert result.returncode == 0, result.stderr
        assert checks.status.eq("WARN").any()
    else:
        assert result.returncode != 0
        assert checks.loc[checks.item.eq("n_models"), "status"].iloc[0] == "FAIL"
        assert "DONE" not in result.stdout
