"""The legacy transfer runner resolves its checkout when project root is omitted."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "prediction_modeling_pipeline/spatial_transfer_inference_model/scripts/00_run_spatial_transfer_inference_model.py"


@pytest.mark.parametrize("explicit_root", [False, True])
def test_transfer_project_root_reaches_dispatch(tmp_path, monkeypatch, explicit_root):
    monkeypatch.syspath_prepend(str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("legacy_transfer_release_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "explicit_checkout" if explicit_root else ROOT
    output = tmp_path / "transfer-output"
    args = [str(RUNNER), "--pim-run-root", str(tmp_path / "completed_pim"),
            "--output-root", str(output), "--steps", "01"]
    if explicit_root:
        args.extend(["--project-root", str(expected)])
    monkeypatch.setattr(sys, "argv", args)
    calls = []

    def capture_step(**kwargs):
        calls.append(kwargs)
        return {"step": kwargs["step"], "status": "pass", "return_code": 0}

    monkeypatch.setattr(module, "run_step", capture_step)
    assert module.main() == 0
    assert len(calls) == 1
    assert calls[0]["project_root"] == expected
    assert calls[0]["python_exe"] == sys.executable
    summary = json.loads((output / "spatial_transfer_inference_model_run_summary.json").read_text())
    assert Path(summary["project_root"]) == expected
    assert summary["status"] == "pass"
