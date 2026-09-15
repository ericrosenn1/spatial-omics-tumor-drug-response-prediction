"""Public execution defaults must not silently change the analysis or interpreter."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "prediction_modeling_pipeline/spatial_prediction_model_V2/scripts/00_run_spatial_prediction_model_v2.py"


def test_documented_root_profile_source_and_supplied_paths_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    sources = re.findall(r"^\s*Copy-Item\s+(project_profile\.\S+)\s+", readme, re.MULTILINE)
    assert sources, "README root profile setup must identify its source template"
    for source in sources:
        profile = yaml.safe_load((ROOT / source).read_text(encoding="utf-8-sig"))
        supplied = profile["precomputed_handoffs"]
        for key in ["fused_teacher_table_gz", "authority_manifest", "spatial_features", "feature_manifest"]:
            assert (ROOT / supplied[key]).is_file(), supplied[key]
        for config in profile["module_configs"].values():
            assert (ROOT / config["example_config"]).is_file(), config["example_config"]
            if "precomputed_entry_point" in config:
                assert (ROOT / config["precomputed_entry_point"]).is_file(), config["precomputed_entry_point"]


def test_active_python_is_preserved():
    spec = importlib.util.spec_from_file_location("release_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolve_python("") == sys.executable
    assert module.resolve_python("custom-python") == "custom-python"


def test_standalone_step09_defaults_match_full_analysis():
    sys.path.insert(0, str(ROOT / "prediction_modeling_pipeline/spatial_prediction_model_V2/src"))
    from spm_v2.conditional_validation import parser
    args = parser().parse_args([])
    assert (args.n_shuffles, args.n_repeats) == (1000, 5)


@pytest.mark.parametrize("arguments", [
    ["--full-step09-n-shuffles", "100"],
    ["--full-step09-n-repeats", "2"],
])
def test_full_mode_rejects_reduced_validation_before_creating_outputs(tmp_path, arguments):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "full", "--handoff-root", str(tmp_path),
         "--output-root", str(output), *arguments], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert "1000 permutations and 5 repeats" in result.stderr
    assert not output.exists()


def test_model_runner_resolves_relative_outputs_before_dispatch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("relative_release_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--handoff-root", str(tmp_path), "--output-root", "model-output"])
    commands = []
    def capture(*args, **kwargs):
        command = kwargs.get("command", args[1] if len(args) > 1 else None)
        commands.append(command)
        raise RuntimeError("stop before modeling")
    monkeypatch.setattr(module, "run_step", capture)
    # Keep logs in this test's scratch directory too.
    monkeypatch.setattr(module, "V2_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="stop before modeling"):
        module.main()
    command = commands[0]
    assert command[0] == sys.executable
    for flag in ["--run-root", "--output-root", "--handoff-root"]:
        if flag in command:
            assert Path(command[command.index(flag) + 1]).is_absolute()


def test_interpretation_default_project_root_is_this_checkout(monkeypatch):
    path = ROOT / "prediction_modeling_pipeline/prediction_interpretation_model/scripts/00_run_prediction_interpretation_model.py"
    spec = importlib.util.spec_from_file_location("interpretation_release_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--v2-run-root", "completed_run"])
    assert Path(module.parse_args().project_root) == ROOT
