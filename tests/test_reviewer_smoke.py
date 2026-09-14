"""Public integration checks use generated counts and actual maintained functions."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_reviewer_smoke.py"


def execute(output, config=None):
    args = [sys.executable, str(SCRIPT), "--output", str(output)]
    if config:
        args += ["--config", str(config)]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180)
    return result


def test_public_raw_fixture_endpoint_is_deterministic_and_preserves_failures(tmp_path):
    reports = []
    for name in ["first", "second"]:
        destination = tmp_path / name
        result = execute(destination)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((destination / "smoke_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert report["scope"] == "SMOKE_ONLY_NOT_SCIENTIFIC_VALIDATION"
        assert report["checks_failed"] == 0
        assert report["teacher_rows"] == 32
        assert report["teacher_modalities"] == {"both": 10, "expression_only": 10, "histology_only": 12}
        assert report["model_prediction_rows"] == report["alignment_rows"] == 4
        assert all(x["exit_code"] == 0 for x in report["commands"])
        assert len(report["commands"]) == 4  # Fusion, teacher handoff, prediction and alignment CLIs.
        qc = pd.read_csv(destination / "features/spot_qc.tsv", sep="\t")
        assert qc.loaded_spots.eq(48).all() and qc.retained_spots.eq(47).all()
        assert (destination / "raw_visium_fixtures/FIXTURE_TRAIN_000/raw/matrix.mtx").is_file()
        assert (destination / "raw_visium_fixtures/FIXTURE_EXTERNAL_001/processed.h5ad").is_file()
        decisions = pd.read_csv(destination / "conditional_smoke/conditional_decisions_SMOKE_ONLY.tsv", sep="\t")
        # Three permutations cannot attain q <= .05; never force acceptance for a demo.
        assert not decisions.validated_for_step10.any()
        assert report["conditional_smoke_accepted_profiles"] == 0
        reports.append(report)
    assert reports[0]["numerical_fingerprint"] == reports[1]["numerical_fingerprint"]
    protected = (tmp_path / "first/smoke_report.json").read_bytes()
    retry = execute(tmp_path / "first")
    assert retry.returncode != 0 and "new or empty directory" in retry.stderr
    assert (tmp_path / "first/smoke_report.json").read_bytes() == protected


def test_smoke_refuses_unlabeled_production_scope(tmp_path):
    config = json.loads((ROOT / "configs/reviewer_smoke.json").read_text(encoding="utf-8"))
    config["scope"] = "production"
    path = tmp_path / "invalid_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "invalid_run"
    result = execute(output, path)
    assert result.returncode != 0
    report = json.loads((output / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "Explicit smoke-only" in report["error"]
