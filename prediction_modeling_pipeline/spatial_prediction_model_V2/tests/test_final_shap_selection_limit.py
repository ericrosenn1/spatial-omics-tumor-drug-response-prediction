"""Regression checks for optional final-SHAP model curation limits."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_train_filtered_per_treatment_residual_models.py"
spec = importlib.util.spec_from_file_location("v2_step07_final_selection", SCRIPT)
step07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step07)


def make_screening_table(n_qualifying=35):
    rows = [
        {
            "drug_key": f"treatment_{i:03d}",
            "selected_for_final_shap": True,
            "test_pearson_mean": 0.90 - i / 1000,
            "test_r2_mean": 0.40,
            "rmse_improvement_vs_baseline_mean": 0.05,
        }
        for i in range(n_qualifying)
    ]
    rows.append({
        "drug_key": "below_threshold",
        "selected_for_final_shap": False,
        "test_pearson_mean": 0.99,
        "test_r2_mean": 0.90,
        "rmse_improvement_vs_baseline_mean": 0.10,
    })
    return pd.DataFrame(rows)


def test_unlimited_default_retains_all_qualifying_models():
    selected = step07.select_final_shap_candidates(make_screening_table(), None)
    assert len(selected) == 35
    assert selected["drug_key"].iloc[0] == "treatment_000"
    assert selected["drug_key"].iloc[-1] == "treatment_034"
    assert "below_threshold" not in selected["drug_key"].tolist()


def test_explicit_limit_preserves_ranked_top_n():
    full = step07.select_final_shap_candidates(make_screening_table(), None)
    limited = step07.select_final_shap_candidates(make_screening_table(), 30)
    assert len(limited) == 30
    assert limited["drug_key"].tolist() == full["drug_key"].iloc[:30].tolist()


@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_nonpositive_final_model_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        step07.select_final_shap_candidates(make_screening_table(), limit)


def test_full_run_launcher_exposes_optional_limit():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "00_run_spatial_prediction_model_v2.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument(\n        "--max-final-shap-models"' in launcher
    assert 'step07_mode_args.extend(["--max-final-shap-models", str(args.max_final_shap_models)])' in launcher
