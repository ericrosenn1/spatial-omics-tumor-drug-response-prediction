"""Slide scoring supports missing dataset metadata and preserves missing ranks."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def scoring(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "code"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("slide_score", scripts / "04_score_and_label_slides.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_samples_without_dataset_metadata_use_observed_percentile_ranks(scoring):
    table = pd.DataFrame({
        "sample_id": ["SAMPLE_A", "SAMPLE_B", "SAMPLE_C", "SAMPLE_MISSING"],
        "mean__tumor_epithelial_score": [10., 30., 20., np.nan],
    }, index=[4, 6, 8, 10])
    result = scoring.add_program_scores(table)
    np.testing.assert_allclose(result["score__tumor_epithelial"], [1 / 3, 1, 2 / 3, np.nan], equal_nan=True)
    assert result.index.equals(table.index)
    assert result["support_n__tumor_epithelial"].tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize("dataset_id", [None, "recorded_dataset"])
def test_single_sample_has_no_inferred_cohort_rank(scoring, dataset_id):
    table = pd.DataFrame({"sample_id": ["SAMPLE_A"], "mean__tumor_epithelial_score": [10.]})
    if dataset_id is not None:
        table["dataset_id"] = dataset_id
    result = scoring.add_dominant_program(scoring.add_program_scores(table))
    assert result["score__tumor_epithelial"].isna().all()
    assert result["dominant_program"].tolist() == ["none"]
    assert not result["label__tumor_epithelial_high"].any()
    assert not result["label__tumor_epithelial_low"].any()
