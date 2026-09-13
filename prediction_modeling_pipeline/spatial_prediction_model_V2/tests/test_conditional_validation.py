"""Regression evidence for numerical durability and the production seed/model contract."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spm_v2 import conditional_validation as cv
from spm_v2.conditional_validation_audit import audit_run


def config():
    return dict(target_col="fused_residual_vs_prior", n_shuffles=3, n_repeats=2, test_size=.25,
                max_features_per_split=4, n_estimators=4, max_depth=2, learning_rate=.03,
                random_state=42, fdr_threshold=.10)


def task_at(root, original_index=7):
    root.mkdir(parents=True)
    rng = np.random.default_rng(88)
    frame = pd.DataFrame(rng.normal(size=(24, 6)), columns=[f"f{i}" for i in range(6)])
    frame.loc[[1, 7], "f1"] = np.nan
    frame["sample_id"] = [f"verified_{i:03d}" for i in range(len(frame))]
    frame["drug_key"] = "agent a | recorded profile b"
    frame["fused_residual_vs_prior"] = frame.f0 * .3 + rng.normal(size=len(frame)) * .01
    frame.to_parquet(root / "input.parquet", index=False)
    task = dict(drug_key="agent a | recorded profile b", original_task_index=original_index,
                features=[f"f{i}" for i in range(6)], config=config(), run_signature="test", block_size=1,
                slice_sha256=cv.sha256(root / "input.parquet"), sample_ids=frame.sample_id.tolist())
    task["signature"] = cv.stable_hash(task)
    cv.atomic_json(root / "task.json", task)
    return root / "task.json"


def test_interruption_resume_preserves_numerical_results(tmp_path):
    task_path = task_at(tmp_path / "resumed")
    with pytest.raises(InterruptedError, match="Injected interruption"):
        cv.run_task(task_path, stop_after_blocks=2)
    first = {str(path.relative_to(task_path.parent)): cv.sha256(path) for path in (task_path.parent / "blocks").rglob("*.parquet")}
    assert len(first) == 6
    assert not cv.task_is_complete(task_path)
    cv.run_task(task_path)
    assert cv.task_is_complete(task_path)
    for path, digest in first.items():
        assert cv.sha256(task_path.parent / path) == digest
    fresh = task_at(tmp_path / "fresh")
    cv.run_task(fresh)
    for resumed in (task_path.parent / "blocks").rglob("*.parquet"):
        expected = fresh.parent / resumed.relative_to(task_path.parent)
        pd.testing.assert_frame_equal(pd.read_parquet(resumed), pd.read_parquet(expected), check_exact=True)


def test_corrupt_checkpoint_and_changed_input_fail_closed(tmp_path):
    task = task_at(tmp_path / "corrupt")
    cv.run_task(task)
    path = task.parent / "blocks/observed/predictions.parquet"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Corrupt numerical"):
        cv.run_task(task)
    task2 = task_at(tmp_path / "changed")
    with (task2.parent / "input.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="input hash changed"):
        cv.run_task(task2)


def test_seeds_fixed_by_original_task_index():
    settings = config()
    assert cv.seed_plan(settings, 7, -1, 1) == (None, 70043)
    assert cv.seed_plan(settings, 7, 2, 1) == (570044, 572243)
    queued = [9, 2, 7]
    resumed = [7]
    assert {i: cv.seed_plan(settings, i, 4, 1) for i in queued}[7] == {i: cv.seed_plan(settings, i, 4, 1) for i in resumed}[7]


def test_r2_is_not_squared_pearson_and_constants_are_explicit():
    result = cv.metrics_from_predictions([1, 2, 3, 4], [11, 12, 13, 14], [2, 2, 2, 2])
    assert result["test_r2"] == -79
    assert result["test_pearson"] == pytest.approx(1)
    constant = cv.metrics_from_predictions([1, 1, 1], [0, 1, 2], [1, 1, 1])
    assert constant["constant_target"]
    assert np.isnan(constant["test_r2"])
    assert np.isnan(constant["test_pearson"])
    with pytest.raises(ValueError, match="nonfinite"):
        cv.metrics_from_predictions([1, np.nan], [2, 3], [1, 1])


def test_undefined_permutations_keep_denominator():
    assert cv.empirical_p(.6, [.1, np.nan, .8]) == .75
    assert cv.empirical_p(np.nan, [.1, .2]) == 1
    assert np.isnan(cv.strict_mean([.1, np.nan]))
    np.testing.assert_allclose(cv.bh([.01, .04, .03]), [.03, .04, .04])


def test_production_observed_and_null_equivalence(tmp_path):
    task_path = task_at(tmp_path / "equivalence")
    task = json.loads(task_path.read_text())
    data = pd.read_parquet(task_path.parent / "input.parquet")
    legacy = cv.load_legacy_helpers()
    for shuffle in [-1, 2]:
        actual = cv.train_block(task, data, [shuffle])["metrics"]
        seed, _ = cv.seed_plan(task["config"], task["original_task_index"], shuffle, 0)
        y = data[task["config"]["target_col"]]
        if shuffle >= 0:
            y = pd.Series(np.random.default_rng(seed).permutation(y))
        base = 42 + 7 * 10000 + (500000 + shuffle * 100 if shuffle >= 0 else 0)
        expected, _ = legacy.train_repeated_model(task["drug_key"], data, data[task["features"]], y,
                "observed" if shuffle < 0 else "label_shuffle_null", shuffle, 2, .25, 4, 4, 2, .03, base)
        for metric in cv.METRICS:
            np.testing.assert_allclose(actual[metric], expected[metric], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_duplicate_sample_identity_rejected(tmp_path):
    task_path = task_at(tmp_path / "duplicate")
    task = json.loads(task_path.read_text())
    data = pd.read_parquet(task_path.parent / "input.parquet")
    data.loc[1, "sample_id"] = data.loc[0, "sample_id"]
    data.to_parquet(task_path.parent / "input.parquet", index=False)
    task["slice_sha256"] = cv.sha256(task_path.parent / "input.parquet")
    task["signature"] = cv.stable_hash({key: value for key, value in task.items() if key != "signature"})
    cv.atomic_json(task_path, task)
    with pytest.raises(ValueError, match="Duplicate sample"):
        cv.run_task(task_path)


def test_full_numerical_audit_and_corruption_detection(tmp_path):
    first = task_at(tmp_path / "fixture")
    fixture = pd.read_parquet(first.parent / "input.parquet")
    second = fixture.copy()
    second["drug_key"] = "single_agent_b"
    pair = pd.concat([fixture, second], ignore_index=True)
    dataset = tmp_path / "dataset/03_modeling_datasets"
    registry = tmp_path / "registry/03_v2_strict_biology_registry"
    candidates = tmp_path / "candidates/07_label_shuffle_handoff"
    for path in [dataset, registry, candidates]:
        path.mkdir(parents=True)
    pair.to_csv(dataset / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_name": [f"f{i}" for i in range(6)], "biological_theme": "synthetic"}).to_csv(registry / "v2_strict_biology_feature_registry.tsv", sep="\t", index=False)
    pd.DataFrame({"drug_key": ["agent a | recorded profile b", "single_agent_b"]}).to_csv(candidates / "tier1_label_shuffle_candidates.tsv", sep="\t", index=False)
    args = cv.parser().parse_args(["--run-root", str(tmp_path), "--dataset-root", str(dataset.parent),
            "--step05-root", str(registry.parent), "--step08-root", str(candidates.parent),
            "--output-root", str(tmp_path / "results"), "--n-shuffles", "3", "--n-repeats", "2",
            "--n-estimators", "4", "--max-features-per-split", "4", "--block-size", "1"])
    paths = cv.prepare_tasks(args)
    for path in paths:
        cv.run_task(path)
    cv.consolidate(args.output_root, paths)
    result = audit_run(Path(args.output_root))
    assert result["status"] == "PASS"
    assert result["n_split_records_recomputed"] == 16
    table_path = Path(args.output_root) / "04_validation_results/tier1_label_shuffle_validation_results.tsv"
    changed = pd.read_csv(table_path, sep="\t")
    changed.loc[0, "empirical_p_pearson"] = 0.123456
    changed.to_csv(table_path, sep="\t", index=False)
    with pytest.raises(AssertionError, match="empirical plus-one"):
        audit_run(Path(args.output_root))
