"""Independent recomputation of conditional Step 09 numerical results.

The arithmetic here does not call the producer's metric, aggregation, p-value or
BH functions. Predictions remain in durable per-treatment parquet blocks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def compare(actual, expected, label, tolerance=1e-10):
    if not np.allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=tolerance, atol=tolerance, equal_nan=True):
        raise AssertionError(f"Numerical mismatch {label}: {actual} vs {expected}")


def recalculate(y, predicted, baseline):
    error = np.asarray(y) - np.asarray(predicted)
    if not np.isfinite(np.column_stack([y, predicted, baseline])).all():
        raise AssertionError("Nonfinite prediction/target/baseline")
    r = pearsonr(y, predicted).statistic if len(y) >= 3 and np.std(y) > 0 and np.std(predicted) > 0 else np.nan
    rho = spearmanr(y, predicted).statistic if len(y) >= 3 and np.std(y) > 0 and np.std(predicted) > 0 else np.nan
    r2 = r2_score(y, predicted, force_finite=False) if len(y) >= 3 and np.std(y) > 0 else np.nan
    rmse = np.sqrt(mean_squared_error(y, predicted))
    mae = mean_absolute_error(y, predicted)
    base_rmse = np.sqrt(mean_squared_error(y, baseline))
    return {"test_pearson": r, "test_spearman": rho, "test_r2": r2,
            "test_mae": mae, "test_rmse": rmse, "baseline_test_rmse": base_rmse,
            "rmse_improvement_vs_baseline": base_rmse - rmse,
            "mae_improvement_vs_baseline": mean_absolute_error(y, baseline) - mae}


def independent_bh(values):
    pairs = sorted(enumerate(values), key=lambda pair: pair[1])
    output = np.ones(len(values))
    running = 1.0
    for position in range(len(pairs) - 1, -1, -1):
        index, value = pairs[position]
        running = min(running, float(value) * len(values) / (position + 1))
        output[index] = running
    return output


def audit_run(output):
    output = Path(output)
    run = json.loads((output / "run_manifest.json").read_text())
    config = run["config"]
    expected_candidates = run["candidate_order"]
    if len(set(expected_candidates)) != len(expected_candidates):
        raise AssertionError("Duplicate candidate keys")
    for name, info in run["sources"].items():
        if digest(info["path"]) != info["sha256"]:
            raise AssertionError(f"Source changed: {name}")
    for name, expected in run["code"].items():
        code_path = Path(__file__).with_name(name) if name != "09_label_shuffle_validate_tier1.py" else Path(__file__).parents[2] / "scripts" / name
        if digest(code_path) != expected:
            raise AssertionError(f"Code changed since numerical run: {code_path}")
    feature_pool = set(run["features"])
    if len(feature_pool) != len(run["features"]):
        raise AssertionError("Duplicate registry features")
    task_paths = sorted((output / "09_checkpoints_by_treatment").glob("task_*/task.json"))
    tasks = [json.loads(path.read_text()) for path in task_paths]
    if [task["drug_key"] for task in tasks] != expected_candidates:
        raise AssertionError("Task candidate/order mismatch")
    audited_metrics, details, total_predictions, total_features = [], [], 0, 0
    for task_path, task in zip(task_paths, tasks):
        root = task_path.parent
        completion = json.loads((root / "complete.json").read_text())
        if completion["task_signature"] != task["signature"] or task["run_signature"] != run["run_signature"]:
            raise AssertionError("Task/completion run signature mismatch")
        if digest(root / "input.parquet") != task["slice_sha256"]:
            raise AssertionError("Input slice changed")
        data = pd.read_parquet(root / "input.parquet")
        if data.sample_id.duplicated().any() or set(data.drug_key) != {task["drug_key"]}:
            raise AssertionError("Duplicate or mixed task sample identities")
        if data.sample_id.tolist() != task["sample_ids"] or list(data[run["features"]].columns) != run["features"]:
            raise AssertionError("Task sample/feature order mismatch")
        original = data[config["target_col"]].to_numpy(float)
        samples = data.sample_id.to_numpy(str)
        seen = set()
        count_predictions, count_features, constant_targets = 0, 0, 0
        for block_name, expected_hash in completion["block_manifest_hashes"].items():
            block = root / "blocks" / block_name
            if digest(block / "manifest.json") != expected_hash:
                raise AssertionError("Block manifest changed")
            block_manifest = json.loads((block / "manifest.json").read_text())
            if block_manifest["task_signature"] != task["signature"]:
                raise AssertionError("Block signature mismatch")
            for name, info in block_manifest["files"].items():
                if digest(block / name) != info["sha256"]:
                    raise AssertionError(f"Numerical output hash mismatch {block / name}")
            metrics = pd.read_parquet(block / "metrics.parquet")
            predictions = pd.read_parquet(block / "predictions.parquet")
            features = pd.read_parquet(block / "features.parquet")
            for frame in [metrics, predictions, features]:
                if set(frame.drug_key) != {task["drug_key"]} or set(frame.original_task_index) != {task["original_task_index"]}:
                    raise AssertionError("Mixed treatment/original-task identities in block")
            keys = ["shuffle_id", "repeat"]
            if metrics.duplicated(keys).any() or predictions.duplicated(keys + ["sample_id"]).any() or features.duplicated(keys + ["feature_name"]).any():
                raise AssertionError("Duplicate metric/prediction/feature keys")
            if not set(features.feature_name).issubset(feature_pool):
                raise AssertionError("Selected feature outside input registry")
            metric_keys = set(map(tuple, metrics[keys].to_numpy()))
            if metric_keys != set(map(tuple, predictions[keys].drop_duplicates().to_numpy())) or metric_keys != set(map(tuple, features[keys].drop_duplicates().to_numpy())):
                raise AssertionError("Split key mismatch among numerical outputs")
            pred_groups = {key: frame for key, frame in predictions.groupby(keys, sort=False)}
            feature_groups = {key: frame for key, frame in features.groupby(keys, sort=False)}
            for record in metrics.itertuples(index=False):
                key = (int(record.shuffle_id), int(record.repeat))
                if key in seen:
                    raise AssertionError("Split repeated across blocks")
                seen.add(key)
                shuffle, repeat = key
                base = config["random_state"] + task["original_task_index"] * 10000
                permutation_seed = base + 500000 + shuffle if shuffle >= 0 else None
                split_seed = base + repeat if shuffle < 0 else base + 500000 + shuffle * 1100 + repeat
                if int(record.random_state) != split_seed:
                    raise AssertionError("Seed schedule mismatch")
                y = original if shuffle < 0 else np.random.default_rng(permutation_seed).permutation(original)
                train, test = train_test_split(np.arange(len(data)), test_size=config["test_size"], random_state=split_seed)
                p = pred_groups[key]
                f = feature_groups[key]
                if p.row_index.tolist() != test.tolist() or p.sample_id.tolist() != samples[test].tolist():
                    raise AssertionError("Held-out membership/order mismatch")
                if json.loads(record.train_sample_ids) != samples[train].tolist() or set(p.sample_id).intersection(samples[train]):
                    raise AssertionError("Training/test overlap or wrong training membership")
                if len(p) != record.n_test or len(train) != record.n_train or len(data) != record.n_rows:
                    raise AssertionError("Sample counts differ")
                if len(f) != record.n_features_selected or f.feature_order.tolist() != list(range(len(f))):
                    raise AssertionError("Selected-feature count/order mismatch")
                compare(p.y_true, y[test], "target permutation")
                compare(p.original_y, original[test], "original targets")
                compare(p.baseline_prediction, np.repeat(y[train].mean(), len(test)), "training-mean baseline")
                computed = recalculate(p.y_true.to_numpy(), p.prediction.to_numpy(), p.baseline_prediction.to_numpy())
                for metric, value in computed.items():
                    compare(getattr(record, metric), value, f"{task['original_task_index']}:{key}:{metric}")
                audited_metrics.append({"drug_key": task["drug_key"], "shuffle_id": shuffle, "repeat": repeat, **computed})
                constant_targets += int(np.std(y[test]) == 0)
            count_predictions += len(predictions)
            count_features += len(features)
        expected_keys = {(p, r) for p in range(-1, config["n_shuffles"]) for r in range(config["n_repeats"])}
        if seen != expected_keys:
            raise AssertionError(f"Incomplete permutation/split set for {task['drug_key']}")
        details.append({"drug_key": task["drug_key"], "original_task_index": task["original_task_index"],
                        "sample_count": len(data), "feature_count": len(feature_pool), "observed_splits": config["n_repeats"],
                        "unique_permutation_ids": config["n_shuffles"], "null_splits": config["n_repeats"] * config["n_shuffles"],
                        "heldout_prediction_rows": count_predictions, "selected_feature_rows": count_features,
                        "constant_target_splits": constant_targets, "status": "PASS"})
        total_predictions += count_predictions
        total_features += count_features
        print(f"AUDIT task={task['original_task_index']} predictions={count_predictions} splits={len(seen)} PASS", flush=True)
    audit = pd.DataFrame(audited_metrics)
    source_results = pd.read_csv(output / "04_validation_results/tier1_label_shuffle_validation_results.tsv", sep="\t").set_index("drug_key")
    reported_observed = pd.read_csv(output / "02_observed_models/tier1_observed_repeated_split_metrics_long.tsv", sep="\t")
    reported_null = pd.read_csv(output / "03_label_shuffle_null/tier1_label_shuffle_null_metrics_long.tsv", sep="\t")
    summary_observed = pd.read_csv(output / "02_observed_models/tier1_observed_metric_summary.tsv", sep="\t").set_index("drug_key")
    summary_null = pd.read_csv(output / "03_label_shuffle_null/tier1_label_shuffle_null_metric_by_shuffle.tsv", sep="\t").set_index(["drug_key", "shuffle_id"])
    for frame in [reported_observed, reported_null]:
        if frame.duplicated(["drug_key", "shuffle_id", "repeat"]).any():
            raise AssertionError("Duplicate split in consolidated tables")
    consolidated = pd.concat([reported_observed, reported_null]).set_index(["drug_key", "shuffle_id", "repeat"]).sort_index()
    calculated = audit.set_index(["drug_key", "shuffle_id", "repeat"]).sort_index()
    if not consolidated.index.equals(calculated.index):
        raise AssertionError("Consolidated split key set mismatch")
    for metric in calculated.columns:
        compare(consolidated[metric], calculated[metric], f"consolidated {metric}")
    decision_rows = []
    for treatment in expected_candidates:
        frame = audit[audit.drug_key == treatment]
        obs = frame[frame.shuffle_id == -1]
        null = frame[frame.shuffle_id >= 0]
        row = {"drug_key": treatment}
        for metric in [c for c in audit.columns if c.startswith("test_") or c.endswith("baseline") or c == "baseline_test_rmse"]:
            observed_mean = float(np.mean(obs[metric].to_numpy(float)))
            compare(summary_observed.loc[treatment, f"observed_{metric}_mean"], observed_mean, "observed summary")
            null_means = []
            for shuffle, group in null.groupby("shuffle_id", sort=True):
                value = float(np.mean(group[metric].to_numpy(float)))
                compare(summary_null.loc[(treatment, shuffle), f"null_{metric}_mean"], value, "null per-shuffle summary")
                null_means.append(value)
            null_means = np.array(null_means)
            if metric in ["test_pearson", "test_r2"]:
                finite_obs = np.isfinite(observed_mean)
                exceedances = sum((not np.isfinite(value)) or value >= observed_mean for value in null_means)
                pvalue = (1 + exceedances) / (config["n_shuffles"] + 1) if finite_obs else 1.0
                q95 = np.percentile(null_means, 95)
                suffix = metric.replace("test_", "")
                row[f"p_{suffix}"] = pvalue
                row[f"observed_{suffix}"] = observed_mean
                row[f"null95_{suffix}"] = q95
                compare(source_results.loc[treatment, f"empirical_p_{suffix}"], pvalue, "empirical plus-one P")
                compare(source_results.loc[treatment, f"null_{metric}_mean_q95"], q95, "null95")
        row["mean_rmse_improvement"] = float(np.mean(obs.rmse_improvement_vs_baseline.to_numpy()))
        decision_rows.append(row)
    decisions = pd.DataFrame(decision_rows)
    decisions["q_pearson"] = independent_bh(decisions.p_pearson)
    decisions["q_r2"] = independent_bh(decisions.p_r2)
    decisions["accepted"] = ((decisions.q_pearson <= config["fdr_threshold"]) & (decisions.observed_pearson > decisions.null95_pearson) & (decisions.mean_rmse_improvement > 0))
    for row in decisions.itertuples(index=False):
        compare(source_results.loc[row.drug_key, "fdr_q_pearson"], row.q_pearson, "BH Pearson")
        compare(source_results.loc[row.drug_key, "fdr_q_r2"], row.q_r2, "BH R2")
        if bool(source_results.loc[row.drug_key, "validated_for_step10"]) != row.accepted:
            raise AssertionError("Acceptance decision mismatch")
    accepted = set(decisions.loc[decisions.accepted, "drug_key"])
    for filename in ["04_validation_results/tier1_label_shuffle_validated_treatments.tsv", "08_step10_handoff/label_shuffle_validated_treatments_for_step10.tsv"]:
        frame = pd.read_csv(output / filename, sep="\t")
        if frame.drug_key.duplicated().any() or set(frame.drug_key) != accepted:
            raise AssertionError("Accepted downstream family mismatch")
    report = {"status": "PASS", "analysis_scope": run["analysis_role"], "n_candidates": len(tasks),
              "n_conditional_acceptances": len(accepted), "n_permutations_per_treatment": config["n_shuffles"],
              "n_splits_per_permutation": config["n_repeats"], "n_split_records_recomputed": len(audit),
              "n_heldout_predictions_checked": total_predictions, "n_selected_feature_rows_checked": total_features,
              "negative_R2_split_count": int((audit.test_r2 < 0).sum()), "undefined_R2_split_count": int(audit.test_r2.isna().sum()),
              "undefined_Pearson_split_count": int(audit.test_pearson.isna().sum()),
              "independent_discovery_validation": False, "critical_numerical_issues": []}
    audit_root = output / "10_independent_numerical_audit"
    audit_root.mkdir(exist_ok=True)
    pd.DataFrame(details).to_csv(audit_root / "permutation_and_key_audit.tsv", sep="\t", index=False)
    decisions.to_csv(audit_root / "independently_recomputed_decisions.tsv", sep="\t", index=False)
    (audit_root / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for filename in ["v2_step09_tier1_label_shuffle_validation_summary.json", "08_step10_handoff/step10_handoff_summary.json"]:
        path = output / filename
        summary = json.loads(path.read_text())
        if summary["n_label_shuffle_validated_treatments"] != len(accepted) or summary["n_tier1_candidates_tested"] != len(tasks):
            raise AssertionError("Run summary counts mismatch")
        summary["status"] = "pass"
        summary["numerical_audit"] = "PASS"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return report
