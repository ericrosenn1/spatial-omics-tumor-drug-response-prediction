"""Durable, conditional Step 09 validation with auditable numerical checkpoints.

This preserves the production model, candidate ordering, seeds and acceptance
thresholds. The globally learned registry and previously selected candidate set
make this a development/conditional test, not an independent discovery test.
Each atomic block contains every split metric, held-out prediction and selected
feature. Undefined correlations are retained; an undefined null statistic counts
as an exceedance and never shrinks the permutation denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .model_training import make_xgb_pipeline, select_features_training_only
from .conditional_reporting import report_distinct_theme_counts

SCHEMA = "conditional_step09_numerical_v1"
METRICS = ["test_pearson", "test_spearman", "test_r2", "test_mae", "test_rmse",
           "baseline_test_rmse", "rmse_improvement_vs_baseline", "mae_improvement_vs_baseline"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_task_signature(task):
    if stable_hash({key: value for key, value in task.items() if key != "signature"}) != task["signature"]:
        raise ValueError("Task configuration/signature changed")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def save_execution_code(output, code_files, expected_hashes):
    """Atomically preserve the exact executed bytes for later numerical audits.

    A Git checkout may normalize line endings after execution. The snapshot
    preserves the original byte identity; it does not authorize resuming a run
    with different current code or change the run signature.
    """
    destination = Path(output) / "01_inputs/execution_code"
    if destination.exists():
        for name, expected in expected_hashes.items():
            if not (destination / name).is_file() or sha256(destination / name) != expected:
                raise ValueError(f"Execution-code snapshot differs from run: {name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp." + uuid.uuid4().hex)
    temporary.mkdir()
    for source in code_files:
        target = temporary / source.name
        shutil.copy2(source, target)
        if sha256(target) != expected_hashes[source.name]:
            raise ValueError(f"Execution source changed while snapshotting: {source}")
    os.replace(temporary, destination)
    return destination


def metrics_from_predictions(y, prediction, baseline):
    """No finite-row deletion; true R2 is 1-SSE/SST, undefined for constants."""
    y, prediction, baseline = [np.asarray(a, dtype=float) for a in (y, prediction, baseline)]
    if len(y) == 0 or not all(np.isfinite(a).all() for a in (y, prediction, baseline)):
        raise ValueError("Empty/nonfinite held-out predictions or targets")
    error = y - prediction
    sst = float(np.square(y - y.mean()).sum())
    sse = float(np.square(error).sum())
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    base_rmse = float(np.sqrt(np.square(y - baseline).mean()))
    def correlation(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "test_pearson": correlation(y, prediction),
        "test_spearman": correlation(rankdata(y), rankdata(prediction)),
        "test_r2": 1.0 - sse / sst if len(y) >= 3 and sst > 0 else np.nan,
        "test_mae": mae, "test_rmse": rmse, "baseline_test_rmse": base_rmse,
        "rmse_improvement_vs_baseline": base_rmse - rmse,
        "mae_improvement_vs_baseline": float(np.abs(y - baseline).mean()) - mae,
        "constant_target": bool(sst == 0), "constant_prediction": bool(np.std(prediction) == 0),
    }


def seed_plan(config, original_index, shuffle_id, repeat):
    """Exactly the production seed arithmetic, unaffected by queue/resume order."""
    base = int(config["random_state"]) + int(original_index) * 10000
    if shuffle_id < 0:
        return None, base + repeat
    permutation_seed = base + 500000 + shuffle_id
    split_seed = base + 500000 + shuffle_id * 1100 + repeat
    return permutation_seed, split_seed


def train_block(task, data, shuffle_ids):
    config = task["config"]
    features = task["features"]
    x = data[features]
    original_y = data[config["target_col"]].to_numpy(float)
    samples = data.sample_id.astype(str).to_numpy()
    metrics, predictions, evidence = [], [], []
    for shuffle_id in shuffle_ids:
        permutation_seed, _ = seed_plan(config, task["original_task_index"], shuffle_id, 0)
        y = original_y if shuffle_id < 0 else np.random.default_rng(permutation_seed).permutation(original_y)
        label_status = "observed" if shuffle_id < 0 else "label_shuffle_null"
        for repeat in range(config["n_repeats"]):
            _, split_seed = seed_plan(config, task["original_task_index"], shuffle_id, repeat)
            train, test = train_test_split(np.arange(len(data)), test_size=config["test_size"], random_state=split_seed)
            selected = select_features_training_only(
                x.iloc[train], y[train], features,
                max_features=min(config["max_features_per_split"], len(features)), min_variance=1e-12,
            )
            pipe = make_xgb_pipeline(random_state=split_seed, n_estimators=config["n_estimators"],
                                     max_depth=config["max_depth"], learning_rate=config["learning_rate"],
                                     tree_method="hist", n_jobs=1)
            pipe.fit(x.iloc[train][selected], y[train])
            pred = pipe.predict(x.iloc[test][selected]).astype(float)
            baseline = np.repeat(float(y[train].mean()), len(test))
            identity = {"drug_key": task["drug_key"], "original_task_index": task["original_task_index"],
                        "label_status": label_status, "shuffle_id": shuffle_id, "repeat": repeat}
            metrics.append({**identity, "random_state": split_seed, "permutation_seed": permutation_seed,
                            "n_rows": len(data), "n_train": len(train), "n_test": len(test),
                            "n_features_available": len(features), "n_features_selected": len(selected),
                            "train_sample_ids": json.dumps(samples[train].tolist(), separators=(",", ":")),
                            **metrics_from_predictions(y[test], pred, baseline)})
            for idx, predicted, baseline_value in zip(test, pred, baseline):
                predictions.append({**identity, "sample_id": samples[idx], "row_index": int(idx),
                                    "y_true": float(y[idx]), "original_y": float(original_y[idx]),
                                    "prediction": float(predicted), "baseline_prediction": float(baseline_value)})
            for order, (feature, importance) in enumerate(zip(selected, pipe.named_steps["model"].feature_importances_)):
                evidence.append({**identity, "feature_name": feature, "feature_order": order,
                                 "gain_importance": float(importance), "selected": True})
    return {"metrics": pd.DataFrame(metrics), "predictions": pd.DataFrame(predictions), "features": pd.DataFrame(evidence)}


def block_name(shuffle_ids):
    return "observed" if shuffle_ids == [-1] else f"null_{shuffle_ids[0]:04d}_{shuffle_ids[-1]:04d}"


def expected_blocks(task):
    yield [-1]
    for first in range(0, task["config"]["n_shuffles"], task["block_size"]):
        yield list(range(first, min(first + task["block_size"], task["config"]["n_shuffles"])))


def verify_block(path, task, shuffle_ids):
    """Hash/schema/key validation, required both when resuming and consolidating."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest["task_signature"] != task["signature"] or manifest["shuffle_ids"] != shuffle_ids:
        raise ValueError(f"Incompatible checkpoint: {path}")
    for key, expected in manifest["files"].items():
        filename = path / key
        if not filename.exists() or sha256(filename) != expected["sha256"]:
            raise ValueError(f"Corrupt numerical checkpoint: {filename}")
    metrics = pd.read_parquet(path / "metrics.parquet")
    keys = set(zip(metrics.shuffle_id, metrics.repeat))
    expected = {(p, r) for p in shuffle_ids for r in range(task["config"]["n_repeats"])}
    if keys != expected or len(metrics) != len(expected) or set(metrics.drug_key) != {task["drug_key"]}:
        raise ValueError(f"Incomplete/duplicate numerical split keys: {path}")
    return True


def save_block(path, task, shuffle_ids, frames):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    files = {}
    for name, frame in frames.items():
        destination = temporary / f"{name}.parquet"
        frame.to_parquet(destination, index=False)
        files[destination.name] = {"sha256": sha256(destination), "rows": len(frame), "bytes": destination.stat().st_size}
    atomic_json(temporary / "manifest.json", {"schema": SCHEMA, "task_signature": task["signature"],
                                             "shuffle_ids": shuffle_ids, "files": files})
    if path.exists():
        raise FileExistsError(f"Refusing to replace numerical checkpoint {path}")
    os.replace(temporary, path)
    verify_block(path, task, shuffle_ids)


def run_task(task_path, stop_after_blocks=None):
    task_path = Path(task_path)
    task = json.loads(task_path.read_text())
    verify_task_signature(task)
    root = task_path.parent
    data_path = root / "input.parquet"
    if sha256(data_path) != task["slice_sha256"]:
        raise ValueError(f"Treatment input hash changed: {data_path}")
    data = pd.read_parquet(data_path)
    if data.sample_id.duplicated().any() or set(data.drug_key) != {task["drug_key"]}:
        raise ValueError("Duplicate sample-treatment identity or mixed task slice")
    completed = 0
    with threadpool_limits(limits=1):
        for shuffle_ids in expected_blocks(task):
            path = root / "blocks" / block_name(shuffle_ids)
            if not verify_block(path, task, shuffle_ids):
                frames = train_block(task, data, shuffle_ids)
                save_block(path, task, shuffle_ids, frames)
                completed += 1
                atomic_json(root / "progress.json", {"status": "RUNNING", "last_block": path.name,
                                                      "pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                print(f"task={task['original_task_index']} completed={path.name}", flush=True)
                if stop_after_blocks is not None and completed >= stop_after_blocks:
                    raise InterruptedError("Injected interruption after durable numerical block")
    block_hashes = {block_name(ids): sha256(root / "blocks" / block_name(ids) / "manifest.json") for ids in expected_blocks(task)}
    atomic_json(root / "complete.json", {"status": "NUMERICALLY_COMPLETE", "task_signature": task["signature"],
                                         "drug_key": task["drug_key"], "block_manifest_hashes": block_hashes,
                                         "n_shuffles": task["config"]["n_shuffles"], "n_repeats": task["config"]["n_repeats"]})
    return root


def prepare_tasks(args):
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "pair_dataset": Path(args.dataset_root) / "03_modeling_datasets/v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv",
        "registry": Path(args.step05_root) / "03_v2_strict_biology_registry/v2_strict_biology_feature_registry.tsv",
        "candidates": Path(args.step08_root) / "07_label_shuffle_handoff/tier1_label_shuffle_candidates.tsv",
    }
    config = {name: getattr(args, name) for name in ["target_col", "n_shuffles", "n_repeats", "test_size", "max_features_per_split",
                                                   "n_estimators", "max_depth", "learning_rate", "random_state", "fdr_threshold"]}
    if min(config["n_shuffles"], config["n_repeats"], args.block_size) < 1:
        raise ValueError("Counts must be positive")
    registry = pd.read_csv(paths["registry"], sep="\t")
    candidates = pd.read_csv(paths["candidates"], sep="\t")
    if candidates.drug_key.isna().any() or candidates.drug_key.duplicated().any():
        raise ValueError("Duplicate/missing candidate treatment identities")
    if registry.feature_name.isna().any() or registry.feature_name.duplicated().any():
        raise ValueError("Duplicate/missing registry features")
    features = registry.feature_name.astype(str).tolist()
    columns = ["sample_id", "drug_key", config["target_col"], *features]
    pair = pd.read_csv(paths["pair_dataset"], sep="\t", usecols=columns)
    if pair[["sample_id", "drug_key"]].isna().any().any() or pair.duplicated(["sample_id", "drug_key"]).any():
        raise ValueError("Duplicate/missing sample-treatment pair identity")
    pair["sample_id"] = pair.sample_id.astype(str)
    pair["drug_key"] = pair.drug_key.astype(str)
    source = {key: {"path": str(value.resolve()), "sha256": sha256(value)} for key, value in paths.items()}
    code_files = [Path(__file__), Path(__file__).with_name("model_training.py"), Path(__file__).with_name("conditional_validation_audit.py"),
                  Path(__file__).parents[2] / "scripts/09_label_shuffle_validate_tier1.py"]
    code_hashes = {path.name: sha256(path) for path in code_files}
    import sklearn, xgboost, scipy, pyarrow
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                   "xgboost": xgboost.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__}
    signature_base = {"schema": SCHEMA, "sources": source, "code": code_hashes, "config": config,
                      "environment": environment, "features": features, "candidate_order": candidates.drug_key.tolist(), "block_size": args.block_size}
    run_signature = stable_hash(signature_base)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["run_signature"] != run_signature:
            raise ValueError("Run signature mismatch: use a versioned sibling; never reuse incompatible results")
        save_execution_code(output, code_files, code_hashes)
    else:
        save_execution_code(output, code_files, code_hashes)
        atomic_json(manifest_path, {**signature_base, "run_signature": run_signature,
                    "analysis_role": "conditional_development_validation_fixed_global_registry_and_selected_candidates",
                    "aggregation": "arithmetic mean over all repeated splits; any undefined split propagates to undefined mean",
                    "undefined_null_policy": "retain all permutation IDs; undefined null mean counts as exceedance; null q95 unavailable if any undefined",
                    "acceptance": "q_pearson <= fdr_threshold AND observed_mean_pearson > null_95th_percentile AND mean_RMSE_improvement > 0"})
    atomic_table(output / "01_inputs/tier1_label_shuffle_candidates_used.tsv", candidates)
    atomic_table(output / "01_inputs/step05_registry_features_used.tsv", pd.DataFrame({"feature_name": features}))
    atomic_table(output / "01_inputs/source_manifest.tsv", pd.DataFrame([{"source_name": k, **v} for k, v in source.items()]))
    tasks, exclusion_rows = [], []
    for index, key in enumerate(candidates.drug_key.astype(str)):
        subset = pair.loc[pair.drug_key == key, columns].copy()
        y = pd.to_numeric(subset[config["target_col"]], errors="coerce")
        valid = np.isfinite(y)
        for sid in subset.loc[~valid, "sample_id"]:
            exclusion_rows.append({"drug_key": key, "sample_id": sid, "reason": "nonfinite_target"})
        subset = subset.loc[valid].reset_index(drop=True)
        subset[config["target_col"]] = y.loc[valid].to_numpy(float)
        subset[features] = subset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if subset.empty:
            raise ValueError(f"No finite target rows for candidate {key}")
        root = output / "09_checkpoints_by_treatment" / f"task_{index:03d}_{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        root.mkdir(parents=True, exist_ok=True)
        slice_path = root / "input.parquet"
        task_path = root / "task.json"
        if not task_path.exists():
            atomic_table(slice_path, subset)
            task = {"drug_key": key, "original_task_index": index, "features": features, "config": config,
                    "run_signature": run_signature, "block_size": args.block_size, "slice_sha256": sha256(slice_path),
                    "sample_ids": subset.sample_id.tolist()}
            task["signature"] = stable_hash(task)
            atomic_json(task_path, task)
        else:
            task = json.loads(task_path.read_text())
            verify_task_signature(task)
            if task["run_signature"] != run_signature or task["original_task_index"] != index or task["drug_key"] != key:
                raise ValueError(f"Task provenance mismatch: {root}")
            if sha256(slice_path) != task["slice_sha256"]:
                raise ValueError(f"Changed input slice {slice_path}")
        tasks.append(task_path)
    atomic_table(output / "01_inputs/explicit_target_exclusions.tsv", pd.DataFrame(exclusion_rows, columns=["drug_key", "sample_id", "reason"]))
    return tasks


def task_is_complete(path):
    task = json.loads(Path(path).read_text())
    verify_task_signature(task)
    root = Path(path).parent
    if not (root / "complete.json").exists():
        return False
    completion = json.loads((root / "complete.json").read_text())
    if completion["task_signature"] != task["signature"]:
        raise ValueError("Incompatible completed task")
    for ids in expected_blocks(task):
        block = root / "blocks" / block_name(ids)
        if not verify_block(block, task, ids) or sha256(block / "manifest.json") != completion["block_manifest_hashes"][block.name]:
            raise ValueError(f"Unverified completed task {root}")
    return True


def run_controller(task_paths, output, workers=2, task_timeout=7200, deadline_utc=None):
    """Small owned-process supervisor: actual exit codes, bounded serial recovery."""
    output = Path(output)
    log_root = output / "controller_logs"
    log_root.mkdir(exist_ok=True)
    pending = [(p, 0) for p in task_paths if not task_is_complete(p)]
    active, complete, failures = {}, len(task_paths) - len(pending), []
    workers = max(1, min(int(workers) if workers else 2, 2))
    next_heartbeat = 0.0
    deadline = pd.Timestamp(deadline_utc).timestamp() if deadline_utc else float("inf")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    journal = log_root / "commands.jsonl"
    def record(value):
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value}) + "\n")
    def stop_owned(proc):
        try:
            owner = psutil.Process(proc.pid)
            for child in owner.children(recursive=True):
                child.terminate()
            owner.terminate()
            proc.wait(timeout=10)
        except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
            if proc.poll() is None:
                proc.kill()
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        while pending or active:
            if time.time() >= deadline:
                raise TimeoutError("Campaign deadline reached; durable blocks preserved")
            while pending and len(active) < workers:
                path, attempt = pending.pop(0)
                stem = path.parent.name + f"_attempt{attempt}"
                stdout = (log_root / (stem + ".stdout.log")).open("a", encoding="utf-8")
                stderr = (log_root / (stem + ".stderr.log")).open("a", encoding="utf-8")
                command = [sys.executable, "-m", "spm_v2.conditional_validation", "--worker-task", str(path)]
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active[proc.pid] = {"proc": proc, "path": path, "attempt": attempt, "start": time.time(), "handles": (stdout, stderr)}
                record({"event": "START", "pid": proc.pid, "command": command, "attempt": attempt})
            for pid, item in list(active.items()):
                proc = item["proc"]
                timed_out = time.time() - item["start"] > task_timeout
                if timed_out and proc.poll() is None:
                    stop_owned(proc)
                code = proc.poll()
                if code is None:
                    continue
                for handle in item["handles"]:
                    handle.close()
                record({"event": "EXIT", "pid": pid, "returncode": code, "elapsed_seconds": time.time() - item["start"], "timeout": timed_out})
                del active[pid]
                if code == 0 and task_is_complete(item["path"]):
                    complete += 1
                    print(f"NUMERICALLY_COMPLETE {complete}/{len(task_paths)} {item['path'].parent.name}", flush=True)
                else:
                    failure_path = item["path"].parent / "failure.json"
                    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
                    transient = timed_out or code < 0 or code > 255 or failure.get("transient", False)
                    if transient and item["attempt"] < 2:
                        workers = 1
                        pending.append((item["path"], item["attempt"] + 1))
                        record({"event": "RETRY_SERIAL", "task": str(item["path"]), "reason": failure or f"abnormal exit {code}"})
                        time.sleep(min(2 ** item["attempt"], 4))
                    else:
                        failures.append({"task": str(item["path"]), "returncode": code, "timeout": timed_out, "error": failure})
            if time.time() >= next_heartbeat:
                resource_rows = []
                for pid, item in active.items():
                    try:
                        process = psutil.Process(pid)
                        resource_rows.append({"pid": pid, "task": item["path"].parent.name, "rss_bytes": process.memory_info().rss,
                                              "elapsed_seconds": time.time() - item["start"]})
                    except psutil.NoSuchProcess:
                        pass
                state = {"status": "RUNNING", "completed": complete, "total": len(task_paths), "pending": len(pending),
                         "failed": failures, "active": resource_rows, "workers": workers,
                         "available_ram_bytes": psutil.virtual_memory().available, "cpu_percent": psutil.cpu_percent(),
                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                atomic_json(output / "controller_state.json", state)
                record({"event": "HEARTBEAT", **state})
                print(f"HEARTBEAT complete={complete}/{len(task_paths)} active={len(active)} pending={len(pending)} failed={len(failures)}", flush=True)
                next_heartbeat = time.time() + 60
            if active:
                time.sleep(1)
        if failures:
            atomic_json(output / "controller_state.json", {"status": "FAIL", "failed": failures, "completed": complete})
            raise RuntimeError(f"{len(failures)} treatment tasks failed; see controller_state.json")
        atomic_json(output / "controller_state.json", {"status": "NUMERICALLY_COMPLETE", "completed": complete, "total": len(task_paths)})
    finally:
        for item in active.values():
            stop_owned(item["proc"])
            for handle in item["handles"]:
                handle.close()
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def strict_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan


def bh(pvalues):
    values = np.asarray(pvalues, float)
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def empirical_p(observed, null):
    null = np.asarray(null, float)
    if not np.isfinite(observed):
        return 1.0
    return float((1 + np.count_nonzero((null >= observed) | ~np.isfinite(null))) / (1 + len(null)))


def summarize(observed, null, candidates, config):
    observed_rows, null_rows = [], []
    for key, frame in observed.groupby("drug_key", sort=False):
        row = {"drug_key": key, "n_observed_repeats": len(frame), "n_rows": int(frame.n_rows.iloc[0]),
               "n_features_available": int(frame.n_features_available.iloc[0]), "n_features_selected_median": int(frame.n_features_selected.median())}
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            for stat, value in [("mean", strict_mean(values)), ("median", np.median(values)), ("std", np.std(values, ddof=1))]:
                row[f"observed_{metric}_{stat}"] = value
            row[f"observed_{metric}_undefined_splits"] = int((~np.isfinite(values)).sum())
        row["observed_test_pearson_positive_fraction"] = float((frame.test_pearson > 0).mean())
        observed_rows.append(row)
    for (key, shuffle_id), frame in null.groupby(["drug_key", "shuffle_id"], sort=False):
        row = {"drug_key": key, "shuffle_id": int(shuffle_id), "n_repeats": len(frame)}
        for metric in METRICS:
            row[f"null_{metric}_mean"] = strict_mean(frame[metric])
            row[f"null_{metric}_undefined_splits"] = int(frame[metric].isna().sum())
        null_rows.append(row)
    obs = pd.DataFrame(observed_rows)
    nul = pd.DataFrame(null_rows)
    rows, null_summary_rows = [], []
    for _, record in obs.iterrows():
        key = record.drug_key
        frame = nul[nul.drug_key == key]
        row = {"drug_key": key, "n_null_shuffles": len(frame), "original_task_index": int(candidates.index[candidates.drug_key == key][0])}
        for metric in ("test_pearson", "test_r2"):
            values = frame[f"null_{metric}_mean"].to_numpy(float)
            value = record[f"observed_{metric}_mean"]
            row[f"observed_{metric}_mean"] = value
            for stat, result in [("mean", strict_mean(values)), ("median", np.median(values)), ("q95", np.quantile(values, .95)), ("max", np.max(values))]:
                row[f"null_{metric}_mean_{stat}"] = result
            row[f"empirical_p_{metric[5:]}"] = empirical_p(value, values)
            row[f"{metric[5:]}_effect_vs_null_median"] = value - np.median(values)
            row[f"n_undefined_null_{metric}"] = int((~np.isfinite(values)).sum())
        row["observed_rmse_improvement_vs_baseline_mean"] = record.observed_rmse_improvement_vs_baseline_mean
        rows.append(row)
        null_summary_rows.append({k: v for k, v in row.items() if k.startswith("null_") or k in ["drug_key", "n_null_shuffles"]})
    results = pd.DataFrame(rows)
    results["fdr_q_pearson"] = bh(results.empirical_p_pearson)
    results["fdr_q_r2"] = bh(results.empirical_p_r2)
    results["validated_for_step10"] = ((results.fdr_q_pearson <= config["fdr_threshold"]) &
                                        (results.observed_test_pearson_mean > results.null_test_pearson_mean_q95) &
                                        (results.observed_rmse_improvement_vs_baseline_mean > 0))
    results["label_shuffle_validation_status"] = np.where(results.validated_for_step10, "validated_tier1_spatial_signal", "not_label_shuffle_validated")
    results["validation_scope"] = "conditional_on_global_registry_and_preselected_candidates"
    results = results.merge(candidates, on="drug_key", how="left", validate="one_to_one", suffixes=("", "_step08"))
    results = results.sort_values(["validated_for_step10", "fdr_q_pearson", "observed_test_pearson_mean"], ascending=[False, True, False])
    return obs, nul, pd.DataFrame(null_summary_rows), results


def load_legacy_helpers():
    path = Path(__file__).parents[2] / "scripts/09_label_shuffle_validate_tier1.py"
    spec = importlib.util.spec_from_file_location("step09_legacy_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def consolidate(output, task_paths):
    output = Path(output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    observed_all, null_all, observed_features = [], [], []
    numerical_manifest = []
    for path in task_paths:
        if not task_is_complete(path):
            raise ValueError(f"Incomplete task: {path}")
        task = json.loads(path.read_text())
        for ids in expected_blocks(task):
            root = path.parent / "blocks" / block_name(ids)
            metrics = pd.read_parquet(root / "metrics.parquet")
            (observed_all if ids == [-1] else null_all).append(metrics)
            if ids == [-1]:
                observed_features.append(pd.read_parquet(root / "features.parquet"))
            for filename in ["metrics.parquet", "predictions.parquet", "features.parquet", "manifest.json"]:
                p = root / filename
                numerical_manifest.append({"relative_path": str(p.relative_to(output)), "sha256": sha256(p), "size_bytes": p.stat().st_size,
                                           "drug_key": task["drug_key"], "original_task_index": task["original_task_index"]})
    observed, null, features = pd.concat(observed_all, ignore_index=True), pd.concat(null_all, ignore_index=True), pd.concat(observed_features, ignore_index=True)
    candidates = pd.read_csv(output / "01_inputs/tier1_label_shuffle_candidates_used.tsv", sep="\t")
    registry = pd.read_csv(manifest["sources"]["registry"]["path"], sep="\t")
    obs_summary, null_by_shuffle, null_summary, results = summarize(observed, null, candidates, manifest["config"])
    legacy = load_legacy_helpers()
    recurrent = legacy.summarize_feature_recurrence(features, results, registry)
    themes = legacy.summarize_theme_recurrence(recurrent)
    if recurrent.empty:
        recurrent = pd.DataFrame(columns=["feature_name", "validated_treatment_count", "observed_selection_count", "mean_gain_importance", "max_gain_importance", "biological_theme"])
    if themes.empty:
        themes = pd.DataFrame(columns=["biological_theme", "n_features", "validated_treatment_count", "total_gain_importance", "max_gain_importance", "example_features"])
    accepted = set(results.loc[results.validated_for_step10, "drug_key"])
    themes = report_distinct_theme_counts(themes, features, registry, accepted)
    validated = results[results.validated_for_step10].copy()
    rejected = results[~results.validated_for_step10].copy()
    validated["step10_validation_status"] = "label_shuffle_validated"
    rejected["step10_validation_status"] = "not_label_shuffle_validated"
    tables = {
        "02_observed_models/tier1_observed_repeated_split_metrics_long.tsv": observed,
        "02_observed_models/tier1_observed_metric_summary.tsv": obs_summary,
        "02_observed_models/tier1_observed_feature_evidence_long.tsv": features,
        "03_label_shuffle_null/tier1_label_shuffle_null_metrics_long.tsv": null,
        "03_label_shuffle_null/tier1_label_shuffle_null_metric_by_shuffle.tsv": null_by_shuffle,
        "03_label_shuffle_null/tier1_label_shuffle_null_summary.tsv": null_summary,
        "04_validation_results/tier1_label_shuffle_validation_results.tsv": results,
        "04_validation_results/tier1_label_shuffle_validated_treatments.tsv": validated,
        "04_validation_results/tier1_label_shuffle_not_validated_treatments.tsv": rejected,
        "05_validated_features_and_themes/label_shuffle_validated_recurrent_spatial_features.tsv": recurrent,
        "05_validated_features_and_themes/label_shuffle_validated_recurrent_biology_themes.tsv": themes,
        "08_step10_handoff/label_shuffle_validated_treatments_for_step10.tsv": validated,
        "08_step10_handoff/all_label_shuffle_results_for_step10.tsv": results,
        "numerical_results_manifest.tsv": pd.DataFrame(numerical_manifest),
    }
    for name, frame in tables.items():
        atomic_table(output / name, frame)
    summary = {"status": "NUMERICALLY_COMPLETE_AUDIT_PENDING", "official_step": "09_label_shuffle_validate_tier1",
               "analysis_role": manifest["analysis_role"], "output_root": str(output), "target_col": manifest["config"]["target_col"],
               "n_tier1_candidates_tested": len(candidates), "n_shuffles": manifest["config"]["n_shuffles"],
               "n_repeats": manifest["config"]["n_repeats"], "n_features_used": len(manifest["features"]),
               "n_step05_registry_features": len(manifest["features"]), "n_label_shuffle_validated_treatments": len(validated),
               "n_not_validated_treatments": len(rejected), "fdr_threshold": manifest["config"]["fdr_threshold"],
               "independent_discovery_validation": False, "run_signature": manifest["run_signature"]}
    atomic_json(output / "v2_step09_tier1_label_shuffle_validation_summary.json", summary)
    atomic_json(output / "08_step10_handoff/step10_handoff_summary.json", {**summary, "status": "conditional_development_handoff_audit_pending"})
    legacy.save_observed_vs_null_plot(results, output / "06_figures/fig_05_observed_vs_null_q95_pearson.png")
    report = ("Corrected conditional Step 09 validation\n\n"
              "The fixed registry and treatment candidates were learned/selected using this cohort. These P and q values test association conditional on those choices; they do not validate the full discovery procedure independently.\n\n"
              f"Candidates: {len(candidates)}; permutations per candidate: {manifest['config']['n_shuffles']}; repeated splits: {manifest['config']['n_repeats']}. Conditional passes: {len(validated)}.\n"
              "The arithmetic mean includes all split records. Undefined split statistics propagate; undefined null statistics count as exceedances and retain the full denominator. Constant-target R2/correlations remain NA. R2 = 1 - SSE/SST, including negative values.\n"
              "Saved held-out predictions, training membership, selected feature order and importances are indexed in numerical_results_manifest.tsv. Deployment and independent evaluations remain separate.\n")
    (output / "07_reports").mkdir(exist_ok=True)
    (output / "07_reports/v2_step09_tier1_label_shuffle_validation_report.txt").write_text(report, encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-task")
    p.add_argument("--run-root")
    p.add_argument("--dataset-root")
    p.add_argument("--step05-root")
    p.add_argument("--step08-root")
    p.add_argument("--output-root")
    p.add_argument("--target-col", default="fused_residual_vs_prior")
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--test-size", type=float, default=.2)
    p.add_argument("--max-features-per-split", type=int, default=60)
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.03)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fdr-threshold", type=float, default=.10)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--block-size", type=int, default=25)
    p.add_argument("--task-timeout-seconds", type=int, default=7200)
    p.add_argument("--deadline-utc")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_task:
        try:
            run_task(args.worker_task)
        except Exception as exc:
            atomic_json(Path(args.worker_task).parent / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "transient": isinstance(exc, (MemoryError, OSError))})
            raise
        return 0
    for name in ["run_root", "dataset_root", "step05_root", "step08_root", "output_root"]:
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    tasks = prepare_tasks(args)
    if args.prepare_only:
        print(f"Prepared {len(tasks)} provenance-bound treatment slices; no models run.", flush=True)
        return 0
    if not args.audit_only:
        run_controller(tasks, args.output_root, args.max_workers, args.task_timeout_seconds, args.deadline_utc)
        summary = consolidate(args.output_root, tasks)
        print(json.dumps(summary, indent=2), flush=True)
    from .conditional_validation_audit import audit_run
    result = audit_run(Path(args.output_root))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
