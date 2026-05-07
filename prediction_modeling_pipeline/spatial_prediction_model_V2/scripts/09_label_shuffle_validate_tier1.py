"""
Script: 09_label_shuffle_validate_tier1.py

Purpose:
    Validate Step 08 Tier 1 treatment-specific residual models by label
    shuffling.

Pipeline role:
    This step consumes Step 08 Tier 1 candidates, retrains observed treatment-
    specific models, builds within-treatment shuffled-label null models,
    estimates empirical p values, applies FDR correction, and writes the Step 10
    validation handoff.

Scientific role:
    Label-shuffle validation tests whether treatment-specific spatial residual
    signal is stronger than expected when response residuals are permuted within
    the same treatment. This guards against overinterpreting unstable treatment-
    specific screens as spatial biology claims.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP09_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    imports, constants, thresholds, hyperparameters, validation rules,
    output filenames, and return codes must remain unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
# Use a non-interactive backend so validation figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.io_utils import ensure_dir, read_table, write_json, write_table, write_text_report
from spm_v2.model_training import make_xgb_pipeline, select_features_training_only
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest


# =============================================================================
# Helper functions
# =============================================================================

def safe_float(value):
    """Convert a value to float while returning None for invalid or missing values."""

    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def metric_safe_local(y_true, y_pred) -> dict:
    """Compute regression metrics while handling constant or invalid inputs."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    out = {
        "n": int(len(y_true)),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }

    if len(y_true) == 0:
        return out

    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)

    if len(y_true) >= 3 and np.nanstd(y_true) > 0:
        try:
            out["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            pass

    if len(y_true) >= 3 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            try:
                out["pearson"] = float(pearsonr(y_true, y_pred)[0])
            except Exception:
                pass
            try:
                out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
            except Exception:
                pass

    return out


def bh_fdr(p_values) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to p-values."""

    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)

    n = len(p)
    if n == 0:
        return np.array([])

    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        value = min(prev, p[idx] * n / rank)
        q[idx] = value
        prev = value

    return q


def build_source_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    """Build a source-file manifest with path existence and size metadata."""

    rows = []
    for key, path in paths.items():
        exists = bool(path.exists())
        rows.append({
            "source_name": key,
            "path": str(path),
            "exists": exists,
            "is_file": bool(exists and path.is_file()),
            "size_bytes": int(path.stat().st_size) if exists and path.is_file() else "",
        })
    return pd.DataFrame(rows)


def resolve_feature_cols(pair_df: pd.DataFrame, registry: pd.DataFrame) -> list[str]:
    """Resolve Step 05 registry features that are present in the pair-level dataset."""

    if "feature_name" not in registry.columns:
        raise ValueError("Step 05 strict biology registry must contain feature_name.")

    features = [str(x) for x in registry["feature_name"].astype(str).tolist() if str(x) in pair_df.columns]
    features = list(dict.fromkeys(features))

    if len(features) < 10:
        raise ValueError(f"Too few Step 05 registry features are present in pair dataset: {len(features)}")

    return features


def prepare_treatment_frame(
    pair_df: pd.DataFrame,
    drug_key: str,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Subset one treatment and return aligned feature and residual target arrays."""

    sub = pair_df[pair_df["drug_key"].astype(str) == str(drug_key)].copy()

    if sub.empty:
        raise ValueError(f"No rows found for treatment: {drug_key}")

    y = pd.to_numeric(sub[target_col], errors="coerce")
    valid = y.notna()

    sub = sub.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(float).reset_index(drop=True)

    x = sub[feature_cols].copy()

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return sub, x, y


def train_repeated_model(
    drug_key: str,
    sub: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    label_status: str,
    shuffle_id: int,
    n_repeats: int,
    test_size: float,
    max_features_per_split: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train repeated observed or shuffled-label residual models for one treatment."""

    metric_rows = []
    feature_rows = []

    feature_cols = list(x.columns)

    for repeat in range(n_repeats):
        seed = random_state + repeat + max(shuffle_id, 0) * 1000

        train_idx, test_idx = train_test_split(
            np.arange(len(sub)),
            test_size=test_size,
            random_state=seed,
        )

        selected = select_features_training_only(
            x_train=x.iloc[train_idx],
            y_train=y.iloc[train_idx],
            feature_cols=feature_cols,
            max_features=min(max_features_per_split, len(feature_cols)),
            min_variance=1e-12,
        )

        pipe = make_xgb_pipeline(
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            n_jobs=1,
        )

        pipe.fit(x.iloc[train_idx][selected], y.iloc[train_idx])

        pred_test = pipe.predict(x.iloc[test_idx][selected])
        baseline_test = np.repeat(float(y.iloc[train_idx].mean()), len(test_idx))

        test_m = metric_safe_local(y.iloc[test_idx], pred_test)
        base_m = metric_safe_local(y.iloc[test_idx], baseline_test)

        metric_rows.append({
            "drug_key": drug_key,
            "label_status": label_status,
            "shuffle_id": int(shuffle_id),
            "repeat": int(repeat),
            "random_state": int(seed),
            "n_rows": int(len(sub)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features_available": int(len(feature_cols)),
            "n_features_selected": int(len(selected)),
            "test_pearson": test_m["pearson"],
            "test_spearman": test_m["spearman"],
            "test_r2": test_m["r2"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "baseline_test_rmse": base_m["rmse"],
            "rmse_improvement_vs_baseline": base_m["rmse"] - test_m["rmse"],
            "mae_improvement_vs_baseline": base_m["mae"] - test_m["mae"],
        })

        for feature, importance in zip(selected, pipe.named_steps["model"].feature_importances_):
            feature_rows.append({
                "drug_key": drug_key,
                "label_status": label_status,
                "shuffle_id": int(shuffle_id),
                "repeat": int(repeat),
                "feature_name": feature,
                "gain_importance": float(importance),
                "selected": True,
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(feature_rows)


def run_one_treatment(args_dict: dict) -> dict:
    """Run observed and shuffled-label validation for one Tier 1 treatment candidate."""

    drug_key = args_dict["drug_key"]
    pair_path = Path(args_dict["pair_path"])
    registry_path = Path(args_dict["registry_path"])
    target_col = args_dict["target_col"]
    n_shuffles = int(args_dict["n_shuffles"])
    n_repeats = int(args_dict["n_repeats"])
    test_size = float(args_dict["test_size"])
    max_features_per_split = int(args_dict["max_features_per_split"])
    n_estimators = int(args_dict["n_estimators"])
    max_depth = int(args_dict["max_depth"])
    learning_rate = float(args_dict["learning_rate"])
    random_state = int(args_dict["random_state"])
    treatment_idx = int(args_dict["treatment_idx"])

    pair_df = read_table(pair_path)
    registry = read_table(registry_path)

    pair_df["drug_key"] = pair_df["drug_key"].astype(str)
    pair_df["sample_id"] = pair_df["sample_id"].astype(str)

    feature_cols = resolve_feature_cols(pair_df, registry)

    sub, x, y_true = prepare_treatment_frame(
        pair_df=pair_df,
        drug_key=drug_key,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    observed_metrics, observed_feature = train_repeated_model(
        drug_key=drug_key,
        sub=sub,
        x=x,
        y=y_true,
        label_status="observed",
        shuffle_id=-1,
        n_repeats=n_repeats,
        test_size=test_size,
        max_features_per_split=max_features_per_split,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state + treatment_idx * 10000,
    )

    null_metrics_all = []
    base_seed = random_state + treatment_idx * 10000 + 500000

    for shuffle_id in range(n_shuffles):
        rng = np.random.default_rng(base_seed + shuffle_id)
        # Shuffle labels within each treatment to form the null distribution without changing features.
        y_shuffled = pd.Series(rng.permutation(y_true.values), index=y_true.index, dtype=float)

        null_metrics, _ = train_repeated_model(
            drug_key=drug_key,
            sub=sub,
            x=x,
            y=y_shuffled,
            label_status="label_shuffle_null",
            shuffle_id=shuffle_id,
            n_repeats=n_repeats,
            test_size=test_size,
            max_features_per_split=max_features_per_split,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=base_seed + shuffle_id * 100,
        )

        null_metrics_all.append(null_metrics)

    null_metrics = pd.concat(null_metrics_all, ignore_index=True) if null_metrics_all else pd.DataFrame()

    return {
        "drug_key": drug_key,
        "observed_metrics": observed_metrics.to_dict(orient="records"),
        "null_metrics": null_metrics.to_dict(orient="records"),
        "observed_feature": observed_feature.to_dict(orient="records"),
        "n_rows": int(len(sub)),
        "n_features": int(len(feature_cols)),
    }


def summarize_observed(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed repeated-split metrics by treatment."""

    rows = []

    for drug_key, sub in metrics.groupby("drug_key", dropna=False):
        row = {
            "drug_key": str(drug_key),
            "n_observed_repeats": int(len(sub)),
            "n_rows": int(pd.to_numeric(sub["n_rows"], errors="coerce").median()),
            "n_features_available": int(pd.to_numeric(sub["n_features_available"], errors="coerce").median()),
            "n_features_selected_median": int(pd.to_numeric(sub["n_features_selected"], errors="coerce").median()),
        }

        for col in [
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_mae",
            "test_rmse",
            "baseline_test_rmse",
            "rmse_improvement_vs_baseline",
            "mae_improvement_vs_baseline",
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"observed_{col}_mean"] = safe_float(vals.mean()) if vals.notna().sum() else None
            row[f"observed_{col}_median"] = safe_float(vals.median()) if vals.notna().sum() else None
            row[f"observed_{col}_std"] = safe_float(vals.std(ddof=1)) if vals.notna().sum() > 1 else None

        vals = pd.to_numeric(sub["test_pearson"], errors="coerce")
        row["observed_test_pearson_positive_fraction"] = safe_float((vals > 0).mean()) if vals.notna().sum() else None
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_null(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize label-shuffle null metrics by treatment and shuffle."""

    shuffle_rows = []

    for (drug_key, shuffle_id), sub in metrics.groupby(["drug_key", "shuffle_id"], dropna=False):
        row = {
            "drug_key": str(drug_key),
            "shuffle_id": int(shuffle_id),
            "n_repeats": int(len(sub)),
        }

        for col in [
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_rmse",
            "rmse_improvement_vs_baseline",
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"null_{col}_mean"] = safe_float(vals.mean()) if vals.notna().sum() else None

        shuffle_rows.append(row)

    null_by_shuffle = pd.DataFrame(shuffle_rows)

    summary_rows = []
    for drug_key, sub in null_by_shuffle.groupby("drug_key", dropna=False):
        row = {
            "drug_key": str(drug_key),
            "n_null_shuffles": int(len(sub)),
        }

        for col in [
            "null_test_pearson_mean",
            "null_test_spearman_mean",
            "null_test_r2_mean",
            "null_test_rmse_mean",
            "null_rmse_improvement_vs_baseline_mean",
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = safe_float(vals.mean()) if vals.notna().sum() else None
            row[f"{col}_median"] = safe_float(vals.median()) if vals.notna().sum() else None
            row[f"{col}_q95"] = safe_float(vals.quantile(0.95)) if vals.notna().sum() else None
            row[f"{col}_max"] = safe_float(vals.max()) if vals.notna().sum() else None

        summary_rows.append(row)

    null_summary = pd.DataFrame(summary_rows)
    return null_by_shuffle, null_summary


def validate_against_null(
    observed_summary: pd.DataFrame,
    null_by_shuffle: pd.DataFrame,
    candidates: pd.DataFrame,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Compare observed treatment metrics with shuffled-label null distributions."""

    rows = []
    candidate_meta = candidates.copy()

    for _, obs in observed_summary.iterrows():
        drug_key = str(obs["drug_key"])
        null = null_by_shuffle[null_by_shuffle["drug_key"].astype(str) == drug_key].copy()

        obs_pearson = safe_float(obs.get("observed_test_pearson_mean"))
        obs_r2 = safe_float(obs.get("observed_test_r2_mean"))
        obs_rmse_improvement = safe_float(obs.get("observed_rmse_improvement_vs_baseline_mean"))

        null_pearson = pd.to_numeric(null.get("null_test_pearson_mean"), errors="coerce").dropna()
        null_r2 = pd.to_numeric(null.get("null_test_r2_mean"), errors="coerce").dropna()

        if obs_pearson is None or len(null_pearson) == 0:
            # Empirical p-values compare observed performance with treatment-specific shuffled-label nulls.
            empirical_p_pearson = 1.0
        else:
            empirical_p_pearson = float((1 + (null_pearson >= obs_pearson).sum()) / (1 + len(null_pearson)))

        if obs_r2 is None or len(null_r2) == 0:
            empirical_p_r2 = 1.0
        else:
            empirical_p_r2 = float((1 + (null_r2 >= obs_r2).sum()) / (1 + len(null_r2)))

        row = {
            "drug_key": drug_key,
            "observed_test_pearson_mean": obs_pearson,
            "observed_test_r2_mean": obs_r2,
            "observed_rmse_improvement_vs_baseline_mean": obs_rmse_improvement,
            "null_test_pearson_mean_mean": safe_float(null_pearson.mean()) if len(null_pearson) else None,
            "null_test_pearson_mean_median": safe_float(null_pearson.median()) if len(null_pearson) else None,
            "null_test_pearson_mean_q95": safe_float(null_pearson.quantile(0.95)) if len(null_pearson) else None,
            "null_test_pearson_mean_max": safe_float(null_pearson.max()) if len(null_pearson) else None,
            "null_test_r2_mean_mean": safe_float(null_r2.mean()) if len(null_r2) else None,
            "null_test_r2_mean_median": safe_float(null_r2.median()) if len(null_r2) else None,
            "null_test_r2_mean_q95": safe_float(null_r2.quantile(0.95)) if len(null_r2) else None,
            "null_test_r2_mean_max": safe_float(null_r2.max()) if len(null_r2) else None,
            "empirical_p_pearson": empirical_p_pearson,
            "empirical_p_r2": empirical_p_r2,
            "pearson_effect_vs_null_median": safe_float(obs_pearson - null_pearson.median()) if obs_pearson is not None and len(null_pearson) else None,
            "r2_effect_vs_null_median": safe_float(obs_r2 - null_r2.median()) if obs_r2 is not None and len(null_r2) else None,
            "n_null_shuffles": int(len(null_pearson)),
        }
        rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    # Apply FDR across Tier 1 candidates before declaring validated treatment-specific spatial signal.
    results["fdr_q_pearson"] = bh_fdr(results["empirical_p_pearson"].values)
    results["fdr_q_r2"] = bh_fdr(results["empirical_p_r2"].values)

    results = results.merge(candidate_meta, on="drug_key", how="left", suffixes=("", "_step08"))

    results["label_shuffle_validation_status"] = np.where(
        (pd.to_numeric(results["fdr_q_pearson"], errors="coerce") <= fdr_threshold)
        & (pd.to_numeric(results["observed_test_pearson_mean"], errors="coerce") > pd.to_numeric(results["null_test_pearson_mean_q95"], errors="coerce"))
        & (pd.to_numeric(results["observed_rmse_improvement_vs_baseline_mean"], errors="coerce") > 0),
        "validated_tier1_spatial_signal",
        "not_label_shuffle_validated",
    )

    results["validated_for_step10"] = results["label_shuffle_validation_status"].eq("validated_tier1_spatial_signal")
    results = results.sort_values(
        ["validated_for_step10", "fdr_q_pearson", "observed_test_pearson_mean"],
        ascending=[False, True, False],
    )
    return results


def summarize_feature_recurrence(feature_evidence: pd.DataFrame, validation_results: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Summarize recurrent spatial features among label-shuffle-validated treatments."""

    if feature_evidence.empty or validation_results.empty:
        return pd.DataFrame()

    validated_drugs = validation_results.loc[
        validation_results["validated_for_step10"] == True,
        "drug_key"
    ].astype(str).tolist()

    subset = feature_evidence[
        (feature_evidence["label_status"].astype(str) == "observed")
        & (feature_evidence["drug_key"].astype(str).isin(validated_drugs))
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset
        .groupby("feature_name", dropna=False)
        .agg(
            validated_treatment_count=("drug_key", lambda s: int(s.nunique())),
            observed_selection_count=("feature_name", "count"),
            mean_gain_importance=("gain_importance", "mean"),
            max_gain_importance=("gain_importance", "max"),
        )
        .reset_index()
        .sort_values(["validated_treatment_count", "mean_gain_importance"], ascending=False)
    )

    if "feature_name" in registry.columns:
        keep_cols = ["feature_name"]
        for col in ["feature_original", "feature_group", "feature_axis", "biological_theme", "interpretation_class", "interpretation_note"]:
            if col in registry.columns:
                keep_cols.append(col)
        grouped = grouped.merge(registry[keep_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

    examples = []
    for feature in grouped["feature_name"].astype(str).tolist():
        sub = subset[subset["feature_name"].astype(str) == feature].copy()
        sub = sub.sort_values("gain_importance", ascending=False)
        examples.append("; ".join(sub["drug_key"].astype(str).drop_duplicates().head(5).tolist()))

    grouped["example_validated_treatments"] = examples
    return grouped


def summarize_theme_recurrence(feature_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize recurrent biological themes among validated feature evidence."""

    if feature_summary.empty or "biological_theme" not in feature_summary.columns:
        return pd.DataFrame()

    out = (
        feature_summary
        .groupby("biological_theme", dropna=False)
        .agg(
            n_features=("feature_name", "count"),
            validated_treatment_count=("validated_treatment_count", "sum"),
            total_gain_importance=("mean_gain_importance", "sum"),
            max_gain_importance=("mean_gain_importance", "max"),
        )
        .reset_index()
        .sort_values("total_gain_importance", ascending=False)
    )

    examples = []
    for theme in out["biological_theme"].astype(str).tolist():
        sub = feature_summary[feature_summary["biological_theme"].astype(str) == theme].copy()
        sub = sub.sort_values("mean_gain_importance", ascending=False)
        names = sub.get("feature_original", sub["feature_name"]).astype(str).head(4).tolist()
        examples.append("; ".join(names))
    out["example_features"] = examples
    return out


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, output_path: Path, title: str, xlabel: str, top_n: int = 30) -> None:
    """Save a ranked horizontal bar plot for model, feature, or theme evidence."""

    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    plot = df.copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)

    if plot.empty:
        return

    labels = plot[label_col].astype(str).tolist()
    labels = [x if len(x) <= 78 else x[:75] + "..." for x in labels]
    y = np.arange(len(plot))[::-1]
    values = plot[value_col].to_numpy()[::-1]
    labels = labels[::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, max(6, len(plot) * 0.34)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_observed_vs_null_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Plot observed treatment performance against the shuffled-label null threshold."""

    if results.empty:
        return

    plot = results.copy()
    plot["observed_test_pearson_mean"] = pd.to_numeric(plot["observed_test_pearson_mean"], errors="coerce")
    plot["null_test_pearson_mean_q95"] = pd.to_numeric(plot["null_test_pearson_mean_q95"], errors="coerce")
    plot = plot.dropna(subset=["observed_test_pearson_mean", "null_test_pearson_mean_q95"])

    if plot.empty:
        return

    plot = plot.sort_values("observed_test_pearson_mean", ascending=False)
    labels = [x if len(x) <= 65 else x[:62] + "..." for x in plot["drug_key"].astype(str).tolist()]
    x = np.arange(len(plot))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(plot) * 1.2), 6))
    plt.plot(x, plot["observed_test_pearson_mean"].values, marker="o", label="observed")
    plt.plot(x, plot["null_test_pearson_mean_q95"].values, marker="o", label="null q95")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Mean test Pearson")
    plt.title("V2 Step 09 observed versus shuffled null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Run this spatial_prediction_model_V2 step and write tables, reports, provenance, and summaries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--step05-root", required=True)
    parser.add_argument("--step08-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-col", default="fused_residual_vs_prior")
    parser.add_argument("--n-shuffles", type=int, default=25)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-features-per-split", type=int, default=60)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fdr-threshold", type=float, default=0.10)
    parser.add_argument("--max-workers", type=int, default=0, help="Maximum parallel treatment workers. Use 0 for automatic half-CPU capped worker selection.")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    step05_root = Path(args.step05_root)
    step08_root = Path(args.step08_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_observed_models")
    d03 = ensure_dir(output_root / "03_label_shuffle_null")
    d04 = ensure_dir(output_root / "04_validation_results")
    d05 = ensure_dir(output_root / "05_validated_features_and_themes")
    d06 = ensure_dir(output_root / "06_figures")
    d07 = ensure_dir(output_root / "07_reports")
    d08 = ensure_dir(output_root / "08_step10_handoff")
    d09 = ensure_dir(output_root / "09_checkpoints_by_treatment")

    pair_path = dataset_root / "03_modeling_datasets" / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"
    registry_path = step05_root / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv"
    candidate_path = step08_root / "07_label_shuffle_handoff" / "tier1_label_shuffle_candidates.tsv"
    candidate_summary_path = step08_root / "07_label_shuffle_handoff" / "tier1_label_shuffle_handoff_summary.json"
    curated_table_path = step08_root / "02_curated_treatment_models" / "curated_treatment_model_table.tsv"
    step08_summary_path = step08_root / "v2_step08_curated_per_treatment_residual_models_summary.json"

    pair_df_head = read_table(pair_path)
    registry = read_table(registry_path)
    candidates = read_table(candidate_path)

    if pair_df_head.empty:
        raise FileNotFoundError(f"Pair dataset missing or empty: {pair_path}")
    if registry.empty:
        raise FileNotFoundError(f"Step 05 strict biology registry missing or empty: {registry_path}")
    if candidates.empty:
        raise FileNotFoundError(f"Step 08 Tier 1 label shuffle candidate table missing or empty: {candidate_path}")

    pair_df_head["drug_key"] = pair_df_head["drug_key"].astype(str)
    feature_cols = resolve_feature_cols(pair_df_head, registry)

    candidate_keys = candidates["drug_key"].astype(str).drop_duplicates().tolist()
    requested_workers = int(args.max_workers)
    if requested_workers <= 0:
        auto_cap = max(1, (os.cpu_count() or 2) // 2)
        useful_workers = max(1, min(auto_cap, len(candidate_keys)))
        worker_policy = f"auto_half_cpu_cap_{auto_cap}"
    else:
        useful_workers = max(1, min(requested_workers, len(candidate_keys)))
        worker_policy = "user_requested"

    source_manifest = build_source_manifest({
        "run_root": run_root,
        "dataset_root": dataset_root,
        "step05_root": step05_root,
        "step08_root": step08_root,
        "pair_level_dataset": pair_path,
        "step05_strict_biology_registry": registry_path,
        "tier1_label_shuffle_candidates": candidate_path,
        "tier1_label_shuffle_handoff_summary": candidate_summary_path,
        "curated_treatment_model_table": curated_table_path,
        "step08_summary": step08_summary_path,
    })

    write_table(source_manifest, d01 / "source_manifest.tsv")
    write_table(candidates, d01 / "tier1_label_shuffle_candidates_used.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), d01 / "step05_registry_features_used.tsv")

    tasks = []
    for treatment_idx, drug_key in enumerate(candidate_keys):
        tasks.append({
            "drug_key": drug_key,
            "pair_path": str(pair_path),
            "registry_path": str(registry_path),
            "target_col": args.target_col,
            "n_shuffles": args.n_shuffles,
            "n_repeats": args.n_repeats,
            "test_size": args.test_size,
            "max_features_per_split": args.max_features_per_split,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
            "treatment_idx": treatment_idx,
        })

    print(f"Running Step 09 with {useful_workers} worker(s) across {len(tasks)} Tier 1 treatment(s). Worker policy: {worker_policy}.", flush=True)

    observed_metrics_all = []
    null_metrics_all = []
    observed_feature_all = []
    checkpoint_rows = []

    if useful_workers == 1:
        for task in tasks:
            result = run_one_treatment(task)
            print(f"Completed treatment: {result['drug_key']} rows={result['n_rows']} features={result['n_features']}", flush=True)
            observed_metrics_all.extend(result["observed_metrics"])
            null_metrics_all.extend(result["null_metrics"])
            observed_feature_all.extend(result["observed_feature"])
            checkpoint_rows.append({
                "drug_key": result["drug_key"],
                "n_rows": result["n_rows"],
                "n_features": result["n_features"],
                "status": "complete",
            })
            write_table(pd.DataFrame(checkpoint_rows), d09 / "completed_treatments_checkpoint.tsv")
    else:
        # Parallelism is across treatments; each model still uses n_jobs=1 to avoid nested oversubscription.
        with ProcessPoolExecutor(max_workers=useful_workers) as executor:
            future_map = {executor.submit(run_one_treatment, task): task["drug_key"] for task in tasks}

            for future in as_completed(future_map):
                drug_key = future_map[future]
                try:
                    result = future.result()
                    print(f"Completed treatment: {result['drug_key']} rows={result['n_rows']} features={result['n_features']}", flush=True)
                    observed_metrics_all.extend(result["observed_metrics"])
                    null_metrics_all.extend(result["null_metrics"])
                    observed_feature_all.extend(result["observed_feature"])
                    checkpoint_rows.append({
                        "drug_key": result["drug_key"],
                        "n_rows": result["n_rows"],
                        "n_features": result["n_features"],
                        "status": "complete",
                    })
                except Exception as exc:
                    print(f"Failed treatment: {drug_key}: {exc}", flush=True)
                    checkpoint_rows.append({
                        "drug_key": drug_key,
                        "n_rows": "",
                        "n_features": "",
                        "status": "failed",
                        "error": str(exc),
                    })
                    write_table(pd.DataFrame(checkpoint_rows), d09 / "completed_treatments_checkpoint.tsv")
                    raise

                write_table(pd.DataFrame(checkpoint_rows), d09 / "completed_treatments_checkpoint.tsv")

    observed_metrics = pd.DataFrame(observed_metrics_all)
    null_metrics = pd.DataFrame(null_metrics_all)
    observed_feature_evidence = pd.DataFrame(observed_feature_all)

    observed_summary = summarize_observed(observed_metrics)
    null_by_shuffle, null_summary = summarize_null(null_metrics)

    validation_results = validate_against_null(
        observed_summary=observed_summary,
        null_by_shuffle=null_by_shuffle,
        candidates=candidates,
        fdr_threshold=args.fdr_threshold,
    )

    recurrent_features = summarize_feature_recurrence(
        observed_feature_evidence,
        validation_results,
        registry,
    )

    recurrent_themes = summarize_theme_recurrence(recurrent_features)

    validated = validation_results[validation_results["validated_for_step10"] == True].copy()
    not_validated = validation_results[validation_results["validated_for_step10"] != True].copy()

    validated["step10_validation_status"] = "label_shuffle_validated"
    not_validated["step10_validation_status"] = "not_label_shuffle_validated"

    write_table(observed_metrics, d02 / "tier1_observed_repeated_split_metrics_long.tsv")
    write_table(observed_summary, d02 / "tier1_observed_metric_summary.tsv")
    write_table(observed_feature_evidence, d02 / "tier1_observed_feature_evidence_long.tsv")

    write_table(null_metrics, d03 / "tier1_label_shuffle_null_metrics_long.tsv")
    write_table(null_by_shuffle, d03 / "tier1_label_shuffle_null_metric_by_shuffle.tsv")
    write_table(null_summary, d03 / "tier1_label_shuffle_null_summary.tsv")

    write_table(validation_results, d04 / "tier1_label_shuffle_validation_results.tsv")
    write_table(validated, d04 / "tier1_label_shuffle_validated_treatments.tsv")
    write_table(not_validated, d04 / "tier1_label_shuffle_not_validated_treatments.tsv")

    write_table(recurrent_features, d05 / "label_shuffle_validated_recurrent_spatial_features.tsv")
    write_table(recurrent_themes, d05 / "label_shuffle_validated_recurrent_biology_themes.tsv")

    write_table(validated, d08 / "label_shuffle_validated_treatments_for_step10.tsv")
    write_table(validation_results, d08 / "all_label_shuffle_results_for_step10.tsv")

    # The Step 10 handoff records whether validated treatments are available for integrated interpretation.
    step10_handoff_summary = {
        "status": "ready_for_step10" if len(validated) > 0 else "no_validated_tier1_models",
        "n_tier1_candidates_tested": int(len(candidate_keys)),
        "n_label_shuffle_validated_treatments": int(len(validated)),
        "target_col": args.target_col,
        "feature_set": "v2_step05_strict_biology_registry",
        "source_step": "09_label_shuffle_validate_tier1",
        "fdr_threshold": float(args.fdr_threshold),
    }
    write_json(step10_handoff_summary, d08 / "step10_handoff_summary.json")

    save_bar(
        validation_results,
        "drug_key",
        "observed_test_pearson_mean",
        d06 / "fig_01_tier1_observed_mean_test_pearson.png",
        "V2 Step 09 Tier 1 observed model performance",
        "Observed mean test Pearson",
        top_n=30,
    )

    save_bar(
        validation_results,
        "drug_key",
        "empirical_p_pearson",
        d06 / "fig_02_tier1_label_shuffle_empirical_p_values.png",
        "V2 Step 09 empirical label shuffle p values",
        "Empirical p value",
        top_n=30,
    )

    save_bar(
        validation_results,
        "drug_key",
        "fdr_q_pearson",
        d06 / "fig_03_tier1_label_shuffle_fdr_q_values.png",
        "V2 Step 09 label shuffle FDR q values",
        "FDR q value",
        top_n=30,
    )

    save_bar(
        validation_results,
        "drug_key",
        "pearson_effect_vs_null_median",
        d06 / "fig_04_tier1_observed_minus_null_median_pearson.png",
        "V2 Step 09 observed minus null median Pearson",
        "Observed minus null median Pearson",
        top_n=30,
    )

    save_observed_vs_null_plot(
        validation_results,
        d06 / "fig_05_observed_vs_null_q95_pearson.png",
    )

    if not recurrent_features.empty:
        save_bar(
            recurrent_features,
            "feature_name",
            "mean_gain_importance",
            d06 / "fig_06_label_shuffle_validated_recurrent_spatial_features.png",
            "V2 Step 09 validated recurrent spatial features",
            "Mean gain importance",
            top_n=30,
        )

    if not recurrent_themes.empty:
        save_bar(
            recurrent_themes,
            "biological_theme",
            "total_gain_importance",
            d06 / "fig_07_label_shuffle_validated_recurrent_biology_themes.png",
            "V2 Step 09 validated recurrent biology themes",
            "Total gain importance",
            top_n=15,
        )

    best = validation_results.head(1)
    best_treatment = str(best.iloc[0]["drug_key"]) if not best.empty else ""
    best_p = safe_float(best.iloc[0].get("empirical_p_pearson")) if not best.empty else None
    best_q = safe_float(best.iloc[0].get("fdr_q_pearson")) if not best.empty else None
    best_pearson = safe_float(best.iloc[0].get("observed_test_pearson_mean")) if not best.empty else None

    run_summary = {
        "status": "pass",
        "official_step": "09_label_shuffle_validate_tier1",
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "step05_root": str(step05_root),
        "step08_root": str(step08_root),
        "output_root": str(output_root),
        "target_col": args.target_col,
        "n_tier1_candidates_tested": int(len(candidate_keys)),
        "n_shuffles": int(args.n_shuffles),
        "n_repeats": int(args.n_repeats),
        "max_workers_requested": int(args.max_workers),
        "max_workers_policy": worker_policy,
        "max_workers_used": int(useful_workers),
        "n_step05_registry_features": int(len(registry)),
        "n_features_used": int(len(feature_cols)),
        "n_label_shuffle_validated_treatments": int(len(validated)),
        "n_not_validated_treatments": int(len(not_validated)),
        "best_treatment": best_treatment,
        "best_empirical_p_pearson": best_p,
        "best_fdr_q_pearson": best_q,
        "best_observed_test_pearson_mean": best_pearson,
        "fdr_threshold": float(args.fdr_threshold),
        "production_dependency_on_v1_outputs": "no",
        "uses_step05_v2_registry": "yes",
        "uses_treatment_identity_features": "no",
        "ready_for_step10_integrated_package": "yes" if len(validated) > 0 else "no",
    }

    write_json(run_summary, output_root / "v2_step09_tier1_label_shuffle_validation_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 09 TIER 1 LABEL SHUFFLE VALIDATION REPORT")
    report_lines.append("")

    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("1. Source files")
    report_lines.append(source_manifest.to_string(index=False))

    report_lines.append("")
    report_lines.append("2. Validation design")
    report_lines.append("For each Step 08 Tier 1 candidate, Step 09 retrains observed treatment-specific residual models.")
    report_lines.append("It creates a label shuffle null by permuting fused_residual_vs_prior labels within the same treatment.")
    report_lines.append("Feature selection is repeated inside each train split.")
    report_lines.append("No treatment identity features are used.")
    report_lines.append("The empirical p value is computed as 1 plus the number of null shuffles at least as strong as observed, divided by 1 plus the number of shuffles.")
    report_lines.append("Benjamini-Hochberg FDR is applied across Tier 1 candidates.")

    report_lines.append("")
    report_lines.append("3. Label shuffle validation results")
    if validation_results.empty:
        report_lines.append("No validation results generated.")
    else:
        show_cols = [
            "drug_key",
            "observed_test_pearson_mean",
            "observed_test_r2_mean",
            "observed_rmse_improvement_vs_baseline_mean",
            "null_test_pearson_mean_median",
            "null_test_pearson_mean_q95",
            "empirical_p_pearson",
            "fdr_q_pearson",
            "pearson_effect_vs_null_median",
            "label_shuffle_validation_status",
        ]
        show_cols = [c for c in show_cols if c in validation_results.columns]
        report_lines.append(validation_results[show_cols].to_string(index=False))

    report_lines.append("")
    report_lines.append("4. Validated treatment models for Step 10")
    if validated.empty:
        report_lines.append("No treatments passed label shuffle validation.")
    else:
        show_cols = [
            "drug_key",
            "observed_test_pearson_mean",
            "observed_test_r2_mean",
            "empirical_p_pearson",
            "fdr_q_pearson",
            "pearson_effect_vs_null_median",
            "label_shuffle_validation_status",
        ]
        show_cols = [c for c in show_cols if c in validated.columns]
        report_lines.append(validated[show_cols].to_string(index=False))

    report_lines.append("")
    report_lines.append("5. Recurrent spatial features among validated treatments")
    if recurrent_features.empty:
        report_lines.append("No validated recurrent feature table generated.")
    else:
        report_lines.append(recurrent_features.head(80).to_string(index=False))

    report_lines.append("")
    report_lines.append("6. Recurrent biology themes among validated treatments")
    if recurrent_themes.empty:
        report_lines.append("No validated recurrent theme table generated.")
    else:
        report_lines.append(recurrent_themes.to_string(index=False))

    report_lines.append("")
    report_lines.append("7. Interpretation")
    report_lines.append("Step 09 converts Tier 1 screening signals into label-shuffle-validated treatment-specific spatial biology claims.")
    report_lines.append("Passing Step 09 means observed treatment-specific residual prediction was stronger than the shuffled-label null after FDR correction.")
    report_lines.append("These validated treatments are the handoff to Step 10 integrated interpretation packaging.")

    report_path = write_text_report(d07 / "v2_step09_tier1_label_shuffle_validation_report.txt", "\n".join(report_lines))

    slide_lines = []
    slide_lines.append("V2 STEP 09 TIER 1 LABEL SHUFFLE VALIDATION SLIDE NOTES")
    slide_lines.append("")
    slide_lines.append(f"Tier 1 candidates tested: {len(candidate_keys)}")
    slide_lines.append(f"Label shuffles per treatment: {args.n_shuffles}")
    slide_lines.append(f"Repeated splits per observed or shuffled model: {args.n_repeats}")
    slide_lines.append(f"Workers used: {useful_workers}")
    slide_lines.append(f"Validated treatments: {len(validated)}")
    slide_lines.append(f"FDR threshold: {args.fdr_threshold}")
    slide_lines.append(f"Best treatment: {best_treatment}")
    slide_lines.append(f"Best observed mean test Pearson: {best_pearson}")
    slide_lines.append(f"Best empirical p value: {best_p}")
    slide_lines.append(f"Best FDR q value: {best_q}")
    slide_lines.append("")
    slide_lines.append("Validated recurrent biology themes:")
    if recurrent_themes.empty:
        slide_lines.append("No validated recurrent themes.")
    else:
        for theme in recurrent_themes.head(8)["biological_theme"].astype(str).tolist():
            slide_lines.append(theme)
    slide_lines.append("")
    slide_lines.append("Caveat: this is a smoke label-shuffle validation. The full run should increase shuffles for finer p value resolution.")

    write_text_report(d07 / "v2_step09_tier1_label_shuffle_validation_slide_notes.txt", "\n".join(slide_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Dataset root: {dataset_root}",
        f"Step 05 root: {step05_root}",
        f"Step 08 root: {step08_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Tier 1 candidates tested: {len(candidate_keys)}",
        f"Label shuffles per treatment: {args.n_shuffles}",
        f"Repeated splits per model: {args.n_repeats}",
        f"Workers used: {useful_workers}",
        f"Worker policy: {worker_policy}",
        f"Step 05 registry features: {len(registry)}",
        f"Features used: {len(feature_cols)}",
        f"Validated treatments: {len(validated)}",
        f"Not validated treatments: {len(not_validated)}",
        f"Best treatment: {best_treatment}",
        f"Best observed mean test Pearson: {best_pearson}",
        f"Best empirical p value: {best_p}",
        f"Best FDR q value: {best_q}",
        "Treatment identity features used: no",
        "Production dependency on V1 outputs: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 09 TIER 1 LABEL SHUFFLE VALIDATION COMPLETE", terminal_lines))
    print("")

    print("Label shuffle validation results")
    show_cols = [
        "drug_key",
        "observed_test_pearson_mean",
        "observed_test_r2_mean",
        "null_test_pearson_mean_median",
        "null_test_pearson_mean_q95",
        "empirical_p_pearson",
        "fdr_q_pearson",
        "pearson_effect_vs_null_median",
        "label_shuffle_validation_status",
    ]
    show_cols = [c for c in show_cols if c in validation_results.columns]
    print(validation_results[show_cols].to_string(index=False))
    print("")

    if not recurrent_features.empty:
        print("Validated recurrent spatial features")
        show_cols = [
            "feature_name",
            "feature_original",
            "biological_theme",
            "validated_treatment_count",
            "mean_gain_importance",
            "max_gain_importance",
            "example_validated_treatments",
        ]
        show_cols = [c for c in show_cols if c in recurrent_features.columns]
        print(recurrent_features[show_cols].head(30).to_string(index=False))
        print("")

    if not recurrent_themes.empty:
        print("Validated recurrent biology themes")
        print(recurrent_themes.head(15).to_string(index=False))
        print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
