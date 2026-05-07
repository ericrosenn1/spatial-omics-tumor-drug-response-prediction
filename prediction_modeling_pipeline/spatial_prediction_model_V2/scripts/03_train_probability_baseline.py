"""
Script: 03_train_probability_baseline.py

Purpose:
    Train the Step 03 probability-baseline model for spatial_prediction_model_V2.

Pipeline role:
    This script consumes the governed modeling dataset built in Step 02 and fits
    repeated grouped train/test baselines for the selected target column. It is a
    diagnostic baseline rather than the final biological residual model: it tests
    whether spatial candidate features and treatment-identity controls can predict
    the governed teacher response signal under sample-grouped splits.

Scientific role:
    The probability baseline separates two questions that matter for downstream
    interpretation. First, it measures whether the current broad spatial feature
    pool carries predictive signal. Second, it reports how much model importance
    is attributed to spatial features versus treatment-identity dummy variables,
    helping distinguish spatial morphology signal from treatment-prior or drug
    identity effects.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP03_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, imports, constants,
    hyperparameters, split logic, feature-selection rules, output filenames, and
    return codes must remain unchanged.
"""


# =============================================================================
# Imports and local package setup
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
# Use a non-interactive backend so figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent
SRC_ROOT = V2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spm_v2.io_utils import ensure_dir, read_table, write_json, write_table, write_text_report
from spm_v2.model_training import make_xgb_pipeline, select_features_training_only
from spm_v2.provenance import write_run_provenance
from spm_v2.reporting import terminal_block, write_output_manifest
from spm_v2.validation import metric_safe, summarize_repeated_split_metrics


# =============================================================================
# Helper functions
# =============================================================================

def subset_pair_dataset(df: pd.DataFrame, max_samples: int, max_treatments: int, random_state: int) -> pd.DataFrame:
    """Return a smoke-test subset by limiting samples and high-frequency treatments."""

    out = df.copy()
    rng = np.random.default_rng(random_state)

    out["sample_id"] = out["sample_id"].astype(str)
    out["drug_key"] = out["drug_key"].astype(str)

    if max_samples and max_samples > 0:
        samples = np.array(sorted(out["sample_id"].dropna().unique()))
        if len(samples) > max_samples:
            samples = rng.choice(samples, size=max_samples, replace=False)
        out = out[out["sample_id"].isin(samples)].copy()

    if max_treatments and max_treatments > 0:
        keep = out["drug_key"].value_counts().head(max_treatments).index.astype(str).tolist()
        out = out[out["drug_key"].isin(keep)].copy()

    return out.reset_index(drop=True)


def add_treatment_dummies(df: pd.DataFrame, max_treatment_dummies: int) -> tuple[pd.DataFrame, list[str]]:
    """Add treatment-identity dummy variables used as baseline control features."""

    out = df.copy()
    counts = out["drug_key"].astype(str).value_counts()
    keep = counts.head(max_treatment_dummies).index.astype(str).tolist()
    label = out["drug_key"].astype(str).where(out["drug_key"].astype(str).isin(keep), "__other_treatment__")
    # Treatment dummies estimate how much predictive signal can be explained by drug identity alone.
    dummies = pd.get_dummies(label, prefix="treatment_identity", dtype=float)
    out = pd.concat([out, dummies], axis=1)
    return out, dummies.columns.astype(str).tolist()


def grouped_sample_split(df: pd.DataFrame, test_size: float, random_state: int):
    # Group by sample_id so rows from the same sample do not appear in both train and test sets.
    """Create one train/test split grouped by sample ID to avoid sample leakage."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["sample_id"].astype(str).values
    return next(splitter.split(df, groups=groups))


def summarize_feature_evidence(rows: list[dict]) -> pd.DataFrame:
    """Aggregate repeated-split feature selection and gain-importance evidence."""

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["feature_name", "feature_type"], dropna=False)
        .agg(
            selection_count=("selected", "sum"),
            mean_gain_importance=("gain_importance", "mean"),
            max_gain_importance=("gain_importance", "max"),
        )
        .reset_index()
    )

    repeats = max(int(df["repeat"].nunique()), 1)
    summary["selection_frequency"] = summary["selection_count"] / repeats
    summary = summary.sort_values(["mean_gain_importance", "selection_frequency"], ascending=False)
    return summary


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, xlabel: str, top_n: int = 30) -> None:
    """Write a horizontal bar plot for ranked feature evidence."""

    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    plot = df.copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)

    if plot.empty:
        return

    labels = [str(x) if len(str(x)) <= 70 else str(x)[:67] + "..." for x in plot[label_col].tolist()]
    y = np.arange(len(plot))[::-1]

    plt.figure(figsize=(11, max(6, len(plot) * 0.32)))
    plt.barh(y, plot[value_col].to_numpy()[::-1])
    plt.yticks(y, labels[::-1], fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Train repeated probability-baseline models and write metrics, figures, and reports."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--target-col", default="fused_prob_responder")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-treatments", type=int, default=160)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-features-per-split", type=int, default=120)
    parser.add_argument("--max-treatment-dummies", type=int, default=250)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_metrics")
    d03 = ensure_dir(output_root / "03_feature_evidence")
    d04 = ensure_dir(output_root / "04_figures")
    d05 = ensure_dir(output_root / "05_reports")

    pair_path = dataset_root / "03_modeling_datasets" / "v2_pair_level_residual_dataset_broad_governed_candidate_pool.tsv"
    feature_path = dataset_root / "02_feature_governance" / "v2_broad_governed_candidate_features.tsv"

    pair_df = read_table(pair_path)
    feature_meta = read_table(feature_path)

    if pair_df.empty:
        raise FileNotFoundError(f"Pair dataset missing or empty: {pair_path}")

    if feature_meta.empty:
        raise FileNotFoundError(f"Broad feature table missing or empty: {feature_path}")

    if args.target_col not in pair_df.columns:
        raise ValueError(f"Required target column missing: {args.target_col}")

    if args.mode == "smoke":
        model_df = subset_pair_dataset(pair_df, args.max_samples, args.max_treatments, args.random_state)
    else:
        model_df = pair_df.copy()

    broad_cols = [str(x) for x in feature_meta["feature_name"].tolist() if str(x) in model_df.columns]

    if len(broad_cols) < 10:
        raise ValueError(f"Too few broad spatial candidate features: {len(broad_cols)}")

    model_df, treatment_cols = add_treatment_dummies(model_df, args.max_treatment_dummies)
    feature_cols = broad_cols + treatment_cols

    for col in broad_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    y_all = pd.to_numeric(model_df[args.target_col], errors="coerce").astype(float)
    x_all = model_df[feature_cols].copy()

    metrics_rows = []
    prediction_rows = []
    evidence_rows = []

    for repeat in range(args.n_repeats):
        seed = args.random_state + repeat
        train_idx, test_idx = grouped_sample_split(model_df, args.test_size, seed)

        # Feature selection is fit on the training split only to avoid test-set leakage.
        selected = select_features_training_only(
            x_train=x_all.iloc[train_idx],
            y_train=y_all.iloc[train_idx],
            feature_cols=feature_cols,
            max_features=args.max_features_per_split,
            min_variance=1e-12,
        )

        pipe = make_xgb_pipeline(
            random_state=seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            tree_method="hist",
            n_jobs=1,
        )

        pipe.fit(x_all.iloc[train_idx][selected], y_all.iloc[train_idx])
        pred = pipe.predict(x_all.iloc[test_idx][selected])
        baseline = np.repeat(float(y_all.iloc[train_idx].mean()), len(test_idx))

        test_m = metric_safe(y_all.iloc[test_idx], pred)
        base_m = metric_safe(y_all.iloc[test_idx], baseline)

        metrics_rows.append({
            "model_family": "probability_baseline",
            "target_col": args.target_col,
            "repeat": repeat,
            "random_state": seed,
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

        pred_df = pd.DataFrame({
            "repeat": repeat,
            "sample_id": model_df.iloc[test_idx]["sample_id"].astype(str).values,
            "drug_key": model_df.iloc[test_idx]["drug_key"].astype(str).values,
            "target": y_all.iloc[test_idx].values,
            "prediction": pred,
        })
        prediction_rows.append(pred_df)

        importances = pipe.named_steps["model"].feature_importances_
        for feature, importance in zip(selected, importances):
            evidence_rows.append({
                "repeat": repeat,
                "feature_name": feature,
                "feature_type": "treatment_identity" if feature.startswith("treatment_identity_") else "spatial_candidate",
                "gain_importance": float(importance),
                "selected": True,
            })

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    feature_evidence = summarize_feature_evidence(evidence_rows)

    metric_summary = summarize_repeated_split_metrics(metrics, ["model_family", "target_col"])

    total_gain = float(feature_evidence["mean_gain_importance"].sum()) if not feature_evidence.empty else 0.0
    spatial_gain = float(feature_evidence.loc[feature_evidence["feature_type"].eq("spatial_candidate"), "mean_gain_importance"].sum()) if not feature_evidence.empty else 0.0
    treatment_gain = float(feature_evidence.loc[feature_evidence["feature_type"].eq("treatment_identity"), "mean_gain_importance"].sum()) if not feature_evidence.empty else 0.0

    # Report spatial-vs-treatment contribution so downstream interpretation is not based on accuracy alone.
    contribution = pd.DataFrame([{
        "model_family": "probability_baseline",
        "spatial_gain_total": spatial_gain,
        "treatment_identity_gain_total": treatment_gain,
        "spatial_feature_fraction": spatial_gain / total_gain if total_gain > 0 else np.nan,
        "treatment_identity_fraction": treatment_gain / total_gain if total_gain > 0 else np.nan,
    }])

    write_table(pd.DataFrame({
        "source_name": ["pair_dataset", "broad_feature_manifest"],
        "path": [str(pair_path), str(feature_path)],
        "exists": [pair_path.exists(), feature_path.exists()],
    }), d01 / "source_manifest.tsv")

    write_table(metrics, d02 / "probability_baseline_metrics_long.tsv")
    write_table(metric_summary, d02 / "probability_baseline_metric_summary.tsv")
    write_table(predictions, d02 / "probability_baseline_test_predictions.tsv")
    write_table(feature_evidence, d03 / "probability_baseline_feature_evidence_summary.tsv")
    write_table(contribution, d03 / "probability_baseline_spatial_vs_treatment_contribution.tsv")

    save_bar(
        feature_evidence,
        "feature_name",
        "mean_gain_importance",
        d04 / "fig_01_probability_baseline_top_features.png",
        "V2 step 03 probability baseline feature evidence",
        "Mean XGBoost gain importance",
        top_n=30,
    )

    summary_row = metric_summary.iloc[0].to_dict() if not metric_summary.empty else {}

    run_summary = {
        "status": "pass",
        "official_step": "03_train_probability_baseline",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "target_col": args.target_col,
        "mode": args.mode,
        "n_rows_used": int(len(model_df)),
        "n_samples_used": int(model_df["sample_id"].astype(str).nunique()),
        "n_treatments_used": int(model_df["drug_key"].astype(str).nunique()),
        "n_broad_spatial_features": int(len(broad_cols)),
        "n_treatment_identity_features": int(len(treatment_cols)),
        "n_repeats": int(args.n_repeats),
        "production_dependency_on_v1_outputs": "no",
        "spatial_feature_fraction": float(contribution.iloc[0]["spatial_feature_fraction"]) if not contribution.empty else np.nan,
        **{k: float(v) if isinstance(v, (np.floating, float)) and pd.notna(v) else v for k, v in summary_row.items() if k.startswith("test_") or k.startswith("rmse_")}
    }

    write_json(run_summary, output_root / "v2_step03_probability_baseline_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 03 PROBABILITY BASELINE REPORT")
    report_lines.append("")
    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    report_lines.append("Spatial versus treatment identity contribution")
    report_lines.append(contribution.to_string(index=False))
    report_lines.append("")
    report_lines.append("Top feature evidence")
    report_lines.append(feature_evidence.head(40).to_string(index=False))

    report_path = write_text_report(d05 / "v2_step03_probability_baseline_report.txt", "\n".join(report_lines))
    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Dataset root: {dataset_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Rows used: {len(model_df)}",
        f"Samples used: {model_df['sample_id'].astype(str).nunique()}",
        f"Treatments used: {model_df['drug_key'].astype(str).nunique()}",
        f"Broad spatial features: {len(broad_cols)}",
        f"Treatment identity features: {len(treatment_cols)}",
        f"Mean test Pearson: {run_summary.get('test_pearson_mean', '')}",
        f"Mean test R2: {run_summary.get('test_r2_mean', '')}",
        f"Spatial feature fraction: {run_summary.get('spatial_feature_fraction', '')}",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 03 PROBABILITY BASELINE COMPLETE", terminal_lines))
    print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
