"""
Script: 04_train_pair_level_residual_model.py

Purpose:
    Train the Step 04 pair-level residual model for spatial_prediction_model_V2.

Pipeline role:
    This step consumes the Step 02 pair-level residual dataset and trains a
    pair-level residual model that evaluates sample-treatment rows. Its outputs
    provide residual prediction metrics and feature evidence used by the Step 05
    residual-biology registry.

Scientific role:
    The pair-level residual model tests whether spatial features explain
    response deviation after treatment-prior governance. It is an evidence-
    generation step for curating spatial biology features, not a final
    treatment-specific claim by itself.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP04_DOC_POLISH_V2

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    imports, constants, thresholds, hyperparameters, feature-selection
    rules, output filenames, and return codes must remain unchanged.
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
    """Helper used by this spatial_prediction_model_V2 workflow step."""

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
    """Build or summarize treatment-level residual modeling evidence."""

    out = df.copy()
    counts = out["drug_key"].astype(str).value_counts()
    keep = counts.head(max_treatment_dummies).index.astype(str).tolist()
    label = out["drug_key"].astype(str).where(out["drug_key"].astype(str).isin(keep), "__other_treatment__")
    dummies = pd.get_dummies(label, prefix="treatment_identity", dtype=float)
    out = pd.concat([out, dummies], axis=1)
    return out, dummies.columns.astype(str).tolist()


def grouped_sample_split(df: pd.DataFrame, test_size: float, random_state: int):
    """Create train/test split indices for model evaluation."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["sample_id"].astype(str).values
    return next(splitter.split(df, groups=groups))


def run_shap_if_requested(pipe, x_eval: pd.DataFrame, selected: list[str], max_rows: int, seed: int) -> pd.DataFrame:
    """Helper used by this spatial_prediction_model_V2 workflow step."""

    try:
        import shap
    except Exception as exc:
        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": f"not_available: {exc}",
        })

    try:
        rng = np.random.default_rng(seed)
        if len(x_eval) > max_rows:
            idx = rng.choice(np.arange(len(x_eval)), size=max_rows, replace=False)
            x_eval = x_eval.iloc[idx].copy()

        x_imp = pd.DataFrame(
            pipe.named_steps["imputer"].transform(x_eval[selected]),
            columns=selected,
        )

        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        values = explainer.shap_values(x_imp)

        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "shap_status": "success",
        })
    except Exception as exc:
        return pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": f"failed: {exc}",
        })


def summarize_feature_evidence(rows: list[dict]) -> pd.DataFrame:
    """Aggregate feature evidence across residual models."""

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["feature_name", "feature_type"], dropna=False)
        .agg(
            selection_count=("selected", "sum"),
            mean_gain_importance=("gain_importance", "mean"),
            max_gain_importance=("gain_importance", "max"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            shap_success_count=("shap_status", lambda s: int((s.astype(str) == "success").sum())),
        )
        .reset_index()
    )

    repeats = max(int(df["repeat"].nunique()), 1)
    summary["selection_frequency"] = summary["selection_count"] / repeats
    summary = summary.sort_values(["mean_abs_shap", "mean_gain_importance", "selection_frequency"], ascending=False)
    return summary


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, xlabel: str, top_n: int = 30) -> None:
    """Save a ranked horizontal bar plot for model, feature, or theme evidence."""

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
    """Run this spatial_prediction_model_V2 step and write tables, figures, reports, and provenance."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--target-col", default="fused_residual_vs_prior")
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
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--max-shap-rows", type=int, default=2500)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_metrics")
    d03 = ensure_dir(output_root / "03_feature_evidence_for_step05")
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

        # Feature selection is fit on training data only to avoid test-set leakage.
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
            "model_family": "pair_level_prior_adjusted_residual_model",
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

        importances = pd.DataFrame({
            "feature_name": selected,
            "gain_importance": pipe.named_steps["model"].feature_importances_,
        })

        if args.run_shap:
            shap_df = run_shap_if_requested(pipe, x_all.iloc[test_idx][selected], selected, args.max_shap_rows, seed)
        else:
            shap_df = pd.DataFrame({
                "feature_name": selected,
                "mean_abs_shap": np.nan,
                "shap_status": "not_run",
            })

        evidence = importances.merge(shap_df, on="feature_name", how="left")

        for _, row in evidence.iterrows():
            feature = str(row["feature_name"])
            evidence_rows.append({
                "repeat": repeat,
                "feature_name": feature,
                "feature_type": "treatment_identity" if feature.startswith("treatment_identity_") else "spatial_candidate",
                "gain_importance": float(row["gain_importance"]) if pd.notna(row["gain_importance"]) else np.nan,
                "mean_abs_shap": float(row["mean_abs_shap"]) if pd.notna(row["mean_abs_shap"]) else np.nan,
                "shap_status": str(row.get("shap_status", "")),
                "selected": True,
            })

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    evidence_long = pd.DataFrame(evidence_rows)
    feature_evidence = summarize_feature_evidence(evidence_rows)

    metric_summary = summarize_repeated_split_metrics(metrics, ["model_family", "target_col"])

    score_col = "mean_abs_shap" if feature_evidence["mean_abs_shap"].notna().sum() > 0 else "mean_gain_importance"
    total_score = float(pd.to_numeric(feature_evidence[score_col], errors="coerce").fillna(0.0).sum()) if not feature_evidence.empty else 0.0
    spatial_score = float(pd.to_numeric(feature_evidence.loc[feature_evidence["feature_type"].eq("spatial_candidate"), score_col], errors="coerce").fillna(0.0).sum()) if not feature_evidence.empty else 0.0
    treatment_score = float(pd.to_numeric(feature_evidence.loc[feature_evidence["feature_type"].eq("treatment_identity"), score_col], errors="coerce").fillna(0.0).sum()) if not feature_evidence.empty else 0.0

    # Report spatial-vs-control contribution so interpretation is not based on accuracy alone.
    contribution = pd.DataFrame([{
        "model_family": "pair_level_prior_adjusted_residual_model",
        "score_col": score_col,
        "spatial_score_total": spatial_score,
        "treatment_identity_score_total": treatment_score,
        "spatial_feature_fraction": spatial_score / total_score if total_score > 0 else np.nan,
        "treatment_identity_fraction": treatment_score / total_score if total_score > 0 else np.nan,
    }])

    spatial_evidence_for_step05 = feature_evidence[feature_evidence["feature_type"].eq("spatial_candidate")].copy()
    spatial_evidence_for_step05 = spatial_evidence_for_step05.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_feature_manifest"))

    write_table(pd.DataFrame({
        "source_name": ["pair_dataset", "broad_feature_manifest"],
        "path": [str(pair_path), str(feature_path)],
        "exists": [pair_path.exists(), feature_path.exists()],
    }), d01 / "source_manifest.tsv")

    write_table(metrics, d02 / "pair_level_residual_metrics_long.tsv")
    write_table(metric_summary, d02 / "pair_level_residual_metric_summary.tsv")
    write_table(predictions, d02 / "pair_level_residual_test_predictions.tsv")
    write_table(evidence_long, d03 / "pair_level_residual_feature_evidence_long.tsv")
    write_table(feature_evidence, d03 / "pair_level_residual_feature_evidence_summary.tsv")
    write_table(spatial_evidence_for_step05, d03 / "spatial_feature_evidence_for_step05.tsv")
    write_table(contribution, d03 / "pair_level_residual_spatial_vs_treatment_contribution.tsv")

    save_bar(
        spatial_evidence_for_step05,
        "feature_name",
        score_col,
        d04 / "fig_01_pair_level_residual_top_spatial_features_for_step05.png",
        "V2 step 04 residual spatial feature evidence for step 05",
        score_col,
        top_n=30,
    )

    save_bar(
        feature_evidence,
        "feature_name",
        score_col,
        d04 / "fig_02_pair_level_residual_top_all_features.png",
        "V2 step 04 residual model top all features",
        score_col,
        top_n=30,
    )

    summary_row = metric_summary.iloc[0].to_dict() if not metric_summary.empty else {}

    run_summary = {
        "status": "pass",
        "official_step": "04_train_pair_level_residual_model",
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
        "feature_evidence_for_step05_generated": "yes",
        "strict_biology_registry_generated_by_step04": "no",
        "strict_biology_registry_generation_step": "05_build_residual_biology_registry.py",
        "production_dependency_on_v1_outputs": "no",
        "spatial_feature_fraction": float(contribution.iloc[0]["spatial_feature_fraction"]) if not contribution.empty else np.nan,
        **{k: float(v) if isinstance(v, (np.floating, float)) and pd.notna(v) else v for k, v in summary_row.items() if k.startswith("test_") or k.startswith("rmse_")}
    }

    write_json(run_summary, output_root / "v2_step04_pair_level_residual_model_summary.json")
    # Provenance records connect this model run back to the V2 code and input handoff.
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 04 PAIR LEVEL RESIDUAL MODEL REPORT")
    report_lines.append("")
    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    report_lines.append("Interpretation")
    report_lines.append("Step 04 generates residual pair model evidence only.")
    report_lines.append("Step 04 does not generate the official strict biology registry.")
    report_lines.append("Step 05 will classify spatial_feature_evidence_for_step05.tsv into strict biology, caution, and excluded feature sets.")
    report_lines.append("")
    report_lines.append("Spatial versus treatment identity contribution")
    report_lines.append(contribution.to_string(index=False))
    report_lines.append("")
    report_lines.append("Top spatial feature evidence for step 05")
    show_cols = [c for c in ["feature_name", "feature_original", "feature_group", "feature_axis", "selection_frequency", "mean_abs_shap", "mean_gain_importance"] if c in spatial_evidence_for_step05.columns]
    report_lines.append(spatial_evidence_for_step05[show_cols].head(50).to_string(index=False))

    report_path = write_text_report(d05 / "v2_step04_pair_level_residual_model_report.txt", "\n".join(report_lines))
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
        "Strict registry generated by step 04: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 04 PAIR LEVEL RESIDUAL MODEL COMPLETE", terminal_lines))
    print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
