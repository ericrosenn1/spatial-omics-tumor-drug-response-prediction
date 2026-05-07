"""
Script: 06_train_broad_residual_model.py

Purpose:
    Train the Step 06 broad residual spatial model.

Pipeline role:
    This step uses the Step 05 strict biology registry and the Step 02 broad
    residual dataset to screen sample-level residual endpoints with spatial-only
    features.

Scientific role:
    The broad residual model tests whether curated spatial-biology features
    explain response deviation above or below treatment priors without using
    treatment-identity dummy variables. Its outputs identify candidate spatial
    phenotypes and biological themes for later treatment-specific analysis.

Documentation polish marker:
    SPATIAL_PREDICTION_MODEL_V2_STEP06_DOC_POLISH_V2

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
# Use a non-interactive backend so broad-model figures can be generated from batch/PowerShell runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
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
from spm_v2.validation import metric_safe


TARGET_CANDIDATES = [
    "mean_residual",
    "broad_resistance_score",
    "strong_negative_fraction_m005",
    "residual_iqr",
    "median_residual",
    "residual_std",
    "positive_residual_fraction",
    "strong_positive_fraction_005",
    "top5_mean_residual",
    "top10_mean_residual",
    "bottom5_mean_residual",
    "bottom10_mean_residual",
]


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


def split_indices(n: int, test_size: float, random_state: int):
    """Create train/test index splits for residual-model evaluation."""

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    return train_idx, test_idx


def discover_targets(broad_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Discover residual target columns available for broad model screening."""

    known = [c for c in TARGET_CANDIDATES if c in broad_df.columns]

    if known:
        return known

    excluded = set(["sample_id", "drug_key", "drug", "treatment", "treatment_name"])
    excluded.update(feature_cols)

    candidates = []

    for col in broad_df.columns:
        if col in excluded:
            continue

        vals = pd.to_numeric(broad_df[col], errors="coerce")
        if vals.notna().sum() >= 20 and vals.nunique(dropna=True) >= 5:
            candidates.append(col)

    return candidates


def run_one_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    n_repeats: int,
    test_size: float,
    max_features_per_split: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    run_shap: bool,
    max_shap_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train repeated broad residual models for one target column."""

    try:
        import shap
        shap_available = True
    except Exception:
        shap = None
        shap_available = False

    y_all = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y_all.notna()

    work = df.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask].astype(float).reset_index(drop=True)
    work = work.reset_index(drop=True)

    x_all = work[feature_cols].copy()

    for col in feature_cols:
        x_all[col] = pd.to_numeric(x_all[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    metric_rows = []
    prediction_rows = []
    evidence_rows = []

    for repeat in range(n_repeats):
        seed = random_state + repeat
        train_idx, test_idx = split_indices(len(work), test_size, seed)

        selected = select_features_training_only(
            x_train=x_all.iloc[train_idx],
            y_train=y_all.iloc[train_idx],
            feature_cols=feature_cols,
            max_features=max_features_per_split,
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

        pipe.fit(x_all.iloc[train_idx][selected], y_all.iloc[train_idx])

        pred_test = pipe.predict(x_all.iloc[test_idx][selected])
        baseline_test = np.repeat(float(y_all.iloc[train_idx].mean()), len(test_idx))

        test_m = metric_safe(y_all.iloc[test_idx], pred_test)
        base_m = metric_safe(y_all.iloc[test_idx], baseline_test)

        metric_rows.append({
            "target_col": target_col,
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
            "target_col": target_col,
            "repeat": repeat,
            "sample_id": work.iloc[test_idx]["sample_id"].astype(str).values if "sample_id" in work.columns else test_idx,
            "target": y_all.iloc[test_idx].values,
            "prediction": pred_test,
        })
        pred_df["prediction_error"] = pred_df["prediction"] - pred_df["target"]
        prediction_rows.append(pred_df)

        gain_df = pd.DataFrame({
            "target_col": target_col,
            "repeat": repeat,
            "feature_name": selected,
            "gain_importance": pipe.named_steps["model"].feature_importances_,
        })

        shap_df = pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": "not_run",
        })

        if run_shap and shap_available:
            try:
                x_eval = x_all.iloc[test_idx][selected].copy()

                if len(x_eval) > max_shap_rows:
                    rng = np.random.default_rng(seed)
                    keep = rng.choice(np.arange(len(x_eval)), size=max_shap_rows, replace=False)
                    x_eval = x_eval.iloc[keep].copy()

                x_imp = pd.DataFrame(
                    pipe.named_steps["imputer"].transform(x_eval),
                    columns=selected,
                )

                explainer = shap.TreeExplainer(pipe.named_steps["model"])
                values = explainer.shap_values(x_imp)

                shap_df = pd.DataFrame({
                    "feature_name": selected,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                    "shap_status": "success",
                })
            except Exception as exc:
                shap_df = pd.DataFrame({
                    "feature_name": selected,
                    "mean_abs_shap": np.nan,
                    "shap_status": "failed: " + str(exc),
                })

        evidence = gain_df.merge(shap_df, on="feature_name", how="left")
        evidence_rows.append(evidence)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    evidence = pd.concat(evidence_rows, ignore_index=True) if evidence_rows else pd.DataFrame()

    return metrics, predictions, evidence


def summarize_targets(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated-split metrics for each broad residual target."""

    rows = []

    for target, sub in metrics.groupby("target_col", dropna=False):
        row = {
            "target_col": target,
            "n_repeats": int(len(sub)),
            "n_features_available": int(sub["n_features_available"].median()) if "n_features_available" in sub.columns else 0,
            "n_features_selected_median": int(sub["n_features_selected"].median()) if "n_features_selected" in sub.columns else 0,
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
            vals = pd.to_numeric(sub[col], errors="coerce") if col in sub.columns else pd.Series(dtype=float)
            row[f"{col}_mean"] = safe_float(vals.mean()) if len(vals.dropna()) else None
            row[f"{col}_median"] = safe_float(vals.median()) if len(vals.dropna()) else None
            row[f"{col}_std"] = safe_float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else None
            row[f"{col}_q025"] = safe_float(vals.quantile(0.025)) if len(vals.dropna()) else None
            row[f"{col}_q975"] = safe_float(vals.quantile(0.975)) if len(vals.dropna()) else None

        vals = pd.to_numeric(sub["test_pearson"], errors="coerce")
        row["test_pearson_positive_fraction"] = safe_float((vals > 0).mean()) if len(vals.dropna()) else None

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["test_pearson_mean", "test_r2_mean"], ascending=False)

    return out


def summarize_feature_evidence(evidence: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature evidence across residual models."""

    if evidence.empty:
        return pd.DataFrame()

    ev = evidence.copy()

    ev["gain_importance"] = pd.to_numeric(ev["gain_importance"], errors="coerce").fillna(0.0)
    ev["mean_abs_shap"] = pd.to_numeric(ev["mean_abs_shap"], errors="coerce")

    grouped = (
        ev.groupby("feature_name", dropna=False)
        .agg(
            target_count=("target_col", lambda s: int(s.nunique())),
            selection_count=("feature_name", "count"),
            mean_gain_importance=("gain_importance", "mean"),
            max_gain_importance=("gain_importance", "max"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            max_abs_shap=("mean_abs_shap", "max"),
            shap_success_count=("shap_status", lambda s: int((s.astype(str) == "success").sum())),
        )
        .reset_index()
    )

    keep_cols = ["feature_name"]

    for col in [
        "feature_original",
        "feature_group",
        "feature_axis",
        "biological_theme",
        "interpretation_class",
        "interpretation_note",
    ]:
        if col in registry.columns:
            keep_cols.append(col)

    reg = registry[keep_cols].drop_duplicates("feature_name")
    grouped = grouped.merge(reg, on="feature_name", how="left")

    score_col = "mean_abs_shap" if grouped["mean_abs_shap"].notna().sum() > 0 else "mean_gain_importance"

    grouped[score_col] = pd.to_numeric(grouped[score_col], errors="coerce").fillna(0.0)
    grouped = grouped.sort_values([score_col, "target_count", "selection_count"], ascending=False)

    return grouped


def summarize_theme_evidence(feature_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate residual-model feature evidence by biological theme."""

    if feature_summary.empty or "biological_theme" not in feature_summary.columns:
        return pd.DataFrame()

    score_col = "mean_abs_shap" if "mean_abs_shap" in feature_summary.columns and feature_summary["mean_abs_shap"].notna().sum() > 0 else "mean_gain_importance"

    out = (
        feature_summary
        .groupby("biological_theme", dropna=False)
        .agg(
            n_features=("feature_name", "count"),
            total_score=(score_col, "sum"),
            max_score=(score_col, "max"),
            mean_score=(score_col, "mean"),
            total_target_count=("target_count", "sum"),
        )
        .reset_index()
        .sort_values("total_score", ascending=False)
    )

    examples = []

    for theme in out["biological_theme"].astype(str).tolist():
        sub = feature_summary[feature_summary["biological_theme"].astype(str) == theme].copy()
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0)
        sub = sub.sort_values(score_col, ascending=False)
        names = sub.head(3).get("feature_original", sub["feature_name"]).astype(str).tolist()
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
    labels = [x if len(x) <= 72 else x[:69] + "..." for x in labels]

    y = np.arange(len(plot))[::-1]
    values = plot[value_col].to_numpy()[::-1]
    labels = labels[::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(6, len(plot) * 0.33)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    """Run this spatial_prediction_model_V2 step and write tables, figures, reports, and provenance."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--step05-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-features-per-split", type=int, default=80)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--max-shap-rows", type=int, default=1000)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    step05_root = Path(args.step05_root)
    output_root = ensure_dir(args.output_root)

    d01 = ensure_dir(output_root / "01_inputs")
    d02 = ensure_dir(output_root / "02_model_metrics")
    d03 = ensure_dir(output_root / "03_feature_evidence")
    d04 = ensure_dir(output_root / "04_figures")
    d05 = ensure_dir(output_root / "05_reports")

    broad_path = dataset_root / "03_modeling_datasets" / "v2_broad_residual_dataset_broad_governed_candidate_pool.tsv"
    registry_path = step05_root / "03_v2_strict_biology_registry" / "v2_strict_biology_feature_registry.tsv"
    theme_path = step05_root / "04_theme_summary" / "v2_residual_biology_theme_summary.tsv"
    step05_summary_path = step05_root / "v2_step05_residual_biology_registry_summary.json"

    broad_df = read_table(broad_path)
    registry = read_table(registry_path)
    step05_theme_summary = read_table(theme_path)

    if broad_df.empty:
        raise FileNotFoundError(f"Broad residual dataset missing or empty: {broad_path}")

    if registry.empty:
        raise FileNotFoundError(f"Step 05 strict biology registry missing or empty: {registry_path}")

    if "feature_name" not in registry.columns:
        raise ValueError("Step 05 registry must contain feature_name.")

    feature_cols = [str(x) for x in registry["feature_name"].astype(str).tolist() if str(x) in broad_df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    if len(feature_cols) < 10:
        raise ValueError(f"Too few Step 05 registry features found in broad dataset: {len(feature_cols)}")

    # Discover broad residual targets from available Step 02 summaries rather than hard-coding one endpoint.
    targets = discover_targets(broad_df, feature_cols)

    target_rows = []

    for target in targets:
        vals = pd.to_numeric(broad_df[target], errors="coerce")
        target_rows.append({
            "target_col": target,
            "nonmissing": int(vals.notna().sum()),
            "n_unique": int(vals.nunique(dropna=True)),
            "mean": safe_float(vals.mean()),
            "std": safe_float(vals.std()),
            "min": safe_float(vals.min()),
            "max": safe_float(vals.max()),
            "eligible": bool(vals.notna().sum() >= 30 and vals.nunique(dropna=True) >= 5 and vals.std() > 0),
        })

    target_manifest = pd.DataFrame(target_rows)
    # Eligibility filters avoid training broad models on sparse or near-constant targets.
    eligible_targets = target_manifest.loc[target_manifest["eligible"] == True, "target_col"].astype(str).tolist()

    if len(eligible_targets) == 0:
        raise ValueError("No eligible broad residual targets found.")

    source_manifest = pd.DataFrame({
        "source_name": [
            "run_root",
            "broad_residual_dataset",
            "step05_strict_biology_registry",
            "step05_theme_summary",
            "step05_summary",
        ],
        "path": [
            str(run_root),
            str(broad_path),
            str(registry_path),
            str(theme_path),
            str(step05_summary_path),
        ],
        "exists": [
            run_root.exists(),
            broad_path.exists(),
            registry_path.exists(),
            theme_path.exists(),
            step05_summary_path.exists(),
        ],
    })

    write_table(source_manifest, d01 / "source_manifest.tsv")
    write_table(target_manifest, d01 / "broad_residual_target_manifest.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), d01 / "step05_registry_features_used.tsv")

    all_metrics = []
    all_predictions = []
    all_evidence = []

    for target in eligible_targets:
        # Each residual endpoint is modeled separately so target-specific signal can be compared.
        metrics, predictions, evidence = run_one_target(
            df=broad_df,
            target_col=target,
            feature_cols=feature_cols,
            n_repeats=args.n_repeats,
            test_size=args.test_size,
            max_features_per_split=min(args.max_features_per_split, len(feature_cols)),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            run_shap=args.run_shap,
            max_shap_rows=args.max_shap_rows,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_evidence.append(evidence)

    metrics_long = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    evidence_long = pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame()

    target_summary = summarize_targets(metrics_long)
    # Feature summaries identify recurring registry features across broad residual targets.
    feature_summary = summarize_feature_evidence(evidence_long, registry)
    theme_summary = summarize_theme_evidence(feature_summary)

    write_table(metrics_long, d02 / "broad_residual_metrics_long.tsv")
    write_table(target_summary, d02 / "broad_residual_target_summary.tsv")
    write_table(predictions_long, d02 / "broad_residual_test_predictions_long.tsv")

    write_table(evidence_long, d03 / "broad_residual_feature_evidence_long.tsv")
    write_table(feature_summary, d03 / "broad_residual_feature_evidence_summary.tsv")
    write_table(theme_summary, d03 / "broad_residual_theme_evidence_summary.tsv")

    score_col = "mean_abs_shap" if "mean_abs_shap" in feature_summary.columns and feature_summary["mean_abs_shap"].notna().sum() > 0 else "mean_gain_importance"

    save_bar(
        target_summary,
        "target_col",
        "test_pearson_mean",
        d04 / "fig_01_broad_residual_target_mean_test_pearson.png",
        "V2 Step 06 broad residual target comparison: Pearson",
        "Mean test Pearson",
        top_n=20,
    )

    save_bar(
        target_summary,
        "target_col",
        "test_r2_mean",
        d04 / "fig_02_broad_residual_target_mean_test_r2.png",
        "V2 Step 06 broad residual target comparison: R2",
        "Mean test R2",
        top_n=20,
    )

    save_bar(
        feature_summary,
        "feature_name",
        score_col,
        d04 / "fig_03_broad_residual_top_spatial_features.png",
        "V2 Step 06 broad residual top spatial features",
        score_col,
        top_n=30,
    )

    save_bar(
        theme_summary,
        "biological_theme",
        "total_score",
        d04 / "fig_04_broad_residual_theme_contribution.png",
        "V2 Step 06 broad residual biology theme contribution",
        "Total feature score",
        top_n=15,
    )

    best = target_summary.head(1)

    if best.empty:
        best_target = ""
        best_pearson = None
        best_r2 = None
        best_rmse_improvement = None
    else:
        best_target = str(best.iloc[0]["target_col"])
        best_pearson = safe_float(best.iloc[0].get("test_pearson_mean"))
        best_r2 = safe_float(best.iloc[0].get("test_r2_mean"))
        best_rmse_improvement = safe_float(best.iloc[0].get("rmse_improvement_vs_baseline_mean"))

    run_summary = {
        "status": "pass",
        "official_step": "06_train_broad_residual_model",
        "mode": args.mode,
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "step05_root": str(step05_root),
        "output_root": str(output_root),
        "broad_residual_dataset": str(broad_path),
        "step05_strict_biology_registry": str(registry_path),
        "n_samples": int(broad_df["sample_id"].astype(str).nunique()) if "sample_id" in broad_df.columns else int(len(broad_df)),
        "n_rows": int(len(broad_df)),
        "n_step05_registry_features": int(len(registry)),
        "n_features_used": int(len(feature_cols)),
        "n_targets_found": int(len(targets)),
        "n_targets_eligible": int(len(eligible_targets)),
        "n_repeats": int(args.n_repeats),
        "best_target": best_target,
        "best_target_test_pearson_mean": best_pearson,
        "best_target_test_r2_mean": best_r2,
        "best_target_rmse_improvement_vs_baseline_mean": best_rmse_improvement,
        "production_dependency_on_v1_outputs": "no",
        "uses_step05_v2_registry": "yes",
    }

    write_json(run_summary, output_root / "v2_step06_broad_residual_model_summary.json")
    write_run_provenance(output_root, V2_ROOT, extra=run_summary)

    report_lines = []
    report_lines.append("SPATIAL PREDICTION MODEL V2 STEP 06 BROAD RESIDUAL MODEL REPORT")
    report_lines.append("")

    for key, value in run_summary.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("1. Source files")
    report_lines.append(source_manifest.to_string(index=False))

    report_lines.append("")
    report_lines.append("2. Target manifest")
    report_lines.append(target_manifest.to_string(index=False))

    report_lines.append("")
    report_lines.append("3. Target performance summary")
    report_lines.append(target_summary.to_string(index=False))

    report_lines.append("")
    report_lines.append("4. Top broad residual spatial features")
    show_cols = [
        "feature_name",
        "feature_original",
        "biological_theme",
        "target_count",
        "selection_count",
        "mean_abs_shap",
        "mean_gain_importance",
    ]
    show_cols = [c for c in show_cols if c in feature_summary.columns]

    if feature_summary.empty:
        report_lines.append("No feature evidence generated.")
    else:
        report_lines.append(feature_summary[show_cols].head(60).to_string(index=False))

    report_lines.append("")
    report_lines.append("5. Broad residual theme evidence")
    if theme_summary.empty:
        report_lines.append("No theme evidence generated.")
    else:
        report_lines.append(theme_summary.to_string(index=False))

    report_lines.append("")
    report_lines.append("6. Interpretation")
    report_lines.append("Step 06 is a sample level spatial only broad residual screen.")
    report_lines.append("It uses only the V2 Step 05 strict biology registry and does not include treatment identity features.")
    report_lines.append("This smoke run checks structure and signal direction. The full run should use the full Step 05 registry generated from full Step 04 evidence.")

    report_path = write_text_report(d05 / "v2_step06_broad_residual_model_report.txt", "\n".join(report_lines))

    slide_lines = []
    slide_lines.append("V2 STEP 06 BROAD RESIDUAL MODEL SLIDE NOTES")
    slide_lines.append("")
    slide_lines.append(f"Best target: {best_target}")
    slide_lines.append(f"Best target mean test Pearson: {best_pearson}")
    slide_lines.append(f"Best target mean test R2: {best_r2}")
    slide_lines.append(f"Registry features used: {len(feature_cols)}")
    slide_lines.append(f"Eligible targets: {len(eligible_targets)}")
    slide_lines.append("")
    slide_lines.append("Top themes:")
    if theme_summary.empty:
        slide_lines.append("No themes generated.")
    else:
        for theme in theme_summary.head(8)["biological_theme"].astype(str).tolist():
            slide_lines.append(theme)

    write_text_report(d05 / "v2_step06_broad_residual_slide_notes.txt", "\n".join(slide_lines))

    output_manifest = write_output_manifest(output_root)

    terminal_lines = [
        "Status: pass",
        f"Run root: {run_root}",
        f"Dataset root: {dataset_root}",
        f"Step 05 root: {step05_root}",
        f"Output root: {output_root}",
        f"Report: {report_path}",
        f"Samples: {run_summary['n_samples']}",
        f"Step 05 registry features: {len(registry)}",
        f"Features used: {len(feature_cols)}",
        f"Targets eligible: {len(eligible_targets)}",
        f"Repeats: {args.n_repeats}",
        f"Best target: {best_target}",
        f"Best target mean test Pearson: {best_pearson}",
        f"Best target mean test R2: {best_r2}",
        "Production dependency on V1 outputs: no",
        f"Output manifest rows: {len(output_manifest)}",
    ]

    print("")
    print(terminal_block("V2 STEP 06 BROAD RESIDUAL MODEL COMPLETE", terminal_lines))
    print("")

    print("Target performance summary")
    print(target_summary.head(20).to_string(index=False))
    print("")

    if not feature_summary.empty:
        print("Top broad residual features")
        print(feature_summary[show_cols].head(30).to_string(index=False))
        print("")

    if not theme_summary.empty:
        print("Broad residual theme summary")
        print(theme_summary.head(15).to_string(index=False))
        print("")

    return 0


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
