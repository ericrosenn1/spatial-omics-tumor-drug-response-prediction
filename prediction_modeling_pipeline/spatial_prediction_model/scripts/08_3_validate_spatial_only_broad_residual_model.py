"""
Script:
    08_N_validate_spatial_only_broad_residual_model.py

Purpose:
    Quick validation of the spatial-only broad residual sample-level model.

Design:
    New experimental validation script.
    Does not overwrite canonical scripts.
    Uses repeated train/test splits.
    Quantifies stability of test metrics.
    Quantifies recurrence of SHAP features across splits.

Text report convention:
    Every generated .txt report starts with its own filepath.
"""

from pathlib import Path
import argparse
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
except Exception as exc:
    raise ImportError("xgboost is required") from exc

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-shap-repeats", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument(
        "--targets",
        default="mean_residual,broad_resistance_score,strong_negative_fraction_m005,residual_iqr",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_any(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def write_table(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def write_text_report(path, lines):
    path = Path(path)
    ensure_dir(path.parent)
    lines = [f"FILEPATH: {path}"] + list(lines)
    path.write_text("\n".join(lines), encoding="utf-8")


def savefig(path):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def short_label(value, width=45):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + "..."


def metric_safe(y_true, y_pred):
    out = {
        "n": int(len(y_true)),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    out["n"] = int(len(y_true))

    if len(y_true) == 0:
        return out

    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)

    if len(y_true) >= 3 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        out["r2"] = float(r2_score(y_true, y_pred))
        try:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pass
        try:
            out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            pass

    return out


def q025(x):
    return float(np.nanquantile(x, 0.025))


def q975(x):
    return float(np.nanquantile(x, 0.975))


def train_predict_one(X, y, train_idx, test_idx, args):
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=args.random_state,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred_train = pipe.predict(X.iloc[train_idx])
    pred_test = pipe.predict(X.iloc[test_idx])

    baseline_value = float(y.iloc[train_idx].mean())
    baseline_test = np.repeat(baseline_value, len(test_idx))

    train_metrics = metric_safe(y.iloc[train_idx], pred_train)
    test_metrics = metric_safe(y.iloc[test_idx], pred_test)
    baseline_metrics = metric_safe(y.iloc[test_idx], baseline_test)

    return pipe, train_metrics, test_metrics, baseline_metrics


def main():
    args = parse_args()

    broad_root = Path(args.broad_run_root)
    output_root = Path(args.output_root)

    ensure_dir(output_root)

    tables_dir = output_root / "01_validation_tables"
    models_dir = output_root / "02_validation_models"
    figures_dir = output_root / "03_figures"
    reports_dir = output_root / "04_reports"

    for folder in [tables_dir, models_dir, figures_dir, reports_dir]:
        ensure_dir(folder)

    modeling_path = broad_root / "01_prepared_sample_level_data" / "sample_level_modeling_table.tsv"
    feature_meta_path = broad_root / "01_prepared_sample_level_data" / "selected_feature_metadata.tsv"
    selected_features_path = broad_root / "01_prepared_sample_level_data" / "selected_spatial_features.tsv"

    modeling = read_any(modeling_path)
    feature_meta = read_any(feature_meta_path)
    selected_features = read_any(selected_features_path)

    if "feature_name" not in selected_features.columns:
        raise ValueError("selected_spatial_features.tsv must contain feature_name")

    feature_cols = selected_features["feature_name"].astype(str).tolist()
    feature_cols = [c for c in feature_cols if c in modeling.columns]

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    targets = [t for t in targets if t in modeling.columns]

    if not targets:
        raise ValueError("No requested targets were found in sample_level_modeling_table.tsv")

    if len(feature_cols) < 10:
        raise ValueError("Too few feature columns available for validation")

    modeling = modeling.copy()
    modeling["sample_id"] = modeling["sample_id"].astype(str)

    for col in feature_cols:
        modeling[col] = pd.to_numeric(modeling[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    write_table(pd.DataFrame({"target_name": targets}), tables_dir / "validated_targets.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), tables_dir / "validated_features.tsv")

    all_metric_rows = []
    all_shap_rows = []
    all_prediction_rows = []

    for target in targets:
        print("")
        print("=" * 90)
        print("Validating target:", target)
        print("=" * 90)

        y = pd.to_numeric(modeling[target], errors="coerce")
        keep = y.notna()
        data = modeling.loc[keep].copy()
        y = y.loc[keep].astype(float)
        X = data[feature_cols].copy()

        for repeat in range(args.n_repeats):
            seed = args.random_state + repeat

            train_idx, test_idx = train_test_split(
                np.arange(len(data)),
                test_size=args.test_size,
                random_state=seed,
            )

            pipe, train_m, test_m, baseline_m = train_predict_one(X, y, train_idx, test_idx, args)

            row = {
                "target_name": target,
                "repeat": repeat,
                "random_state": seed,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_mae": train_m["mae"],
                "train_rmse": train_m["rmse"],
                "train_r2": train_m["r2"],
                "train_pearson": train_m["pearson"],
                "train_spearman": train_m["spearman"],
                "test_mae": test_m["mae"],
                "test_rmse": test_m["rmse"],
                "test_r2": test_m["r2"],
                "test_pearson": test_m["pearson"],
                "test_spearman": test_m["spearman"],
                "baseline_test_mae": baseline_m["mae"],
                "baseline_test_rmse": baseline_m["rmse"],
                "baseline_test_r2": baseline_m["r2"],
                "mae_improvement_vs_baseline": baseline_m["mae"] - test_m["mae"],
                "rmse_improvement_vs_baseline": baseline_m["rmse"] - test_m["rmse"],
            }
            all_metric_rows.append(row)

            pred_test = pipe.predict(X.iloc[test_idx])
            pred_rows = pd.DataFrame({
                "target_name": target,
                "repeat": repeat,
                "sample_id": data.iloc[test_idx]["sample_id"].values,
                "target": y.iloc[test_idx].values,
                "prediction": pred_test,
            })
            pred_rows["residual"] = pred_rows["prediction"] - pred_rows["target"]
            all_prediction_rows.append(pred_rows)

            if HAS_SHAP and repeat < args.n_shap_repeats:
                try:
                    model = pipe.named_steps["model"]
                    X_imp = pd.DataFrame(
                        pipe.named_steps["imputer"].transform(X),
                        columns=feature_cols,
                    )
                    explainer = shap.TreeExplainer(model)
                    values = explainer.shap_values(X_imp)
                    mean_abs = np.abs(values).mean(axis=0)

                    shap_df = pd.DataFrame({
                        "target_name": target,
                        "repeat": repeat,
                        "feature_name": feature_cols,
                        "mean_abs_shap": mean_abs,
                    })
                    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                    shap_df["rank"] = np.arange(1, len(shap_df) + 1)
                    all_shap_rows.append(shap_df)
                except Exception as exc:
                    print("SHAP failed for", target, "repeat", repeat, ":", exc)

        done = pd.DataFrame([r for r in all_metric_rows if r["target_name"] == target])
        print("Repeats:", len(done))
        print("Mean test Pearson:", float(done["test_pearson"].mean()))
        print("Median test Pearson:", float(done["test_pearson"].median()))
        print("Mean test R2:", float(done["test_r2"].mean()))
        print("Mean test MAE:", float(done["test_mae"].mean()))
        print("Mean RMSE improvement vs baseline:", float(done["rmse_improvement_vs_baseline"].mean()))

    metrics_long = pd.DataFrame(all_metric_rows)
    predictions_long = pd.concat(all_prediction_rows, ignore_index=True) if all_prediction_rows else pd.DataFrame()
    shap_long = pd.concat(all_shap_rows, ignore_index=True) if all_shap_rows else pd.DataFrame()

    write_table(metrics_long, tables_dir / "repeated_split_metrics_long.tsv")
    write_table(predictions_long, tables_dir / "repeated_split_predictions_long.tsv")
    write_table(shap_long, tables_dir / "repeated_split_shap_long.tsv")

    metric_summary = (
        metrics_long
        .groupby("target_name", dropna=False)
        .agg(
            n_repeats=("repeat", "count"),
            test_pearson_mean=("test_pearson", "mean"),
            test_pearson_median=("test_pearson", "median"),
            test_pearson_std=("test_pearson", "std"),
            test_pearson_q025=("test_pearson", q025),
            test_pearson_q975=("test_pearson", q975),
            test_r2_mean=("test_r2", "mean"),
            test_r2_median=("test_r2", "median"),
            test_r2_q025=("test_r2", q025),
            test_r2_q975=("test_r2", q975),
            test_mae_mean=("test_mae", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            mae_improvement_vs_baseline_mean=("mae_improvement_vs_baseline", "mean"),
            rmse_improvement_vs_baseline_mean=("rmse_improvement_vs_baseline", "mean"),
        )
        .reset_index()
        .sort_values("test_pearson_mean", ascending=False)
    )

    write_table(metric_summary, tables_dir / "repeated_split_metric_summary.tsv")

    if not shap_long.empty:
        shap_with_meta = shap_long.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))

        shap_stability = (
            shap_with_meta
            .groupby(["target_name", "feature_name"], dropna=False)
            .agg(
                n_shap_repeats=("repeat", "nunique"),
                mean_abs_shap_mean=("mean_abs_shap", "mean"),
                mean_abs_shap_median=("mean_abs_shap", "median"),
                mean_rank=("rank", "mean"),
                top10_frequency=("rank", lambda s: float((s <= 10).mean())),
                top20_frequency=("rank", lambda s: float((s <= 20).mean())),
            )
            .reset_index()
        )

        meta_cols = [c for c in ["feature_name", "feature_original", "feature_group", "feature_axis", "biological_theme"] if c in feature_meta.columns]
        if meta_cols:
            shap_stability = shap_stability.merge(feature_meta[meta_cols].drop_duplicates("feature_name"), on="feature_name", how="left")

        shap_stability = shap_stability.sort_values(
            ["target_name", "top10_frequency", "mean_abs_shap_mean"],
            ascending=[True, False, False],
        )

        write_table(shap_stability, tables_dir / "shap_feature_stability.tsv")

        theme_stability = (
            shap_stability
            .groupby(["target_name", "biological_theme"], dropna=False)
            .agg(
                n_features=("feature_name", "count"),
                mean_top10_frequency=("top10_frequency", "mean"),
                max_top10_frequency=("top10_frequency", "max"),
                total_mean_abs_shap=("mean_abs_shap_mean", "sum"),
            )
            .reset_index()
            .sort_values(["target_name", "total_mean_abs_shap"], ascending=[True, False])
        )
        write_table(theme_stability, tables_dir / "shap_theme_stability.tsv")
    else:
        shap_stability = pd.DataFrame()
        theme_stability = pd.DataFrame()

    model_summary = {
        "broad_run_root": str(broad_root),
        "output_root": str(output_root),
        "n_samples": int(modeling["sample_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "targets": targets,
        "n_repeats": int(args.n_repeats),
        "n_shap_repeats": int(args.n_shap_repeats),
        "best_target_by_mean_test_pearson": str(metric_summary.iloc[0]["target_name"]),
        "best_mean_test_pearson": float(metric_summary.iloc[0]["test_pearson_mean"]),
        "best_median_test_pearson": float(metric_summary.iloc[0]["test_pearson_median"]),
    }
    (output_root / "quick_validation_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 6))
    plot_targets = metric_summary["target_name"].tolist()
    data = [metrics_long.loc[metrics_long["target_name"] == t, "test_pearson"].astype(float).values for t in plot_targets]
    plt.boxplot(data, labels=[short_label(t, 35) for t in plot_targets], showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test Pearson across repeated splits")
    plt.title("Repeated split validation: test Pearson")
    savefig(figures_dir / "fig_01_repeated_split_test_pearson_boxplot.png")

    plt.figure(figsize=(10, 6))
    data = [metrics_long.loc[metrics_long["target_name"] == t, "test_r2"].astype(float).values for t in plot_targets]
    plt.boxplot(data, labels=[short_label(t, 35) for t in plot_targets], showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test R2 across repeated splits")
    plt.title("Repeated split validation: test R2")
    savefig(figures_dir / "fig_02_repeated_split_test_r2_boxplot.png")

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(metric_summary)), metric_summary["test_pearson_mean"].astype(float).values)
    plt.xticks(np.arange(len(metric_summary)), [short_label(t, 35) for t in metric_summary["target_name"]], rotation=35, ha="right")
    plt.ylabel("Mean test Pearson across repeated splits")
    plt.title("Mean test Pearson by target")
    savefig(figures_dir / "fig_03_mean_test_pearson_by_target.png")

    if not shap_stability.empty:
        best_target = metric_summary.iloc[0]["target_name"]
        top_features = shap_stability[shap_stability["target_name"] == best_target].copy()
        top_features = top_features.sort_values(["top10_frequency", "mean_abs_shap_mean"], ascending=False).head(25)

        labels = [
            short_label(row.get("feature_original", row.get("feature_name", "")), 65)
            for _, row in top_features.iterrows()
        ]
        vals = top_features["top10_frequency"].astype(float).values
        order = np.arange(len(top_features))[::-1]

        plt.figure(figsize=(10, max(6, len(top_features) * 0.34)))
        plt.barh(order, vals[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Top 10 SHAP recurrence frequency")
        plt.title("Stable SHAP features for " + str(best_target))
        savefig(figures_dir / "fig_04_stable_shap_features_best_target.png")

    report_path = reports_dir / "quick_validation_report.txt"

    lines = []
    lines.append("QUICK VALIDATION REPORT FOR SPATIAL-ONLY BROAD RESIDUAL MODEL")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"Broad model run root: {broad_root}")
    lines.append(f"Validation output root: {output_root}")
    lines.append("")
    lines.append("1. Validation design")
    lines.append("-" * 95)
    lines.append(f"Repeated random train/test splits: {args.n_repeats}")
    lines.append(f"SHAP stability repeats per target: {args.n_shap_repeats}")
    lines.append(f"Samples: {modeling['sample_id'].nunique()}")
    lines.append(f"Spatial biology features: {len(feature_cols)}")
    lines.append(f"Targets: {', '.join(targets)}")
    lines.append("")
    lines.append("2. Metric summary")
    lines.append("-" * 95)
    lines.append(metric_summary.to_string(index=False))
    lines.append("")
    lines.append("3. Interpretation")
    lines.append("-" * 95)
    lines.append("This validation tests whether the original broad residual signal survives repeated train/test splitting.")
    lines.append("Positive mean test Pearson across repeated splits supports a real sample-level spatial signal.")
    lines.append("Wide intervals or negative R2 indicate the result is still screening-level and should not be treated as final proof.")
    lines.append("")
    lines.append("4. Best target")
    lines.append("-" * 95)
    lines.append(f"Best target by mean test Pearson: {model_summary['best_target_by_mean_test_pearson']}")
    lines.append(f"Mean test Pearson: {model_summary['best_mean_test_pearson']:.4f}")
    lines.append(f"Median test Pearson: {model_summary['best_median_test_pearson']:.4f}")
    lines.append("")
    lines.append("5. Recommended decision rule")
    lines.append("-" * 95)
    lines.append("If mean_residual or broad_resistance_score has mean test Pearson clearly above zero and improves RMSE versus baseline, keep it as the broad sample-level spatial phenotype.")
    lines.append("If the signal is unstable, treat broad residual modeling as exploratory and prioritize filtered per-treatment residual models.")
    lines.append("")
    if not shap_stability.empty:
        lines.append("6. Stable SHAP features for best target")
        lines.append("-" * 95)
        best_target = metric_summary.iloc[0]["target_name"]
        best_features = shap_stability[shap_stability["target_name"] == best_target].copy()
        best_features = best_features.sort_values(["top10_frequency", "mean_abs_shap_mean"], ascending=False).head(30)
        show_cols = [c for c in ["feature_name", "feature_original", "biological_theme", "top10_frequency", "top20_frequency", "mean_abs_shap_mean", "mean_rank"] if c in best_features.columns]
        lines.append(best_features[show_cols].to_string(index=False))
        lines.append("")

    write_text_report(report_path, lines)

    script_provenance = pd.DataFrame([{
        "script": str(Path(__file__)),
        "exists": Path(__file__).exists(),
        "sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        "note": "new validation script, no canonical script overwritten",
    }])
    write_table(script_provenance, output_root / "script_provenance.tsv")

    print("")
    print("=" * 95)
    print("QUICK VALIDATION COMPLETE")
    print("=" * 95)
    print("Output root:", output_root)
    print("Report:", report_path)
    print("Best target:", model_summary["best_target_by_mean_test_pearson"])
    print("Best mean test Pearson:", model_summary["best_mean_test_pearson"])
    print("Best median test Pearson:", model_summary["best_median_test_pearson"])
    print("")
    print("Metric summary:")
    print(metric_summary.to_string(index=False))
    print("")
    print("Generated output folders:")
    for folder in [tables_dir, models_dir, figures_dir, reports_dir]:
        print(" ", folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
