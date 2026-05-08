"""
Script:
    11_N_label_shuffle_validate_tier1_per_treatment_residual_models.py

Purpose:
    Label shuffle validation for Tier 1 per treatment residual models.

Design:
    New downstream validation script.
    Does not overwrite canonical scripts.
    Does not modify prior outputs.
    Tests whether Tier 1 per treatment model performance exceeds a shuffled label null.
    Every generated .txt report starts with its own filepath.
"""

from pathlib import Path
import argparse
import json
import hashlib
import re
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

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
except Exception as exc:
    raise ImportError("xgboost is required") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--curated-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-treatments", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-features-per-model", type=int, default=40)
    parser.add_argument("--min-feature-variance", type=float, default=1e-12)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_table(path):
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
    path.write_text("\n".join([f"FILEPATH: {path}"] + list(lines)), encoding="utf-8")


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def savefig(path):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def short_label(value, width=55):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + "..."


def safe_name(value, max_len=120):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "unnamed"
    return value[:max_len]


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
        try:
            out["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            pass
        try:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pass
        try:
            out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            pass

    return out


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def locate_handoff(project_root, full_run_root):
    summary = load_json(Path(full_run_root) / "filtered_per_treatment_residual_summary.json")
    derived = summary.get("derived_handoff", "")

    if derived:
        derived = Path(derived)
        handoff = derived / "full102_handoff"
        if handoff.exists():
            return handoff

    spm = Path(project_root) / "prediction_modeling_pipeline" / "spatial_prediction_model"
    candidates = sorted(
        (spm / "outputs" / "_derived_handoffs").glob("residual_prior_adjusted_filtered_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if candidates:
        handoff = candidates[0] / "full102_handoff"
        if handoff.exists():
            return handoff

    raise FileNotFoundError("Could not locate derived residual full102 handoff")


def select_features_training_only(X_train, y_train, feature_cols, max_features, min_var):
    values = X_train[feature_cols].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)

    variances = values.var(axis=0, ddof=1)
    candidates = variances[variances > min_var].index.tolist()

    if not candidates:
        candidates = feature_cols

    y_series = pd.Series(np.asarray(y_train, dtype=float), index=values.index)

    if y_series.std(ddof=1) <= 0:
        return variances.sort_values(ascending=False).head(max_features).index.tolist()

    x_rank = values[candidates].rank(axis=0)
    y_rank = y_series.rank()
    corr = x_rank.corrwith(y_rank).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = pd.DataFrame({
        "feature_name": corr.index,
        "abs_spearman_train": corr.values,
        "variance": variances.loc[corr.index].values,
    })

    score = score.sort_values(["abs_spearman_train", "variance"], ascending=[False, False])
    selected = score["feature_name"].head(max_features).tolist()

    if len(selected) < min(5, len(feature_cols)):
        selected = variances.sort_values(ascending=False).head(max_features).index.tolist()

    return selected


def fit_xgb(X_train, y_train, args, seed):
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    return pipe


def prepare_treatment_data(teacher, spatial, feature_cols, drug_key):
    target = "fused_residual_vs_prior"

    sub = teacher[teacher["drug_key"].astype(str) == str(drug_key)].copy()
    sub["sample_id"] = sub["sample_id"].astype(str)
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub = sub.dropna(subset=[target]).copy()

    merged = sub[["sample_id", "drug_key", target]].merge(spatial[["sample_id"] + feature_cols], on="sample_id", how="inner")
    merged = merged.dropna(subset=[target]).reset_index(drop=True)

    y = pd.to_numeric(merged[target], errors="coerce").astype(float)
    X = merged[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return merged, X, y


def precompute_splits(n_rows, args):
    splits = []
    for repeat in range(args.n_repeats):
        seed = args.random_state + repeat
        train_idx, test_idx = train_test_split(
            np.arange(n_rows),
            test_size=args.test_size,
            random_state=seed,
        )
        splits.append((repeat, seed, train_idx, test_idx))
    return splits


def evaluate_repeated_splits(X, y, feature_cols, splits, args, label_kind, permutation_id=None):
    rows = []

    for repeat, seed, train_idx, test_idx in splits:
        selected = select_features_training_only(
            X.iloc[train_idx],
            y.iloc[train_idx],
            feature_cols,
            args.max_features_per_model,
            args.min_feature_variance,
        )

        pipe = fit_xgb(X.iloc[train_idx][selected], y.iloc[train_idx], args, seed)
        pred_train = pipe.predict(X.iloc[train_idx][selected])
        pred_test = pipe.predict(X.iloc[test_idx][selected])

        baseline = np.repeat(float(y.iloc[train_idx].mean()), len(test_idx))

        train_m = metric_safe(y.iloc[train_idx], pred_train)
        test_m = metric_safe(y.iloc[test_idx], pred_test)
        baseline_m = metric_safe(y.iloc[test_idx], baseline)

        rows.append({
            "label_kind": label_kind,
            "permutation_id": permutation_id,
            "repeat": repeat,
            "random_state": seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "selected_feature_count": int(len(selected)),
            "train_pearson": train_m["pearson"],
            "train_r2": train_m["r2"],
            "test_pearson": test_m["pearson"],
            "test_spearman": test_m["spearman"],
            "test_r2": test_m["r2"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "baseline_test_rmse": baseline_m["rmse"],
            "rmse_improvement_vs_baseline": baseline_m["rmse"] - test_m["rmse"],
        })

    return pd.DataFrame(rows)


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    n = len(p)
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


def make_figures(summary, output_root):
    figures_dir = Path(output_root) / "03_figures"
    ensure_dir(figures_dir)

    if len(summary) == 0:
        return

    plot = summary.sort_values("observed_test_pearson_mean", ascending=False).copy()
    labels = [short_label(x, 50) for x in plot["drug_label"].fillna(plot["drug_key"])]
    values = plot["observed_test_pearson_mean"].astype(float).values
    order = np.arange(len(plot))[::-1]

    plt.figure(figsize=(11, max(6, len(plot) * 0.32)))
    plt.barh(order, values[::-1])
    plt.yticks(order, labels[::-1], fontsize=8)
    plt.xlabel("Observed mean test Pearson")
    plt.title("Tier 1 models entering label shuffle validation")
    savefig(figures_dir / "fig_01_observed_tier1_test_pearson.png")

    plot2 = summary.sort_values("empirical_p_value", ascending=True).copy()
    labels = [short_label(x, 50) for x in plot2["drug_label"].fillna(plot2["drug_key"])]
    values = plot2["empirical_p_value"].astype(float).values
    order = np.arange(len(plot2))[::-1]

    plt.figure(figsize=(11, max(6, len(plot2) * 0.32)))
    plt.barh(order, values[::-1])
    plt.yticks(order, labels[::-1], fontsize=8)
    plt.xlabel("Empirical label shuffle p value")
    plt.title("Tier 1 label shuffle validation")
    savefig(figures_dir / "fig_02_empirical_p_values.png")

    plot3 = summary.sort_values("fdr_q_value", ascending=True).copy()
    labels = [short_label(x, 50) for x in plot3["drug_label"].fillna(plot3["drug_key"])]
    values = plot3["fdr_q_value"].astype(float).values
    order = np.arange(len(plot3))[::-1]

    plt.figure(figsize=(11, max(6, len(plot3) * 0.32)))
    plt.barh(order, values[::-1])
    plt.yticks(order, labels[::-1], fontsize=8)
    plt.xlabel("Benjamini Hochberg FDR q value")
    plt.title("Tier 1 label shuffle FDR")
    savefig(figures_dir / "fig_03_fdr_q_values.png")


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    full_run_root = Path(args.full_run_root)
    curated_run_root = Path(args.curated_run_root)
    output_root = Path(args.output_root)

    inputs_dir = output_root / "01_inputs"
    tables_dir = output_root / "02_label_shuffle_tables"
    figures_dir = output_root / "03_figures"
    reports_dir = output_root / "04_reports"

    for folder in [inputs_dir, tables_dir, figures_dir, reports_dir]:
        ensure_dir(folder)

    tier1_path = curated_run_root / "01_curated_tables" / "tier1_high_confidence_treatment_models.tsv"
    feature_path = full_run_root / "01_inputs_and_eligibility" / "spatial_feature_list.tsv"

    tier1 = read_table(tier1_path)
    feature_table = read_table(feature_path)

    if "feature_name" not in feature_table.columns:
        raise ValueError("spatial_feature_list.tsv must contain feature_name")

    feature_cols = feature_table["feature_name"].astype(str).tolist()

    if args.max_treatments and args.max_treatments > 0:
        selected = tier1.head(args.max_treatments).copy()
    else:
        selected = tier1.copy()

    handoff = locate_handoff(project_root, full_run_root)
    teacher = read_table(handoff / "visium_fused_teacher_table.tsv")
    spatial = read_table(handoff / "model_input_numeric.csv")

    teacher = teacher.copy()
    teacher["sample_id"] = teacher["sample_id"].astype(str)
    teacher["drug_key"] = teacher["drug_key"].astype(str)

    spatial = spatial.copy()
    spatial["sample_id"] = spatial["sample_id"].astype(str)

    feature_cols = [c for c in feature_cols if c in spatial.columns]

    write_table(tier1, inputs_dir / "tier1_treatments_all.tsv")
    write_table(selected, inputs_dir / "tier1_treatments_selected.tsv")
    write_table(pd.DataFrame({"feature_name": feature_cols}), inputs_dir / "spatial_features_used.tsv")

    all_observed = []
    all_null = []
    summary_rows = []

    print("")
    print("=" * 100)
    print("TIER 1 LABEL SHUFFLE VALIDATION")
    print("=" * 100)
    print("Mode:", args.mode)
    print("Selected Tier 1 treatments:", len(selected))
    print("Permutations per treatment:", args.n_permutations)
    print("Repeated splits per observed or shuffled fit:", args.n_repeats)
    print("Spatial features:", len(feature_cols))
    print("Handoff:", handoff)

    for idx, row in selected.reset_index(drop=True).iterrows():
        drug_key = str(row["drug_key"])
        drug_label = str(row.get("drug_label", drug_key))

        print("")
        print(f"[{idx + 1}/{len(selected)}] {drug_label}")

        data, X, y = prepare_treatment_data(teacher, spatial, feature_cols, drug_key)
        splits = precompute_splits(len(data), args)

        observed = evaluate_repeated_splits(X, y, feature_cols, splits, args, "observed", None)
        observed.insert(0, "drug_key", drug_key)
        observed.insert(1, "drug_label", drug_label)
        all_observed.append(observed)

        observed_mean = float(pd.to_numeric(observed["test_pearson"], errors="coerce").mean())
        observed_median = float(pd.to_numeric(observed["test_pearson"], errors="coerce").median())
        observed_r2_mean = float(pd.to_numeric(observed["test_r2"], errors="coerce").mean())
        observed_rmse_improvement = float(pd.to_numeric(observed["rmse_improvement_vs_baseline"], errors="coerce").mean())

        rng = np.random.default_rng(args.random_state + 10000 + idx)
        null_perm_rows = []

        for perm in range(args.n_permutations):
            shuffled_values = np.asarray(y.values, dtype=float).copy()
            rng.shuffle(shuffled_values)
            y_perm = pd.Series(shuffled_values, index=y.index)

            null_metrics = evaluate_repeated_splits(X, y_perm, feature_cols, splits, args, "shuffled", perm)
            null_metrics.insert(0, "drug_key", drug_key)
            null_metrics.insert(1, "drug_label", drug_label)
            all_null.append(null_metrics)

            null_perm_rows.append({
                "drug_key": drug_key,
                "drug_label": drug_label,
                "permutation_id": perm,
                "null_test_pearson_mean": float(pd.to_numeric(null_metrics["test_pearson"], errors="coerce").mean()),
                "null_test_r2_mean": float(pd.to_numeric(null_metrics["test_r2"], errors="coerce").mean()),
                "null_rmse_improvement_mean": float(pd.to_numeric(null_metrics["rmse_improvement_vs_baseline"], errors="coerce").mean()),
            })

        null_perm = pd.DataFrame(null_perm_rows)
        null_mean = float(null_perm["null_test_pearson_mean"].mean())
        null_sd = float(null_perm["null_test_pearson_mean"].std(ddof=1)) if len(null_perm) > 1 else np.nan

        empirical_p = float((1 + (null_perm["null_test_pearson_mean"] >= observed_mean).sum()) / (1 + len(null_perm)))
        z_score = float((observed_mean - null_mean) / null_sd) if null_sd and np.isfinite(null_sd) and null_sd > 0 else np.nan

        summary_rows.append({
            "drug_key": drug_key,
            "drug_label": drug_label,
            "n_samples": int(len(data)),
            "n_features": int(len(feature_cols)),
            "n_repeats": int(args.n_repeats),
            "n_permutations": int(args.n_permutations),
            "observed_test_pearson_mean": observed_mean,
            "observed_test_pearson_median": observed_median,
            "observed_test_r2_mean": observed_r2_mean,
            "observed_rmse_improvement_mean": observed_rmse_improvement,
            "null_test_pearson_mean": null_mean,
            "null_test_pearson_sd": null_sd,
            "null_test_pearson_q95": float(null_perm["null_test_pearson_mean"].quantile(0.95)),
            "null_test_pearson_max": float(null_perm["null_test_pearson_mean"].max()),
            "empirical_p_value": empirical_p,
            "observed_minus_null_mean": float(observed_mean - null_mean),
            "z_score_vs_null": z_score,
            "source_curated_test_pearson_mean": row.get("test_pearson_mean", np.nan),
            "source_curated_test_r2_mean": row.get("test_r2_mean", np.nan),
            "source_interpretation_tier": row.get("interpretation_tier", ""),
            "source_caution_flags": row.get("caution_flags", ""),
        })

        print("  observed mean test Pearson:", observed_mean)
        print("  null mean test Pearson:", null_mean)
        print("  empirical p:", empirical_p)
        print("  z score:", z_score)

    observed_long = pd.concat(all_observed, ignore_index=True) if all_observed else pd.DataFrame()
    null_long = pd.concat(all_null, ignore_index=True) if all_null else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    if len(summary) > 0:
        summary["fdr_q_value"] = bh_fdr(summary["empirical_p_value"].values)
        summary["passes_p05"] = summary["empirical_p_value"] <= 0.05
        summary["passes_fdr10"] = summary["fdr_q_value"] <= 0.10
        summary["passes_fdr20"] = summary["fdr_q_value"] <= 0.20
        summary = summary.sort_values(["fdr_q_value", "empirical_p_value", "observed_test_pearson_mean"], ascending=[True, True, False])

    write_table(observed_long, tables_dir / "observed_repeated_split_metrics_long.tsv")
    write_table(null_long, tables_dir / "shuffled_repeated_split_metrics_long.tsv")
    write_table(summary, tables_dir / "label_shuffle_treatment_summary.tsv")

    make_figures(summary, output_root)

    script_provenance = pd.DataFrame([{
        "script": str(Path(__file__)),
        "exists": Path(__file__).exists(),
        "sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        "note": "new Tier 1 label shuffle validation script, no canonical script overwritten"
    }])
    write_table(script_provenance, output_root / "script_provenance.tsv")

    report_path = reports_dir / "tier1_label_shuffle_validation_report.txt"

    show_cols = [
        "drug_key",
        "drug_label",
        "n_samples",
        "observed_test_pearson_mean",
        "null_test_pearson_mean",
        "null_test_pearson_q95",
        "empirical_p_value",
        "fdr_q_value",
        "passes_p05",
        "passes_fdr10",
        "observed_minus_null_mean",
        "z_score_vs_null",
        "source_caution_flags",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]

    lines = []
    lines.append("TIER 1 PER TREATMENT RESIDUAL LABEL SHUFFLE VALIDATION REPORT")
    lines.append("=" * 110)
    lines.append("")
    lines.append(f"Full per treatment run root: {full_run_root}")
    lines.append(f"Curated run root: {curated_run_root}")
    lines.append(f"Validation output root: {output_root}")
    lines.append(f"Mode: {args.mode}")
    lines.append("")
    lines.append("1. Validation design")
    lines.append("-" * 110)
    lines.append("This test compares each Tier 1 treatment model against a shuffled residual label null.")
    lines.append("For each treatment, the spatial feature matrix is unchanged and fused_residual_vs_prior is shuffled within that treatment.")
    lines.append("Feature selection is repeated inside each training split, including shuffled null fits.")
    lines.append("The empirical p value is the fraction of shuffled mean test Pearson values greater than or equal to the observed mean test Pearson.")
    lines.append("")
    lines.append("2. Inputs")
    lines.append("-" * 110)
    lines.append(f"Tier 1 treatments available: {len(tier1)}")
    lines.append(f"Tier 1 treatments tested: {len(selected)}")
    lines.append(f"Spatial features: {len(feature_cols)}")
    lines.append(f"Repeated splits per treatment: {args.n_repeats}")
    lines.append(f"Label permutations per treatment: {args.n_permutations}")
    lines.append(f"Handoff used: {handoff}")
    lines.append("")
    lines.append("3. Summary")
    lines.append("-" * 110)
    if len(summary) > 0:
        lines.append(f"Treatments with empirical p <= 0.05: {int(summary['passes_p05'].sum())}")
        lines.append(f"Treatments with FDR q <= 0.10: {int(summary['passes_fdr10'].sum())}")
        lines.append(f"Treatments with FDR q <= 0.20: {int(summary['passes_fdr20'].sum())}")
    else:
        lines.append("No treatments were evaluated.")
    lines.append("")
    lines.append("4. Treatment results")
    lines.append("-" * 110)
    if len(summary) > 0:
        lines.append(summary[show_cols].to_string(index=False))
    else:
        lines.append("No summary table available.")
    lines.append("")
    lines.append("5. Interpretation")
    lines.append("-" * 110)
    lines.append("Tier 1 treatments that pass label shuffle validation are the strongest treatment specific spatial biology candidates.")
    lines.append("Tier 1 treatments that do not pass label shuffle validation should remain screening findings only.")
    lines.append("Because many treatments were screened, FDR controlled results should be prioritized over raw empirical p values.")
    lines.append("")
    lines.append("6. Output files")
    lines.append("-" * 110)
    lines.append(f"Observed repeated split metrics: {tables_dir / 'observed_repeated_split_metrics_long.tsv'}")
    lines.append(f"Shuffled repeated split metrics: {tables_dir / 'shuffled_repeated_split_metrics_long.tsv'}")
    lines.append(f"Treatment summary: {tables_dir / 'label_shuffle_treatment_summary.tsv'}")
    lines.append(f"Figures: {figures_dir}")

    write_text_report(report_path, lines)

    summary_json = {
        "full_run_root": str(full_run_root),
        "curated_run_root": str(curated_run_root),
        "output_root": str(output_root),
        "report": str(report_path),
        "mode": args.mode,
        "n_tier1_available": int(len(tier1)),
        "n_tier1_tested": int(len(selected)),
        "n_permutations": int(args.n_permutations),
        "n_repeats": int(args.n_repeats),
        "n_pass_p05": int(summary["passes_p05"].sum()) if len(summary) else 0,
        "n_pass_fdr10": int(summary["passes_fdr10"].sum()) if len(summary) else 0,
        "n_pass_fdr20": int(summary["passes_fdr20"].sum()) if len(summary) else 0,
    }
    (output_root / "tier1_label_shuffle_validation_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("")
    print("=" * 110)
    print("TIER 1 LABEL SHUFFLE VALIDATION COMPLETE")
    print("=" * 110)
    print("Output root:", output_root)
    print("Report:", report_path)
    print("Mode:", args.mode)
    print("Tier 1 treatments tested:", len(selected))
    print("Permutations per treatment:", args.n_permutations)
    print("Repeated splits per treatment:", args.n_repeats)
    if len(summary) > 0:
        print("Pass empirical p <= 0.05:", int(summary["passes_p05"].sum()))
        print("Pass FDR q <= 0.10:", int(summary["passes_fdr10"].sum()))
        print("Pass FDR q <= 0.20:", int(summary["passes_fdr20"].sum()))
        print("")
        print("Top label shuffle results:")
        print(summary[show_cols].head(30).to_string(index=False))
    print("")
    print("Generated output folders:")
    for folder in [inputs_dir, tables_dir, figures_dir, reports_dir]:
        print(" ", folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
