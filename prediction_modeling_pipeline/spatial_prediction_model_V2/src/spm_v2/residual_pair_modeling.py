from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .interpretation import attach_biology_theme
from .model_training import make_xgb_pipeline, select_features_training_only
from .validation import metric_safe


def subset_pair_dataset(
    df: pd.DataFrame,
    max_samples: int = 60,
    max_treatments: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
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
        treatment_counts = out["drug_key"].value_counts()
        keep_treatments = treatment_counts.head(max_treatments).index.astype(str).tolist()
        out = out[out["drug_key"].isin(keep_treatments)].copy()

    return out.reset_index(drop=True)


def make_split_indices(
    df: pd.DataFrame,
    split_mode: str = "grouped_sample_split",
    test_size: float = 0.20,
    random_state: int = 42,
):
    if split_mode == "grouped_sample_split":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df["sample_id"].astype(str).values
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return train_idx, test_idx

    if split_mode == "random_pair_split":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
        )
        return train_idx, test_idx

    raise ValueError(f"Unsupported split_mode: {split_mode}")


def normalized_score(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    vmax = float(vals.max()) if len(vals) else 0.0
    if vmax <= 0:
        return vals * 0.0
    return vals / vmax


def train_repeated_residual_pair_model(
    pair_df: pd.DataFrame,
    feature_meta: pd.DataFrame,
    feature_cols: list[str],
    residual_col: str = "fused_residual_vs_prior",
    n_repeats: int = 3,
    split_mode: str = "grouped_sample_split",
    test_size: float = 0.20,
    max_features_per_split: int = 75,
    n_estimators: int = 120,
    max_depth: int = 2,
    learning_rate: float = 0.03,
    random_state: int = 42,
    run_shap: bool = True,
    max_shap_rows: int = 2500,
) -> dict:
    try:
        import shap
        has_shap = True
    except Exception:
        shap = None
        has_shap = False

    df = pair_df.copy()
    df[residual_col] = pd.to_numeric(df[residual_col], errors="coerce")
    df = df.dropna(subset=[residual_col]).reset_index(drop=True)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    metrics_rows = []
    prediction_rows = []
    evidence_rows = []
    selected_rows = []

    x_all = df[feature_cols].copy()
    y_all = df[residual_col].astype(float)

    for repeat in range(n_repeats):
        seed = random_state + repeat
        train_idx, test_idx = make_split_indices(
            df,
            split_mode=split_mode,
            test_size=test_size,
            random_state=seed,
        )

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

        pred_train = pipe.predict(x_all.iloc[train_idx][selected])
        pred_test = pipe.predict(x_all.iloc[test_idx][selected])

        baseline_test = np.repeat(float(y_all.iloc[train_idx].mean()), len(test_idx))

        train_m = metric_safe(y_all.iloc[train_idx], pred_train)
        test_m = metric_safe(y_all.iloc[test_idx], pred_test)
        baseline_m = metric_safe(y_all.iloc[test_idx], baseline_test)

        for split_name, metric_dict in [("train", train_m), ("test", test_m), ("baseline_test", baseline_m)]:
            row = {
                "repeat": repeat,
                "random_state": seed,
                "split": split_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_selected_features": int(len(selected)),
            }
            row.update(metric_dict)
            if split_name == "test":
                row["rmse_improvement_vs_baseline"] = baseline_m["rmse"] - test_m["rmse"]
                row["mae_improvement_vs_baseline"] = baseline_m["mae"] - test_m["mae"]
            else:
                row["rmse_improvement_vs_baseline"] = np.nan
                row["mae_improvement_vs_baseline"] = np.nan
            metrics_rows.append(row)

        pred_df = pd.DataFrame({
            "repeat": repeat,
            "split": "test",
            "sample_id": df.iloc[test_idx]["sample_id"].astype(str).values,
            "drug_key": df.iloc[test_idx]["drug_key"].astype(str).values,
            "target": y_all.iloc[test_idx].values,
            "prediction": pred_test,
        })
        pred_df["prediction_error"] = pred_df["prediction"] - pred_df["target"]
        prediction_rows.append(pred_df)

        model = pipe.named_steps["model"]

        gains = pd.DataFrame({
            "feature_name": selected,
            "gain_importance": model.feature_importances_,
        })

        shap_df = pd.DataFrame({
            "feature_name": selected,
            "mean_abs_shap": np.nan,
            "shap_status": "not_run",
        })

        if run_shap and has_shap:
            try:
                rng = np.random.default_rng(seed)
                eval_pool = np.arange(len(df))
                if len(eval_pool) > max_shap_rows:
                    eval_idx = rng.choice(eval_pool, size=max_shap_rows, replace=False)
                else:
                    eval_idx = eval_pool

                x_eval = x_all.iloc[eval_idx][selected]
                x_imp = pd.DataFrame(
                    pipe.named_steps["imputer"].transform(x_eval),
                    columns=selected,
                )

                explainer = shap.TreeExplainer(model)
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

        evidence = gains.merge(shap_df, on="feature_name", how="left")
        evidence.insert(0, "repeat", repeat)
        evidence.insert(1, "random_state", seed)
        evidence_rows.append(evidence)

        selected_rows.append(pd.DataFrame({
            "repeat": repeat,
            "feature_name": selected,
            "selected": True,
        }))

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    evidence_long = pd.concat(evidence_rows, ignore_index=True) if evidence_rows else pd.DataFrame()
    selected_long = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    feature_base = pd.DataFrame({"feature_name": feature_cols})
    selected_summary = (
        selected_long
        .groupby("feature_name", dropna=False)
        .agg(selection_count=("selected", "sum"))
        .reset_index()
    )

    evidence_summary = (
        evidence_long
        .groupby("feature_name", dropna=False)
        .agg(
            mean_gain_importance=("gain_importance", "mean"),
            max_gain_importance=("gain_importance", "max"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            max_abs_shap=("mean_abs_shap", "max"),
            shap_success_count=("shap_status", lambda s: int((s.astype(str) == "success").sum())),
        )
        .reset_index()
    )

    feature_stability = feature_base.merge(selected_summary, on="feature_name", how="left")
    feature_stability = feature_stability.merge(evidence_summary, on="feature_name", how="left")
    feature_stability["selection_count"] = feature_stability["selection_count"].fillna(0).astype(int)
    feature_stability["selection_frequency"] = feature_stability["selection_count"] / max(n_repeats, 1)

    for col in ["mean_gain_importance", "max_gain_importance", "mean_abs_shap", "max_abs_shap"]:
        if col in feature_stability.columns:
            feature_stability[col] = pd.to_numeric(feature_stability[col], errors="coerce").fillna(0.0)

    feature_stability["gain_score_norm"] = normalized_score(feature_stability["mean_gain_importance"])
    feature_stability["shap_score_norm"] = normalized_score(feature_stability["mean_abs_shap"])
    feature_stability["composite_registry_score"] = (
        feature_stability["selection_frequency"]
        + feature_stability["gain_score_norm"]
        + feature_stability["shap_score_norm"]
    )

    feature_stability = feature_stability.merge(feature_meta, on="feature_name", how="left", suffixes=("", "_meta"))
    feature_stability = attach_biology_theme(feature_stability, feature_col="feature_name")
    feature_stability = feature_stability.sort_values(
        ["composite_registry_score", "selection_frequency", "mean_abs_shap", "mean_gain_importance"],
        ascending=False,
    )

    test_metrics = metrics[metrics["split"] == "test"].copy()

    metric_summary = {}
    if not test_metrics.empty:
        for col in ["pearson", "spearman", "r2", "mae", "rmse", "rmse_improvement_vs_baseline", "mae_improvement_vs_baseline"]:
            vals = pd.to_numeric(test_metrics[col], errors="coerce") if col in test_metrics.columns else pd.Series(dtype=float)
            metric_summary[f"test_{col}_mean"] = float(vals.mean()) if len(vals.dropna()) else np.nan
            metric_summary[f"test_{col}_median"] = float(vals.median()) if len(vals.dropna()) else np.nan
            metric_summary[f"test_{col}_q025"] = float(vals.quantile(0.025)) if len(vals.dropna()) else np.nan
            metric_summary[f"test_{col}_q975"] = float(vals.quantile(0.975)) if len(vals.dropna()) else np.nan

    return {
        "metrics_long": metrics,
        "prediction_long": predictions,
        "feature_evidence_long": evidence_long,
        "feature_stability": feature_stability,
        "metric_summary": metric_summary,
    }


def build_strict_registry(
    feature_stability: pd.DataFrame,
    max_features: int = 150,
    min_selection_frequency: float = 0.34,
    min_composite_score: float = 0.05,
) -> pd.DataFrame:
    df = feature_stability.copy()
    df["selection_frequency"] = pd.to_numeric(df["selection_frequency"], errors="coerce").fillna(0.0)
    df["composite_registry_score"] = pd.to_numeric(df["composite_registry_score"], errors="coerce").fillna(0.0)

    filtered = df[
        (df["selection_frequency"] >= min_selection_frequency)
        | (df["composite_registry_score"] >= min_composite_score)
    ].copy()

    if filtered.empty:
        filtered = df.sort_values("composite_registry_score", ascending=False).head(max_features).copy()
    else:
        filtered = filtered.sort_values("composite_registry_score", ascending=False).head(max_features).copy()

    filtered["include_for_v2_strict_biology_registry"] = True
    filtered["v2_strict_registry_status"] = "generated_from_residual_pair_smoke_evidence"
    filtered["registry_generation_rule"] = (
        f"selection_frequency >= {min_selection_frequency} OR composite_registry_score >= {min_composite_score}; "
        f"top {max_features} by composite score"
    )

    return filtered.reset_index(drop=True)
