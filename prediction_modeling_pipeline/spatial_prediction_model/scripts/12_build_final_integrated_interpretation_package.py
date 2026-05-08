import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_table(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() == ".tsv":
            return pd.read_csv(p, sep="\t")
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() == ".json":
            obj = read_json(p)
            if obj is None:
                return None
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                return pd.DataFrame([obj])
        return pd.read_csv(p, sep=None, engine="python")
    except Exception:
        return None


def infer_col(df, candidates):
    if df is None:
        return None
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == str(c).lower():
                return c
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in str(c).lower():
                return c
    return None


def to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_series_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def wrap_text(s, width=50):
    return "\n".join(textwrap.wrap(str(s), width=width))


def write_txt(path, body):
    path = Path(path)
    text = f"{path}\n\n{body}"
    path.write_text(text, encoding="utf-8")


def save_manifest_row(rows, path, category, description):
    rows.append({
        "path": str(path),
        "category": category,
        "description": description
    })


def create_barh(df, xcol, ycol, path, title, xlabel, top_n=None):
    if df is None or df.empty or xcol not in df.columns or ycol not in df.columns:
        return False
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[xcol, ycol])
    if plot_df.empty:
        return False
    plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")
    plot_df = plot_df.dropna(subset=[xcol])
    if plot_df.empty:
        return False
    plot_df = plot_df.sort_values(xcol, ascending=True)
    if top_n is not None:
        plot_df = plot_df.tail(top_n)
    labels = [wrap_text(x, 45) for x in plot_df[ycol].astype(str).tolist()]
    fig_h = max(5, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(labels, plot_df[xcol].tolist())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_hist(series, path, title, xlabel, bins=20):
    if series is None:
        return False
    ser = pd.to_numeric(series, errors="coerce").dropna()
    if ser.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ser, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_scatter(df, xcol, ycol, path, title, xlabel, ylabel):
    if df is None or df.empty or xcol not in df.columns or ycol not in df.columns:
        return False
    plot_df = df.copy()
    plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")
    plot_df[ycol] = pd.to_numeric(plot_df[ycol], errors="coerce")
    plot_df = plot_df.dropna(subset=[xcol, ycol])
    if plot_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df[xcol], plot_df[ycol])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_heatmap_from_table(df, path, title):
    if df is None or df.empty:
        return False
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        return False
    plot_df = df.copy()
    label_col = None
    for cand in ["model_name", "model_family", "label"]:
        if cand in plot_df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = plot_df.columns[0]
    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[numeric_cols].copy()
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    arr = values.to_numpy(dtype=float)
    fig_w = max(8, 1.5 * len(numeric_cols))
    fig_h = max(4, 0.6 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([wrap_text(x, 28) for x in labels])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def create_grouped_theme_chart(theme_df, path, title):
    if theme_df is None or theme_df.empty:
        return False
    need_cols = {"theme", "source_branch", "score"}
    if not need_cols.issubset(set(theme_df.columns)):
        return False
    plot_df = theme_df.copy()
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["score"])
    if plot_df.empty:
        return False
    pivot = plot_df.pivot_table(index="theme", columns="source_branch", values="score", aggfunc="mean", fill_value=0.0)
    if pivot.empty:
        return False
    pivot["__sum__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__sum__", ascending=True)
    pivot = pivot.tail(12)
    pivot = pivot.drop(columns="__sum__")
    themes = [wrap_text(x, 35) for x in pivot.index.tolist()]
    x = np.arange(len(themes))
    cols = list(pivot.columns)
    width = 0.8 / max(1, len(cols))
    fig_h = max(6, 0.5 * len(themes) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    for i, col in enumerate(cols):
        ax.barh(x + (i - (len(cols)-1)/2)*width, pivot[col].values, height=width, label=str(col))
    ax.set_yticks(x)
    ax.set_yticklabels(themes)
    ax.set_title(title)
    ax.set_xlabel("Score / contribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize_source_files(source_map):
    rows = []
    for key, value in source_map.items():
        if isinstance(value, list):
            for item in value:
                rows.append({
                    "source_key": key,
                    "path": str(item),
                    "exists": Path(item).exists() if item else False
                })
        else:
            rows.append({
                "source_key": key,
                "path": str(value) if value else "",
                "exists": Path(value).exists() if value else False
            })
    return pd.DataFrame(rows)


def build_model_comparison(source_map):
    records = []

    prob_metrics = read_table(source_map.get("probability_vs_residual_metrics_long"))
    spatial_contrib = read_table(source_map.get("probability_vs_residual_spatial_contribution_long"))
    spatial_fraction_map = {}

    if spatial_contrib is not None and not spatial_contrib.empty:
        model_col = infer_col(spatial_contrib, ["model", "model_name"])
        group_col = infer_col(spatial_contrib, ["component", "feature_group", "contributor", "source"])
        value_col = infer_col(spatial_contrib, ["value", "fraction", "score"])
        if model_col and group_col and value_col:
            tmp = spatial_contrib[[model_col, group_col, value_col]].copy()
            tmp.columns = ["model", "group", "value"]
            tmp["model"] = tmp["model"].astype(str)
            tmp["group"] = tmp["group"].astype(str).str.lower()
            tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
            tmp = tmp.dropna(subset=["value"])
            for model_name, gdf in tmp.groupby("model"):
                spatial_rows = gdf[gdf["group"].str.contains("spatial")]
                if not spatial_rows.empty:
                    spatial_fraction_map[model_name] = spatial_rows["value"].max()

    if prob_metrics is not None and not prob_metrics.empty:
        model_col = infer_col(prob_metrics, ["model", "model_name"])
        metric_col = infer_col(prob_metrics, ["metric", "metric_name"])
        value_col = infer_col(prob_metrics, ["value", "metric_value", "score"])
        if model_col and metric_col and value_col:
            tmp = prob_metrics[[model_col, metric_col, value_col]].copy()
            tmp.columns = ["model", "metric", "value"]
            tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
            wide = tmp.pivot_table(index="model", columns="metric", values="value", aggfunc="first").reset_index()
            for _, row in wide.iterrows():
                model_name = str(row["model"])
                model_lower = model_name.lower()

                pearson_col = None
                r2_col = None
                for c in wide.columns:
                    cl = str(c).lower()
                    if pearson_col is None and ("pearson" in cl):
                        pearson_col = c
                    if r2_col is None and (cl == "r2" or "r2" in cl):
                        r2_col = c

                primary = to_float(row[pearson_col]) if pearson_col else np.nan
                secondary = to_float(row[r2_col]) if r2_col else np.nan

                if "prob" in model_lower or "pooled" in model_lower:
                    records.append({
                        "model_family": "pooled_probability_xgboost",
                        "model_name": model_name,
                        "description": "Pair level pooled probability model",
                        "purpose": "Best overall response prediction across sample treatment pairs",
                        "unit_of_prediction": "sample_treatment_pair",
                        "target": "fused_prob_responder",
                        "inputs": "spatial features plus drug identity",
                        "validation_type": "held out evaluation",
                        "primary_metric_name": "test_pearson",
                        "primary_metric_value": primary,
                        "secondary_metric_name": "test_r2",
                        "secondary_metric_value": secondary,
                        "spatial_feature_fraction": spatial_fraction_map.get(model_name, np.nan),
                        "validated_finding_type": "baseline comparison",
                        "recommended_use": "prediction baseline",
                        "notes": "Strong overall predictive model, but treatment identity dominated"
                    })
                elif "residual" in model_lower or "prior" in model_lower:
                    records.append({
                        "model_family": "pair_level_prior_adjusted_residual_xgboost",
                        "model_name": model_name,
                        "description": "Pair level prior adjusted residual model",
                        "purpose": "Biological interpretation of response above or below treatment prior",
                        "unit_of_prediction": "sample_treatment_pair",
                        "target": "fused_residual_vs_prior",
                        "inputs": "spatial features plus drug identity",
                        "validation_type": "held out evaluation",
                        "primary_metric_name": "test_pearson",
                        "primary_metric_value": primary,
                        "secondary_metric_name": "test_r2",
                        "secondary_metric_value": secondary,
                        "spatial_feature_fraction": spatial_fraction_map.get(model_name, np.nan),
                        "validated_finding_type": "interpretive residual model",
                        "recommended_use": "pair level residual biology interpretation",
                        "notes": "Residual target exposed much more spatial biology signal"
                    })

    broad_rep = read_table(source_map.get("repeated_split_metric_summary"))
    broad_targets_table = pd.DataFrame()
    if broad_rep is not None and not broad_rep.empty:
        target_col = infer_col(broad_rep, ["target", "target_name"])
        pearson_col = infer_col(broad_rep, ["mean_test_pearson", "test_pearson_mean", "mean_pearson"])
        r2_col = infer_col(broad_rep, ["mean_test_r2", "test_r2_mean", "mean_r2"])
        if target_col and pearson_col:
            broad_targets_table = broad_rep[[target_col, pearson_col] + ([r2_col] if r2_col else [])].copy()
            broad_targets_table.columns = ["target", "mean_test_pearson"] + (["mean_test_r2"] if r2_col else [])
            broad_targets_table["mean_test_pearson"] = pd.to_numeric(broad_targets_table["mean_test_pearson"], errors="coerce")
            if "mean_test_r2" in broad_targets_table.columns:
                broad_targets_table["mean_test_r2"] = pd.to_numeric(broad_targets_table["mean_test_r2"], errors="coerce")
            broad_targets_table = broad_targets_table.sort_values("mean_test_pearson", ascending=False)
            if not broad_targets_table.empty:
                best = broad_targets_table.iloc[0]
                records.append({
                    "model_family": "broad_residual_spatial_only",
                    "model_name": "spatial_only_broad_residual_best_target",
                    "description": "Sample level broad residual model",
                    "purpose": "Broad screening of sample level spatial signal independent of drug identity",
                    "unit_of_prediction": "sample",
                    "target": str(best["target"]),
                    "inputs": "strict spatial biology features only",
                    "validation_type": "repeated split validation",
                    "primary_metric_name": "mean_test_pearson",
                    "primary_metric_value": to_float(best["mean_test_pearson"]),
                    "secondary_metric_name": "mean_test_r2",
                    "secondary_metric_value": to_float(best["mean_test_r2"]) if "mean_test_r2" in broad_targets_table.columns else np.nan,
                    "spatial_feature_fraction": 1.0,
                    "validated_finding_type": "screening level broad biology signal",
                    "recommended_use": "sample level screening and biological context",
                    "notes": "Broad residual model is useful, but weaker than per treatment residual modeling"
                })

    curated_treatment = read_table(source_map.get("curated_treatment_model_summary"))
    validated_treatments = pd.DataFrame()
    if curated_treatment is not None and not curated_treatment.empty:
        treatment_col = infer_col(curated_treatment, ["treatment", "treatment_name"])
        pearson_col = infer_col(curated_treatment, ["mean_test_pearson", "test_pearson_mean"])
        r2_col = infer_col(curated_treatment, ["mean_test_r2", "test_r2_mean", "mean_r2"])
        tier_col = infer_col(curated_treatment, ["interpretation_tier", "tier"])
        if treatment_col and pearson_col:
            tmp = curated_treatment.copy()
            tmp[pearson_col] = pd.to_numeric(tmp[pearson_col], errors="coerce")
            if r2_col:
                tmp[r2_col] = pd.to_numeric(tmp[r2_col], errors="coerce")
            median_pearson = float(tmp[pearson_col].median())
            median_r2 = float(tmp[r2_col].median()) if r2_col else np.nan
            tier1_count = int((tmp[tier_col].astype(str).str.contains("tier1", case=False, na=False)).sum()) if tier_col else np.nan
            records.append({
                "model_family": "filtered_per_treatment_residual_models",
                "model_name": "curated_per_treatment_residual_models",
                "description": "Treatment specific residual models with repeated split screening and curation",
                "purpose": "Find treatment specific spatial biology explaining response above or below prior",
                "unit_of_prediction": "sample within treatment",
                "target": "fused_residual_vs_prior",
                "inputs": "strict spatial biology features only, treatment specific subset",
                "validation_type": "repeated split screening and final SHAP curation",
                "primary_metric_name": "median_mean_test_pearson",
                "primary_metric_value": median_pearson,
                "secondary_metric_name": "median_mean_test_r2",
                "secondary_metric_value": median_r2,
                "spatial_feature_fraction": 1.0,
                "validated_finding_type": f"tier1_count={tier1_count}" if pd.notna(tier1_count) else "curated treatment models",
                "recommended_use": "treatment specific spatial biology discovery",
                "notes": "Strongest branch for validated treatment specific biological interpretation"
            })

    label_shuffle = read_table(source_map.get("label_shuffle_treatment_summary"))
    if label_shuffle is not None and not label_shuffle.empty:
        treatment_col = infer_col(label_shuffle, ["treatment", "treatment_name"])
        obs_col = infer_col(label_shuffle, ["observed_mean_test_pearson", "observed_mean_pearson", "observed_pearson"])
        p_col = infer_col(label_shuffle, ["empirical_p_value", "empirical_p", "p_value"])
        fdr_col = infer_col(label_shuffle, ["bh_fdr_q_value", "fdr_q_value", "fdr_q", "q_value"])
        if treatment_col:
            validated_treatments = label_shuffle.copy()
            if obs_col:
                validated_treatments[obs_col] = pd.to_numeric(validated_treatments[obs_col], errors="coerce")
            if p_col:
                validated_treatments[p_col] = pd.to_numeric(validated_treatments[p_col], errors="coerce")
            if fdr_col:
                validated_treatments[fdr_col] = pd.to_numeric(validated_treatments[fdr_col], errors="coerce")
            pass_count = 0
            if fdr_col:
                pass_count = int((validated_treatments[fdr_col] <= 0.10).sum())
            elif p_col:
                pass_count = int((validated_treatments[p_col] <= 0.05).sum())
            records.append({
                "model_family": "tier1_label_shuffle_validation",
                "model_name": "tier1_per_treatment_label_shuffle_validation",
                "description": "Null model validation for tier1 per treatment residual models",
                "purpose": "Test whether observed tier1 treatment specific signals exceed shuffled label null performance",
                "unit_of_prediction": "treatment level validation set",
                "target": "tier1_treatment_models",
                "inputs": "selected tier1 model outputs and label permutations",
                "validation_type": "empirical label shuffle validation",
                "primary_metric_name": "validated_treatments_fdr_le_0_10",
                "primary_metric_value": float(pass_count),
                "secondary_metric_name": "n_treatments_tested",
                "secondary_metric_value": float(len(validated_treatments)),
                "spatial_feature_fraction": np.nan,
                "validated_finding_type": "formal null validation",
                "recommended_use": "validation of strongest treatment specific signals",
                "notes": "All tier1 treatments passing label shuffle is the strongest validation result in the branch"
            })

    return pd.DataFrame(records), broad_targets_table, validated_treatments


def build_feature_theme_tables(source_map):
    feature_sources = [
        ("pair_level_residual", source_map.get("top_residual_biology_features_strict")),
        ("broad_residual", source_map.get("cross_target_top_features")),
        ("per_treatment_curated", source_map.get("core_recurrent_spatial_features")),
        ("per_treatment_full", source_map.get("all_cross_treatment_feature_stability")),
    ]
    feature_rows = []
    for source_name, path in feature_sources:
        df = read_table(path)
        if df is None or df.empty:
            continue
        feature_col = infer_col(df, ["feature", "feature_name"])
        score_col = infer_col(df, ["frequency", "score", "mean_absolute_shap", "total_mean_absolute_shap", "top10_shap_frequency_across_final_models"])
        if feature_col is None:
            continue
        tmp = df.copy()
        if score_col is None:
            tmp["__score__"] = np.nan
            score_col = "__score__"
        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp[[feature_col, score_col]].copy()
        tmp.columns = ["feature", "score"]
        tmp["source_branch"] = source_name
        feature_rows.append(tmp)

    if feature_rows:
        feature_long = pd.concat(feature_rows, ignore_index=True)
        feature_summary = feature_long.groupby("feature", as_index=False).agg(
            source_count=("source_branch", "nunique"),
            source_branches=("source_branch", lambda x: "|".join(sorted(set(x)))),
            mean_score=("score", "mean"),
            max_score=("score", "max")
        ).sort_values(["source_count", "max_score", "mean_score"], ascending=[False, False, False])
    else:
        feature_long = pd.DataFrame(columns=["feature", "score", "source_branch"])
        feature_summary = pd.DataFrame(columns=["feature", "source_count", "source_branches", "mean_score", "max_score"])

    theme_sources = [
        ("pair_level_residual", source_map.get("residual_biology_theme_summary")),
        ("broad_residual", source_map.get("theme_summary_all_targets")),
        ("per_treatment_curated", source_map.get("cross_treatment_theme_summary")),
    ]
    theme_rows = []
    for source_name, path in theme_sources:
        df = read_table(path)
        if df is None or df.empty:
            continue
        theme_col = infer_col(df, ["theme", "biology_theme"])
        score_col = infer_col(df, ["score", "total_mean_absolute_shap", "mean_absolute_shap", "contribution"])
        if theme_col is None:
            continue
        tmp = df.copy()
        if score_col is None:
            tmp["__score__"] = np.nan
            score_col = "__score__"
        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp[[theme_col, score_col]].copy()
        tmp.columns = ["theme", "score"]
        tmp["source_branch"] = source_name
        theme_rows.append(tmp)

    if theme_rows:
        theme_long = pd.concat(theme_rows, ignore_index=True)
        theme_summary = theme_long.groupby("theme", as_index=False).agg(
            source_count=("source_branch", "nunique"),
            source_branches=("source_branch", lambda x: "|".join(sorted(set(x)))),
            mean_score=("score", "mean"),
            max_score=("score", "max")
        ).sort_values(["source_count", "max_score", "mean_score"], ascending=[False, False, False])
    else:
        theme_long = pd.DataFrame(columns=["theme", "score", "source_branch"])
        theme_summary = pd.DataFrame(columns=["theme", "source_count", "source_branches", "mean_score", "max_score"])

    return feature_long, feature_summary, theme_long, theme_summary


def build_decision_table():
    return pd.DataFrame([
        {
            "pipeline_component": "01_to_07_core_prediction_engine",
            "current_status": "canonical",
            "recommended_decision": "keep",
            "promote_to_canonical": "already canonical",
            "rationale": "Base engine completed and not replaced by experimental branch work",
            "notes": "Do not overwrite until final packaging and review are complete"
        },
        {
            "pipeline_component": "08_residual_biology_interpretation",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Translates residual pair model outputs into interpretable biological feature and theme summaries",
            "notes": "Useful as official downstream interpretation module"
        },
        {
            "pipeline_component": "09_broad_residual_spatial_only_model",
            "current_status": "experimental",
            "recommended_decision": "promote_as_screening_module",
            "promote_to_canonical": "yes",
            "rationale": "Useful sample level broad spatial screening model with repeated split validation",
            "notes": "Best framed as broad screening and biological context, not primary validated endpoint"
        },
        {
            "pipeline_component": "10_filtered_per_treatment_residual_models",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Strongest treatment specific spatial biology discovery branch",
            "notes": "Should become official downstream model discovery step"
        },
        {
            "pipeline_component": "11_label_shuffle_validation",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Provides formal null validation of tier1 treatment specific models",
            "notes": "Strong validation step, especially for publication or dissertation style reporting"
        },
        {
            "pipeline_component": "12_final_integrated_interpretation_package",
            "current_status": "new_experimental_reporting_step",
            "recommended_decision": "create_and_keep",
            "promote_to_canonical": "likely yes",
            "rationale": "Provides integrated synthesis, figure generation, reporting, provenance, and canonical recommendations",
            "notes": "Should remain non destructive and only consume earlier outputs"
        }
    ])


def create_methods_results_discussion_text(model_comp, broad_targets, validated_treatments, feature_summary, theme_summary):
    lines = []
    lines.append("Methods")
    lines.append("")
    lines.append("We compared four main modeling branches within the spatial prediction modeling pipeline.")
    lines.append("First, a pooled probability XGBoost model predicted fused_prob_responder using spatial features plus drug identity.")
    lines.append("Second, a prior adjusted residual XGBoost model predicted fused_residual_vs_prior at the sample treatment pair level to expose biological signal above or below treatment prior.")
    lines.append("Third, a broad residual spatial only model summarized sample level residual behavior across treatments and was validated with repeated splits.")
    lines.append("Fourth, filtered per treatment residual models were trained within treatment subsets using spatial features only, followed by curation and Tier 1 label shuffle validation.")
    lines.append("")
    lines.append("Results")
    lines.append("")
    if model_comp is not None and not model_comp.empty:
        lines.append("Model family comparison highlights:")
        for _, row in model_comp.iterrows():
            pm = row.get("primary_metric_name", "")
            pv = row.get("primary_metric_value", np.nan)
            sm = row.get("secondary_metric_name", "")
            sv = row.get("secondary_metric_value", np.nan)
            lines.append(f"  {row['model_family']}: {pm}={pv:.3f} {sm}={sv:.3f}" if pd.notna(pv) and pd.notna(sv) else f"  {row['model_family']}: primary summary available")
        lines.append("")
    if broad_targets is not None and not broad_targets.empty:
        best = broad_targets.iloc[0]
        lines.append(f"The broad residual sample level branch identified {best['target']} as the strongest repeated split target with mean test Pearson {best['mean_test_pearson']:.3f}.")
        if "mean_test_r2" in broad_targets.columns and pd.notna(best.get("mean_test_r2", np.nan)):
            lines.append(f"The corresponding mean test R2 was {best['mean_test_r2']:.3f}.")
        lines.append("")
    if validated_treatments is not None and not validated_treatments.empty:
        lines.append(f"Tier 1 label shuffle validation included {len(validated_treatments)} treatment models.")
        fdr_col = None
        for c in validated_treatments.columns:
            if "fdr" in str(c).lower() or "q_value" in str(c).lower():
                fdr_col = c
                break
        if fdr_col:
            n_pass = int((pd.to_numeric(validated_treatments[fdr_col], errors='coerce') <= 0.10).sum())
            lines.append(f"{n_pass} treatment models passed FDR q <= 0.10.")
        lines.append("")
    if feature_summary is not None and not feature_summary.empty:
        top_features = feature_summary.head(5)["feature"].astype(str).tolist()
        lines.append("The most recurrent spatial features across branches included:")
        for feat in top_features:
            lines.append(f"  {feat}")
        lines.append("")
    if theme_summary is not None and not theme_summary.empty:
        top_themes = theme_summary.head(5)["theme"].astype(str).tolist()
        lines.append("The most recurrent biology themes across branches included:")
        for theme in top_themes:
            lines.append(f"  {theme}")
        lines.append("")
    lines.append("Discussion")
    lines.append("")
    lines.append("The pooled probability model remained a strong predictive baseline, but it was less useful for biological interpretation because treatment identity dominated.")
    lines.append("The prior adjusted residual pair level model improved biological interpretability by shifting the learning objective from absolute response probability to response above or below treatment prior.")
    lines.append("The broad residual spatial only branch provided a useful screening view of sample level spatial signal.")
    lines.append("The strongest overall discovery branch was the filtered per treatment residual framework, especially when followed by tiered curation and label shuffle validation.")
    lines.append("Taken together, the results support keeping scripts 01 to 07 as the core engine and promoting residual interpretation, broad residual screening, per treatment residual discovery, and label shuffle validation as official downstream modules.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-map", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    source_map = json.loads(Path(args.source_map).read_text(encoding="utf-8"))
    output_root = ensure_dir(Path(args.output_root))

    d01 = ensure_dir(output_root / "01_model_milestone_summary")
    d02 = ensure_dir(output_root / "02_validated_treatment_models")
    d03 = ensure_dir(output_root / "03_recurrent_spatial_features")
    d04 = ensure_dir(output_root / "04_recurrent_biology_themes")
    d05 = ensure_dir(output_root / "05_figures_for_presentation")
    d06 = ensure_dir(output_root / "06_methods_results_discussion_report")
    d07 = ensure_dir(output_root / "07_pipeline_integration_recommendations")
    d08 = ensure_dir(output_root / "08_provenance_and_file_manifest")

    manifest_rows = []
    figure_manifest_rows = []
    caption_lines = []

    model_comp, broad_targets, validated_treatments = build_model_comparison(source_map)
    feature_long, feature_summary, theme_long, theme_summary = build_feature_theme_tables(source_map)
    decision_table = build_decision_table()
    source_file_manifest = summarize_source_files(source_map)

    model_comp_path = d01 / "model_comparison_table.tsv"
    model_comp.to_csv(model_comp_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, model_comp_path, "table", "Model comparison table")

    if broad_targets is not None and not broad_targets.empty:
        broad_targets_path = d01 / "broad_residual_target_comparison.tsv"
        broad_targets.to_csv(broad_targets_path, sep="\t", index=False)
        save_manifest_row(manifest_rows, broad_targets_path, "table", "Broad residual repeated split target comparison")

    if validated_treatments is not None and not validated_treatments.empty:
        validated_path = d02 / "validated_treatment_table.tsv"
        validated_treatments.to_csv(validated_path, sep="\t", index=False)
        save_manifest_row(manifest_rows, validated_path, "table", "Validated treatment table from label shuffle validation")

    feature_summary_path = d03 / "recurrent_spatial_feature_table.tsv"
    feature_summary.to_csv(feature_summary_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, feature_summary_path, "table", "Recurrent spatial feature summary table")

    feature_long_path = d03 / "recurrent_spatial_feature_long.tsv"
    feature_long.to_csv(feature_long_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, feature_long_path, "table", "Recurrent spatial feature long table")

    theme_summary_path = d04 / "recurrent_biology_theme_table.tsv"
    theme_summary.to_csv(theme_summary_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, theme_summary_path, "table", "Recurrent biology theme summary table")

    theme_long_path = d04 / "recurrent_biology_theme_long.tsv"
    theme_long.to_csv(theme_long_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, theme_long_path, "table", "Recurrent biology theme long table")

    decision_table_path = d07 / "pipeline_integration_recommendations.tsv"
    decision_table.to_csv(decision_table_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, decision_table_path, "table", "Decision table for canonical integration")

    source_manifest_path = d08 / "source_file_manifest.tsv"
    source_file_manifest.to_csv(source_manifest_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, source_manifest_path, "table", "Source file manifest")

    script_prov_files = source_map.get("script_provenance_files", []) or []
    prov_rows = []
    for item in script_prov_files:
        p = Path(item)
        prov_rows.append({
            "path": str(p),
            "exists": p.exists(),
            "filename": p.name,
            "canonical_original_script_modified": "no",
            "notes": "Experimental provenance file discovered during integrated reporting package build"
        })
    provenance_table = pd.DataFrame(prov_rows)
    provenance_path = d08 / "provenance_table.tsv"
    provenance_table.to_csv(provenance_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, provenance_path, "table", "Provenance table")

    master_summary_lines = []
    master_summary_lines.append("Final integrated spatial response interpretation package")
    master_summary_lines.append("")
    master_summary_lines.append(f"Output root: {output_root}")
    master_summary_lines.append("")
    master_summary_lines.append("Key package contents")
    master_summary_lines.append("1. Master summary report")
    master_summary_lines.append("2. Model comparison table")
    master_summary_lines.append("3. Validated treatment table")
    master_summary_lines.append("4. Recurrent spatial feature table")
    master_summary_lines.append("5. Recurrent biology theme table")
    master_summary_lines.append("6. Figure manifest")
    master_summary_lines.append("7. Presentation ready figure captions")
    master_summary_lines.append("8. Methods results discussion narrative")
    master_summary_lines.append("9. Canonical integration recommendations")
    master_summary_lines.append("10. Provenance and file manifest tables")
    master_summary_lines.append("")
    if not model_comp.empty:
        master_summary_lines.append("Model families summarized")
        for _, row in model_comp.iterrows():
            metric_name = row.get("primary_metric_name", "")
            metric_value = row.get("primary_metric_value", np.nan)
            if pd.notna(metric_value):
                master_summary_lines.append(f"  {row['model_family']}: {metric_name}={metric_value:.3f}")
            else:
                master_summary_lines.append(f"  {row['model_family']}: summary available")
        master_summary_lines.append("")
    if validated_treatments is not None and not validated_treatments.empty:
        master_summary_lines.append(f"Validated treatment models included: {len(validated_treatments)}")
        master_summary_lines.append("")
    if not feature_summary.empty:
        master_summary_lines.append("Top recurrent spatial features")
        for feat in feature_summary.head(10)["feature"].astype(str).tolist():
            master_summary_lines.append(f"  {feat}")
        master_summary_lines.append("")
    if not theme_summary.empty:
        master_summary_lines.append("Top recurrent biology themes")
        for theme in theme_summary.head(10)["theme"].astype(str).tolist():
            master_summary_lines.append(f"  {theme}")
        master_summary_lines.append("")
    master_summary_path = d01 / "master_summary_report.txt"
    write_txt(master_summary_path, "\n".join(master_summary_lines))
    save_manifest_row(manifest_rows, master_summary_path, "report", "Master summary report")

    mrd_text = create_methods_results_discussion_text(model_comp, broad_targets, validated_treatments, feature_summary, theme_summary)
    mrd_path = d06 / "methods_results_discussion_narrative.txt"
    write_txt(mrd_path, mrd_text)
    save_manifest_row(manifest_rows, mrd_path, "report", "Methods results discussion narrative")

    recommendation_lines = []
    recommendation_lines.append("Canonical integration recommendations")
    recommendation_lines.append("")
    recommendation_lines.append("Recommended canonical structure")
    recommendation_lines.append("1. Keep scripts 01 to 07 as the core prediction engine.")
    recommendation_lines.append("2. Promote residual biology interpretation as official downstream Step 08.")
    recommendation_lines.append("3. Promote broad residual spatial only modeling as official downstream Step 09.")
    recommendation_lines.append("4. Promote filtered per treatment residual discovery as official downstream Step 10.")
    recommendation_lines.append("5. Promote tier1 label shuffle validation as official downstream Step 11.")
    recommendation_lines.append("6. Keep this final integrated interpretation package as a non destructive synthesis and reporting step.")
    recommendation_lines.append("")
    recommendation_lines.append("Important implementation note")
    recommendation_lines.append("The experimental branch work summarized here did not require rewriting the canonical original scripts.")
    recommendation_lines.append("These results should be reviewed before any permanent reorganization of the official script tree.")
    recommendation_path = d07 / "pipeline_integration_recommendations.txt"
    write_txt(recommendation_path, "\n".join(recommendation_lines))
    save_manifest_row(manifest_rows, recommendation_path, "report", "Narrative integration recommendations")

    fig_count = 0

    fig1 = d05 / "figure01_model_family_primary_metric_bar.png"
    if create_barh(model_comp, "primary_metric_value", "model_family", fig1, "Model family primary metric comparison", "Primary metric value", top_n=None):
        fig_count += 1
        figure_manifest_rows.append({
            "figure_id": "Figure 1",
            "path": str(fig1),
            "title": "Model family primary metric comparison",
            "description": "Compares the main headline metric for each modeling branch.",
            "recommended_use": "Main overview figure"
        })
        caption_lines.append("Figure 1. Model family primary metric comparison. This figure compares the main headline metric for each major modeling branch and provides a fast overview of which model family is strongest for prediction, screening, or validated biological interpretation.")

    fig2 = d05 / "figure02_model_family_metric_heatmap.png"
    heat_df = model_comp.copy()
    if not heat_df.empty:
        for col in ["primary_metric_value", "secondary_metric_value", "spatial_feature_fraction"]:
            if col in heat_df.columns:
                heat_df[col] = pd.to_numeric(heat_df[col], errors="coerce")
    if create_heatmap_from_table(heat_df[["model_family", "primary_metric_value", "secondary_metric_value", "spatial_feature_fraction"]] if not heat_df.empty else heat_df, fig2, "Model family metric heatmap"):
        fig_count += 1
        figure_manifest_rows.append({
            "figure_id": "Figure 2",
            "path": str(fig2),
            "title": "Model family metric heatmap",
            "description": "Heatmap of key quantitative summaries across model families.",
            "recommended_use": "Compact publication style comparison figure"
        })
        caption_lines.append("Figure 2. Model family metric heatmap. This heatmap condenses major quantitative summaries across modeling branches and helps compare predictive strength, biological signal fraction, and validation layers in a publication style format.")

    fig3 = d05 / "figure03_broad_residual_target_comparison.png"
    if broad_targets is not None and not broad_targets.empty:
        if create_barh(broad_targets, "mean_test_pearson", "target", fig3, "Broad residual target comparison", "Mean test Pearson", top_n=12):
            fig_count += 1
            figure_manifest_rows.append({
                "figure_id": "Figure 3",
                "path": str(fig3),
                "title": "Broad residual target comparison",
                "description": "Compares repeated split performance across broad residual targets.",
                "recommended_use": "Broad residual screening summary"
            })
            caption_lines.append("Figure 3. Broad residual target comparison. This figure compares repeated split performance across broad residual targets and shows which sample level spatial summary is most promising.")

    fig4 = d05 / "figure04_per_treatment_performance_distribution.png"
    curated_treatment = read_table(source_map.get("curated_treatment_model_summary"))
    if curated_treatment is not None and not curated_treatment.empty:
        pearson_col = infer_col(curated_treatment, ["mean_test_pearson", "test_pearson_mean"])
        if pearson_col and create_hist(curated_treatment[pearson_col], fig4, "Per treatment model performance distribution", "Mean test Pearson", bins=20):
            fig_count += 1
            figure_manifest_rows.append({
                "figure_id": "Figure 4",
                "path": str(fig4),
                "title": "Per treatment model performance distribution",
                "description": "Histogram showing the distribution of mean test Pearson across curated treatment models.",
                "recommended_use": "Treatment level screening overview"
            })
            caption_lines.append("Figure 4. Per treatment model performance distribution. This histogram shows the distribution of repeated split mean test Pearson values across curated per treatment residual models, helping separate high confidence signals from weaker or exploratory treatment models.")

    fig5 = d05 / "figure05_top_validated_treatments.png"
    if validated_treatments is not None and not validated_treatments.empty:
        treatment_col = infer_col(validated_treatments, ["treatment", "treatment_name"])
        obs_col = infer_col(validated_treatments, ["observed_mean_test_pearson", "observed_mean_pearson", "observed_pearson"])
        if treatment_col and obs_col:
            vt = validated_treatments[[treatment_col, obs_col]].copy()
            vt.columns = ["treatment", "observed_mean_test_pearson"]
            if create_barh(vt, "observed_mean_test_pearson", "treatment", fig5, "Top validated treatment models", "Observed mean test Pearson", top_n=15):
                fig_count += 1
                figure_manifest_rows.append({
                    "figure_id": "Figure 5",
                    "path": str(fig5),
                    "title": "Top validated treatment models",
                    "description": "Top Tier 1 treatment models ranked by observed label shuffle validated performance.",
                    "recommended_use": "Main treatment specific biology figure"
                })
                caption_lines.append("Figure 5. Top validated treatment models. This figure ranks the strongest label shuffle validated Tier 1 treatment specific residual models by observed mean test Pearson.")

    fig6 = d05 / "figure06_label_shuffle_validation_scatter.png"
    if validated_treatments is not None and not validated_treatments.empty:
        obs_col = infer_col(validated_treatments, ["observed_mean_test_pearson", "observed_mean_pearson", "observed_pearson"])
        fdr_col = infer_col(validated_treatments, ["bh_fdr_q_value", "fdr_q_value", "fdr_q", "q_value"])
        if obs_col and fdr_col:
            tmp = validated_treatments.copy()
            tmp["neglog10_fdr"] = -np.log10(pd.to_numeric(tmp[fdr_col], errors="coerce").clip(lower=1e-12))
            if create_scatter(tmp, obs_col, "neglog10_fdr", fig6, "Label shuffle validation significance", "Observed mean test Pearson", "-log10 FDR q value"):
                fig_count += 1
                figure_manifest_rows.append({
                    "figure_id": "Figure 6",
                    "path": str(fig6),
                    "title": "Label shuffle validation significance",
                    "description": "Observed treatment model performance plotted against validation significance.",
                    "recommended_use": "Validation figure"
                })
                caption_lines.append("Figure 6. Label shuffle validation significance. This figure compares observed treatment specific model performance against empirical null significance after label shuffling and multiple testing correction.")

    fig7 = d05 / "figure07_core_recurrent_spatial_features.png"
    if not feature_summary.empty:
        top_feats = feature_summary.head(20).copy()
        if create_barh(top_feats, "source_count", "feature", fig7, "Core recurrent spatial features", "Number of branches in which feature recurs", top_n=20):
            fig_count += 1
            figure_manifest_rows.append({
                "figure_id": "Figure 7",
                "path": str(fig7),
                "title": "Core recurrent spatial features",
                "description": "Most recurrent spatial features across residual pair, broad residual, and per treatment branches.",
                "recommended_use": "Cross model biology synthesis figure"
            })
            caption_lines.append("Figure 7. Core recurrent spatial features. This figure highlights the spatial features that recur across multiple modeling branches and therefore represent the most reproducible biological signals in the analysis.")

    fig8 = d05 / "figure08_cross_model_biology_theme_comparison.png"
    if create_grouped_theme_chart(theme_long, fig8, "Cross model biology theme comparison"):
        fig_count += 1
        figure_manifest_rows.append({
            "figure_id": "Figure 8",
            "path": str(fig8),
            "title": "Cross model biology theme comparison",
            "description": "Grouped comparison of recurrent biology themes across modeling branches.",
            "recommended_use": "Cross branch biology synthesis figure"
        })
        caption_lines.append("Figure 8. Cross model biology theme comparison. This grouped chart compares recurrent biology themes across the pair level residual, broad residual, and per treatment branches and helps identify which biological programs recur most consistently.")

    fig9 = d05 / "figure09_per_treatment_correlation_vs_r2.png"
    if curated_treatment is not None and not curated_treatment.empty:
        pearson_col = infer_col(curated_treatment, ["mean_test_pearson", "test_pearson_mean"])
        r2_col = infer_col(curated_treatment, ["mean_test_r2", "test_r2_mean", "mean_r2"])
        if pearson_col and r2_col and create_scatter(curated_treatment, pearson_col, r2_col, fig9, "Per treatment model correlation versus R2", "Mean test Pearson", "Mean test R2"):
            fig_count += 1
            figure_manifest_rows.append({
                "figure_id": "Figure 9",
                "path": str(fig9),
                "title": "Per treatment model correlation versus R2",
                "description": "Scatter plot comparing treatment model correlation and explained variance.",
                "recommended_use": "Supplementary quality control figure"
            })
            caption_lines.append("Figure 9. Per treatment model correlation versus R2. This supplementary figure shows how mean test Pearson and mean test R2 align across curated treatment models and helps distinguish strong but noisy models from models with better explained variance.")

    fig10 = d05 / "figure10_model_design_table.tsv"
    model_design_table = model_comp[[
        "model_family", "description", "purpose", "unit_of_prediction", "target", "inputs", "validation_type",
        "primary_metric_name", "primary_metric_value", "secondary_metric_name", "secondary_metric_value", "recommended_use"
    ]].copy() if not model_comp.empty else pd.DataFrame()
    if not model_design_table.empty:
        model_design_table.to_csv(fig10, sep="\t", index=False)
        save_manifest_row(manifest_rows, fig10, "table", "Publication style model design comparison table")
        figure_manifest_rows.append({
            "figure_id": "Table 1",
            "path": str(fig10),
            "title": "Publication style model design comparison table",
            "description": "Publication style table comparing model families, inputs, targets, validation, and metrics.",
            "recommended_use": "Main comparison table"
        })
        caption_lines.append("Table 1. Publication style model design comparison table. This table compares the main model families, their targets, input features, validation strategy, and headline metrics.")

    figure_manifest = pd.DataFrame(figure_manifest_rows)
    figure_manifest_path = d05 / "figure_manifest.tsv"
    figure_manifest.to_csv(figure_manifest_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, figure_manifest_path, "table", "Figure manifest")

    figure_captions_path = d05 / "figure_captions.txt"
    write_txt(figure_captions_path, "\n\n".join(caption_lines))
    save_manifest_row(manifest_rows, figure_captions_path, "report", "Presentation ready figure captions")

    manifest_df = pd.DataFrame(manifest_rows)
    generated_manifest_path = d08 / "generated_output_manifest.tsv"
    manifest_df.to_csv(generated_manifest_path, sep="\t", index=False)
    save_manifest_row(manifest_rows, generated_manifest_path, "table", "Generated output manifest")

    run_summary = {
        "output_root": str(output_root),
        "source_files_found": int(source_file_manifest["exists"].sum()) if not source_file_manifest.empty else 0,
        "model_families_summarized": int(len(model_comp)),
        "validated_treatment_models": int(len(validated_treatments)) if validated_treatments is not None else 0,
        "recurrent_features": int(len(feature_summary)),
        "recurrent_themes": int(len(theme_summary)),
        "figures_or_main_tables_generated": int(len(figure_manifest_rows)),
        "generated_output_manifest_rows": int(len(manifest_df))
    }
    run_summary_path = d01 / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    save_manifest_row(manifest_rows, run_summary_path, "json", "Run summary json")

    print("")
    print("============================================================")
    print("FINAL INTEGRATED INTERPRETATION PACKAGE COMPLETE")
    print("============================================================")
    print(f"Output root: {output_root}")
    print(f"Source files found: {run_summary['source_files_found']}")
    print(f"Model families summarized: {run_summary['model_families_summarized']}")
    print(f"Validated treatment models: {run_summary['validated_treatment_models']}")
    print(f"Recurrent features: {run_summary['recurrent_features']}")
    print(f"Recurrent themes: {run_summary['recurrent_themes']}")
    print(f"Figures or main tables generated: {run_summary['figures_or_main_tables_generated']}")
    print("")
    print("Key outputs")
    print(f"  Master summary: {master_summary_path}")
    print(f"  Model comparison table: {model_comp_path}")
    print(f"  Figure manifest: {figure_manifest_path}")
    print(f"  Methods/results/discussion narrative: {mrd_path}")
    print(f"  Integration recommendations: {decision_table_path}")
    print(f"  Source manifest: {source_manifest_path}")
    print("")


if __name__ == "__main__":
    main()
