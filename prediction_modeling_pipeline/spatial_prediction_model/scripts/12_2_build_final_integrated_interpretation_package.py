import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    path = Path(path)
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
            return pd.read_csv(p, sep="\t", low_memory=False)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, low_memory=False)
        if p.suffix.lower() == ".json":
            obj = read_json(p)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                return pd.DataFrame([obj])
        return pd.read_csv(p, sep=None, engine="python", low_memory=False)
    except Exception as exc:
        print(f"WARNING: Could not read table {p}: {exc}")
        return None


def write_table(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    if df is None:
        df = pd.DataFrame()
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def write_txt(path, body):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(f"FILEPATH: {path}\n\n{body}", encoding="utf-8")


def find_col(df, names):
    if df is None or df.empty:
        return None

    lower = {str(c).lower(): c for c in df.columns}

    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]

    for name in names:
        needle = name.lower()
        for c in df.columns:
            if needle in str(c).lower():
                return c

    return None


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def first_existing(source_map, key):
    value = source_map.get(key, "")
    if isinstance(value, list):
        for item in value:
            if item and Path(item).exists():
                return str(item)
        return ""
    if value and Path(value).exists():
        return str(value)
    return ""


def clean_label(x, width=70):
    x = str(x)
    return x if len(x) <= width else x[: width - 3] + "..."


def short_wrap(x, width=42):
    x = str(x)
    words = x.split()
    lines = []
    cur = ""
    for word in words:
        nxt = word if not cur else cur + " " + word
        if len(nxt) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def plot_barh(df, value_col, label_col, out, title, xlabel, top_n=30):
    if df is None or df.empty:
        return False
    if value_col not in df.columns or label_col not in df.columns:
        return False

    d = df.copy()
    d[value_col] = numeric(d[value_col])
    d = d.dropna(subset=[value_col, label_col])
    if d.empty:
        return False

    d = d.sort_values(value_col, ascending=True)
    if top_n:
        d = d.tail(top_n)

    labels = [short_wrap(v, 48) for v in d[label_col].astype(str).tolist()]
    values = d[value_col].astype(float).tolist()

    h = max(5.5, 0.42 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(12, h))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_hist(df, value_col, out, title, xlabel, bins=24):
    if df is None or df.empty or value_col not in df.columns:
        return False

    vals = numeric(df[value_col]).dropna()
    if vals.empty:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scatter(df, xcol, ycol, out, title, xlabel, ylabel):
    if df is None or df.empty:
        return False
    if xcol not in df.columns or ycol not in df.columns:
        return False

    d = df.copy()
    d[xcol] = numeric(d[xcol])
    d[ycol] = numeric(d[ycol])
    d = d.dropna(subset=[xcol, ycol])
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(d[xcol], d[ycol])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_heatmap(df, out, title, label_col, metric_cols):
    if df is None or df.empty:
        return False

    cols = [c for c in metric_cols if c in df.columns]
    if not cols or label_col not in df.columns:
        return False

    d = df[[label_col] + cols].copy()
    for c in cols:
        d[c] = numeric(d[c])

    d = d.replace([np.inf, -np.inf], np.nan)
    if d[cols].isna().all().all():
        return False

    mat = d[cols].fillna(0.0).to_numpy(dtype=float)
    labels = [short_wrap(v, 34) for v in d[label_col].astype(str).tolist()]

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(cols)), max(4.5, 0.7 * len(labels) + 2)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def source_manifest(source_map):
    rows = []
    for k, v in source_map.items():
        if isinstance(v, list):
            if len(v) == 0:
                rows.append({"source_key": k, "path": "", "exists": False})
            for item in v:
                rows.append({"source_key": k, "path": str(item), "exists": bool(item and Path(item).exists())})
        else:
            rows.append({"source_key": k, "path": str(v), "exists": bool(v and Path(v).exists())})
    return pd.DataFrame(rows)


def model_comparison(source_map):
    rows = []

    metrics = read_table(first_existing(source_map, "probability_vs_residual_metrics_long"))
    contrib = read_table(first_existing(source_map, "probability_vs_residual_spatial_contribution_long"))

    spatial_fraction = {}
    if contrib is not None and not contrib.empty:
        model_col = find_col(contrib, ["model", "model_name"])
        class_col = find_col(contrib, ["feature_class", "component", "feature_group", "group"])
        frac_col = find_col(contrib, ["fraction_of_total_score", "fraction", "value"])
        if model_col and class_col and frac_col:
            c = contrib.copy()
            c[frac_col] = numeric(c[frac_col])
            for model, sub in c.groupby(model_col):
                ss = sub[sub[class_col].astype(str).str.lower().str.contains("spatial", na=False)]
                if not ss.empty:
                    spatial_fraction[str(model)] = float(ss[frac_col].max())

    if metrics is not None and not metrics.empty:
        model_col = find_col(metrics, ["model", "model_name"])
        split_col = find_col(metrics, ["split"])
        pearson_col = find_col(metrics, ["pearson", "test_pearson"])
        r2_col = find_col(metrics, ["r2", "test_r2"])
        target_col = find_col(metrics, ["target"])

        m = metrics.copy()
        if split_col:
            test_m = m[m[split_col].astype(str).str.lower().eq("test")].copy()
            if test_m.empty:
                test_m = m.copy()
        else:
            test_m = m.copy()

        if model_col and pearson_col:
            for _, r in test_m.iterrows():
                model = str(r[model_col])
                lower = model.lower()
                if "prob" in lower or "pooled" in lower:
                    family = "pooled_probability_xgboost"
                    desc = "Pooled probability model"
                    target = "fused_prob_responder"
                    purpose = "Overall pair level response prediction baseline"
                    note = "Strong predictive baseline but less biologically specific because treatment identity dominated"
                elif "residual" in lower or "prior" in lower:
                    family = "pair_level_prior_adjusted_residual_xgboost"
                    desc = "Pair level prior adjusted residual model"
                    target = "fused_residual_vs_prior"
                    purpose = "Biological interpretation of response above or below treatment prior"
                    note = "Residual target exposed substantially more spatial biology"
                else:
                    continue

                rows.append({
                    "model_family": family,
                    "model_name": model,
                    "description": desc,
                    "purpose": purpose,
                    "unit_of_prediction": "sample_treatment_pair",
                    "target": str(r[target_col]) if target_col else target,
                    "inputs": "spatial features plus drug identity",
                    "validation_type": "held out test split",
                    "primary_metric_name": "test_pearson",
                    "primary_metric_value": float(r[pearson_col]) if pd.notna(r[pearson_col]) else np.nan,
                    "secondary_metric_name": "test_r2",
                    "secondary_metric_value": float(r[r2_col]) if r2_col and pd.notna(r[r2_col]) else np.nan,
                    "spatial_feature_fraction": spatial_fraction.get(model, np.nan),
                    "validated_finding_type": "model benchmark",
                    "recommended_use": "prediction baseline" if family.startswith("pooled") else "pair level residual biology",
                    "notes": note
                })

    broad = read_table(first_existing(source_map, "repeated_split_metric_summary"))
    broad_targets = pd.DataFrame()
    if broad is not None and not broad.empty:
        target_col = find_col(broad, ["target_name", "target"])
        pearson_col = find_col(broad, ["test_pearson_mean", "mean_test_pearson"])
        r2_col = find_col(broad, ["test_r2_mean", "mean_test_r2"])
        if target_col and pearson_col:
            broad_targets = broad.copy()
            broad_targets[pearson_col] = numeric(broad_targets[pearson_col])
            if r2_col:
                broad_targets[r2_col] = numeric(broad_targets[r2_col])
            broad_targets = broad_targets.sort_values(pearson_col, ascending=False)
            best = broad_targets.iloc[0]
            rows.append({
                "model_family": "broad_residual_spatial_only",
                "model_name": "spatial_only_broad_residual_best_target",
                "description": "Sample level broad residual spatial only model",
                "purpose": "Screen whether spatial biology predicts broad residual behavior across treatments",
                "unit_of_prediction": "sample",
                "target": str(best[target_col]),
                "inputs": "strict spatial biology features only",
                "validation_type": "100 repeated train test splits",
                "primary_metric_name": "mean_test_pearson",
                "primary_metric_value": float(best[pearson_col]),
                "secondary_metric_name": "mean_test_r2",
                "secondary_metric_value": float(best[r2_col]) if r2_col and pd.notna(best[r2_col]) else np.nan,
                "spatial_feature_fraction": 1.0,
                "validated_finding_type": "screening level broad phenotype",
                "recommended_use": "sample level screening context",
                "notes": "Useful broad spatial phenotype but weaker than treatment specific branch"
            })

    curated = read_table(first_existing(source_map, "curated_treatment_model_summary"))
    if curated is not None and not curated.empty:
        pearson_col = find_col(curated, ["test_pearson_mean", "mean_test_pearson"])
        r2_col = find_col(curated, ["test_r2_mean", "mean_test_r2"])
        tier_col = find_col(curated, ["interpretation_tier"])
        selected_col = find_col(curated, ["selected_for_final_model"])
        c = curated.copy()
        if pearson_col:
            c[pearson_col] = numeric(c[pearson_col])
        if r2_col:
            c[r2_col] = numeric(c[r2_col])
        if selected_col:
            c_selected = c[c[selected_col].astype(str).str.lower().isin(["true", "1", "yes"])]
            if c_selected.empty:
                c_selected = c
        else:
            c_selected = c
        tier1_count = int(c[tier_col].astype(str).str.contains("tier1", case=False, na=False).sum()) if tier_col else np.nan
        rows.append({
            "model_family": "filtered_per_treatment_residual_models",
            "model_name": "curated_per_treatment_residual_models",
            "description": "Treatment specific residual models",
            "purpose": "Identify treatment specific spatial biology explaining response above or below treatment prior",
            "unit_of_prediction": "sample within treatment",
            "target": "fused_residual_vs_prior",
            "inputs": "strict spatial biology features only within each treatment",
            "validation_type": "repeated split screening plus curation",
            "primary_metric_name": "median_mean_test_pearson_selected",
            "primary_metric_value": float(c_selected[pearson_col].median()) if pearson_col else np.nan,
            "secondary_metric_name": "median_mean_test_r2_selected",
            "secondary_metric_value": float(c_selected[r2_col].median()) if r2_col else np.nan,
            "spatial_feature_fraction": 1.0,
            "validated_finding_type": f"tier1_count={tier1_count}",
            "recommended_use": "treatment specific spatial biology discovery",
            "notes": "Strongest discovery branch before label shuffle validation"
        })

    label = read_table(first_existing(source_map, "label_shuffle_treatment_summary"))
    validated = pd.DataFrame()
    if label is not None and not label.empty:
        validated = label.copy()
        fdr_col = find_col(validated, ["fdr_q_value", "bh_fdr_q_value", "q_value"])
        p_col = find_col(validated, ["empirical_p_value", "empirical_p"])
        obs_col = find_col(validated, ["observed_test_pearson_mean", "observed_mean_test_pearson"])
        if fdr_col:
            validated[fdr_col] = numeric(validated[fdr_col])
            n_pass = int((validated[fdr_col] <= 0.10).sum())
        elif p_col:
            validated[p_col] = numeric(validated[p_col])
            n_pass = int((validated[p_col] <= 0.05).sum())
        else:
            n_pass = 0

        rows.append({
            "model_family": "tier1_label_shuffle_validation",
            "model_name": "tier1_per_treatment_label_shuffle_validation",
            "description": "Label shuffle validation for Tier 1 treatment models",
            "purpose": "Test whether Tier 1 treatment specific signals exceed shuffled label null performance",
            "unit_of_prediction": "treatment validation summary",
            "target": "Tier 1 per treatment residual models",
            "inputs": "observed treatment models plus shuffled residual labels",
            "validation_type": "100 label permutations per Tier 1 treatment",
            "primary_metric_name": "validated_treatments_fdr_le_0_10",
            "primary_metric_value": float(n_pass),
            "secondary_metric_name": "n_treatments_tested",
            "secondary_metric_value": float(len(validated)),
            "spatial_feature_fraction": np.nan,
            "validated_finding_type": "formal null validation",
            "recommended_use": "validated treatment specific spatial biology",
            "notes": "This is the strongest validation layer"
        })

    return pd.DataFrame(rows), broad_targets, validated


def normalize_feature_table(df, source_branch):
    if df is None or df.empty:
        return pd.DataFrame()

    feature_col = find_col(df, ["feature_original", "feature_name", "feature"])
    theme_col = find_col(df, ["biological_theme", "theme"])
    score_col = find_col(df, ["top10_frequency", "total_mean_abs_shap", "mean_abs_shap", "mean_abs_shap_mean", "max_mean_abs_shap"])

    if not feature_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["feature"] = df[feature_col].astype(str)
    out["source_branch"] = source_branch
    out["biological_theme"] = df[theme_col].astype(str) if theme_col else ""
    out["score"] = numeric(df[score_col]) if score_col else np.nan
    return out


def normalize_theme_table(df, source_branch):
    if df is None or df.empty:
        return pd.DataFrame()

    theme_col = find_col(df, ["biological_theme", "theme"])
    score_col = find_col(df, ["total_mean_abs_shap", "mean_abs_shap", "mean_abs_shap_mean", "max_mean_abs_shap", "score"])

    if not theme_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["theme"] = df[theme_col].astype(str)
    out["source_branch"] = source_branch
    out["score"] = numeric(df[score_col]) if score_col else np.nan
    return out


def recurrent_feature_theme_tables(source_map):
    feature_parts = [
        normalize_feature_table(read_table(first_existing(source_map, "top_residual_biology_features_strict")), "pair_level_residual"),
        normalize_feature_table(read_table(first_existing(source_map, "cross_target_top_features")), "broad_residual"),
        normalize_feature_table(read_table(first_existing(source_map, "core_recurrent_spatial_features")), "per_treatment_curated"),
        normalize_feature_table(read_table(first_existing(source_map, "all_cross_treatment_feature_stability")), "per_treatment_full"),
    ]
    feature_long = pd.concat([x for x in feature_parts if x is not None and not x.empty], ignore_index=True) if any((x is not None and not x.empty) for x in feature_parts) else pd.DataFrame(columns=["feature", "source_branch", "biological_theme", "score"])

    if not feature_long.empty:
        feature_summary = (
            feature_long
            .groupby("feature", dropna=False)
            .agg(
                source_count=("source_branch", "nunique"),
                source_branches=("source_branch", lambda s: "|".join(sorted(set(map(str, s))))),
                biological_themes=("biological_theme", lambda s: "|".join(sorted(set([x for x in map(str, s) if x and x != "nan"])))),
                mean_score=("score", "mean"),
                max_score=("score", "max")
            )
            .reset_index()
            .sort_values(["source_count", "max_score", "mean_score"], ascending=[False, False, False])
        )
    else:
        feature_summary = pd.DataFrame(columns=["feature", "source_count", "source_branches", "biological_themes", "mean_score", "max_score"])

    theme_parts = [
        normalize_theme_table(read_table(first_existing(source_map, "residual_biology_theme_summary")), "pair_level_residual"),
        normalize_theme_table(read_table(first_existing(source_map, "theme_summary_all_targets")), "broad_residual"),
        normalize_theme_table(read_table(first_existing(source_map, "cross_treatment_theme_summary")), "per_treatment_curated"),
    ]
    theme_long = pd.concat([x for x in theme_parts if x is not None and not x.empty], ignore_index=True) if any((x is not None and not x.empty) for x in theme_parts) else pd.DataFrame(columns=["theme", "source_branch", "score"])

    if not theme_long.empty:
        theme_summary = (
            theme_long
            .groupby("theme", dropna=False)
            .agg(
                source_count=("source_branch", "nunique"),
                source_branches=("source_branch", lambda s: "|".join(sorted(set(map(str, s))))),
                mean_score=("score", "mean"),
                max_score=("score", "max")
            )
            .reset_index()
            .sort_values(["source_count", "max_score", "mean_score"], ascending=[False, False, False])
        )
    else:
        theme_summary = pd.DataFrame(columns=["theme", "source_count", "source_branches", "mean_score", "max_score"])

    return feature_long, feature_summary, theme_long, theme_summary


def integration_recommendations():
    return pd.DataFrame([
        {
            "pipeline_component": "01_to_07_core_prediction_engine",
            "current_status": "canonical",
            "recommended_decision": "keep",
            "promote_to_canonical": "already canonical",
            "rationale": "Core prediction engine completed successfully and should remain the base pipeline",
            "notes": "Do not rewrite canonical scripts until migration is reviewed"
        },
        {
            "pipeline_component": "08_residual_biology_interpretation",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Maps residual SHAP features to spatial biology and creates interpretable themes",
            "notes": "Official downstream interpretation module"
        },
        {
            "pipeline_component": "09_broad_residual_spatial_only_model",
            "current_status": "experimental",
            "recommended_decision": "promote_as_screening_module",
            "promote_to_canonical": "yes",
            "rationale": "Provides sample level broad spatial phenotype screening",
            "notes": "Use as exploratory context, not final validated endpoint"
        },
        {
            "pipeline_component": "10_filtered_per_treatment_residual_models",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Strongest branch for treatment specific spatial response biology",
            "notes": "Should become official downstream discovery step"
        },
        {
            "pipeline_component": "11_tier1_label_shuffle_validation",
            "current_status": "experimental",
            "recommended_decision": "promote",
            "promote_to_canonical": "yes",
            "rationale": "Provides formal null validation for Tier 1 treatment specific models",
            "notes": "Important for publication style claims"
        },
        {
            "pipeline_component": "12_final_integrated_interpretation_package",
            "current_status": "new_experimental_reporting_step",
            "recommended_decision": "keep_and_promote_after_review",
            "promote_to_canonical": "likely yes",
            "rationale": "Creates final reports, comparison figures, file manifest, and canonical recommendations",
            "notes": "Non destructive synthesis step"
        }
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-map", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    source_map = json.loads(Path(args.source_map).read_text(encoding="utf-8"))

    out_root = ensure_dir(Path(args.output_root))
    d01 = ensure_dir(out_root / "01_model_milestone_summary")
    d02 = ensure_dir(out_root / "02_validated_treatment_models")
    d03 = ensure_dir(out_root / "03_recurrent_spatial_features")
    d04 = ensure_dir(out_root / "04_recurrent_biology_themes")
    d05 = ensure_dir(out_root / "05_figures_for_presentation")
    d06 = ensure_dir(out_root / "06_methods_results_discussion_report")
    d07 = ensure_dir(out_root / "07_pipeline_integration_recommendations")
    d08 = ensure_dir(out_root / "08_provenance_and_file_manifest")

    generated = []
    figures = []

    def add_generated(path, category, description):
        generated.append({"path": str(path), "category": category, "description": description})

    def add_figure(fig_id, path, title, description, use):
        figures.append({"figure_id": fig_id, "path": str(path), "title": title, "description": description, "recommended_use": use})

    src_manifest = source_manifest(source_map)
    src_manifest_path = d08 / "source_file_manifest.tsv"
    write_table(src_manifest, src_manifest_path)
    add_generated(src_manifest_path, "table", "Source file manifest")

    model_comp, broad_targets, validated = model_comparison(source_map)
    feature_long, feature_summary, theme_long, theme_summary = recurrent_feature_theme_tables(source_map)
    decisions = integration_recommendations()

    model_comp_path = d01 / "model_comparison_table.tsv"
    write_table(model_comp, model_comp_path)
    add_generated(model_comp_path, "table", "Model comparison table")

    if broad_targets is not None and not broad_targets.empty:
        target_col = find_col(broad_targets, ["target_name", "target"])
        pearson_col = find_col(broad_targets, ["test_pearson_mean", "mean_test_pearson"])
        r2_col = find_col(broad_targets, ["test_r2_mean", "mean_test_r2"])
        cols = [c for c in [target_col, pearson_col, r2_col] if c]
        broad_out = broad_targets[cols].copy()
        broad_targets_path = d01 / "broad_residual_target_comparison.tsv"
        write_table(broad_out, broad_targets_path)
        add_generated(broad_targets_path, "table", "Broad residual target comparison")

    if validated is not None and not validated.empty:
        drug_col = find_col(validated, ["drug_label", "treatment", "treatment_name"])
        obs_col = find_col(validated, ["observed_test_pearson_mean", "observed_mean_test_pearson"])
        fdr_col = find_col(validated, ["fdr_q_value", "bh_fdr_q_value", "q_value"])
        if obs_col:
            validated[obs_col] = numeric(validated[obs_col])
            validated = validated.sort_values(obs_col, ascending=False)
        if fdr_col:
            validated[fdr_col] = numeric(validated[fdr_col])
        valid_path = d02 / "validated_treatment_table.tsv"
        write_table(validated, valid_path)
        add_generated(valid_path, "table", "Validated treatment table from label shuffle validation")

    feature_long_path = d03 / "recurrent_spatial_feature_long.tsv"
    feature_summary_path = d03 / "recurrent_spatial_feature_table.tsv"
    write_table(feature_long, feature_long_path)
    write_table(feature_summary, feature_summary_path)
    add_generated(feature_long_path, "table", "Long recurrent spatial feature table")
    add_generated(feature_summary_path, "table", "Recurrent spatial feature summary")

    theme_long_path = d04 / "recurrent_biology_theme_long.tsv"
    theme_summary_path = d04 / "recurrent_biology_theme_table.tsv"
    write_table(theme_long, theme_long_path)
    write_table(theme_summary, theme_summary_path)
    add_generated(theme_long_path, "table", "Long recurrent biology theme table")
    add_generated(theme_summary_path, "table", "Recurrent biology theme summary")

    decision_path = d07 / "pipeline_integration_recommendations.tsv"
    write_table(decisions, decision_path)
    add_generated(decision_path, "table", "Canonical integration recommendation table")

    script_prov_files = source_map.get("script_provenance_files", []) or []
    prov_rows = []
    for p in script_prov_files:
        prov_rows.append({
            "path": str(p),
            "exists": bool(p and Path(p).exists()),
            "canonical_original_script_modified": "no",
            "note": "Experimental provenance source discovered during final package build"
        })
    prov = pd.DataFrame(prov_rows)
    prov_path = d08 / "provenance_table.tsv"
    write_table(prov, prov_path)
    add_generated(prov_path, "table", "Provenance table")

    fig1 = d05 / "figure01_model_family_primary_metric_bar.png"
    if plot_barh(model_comp, "primary_metric_value", "model_family", fig1, "Model family primary metric comparison", "Primary metric value", None):
        add_figure("Figure 1", fig1, "Model family primary metric comparison", "Compares headline metric values across model families", "Overview figure")

    fig2 = d05 / "figure02_model_family_metric_heatmap.png"
    if plot_heatmap(model_comp, fig2, "Model family metric heatmap", "model_family", ["primary_metric_value", "secondary_metric_value", "spatial_feature_fraction"]):
        add_figure("Figure 2", fig2, "Model family metric heatmap", "Heatmap of primary, secondary, and spatial contribution metrics", "Publication style comparison")

    contrib = read_table(first_existing(source_map, "probability_vs_residual_spatial_contribution_long"))
    if contrib is not None and not contrib.empty:
        model_col = find_col(contrib, ["model", "model_name"])
        class_col = find_col(contrib, ["feature_class", "component", "feature_group"])
        frac_col = find_col(contrib, ["fraction_of_total_score", "fraction"])
        if model_col and class_col and frac_col:
            c = contrib.copy()
            c[frac_col] = numeric(c[frac_col])
            c = c[c[class_col].astype(str).str.lower().str.contains("spatial", na=False)]
            fig3 = d05 / "figure03_probability_vs_residual_spatial_fraction.png"
            if plot_barh(c, frac_col, model_col, fig3, "Spatial contribution by pair level model", "Spatial fraction of explanation score", None):
                add_figure("Figure 3", fig3, "Spatial contribution by pair level model", "Shows how residual modeling increased spatial contribution", "Residual model justification")

    if broad_targets is not None and not broad_targets.empty:
        target_col = find_col(broad_targets, ["target_name", "target"])
        pearson_col = find_col(broad_targets, ["test_pearson_mean", "mean_test_pearson"])
        if target_col and pearson_col:
            fig4 = d05 / "figure04_broad_residual_target_comparison.png"
            if plot_barh(broad_targets, pearson_col, target_col, fig4, "Broad residual target comparison", "Mean test Pearson", 12):
                add_figure("Figure 4", fig4, "Broad residual target comparison", "Compares broad sample level residual targets", "Broad residual screening summary")

    curated = read_table(first_existing(source_map, "curated_treatment_model_summary"))
    if curated is not None and not curated.empty:
        pearson_col = find_col(curated, ["test_pearson_mean", "mean_test_pearson"])
        r2_col = find_col(curated, ["test_r2_mean", "mean_test_r2"])
        drug_col = find_col(curated, ["drug_label", "treatment", "treatment_name"])
        if pearson_col:
            fig5 = d05 / "figure05_per_treatment_model_performance_distribution.png"
            if plot_hist(curated, pearson_col, fig5, "Per treatment model performance distribution", "Mean test Pearson", 24):
                add_figure("Figure 5", fig5, "Per treatment model performance distribution", "Distribution of curated treatment model performance", "Treatment screening overview")
        if pearson_col and r2_col:
            fig6 = d05 / "figure06_per_treatment_pearson_vs_r2.png"
            if plot_scatter(curated, pearson_col, r2_col, fig6, "Per treatment Pearson versus R2", "Mean test Pearson", "Mean test R2"):
                add_figure("Figure 6", fig6, "Per treatment Pearson versus R2", "Compares correlation and explained variance across treatment models", "Supplementary QC figure")
        if pearson_col and drug_col:
            tier_col = find_col(curated, ["interpretation_tier"])
            ctop = curated.copy()
            if tier_col:
                tier1 = ctop[ctop[tier_col].astype(str).str.contains("tier1", case=False, na=False)]
                if not tier1.empty:
                    ctop = tier1
            fig7 = d05 / "figure07_top_curated_tier1_treatment_models.png"
            if plot_barh(ctop, pearson_col, drug_col, fig7, "Top curated Tier 1 treatment models", "Mean test Pearson", 20):
                add_figure("Figure 7", fig7, "Top curated Tier 1 treatment models", "Ranks strongest curated treatment specific residual models", "Treatment specific headline figure")

    if validated is not None and not validated.empty:
        drug_col = find_col(validated, ["drug_label", "treatment", "treatment_name"])
        obs_col = find_col(validated, ["observed_test_pearson_mean", "observed_mean_test_pearson"])
        p_col = find_col(validated, ["empirical_p_value", "empirical_p"])
        fdr_col = find_col(validated, ["fdr_q_value", "bh_fdr_q_value", "q_value"])

        if drug_col and obs_col:
            fig8 = d05 / "figure08_label_shuffle_validated_treatments.png"
            if plot_barh(validated, obs_col, drug_col, fig8, "Label shuffle validated treatments", "Observed mean test Pearson", 20):
                add_figure("Figure 8", fig8, "Label shuffle validated treatments", "Ranks validated Tier 1 treatments by observed performance", "Validated treatment figure")

        if obs_col and fdr_col:
            v = validated.copy()
            v[fdr_col] = numeric(v[fdr_col]).clip(lower=1e-12)
            v["negative_log10_fdr"] = -np.log10(v[fdr_col])
            fig9 = d05 / "figure09_label_shuffle_significance.png"
            if plot_scatter(v, obs_col, "negative_log10_fdr", fig9, "Label shuffle validation significance", "Observed mean test Pearson", "negative log10 FDR q"):
                add_figure("Figure 9", fig9, "Label shuffle validation significance", "Observed performance versus label shuffle FDR", "Validation figure")

    if not feature_summary.empty:
        fig10 = d05 / "figure10_recurrent_spatial_features.png"
        if plot_barh(feature_summary, "source_count", "feature", fig10, "Recurrent spatial features across model branches", "Number of model branches", 25):
            add_figure("Figure 10", fig10, "Recurrent spatial features", "Features recurring across residual pair, broad residual, and per treatment branches", "Biology synthesis figure")

    if not theme_summary.empty:
        fig11 = d05 / "figure11_recurrent_biology_themes.png"
        if plot_barh(theme_summary, "source_count", "theme", fig11, "Recurrent biology themes across model branches", "Number of model branches", 15):
            add_figure("Figure 11", fig11, "Recurrent biology themes", "Themes recurring across model branches", "Biology synthesis figure")

    design_table = model_comp[[
        "model_family",
        "description",
        "purpose",
        "unit_of_prediction",
        "target",
        "inputs",
        "validation_type",
        "primary_metric_name",
        "primary_metric_value",
        "secondary_metric_name",
        "secondary_metric_value",
        "recommended_use",
        "notes"
    ]].copy() if not model_comp.empty else pd.DataFrame()
    design_path = d05 / "table01_publication_style_model_comparison.tsv"
    write_table(design_table, design_path)
    add_figure("Table 1", design_path, "Publication style model comparison table", "Compares model name, purpose, target, inputs, validation, and metrics", "Main manuscript style table")
    add_generated(design_path, "table", "Publication style model comparison table")

    fig_manifest = pd.DataFrame(figures)
    fig_manifest_path = d05 / "figure_manifest.tsv"
    write_table(fig_manifest, fig_manifest_path)
    add_generated(fig_manifest_path, "table", "Figure manifest")

    caption_lines = []
    for r in figures:
        caption_lines.append(f"{r['figure_id']}. {r['title']}. {r['description']} Recommended use: {r['recommended_use']}.")
    captions_path = d05 / "figure_captions.txt"
    write_txt(captions_path, "\n\n".join(caption_lines))
    add_generated(captions_path, "txt", "Presentation ready figure captions")

    pass_count = 0
    if validated is not None and not validated.empty:
        fdr_col = find_col(validated, ["fdr_q_value", "bh_fdr_q_value", "q_value"])
        if fdr_col:
            pass_count = int((numeric(validated[fdr_col]) <= 0.10).sum())

    master_lines = []
    master_lines.append("FINAL INTEGRATED SPATIAL RESPONSE INTERPRETATION PACKAGE")
    master_lines.append("")
    master_lines.append(f"Output root: {out_root}")
    master_lines.append("")
    master_lines.append("Summary")
    master_lines.append(f"Model families summarized: {len(model_comp)}")
    master_lines.append(f"Validated treatment models in label shuffle table: {len(validated) if validated is not None else 0}")
    master_lines.append(f"Validated treatment models passing FDR q <= 0.10: {pass_count}")
    master_lines.append(f"Recurrent spatial features summarized: {len(feature_summary)}")
    master_lines.append(f"Recurrent biology themes summarized: {len(theme_summary)}")
    master_lines.append("")
    master_lines.append("Interpretation")
    master_lines.append("The pooled probability model is the best baseline prediction model, but it is less biologically specific because treatment identity dominates.")
    master_lines.append("The prior adjusted residual model is the best pair level biological interpretation model because it asks which spatial features explain response above or below treatment prior.")
    master_lines.append("The broad residual spatial only model provides a useful sample level screening phenotype.")
    master_lines.append("The filtered per treatment residual models, followed by Tier 1 curation and label shuffle validation, provide the strongest treatment specific spatial biology results.")
    master_lines.append("")
    if not theme_summary.empty:
        master_lines.append("Top recurrent biology themes")
        for theme in theme_summary.head(10)["theme"].astype(str).tolist():
            master_lines.append(f"  {theme}")
        master_lines.append("")
    if not feature_summary.empty:
        master_lines.append("Top recurrent spatial features")
        for feat in feature_summary.head(10)["feature"].astype(str).tolist():
            master_lines.append(f"  {feat}")

    master_path = d01 / "master_summary_report.txt"
    write_txt(master_path, "\n".join(master_lines))
    add_generated(master_path, "txt", "Master summary report")

    mrd_lines = []
    mrd_lines.append("METHODS, RESULTS, AND DISCUSSION NARRATIVE")
    mrd_lines.append("")
    mrd_lines.append("Methods")
    mrd_lines.append("The analysis compared pooled probability modeling, prior adjusted residual pair level modeling, broad residual spatial only modeling, filtered per treatment residual modeling, and Tier 1 label shuffle validation.")
    mrd_lines.append("The residual target fused_residual_vs_prior was used to focus on response above or below treatment prior.")
    mrd_lines.append("The broad residual model collapsed treatment residuals to sample level targets, while per treatment residual models were fit within eligible treatment subsets using spatial features only.")
    mrd_lines.append("")
    mrd_lines.append("Results")
    if not model_comp.empty:
        for _, r in model_comp.iterrows():
            val = r.get("primary_metric_value", np.nan)
            if pd.notna(val):
                mrd_lines.append(f"{r['model_family']} had {r['primary_metric_name']} = {val:.4g}.")
    if validated is not None and not validated.empty:
        mrd_lines.append(f"Tier 1 label shuffle validation summarized {len(validated)} treatment models, with {pass_count} passing FDR q <= 0.10.")
    if not theme_summary.empty:
        mrd_lines.append("The most recurrent biological themes included " + ", ".join(theme_summary.head(6)["theme"].astype(str).tolist()) + ".")
    mrd_lines.append("")
    mrd_lines.append("Discussion")
    mrd_lines.append("The modeling framework improved biological interpretability by moving from treatment dominated probability prediction to prior adjusted residual response modeling.")
    mrd_lines.append("The strongest validated branch is the treatment specific residual modeling branch with label shuffle validation.")
    mrd_lines.append("Canonical scripts 01 to 07 should remain the core prediction engine, while residual interpretation, broad residual modeling, per treatment residual modeling, label shuffle validation, and final integrated reporting should be promoted as downstream modules after review.")

    mrd_path = d06 / "methods_results_discussion_narrative.txt"
    write_txt(mrd_path, "\n".join(mrd_lines))
    add_generated(mrd_path, "txt", "Methods results discussion narrative")

    rec_lines = []
    rec_lines.append("PIPELINE INTEGRATION RECOMMENDATIONS")
    rec_lines.append("")
    rec_lines.append("Keep scripts 01 to 07 as the core prediction engine.")
    rec_lines.append("Promote residual biology interpretation as downstream Step 08.")
    rec_lines.append("Promote broad residual spatial only modeling as downstream Step 09.")
    rec_lines.append("Promote filtered per treatment residual models as downstream Step 10.")
    rec_lines.append("Promote Tier 1 label shuffle validation as downstream Step 11.")
    rec_lines.append("Promote final integrated interpretation packaging as downstream Step 12.")
    rec_lines.append("")
    rec_lines.append("No canonical original scripts were modified by this final packaging step.")

    rec_txt = d07 / "pipeline_integration_recommendations.txt"
    write_txt(rec_txt, "\n".join(rec_lines))
    add_generated(rec_txt, "txt", "Pipeline integration recommendation narrative")

    generated_manifest = pd.DataFrame(generated)
    generated_manifest_path = d08 / "generated_output_manifest.tsv"
    write_table(generated_manifest, generated_manifest_path)

    run_summary = {
        "output_root": str(out_root),
        "source_files_found": int(src_manifest["exists"].sum()) if not src_manifest.empty else 0,
        "model_families_summarized": int(len(model_comp)),
        "validated_treatment_models": int(len(validated) if validated is not None else 0),
        "validated_treatment_models_passing_fdr_0_10": int(pass_count),
        "recurrent_spatial_features": int(len(feature_summary)),
        "recurrent_biology_themes": int(len(theme_summary)),
        "figures_and_main_tables": int(len(fig_manifest)),
        "master_summary_report": str(master_path),
        "model_comparison_table": str(model_comp_path),
        "figure_manifest": str(fig_manifest_path),
        "methods_results_discussion": str(mrd_path),
        "pipeline_integration_recommendations": str(decision_path)
    }

    run_summary_path = d01 / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("")
    print("=" * 90)
    print("FINAL INTEGRATED INTERPRETATION PACKAGE COMPLETE")
    print("=" * 90)
    print("Output root:", out_root)
    print("Source files found:", run_summary["source_files_found"])
    print("Model families summarized:", run_summary["model_families_summarized"])
    print("Validated treatment models:", run_summary["validated_treatment_models"])
    print("Validated treatment models passing FDR q <= 0.10:", run_summary["validated_treatment_models_passing_fdr_0_10"])
    print("Recurrent spatial features:", run_summary["recurrent_spatial_features"])
    print("Recurrent biology themes:", run_summary["recurrent_biology_themes"])
    print("Figures and main tables:", run_summary["figures_and_main_tables"])
    print("")
    print("Key outputs")
    print("Master summary:", master_path)
    print("Model comparison table:", model_comp_path)
    print("Validated treatment table:", d02 / "validated_treatment_table.tsv")
    print("Recurrent feature table:", feature_summary_path)
    print("Recurrent theme table:", theme_summary_path)
    print("Figure manifest:", fig_manifest_path)
    print("Figure captions:", captions_path)
    print("Methods/results/discussion:", mrd_path)
    print("Integration recommendations:", rec_txt)
    print("Source manifest:", src_manifest_path)
    print("Generated manifest:", generated_manifest_path)


if __name__ == "__main__":
    main()
