"""
Script:
    10_N_curate_per_treatment_residual_models.py

Purpose:
    Curate, audit, and summarize the full filtered per treatment residual model run.

Design:
    New downstream analysis script.
    Does not overwrite canonical scripts.
    Does not retrain models.
    Separates high confidence, screening, and caution treatment models.
    Summarizes recurring SHAP features and biology themes.
    Audits logs for real errors versus known constant input warnings.

Text report convention:
    Every generated .txt report starts with its own filepath.
"""

from pathlib import Path
import argparse
import json
import hashlib
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", default="")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_table(path):
    path = Path(path)
    if not path.exists():
        return None
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


def short_label(value, width=65):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + "..."


def as_bool_series(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def numeric(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)


def add_caution_flags(row):
    flags = []

    if not bool(row.get("selected_for_final_model_bool", False)):
        flags.append("not selected for final model")

    if not bool(row.get("shap_success_bool", False)):
        flags.append("SHAP not successful")

    if pd.notna(row.get("test_r2_mean", np.nan)) and row.get("test_r2_mean", np.nan) < 0:
        flags.append("negative mean test R2")

    if pd.notna(row.get("rmse_improvement_mean", np.nan)) and row.get("rmse_improvement_mean", np.nan) <= 0:
        flags.append("no RMSE improvement versus baseline")

    if pd.notna(row.get("test_pearson_positive_fraction", np.nan)) and row.get("test_pearson_positive_fraction", np.nan) < 0.875:
        flags.append("less than 87.5 percent positive splits")

    if "test_pearson_q025" in row.index and pd.notna(row.get("test_pearson_q025", np.nan)) and row.get("test_pearson_q025", np.nan) < 0:
        flags.append("lower repeated split interval crosses zero")

    if pd.notna(row.get("target_std", np.nan)) and row.get("target_std", np.nan) < 0.03:
        flags.append("low residual target standard deviation")

    return "; ".join(flags)


def audit_logs(log_root):
    rows = []

    if not log_root:
        return pd.DataFrame([{
            "log_file": "",
            "exists": False,
            "n_lines": 0,
            "n_constant_input_warnings": 0,
            "n_tracebacks": 0,
            "n_error_like_lines": 0,
            "status": "no log root provided"
        }])

    log_root = Path(log_root)

    if not log_root.exists():
        return pd.DataFrame([{
            "log_file": str(log_root),
            "exists": False,
            "n_lines": 0,
            "n_constant_input_warnings": 0,
            "n_tracebacks": 0,
            "n_error_like_lines": 0,
            "status": "log root not found"
        }])

    for path in sorted(log_root.glob("*")):
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        constant_warnings = [x for x in lines if "ConstantInputWarning" in x]
        tracebacks = [x for x in lines if "Traceback" in x]

        error_like = []
        for line in lines:
            lower = line.lower()
            if "constantinputwarning" in lower:
                continue
            if "warning" in lower:
                continue
            if "error" in lower or "failed" in lower or "traceback" in lower or "exception" in lower:
                error_like.append(line)

        if tracebacks or error_like:
            status = "review"
        elif constant_warnings:
            status = "warnings_only"
        else:
            status = "ok"

        rows.append({
            "log_file": str(path),
            "exists": True,
            "n_lines": len(lines),
            "n_constant_input_warnings": len(constant_warnings),
            "n_tracebacks": len(tracebacks),
            "n_error_like_lines": len(error_like),
            "status": status
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    full_root = Path(args.full_run_root)
    output_root = Path(args.output_root)
    log_root = Path(args.log_root) if args.log_root else None

    tables_dir = output_root / "01_curated_tables"
    figures_dir = output_root / "02_figures"
    reports_dir = output_root / "03_reports"

    for folder in [tables_dir, figures_dir, reports_dir]:
        ensure_dir(folder)

    paths = {
        "treatment_summary": full_root / "04_analysis" / "treatment_model_summary.tsv",
        "feature_stability": full_root / "04_analysis" / "cross_treatment_feature_stability.tsv",
        "theme_summary": full_root / "04_analysis" / "cross_treatment_theme_summary.tsv",
        "shap_all": full_root / "04_analysis" / "final_model_shap_importance_all.tsv",
        "metrics_long": full_root / "02_repeated_split_validation" / "per_treatment_repeated_split_metrics_long.tsv",
        "final_inventory": full_root / "04_analysis" / "final_model_inventory.tsv",
        "promising": full_root / "04_analysis" / "promising_treatments.tsv",
        "selected_final": full_root / "04_analysis" / "selected_final_model_treatments.tsv",
        "source_report": full_root / "05_reports" / "filtered_per_treatment_residual_model_report.txt",
        "source_summary_json": full_root / "filtered_per_treatment_residual_summary.json",
    }

    treatment = read_table(paths["treatment_summary"])
    features = read_table(paths["feature_stability"])
    themes = read_table(paths["theme_summary"])
    shap_all = read_table(paths["shap_all"])
    metrics_long = read_table(paths["metrics_long"])
    final_inventory = read_table(paths["final_inventory"])
    promising = read_table(paths["promising"])
    selected_final = read_table(paths["selected_final"])

    missing = [name for name, path in paths.items() if name not in ["source_summary_json"] and not Path(path).exists()]

    if treatment is None:
        raise FileNotFoundError(paths["treatment_summary"])

    treatment = treatment.copy()

    for col in [
        "n_samples",
        "target_std",
        "target_range",
        "test_pearson_mean",
        "test_pearson_median",
        "test_pearson_std",
        "test_pearson_q025",
        "test_pearson_q975",
        "test_pearson_positive_fraction",
        "test_r2_mean",
        "test_r2_median",
        "test_mae_mean",
        "test_rmse_mean",
        "mae_improvement_mean",
        "rmse_improvement_mean",
        "model_selection_score",
        "all_pearson",
        "all_r2",
    ]:
        if col in treatment.columns:
            treatment[col] = pd.to_numeric(treatment[col], errors="coerce")

    if "selected_for_final_model" in treatment.columns:
        treatment["selected_for_final_model_bool"] = as_bool_series(treatment["selected_for_final_model"])
    else:
        treatment["selected_for_final_model_bool"] = False

    if "promising" in treatment.columns:
        treatment["promising_bool"] = as_bool_series(treatment["promising"])
    else:
        treatment["promising_bool"] = False

    if "shap_status" in treatment.columns:
        treatment["shap_success_bool"] = treatment["shap_status"].astype(str).str.lower().str.contains("success", na=False)
    else:
        treatment["shap_success_bool"] = False

    treatment["caution_flags"] = treatment.apply(add_caution_flags, axis=1)

    tier1 = (
        treatment["selected_for_final_model_bool"]
        & treatment["shap_success_bool"]
        & (numeric(treatment, "test_pearson_mean") >= 0.60)
        & (numeric(treatment, "test_pearson_positive_fraction") >= 0.875)
        & (numeric(treatment, "rmse_improvement_mean") > 0)
        & (numeric(treatment, "test_r2_mean") > 0)
    )

    tier2 = (
        treatment["selected_for_final_model_bool"]
        & treatment["shap_success_bool"]
        & (numeric(treatment, "test_pearson_mean") >= 0.40)
        & (numeric(treatment, "test_pearson_positive_fraction") >= 0.75)
        & (numeric(treatment, "rmse_improvement_mean") > 0)
        & ~tier1
    )

    tier3 = treatment["selected_for_final_model_bool"] & treatment["shap_success_bool"] & ~(tier1 | tier2)

    treatment["interpretation_tier"] = "not_selected_or_not_successful"
    treatment.loc[tier3, "interpretation_tier"] = "tier3_final_model_caution"
    treatment.loc[tier2, "interpretation_tier"] = "tier2_screening_signal"
    treatment.loc[tier1, "interpretation_tier"] = "tier1_high_confidence_screen"

    treatment = treatment.sort_values(
        ["interpretation_tier", "model_selection_score", "test_pearson_mean"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    tier1_df = treatment[treatment["interpretation_tier"] == "tier1_high_confidence_screen"].copy()
    tier2_df = treatment[treatment["interpretation_tier"] == "tier2_screening_signal"].copy()
    tier3_df = treatment[treatment["interpretation_tier"] == "tier3_final_model_caution"].copy()
    caution_df = treatment[treatment["caution_flags"].astype(str).str.len() > 0].copy()

    write_table(treatment, tables_dir / "curated_treatment_model_summary.tsv")
    write_table(tier1_df, tables_dir / "tier1_high_confidence_treatment_models.tsv")
    write_table(tier2_df, tables_dir / "tier2_screening_treatment_models.tsv")
    write_table(tier3_df, tables_dir / "tier3_final_model_caution_treatments.tsv")
    write_table(caution_df, tables_dir / "caution_flags_for_selected_models.tsv")

    if features is not None and len(features) > 0:
        features = features.copy()
        for col in ["top10_frequency", "top20_frequency", "total_mean_abs_shap", "mean_rank", "n_treatments_present"]:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce")

        core_features = features[
            (features.get("top10_frequency", pd.Series(0, index=features.index)) >= 0.20)
            | (features.get("top20_frequency", pd.Series(0, index=features.index)) >= 0.50)
        ].copy()

        core_features = core_features.sort_values(["top10_frequency", "total_mean_abs_shap"], ascending=False)
        write_table(features, tables_dir / "all_cross_treatment_feature_stability.tsv")
        write_table(core_features, tables_dir / "core_recurrent_spatial_features.tsv")
    else:
        core_features = pd.DataFrame()

    if themes is not None and len(themes) > 0:
        themes = themes.copy()
        for col in ["n_features", "n_treatments", "total_mean_abs_shap", "mean_abs_shap"]:
            if col in themes.columns:
                themes[col] = pd.to_numeric(themes[col], errors="coerce")
        max_treatments = max(float(themes["n_treatments"].max()), 1.0) if "n_treatments" in themes.columns else 1.0
        themes["fraction_of_shap_treatments"] = themes["n_treatments"] / max_treatments if "n_treatments" in themes.columns else np.nan
        themes = themes.sort_values("total_mean_abs_shap", ascending=False)
        write_table(themes, tables_dir / "curated_cross_treatment_theme_summary.tsv")

    if shap_all is not None and len(shap_all) > 0:
        shap_all = shap_all.copy()
        for col in ["mean_abs_shap", "rank"]:
            if col in shap_all.columns:
                shap_all[col] = pd.to_numeric(shap_all[col], errors="coerce")

        top_shap = shap_all[shap_all["rank"] <= 10].copy() if "rank" in shap_all.columns else shap_all.sort_values("mean_abs_shap", ascending=False).groupby("drug_key").head(10)
        write_table(top_shap, tables_dir / "per_treatment_top10_shap_features.tsv")

    if metrics_long is not None and len(metrics_long) > 0:
        write_table(metrics_long, tables_dir / "source_repeated_split_metrics_long.tsv")

    if final_inventory is not None and len(final_inventory) > 0:
        write_table(final_inventory, tables_dir / "source_final_model_inventory.tsv")

    log_audit = audit_logs(log_root)
    write_table(log_audit, tables_dir / "log_audit.tsv")

    source_report_first_line = ""
    if paths["source_report"].exists():
        try:
            source_report_first_line = paths["source_report"].read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        except Exception:
            source_report_first_line = ""

    source_report_has_filepath = source_report_first_line.startswith("FILEPATH:")

    script_provenance = pd.DataFrame([{
        "script": str(Path(__file__)),
        "exists": Path(__file__).exists(),
        "sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        "note": "new curation script, no canonical script overwritten"
    }])
    write_table(script_provenance, output_root / "script_provenance.tsv")

    tier_counts = (
        treatment["interpretation_tier"]
        .value_counts(dropna=False)
        .rename_axis("interpretation_tier")
        .reset_index(name="n_treatments")
    )
    write_table(tier_counts, tables_dir / "interpretation_tier_counts.tsv")

    if len(tier1_df) > 0:
        top = tier1_df.sort_values("test_pearson_mean", ascending=False).head(30)
    else:
        top = treatment[treatment["selected_for_final_model_bool"]].sort_values("test_pearson_mean", ascending=False).head(30)

    if len(top) > 0:
        labels = [short_label(x, 55) for x in top["drug_label"].fillna(top["drug_key"])]
        values = top["test_pearson_mean"].astype(float).values
        order = np.arange(len(top))[::-1]

        plt.figure(figsize=(11, max(6, len(top) * 0.32)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Mean test Pearson across repeated splits")
        plt.title("Curated top per treatment residual models")
        savefig(figures_dir / "fig_01_curated_top_treatments_by_test_pearson.png")

    selected = treatment[treatment["selected_for_final_model_bool"]].copy()
    if len(selected) > 0 and "test_r2_mean" in selected.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(selected["test_pearson_mean"].astype(float), selected["test_r2_mean"].astype(float))
        plt.xlabel("Mean test Pearson")
        plt.ylabel("Mean test R2")
        plt.title("Per treatment model correlation versus R2")
        savefig(figures_dir / "fig_02_test_pearson_vs_test_r2.png")

    if len(tier_counts) > 0:
        plot_counts = tier_counts.copy()
        plt.figure(figsize=(8, 5))
        plt.bar(np.arange(len(plot_counts)), plot_counts["n_treatments"].astype(float).values)
        plt.xticks(np.arange(len(plot_counts)), [short_label(x, 35) for x in plot_counts["interpretation_tier"]], rotation=25, ha="right")
        plt.ylabel("Treatment count")
        plt.title("Interpretation tier counts")
        savefig(figures_dir / "fig_03_interpretation_tier_counts.png")

    if len(core_features) > 0:
        topf = core_features.sort_values(["top10_frequency", "total_mean_abs_shap"], ascending=False).head(30)
        labels = [short_label(x, 65) for x in topf["feature_original"].fillna(topf["feature_name"])]
        values = topf["top10_frequency"].astype(float).values
        order = np.arange(len(topf))[::-1]

        plt.figure(figsize=(11, max(6, len(topf) * 0.32)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Top 10 SHAP frequency across final models")
        plt.title("Core recurrent spatial features")
        savefig(figures_dir / "fig_04_core_recurrent_spatial_features.png")

    if themes is not None and len(themes) > 0:
        top_theme = themes.sort_values("total_mean_abs_shap", ascending=False).head(20)
        labels = [short_label(x, 55) for x in top_theme["biological_theme"].fillna("unmapped")]
        values = top_theme["total_mean_abs_shap"].astype(float).values
        order = np.arange(len(top_theme))[::-1]

        plt.figure(figsize=(10, max(5, len(top_theme) * 0.35)))
        plt.barh(order, values[::-1])
        plt.yticks(order, labels[::-1], fontsize=8)
        plt.xlabel("Total mean absolute SHAP across final models")
        plt.title("Curated cross treatment biology themes")
        savefig(figures_dir / "fig_05_curated_biology_theme_contribution.png")

    report_path = reports_dir / "curated_per_treatment_residual_analysis_report.txt"

    show_cols = [
        "drug_key",
        "drug_label",
        "n_samples",
        "target_std",
        "test_pearson_mean",
        "test_pearson_median",
        "test_pearson_positive_fraction",
        "test_r2_mean",
        "rmse_improvement_mean",
        "interpretation_tier",
        "caution_flags",
    ]
    show_cols = [c for c in show_cols if c in treatment.columns]

    feature_show_cols = [c for c in ["feature_name", "feature_original", "biological_theme", "top10_frequency", "top20_frequency", "total_mean_abs_shap", "mean_rank"] if len(core_features) and c in core_features.columns]

    theme_show_cols = [c for c in ["biological_theme", "n_features", "n_treatments", "total_mean_abs_shap", "mean_abs_shap", "fraction_of_shap_treatments"] if themes is not None and c in themes.columns]

    lines = []
    lines.append("CURATED PER TREATMENT RESIDUAL ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Full per treatment run root: {full_root}")
    lines.append(f"Curated analysis output root: {output_root}")
    lines.append(f"Log root audited: {log_root if log_root else ''}")
    lines.append("")
    lines.append("1. Run status")
    lines.append("-" * 100)
    lines.append(f"Treatment summary rows: {len(treatment)}")
    lines.append(f"Selected final models: {int(treatment['selected_for_final_model_bool'].sum())}")
    lines.append(f"SHAP success models: {int(treatment['shap_success_bool'].sum())}")
    lines.append(f"Source report starts with FILEPATH: {source_report_has_filepath}")
    lines.append(f"Missing expected source files: {', '.join(missing) if missing else 'none'}")
    lines.append("")
    lines.append("2. Log audit")
    lines.append("-" * 100)
    lines.append(log_audit.to_string(index=False))
    lines.append("")
    lines.append("3. Interpretation tier counts")
    lines.append("-" * 100)
    lines.append(tier_counts.to_string(index=False))
    lines.append("")
    lines.append("4. Tier 1 high confidence screening models")
    lines.append("-" * 100)
    if len(tier1_df) > 0:
        lines.append(tier1_df[show_cols].head(40).to_string(index=False))
    else:
        lines.append("No treatments met tier 1 criteria.")
    lines.append("")
    lines.append("5. Tier 2 screening models")
    lines.append("-" * 100)
    if len(tier2_df) > 0:
        lines.append(tier2_df[show_cols].head(40).to_string(index=False))
    else:
        lines.append("No treatments met tier 2 criteria.")
    lines.append("")
    lines.append("6. Caution flags")
    lines.append("-" * 100)
    if len(caution_df) > 0:
        lines.append(caution_df[show_cols].head(60).to_string(index=False))
    else:
        lines.append("No caution flags found.")
    lines.append("")
    lines.append("7. Core recurrent spatial features")
    lines.append("-" * 100)
    if len(core_features) > 0:
        lines.append(core_features[feature_show_cols].head(50).to_string(index=False))
    else:
        lines.append("No core recurrent features table available.")
    lines.append("")
    lines.append("8. Cross treatment biology themes")
    lines.append("-" * 100)
    if themes is not None and len(themes) > 0:
        lines.append(themes[theme_show_cols].head(30).to_string(index=False))
    else:
        lines.append("No theme summary table available.")
    lines.append("")
    lines.append("9. Interpretation")
    lines.append("-" * 100)
    lines.append("The per treatment residual models identify treatment specific spatial biology after treatment prior adjustment.")
    lines.append("Validation should be based on repeated split metrics, not final all labeled model fit.")
    lines.append("Tier 1 models are the safest candidates for biological interpretation, but they are still screening level because many treatments were tested.")
    lines.append("The recurring biology is consistent with the prior residual and broad residual models: myeloid macrophage tumor ecology, tumor access and boundary penetration, immune and T cell organization, tryptophan kynurenine metabolism, hypoxia context, stromal ECM, and vascular or metabolic context.")
    lines.append("")
    lines.append("10. Recommended next step")
    lines.append("-" * 100)
    lines.append("Run permutation or label shuffle validation on tier 1 treatment models to estimate how much of the apparent treatment specific signal exceeds chance after screening many treatments.")
    lines.append("")
    lines.append("11. Output files")
    lines.append("-" * 100)
    lines.append(f"Curated treatment summary: {tables_dir / 'curated_treatment_model_summary.tsv'}")
    lines.append(f"Tier 1 treatments: {tables_dir / 'tier1_high_confidence_treatment_models.tsv'}")
    lines.append(f"Tier 2 treatments: {tables_dir / 'tier2_screening_treatment_models.tsv'}")
    lines.append(f"Caution treatments: {tables_dir / 'caution_flags_for_selected_models.tsv'}")
    lines.append(f"Core recurrent features: {tables_dir / 'core_recurrent_spatial_features.tsv'}")
    lines.append(f"Curated themes: {tables_dir / 'curated_cross_treatment_theme_summary.tsv'}")
    lines.append(f"Figures: {figures_dir}")

    write_text_report(report_path, lines)

    summary_json = {
        "full_run_root": str(full_root),
        "output_root": str(output_root),
        "report": str(report_path),
        "n_treatment_summary_rows": int(len(treatment)),
        "n_selected_final_models": int(treatment["selected_for_final_model_bool"].sum()),
        "n_shap_success_models": int(treatment["shap_success_bool"].sum()),
        "n_tier1_high_confidence": int(len(tier1_df)),
        "n_tier2_screening": int(len(tier2_df)),
        "n_tier3_caution": int(len(tier3_df)),
        "source_report_starts_with_filepath": bool(source_report_has_filepath),
        "n_log_files_reviewed": int(len(log_audit)),
        "n_log_files_with_review_status": int((log_audit["status"] == "review").sum()) if "status" in log_audit.columns else None,
    }

    (output_root / "curated_per_treatment_residual_analysis_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("")
    print("=" * 100)
    print("CURATED PER TREATMENT RESIDUAL ANALYSIS COMPLETE")
    print("=" * 100)
    print("Full run root:", full_root)
    print("Curated output root:", output_root)
    print("Report:", report_path)
    print("Selected final models:", summary_json["n_selected_final_models"])
    print("SHAP success models:", summary_json["n_shap_success_models"])
    print("Tier 1 high confidence models:", summary_json["n_tier1_high_confidence"])
    print("Tier 2 screening models:", summary_json["n_tier2_screening"])
    print("Tier 3 caution final models:", summary_json["n_tier3_caution"])
    print("Log files needing review:", summary_json["n_log_files_with_review_status"])
    print("")
    print("Tier counts:")
    print(tier_counts.to_string(index=False))
    print("")
    print("Top Tier 1 treatments:")
    if len(tier1_df) > 0:
        print(tier1_df[show_cols].head(20).to_string(index=False))
    else:
        print("No Tier 1 treatments.")
    print("")
    print("Top core recurrent features:")
    if len(core_features) > 0:
        print(core_features[feature_show_cols].head(20).to_string(index=False))
    else:
        print("No core recurrent feature table.")
    print("")
    print("Top biology themes:")
    if themes is not None and len(themes) > 0:
        print(themes[theme_show_cols].head(15).to_string(index=False))
    else:
        print("No theme table.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
