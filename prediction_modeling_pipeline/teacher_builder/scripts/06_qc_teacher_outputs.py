"""
Script: 06_qc_teacher_outputs.py

Purpose:
    Run final governed teacher_builder quality-control summaries and figures.

Project context:
    This is Step 06 of the governed teacher_builder workflow. It reads the
    prediction-ready teacher outputs from Step 05, computes sample-, treatment-,
    feature-, and run-level QC summaries, applies warning/failure checks for
    saturated or missing teacher labels, creates diagnostic figures, and writes
    the final QC decision.

Scientific role:
    This step does not modify modeling inputs or train a model. It is an audit
    and reporting layer that helps reviewers determine whether the governed
    teacher labels are usable for downstream spatial prediction. The checks focus
    on exact 0/1 saturation, treatment-prior availability, residual-target
    availability, label-quality fields, and excessive excluded labels.

Documentation polish marker:
    TEACHER_BUILDER_STEP06_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, paths, thresholds,
    schemas, figure definitions, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 06 uses pandas/numpy summaries and matplotlib figures for final teacher QC.

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import matplotlib


# =========================
# Headless plotting backend
# =========================
# The Agg backend allows figure generation from PowerShell or batch runs without an interactive display.

# Use a non-interactive backend so QC figures can be generated in scripted runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# =========================
# Shared governance helper imports
# =========================
# Shared helpers keep config loading, directory creation, table IO, and numeric summaries consistent.

from teacher_governance_lib import (
    load_config,
    ensure_dir,
    read_table,
    write_table,
    summarize_series,
)




# =========================
# Command-line interface
# =========================
# The governed runner passes the YAML config path into this final QC step.

def parse_args():
    """Parse the governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Numeric coercion helper
# =========================
# QC summaries repeatedly coerce table columns to numeric values while preserving NaN for invalid entries.

def numeric(s):
    # Invalid values become NaN so summaries do not fail on mixed-type columns.
    """Coerce a Series-like object to numeric values with invalid entries set to NaN."""

    return pd.to_numeric(s, errors="coerce")




# =========================
# Figure writer helper
# =========================
# All QC plots use the same save/close behavior to prevent overlapping figures.

def savefig(path):
    # Tight layout keeps axis labels readable across all saved QC figures.
    """Save the current matplotlib figure with consistent layout, resolution, and cleanup."""

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()




# =========================
# QC figure generation
# =========================
# Figures visualize fused probabilities, residuals, modality composition, label quality, priors, heatmaps, and shrinkage.

def make_figures(teacher, training, manifest, out_dir):


    # =========================
    # Fused probability histogram
    # =========================
    # The first diagnostic shows the overall distribution of governed response labels.

    # Main label distribution should be bounded and not collapsed at extremes.
    """Generate governed teacher QC figures from teacher, training, and manifest tables."""

    vals = numeric(teacher["fused_prob_responder"]).dropna()
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=30)
    plt.xlabel("Fused probability of response")
    plt.ylabel("Sample by treatment rows")
    plt.title("Governed fused teacher probabilities")
    savefig(out_dir / "fig_01_fused_probability_histogram.png")



    # =========================
    # Residual target histogram
    # =========================
    # Residuals show how far governed teacher labels move away from treatment priors.

    # Residual targets are the downstream prior-adjusted supervision signal.
    if "fused_residual_vs_prior" in teacher.columns:
        vals = numeric(teacher["fused_residual_vs_prior"]).dropna()
        plt.figure(figsize=(7, 5))
        plt.hist(vals, bins=30)
        plt.xlabel("Fused probability minus treatment prior")
        plt.ylabel("Sample by treatment rows")
        plt.title("Teacher residual versus treatment prior")
        savefig(out_dir / "fig_02_fused_residual_histogram.png")



    # =========================
    # Modality composition figure
    # =========================
    # Modality counts show whether labels are expression-only, histology-only, or fused from both.

    # Modality composition confirms whether teacher labels are expression-only, histology-only, or fused.
    if "modality_used" in teacher.columns:
        counts = teacher["modality_used"].astype(str).value_counts()
        plt.figure(figsize=(7, 5))
        plt.bar(counts.index, counts.values)
        plt.ylabel("Rows")
        plt.title("Teacher modality composition")
        plt.xticks(rotation=30, ha="right")
        savefig(out_dir / "fig_03_modality_counts.png")



    # =========================
    # Label-quality figure
    # =========================
    # Label-quality counts make weak or excluded labels visible before downstream modeling.

    # Label-quality flags expose weak, clipped, or excluded labels for reviewer triage.
    if "label_quality_flag" in teacher.columns:
        counts = teacher["label_quality_flag"].astype(str).value_counts()
        plt.figure(figsize=(7, 5))
        plt.bar(counts.index, counts.values)
        plt.ylabel("Rows")
        plt.title("Label quality flags")
        savefig(out_dir / "fig_04_label_quality_flags.png")



    # =========================
    # Prior-versus-fused scatter
    # =========================
    # This figure checks whether fused labels remain anchored to treatment priors.

    # Prior-versus-fused scatter checks whether fusion remains anchored to treatment priors.
    if "treatment_prior" in teacher.columns:
        plt.figure(figsize=(6, 6))
        plt.scatter(numeric(teacher["treatment_prior"]), numeric(teacher["fused_prob_responder"]), s=12)
        plt.xlabel("Treatment prior")
        plt.ylabel("Fused probability")
        plt.title("Treatment prior versus fused probability")
        savefig(out_dir / "fig_05_prior_vs_fused.png")



    # =========================
    # Sample-treatment heatmap
    # =========================
    # The heatmap visualizes the governed response surface across samples and treatments.

    # Heatmap summarizes the full sample-by-treatment teacher response surface.
    heat = teacher.pivot_table(index="sample_id", columns="drug_key", values="fused_prob_responder", aggfunc="mean")
    if heat.shape[0] and heat.shape[1]:
        plt.figure(figsize=(max(8, heat.shape[1] * 0.25), max(6, heat.shape[0] * 0.08)))
        plt.imshow(heat.fillna(np.nan).to_numpy(), aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Fused probability")
        plt.yticks(range(len(heat.index)), heat.index.astype(str), fontsize=4)
        plt.xticks(range(len(heat.columns)), heat.columns.astype(str), rotation=90, fontsize=5)
        plt.title("Sample by treatment governed teacher heatmap")
        savefig(out_dir / "fig_06_sample_treatment_heatmap.png")



    # =========================
    # Treatment mean-response figure
    # =========================
    # Treatment-level means highlight high- and low-response teacher distributions.

    # Treatment means identify therapies with consistently high or low governed response scores.
    by_drug = (
        teacher.groupby("drug_key", as_index=False)
        .agg(mean_fused_prob_responder=("fused_prob_responder", "mean"))
        .sort_values("mean_fused_prob_responder", ascending=False)
        .head(30)
    )
    plt.figure(figsize=(8, max(5, len(by_drug) * 0.25)))
    plt.barh(by_drug["drug_key"].astype(str), by_drug["mean_fused_prob_responder"])
    plt.xlabel("Mean fused response")
    plt.title("Mean governed response by treatment")
    plt.gca().invert_yaxis()
    savefig(out_dir / "fig_07_treatment_mean_response.png")



    # =========================
    # Expression shrinkage diagnostic
    # =========================
    # Raw-versus-shrunk expression plots show how governance changes expression probabilities.

    # Expression shrinkage diagnostic compares raw/calibrated and governed expression probabilities.
    if "expression_prob_raw" in teacher.columns and "expression_prob_shrunk" in teacher.columns:
        sub = teacher.dropna(subset=["expression_prob_raw", "expression_prob_shrunk"])
        if len(sub):
            plt.figure(figsize=(6, 6))
            plt.scatter(numeric(sub["expression_prob_raw"]), numeric(sub["expression_prob_shrunk"]), s=12)
            plt.xlabel("Expression raw or calibrated")
            plt.ylabel("Expression shrunk")
            plt.title("Expression teacher shrinkage")
            savefig(out_dir / "fig_08_expression_raw_vs_shrunk.png")



    # =========================
    # Histology shrinkage diagnostic
    # =========================
    # Raw-versus-shrunk histology plots show how governance changes image-derived probabilities.

    # Histology shrinkage diagnostic compares raw and governed histology probabilities.
    if "histology_prob_raw" in teacher.columns and "histology_prob_shrunk" in teacher.columns:
        sub = teacher.dropna(subset=["histology_prob_raw", "histology_prob_shrunk"])
        if len(sub):
            plt.figure(figsize=(6, 6))
            plt.scatter(numeric(sub["histology_prob_raw"]), numeric(sub["histology_prob_shrunk"]), s=12)
            plt.xlabel("Histology raw")
            plt.ylabel("Histology shrunk")
            plt.title("Histology teacher shrinkage")
            savefig(out_dir / "fig_09_histology_raw_vs_shrunk.png")




# =========================
# Teacher QC workflow
# =========================
# Main workflow: load prediction-ready outputs, summarize tables, apply checks, write QC artifacts, and print the report.

def main():
    """Build QC tables, figures, summary text, decision file, and run configuration."""

    args = parse_args()
    cfg = load_config(args.config)



    # =========================
    # Output directory setup
    # =========================
    # QC artifacts are written under the governed output root in 06_teacher_qc.

    # All QC outputs are grouped under the governed teacher_builder output root.
    out_root = Path(cfg["output_root"])
    out_dir = ensure_dir(out_root / "06_teacher_qc")



    # =========================
    # Prediction-ready input paths
    # =========================
    # Step 06 consumes the model input, teacher table, training table, and feature manifest written by Step 05.

    # Step 05 outputs are the required inputs for this final QC pass.
    base = out_root / "05_prediction_ready_teacher"
    model_input = read_table(base / "model_input_numeric.csv")
    teacher = read_table(base / "visium_fused_teacher_table.tsv")
    training = read_table(base / "prediction_ready_training_table.tsv")
    manifest = read_table(base / "feature_manifest.csv")

    # Reuse numeric fused probabilities for checks and run-level summaries.
    vals = numeric(teacher["fused_prob_responder"])



    # =========================
    # Sample-level QC summary
    # =========================
    # Sample summaries report treatment counts, fused probabilities, residuals, and confidence per sample.

    # Sample-level summaries help identify samples with unusual teacher distributions.
    by_sample = (
        teacher.groupby("sample_id", as_index=False)
        .agg(
            n_teacher_rows=("drug_key", "size"),
            n_treatments=("drug_key", "nunique"),
            mean_fused_prob_responder=("fused_prob_responder", "mean"),
            median_fused_prob_responder=("fused_prob_responder", "median"),
            min_fused_prob_responder=("fused_prob_responder", "min"),
            max_fused_prob_responder=("fused_prob_responder", "max"),
            mean_fused_residual_vs_prior=("fused_residual_vs_prior", "mean"),
            mean_fused_confidence=("fused_confidence", "mean"),
        )
    )



    # =========================
    # Treatment-level QC summary
    # =========================
    # Treatment summaries report response distributions, prior values, modality counts, and effective weights.

    # Treatment-level summaries help identify saturated or highly variable drug labels.
    by_treatment = (
        teacher.groupby("drug_key", as_index=False)
        .agg(
            drug=("drug", "first") if "drug" in teacher.columns else ("drug_key", "first"),
            n_samples=("sample_id", "nunique"),
            mean_fused_prob_responder=("fused_prob_responder", "mean"),
            median_fused_prob_responder=("fused_prob_responder", "median"),
            std_fused_prob_responder=("fused_prob_responder", "std"),
            min_fused_prob_responder=("fused_prob_responder", "min"),
            max_fused_prob_responder=("fused_prob_responder", "max"),
            mean_treatment_prior=("treatment_prior", "mean"),
            mean_fused_residual_vs_prior=("fused_residual_vs_prior", "mean"),
            mean_expression_effective_weight=("expression_effective_weight", "mean"),
            mean_histology_effective_weight=("histology_effective_weight", "mean"),
            n_modality_both=("modality_used", lambda s: int((s == "both").sum())),
            n_modality_expression_only=("modality_used", lambda s: int((s == "expression_only").sum())),
            n_modality_histology_only=("modality_used", lambda s: int((s == "histology_only").sum())),
        )
        .sort_values("mean_fused_prob_responder", ascending=False)
    )



    # =========================
    # Feature-level QC summary
    # =========================
    # Feature summaries report missingness, uniqueness, and numeric range for downstream model features.

    # Feature QC mirrors the numeric model input table delivered to downstream models.
    feature_rows = []
    sample_col = cfg.get("sample_col", "sample_id")
    for c in model_input.columns:
        if c == sample_col:
            continue
        s = numeric(model_input[c])
        feature_rows.append(
            {
                "feature": c,
                "nonmissing_fraction": float(s.notna().mean()),
                "missing_fraction": float(s.isna().mean()),
                "n_unique": int(s.dropna().nunique()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().any() else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )

    by_feature = pd.DataFrame(feature_rows)



    # =========================
    # QC check registry
    # =========================
    # Checks are collected in a table so PASS/WARN/FAIL decisions remain auditable.

    # Checks are collected before a final PASS/WARN/FAIL decision is assigned.
    checks = []



    # =========================
    # QC check writer helper
    # =========================
    # Each check records a boolean result, observed value, and reviewer-facing detail.

    def add_check(name, passed, value, detail):
        # Store every check as a row for qc_checks.tsv.
        """Append one named QC check with boolean pass status, value, and explanation."""

        checks.append({"check": name, "passed": bool(passed), "value": value, "detail": detail})



    # =========================
    # Core probability saturation checks
    # =========================
    # Core checks guard against exact 0/1 labels, extreme dominance, saturated treatments, and missing governance fields.

    # Exact 0/1 probabilities indicate ungoverned saturation and are hard-fail checks.
    add_check("no_exact_zero_or_one", int(((vals <= 0) | (vals >= 1)).sum()) == 0, int(((vals <= 0) | (vals >= 1)).sum()), "fused probabilities should be clipped away from exact 0 and 1")
    add_check("not_many_extreme_values", float(((vals < 0.02) | (vals > 0.98)).mean()) < 0.25, float(((vals < 0.02) | (vals > 0.98)).mean()), "extreme fused probabilities should not dominate")
    # Saturated treatments are near-constant at extreme mean response probabilities.
    saturated_treatments = by_treatment[
        ((by_treatment["mean_fused_prob_responder"] > 0.98) | (by_treatment["mean_fused_prob_responder"] < 0.02))
        & (by_treatment["std_fused_prob_responder"].fillna(0) < 0.02)
    ]
    add_check("no_saturated_treatments", len(saturated_treatments) == 0, int(len(saturated_treatments)), "treatments should not be near constant at extremes")
    add_check("treatment_prior_present", "treatment_prior" in teacher.columns and numeric(teacher["treatment_prior"]).notna().all(), int(numeric(teacher.get("treatment_prior", pd.Series(dtype=float))).notna().sum()), "all rows require priors")
    add_check("residual_target_present", "fused_residual_vs_prior" in teacher.columns, "present" if "fused_residual_vs_prior" in teacher.columns else "missing", "downstream residual target should be available")
    add_check("label_quality_present", "label_quality_flag" in teacher.columns, "present" if "label_quality_flag" in teacher.columns else "missing", "label quality flags should be present")

    if "label_quality_flag" in teacher.columns:
        exclude_frac = float((teacher["label_quality_flag"] == "exclude").mean())
        add_check("low_exclude_fraction", exclude_frac < 0.10, exclude_frac, "too many excluded labels means fusion is not usable")



    # =========================
    # QC decision logic
    # =========================
    # Overall decision is PASS, WARN, or FAIL based on required and warning-level checks.

    checks_df = pd.DataFrame(checks)
    # Any warning-level check failure downgrades PASS to WARN before hard-fail checks are applied.
    decision = "PASS" if checks_df["passed"].all() else "WARN"
    # Missing priors/residuals or exact saturation invalidates the teacher handoff.
    if not checks_df.loc[checks_df["check"].isin(["no_exact_zero_or_one", "treatment_prior_present", "residual_target_present"]), "passed"].all():
        decision = "FAIL"



    # =========================
    # Run-level summary table
    # =========================
    # The compact summary captures counts, score moments, extreme fractions, and saturated treatment count.

    summary = pd.DataFrame(
        [
            {
                "qc_decision": decision,
                "n_model_input_samples": int(model_input[sample_col].nunique()) if sample_col in model_input.columns else int(len(model_input)),
                "n_numeric_features": int(model_input.shape[1] - 1),
                "n_teacher_rows": int(len(teacher)),
                "n_teacher_samples": int(teacher["sample_id"].nunique()),
                "n_teacher_treatments": int(teacher["drug_key"].nunique()),
                "n_training_rows": int(len(training)),
                "n_training_samples": int(training["sample_id"].nunique()),
                "n_duplicate_teacher_sample_treatment": int(teacher.duplicated(["sample_id", "drug_key"]).sum()),
                "mean_fused_prob_responder": float(vals.mean()),
                "std_fused_prob_responder": float(vals.std()),
                "min_fused_prob_responder": float(vals.min()),
                "max_fused_prob_responder": float(vals.max()),
                "fraction_extreme_lt_0_02_or_gt_0_98": float(((vals < 0.02) | (vals > 0.98)).mean()),
                "mean_fused_residual_vs_prior": float(numeric(teacher["fused_residual_vs_prior"]).mean()) if "fused_residual_vs_prior" in teacher.columns else np.nan,
                "n_saturated_treatments": int(len(saturated_treatments)),
            }
        ]
    )



    # =========================
    # QC table output writing
    # =========================
    # Step 06 writes summary, check, sample, treatment, feature, and fusion-audit tables.

    # Tabular QC outputs are intended for reproducible downstream audit.
    write_table(summary, out_dir / "qc_summary.tsv")
    write_table(checks_df, out_dir / "qc_checks.tsv")
    write_table(by_sample, out_dir / "qc_by_sample.tsv")
    write_table(by_treatment, out_dir / "qc_by_treatment.tsv")
    write_table(by_feature, out_dir / "qc_by_feature.tsv")
    write_table(teacher, out_dir / "teacher_fusion_audit.tsv")



    # =========================
    # QC figure rendering
    # =========================
    # All diagnostic PNGs are regenerated from the current prediction-ready tables.

    # Figures provide quick visual diagnostics but do not alter any modeling inputs.
    make_figures(teacher, training, manifest, out_dir)



    # =========================
    # Human-readable QC report
    # =========================
    # The text report mirrors the tabular QC outputs for quick review.

    lines = []
    lines.append("Teacher builder governed QC report")
    lines.append("")
    lines.append(f"QC decision: {decision}")
    lines.append("")
    lines.append("Core counts:")
    for k, v in summary.iloc[0].to_dict().items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("QC checks:")
    for _, r in checks_df.iterrows():
        status = "PASS" if r["passed"] else "WARN"
        lines.append(f"  {status} {r['check']}: {r['value']} | {r['detail']}")
    lines.append("")
    lines.append("Modality counts:")
    if "modality_used" in teacher.columns:
        lines.append(teacher["modality_used"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Label quality counts:")
    if "label_quality_flag" in teacher.columns:
        lines.append(teacher["label_quality_flag"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Top treatments by mean fused response:")
    lines.append(by_treatment.head(12)[["drug_key", "mean_fused_prob_responder", "std_fused_prob_responder", "mean_treatment_prior", "mean_fused_residual_vs_prior"]].to_string(index=False))
    lines.append("")
    lines.append("Figures:")
    for p in sorted(out_dir.glob("fig_*.png")):
        lines.append(f"  {p.name}")

    # The text summary is the reviewer-facing QC artifact.
    text = "\n".join(lines)
    (out_dir / "qc_summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "teacher_qc_decision.txt").write_text(decision, encoding="utf-8")



    # =========================
    # Machine-readable QC run configuration
    # =========================
    # The JSON run config records the input config, output directory, and final QC decision.

    with open(out_dir / "qc_run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": str(Path(args.config).resolve()), "output_dir": str(out_dir), "qc_decision": decision}, f, indent=2)

    print("")
    print(text)
    print("")
    print("DONE")
    print(out_dir)


if __name__ == "__main__":
    main()
