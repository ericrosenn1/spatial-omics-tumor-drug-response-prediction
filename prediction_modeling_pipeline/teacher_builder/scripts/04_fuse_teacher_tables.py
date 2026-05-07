"""
Script: 04_fuse_teacher_tables.py

Purpose:
    Fuse governed expression and histology teacher tables into one sample-by-
    treatment teacher label table.

Project context:
    This is Step 04 of the governed teacher_builder workflow. It consumes the
    expression teacher scores from Step 02, histology teacher scores from Step 03,
    and treatment priors from Step 01. It standardizes the two modality tables,
    merges sample-treatment rows, shrinks each modality probability toward the
    treatment prior, combines reliability-supported deltas, clips the final
    probability, and writes fused teacher outputs for downstream spatial modeling.

Scientific role:
    The fused teacher table is the central response-supervision handoff. It keeps
    treatment priors, modality availability, raw probabilities, shrunk probabilities,
    effective modality weights, residual target values, fusion formula provenance,
    and label-quality flags so downstream spatial models can use governed labels
    rather than unqualified raw teacher probabilities.

Documentation polish marker:
    TEACHER_BUILDER_STEP04_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, paths, thresholds,
    schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 04 uses pandas/numpy plus governance helpers for shrinkage, priors, and label quality.

from pathlib import Path
import argparse
import numpy as np
import pandas as pd



# =========================
# Shared governance helper imports
# =========================
# Fusion relies on shared treatment keys, prior lookup, probability shrinkage, and label quality helpers.

from teacher_governance_lib import (
    load_config,
    ensure_dir,
    read_table,
    write_table,
    normalize_key,
    display_drug_name,
    lookup_prior,
    safe_float,
    confidence_from_probability,
    shrink_probability,
    label_quality_for_row,
    summarize_series,
)




# =========================
# Command-line interface
# =========================
# The governed runner passes the YAML config path into this fusion step.

def parse_args():
    """Parse the governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Expression teacher standardization
# =========================
# Expression outputs are normalized to common sample, treatment, probability, reliability, and confidence columns.

def standardize_expression(expr: pd.DataFrame) -> pd.DataFrame:
    # Empty expression input is allowed so histology-only fusion can still proceed.
    """Normalize expression teacher scores into the fusion-table schema."""

    if expr.empty:
        return pd.DataFrame()

    df = expr.copy()

    # Normalize identifiers before sample-treatment merges.
    df["sample_id"] = df.get("sample_id", df.get("slide_id")).astype(str)
    df["slide_id"] = df.get("slide_id", df["sample_id"]).astype(str)
    # Canonical drug keys define the treatment join unit for fusion.
    df["drug_key"] = df["drug_key"].map(normalize_key)
    df["drug"] = df.get("drug", df["drug_key"].map(display_drug_name))

    # Use responder probability as calibrated probability when upstream scoring did not provide both fields.
    if "expression_prob_calibrated" not in df.columns:
        df["expression_prob_calibrated"] = pd.to_numeric(df.get("expression_prob_responder", np.nan), errors="coerce")

    if "expression_prob_raw" not in df.columns:
        df["expression_prob_raw"] = pd.to_numeric(df["expression_prob_calibrated"], errors="coerce")

    if "expression_reliability_weight" not in df.columns:
        df["expression_reliability_weight"] = pd.to_numeric(df.get("expression_model_weight", 0), errors="coerce").fillna(0)

    # Confidence defaults to distance from 0.5 for expression probabilities.
    if "expression_sample_confidence" not in df.columns:
        df["expression_sample_confidence"] = df["expression_prob_calibrated"].map(confidence_from_probability)

    df["expression_available"] = True

    keep = [
        "sample_id",
        "slide_id",
        "drug",
        "drug_key",
        "expression_available",
        "expression_prob_raw",
        "expression_prob_calibrated",
        "expression_prob_responder",
        "expression_sample_confidence",
        "expression_reliability_weight",
        "expression_model_weight",
        "expression_model_source",
        "expression_teacher_mode",
        "expression_training_rows",
        "expression_auc",
        "expression_brier_improvement",
    ]

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep].drop_duplicates(["sample_id", "drug_key"], keep="first")




# =========================
# Histology teacher standardization
# =========================
# Histology outputs are normalized to the same fusion schema while preserving image-control metadata.

def standardize_histology(hist: pd.DataFrame) -> pd.DataFrame:
    # Empty histology input is allowed so expression-only fusion can still proceed.
    """Normalize histology teacher scores into the fusion-table schema."""

    if hist.empty:
        return pd.DataFrame()

    df = hist.copy()

    df["sample_id"] = df.get("sample_id", df.get("slide_id")).astype(str)
    df["slide_id"] = df.get("slide_id", df["sample_id"]).astype(str)
    df["drug_key"] = df["drug_key"].map(normalize_key)
    df["drug"] = df.get("drug", df["drug_key"].map(display_drug_name))

    # Histology responder probabilities are carried into calibrated probability when no separate calibrated column exists.
    if "histology_prob_calibrated" not in df.columns:
        df["histology_prob_calibrated"] = pd.to_numeric(df.get("histology_prob_responder", np.nan), errors="coerce")

    if "histology_prob_raw" not in df.columns:
        df["histology_prob_raw"] = pd.to_numeric(df["histology_prob_calibrated"], errors="coerce")

    if "histology_reliability_weight" not in df.columns:
        df["histology_reliability_weight"] = pd.to_numeric(df.get("histology_model_weight", 0), errors="coerce").fillna(0)

    # Histology confidence also defaults to distance from uncertainty at 0.5.
    if "histology_sample_confidence" not in df.columns:
        df["histology_sample_confidence"] = df["histology_prob_calibrated"].map(confidence_from_probability)

    df["histology_available"] = True

    keep = [
        "sample_id",
        "slide_id",
        "dataset_id",
        "cancer_type",
        "drug",
        "drug_key",
        "histology_available",
        "histology_prob_raw",
        "histology_prob_calibrated",
        "histology_prob_responder",
        "histology_sample_confidence",
        "histology_reliability_weight",
        "histology_model_weight",
        "histology_blank_control_mean",
        "histology_blank_control_std",
        "histology_noise_control_mean",
        "histology_noise_control_std",
        "histology_control_warning",
        "histology_control_factor",
        "n_tiles_used",
        "hires_image_path",
    ]

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep].drop_duplicates(["sample_id", "drug_key"], keep="first")




# =========================
# Governed teacher fusion workflow
# =========================
# Main workflow: load modality tables, align sample-treatment keys, shrink probabilities, fuse deltas, and write outputs.

def main():
    """Fuse expression and histology teacher tables with treatment-prior shrinkage."""

    args = parse_args()
    cfg = load_config(args.config)

    out_root = Path(cfg["output_root"])
    out_dir = ensure_dir(out_root / "04_fused_teacher")



    # =========================
    # Fusion input paths
    # =========================
    # Step 04 reads expression scores, histology scores, and treatment priors from prior governed steps.

    expr_path = out_root / "02_expression_teacher" / "expression_teacher_scores.tsv"
    hist_path = out_root / "03_histology_teacher" / "histology_teacher_scores.tsv"
    priors_path = out_root / "01_input_validation" / "treatment_priors.tsv"

    expr = read_table(expr_path) if expr_path.exists() else pd.DataFrame()
    hist = read_table(hist_path) if hist_path.exists() else pd.DataFrame()
    priors = read_table(priors_path) if priors_path.exists() else pd.DataFrame()



    # =========================
    # Input standardization
    # =========================
    # Both modality tables are standardized before merge so schema differences do not leak into fusion logic.

    # Standardization makes the downstream merge robust to upstream column naming differences.
    expr = standardize_expression(expr)
    hist = standardize_histology(hist)

    print("")
    print("Governed teacher fusion")
    print("=" * 70)
    print("expression_rows:", len(expr))
    print("histology_rows:", len(hist))



    # =========================
    # Sample-treatment universe
    # =========================
    # The fused table is built from the union of sample-treatment pairs observed in either teacher modality.

    keys = []

    if not expr.empty:
        keys.append(expr[["sample_id", "slide_id", "drug_key", "drug"]])

    if not hist.empty:
        keys.append(hist[["sample_id", "slide_id", "drug_key", "drug"]])

    # A fusion run must have at least one teacher modality row.
    if not keys:
        raise ValueError("No teacher rows available for fusion.")



    # =========================
    # Fusion merge table
    # =========================
    # The base table is merged to expression and histology scores on sample, slide, treatment key, and display name.

    # Use one row per sample-treatment pair before modality-specific scores are merged in.
    base = pd.concat(keys, ignore_index=True).drop_duplicates(["sample_id", "drug_key"], keep="first")

    fused = base.merge(expr, on=["sample_id", "slide_id", "drug_key", "drug"], how="left")
    fused = fused.merge(hist, on=["sample_id", "slide_id", "drug_key", "drug"], how="left", suffixes=("", "_hist"))

    # Missing modality rows are represented as explicit availability flags.
    fused["expression_available"] = fused["expression_available"].fillna(False).astype(bool)
    fused["histology_available"] = fused["histology_available"].fillna(False).astype(bool)

    modality = []
    rows = []



    # =========================
    # Governance thresholds
    # =========================
    # Probability clipping and histology-control shrinkage factors are config-driven.

    low = float(cfg.get("governance", {}).get("probability_clip_low", 0.01))
    high = float(cfg.get("governance", {}).get("probability_clip_high", 0.99))
    default_hist_control = float(cfg.get("governance", {}).get("histology_control_factor_if_warning", 0.5))



    # =========================
    # Row-wise fusion calculation
    # =========================
    # Each sample-treatment row is fused independently so provenance and quality flags remain row-specific.

    for _, r in fused.iterrows():
        if r["expression_available"] and r["histology_available"]:
            m = "both"
        elif r["expression_available"]:
            m = "expression_only"
        elif r["histology_available"]:
            m = "histology_only"
        else:
            m = "none"

        # Treatment priors anchor every fused label before modality deltas are applied.
        prior = lookup_prior(r["drug_key"], priors)



        # =========================
        # Expression shrinkage
        # =========================
        # Expression probability deltas are weighted by model reliability and sample confidence.

        # Expression contribution is reliability- and confidence-weighted around the prior.
        expr_shrink = shrink_probability(
            r.get("expression_prob_calibrated"),
            prior["treatment_prior"],
            r.get("expression_reliability_weight", 0.0),
            r.get("expression_sample_confidence", 0.0),
            1.0,
        )



        # =========================
        # Histology control penalty
        # =========================
        # Histology control warnings can reduce the effective histology contribution before fusion.

        hist_control = safe_float(r.get("histology_control_factor"), np.nan)
        if not np.isfinite(hist_control):
            hist_control = default_hist_control if str(r.get("histology_control_warning", "")).strip() else 1.0

        # Histology contribution uses the same prior-anchored shrinkage contract.
        hist_shrink = shrink_probability(
            r.get("histology_prob_calibrated"),
            prior["treatment_prior"],
            r.get("histology_reliability_weight", 0.0),
            r.get("histology_sample_confidence", 0.0),
            hist_control,
        )



        # =========================
        # Prior-anchored fused probability
        # =========================
        # Fusion starts at the treatment prior and adds only reliability-supported modality deltas.

        # The final score is the prior plus the two supported modality deltas.
        fused_prob = prior["treatment_prior"] + expr_shrink["delta"] + hist_shrink["delta"]
        fused_prob = float(np.clip(fused_prob, low, high))

        out = r.to_dict()
        out.update(prior)
        out["modality_used"] = m
        out["expression_effective_weight"] = expr_shrink["effective_weight"]
        out["histology_effective_weight"] = hist_shrink["effective_weight"]
        out["expression_prob_shrunk"] = expr_shrink["shrunk_prob"]
        out["histology_prob_shrunk"] = hist_shrink["shrunk_prob"]
        out["expression_delta_vs_prior"] = expr_shrink["delta"]
        out["histology_delta_vs_prior"] = hist_shrink["delta"]
        out["histology_control_factor_used"] = hist_control
        out["fused_prob_responder"] = fused_prob
        out["fused_residual_vs_prior"] = float(fused_prob - prior["treatment_prior"])
        out["fused_confidence"] = float(np.clip(expr_shrink["effective_weight"] + hist_shrink["effective_weight"], 0.0, 1.0))
        out["fusion_formula"] = "prior + expression_effective_weight*(expression_prob_calibrated-prior) + histology_effective_weight*(histology_prob_calibrated-prior)"


        # =========================
        # Label quality assignment
        # =========================
        # Label quality flags preserve weak-prior, low-reliability, control-warning, and clipping concerns.

        # Label-quality flags let downstream models filter or stratify weak labels.
        qflag, qreason = label_quality_for_row(pd.Series(out))
        out["label_quality_flag"] = qflag
        out["label_quality_reason"] = qreason
        rows.append(out)

    out = pd.DataFrame(rows)



    # =========================
    # Primary fused output writing
    # =========================
    # The full fused table is written under both canonical and compatibility filenames.

    # Write both canonical and compatibility filenames for downstream consumers.
    write_table(out, out_dir / "fused_teacher_table.tsv")
    write_table(out, out_dir / "visium_fused_teacher_table.tsv")



    # =========================
    # Sample-level summary
    # =========================
    # Sample summaries describe row counts, treatment counts, fused probabilities, confidence, and label quality.

    by_sample = (
        out.groupby("sample_id", as_index=False)
        .agg(
            n_rows=("drug_key", "size"),
            n_treatments=("drug_key", "nunique"),
            mean_fused_prob_responder=("fused_prob_responder", "mean"),
            mean_fused_residual_vs_prior=("fused_residual_vs_prior", "mean"),
            mean_fused_confidence=("fused_confidence", "mean"),
            n_ok=("label_quality_flag", lambda s: int((s == "ok").sum())),
            n_warn=("label_quality_flag", lambda s: int((s == "warn").sum())),
            n_exclude=("label_quality_flag", lambda s: int((s == "exclude").sum())),
        )
    )



    # =========================
    # Treatment-level summary
    # =========================
    # Treatment summaries describe modality coverage, priors, fused response distributions, and effective weights.

    by_drug = (
        out.groupby("drug_key", as_index=False)
        .agg(
            drug=("drug", "first"),
            n_samples=("sample_id", "nunique"),
            n_rows=("sample_id", "size"),
            n_both=("modality_used", lambda s: int((s == "both").sum())),
            n_expression_only=("modality_used", lambda s: int((s == "expression_only").sum())),
            n_histology_only=("modality_used", lambda s: int((s == "histology_only").sum())),
            mean_treatment_prior=("treatment_prior", "mean"),
            mean_fused_prob_responder=("fused_prob_responder", "mean"),
            std_fused_prob_responder=("fused_prob_responder", "std"),
            mean_fused_residual_vs_prior=("fused_residual_vs_prior", "mean"),
            mean_expression_effective_weight=("expression_effective_weight", "mean"),
            mean_histology_effective_weight=("histology_effective_weight", "mean"),
        )
    ).sort_values("mean_fused_prob_responder", ascending=False)



    # =========================
    # Missingness and run summary
    # =========================
    # Missingness and compact run summaries make downstream QC easier to audit.

    missingness = (
        out.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "n_missing"})
    )
    missingness["fraction_missing"] = missingness["n_missing"] / max(len(out), 1)

    summary = pd.DataFrame(
        [
            {
                "n_fused_rows": len(out),
                "n_samples": out["sample_id"].nunique(),
                "n_drugs": out["drug_key"].nunique(),
                "n_expression_rows": len(expr),
                "n_histology_rows": len(hist),
                "n_modality_both": int((out["modality_used"] == "both").sum()),
                "n_modality_expression_only": int((out["modality_used"] == "expression_only").sum()),
                "n_modality_histology_only": int((out["modality_used"] == "histology_only").sum()),
                "mean_fused_prob_responder": float(out["fused_prob_responder"].mean()),
                "std_fused_prob_responder": float(out["fused_prob_responder"].std()),
                "min_fused_prob_responder": float(out["fused_prob_responder"].min()),
                "max_fused_prob_responder": float(out["fused_prob_responder"].max()),
                "mean_fused_residual_vs_prior": float(out["fused_residual_vs_prior"].mean()),
                "n_label_ok": int((out["label_quality_flag"] == "ok").sum()),
                "n_label_warn": int((out["label_quality_flag"] == "warn").sum()),
                "n_label_exclude": int((out["label_quality_flag"] == "exclude").sum()),
            }
        ]
    )

    write_table(by_sample, out_dir / "fused_teacher_by_sample.tsv")
    write_table(by_drug, out_dir / "fused_teacher_by_drug.tsv")
    write_table(missingness, out_dir / "fused_teacher_missingness.tsv")
    write_table(summary, out_dir / "fused_teacher_summary.tsv")
    write_table(out, out_dir / "teacher_fusion_audit.tsv")

    lines = ["Fused teacher governed summary", ""]
    for k, v in summary.iloc[0].to_dict().items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("modality_used:")
    lines.append(out["modality_used"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("label_quality_flag:")
    lines.append(out["label_quality_flag"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("top treatments:")
    lines.append(by_drug.head(10)[["drug", "mean_fused_prob_responder", "std_fused_prob_responder", "mean_treatment_prior"]].to_string(index=False))

    summary_text = "\n".join(lines)
    (out_dir / "fused_teacher_summary.txt").write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print("DONE")
    print(out_dir)


if __name__ == "__main__":
    main()
