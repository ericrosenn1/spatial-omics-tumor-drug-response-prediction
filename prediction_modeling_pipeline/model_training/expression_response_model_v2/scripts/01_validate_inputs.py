"""
Script: 01_validate_inputs.py

Purpose:
    Validate the configured expression-response training table before model
    construction.

Project context:
    This is Step 01 of expression_response_model_v2. It verifies that the
    configured training table exists, contains the required case, treatment,
    response, and gene-expression columns, and has enough labeled treatment
    examples to justify downstream deployable model training.

Scientific role:
    This validation step protects the response-modeling workflow from silent
    input drift. It summarizes labeled cases, canonical treatment coverage,
    responder/non-responder counts, detected gene features, and treatments
    eligible for model training under the configured minimum support rules.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP01_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# Input validation uses lightweight file handling plus pandas/numpy
# summaries before any deployable model training occurs.

from pathlib import Path
import argparse
import pandas as pd
import numpy as np



# =========================
# Shared expression-model helper imports
# =========================
# Shared helpers keep config loading, path resolution, response encoding,
# treatment harmonization, and gene-column detection consistent.

from expression_model_v2_lib import (
    load_config,
    resolve_path,
    ensure_dir,
    read_table,
    encode_binary_response,
    canonical_drug_name,
    find_gene_columns,
)




# =========================
# Command-line interface
# =========================
# The runner supplies the YAML config path to every numbered script.

def parse_args():
    """Parse the required YAML config path for Step 01 input validation."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Input validation workflow
# =========================
# This block checks the configured training table and writes reviewable
# validation summaries for downstream model-training decisions.

def main():
    """Validate required columns, gene features, labels, and eligible treatment counts."""

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve all configured paths relative to project_root for reproducibility.
    training_path = resolve_path(cfg, cfg["input_training_table"])
    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    validation_dir = ensure_dir(out_root / "validation")

    # Column names are config-driven so the workflow can tolerate schema variants.
    case_col = cfg.get("case_col", "cases.case_id")
    drug_col = cfg.get("drug_col", "resolved_drug")
    response_col = cfg.get("response_col", "resolved_episode_binary_response")
    gene_prefix = cfg.get("gene_prefix", "ENSG")



# =========================
# Required input checks
# =========================
# Validation accumulates missing-column and missing-feature issues before
# raising a readable error report.

    issues = []

    if not training_path.exists():
        raise FileNotFoundError(training_path)

    # Read as strings to avoid coercing identifiers, treatment labels, or response labels.
    df = read_table(training_path, sep="\t", dtype=str)

    for col in [case_col, drug_col, response_col]:
        if col not in df.columns:
            issues.append(f"missing required column: {col}")

    # Gene-column discovery verifies that the expression matrix is present in this table.
    gene_cols = find_gene_columns(df, gene_prefix=gene_prefix)
    if not gene_cols:
        issues.append(f"no gene columns found with prefix {gene_prefix}")

    if issues:
        summary = "\n".join(["Expression response model v2 input validation", "", "ISSUES:"] + [f"  {x}" for x in issues])
        (validation_dir / "input_validation_summary.txt").write_text(summary, encoding="utf-8")
        raise ValueError(summary)



# =========================
# Minimal validation working table
# =========================
# Only metadata columns and a small gene preview are copied for validation
# summaries; the full expression matrix is not transformed here.

    work = df[[case_col, drug_col, response_col] + gene_cols[:5]].copy()
    # Encode response labels into binary values before counting treatment support.
    work["y"] = df[response_col].map(encode_binary_response)
    # Canonical treatment names align validation counts with later model artifact keys.
    work["canonical_drug"] = df[drug_col].map(canonical_drug_name)
    labeled = work.dropna(subset=["y"]).copy()



# =========================
# Treatment-level label support summary
# =========================
# Responder/non-responder support is summarized after canonical treatment
# harmonization so eligibility reflects downstream model keys.

    drug_summary = (
        labeled.groupby("canonical_drug")
        .agg(
            n_rows=(drug_col, "size"),
            n_cases=(case_col, pd.Series.nunique),
            n_responder=("y", lambda s: int((s == 1).sum())),
            n_non_responder=("y", lambda s: int((s == 0).sum())),
        )
        .reset_index()
        .sort_values(["n_rows", "canonical_drug"], ascending=[False, True])
    )



# =========================
# Eligibility filter
# =========================
# Configured support thresholds define which treatments are plausible
# candidates for deployable expression-response models.

    eligible = drug_summary[
        (drug_summary["n_rows"] >= int(cfg.get("min_rows_per_drug", 30)))
        & (drug_summary["n_cases"] >= int(cfg.get("min_cases_per_drug", 30)))
        & (drug_summary["n_responder"] >= int(cfg.get("min_rows_per_class", 8)))
        & (drug_summary["n_non_responder"] >= int(cfg.get("min_rows_per_class", 8)))
    ].copy()

    # Tabular outputs support audit, troubleshooting, and publication-ready provenance.
    drug_summary.to_csv(validation_dir / "drug_count_summary.tsv", sep="\t", index=False)
    eligible.to_csv(validation_dir / "eligible_drugs_preview.tsv", sep="\t", index=False)



# =========================
# Human-readable validation report
# =========================
# The text report gives reviewers a compact summary of cohort size, label
# coverage, detected gene columns, and top eligible treatments.

    lines = []
    lines.append("Expression response model v2 input validation")
    lines.append("")
    lines.append(f"training_table: {training_path}")
    lines.append(f"rows_total: {len(df)}")
    lines.append(f"rows_labeled: {len(labeled)}")
    lines.append(f"cases_labeled: {labeled[case_col].nunique()}")
    lines.append(f"source_treatments: {df[drug_col].nunique()}")
    lines.append(f"canonical_treatments_labeled: {labeled['canonical_drug'].nunique()}")
    lines.append(f"gene_columns: {len(gene_cols)}")
    lines.append(f"eligible_treatments: {len(eligible)}")
    lines.append("")
    lines.append("Top eligible treatments:")
    for _, row in eligible.head(20).iterrows():
        lines.append(
            f"  {row['canonical_drug']}: rows={row['n_rows']} cases={row['n_cases']} responders={row['n_responder']} nonresponders={row['n_non_responder']}"
        )

    # The plain-text summary is intended for quick terminal and GitHub review.
    (validation_dir / "input_validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print("Wrote:", validation_dir / "input_validation_summary.txt")
    print("Wrote:", validation_dir / "drug_count_summary.tsv")


if __name__ == "__main__":
    main()
