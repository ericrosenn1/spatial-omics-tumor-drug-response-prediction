"""
Script: 02_build_canonical_training_table.py

Purpose:
    Build the canonical case-by-treatment expression-response training table.

Project context:
    This is Step 02 of expression_response_model_v2. It reads the configured
    GDC expression-response training table, standardizes treatment names,
    encodes binary response labels, removes ambiguous case-treatment labels,
    writes canonical metadata, and records the expression gene columns used by
    deployable model training.

Scientific role:
    This step converts a large clinical-expression source table into an
    auditable modeling table. It enforces stable case identifiers, canonical
    treatment keys, binary response labels, conflict removal, deduplication,
    and explicit gene-feature provenance before any classifier is trained.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP02_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# Canonical table construction uses pandas/numpy plus shared
# expression_model_v2_lib helpers for consistent labels and paths.

from pathlib import Path
import argparse
import pandas as pd
import numpy as np



# =========================
# Shared expression-model helper imports
# =========================
# These helpers centralize config handling, text cleanup, response
# encoding, treatment harmonization, and gene-column discovery.

from expression_model_v2_lib import (
    load_config,
    resolve_path,
    ensure_dir,
    read_table,
    clean_text,
    encode_binary_response,
    canonical_drug_name,
    normalize_key,
    find_gene_columns,
)




# =========================
# Command-line interface
# =========================
# The runner passes the same YAML config path through all numbered steps.

def parse_args():
    """Parse the required YAML config path for Step 02 canonical table construction."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Canonical training-table workflow
# =========================
# This block creates the model-ready expression-response table and all
# audit sidecars needed to understand filtering decisions.

def main():
    """Build canonical expression-response training, metadata, conflict, deduplication, and gene-column outputs."""

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve configured input/output paths relative to project_root.
    training_path = resolve_path(cfg, cfg["input_training_table"])
    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    data_dir = ensure_dir(out_root / "data")

    # Column names are config-driven so this step can tolerate source-table schema variants.
    case_col = cfg.get("case_col", "cases.case_id")
    drug_col = cfg.get("drug_col", "resolved_drug")
    response_col = cfg.get("response_col", "resolved_episode_binary_response")
    gene_prefix = cfg.get("gene_prefix", "ENSG")



    # =========================
    # Canonical output paths
    # =========================
    # Step 02 writes the primary training table plus metadata, conflict,
    # deduplication, and gene-feature provenance files.

    canonical_path = data_dir / "training_table_canonical.tsv"
    metadata_path = data_dir / "training_metadata_canonical.tsv"
    conflict_path = data_dir / "conflicting_case_drug_labels.tsv"
    dedup_report_path = data_dir / "deduplication_report.tsv"
    gene_columns_path = data_dir / "gene_columns.txt"

    # Read the source table as strings to preserve identifiers and raw labels.
    df = read_table(training_path, sep="\t", dtype=str)



    # =========================
    # Required column validation
    # =========================
    # The canonical table requires stable case IDs, raw treatment labels,
    # binary-response source labels, and detectable expression columns.

    required = [case_col, drug_col, response_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Persisting the detected gene list makes the trained model feature space auditable.
    gene_cols = find_gene_columns(df, gene_prefix=gene_prefix)
    if not gene_cols:
        raise ValueError("No expression gene columns detected.")



    # =========================
    # Core text normalization
    # =========================
    # Identifiers, treatment labels, and response labels are cleaned before
    # canonical keys and model labels are derived.

    for c in [case_col, drug_col, response_col]:
        df[c] = df[c].map(clean_text)

    # Preserve the raw treatment label beside the canonical treatment representation.
    df["resolved_drug_original"] = df[drug_col]
    # Canonical drug names define the treatment keys used for model grouping.
    df["canonical_drug"] = df[drug_col].map(canonical_drug_name)
    df["drug_key"] = df["canonical_drug"].map(normalize_key)
    # Convert source response labels into the binary target used by classifiers.
    df["y"] = df[response_col].map(encode_binary_response)

    before = len(df)
    # Rows without interpretable binary response labels cannot supervise model training.
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)



    # =========================
    # Conflict detection
    # =========================
    # Case-treatment pairs with contradictory response labels are exported
    # and removed to avoid training on ambiguous supervision.

    grouped = (
        df.groupby([case_col, "drug_key"])["y"]
        .nunique()
        .reset_index(name="n_distinct_labels")
    )

    conflict_keys = grouped[grouped["n_distinct_labels"] > 1][[case_col, "drug_key"]]
    if len(conflict_keys):
        conflict_df = df.merge(conflict_keys, on=[case_col, "drug_key"], how="inner")
    else:
        conflict_df = pd.DataFrame(columns=df.columns)

    # Export conflicting labels even though they are removed from the canonical table.
    conflict_df.to_csv(conflict_path, sep="\t", index=False)

    if len(conflict_keys):
        df = df.merge(conflict_keys.assign(_conflict=1), on=[case_col, "drug_key"], how="left")
        df = df[df["_conflict"].isna()].drop(columns=["_conflict"]).copy()



    # =========================
    # Deduplication and confidence ordering
    # =========================
    # When available, episode confidence and episode key guide which duplicate
    # case-treatment-response row is retained.

    before_dedup = len(df)
    sort_cols = []
    if "episode_confidence" in df.columns:
        df["_episode_confidence_rank"] = df["episode_confidence"].map(lambda x: 0 if str(x).lower() == "high" else 1)
        sort_cols.append("_episode_confidence_rank")
    if "episode_key" in df.columns:
        sort_cols.append("episode_key")

    if sort_cols:
        df = df.sort_values(sort_cols)

    # Retain one representative row per case, canonical treatment, and binary label.
    df = df.drop_duplicates(subset=[case_col, "drug_key", "y"], keep="first").copy()

    if "_episode_confidence_rank" in df.columns:
        df = df.drop(columns=["_episode_confidence_rank"])



    # =========================
    # Metadata sidecar construction
    # =========================
    # The metadata file retains review fields useful for provenance, cohort
    # inspection, and downstream audit without altering the modeling table.

    metadata_cols = [
        c for c in [
            "episode_key",
            "episode_number",
            case_col,
            "cases.submitter_id",
            "project.project_id",
            "cases.primary_site",
            "cases.disease_type",
            "diagnoses.primary_diagnosis",
            "diagnoses.ajcc_clinical_stage",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.tumor_grade",
            "demographic.age_at_index",
            "demographic.gender",
            "demographic.race",
            "demographic.ethnicity",
            "demographic.vital_status",
            "resolved_drug_original",
            "canonical_drug",
            "drug_key",
            response_col,
            "y",
        ]
        if c in df.columns
    ]

    metadata = df[metadata_cols].copy()
    # Metadata sidecars make downstream model artifacts traceable to source clinical fields.
    metadata.to_csv(metadata_path, sep="\t", index=False)

    df.to_csv(canonical_path, sep="\t", index=False)
    # The gene-column list is the explicit expression feature-space contract for Step 03.
    gene_columns_path.write_text("\n".join(gene_cols), encoding="utf-8")



    # =========================
    # Deduplication report
    # =========================
    # The report records row counts, removed conflicts, duplicate removals,
    # canonical treatment counts, and gene-feature counts.

    dedup_report = pd.DataFrame([
        {"metric": "input_rows", "value": before},
        {"metric": "labeled_rows", "value": int((pd.read_csv(training_path, sep="\t", usecols=[response_col], dtype=str)[response_col].map(encode_binary_response).notna()).sum())},
        {"metric": "rows_after_label_filter", "value": before_dedup + len(conflict_df)},
        {"metric": "conflicting_case_drug_rows_removed", "value": len(conflict_df)},
        {"metric": "conflicting_case_drug_pairs", "value": len(conflict_keys)},
        {"metric": "duplicate_case_drug_response_rows_removed", "value": before_dedup - len(df)},
        {"metric": "canonical_rows", "value": len(df)},
        {"metric": "canonical_cases", "value": df[case_col].nunique()},
        {"metric": "canonical_drugs", "value": df["drug_key"].nunique()},
        {"metric": "gene_columns", "value": len(gene_cols)},
    ])
    dedup_report.to_csv(dedup_report_path, sep="\t", index=False)

    print("DONE")
    print("Input rows:", before)
    print("Canonical rows:", len(df))
    print("Canonical cases:", df[case_col].nunique())
    print("Canonical drugs:", df["drug_key"].nunique())
    print("Conflicting rows removed:", len(conflict_df))
    print("Wrote:", canonical_path)
    print("Wrote:", metadata_path)
    print("Wrote:", dedup_report_path)
    print("Wrote:", gene_columns_path)


if __name__ == "__main__":
    main()
