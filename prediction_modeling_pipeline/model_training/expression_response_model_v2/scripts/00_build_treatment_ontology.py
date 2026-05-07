"""
Script: 00_build_treatment_ontology.py

Purpose:
    Build the treatment ontology and alias table used by expression_response_model_v2.

Project context:
    This is Step 00 of the deployable expression response model workflow. It reads
    the configured GDC expression-response training table, extracts raw treatment
    labels, maps them to canonical drug names and regimen keys, and writes the
    ontology artifacts used by later validation, canonical table construction,
    model training, audit, and Visium teacher scoring steps.

Scientific role:
    Treatment harmonization is a core reproducibility requirement for response
    modeling. Raw clinical treatment names can differ by capitalization, salt form,
    brand name, punctuation, or combination-regimen notation. This step makes the
    mapping explicit so downstream response labels and trained model artifacts use
    stable treatment keys rather than ad hoc text labels.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP00_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 00 only needs lightweight file, argument, pandas, and shared
# expression_model_v2_lib helpers.

from pathlib import Path
import argparse
import pandas as pd



# =========================
# Shared expression-model helper imports
# =========================
# These helpers keep config loading, path resolution, text cleanup,
# and treatment harmonization consistent across all numbered steps.

from expression_model_v2_lib import (
    load_config,
    resolve_path,
    ensure_dir,
    read_table,
    clean_text,
    normalize_key,
    canonical_drug_name,
    canonical_regimen_key,
    DEFAULT_ALIASES,
)




# =========================
# Command-line interface
# =========================
# The runner passes the YAML config path into every numbered script.

def parse_args():
    """Parse the required YAML config path for Step 00 ontology construction."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Treatment ontology construction
# =========================
# This block reads raw treatment labels, creates canonical keys, and
# writes auditable ontology tables for downstream model steps.

def main():
    """Build and write canonical treatment ontology and alias tables from the configured training table."""

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve configured paths relative to project_root for portable execution.
    training_path = resolve_path(cfg, cfg["input_training_table"])
    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    ontology_path = out_root / "treatment_ontology.tsv"
    alias_path = out_root / "treatment_aliases.tsv"

    # The configured drug column contains the raw clinical treatment labels.
    drug_col = cfg.get("drug_col", "resolved_drug")

    if not training_path.exists():
        raise FileNotFoundError(training_path)

    # Read as strings so treatment labels are not coerced or reformatted.
    df = read_table(training_path, sep="\t", dtype=str)

    if drug_col not in df.columns:
        raise ValueError(f"Training table missing drug column: {drug_col}")



# =========================
# Raw treatment discovery
# =========================
# Only non-empty unique source labels are retained before canonicalization.

    drugs = (
        df[drug_col]
        .dropna()
        .map(clean_text)
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .sort_values()
    )



# =========================
# Canonical ontology rows
# =========================
# Each raw source treatment is paired with canonical single-agent and
# regimen-level representations.

    rows = []
    for drug in drugs:
        # Canonicalize each source label before writing downstream keys.
        canonical = canonical_drug_name(drug)
        rows.append({
            "source_treatment": drug,
            "source_treatment_key": normalize_key(drug),
            "canonical_drug": canonical,
            "canonical_drug_key": normalize_key(canonical),
            "canonical_regimen_key": canonical_regimen_key(drug),
            "is_alias_changed": normalize_key(drug) != normalize_key(canonical),
        })

    ontology = pd.DataFrame(rows)
    # This ontology is the auditable treatment-key contract for later steps.
    ontology.to_csv(ontology_path, sep="\t", index=False)



# =========================
# Alias table output
# =========================
# The explicit alias table documents all built-in treatment name rewrites.

    alias_rows = [
        {
            "alias": alias,
            "canonical_drug": canonical,
            "alias_key": normalize_key(alias),
            "canonical_drug_key": normalize_key(canonical),
        }
        for alias, canonical in sorted(DEFAULT_ALIASES.items())
    ]
    aliases = pd.DataFrame(alias_rows)
    # Alias output makes every built-in treatment rewrite visible for review.
    aliases.to_csv(alias_path, sep="\t", index=False)

    print("DONE")
    print("Training table:", training_path)
    print("Unique source treatments:", len(ontology))
    print("Alias changes:", int(ontology["is_alias_changed"].sum()))
    print("Wrote:", ontology_path)
    print("Wrote:", alias_path)


if __name__ == "__main__":
    main()
