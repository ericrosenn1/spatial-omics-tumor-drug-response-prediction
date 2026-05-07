"""
Script: 00_build_treatment_ontology.py

Purpose:
    Build a canonical treatment ontology from clinical treatment fields.

Pipeline role:
    Step 00 of histology_response_model_v2. This step scans raw treatment
    strings, removes non-specific therapy labels, harmonizes named agents and
    regimens, and writes the treatment ontology used by downstream case labeling.

Scientific context:
    Treatment harmonization is required before histology response modeling because
    raw clinical exports can contain aliases, salt forms, combinations, and
    generic therapy labels. The downstream teacher model should learn named
    treatment-response structure rather than inconsistent source text.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse
import pandas as pd
from histology_model_v2_lib import load_yaml, output_root, ensure_dir, resolve_path, read_table, clean_text, canonical_component, canonical_regimen, treatment_components



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = ensure_dir(output_root(cfg) / "00_treatment_ontology")

    cptac = resolve_path(cfg, cfg["paths"]["cptac_data_dir"])
    clinical_path = cptac / cfg["clinical_files"]["clinical"]
    clinical = read_table(clinical_path, sep="\t")

    candidates = []
    # Candidate treatment columns are intentionally explicit so the ontology reflects auditable clinical fields.
    cols = [
        "treatments.therapeutic_agents",
        "treatments.regimen_or_line_of_therapy",
        "treatments.drug_category",
        "treatments.treatment_type",
        "treatments.treatment_or_therapy",
    ]
    for col in cols:
        if col not in clinical.columns:
            continue
        for value in clinical[col].dropna().astype(str).unique():
            # Generic therapy labels are filtered inside treatment_components; only named components are retained.
            comps = treatment_components(value)
            if not comps:
                continue
            candidates.append({
                "source_column": col,
                "raw_value": clean_text(value),
                "canonical_treatment_key": canonical_regimen(value),
                "n_components": len(comps),
                "component_drug_keys": " | ".join(comps),
            })

    ont = pd.DataFrame(candidates).drop_duplicates().sort_values(["canonical_treatment_key", "raw_value"])
    # The ontology is reused downstream to keep treatment keys stable across case labels and model training.
    ont.to_csv(out / "treatment_ontology.tsv", sep="\t", index=False)

    summary = [
        "Treatment ontology build summary",
        f"clinical_path: {clinical_path}",
        f"clinical_rows: {len(clinical)}",
        f"ontology_rows: {len(ont)}",
        f"canonical_treatments: {ont['canonical_treatment_key'].nunique() if not ont.empty else 0}",
    ]
    (out / "treatment_ontology_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("DONE")
    print(out)

if __name__ == "__main__":
    main()
