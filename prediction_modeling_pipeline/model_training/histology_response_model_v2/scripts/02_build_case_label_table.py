"""
Script: 02_build_case_label_table.py

Purpose:
    Build case-level treatment-response labels from clinical tables.

Pipeline role:
    Step 02 of histology_response_model_v2. This step collapses clinical rows to
    case-level records, resolves binary response labels from configured priority
    columns, harmonizes named treatments, and marks strict usable cases.

Scientific context:
    The histology teacher depends on reliable patient/case labels. This step
    records response provenance, treatment specificity, and strict usability so
    downstream slide linkage and patient-level splitting are auditable.

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
from histology_model_v2_lib import (
    load_yaml, output_root, ensure_dir, resolve_path, read_table, clean_text,
    canonical_regimen, treatment_components, is_specific_treatment, response_label
)



# =============================================================================
# Helper functions
# =============================================================================

def first_nonempty(row, cols):
    """Return the first non-empty value from a prioritized list of row columns."""
    for c in cols:
        if c in row.index:
            v = clean_text(row[c])
            if v:
                return v, c
    return "", ""


def collapse_to_case(df, case_col):
    """Collapse repeated clinical rows into one case-level row."""
    def join_unique(s):
        """Join unique non-empty clinical values while preserving first observed order."""
        vals = []
        for v in s:
            v = clean_text(v)
            if v and v not in vals:
                vals.append(v)
        return " | ".join(vals)
    agg = {c: join_unique for c in df.columns if c != case_col}
    return df.groupby(case_col, dropna=False).agg(agg).reset_index()



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = ensure_dir(output_root(cfg) / "02_case_labels")

    cptac = resolve_path(cfg, cfg["paths"]["cptac_data_dir"])
    clinical = read_table(cptac / cfg["clinical_files"]["clinical"], sep="\t")
    follow = read_table(cptac / cfg["clinical_files"]["follow_up"], sep="\t") if (cptac / cfg["clinical_files"]["follow_up"]).exists() else pd.DataFrame()
    pathology = read_table(cptac / cfg["clinical_files"]["pathology"], sep="\t") if (cptac / cfg["clinical_files"]["pathology"]).exists() else pd.DataFrame()

    case_col = "cases.case_id"
    submitter_col = "cases.submitter_id"

    frames = []
    for df in [clinical, follow, pathology]:
        if not df.empty and case_col in df.columns:
            frames.append(collapse_to_case(df, case_col))

    if not frames:
        raise ValueError("No case-level clinical tables could be built")

    cases = frames[0]
    for i, df in enumerate(frames[1:], start=1):
        shared = [c for c in df.columns if c in cases.columns and c != case_col]
        df = df.drop(columns=shared)
        cases = cases.merge(df, on=case_col, how="left")

    for col in cases.columns:
        cases[col] = cases[col].apply(clean_text)

    # Response columns are prioritized so the chosen label source remains reproducible.
    response_cols = cfg["labeling"]["response_priority"]
    responder = cfg["labeling"]["responder_labels"]
    non = cfg["labeling"]["non_responder_labels"]

    # Treatment columns are harmonized together to support both single-agent and regimen labels.
    treatment_cols = [
        "treatments.therapeutic_agents",
        "treatments.regimen_or_line_of_therapy",
        "treatments.drug_category",
        "treatments.treatment_type",
        "treatments.treatment_or_therapy",
    ]

    rows = []
    for _, row in cases.iterrows():
        resp_raw, resp_source = first_nonempty(row, response_cols)
        label = response_label(resp_raw, responder, non)
        comps = treatment_components(*[row.get(c, "") for c in treatment_cols])
        regimen = canonical_regimen(*[row.get(c, "") for c in treatment_cols])
        specific = bool(comps)

        rows.append({
            "case_id": row.get(case_col, ""),
            "patient_id": row.get(submitter_col, ""),
            "project_id": row.get("project.project_id", ""),
            "primary_site": row.get("cases.primary_site", ""),
            "disease_type": row.get("cases.disease_type", ""),
            "raw_response": resp_raw,
            "response_source_column": resp_source,
            "binary_response_label": label,
            "binary_response_id": 1 if label == "RESPONDER" else (0 if label == "NON_RESPONDER" else ""),
            "canonical_treatment_key": regimen,
            "component_drug_keys": " | ".join(comps),
            "n_components": len(comps),
            "has_specific_treatment": specific,
            "treatments_therapeutic_agents": row.get("treatments.therapeutic_agents", ""),
            "treatments_regimen_or_line_of_therapy": row.get("treatments.regimen_or_line_of_therapy", ""),
            "diagnosis": row.get("diagnoses.primary_diagnosis", ""),
        })

    lab = pd.DataFrame(rows)
    # Strict usability requires patient identity, binary response, and at least one named treatment.
    lab["usable_strict"] = (
        lab["patient_id"].astype(str).str.len().gt(0)
        & lab["binary_response_label"].astype(str).str.len().gt(0)
        & lab["has_specific_treatment"]
    )

    lab.to_csv(out / "case_label_table.tsv", sep="\t", index=False)
    lab[lab["usable_strict"]].to_csv(out / "case_label_table_strict.tsv", sep="\t", index=False)

    summary = [
        "Case label table summary",
        f"case_rows: {len(lab)}",
        f"usable_strict_cases: {int(lab['usable_strict'].sum())}",
        "response_counts:",
        lab.loc[lab['usable_strict'], 'binary_response_label'].value_counts(dropna=False).to_string(),
        "top_treatments:",
        lab.loc[lab['usable_strict'], 'canonical_treatment_key'].value_counts(dropna=False).head(30).to_string(),
    ]
    (out / "case_label_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("DONE")
    print(out)

if __name__ == "__main__":
    main()
