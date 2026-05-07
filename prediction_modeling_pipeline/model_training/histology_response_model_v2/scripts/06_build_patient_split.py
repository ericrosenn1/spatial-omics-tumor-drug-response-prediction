"""
Script: 06_build_patient_split.py

Purpose:
    Create leakage-safe patient-level train/validation/test splits.

Pipeline role:
    Step 06 of histology_response_model_v2. This step splits patients, not
    tiles, so slides and tiles from the same patient cannot appear in multiple
    model evaluation splits. The resulting patient assignments are then merged
    back to tile rows for Step 07 model training.

Scientific context:
    Patient-level splitting is a core validity requirement for weakly supervised
    histology response modeling. Multiple slides and many tiles can come from the
    same patient, so splitting at tile or slide level would leak patient identity
    into validation and test sets.

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
from sklearn.model_selection import train_test_split

from histology_model_v2_lib import load_yaml, output_root, ensure_dir, read_table



# =============================================================================
# Split helper functions
# =============================================================================

def safe_split(sub, train_frac, val_frac, test_frac, seed):
    """Split one treatment-response stratum while handling very small groups safely."""
    n = len(sub)
    sub = sub.sample(frac=1, random_state=seed).copy()

    if n == 1:
        sub["split"] = "train"
        return sub

    if n == 2:
        sub["split"] = ["train", "test"]
        return sub

    if n == 3:
        sub["split"] = ["train", "val", "test"]
        return sub

    train, temp = train_test_split(
        sub,
        test_size=1 - train_frac,
        random_state=seed,
        shuffle=True,
    )

    if len(temp) < 2:
        train["split"] = "train"
        temp["split"] = "test"
        return pd.concat([train, temp], ignore_index=True)

    rel_test = test_frac / (val_frac + test_frac)

    val, test = train_test_split(
        temp,
        test_size=rel_test,
        random_state=seed,
        shuffle=True,
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)



# =============================================================================
# Summary formatting helpers
# =============================================================================

def table_to_string(df):
    """Render a DataFrame as plain text for inclusion in QC summaries."""
    if df is None or len(df) == 0:
        return ""
    return df.to_string(index=False)



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = ensure_dir(output_root(cfg) / "06_patient_split")
    split_cfg = cfg["split"]

    tile = read_table(output_root(cfg) / "05_training_table" / "tile_training_table.tsv", sep="\t")

    requested_patient_cols = [
        "patient_id",
        "case_id",
        "project_id",
        "primary_site",
        "disease_type",
        "diagnosis",
        "data_source",
        "binary_response_label",
        "binary_response_id",
        "canonical_treatment_key",
        "component_drug_keys",
        "n_components",
        "has_specific_treatment",
        "treatments_therapeutic_agents",
        "treatments_regimen_or_line_of_therapy",
    ]

    patient_cols = [c for c in requested_patient_cols if c in tile.columns]

    required = [
        "patient_id",
        "binary_response_label",
        "canonical_treatment_key",
    ]

    missing_required = [c for c in required if c not in patient_cols]

    if missing_required:
        raise ValueError(f"Missing required columns for patient split: {missing_required}")

    # Splits are assigned at patient level before merging back to tile rows.
    patient = tile[patient_cols].drop_duplicates("patient_id").copy()

    # The stratification key preserves treatment-response balance when group size allows.
    patient["stratify_key"] = (
        patient["canonical_treatment_key"].astype(str)
        + "__"
        + patient["binary_response_label"].astype(str)
    )

    counts = patient["stratify_key"].value_counts()
    patient["stratify_group_size"] = patient["stratify_key"].map(counts)

    pieces = []

    for key, sub in patient.groupby("stratify_key", dropna=False):
        # Very small treatment-response strata are kept in train to avoid invalid split fragments.
        if len(sub) < int(split_cfg.get("min_group_size_for_stratify", 3)):
            sub = sub.copy()
            sub["split"] = "train"
            pieces.append(sub)
        else:
            pieces.append(
                safe_split(
                    sub,
                    split_cfg["train_frac"],
                    split_cfg["val_frac"],
                    split_cfg["test_frac"],
                    split_cfg["random_seed"],
                )
            )

    patient_split = pd.concat(pieces, ignore_index=True)

    if patient_split["patient_id"].duplicated().any():
        dup = patient_split.loc[
            patient_split["patient_id"].duplicated(keep=False),
            ["patient_id", "split", "binary_response_label", "canonical_treatment_key"],
        ]
        raise ValueError("Duplicate patient_id after split\n" + dup.to_string(index=False))

    merge_cols = [
        "patient_id",
        "split",
        "stratify_key",
        "stratify_group_size",
    ]

    # Tile rows inherit the patient split after assignment, preventing patient leakage across splits.
    tile_split = tile.merge(
        patient_split[merge_cols],
        on="patient_id",
        how="left",
    )

    if tile_split["split"].isna().any():
        missing = tile_split.loc[tile_split["split"].isna(), "patient_id"].drop_duplicates().head(20)
        raise ValueError("Some tiles missing split. Example patients:\n" + missing.to_string(index=False))

    patient_split.to_csv(out / "patient_split.tsv", sep="\t", index=False)
    tile_split.to_csv(out / "tile_training_table_split.tsv", sep="\t", index=False)

    leakage = (
        tile_split
        .groupby("patient_id")["split"]
        .nunique()
    )

    # A patient appearing in multiple splits would indicate leakage and must be reported.
    leaky_patients = leakage[leakage > 1]

    duplicate_patient_rows = patient_split[
        patient_split["patient_id"].duplicated(keep=False)
    ]

    patient_split_counts = patient_split["split"].value_counts().rename_axis("split").reset_index(name="n")

    tile_split_counts = tile_split["split"].value_counts().rename_axis("split").reset_index(name="n")

    response_counts = (
        patient_split
        .groupby(["split", "binary_response_label"])
        .size()
        .reset_index(name="n")
        .sort_values(["split", "binary_response_label"])
    )

    treatment_counts = (
        patient_split
        .groupby(["split", "canonical_treatment_key"])
        .size()
        .reset_index(name="n")
        .sort_values(["split", "n"], ascending=[True, False])
        .head(100)
    )

    treatment_response_counts = (
        patient_split
        .groupby(["split", "canonical_treatment_key", "binary_response_label"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(80)
    )

    if "project_id" in patient_split.columns:
        project_counts = (
            patient_split
            .groupby(["split", "project_id"])
            .size()
            .reset_index(name="n")
            .sort_values(["split", "n"], ascending=[True, False])
            .head(80)
        )
    else:
        project_counts = pd.DataFrame()

    min_response_counts = (
        patient_split
        .groupby(["split", "binary_response_label"])
        .size()
        .reset_index(name="n")
        .pivot(index="split", columns="binary_response_label", values="n")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    ok = True
    decision_lines = []

    if len(leaky_patients) > 0:
        ok = False
        decision_lines.append(f"FAIL: patients in multiple splits: {len(leaky_patients)}")

    if len(duplicate_patient_rows) > 0:
        ok = False
        decision_lines.append(f"FAIL: duplicate patient rows: {len(duplicate_patient_rows)}")

    for split_name in ["train", "val", "test"]:
        sub = patient_split[patient_split["split"] == split_name]
        labels = set(sub["binary_response_label"].astype(str))
        if "RESPONDER" not in labels or "NON_RESPONDER" not in labels:
            ok = False
            decision_lines.append(f"FAIL: split lacks both response classes: {split_name}")

    if ok:
        # The summary records whether the split is suitable for downstream Step 07 training.
        decision_lines.append("PASS: split is suitable for step 07 training")

    lines = []

    lines.append("Patient split summary")
    lines.append("=" * 60)

    lines.append("")
    lines.append("Patient split counts")
    lines.append(table_to_string(patient_split_counts))

    lines.append("")
    lines.append("Tile split counts")
    lines.append(table_to_string(tile_split_counts))

    lines.append("")
    lines.append("Response counts by patient split")
    lines.append(table_to_string(response_counts))

    lines.append("")
    lines.append("Minimum response counts per split")
    lines.append(table_to_string(min_response_counts))

    lines.append("")
    lines.append("Patient leakage check")
    lines.append(f"patients_in_multiple_splits: {len(leaky_patients)}")

    lines.append("")
    lines.append("Duplicate patient rows")
    lines.append(f"duplicate_patient_rows: {len(duplicate_patient_rows)}")

    lines.append("")
    lines.append("Project counts by split")
    if len(project_counts):
        lines.append(table_to_string(project_counts))
    else:
        lines.append("project_id column missing")

    lines.append("")
    lines.append("Treatment counts by split")
    lines.append(table_to_string(treatment_counts))

    lines.append("")
    lines.append("Treatment response groups by split")
    lines.append(table_to_string(treatment_response_counts))

    lines.append("")
    lines.append("QC decision")
    lines.extend(decision_lines)

    summary = "\n".join(lines)

    (out / "patient_split_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)

    if not ok:
        raise RuntimeError("Patient split QC failed")


if __name__ == "__main__":
    main()
