"""
Script: 03_build_slide_manifest.py

Purpose:
    Link treatment-response labeled patients to locally available SVS slides.

Pipeline role:
    Step 03 of histology_response_model_v2. This step builds or loads a local
    whole-slide image inventory, reconciles it to case-level treatment-response
    labels, and writes the slide manifest consumed by tiling.

Scientific context:
    This script is the bridge between clinical treatment-response labels and
    image data. Correct patient and slide identifiers are essential because
    downstream tiling, patient-level splitting, leakage checks, and model audit
    all depend on this manifest.

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
import yaml



# =============================================================================
# Local helpers
# =============================================================================

def clean_text(x):
    """Return stripped text while converting missing values to an empty string."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def read_table(path, sep=None):
    """Read a delimited table with an extension-aware delimiter default."""
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in [".tsv", ".txt"] else ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False).fillna("")


def load_config(path):
    """Load the YAML configuration for this standalone script."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_key_recursive(obj, wanted):
    """Search nested config structures for a requested key."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == wanted:
                return v
            found = find_key_recursive(v, wanted)
            if found not in [None, ""]:
                return found

    if isinstance(obj, list):
        for x in obj:
            found = find_key_recursive(x, wanted)
            if found not in [None, ""]:
                return found

    return None


def resolve_path(project_dir, value):
    """Resolve a configured relative or absolute path against the project directory."""
    if value in [None, ""]:
        return None

    p = Path(str(value))
    if p.is_absolute():
        return p

    return Path(project_dir) / p


def make_slide_index(slide_root):
    """Build a local SVS slide inventory keyed by patient and slide identifiers."""
    rows = []

    slide_root = Path(slide_root)

    for svs in sorted(slide_root.rglob("*.svs")):
        patient_id = svs.parent.name.strip()
        rows.append({
            "patient_id": patient_id,
            "slide_id": svs.stem,
            "svs_path": str(svs),
            "slide_bytes": svs.stat().st_size,
            "has_slide": True,
        })

    return pd.DataFrame(rows)


def first_existing(*paths):
    """Return the first candidate path that exists on disk."""
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_dir = Path(cfg.get("project_dir", r"D:\Adv_Omics_Fenyo\project"))

    output_root = Path(cfg["output_root"])
    out_dir = output_root / "03_slide_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    cptac_dir_raw = (
        find_key_recursive(cfg, "cptac_dir")
        or find_key_recursive(cfg, "cptac_data_dir")
        or find_key_recursive(cfg, "clinical_dir")
        or r"D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\data_manifests\cptac_histology_training_data"
    )
    cptac_dir = resolve_path(project_dir, cptac_dir_raw)

    slide_root_raw = (
        find_key_recursive(cfg, "tcia_download_dir")
        or find_key_recursive(cfg, "slide_root")
        or find_key_recursive(cfg, "slides_dir")
        or r"C:\WSI_data\tcia_downloads"
    )
    slide_root = resolve_path(project_dir, slide_root_raw)

    # Step 03 uses the case label table as the clinical authority for patient/treatment/response metadata.
    case_labels_path = output_root / "02_case_labels" / "case_label_table.tsv"

    strict_table_path = first_existing(
        cptac_dir / "combined_case_table_strict.tsv",
        cptac_dir / "combined_case_table.tsv",
    )

    strict_ids_path = first_existing(
        cptac_dir / "strict_submitter_ids.txt",
        cptac_dir / "minimal_case_outputs" / "strict_submitter_ids.txt",
        cptac_dir / "minimal_case_outputs" / "submitter_ids_minimal_spec.txt",
    )

    if not case_labels_path.exists():
        raise FileNotFoundError(f"Missing case label table: {case_labels_path}")

    if not slide_root.exists():
        raise FileNotFoundError(f"Missing slide root: {slide_root}")

    labels = read_table(case_labels_path, sep="\t")

    if "patient_id" not in labels.columns:
        if "cases.submitter_id" in labels.columns:
            labels["patient_id"] = labels["cases.submitter_id"].map(clean_text)
        else:
            raise ValueError("case_label_table.tsv has no patient_id or cases.submitter_id")

    labels["patient_id"] = labels["patient_id"].map(clean_text)

    # The slide index is built from local SVS files so downstream steps only reference files present on disk.
    slide_index = make_slide_index(slide_root)

    if slide_index.empty:
        raise RuntimeError(f"No .svs files found under {slide_root}")

    local_patients = set(slide_index["patient_id"].astype(str))

    use_strict_ids = set()

    if strict_ids_path is not None and strict_ids_path.exists():
        use_strict_ids = {
            x.strip()
            for x in strict_ids_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if x.strip()
        }

    if use_strict_ids:
        selected_labels = labels[labels["patient_id"].isin(use_strict_ids)].copy()

        missing_label_ids = sorted(use_strict_ids - set(selected_labels["patient_id"]))

        if missing_label_ids and strict_table_path is not None and strict_table_path.exists():
            strict = read_table(strict_table_path, sep="\t")

            if "cases.submitter_id" in strict.columns:
                strict["patient_id"] = strict["cases.submitter_id"].map(clean_text)
            elif "patient_id" in strict.columns:
                strict["patient_id"] = strict["patient_id"].map(clean_text)
            else:
                strict["patient_id"] = ""

            add = strict[strict["patient_id"].isin(missing_label_ids)].copy()

            rename_map = {
                "cases.case_id": "case_id",
                "project.project_id": "project_id",
                "cases.primary_site": "primary_site",
                "cases.disease_type": "disease_type",
                "diagnoses.primary_diagnosis": "diagnosis",
                "treatments.therapeutic_agents": "treatments_therapeutic_agents",
                "treatments.regimen_or_line_of_therapy": "treatments_regimen_or_line_of_therapy",
            }

            add = add.rename(columns={k: v for k, v in rename_map.items() if k in add.columns})

            if "case_id" not in add.columns:
                add["case_id"] = ""
            if "project_id" not in add.columns:
                add["project_id"] = ""
            if "primary_site" not in add.columns:
                add["primary_site"] = ""
            if "disease_type" not in add.columns:
                add["disease_type"] = ""
            if "diagnosis" not in add.columns:
                add["diagnosis"] = ""
            if "treatments_therapeutic_agents" not in add.columns:
                add["treatments_therapeutic_agents"] = ""
            if "treatments_regimen_or_line_of_therapy" not in add.columns:
                add["treatments_regimen_or_line_of_therapy"] = ""

            response_col = None
            for c in [
                "treatments.treatment_outcome",
                "follow_ups.disease_response",
                "diagnoses.best_overall_response",
                "raw_response",
            ]:
                if c in add.columns:
                    response_col = c
                    break

            if response_col is None:
                add["raw_response"] = ""
            else:
                add["raw_response"] = add[response_col]

            def map_response(x):
                """Map raw response text to a binary response label when using fallback strict tables."""
                s = clean_text(x).lower()
                if s in ["complete response", "partial response", "cr", "pr", "no evidence of disease", "ned"]:
                    return "RESPONDER"
                if s in ["stable disease", "progressive disease", "persistent disease", "sd", "pd", "with tumor"]:
                    return "NON_RESPONDER"
                return ""

            add["response_source_column"] = response_col or ""
            add["binary_response_label"] = add["raw_response"].apply(map_response)
            add["binary_response_id"] = add["binary_response_label"].map({"NON_RESPONDER": 0, "RESPONDER": 1}).fillna("")
            add["canonical_treatment_key"] = add["treatments_therapeutic_agents"].replace("", "nos").str.lower()
            add["component_drug_keys"] = add["canonical_treatment_key"]
            add["n_components"] = 1
            add["has_specific_treatment"] = True
            add["usable_strict"] = add["binary_response_label"].isin(["RESPONDER", "NON_RESPONDER"])

            keep_cols = [c for c in selected_labels.columns if c in add.columns]
            add = add[keep_cols].copy()

            selected_labels = pd.concat([selected_labels, add], ignore_index=True, sort=False)

        # Patient-level deduplication helps prevent repeated clinical rows from creating ambiguous slide labels.
        selected_labels = selected_labels.drop_duplicates(subset=["patient_id"], keep="first").copy()

    else:
        selected_labels = labels.copy()

    selected_labels["patient_id"] = selected_labels["patient_id"].map(clean_text)

    manifest = selected_labels.merge(
        slide_index,
        on="patient_id",
        how="left",
    )

    manifest["has_slide"] = manifest["svs_path"].astype(str).str.len() > 0

    manifest_path = out_dir / "slide_manifest.tsv"
    # The slide manifest is the file contract between clinical labeling and image tiling.
    manifest.to_csv(manifest_path, sep="\t", index=False)

    matched = manifest[manifest["has_slide"]].copy()
    matched_path = out_dir / "slide_manifest_matched.tsv"
    matched.to_csv(matched_path, sep="\t", index=False)

    strict_local_ids = sorted((use_strict_ids if use_strict_ids else set(selected_labels["patient_id"])) & local_patients)
    manifest_patients = set(manifest.loc[manifest["has_slide"], "patient_id"].astype(str))
    missed_local = sorted(set(strict_local_ids) - manifest_patients)

    pd.DataFrame({"patient_id": missed_local}).to_csv(
        out_dir / "local_svs_patients_missing_from_manifest.tsv",
        sep="\t",
        index=False,
    )

    lines = []
    lines.append("Slide manifest summary")
    lines.append(f"case_label_rows: {len(labels)}")
    lines.append(f"selected_label_patients: {selected_labels['patient_id'].nunique()}")
    lines.append(f"slide_files_found: {len(slide_index)}")
    lines.append(f"local_slide_patients: {len(local_patients)}")
    lines.append(f"manifest_rows: {len(manifest)}")
    lines.append(f"matched_slide_rows: {len(matched)}")
    lines.append(f"patients_with_slides: {matched['patient_id'].nunique() if len(matched) else 0}")
    lines.append(f"patients_without_slides: {selected_labels['patient_id'].nunique() - (matched['patient_id'].nunique() if len(matched) else 0)}")
    lines.append(f"strict_ids_loaded: {len(use_strict_ids)}")
    lines.append(f"strict_local_slide_patients: {len(strict_local_ids)}")
    lines.append(f"strict_local_patients_missed_by_manifest: {len(missed_local)}")

    if missed_local:
        lines.append("")
        lines.append("Missed local patients:")
        lines.extend(missed_local)

    (out_dir / "slide_manifest_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("DONE")
    print(out_dir)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
