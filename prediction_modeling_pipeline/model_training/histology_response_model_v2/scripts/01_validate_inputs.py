"""
Script: 01_validate_inputs.py

Purpose:
    Validate configured clinical, slide, and tile paths before histology modeling.

Pipeline role:
    Step 01 of histology_response_model_v2. This is a preflight audit step that
    confirms expected input folders and clinical tables are present and readable
    before case labeling, slide matching, tiling, and model training.

Scientific context:
    This step helps separate file-contract failures from modeling failures. It
    records missing inputs and readable clinical file dimensions before expensive
    image or neural-network steps are attempted.

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
from histology_model_v2_lib import load_yaml, output_root, ensure_dir, resolve_path, read_table



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = ensure_dir(output_root(cfg) / "01_input_audit")

    # Rows become the machine-readable path audit table written for review.
    rows = []
    # Issues are accumulated and reported together so users can fix all missing inputs at once.
    issues = []

    cptac = resolve_path(cfg, cfg["paths"]["cptac_data_dir"])
    tcia = resolve_path(cfg, cfg["paths"]["tcia_download_dir"])
    tile_dir = resolve_path(cfg, cfg["paths"]["tile_dir"])

    for label, p in [("cptac_data_dir", cptac), ("tcia_download_dir", tcia), ("tile_dir", tile_dir)]:
        rows.append({"item": label, "path": str(p), "exists": p.exists()})
        if label != "tile_dir" and not p.exists():
            issues.append(f"missing {label}: {p}")

    if cptac.exists():
        for key, rel in cfg.get("clinical_files", {}).items():
            p = cptac / rel
            row = {"item": f"clinical_file_{key}", "path": str(p), "exists": p.exists()}
            if p.exists():
                try:
                    df = read_table(p, sep="\t")
                    row["rows"] = len(df)
                    row["cols"] = len(df.columns)
                except Exception as e:
                    row["read_error"] = str(e)
                    issues.append(f"could not read {p}: {e}")
            else:
                issues.append(f"missing clinical file: {p}")
            rows.append(row)

    # Slide counting is a lightweight preflight check; it does not open or tile SVS files.
    svs_count = 0
    if tcia and tcia.exists():
        exts = cfg.get("slide_manifest", {}).get("svs_extensions", [".svs"])
        svs_count = sum(1 for p in tcia.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    rows.append({"item": "slide_file_count", "path": str(tcia), "exists": bool(svs_count), "rows": svs_count})

    report = pd.DataFrame(rows)
    report.to_csv(out / "input_path_report.tsv", sep="\t", index=False)

    lines = ["Histology v2 input audit", "", f"slide_file_count: {svs_count}", ""]
    if issues:
        lines.append("ISSUES:")
        lines.extend([f"  {x}" for x in issues])
    else:
        lines.append("No critical issues detected")
    (out / "input_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("DONE")
    print(out)

if __name__ == "__main__":
    main()
