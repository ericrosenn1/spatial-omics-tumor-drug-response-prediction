"""
Audit canonical spatial feature pipeline outputs and external Step 02 processed samples.

Small canonical reports are stored under spatial_feature_identification_pipeline/outputs.
Large Step 02 processed h5ad intermediates are stored outside the source style
pipeline folder under Visium_samples/processed_samples.
"""

from pathlib import Path
import argparse
import json
import os
from datetime import datetime
import pandas as pd

CHECKS = [
    ("01", "output_01_validate_inputs", "input_validation_report.csv"),
    ("02_reports", "output_02_process_samples_reports", "processing_report.csv"),
    ("03", "output_03_merge_slide_features", "merged_slide_features.csv"),
    ("04", "output_04_score_and_label_slides", "slide_features_scored_labeled.csv"),
    ("05_status", "output_05_build_multi_axis_transcriptome_labels", "multi_axis_label_status.csv"),
    ("05_slide", "output_05_build_multi_axis_transcriptome_labels", "slide_features_with_multi_axis_labels.csv"),
    ("06", "output_06_build_accessibility_profiles", "slide_features_with_accessibility.csv"),
    ("07", "output_07_append_hotspot_metrics", "slide_features_with_hotspot_metrics.csv"),
    ("08", "output_08_context_alignment_and_metabolic_concordance", "slide_features_with_metabolic_concordance.csv"),
    ("09", "output_09_build_motif_tables", "slide_features_with_motif_tables.csv"),
    ("10_model", "output_10_build_model_ready_table", "model_input_numeric.csv"),
    ("10_manifest", "output_10_build_model_ready_table", "feature_manifest.csv"),
    ("11", "output_11_overlay", "overlay_status.csv"),
]

def summarize_csv(path: Path) -> dict:
    """Read one CSV and summarize existence, shape, sample count, and status count."""
    if not path.exists():
        return {
            "exists": False,
            "rows": "",
            "columns": "",
            "sample_count": "",
            "status_counts": "",
            "notes": "missing",
        }

    try:
        df = pd.read_csv(path)
    except Exception as error:
        return {
            "exists": True,
            "rows": "",
            "columns": "",
            "sample_count": "",
            "status_counts": "",
            "notes": f"{type(error).__name__}: {error}",
        }

    sample_count = ""
    if "sample_id" in df.columns:
        sample_count = int(df["sample_id"].astype(str).nunique())

    status_counts = ""
    if "status" in df.columns:
        status_counts = "; ".join(
            f"{k}:{v}"
            for k, v in df["status"].astype(str).value_counts(dropna=False).items()
        )

    return {
        "exists": True,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sample_count": sample_count,
        "status_counts": status_counts,
        "notes": "",
    }

def count_files(path: Path) -> int:
    """Count files recursively under a folder."""
    if not path.exists():
        return 0
    return len([p for p in path.rglob("*") if p.is_file()])

def external_step02_summary(path: Path) -> dict:
    """Summarize the external Step 02 processed sample directory."""
    sample_dirs = sorted([p for p in path.glob("SAMPLE_*") if p.is_dir()]) if path.exists() else []

    loaded = list(path.rglob("01_loaded.h5ad")) if path.exists() else []
    processed = list(path.rglob("02_processed.h5ad")) if path.exists() else []
    slide_rows = list(path.rglob("slide_level_feature_row.csv")) if path.exists() else []
    cluster_rows = list(path.rglob("cluster_summary.csv")) if path.exists() else []

    missing_processed = []
    for sample in sample_dirs:
        expected = sample / "adata" / "02_processed.h5ad"
        if not expected.exists():
            missing_processed.append(sample.name)

    return {
        "exists": path.exists(),
        "sample_folders": len(sample_dirs),
        "loaded_h5ad": len(loaded),
        "processed_h5ad": len(processed),
        "slide_level_feature_rows": len(slide_rows),
        "cluster_summary_rows": len(cluster_rows),
        "missing_processed_samples": ";".join(missing_processed),
    }

def main() -> None:
    """Run output audit and write reports under docs/provenance/output_audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", default=".")
    parser.add_argument(
        "--processed-samples-root",
        default=None,
    )
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    root = Path(args.pipeline_root).resolve()
    outputs = root / "outputs"
    report_dir = root / "docs" / "provenance" / "output_audits"
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for step, folder, filename in CHECKS:
        path = outputs / folder / filename
        rows.append({
            "step": step,
            "folder": folder,
            "file": filename,
            "path": str(path),
            **summarize_csv(path),
        })

    external_step02 = Path(args.processed_samples_root)
    s2 = external_step02_summary(external_step02)

    rows.append({
        "step": "02_data_external",
        "folder": str(external_step02),
        "file": "SAMPLE folders plus h5ad and table counts",
        "path": str(external_step02),
        "exists": s2["exists"],
        "rows": s2["sample_folders"],
        "columns": "",
        "sample_count": s2["sample_folders"],
        "status_counts": f"loaded_h5ad:{s2['loaded_h5ad']}; processed_h5ad:{s2['processed_h5ad']}; slide_rows:{s2['slide_level_feature_rows']}; cluster_rows:{s2['cluster_summary_rows']}",
        "notes": f"external Step 02 processed data; missing processed samples: {s2['missing_processed_samples'] or 'none'}",
    })

    step12 = outputs / "output_12_data_analysis_and_visuals"
    rows.append({
        "step": "12",
        "folder": "output_12_data_analysis_and_visuals",
        "file": "recursive file count",
        "path": str(step12),
        "exists": step12.exists(),
        "rows": count_files(step12),
        "columns": "",
        "sample_count": "",
        "status_counts": "",
        "notes": "visual and report output file count",
    })

    step13_generated = outputs / "output_13_external_study_validation"
    rows.append({
        "step": "13_generated",
        "folder": "output_13_external_study_validation",
        "file": "recursive file count",
        "path": str(step13_generated),
        "exists": step13_generated.exists(),
        "rows": count_files(step13_generated),
        "columns": "",
        "sample_count": "",
        "status_counts": "",
        "notes": "generated external validation outputs",
    })

    step13_assets = outputs / "_external_study_validation"
    rows.append({
        "step": "13_source_assets",
        "folder": "_external_study_validation",
        "file": "recursive file count",
        "path": str(step13_assets),
        "exists": step13_assets.exists(),
        "rows": count_files(step13_assets),
        "columns": "",
        "sample_count": "",
        "status_counts": "",
        "notes": "reference/source assets and study mapping used by Step 13",
    })

    audit = pd.DataFrame(rows)

    csv_path = report_dir / f"pipeline_audit_01_13_{stamp}.csv"
    txt_path = report_dir / f"pipeline_audit_01_13_{stamp}.txt"

    audit.to_csv(csv_path, index=False)

    lines = [str(txt_path), "", "PIPELINE OUTPUT AUDIT", ""]
    lines.append(
        audit[["step", "exists", "rows", "sample_count", "status_counts", "notes"]].to_string(index=False)
    )
    lines.append("")
    lines.append(f"Saved csv: {csv_path}")

    meta_path = outputs / "output_05_build_multi_axis_transcriptome_labels" / "multi_axis_label_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        settings = meta.get("settings", {})
        lines.append("")
        lines.append("REACTOME AUDIT")
        lines.append(f"reactome_max_terms: {settings.get('reactome_max_terms')}")
        lines.append(f"n_reactome_terms_loaded: {meta.get('n_reactome_terms_loaded')}")
        lines.append(f"n_hallmark_terms_loaded: {meta.get('n_hallmark_terms_loaded')}")
        lines.append(f"external_library_status: {meta.get('external_library_status')}")
        lines.append(f"external_library_source: {meta.get('external_library_source')}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("============================================================")
    print("Pipeline output audit")
    print("============================================================")
    print(audit[["step", "exists", "rows", "sample_count", "status_counts", "notes"]].to_string(index=False))
    print("")
    print("Report:", txt_path)
    print("CSV:", csv_path)
    print("DONE")

    if args.open and os.name == "nt":
        os.startfile(str(report_dir))

if __name__ == "__main__":
    main()




