"""
Script: 01_validate_teacher_inputs.py

Purpose:
    Validate governed teacher_builder inputs and write the run governance registry.

Project context:
    This is Step 01 of the governed Visium teacher_builder workflow. It checks
    configured spatial feature, metadata, expression, histology, processed Visium,
    and model-index paths before teacher construction. It also builds the treatment
    prior table and teacher reliability registry consumed by later fusion steps.

Scientific role:
    teacher_builder should not silently fuse unavailable or ungoverned teacher
    signals. This validation step records which samples have processed expression
    inputs and high-resolution histology images, confirms upstream teacher model
    artifacts are available, estimates treatment priors from the expression-response
    training table, and documents expression/histology reliability metadata before
    sample-by-treatment teacher labels are built.

Documentation polish marker:
    TEACHER_BUILDER_STEP01_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic,
    paths, thresholds, schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 01 uses lightweight file handling plus pandas summaries to validate
# inputs before expression, histology, and fusion teacher tables are built.

from pathlib import Path
import argparse
import json
import pandas as pd



# =========================
# Shared governance helper imports
# =========================
# Shared helpers keep config loading, path resolution, treatment priors,
# model-index parsing, and scalar coercion consistent across steps.

from teacher_governance_lib import (
    load_config,
    cfg_path,
    resolve_path,
    ensure_dir,
    read_table,
    write_table,
    save_json,
    build_treatment_priors,
    parse_model_index,
    normalize_key,
    boolish,
    safe_float,
    clean_text,
)




# =========================
# Command-line interface
# =========================
# The governed runner passes the YAML config path into each numbered script.

def parse_args():
    """Parse the required governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Sample selection helper
# =========================
# The same sample list is reused across availability checks and teacher
# construction, with optional truncation for smoke-test runs.

def sample_ids_from_table(df, sample_col, cfg):
    """Return unique sample IDs, respecting configured test-mode truncation."""

    ids = df[sample_col].astype(str).drop_duplicates().tolist()
    # Smoke-test mode intentionally limits the sample list without changing full-run behavior.
    if bool(cfg.get("test_mode", False)):
        ids = ids[: int(cfg.get("test_n_samples", 5))]
    return ids




# =========================
# High-resolution image discovery helper
# =========================
# Histology availability can come from configured slide folders, metadata
# path columns, or raw Visium directory searches.

def discover_hires_for_sample(sample_id, cfg, row=None):
    """Find the best available high-resolution histology image path for one Visium sample."""

    project = Path(cfg["project_dir"])
    patterns = cfg.get("hires_image_patterns", ["tissue_hires_image.png", "*tissue_hires_image.png", "*hires_image.png"])

    candidates = []

    # First search explicit configured slide locations for each sample.
    for key in ["slides_dir", "visium_slides_dir"]:
        p = resolve_path(cfg, cfg.get(key))
        if p is not None:
            candidates += [
                p / sample_id / "spatial" / cfg.get("hires_image_name", "tissue_hires_image.png"),
                p / sample_id / "spatial" / "tissue_hires_image.png",
            ]

    if row is not None:
        for c in row.index:
            # Metadata path-like columns can directly point to the hires image.
            if any(tok in c.lower() for tok in ["image", "hires", "path"]):
                s = clean_text(row[c])
                if s:
                    p = Path(s)
                    if not p.is_absolute():
                        p = project / p
                    candidates.append(p)

    for p in candidates:
        if p.exists():
            return p

    raw = resolve_path(cfg, cfg.get("raw_visium_dir"))

    if raw is not None and raw.exists():
        tokens = [sample_id.lower()]

        if row is not None:
            for c in row.index:
                val = clean_text(row[c])
                if len(val) >= 5 and any(prefix in val.lower() for prefix in ["gsm", "sample", "gse"]):
                    tokens.append(val.lower())

        # Fall back to recursive raw Visium search when explicit paths are absent.
        for pat in patterns:
            for p in raw.rglob(pat):
                name = p.name.lower()
                if any(tok in name for tok in tokens):
                    return p

    return None




# =========================
# Teacher input validation workflow
# =========================
# Main workflow: resolve configured paths, validate tables, audit sample
# availability, build priors, build teacher registry, and write summaries.

def main():
    """Validate teacher inputs and write sample availability, priors, registry, and governance summaries."""

    args = parse_args()
    cfg = load_config(args.config)

    out_dir = ensure_dir(Path(cfg["output_root"]) / "01_input_validation")

    sample_col = cfg.get("sample_col", "sample_id")
    dataset_col = cfg.get("dataset_col", "dataset_id")
    cancer_col = cfg.get("cancer_col", "cancer_type")



    # =========================
    # Configured input path registry
    # =========================
    # This dictionary is the explicit preflight contract for upstream spatial,
    # expression, histology, and model-artifact inputs.

    # Keep all required upstream inputs in one auditable path registry.
    paths = {
        "spatial_feature_table": cfg_path(cfg, "spatial_feature_table"),
        "spatial_feature_manifest": cfg_path(cfg, "spatial_feature_manifest"),
        "metadata_table": cfg_path(cfg, "metadata_table"),
        "expression_training_table": cfg_path(cfg, "expression_training_table"),
        "expression_model_v2_deployable_root": cfg_path(cfg, "expression_model_v2_deployable_root"),
        "expression_model_index": cfg_path(cfg, "expression_model_index"),
        "expression_model_index_approved": cfg_path(cfg, "expression_model_index_approved"),
        "expression_audit_summary": cfg_path(cfg, "expression_audit_summary"),
        "histology_model_index": cfg_path(cfg, "histology_model_index"),
        "histology_model_path": cfg_path(cfg, "histology_model_path"),
        "histology_encoder_path": cfg_path(cfg, "histology_encoder_path"),
        "processed_samples_dir": cfg_path(cfg, "processed_samples_dir"),
        "raw_visium_dir": cfg_path(cfg, "raw_visium_dir"),
    }

    issues = []

    # Missing paths are collected as issues instead of failing at the first problem.
    for name, p in paths.items():
        if p is None:
            issues.append(f"missing config value: {name}")
        elif not p.exists():
            issues.append(f"path does not exist: {name} = {p}")



    # =========================
    # Spatial and metadata table loading
    # =========================
    # Spatial features define sample IDs for the governed teacher run; metadata
    # helps locate image paths and cohort descriptors.

    spatial = read_table(paths["spatial_feature_table"]) if paths["spatial_feature_table"] and paths["spatial_feature_table"].exists() else None
    metadata = read_table(paths["metadata_table"]) if paths["metadata_table"] and paths["metadata_table"].exists() else None

    if spatial is None:
        issues.append("spatial_feature_table could not be loaded")
    elif sample_col not in spatial.columns:
        issues.append(f"spatial_feature_table missing sample column: {sample_col}")

    if metadata is None:
        issues.append("metadata_table could not be loaded")
    elif sample_col not in metadata.columns:
        issues.append(f"metadata_table missing sample column: {sample_col}")

    if spatial is not None and sample_col in spatial.columns:
        sample_ids = sample_ids_from_table(spatial, sample_col, cfg)
    elif metadata is not None and sample_col in metadata.columns:
        sample_ids = sample_ids_from_table(metadata, sample_col, cfg)
    else:
        sample_ids = []



    # =========================
    # Per-sample availability audit
    # =========================
    # Sample-level availability flags make missing processed h5ad or histology
    # images visible before downstream scoring.

    availability_rows = []

    meta_by_sample = {}
    if metadata is not None and sample_col in metadata.columns:
        meta_by_sample = {str(r[sample_col]): r for _, r in metadata.iterrows()}

    for sid in sample_ids:
        row = meta_by_sample.get(str(sid))
        img = discover_hires_for_sample(str(sid), cfg, row=row)
        availability_rows.append(
            {
                "sample_id": sid,
                "has_processed_h5ad": (paths["processed_samples_dir"] / str(sid) / "adata" / "02_processed.h5ad").exists() if paths["processed_samples_dir"] else False,
                "has_hires_image": img is not None,
                "hires_image_path": str(img) if img is not None else "",
            }
        )

    # Availability output separates processed expression presence from histology image presence.
    sample_availability = pd.DataFrame(availability_rows)
    sample_availability.to_csv(out_dir / "sample_availability_report.csv", index=False)



    # =========================
    # Treatment prior construction
    # =========================
    # Treatment priors anchor governed teacher probabilities before expression
    # and histology deltas are fused.

    training = read_table(paths["expression_training_table"]) if paths["expression_training_table"] and paths["expression_training_table"].exists() else None

    if training is not None:
        # Priors are estimated once here and reused by governed fusion.
        priors = build_treatment_priors(
            training,
            min_exact_n=int(cfg.get("governance", {}).get("min_exact_prior_n", 5)),
        )
    else:
        priors = pd.DataFrame()

    write_table(priors, out_dir / "treatment_priors.tsv")



    # =========================
    # Teacher reliability registry
    # =========================
    # The registry records approved teacher models, reliability weights, and
    # known modality limitations before fusion.

    registry_rows = []

    expr_index_path = paths["expression_model_index_approved"]

    if expr_index_path and expr_index_path.exists():
        # Approved expression models are normalized into a common teacher registry schema.
        expr_index = parse_model_index(read_table(expr_index_path))
        for _, r in expr_index.iterrows():
            registry_rows.append(
                {
                    "teacher": "expression",
                    "model_family": "expression_response_model_v2",
                    "canonical_treatment_key": r["canonical_treatment_key"],
                    "approved_for_teacher": bool(r.get("approved_for_teacher", True)),
                    "reliability_weight": safe_float(r.get("reliability_weight"), 0.0),
                    "source_index": str(expr_index_path),
                    "limitation": "",
                }
            )

    hist_index_path = paths["histology_model_index"]

    if hist_index_path and hist_index_path.exists():
        hidx = read_table(hist_index_path)
        hrow = hidx.iloc[0].to_dict() if len(hidx) else {}

        approved = boolish(hrow.get("approved_for_teacher", True))
        reliability = safe_float(hrow.get("reliability_weight"), safe_float(hrow.get("auc_delta_vs_treatment_only"), 0.0957))

        blank_mean = safe_float(hrow.get("blank_mean"), safe_float(hrow.get("histology_blank_control_mean"), float("nan")))
        blank_std = safe_float(hrow.get("blank_std"), safe_float(hrow.get("histology_blank_control_std"), float("nan")))
        noise_mean = safe_float(hrow.get("noise_mean"), safe_float(hrow.get("histology_noise_control_mean"), float("nan")))
        noise_std = safe_float(hrow.get("noise_std"), safe_float(hrow.get("histology_noise_control_std"), float("nan")))

        # Histology control warnings are preserved so downstream fusion can shrink cautiously.
        control_warning = "blank_noise_controls_wide"
        registry_rows.append(
            {
                "teacher": "histology",
                "model_family": "histology_response_model_v2",
                "canonical_treatment_key": "__all_histology_model_treatments__",
                "approved_for_teacher": approved,
                "reliability_weight": reliability,
                "source_index": str(hist_index_path),
                "histology_blank_control_mean": blank_mean,
                "histology_blank_control_std": blank_std,
                "histology_noise_control_mean": noise_mean,
                "histology_noise_control_std": noise_std,
                "histology_control_warning": control_warning,
                "limitation": "histology blank and noise controls produced broad probability distributions; shrink toward treatment prior required",
            }
        )
    else:
        registry_rows.append(
            {
                "teacher": "histology",
                "model_family": "histology_response_model_v2",
                "canonical_treatment_key": "__all_histology_model_treatments__",
                "approved_for_teacher": False,
                "reliability_weight": 0.0,
                "source_index": "",
                "histology_control_warning": "histology_model_index_missing",
                "limitation": "histology model index missing",
            }
        )

    registry = pd.DataFrame(registry_rows)
    write_table(registry, out_dir / "teacher_reliability_registry.tsv")



    # =========================
    # Machine-readable governance configuration
    # =========================
    # This JSON captures resolved paths, clipping bounds, control factors, and
    # validation issues for reproducibility.

    governance = {
        "config": str(Path(args.config).resolve()),
        "output_root": str(Path(cfg["output_root"]).resolve()),
        "sample_col": sample_col,
        "test_mode": bool(cfg.get("test_mode", False)),
        "test_n_samples": int(cfg.get("test_n_samples", 0)),
        "probability_clip_low": float(cfg.get("governance", {}).get("probability_clip_low", 0.01)),
        "probability_clip_high": float(cfg.get("governance", {}).get("probability_clip_high", 0.99)),
        "histology_control_factor_if_warning": float(cfg.get("governance", {}).get("histology_control_factor_if_warning", 0.5)),
        "paths": {k: str(v) if v is not None else "" for k, v in paths.items()},
        "issues": issues,
    }

    # The governance JSON is the machine-readable record of this validation step.
    save_json(governance, out_dir / "teacher_governance_config.json")



    # =========================
    # Human-readable validation summary
    # =========================
    # The text summary is intended for quick review in the terminal, GitHub, and
    # publication audit materials.

    lines = []
    lines.append("Teacher input validation and governance summary")
    lines.append("")
    lines.append(f"config: {Path(args.config).resolve()}")
    lines.append(f"output_root: {Path(cfg['output_root']).resolve()}")
    lines.append(f"test_mode: {bool(cfg.get('test_mode', False))}")
    lines.append(f"test_n_samples: {cfg.get('test_n_samples')}")
    lines.append("")
    lines.append("Resolved paths:")
    for k, v in paths.items():
        lines.append(f"  {k}: {v} | exists={bool(v and v.exists())}")
    lines.append("")
    if spatial is not None:
        lines.append(f"spatial rows: {len(spatial)}")
        lines.append(f"spatial samples: {spatial[sample_col].nunique() if sample_col in spatial.columns else 'NA'}")
        lines.append(f"spatial columns: {spatial.shape[1]}")
    if metadata is not None:
        lines.append(f"metadata rows: {len(metadata)}")
        lines.append(f"metadata samples: {metadata[sample_col].nunique() if sample_col in metadata.columns else 'NA'}")
    lines.append(f"samples selected for run: {len(sample_ids)}")
    if not sample_availability.empty:
        lines.append(f"samples with processed h5ad: {int(sample_availability['has_processed_h5ad'].sum())}")
        lines.append(f"samples with hires image: {int(sample_availability['has_hires_image'].sum())}")
    lines.append("")
    lines.append(f"treatment priors written: {len(priors)}")
    lines.append(f"teacher registry rows written: {len(registry)}")
    lines.append("")
    if issues:
        lines.append("ISSUES:")
        for issue in issues:
            lines.append(f"  {issue}")
    else:
        lines.append("No critical issues detected")

    summary_text = "\n".join(lines)
    # The plain-text summary is the reviewer-facing validation artifact.
    (out_dir / "teacher_input_validation_summary.txt").write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print("DONE")
    print(out_dir)


if __name__ == "__main__":
    main()
