"""
Script: 04_audit_deployable_models.py

Purpose:
    Audit deployable expression-response model artifacts after training.

Project context:
    This is Step 04 of expression_response_model_v2. It reads the model index,
    cross-validation predictions, and skipped-treatment table produced by Step 03,
    then writes a compact audit package describing trained models, teacher-approved
    models, calibration completeness, reliability weights, and warning/failure
    checks.

Scientific role:
    This step is the gate between expression model training and downstream teacher
    use. It does not retrain models. Instead, it verifies that deployable artifacts
    and their performance summaries are present and interpretable before
    teacher_builder or Visium scoring consumes expression-response outputs.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP04_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# The audit step uses pandas/numpy summaries and does not load or retrain
# model joblib artifacts.

from pathlib import Path
import argparse
import pandas as pd
import numpy as np



# =========================
# Shared expression-model helper imports
# =========================
# Shared helpers keep config loading, path resolution, directory creation,
# and table loading consistent with the training steps.

from expression_model_v2_lib import load_config, resolve_path, ensure_dir, read_table




# =========================
# Command-line interface
# =========================
# The runner passes the YAML config path into this audit step.

def parse_args():
    """Parse the required YAML config path for Step 04 deployable model audit."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Deployable model audit workflow
# =========================
# This workflow summarizes trained models, teacher-approved models, skipped
# drugs, CV prediction completeness, and reliability diagnostics.

def main():
    """Audit trained expression-response models and write checks plus a human-readable summary."""

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve the configured deployable output root and audit subdirectory.
    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    audit_dir = ensure_dir(out_root / "audit")



    # =========================
    # Audit input paths
    # =========================
    # Step 04 audits the model index, cross-validation predictions, and skipped
    # drug table written by Step 03.

    model_index_path = out_root / "model_index.tsv"
    cv_pred_path = out_root / "cv" / "cv_fold_predictions.tsv"
    skipped_path = out_root / "skipped_drugs.tsv"

    # The model index is mandatory because it defines all trained artifacts.
    if not model_index_path.exists():
        raise FileNotFoundError(model_index_path)

    # Load the Step 03 tabular products; optional tables fall back to empty frames.
    model_index = read_table(model_index_path, sep="\t")
    cv_preds = read_table(cv_pred_path, sep="\t") if cv_pred_path.exists() else pd.DataFrame()
    skipped = read_table(skipped_path, sep="\t") if skipped_path.exists() else pd.DataFrame()



    # =========================
    # Audit check registry
    # =========================
    # Checks are collected in a structured table so PASS/WARN/FAIL decisions
    # can be reviewed outside the terminal.

    checks = []



    # =========================
    # Check writer helper
    # =========================
    # Each check is both appended to the audit table and printed to the terminal.

    def add_check(item, status, message):
        """Append one audit check to the report table and echo it to the terminal."""

        checks.append({"item": item, "status": status, "message": message})
        print(f"{status} {item}: {message}")



    # =========================
    # Core model-count checks
    # =========================
    # The first audit gate confirms that trained and teacher-approved model rows
    # are present in the model index.

    n_models = len(model_index)
    n_approved = int(model_index["approved_for_teacher"].sum()) if "approved_for_teacher" in model_index.columns and n_models else 0

    # Basic existence and count checks provide the top-level audit gate.
    add_check("model_index_exists", "PASS", str(model_index_path))
    add_check("n_models", "PASS" if n_models > 0 else "FAIL", str(n_models))
    add_check("n_approved_for_teacher", "PASS" if n_approved > 0 else "WARN", str(n_approved))



    # =========================
    # Calibration and reliability checks
    # =========================
    # Optional model-index columns are checked when available to flag weak,
    # extreme, or low-reliability model behavior.

    if "cv_brier_improvement_vs_prior" in model_index.columns:
        # Brier improvement checks whether models beat the treatment-prior baseline.
        n_better = int((pd.to_numeric(model_index["cv_brier_improvement_vs_prior"], errors="coerce") > 0).sum())
        add_check("models_better_than_prior_brier", "PASS" if n_better > 0 else "WARN", str(n_better))

    if "cv_frac_prob_extreme" in model_index.columns:
        # Extreme probability checks protect teacher_builder from saturated labels.
        extreme = pd.to_numeric(model_index["cv_frac_prob_extreme"], errors="coerce")
        n_extreme = int((extreme > 0.35).sum())
        add_check("models_with_extreme_probabilities", "PASS" if n_extreme == 0 else "WARN", str(n_extreme))

    if "reliability_weight" in model_index.columns:
        # Reliability weights are the downstream shrinkage weights used by teacher fusion.
        rel = pd.to_numeric(model_index["reliability_weight"], errors="coerce")
        add_check("median_reliability_weight", "PASS" if rel.median() > 0 else "WARN", f"{rel.median():.4f}")



    # =========================
    # Cross-validation prediction checks
    # =========================
    # The audit verifies that calibrated out-of-fold probabilities were written
    # for downstream review.

    if len(cv_preds):
        # Missing calibrated probabilities are reported because calibration is required for deployable teacher use.
        missing_cal = int(pd.to_numeric(cv_preds.get("cv_prob_calibrated", pd.Series(dtype=float)), errors="coerce").isna().sum())
        add_check("cv_predictions_rows", "PASS", str(len(cv_preds)))
        add_check("cv_calibrated_missing_values", "PASS" if missing_cal == 0 else "WARN", str(missing_cal))
    else:
        add_check("cv_predictions", "WARN", "missing or empty")

    # Copy the model index into audit/ so reviewers can inspect the exact audited table.
    model_index.to_csv(audit_dir / "model_index_for_audit.tsv", sep="\t", index=False)
    pd.DataFrame(checks).to_csv(audit_dir / "deployable_model_audit_checks.tsv", sep="\t", index=False)



    # =========================
    # Human-readable audit summary
    # =========================
    # The text summary is the reviewer-facing description of model approval,
    # skipped treatments, and audit checks.

    lines = []
    lines.append("Deployable expression model audit summary")
    lines.append("")
    lines.append(f"output_root: {out_root}")
    lines.append(f"models_trained: {n_models}")
    lines.append(f"approved_for_teacher: {n_approved}")
    lines.append(f"skipped_drugs: {len(skipped)}")
    lines.append("")
    lines.append("Approved models:")
    if n_approved and "drug" in model_index.columns:
        approved = model_index[model_index["approved_for_teacher"] == True].copy()
        for _, row in approved.sort_values("reliability_weight", ascending=False).iterrows():
            lines.append(
                f"  {row['drug']} | reliability={float(row['reliability_weight']):.3f} | auc={float(row.get('cv_auc', np.nan)):.3f} | brier_improvement={float(row.get('cv_brier_improvement_vs_prior', np.nan)):.4f}"
            )
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Checks:")
    for c in checks:
        lines.append(f"  {c['status']} {c['item']}: {c['message']}")

    summary_path = audit_dir / "deployable_model_audit_summary.txt"
    # The text summary is the main human-readable audit artifact.
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("DONE")
    print("Wrote:", summary_path)
    print("Wrote:", audit_dir / "deployable_model_audit_checks.tsv")


if __name__ == "__main__":
    main()
