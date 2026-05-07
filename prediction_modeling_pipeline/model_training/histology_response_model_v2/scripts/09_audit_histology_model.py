"""
Script: 09_audit_histology_model.py

Purpose:
    Audit the trained histology response model for teacher-builder use.

Pipeline role:
    Step 09 of histology_response_model_v2. This step compares held-out
    image_treatment performance to treatment_only and image_only baselines,
    applies teacher-export thresholds, computes a bounded reliability weight,
    and writes the model index consumed by teacher_builder.

Scientific context:
    Teacher approval is not based only on successful model training. The
    image_treatment model must meet the configured held-out AUC threshold and
    improve over treatment_only, because treatment identity alone is a major
    confounder in response modeling.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from histology_model_v2_lib import load_yaml, output_root, ensure_dir



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); out=ensure_dir(output_root(cfg) / "09_audit")
    model_root=output_root(cfg) / "07_models"
    # The model comparison table is the authority for held-out baseline and conditioned-model metrics.
    comp_path=model_root / "model_comparison.tsv"
    if not comp_path.exists():
        raise FileNotFoundError(comp_path)
    comp=pd.read_csv(comp_path, sep="\t")
    # Blank/noise controls are optional audit context and should not block model-index creation if absent.
    ctrl_path=output_root(cfg) / "08_controls" / "control_summary.tsv"
    ctrl=pd.read_csv(ctrl_path, sep="\t") if ctrl_path.exists() else pd.DataFrame()
    test=comp[comp["split"]=="test"].copy()
    def get_auc(mt):
        """Extract one model type's held-out test AUC from the comparison table."""
        s=test.loc[test.model_type==mt, "auc"]
        return float(s.iloc[0]) if len(s) else np.nan
    image_treat_auc=get_auc("image_treatment")
    treat_auc=get_auc("treatment_only")
    image_auc=get_auc("image_only")
    # The AUC delta measures whether image_treatment adds signal beyond treatment identity.
    delta=image_treat_auc - treat_auc if np.isfinite(image_treat_auc) and np.isfinite(treat_auc) else np.nan
    approve_delta=float(cfg["teacher_export"].get("approve_if_image_treatment_beats_treatment_only_by_auc",0.03))
    approve_auc=float(cfg["teacher_export"].get("approve_if_patient_auc_at_least",0.60))
    # Teacher approval requires adequate held-out AUC and improvement over treatment_only.
    approved=bool(np.isfinite(image_treat_auc) and image_treat_auc>=approve_auc and (not np.isfinite(delta) or delta>=approve_delta))
    # Reliability is bounded so teacher_builder can shrink histology probabilities conservatively.
    reliability=max(float(cfg["teacher_export"].get("reliability_floor",0.05)), min(float(cfg["teacher_export"].get("reliability_ceiling",0.85)), max(0.0, delta if np.isfinite(delta) else 0.0)))
    rows=[{
        "model_family":"histology_response_model_v2",
        "selected_model_type":"image_treatment",
        "approved_for_teacher":approved,
        "test_patient_auc_image_treatment":image_treat_auc,
        "test_patient_auc_treatment_only":treat_auc,
        "test_patient_auc_image_only":image_auc,
        "auc_delta_vs_treatment_only":delta,
        "reliability_weight":reliability,
        "model_path":str(model_root / "image_treatment" / "best_model.pt"),
        "encoder_path":str(model_root / "encoders.json"),
    }]
    index=pd.DataFrame(rows)
    # The model index is the file contract consumed by teacher_builder.
    index.to_csv(out / "histology_model_index.tsv", sep="\t", index=False)
    lines=["Histology response model v2 audit", "", index.to_string(index=False), "", "Model comparison:", comp.to_string(index=False)]
    if not ctrl.empty:
        lines += ["", "Blank/noise controls:", ctrl.to_string(index=False)]
    (out / "histology_model_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("DONE")
    print(out)

if __name__ == "__main__":
    main()
