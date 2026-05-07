"""
Script: 05_score_visium_expression_teacher.py

Purpose:
    Score Visium pseudobulk expression samples with deployable expression
    response models.

Project context:
    This is Step 05 of expression_response_model_v2. It loads trained model
    artifacts from the Step 03 model index, filters to teacher-approved models
    when configured, aligns Visium pseudobulk expression columns to each model's
    gene feature space, applies saved calibrators, optionally shrinks probabilities
    toward treatment priors, and writes expression teacher score tables.

Scientific role:
    This step creates the expression-teacher handoff consumed by teacher_builder.
    The output preserves raw, calibrated, prior-shrunk, confidence-weighted, and
    provenance fields so downstream fusion can distinguish sample-specific model
    signal from treatment-prior support.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP05_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 05 scores existing Visium pseudobulk rows; it does not retrain
# expression-response models.

from pathlib import Path
import argparse
import pandas as pd
import numpy as np



# =========================
# Model artifact loading
# =========================
# Deployable model joblib artifacts are loaded exactly as written by Step 03.

from joblib import load



# =========================
# Shared expression-model helper imports
# =========================
# Shared helpers provide config/path handling, table reading, calibrator
# application, and treatment-prior probability shrinkage.

from expression_model_v2_lib import (
    load_config,
    resolve_path,
    ensure_dir,
    read_table,
    apply_calibrator,
    shrink_probability,
)




# =========================
# Command-line interface
# =========================
# The runner passes the YAML config path into this scoring step.

def parse_args():
    """Parse the required YAML config path for Step 05 Visium teacher scoring."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Probability-derived confidence helper
# =========================
# Confidence is highest when predictions are farthest from an uncertain
# 0.5 response probability, then later scaled by model reliability.

def confidence_from_probability(prob):
    # Confidence is based on distance from the uncertainty point at 0.5.
    """Convert probabilities into distance-from-0.5 confidence values."""

    prob = np.asarray(prob, dtype=float)
    return np.clip(np.abs(prob - 0.5) * 2.0, 0.0, 1.0)




# =========================
# Visium expression-teacher scoring workflow
# =========================
# This workflow loads model artifacts, aligns Visium genes, applies
# calibration and shrinkage, and writes row-level and summary teacher outputs.

def main():
    """Score Visium pseudobulk samples with deployable expression models and write teacher tables."""

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve the deployable model output root created by Steps 03 and 04.
    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    model_index_path = out_root / "model_index.tsv"
    if not model_index_path.exists():
        raise FileNotFoundError(model_index_path)



    # =========================
    # Teacher-scoring configuration
    # =========================
    # Config values define the Visium pseudobulk input, sample column, approval
    # filtering, probability shrinkage, clipping bounds, and output directory.

    # Teacher scoring settings live under the teacher_scoring config block.
    scoring_cfg = cfg.get("teacher_scoring", {})
    pseudo_path = resolve_path(cfg, scoring_cfg.get("visium_pseudobulk_table"))
    output_dir = ensure_dir(resolve_path(cfg, scoring_cfg.get("output_dir")))
    sample_col = scoring_cfg.get("visium_sample_col", "slide_id")
    approved_only = bool(scoring_cfg.get("approved_only", True))
    shrink_to_prior = bool(scoring_cfg.get("shrink_to_prior", True))
    clip_low = float(scoring_cfg.get("clip_low", 0.02))
    clip_high = float(scoring_cfg.get("clip_high", 0.98))

    # The Visium pseudobulk table is required because it supplies sample expression vectors.
    if pseudo_path is None or not pseudo_path.exists():
        raise FileNotFoundError(f"Visium pseudobulk table missing: {pseudo_path}")



    # =========================
    # Model index and Visium pseudobulk loading
    # =========================
    # The model index provides artifact paths; the pseudobulk table provides one
    # expression vector per Visium sample.

    model_index = read_table(model_index_path, sep="\t")
    # By default, only models passing the Step 04 teacher-use audit are scored.
    if approved_only and "approved_for_teacher" in model_index.columns:
        model_index = model_index[model_index["approved_for_teacher"] == True].copy()

    pseudo = read_table(pseudo_path, sep="\t")
    if sample_col not in pseudo.columns:
        raise ValueError(f"Visium pseudobulk table missing sample column: {sample_col}")

    score_rows = []
    summary_rows = []



    # =========================
    # Per-model scoring loop
    # =========================
    # Each approved deployable expression model scores all available Visium
    # pseudobulk samples.

    for _, model_row in model_index.iterrows():
        # Each model-index row points to one deployable joblib artifact.
        artifact_path = Path(model_row["model_path"])
        if not artifact_path.exists():
            continue

        artifact = load(artifact_path)
        gene_cols = artifact["gene_columns"]


        # =========================
        # Gene-space alignment
        # =========================
        # Model gene columns define the expected feature space; missing Visium genes
        # are filled with zero after reindexing.

        # Missing model genes are tracked and filled so scoring can proceed transparently.
        missing_genes = [g for g in gene_cols if g not in pseudo.columns]

        X = pseudo.reindex(columns=gene_cols).apply(pd.to_numeric, errors="coerce")
        if missing_genes:
            X[missing_genes] = 0.0
            X = X[gene_cols]

        base_model = artifact["base_model"]


        # =========================
        # Prediction, calibration, and shrinkage
        # =========================
        # Raw model probabilities are calibrated, optionally shrunk toward the
        # treatment prior, clipped, and converted to reliability-weighted confidence.

        # Score the Visium pseudobulk matrix with the fitted base classifier.
        raw_prob = base_model.predict_proba(X)[:, 1]
        # Apply the saved calibrator before teacher probabilities are exported.
        cal_prob = apply_calibrator(raw_prob, artifact.get("calibration_method", "identity"), artifact.get("calibrator"))

        prior = float(artifact.get("response_prior", np.nan))
        reliability = float(artifact.get("reliability_weight", 0.0))

        if shrink_to_prior and np.isfinite(prior):
            # Shrink unreliable model signal toward the treatment response prior.
            final_prob = shrink_probability(cal_prob, prior=prior, reliability=reliability)
        else:
            final_prob = cal_prob

        # Clipping prevents exact 0/1 teacher labels from propagating downstream.
        final_prob = np.clip(final_prob, clip_low, clip_high)
        confidence = confidence_from_probability(final_prob) * reliability

        drug = artifact.get("drug", model_row.get("drug", ""))
        drug_key = artifact.get("drug_key", model_row.get("drug_key", ""))



        # =========================
        # Row-level teacher score output
        # =========================
        # One output row is written for each Visium sample and scored treatment model.

        for sid, raw, cal, final, conf in zip(pseudo[sample_col].astype(str), raw_prob, cal_prob, final_prob, confidence):
            score_rows.append({
                "slide_id": sid,
                "sample_id": sid,
                "drug": drug,
                "drug_key": drug_key,
                "expression_raw_prob_responder": float(raw),
                "expression_calibrated_prob_responder": float(cal),
                "expression_prob_responder": float(final),
                "expression_sample_confidence": float(conf),
                "expression_model_weight": reliability,
                "expression_response_prior": prior,
                "expression_training_rows": int(artifact.get("n_training_rows", 0)),
                "expression_training_responders": int(artifact.get("n_training_responders", 0)),
                "expression_training_non_responders": int(artifact.get("n_training_non_responders", 0)),
                "expression_available": True,
                "expression_model_path": str(artifact_path),
                "expression_quality_flags": "|".join(artifact.get("quality_flags", [])) if artifact.get("quality_flags", []) else "ok",
            })



        # =========================
        # Per-treatment scoring summary
        # =========================
        # Summary rows capture distribution diagnostics and missing-gene counts for
        # each scored treatment model.

        summary_rows.append({
            "drug": drug,
            "drug_key": drug_key,
            "n_visium_slides_scored": int(len(pseudo)),
            "expression_model_weight": reliability,
            "expression_response_prior": prior,
            "mean_raw_prob": float(np.mean(raw_prob)),
            "mean_calibrated_prob": float(np.mean(cal_prob)),
            "mean_final_prob": float(np.mean(final_prob)),
            "std_final_prob": float(np.std(final_prob, ddof=1)) if len(final_prob) > 1 else 0.0,
            "min_final_prob": float(np.min(final_prob)),
            "max_final_prob": float(np.max(final_prob)),
            "n_missing_model_genes": int(len(missing_genes)),
            "approved_for_teacher": bool(artifact.get("approved_for_teacher", False)),
            "quality_flags": "|".join(artifact.get("quality_flags", [])) if artifact.get("quality_flags", []) else "ok",
        })



    # =========================
    # Output table writing
    # =========================
    # Step 05 writes both row-level teacher scores and treatment-level summaries.

    scores = pd.DataFrame(score_rows)
    summary = pd.DataFrame(summary_rows)

    scores_path = output_dir / "expression_teacher_scores.tsv"
    summary_path = output_dir / "expression_teacher_summary.tsv"
    # Row-level scores and per-drug summaries form the expression teacher handoff.
    scores.to_csv(scores_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)



    # =========================
    # Human-readable scoring summary
    # =========================
    # The text summary records the pseudobulk input, model index, approval filter,
    # models scored, slides scored, and row count.

    lines = []
    lines.append("Deployable expression teacher scoring summary")
    lines.append("")
    lines.append(f"visium_pseudobulk_table: {pseudo_path}")
    lines.append(f"model_index: {model_index_path}")
    lines.append(f"approved_only: {approved_only}")
    lines.append(f"models_scored: {summary['drug_key'].nunique() if len(summary) else 0}")
    lines.append(f"slides_scored: {scores['sample_id'].nunique() if len(scores) else 0}")
    lines.append(f"rows_written: {len(scores)}")
    # The text summary is intended for quick review of the scoring handoff.
    (output_dir / "expression_teacher_scoring_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("DONE")
    print("Models scored:", summary["drug_key"].nunique() if len(summary) else 0)
    print("Slides scored:", scores["sample_id"].nunique() if len(scores) else 0)
    print("Rows written:", len(scores))
    print("Wrote:", scores_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
