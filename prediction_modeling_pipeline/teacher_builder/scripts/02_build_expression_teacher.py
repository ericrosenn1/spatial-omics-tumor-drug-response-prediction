"""
Script: 02_build_expression_teacher.py

Purpose:
    Build governed expression-response teacher scores for Visium samples.

Project context:
    This is Step 02 of the governed teacher_builder workflow. It discovers
    processed Visium h5ad files, builds pseudobulk expression profiles aligned
    to the expression-response model feature space, loads approved deployable
    expression model metadata, scores available treatment models, and writes
    expression teacher tables for downstream fusion.

Scientific role:
    Expression-response models provide one treatment-specific teacher modality.
    This step preserves model reliability, treatment prior, artifact provenance,
    scoring mode, and prior-only fallback information so later fusion can shrink
    and audit teacher labels conservatively.

Documentation polish marker:
    TEACHER_BUILDER_STEP02_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments, section
    headers, and docstrings may be added, but executable logic, paths, thresholds,
    schemas, and outputs must remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 02 uses pandas/numpy, optional h5ad reading, and model artifact loading.

from __future__ import annotations

from pathlib import Path
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

from scipy.io import mmread



# =========================
# Shared governance helper imports
# =========================
# Shared helpers standardize config paths, treatment keys, priors, artifacts, and confidence.

from teacher_governance_lib import (
    load_config,
    cfg_path,
    resolve_path,
    ensure_dir,
    read_table,
    write_table,
    normalize_key,
    display_drug_name,
    find_gene_columns,
    parse_model_index,
    find_artifact_path,
    confidence_from_probability,
    lookup_prior,
    safe_float,
    clean_text,
)




# =========================
# Command-line interface
# =========================
# The governed runner passes the YAML config path into this script.

def parse_args():
    """Parse the governed teacher_builder YAML config path."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Processed Visium discovery
# =========================
# Processed h5ad files define which samples can receive expression teacher scores.

def discover_processed_h5ad(processed_root: Path) -> pd.DataFrame:
    """Discover processed Visium h5ad files and return a sample-level manifest."""

    rows = []
    # Missing processed root returns an empty manifest so the summary can still explain the issue.
    if processed_root is None or not processed_root.exists():
        return pd.DataFrame()
    for path in sorted(processed_root.rglob("02_processed.h5ad")):
        # Processed h5ad files are expected under sample_id/adata/02_processed.h5ad.
        sample_id = path.parents[1].name
        rows.append(
            {
                "slide_id": sample_id,
                "sample_id": sample_id,
                "expr_path": str(path),
                "expr_format": "h5ad_processed",
                "source_priority": 1,
            }
        )
    return pd.DataFrame(rows)




# =========================
# Pseudobulk expression construction
# =========================
# Expression values are collapsed to one sample-level vector in the model feature space.

def pseudobulk_from_matrix(X, gene_ids: list[str], gene_cols: list[str], already_log_normalized: bool) -> dict:
    # Support both gene-by-cell and cell-by-gene matrix orientations.
    """Collapse an expression matrix into a model-aligned pseudobulk gene vector."""

    if X.shape[0] == len(gene_ids):
        values = np.asarray(X.sum(axis=1)).ravel()
    else:
        values = np.asarray(X.sum(axis=0)).ravel()

    values = values.astype(float)

    # Raw-count inputs are converted to CPM-like scale and log1p transformed.
    if not already_log_normalized:
        total = values.sum()
        if total > 0:
            values = values / total * 1_000_000
        else:
            values = np.zeros_like(values)
        values = np.log1p(values)

    gene_to_value = dict(zip(gene_ids, values))
    # Gene IDs may include version suffixes; keep a version-stripped lookup as fallback.
    gene_to_value_nover = {str(g).split(".")[0]: v for g, v in zip(gene_ids, values)}

    out = {}
    for gene in gene_cols:
        out[gene] = gene_to_value.get(gene, gene_to_value_nover.get(str(gene).split(".")[0], 0.0))
    return out




# =========================
# Processed h5ad reader
# =========================
# The reader prefers adata.raw when available and otherwise uses adata.X.

def read_h5ad_pseudobulk(h5ad_path: Path, gene_cols: list[str]) -> dict:
    """Read one processed h5ad file and return a pseudobulk vector for model genes."""

    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)

    # Prefer raw expression when present so pseudobulk values retain maximal gene coverage.
    if adata.raw is not None:
        X = adata.raw.X
        gene_ids = adata.raw.var_names.astype(str).tolist()
    else:
        X = adata.X
        gene_ids = adata.var_names.astype(str).tolist()

    return pseudobulk_from_matrix(X, gene_ids, gene_cols, already_log_normalized=True)




# =========================
# Model artifact loading
# =========================
# Deployable model artifacts may be joblib or pickle compatible.

def load_model(path: Path):
    # Prefer joblib for sklearn artifacts and fall back to pickle for compatibility.
    """Load a serialized model artifact using joblib with pickle fallback."""

    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)




# =========================
# Predictor extraction
# =========================
# Artifacts may wrap the sklearn predictor under several common metadata keys.

def unwrap_predictor(obj):
    # Some artifacts are the predictor itself rather than a metadata dictionary.
    """Extract the predict_proba-capable estimator from a loaded artifact object."""

    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in [
            "calibrated_model",
            "model",
            "pipeline",
            "classifier",
            "estimator",
            "best_model",
        ]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    return None




# =========================
# Probability scoring helper
# =========================
# This helper keeps probability prediction robust to pandas or NumPy feature inputs.

def predict_probability(model, X):
    """Return responder probabilities while suppressing expected feature-name warnings."""

    if model is None:
        return None

    # Suppress expected sklearn feature-name warnings while preserving failure handling.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return model.predict_proba(X.to_numpy())[:, 1]




# =========================
# Expression teacher-building workflow
# =========================
# Main workflow: load feature space, build pseudobulk, score models, and write outputs.

def main():
    """Build pseudobulk expression, score approved expression models, and write teacher tables."""

    args = parse_args()
    cfg = load_config(args.config)

    out_root = Path(cfg["output_root"])
    output_dir = ensure_dir(out_root / "02_expression_teacher")
    validation_dir = out_root / "01_input_validation"

    sample_col = cfg.get("sample_col", "sample_id")



    # =========================
    # Expression feature-space loading
    # =========================
    # The expression training table defines the canonical ENSG feature list.

    training_path = cfg_path(cfg, "expression_training_table")
    if training_path is None or not training_path.exists():
        raise FileNotFoundError(f"expression_training_table not found: {training_path}")

    training = read_table(training_path)
    # The upstream expression-response training table defines the model gene feature space.
    gene_cols = find_gene_columns(training)
    if not gene_cols:
        raise ValueError("No ENSG gene columns detected in expression training table.")



    # =========================
    # Governed sample selection
    # =========================
    # The spatial feature table provides the Visium samples used in this run.

    spatial = read_table(cfg_path(cfg, "spatial_feature_table"))
    # Score only samples represented in the governed spatial feature table.
    selected_samples = spatial[sample_col].astype(str).drop_duplicates().tolist()
    if bool(cfg.get("test_mode", False)):
        selected_samples = selected_samples[: int(cfg.get("test_n_samples", 5))]

    processed_root = cfg_path(cfg, "processed_samples_dir")
    slides = discover_processed_h5ad(processed_root)
    slides = slides[slides["sample_id"].astype(str).isin(selected_samples)].copy()

    slides = slides.sort_values("sample_id").reset_index(drop=True)
    write_table(slides, output_dir / "expression_teacher_slide_manifest.tsv")

    print("")
    print("Expression teacher v2 scoring")
    print("=" * 70)
    print("selected_samples:", len(selected_samples))
    print("expression inputs found:", len(slides))
    print("gene columns:", len(gene_cols))

    pseudo_rows = []
    skipped_sample_rows = []



    # =========================
    # Per-sample pseudobulk loop
    # =========================
    # Each processed sample is collapsed into one pseudobulk row.

    for i, row in slides.iterrows():
        sid = str(row["sample_id"])
        print(f"[{i + 1}/{len(slides)}] pseudobulk {sid}")

        try:
            vals = read_h5ad_pseudobulk(Path(row["expr_path"]), gene_cols)
            vals["sample_id"] = sid
            vals["slide_id"] = sid
            pseudo_rows.append(vals)
        except Exception as e:
            skipped_sample_rows.append(
                {
                    "sample_id": sid,
                    "reason": "pseudobulk_failed",
                    "error": str(e),
                    "expr_path": row["expr_path"],
                }
            )

    if not pseudo_rows:
        raise ValueError("No pseudobulk expression rows could be built.")

    pseudobulk = pd.DataFrame(pseudo_rows)
    pseudobulk = pseudobulk[["sample_id", "slide_id"] + gene_cols]
    # Persist pseudobulk expression so teacher_builder runs are auditable and reusable.
    write_table(pseudobulk, output_dir / "visium_pseudobulk_expression.tsv")

    X = pseudobulk[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)



    # =========================
    # Approved model index and priors
    # =========================
    # Approved model metadata and priors control treatment-level expression scoring.

    approved_path = cfg_path(cfg, "expression_model_index_approved")
    index_path = cfg_path(cfg, "expression_model_index")

    # Prefer approved models, but fall back to the full model index if needed.
    if approved_path is None or not approved_path.exists():
        approved_path = index_path

    if approved_path is None or not approved_path.exists():
        raise FileNotFoundError("No expression model index or approved index found.")

    model_index = parse_model_index(read_table(approved_path))
    # Fusion should consume only expression models approved for teacher use.
    model_index = model_index[model_index["approved_for_teacher"] == True].copy()

    priors_path = validation_dir / "treatment_priors.tsv"
    priors = read_table(priors_path) if priors_path.exists() else pd.DataFrame()

    deploy_root = cfg_path(cfg, "expression_model_v2_deployable_root")
    expr_root = cfg_path(cfg, "expression_model_v2_root")
    base_dirs = [p for p in [deploy_root, expr_root, approved_path.parent] if p is not None]

    score_rows = []
    summary_rows = []
    skipped_drug_rows = []



    # =========================
    # Treatment-specific scoring loop
    # =========================
    # Each approved canonical treatment model scores every available pseudobulk sample.

    for _, mrow in model_index.iterrows():
        drug_key = normalize_key(mrow["canonical_treatment_key"])
        drug = display_drug_name(drug_key)
        rel = safe_float(mrow.get("reliability_weight"), 0.0)

        # Resolve each model artifact from the normalized model-index row and candidate roots.
        artifact = find_artifact_path(mrow, base_dirs)
        model = None
        teacher_mode = "expression_response_model_v2"

        if artifact is not None and artifact.exists():
            try:
                obj = load_model(artifact)
                model = unwrap_predictor(obj)
                if model is None:
                    teacher_mode = "artifact_loaded_no_predict_proba_prior_only"
            except Exception as e:
                skipped_drug_rows.append(
                    {
                        "drug": drug,
                        "drug_key": drug_key,
                        "reason": "model_artifact_load_failed",
                        "model_artifact_path": str(artifact),
                        "error": str(e),
                    }
                )
                teacher_mode = "artifact_load_failed_prior_only"
        else:
            teacher_mode = "model_artifact_missing_prior_only"

        probs = None

        if model is not None:
            try:
                probs = predict_probability(model, X)
            except Exception as e:
                skipped_drug_rows.append(
                    {
                        "drug": drug,
                        "drug_key": drug_key,
                        "reason": "model_predict_failed",
                        "model_artifact_path": str(artifact) if artifact is not None else "",
                        "error": str(e),
                    }
                )
                teacher_mode = "predict_failed_prior_only"

        # Every treatment row carries prior metadata for downstream shrinkage and review.
        prior_info = lookup_prior(drug_key, priors)



        # =========================
        # Prior-only fallback
        # =========================
        # If artifact scoring is unavailable, the treatment prior is emitted with explicit provenance.

        if probs is None:
            # Prior-only fallback preserves a complete sample-by-treatment table while flagging provenance.
            probs = np.repeat(float(prior_info["treatment_prior"]), len(pseudobulk))

        # Clip expression probabilities before fusion so exact 0/1 labels do not propagate.
        probs = np.clip(np.asarray(probs, dtype=float), 0.01, 0.99)

        for sid, slide_id, prob in zip(pseudobulk["sample_id"], pseudobulk["slide_id"], probs):
            score_rows.append(
                {
                    "sample_id": sid,
                    "slide_id": slide_id,
                    "drug": drug,
                    "drug_key": drug_key,
                    "expression_available": True,
                    "expression_prob_raw": float(prob),
                    "expression_prob_calibrated": float(prob),
                    "expression_prob_responder": float(prob),
                    "expression_sample_confidence": confidence_from_probability(prob),
                    "expression_reliability_weight": rel,
                    "expression_model_weight": rel,
                    "expression_model_source": str(artifact) if artifact is not None else "",
                    "expression_teacher_mode": teacher_mode,
                    "expression_training_rows": safe_float(mrow.get("n_rows", mrow.get("training_rows", np.nan)), np.nan),
                    "expression_auc": safe_float(mrow.get("auc"), np.nan),
                    "expression_brier_improvement": safe_float(mrow.get("brier_improvement"), np.nan),
                    "treatment_prior": float(prior_info["treatment_prior"]),
                    "prior_source": prior_info["prior_source"],
                    "prior_n": prior_info["prior_n"],
                }
            )

        summary_rows.append(
            {
                "drug": drug,
                "drug_key": drug_key,
                "n_visium_slides_scored": int(len(pseudobulk)),
                "reliability_weight": rel,
                "teacher_mode": teacher_mode,
                "model_artifact_path": str(artifact) if artifact is not None else "",
                "mean_expression_prob_calibrated": float(np.mean(probs)),
                "std_expression_prob_calibrated": float(np.std(probs, ddof=1)) if len(probs) > 1 else 0.0,
                "prior_prob": float(prior_info["treatment_prior"]),
                "prior_source": prior_info["prior_source"],
                "prior_n": prior_info["prior_n"],
            }
        )



    # =========================
    # Output table writing
    # =========================
    # Expression teacher scores, summaries, and skip tables are consumed by fusion.

    scores = pd.DataFrame(score_rows)
    summary = pd.DataFrame(summary_rows)
    skipped_drugs = pd.DataFrame(skipped_drug_rows)
    skipped_samples = pd.DataFrame(skipped_sample_rows)

    write_table(scores, output_dir / "expression_teacher_scores.tsv")
    write_table(summary, output_dir / "expression_teacher_summary.tsv")
    write_table(skipped_drugs, output_dir / "expression_teacher_skipped_drugs.tsv")
    write_table(skipped_samples, output_dir / "expression_teacher_skipped_samples.tsv")



    # =========================
    # Human-readable summary
    # =========================
    # The text summary records sample/model counts and output provenance.

    lines = [
        "Expression teacher v2 summary",
        "",
        f"samples_selected: {len(selected_samples)}",
        f"samples_scored: {scores['sample_id'].nunique() if not scores.empty else 0}",
        f"approved_drugs_scored: {scores['drug_key'].nunique() if not scores.empty else 0}",
        f"rows: {len(scores)}",
        "",
        "Teacher mode:",
        "  expression_response_model_v2 deployable scoring when artifacts are loadable",
        "  prior only fallback is marked per treatment if artifact loading or prediction fails",
        "",
        "Output role:",
        "  input to 04_fuse_teacher_tables.py",
        "  all expression scores must be shrunk toward treatment priors during fusion",
    ]

    summary_text = "\n".join(lines)
    (output_dir / "expression_teacher_summary.txt").write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print("DONE")
    print(output_dir)


if __name__ == "__main__":
    main()
