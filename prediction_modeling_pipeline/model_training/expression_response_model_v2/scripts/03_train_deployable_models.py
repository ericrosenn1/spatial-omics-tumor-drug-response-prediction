"""
Script: 03_train_deployable_models.py

Purpose:
    Train calibrated, deployable expression-response models for eligible
    canonical treatments.

Project context:
    This is Step 03 of expression_response_model_v2. It consumes the canonical
    training table produced by Step 02, trains one treatment-specific expression
    classifier per eligible canonical drug key, evaluates grouped cross-validation
    predictions, fits a post-hoc probability calibrator, computes reliability
    weights, and writes deployable joblib artifacts plus model-index tables.

Scientific role:
    These models are the expression teacher candidates used downstream by
    teacher_builder. The step is intentionally conservative: every treatment
    model must pass minimum support, grouped cross-validation, calibration,
    probability-extremeness, and reliability checks before being approved for
    teacher use.

Documentation polish marker:
    EXPRESSION_MODEL_V2_STEP03_DOC_POLISH_V1

Important:
    This documentation pass is intentionally non-behavioral. Comments,
    section headers, and docstrings may be added, but executable logic must
    remain unchanged.
"""



# =========================
# Imports
# =========================
# Step 03 trains deployable sklearn models and writes joblib artifacts
# plus audit tables for downstream teacher_builder use.

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd



# =========================
# Model serialization import
# =========================
# Trained model artifacts are serialized with joblib for later loading by
# audit and Visium teacher-scoring steps.

from joblib import dump



# =========================
# Scikit-learn modeling components
# =========================
# The expression model uses a transparent classical ML pipeline:
# imputation, variance filtering, scaling, PCA, logistic regression,
# grouped cross-validation, and optional calibration.

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# =========================
# Shared expression-model helper imports
# =========================
# Shared helpers provide config/path handling, feature discovery,
# calibration utilities, reliability weighting, and JSON writing.

from expression_model_v2_lib import (
    load_config,
    resolve_path,
    ensure_dir,
    read_table,
    find_gene_columns,
    safe_name,
    sigmoid_logit,
    expected_calibration_error,
    reliability_weight,
    apply_calibrator,
    write_json,
)




# =========================
# Command-line interface
# =========================
# The runner passes the YAML config path through each numbered step.

def parse_args():
    """Parse the required YAML config path for Step 03 deployable model training."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()




# =========================
# Model pipeline construction
# =========================
# Each treatment model uses the same preprocessing and classifier structure
# so treatment-level results are comparable and auditable.

def build_pipeline(n_pca, cfg):
    # Pipeline steps are deliberately explicit for auditability.
    """Construct the impute-filter-scale-PCA-logistic-regression pipeline for one treatment model."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(threshold=float(cfg.get("variance_threshold", 0.0)))),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(n_pca), random_state=int(cfg.get("random_state", 42)))),
            (
                "logreg",
                LogisticRegression(
                    max_iter=int(cfg.get("max_iter", 3000)),
                    C=float(cfg.get("logreg_c", 1.0)),
                    class_weight=cfg.get("class_weight", "balanced"),
                    random_state=int(cfg.get("random_state", 42)),
                ),
            ),
        ]
    )




# =========================
# PCA dimension guard
# =========================
# PCA components must be bounded by fold-specific sample and feature counts.

def choose_pca(n_rows, n_features, requested):
    # PCA cannot request more components than fold samples or available genes allow.
    """Choose a PCA component count that is valid for the current sample and feature dimensions."""

    max_allowed = min(int(n_rows) - 1, int(n_features))
    if max_allowed < 2:
        return None
    return int(min(int(requested), max_allowed))




# =========================
# Metric safety helper
# =========================
# Some folds or treatment subsets can make metrics undefined; returning NaN
# keeps the audit table complete without hiding the issue.

def safe_metric(func, *args, **kwargs):
    """Run a metric function and return NaN instead of failing when a metric is undefined."""

    try:
        return float(func(*args, **kwargs))
    except Exception:
        return np.nan




# =========================
# Probability calibration helper
# =========================
# Calibration is fitted on out-of-fold predictions so deployable artifacts
# can avoid using raw classifier probabilities directly.

def fit_calibrator(y_true, raw_prob, method):
    """Fit an identity, sigmoid, or isotonic calibrator from out-of-fold probabilities."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(raw_prob, dtype=float)
    # Calibrators are fit only where out-of-fold probabilities are finite.
    mask = np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-6, 1.0 - 1e-6)

    if len(y) < 10 or len(np.unique(y)) < 2:
        return "identity", None

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p, y)
        return "isotonic", cal

    if method == "sigmoid":
        x = sigmoid_logit(p).reshape(-1, 1)
        cal = LogisticRegression(max_iter=1000, solver="lbfgs")
        cal.fit(x, y)
        return "sigmoid", cal

    return "identity", None




# =========================
# Prediction summary metrics
# =========================
# This metric block combines class balance, discrimination, calibration,
# Brier improvement, and probability-extremeness diagnostics.

def summarize_predictions(y_true, prob, pred):
    """Summarize discrimination, calibration, probability distribution, and classification metrics."""

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = np.asarray(pred, dtype=int)

    # The treatment-specific response rate is the baseline prior comparator.
    response_rate = float(y.mean())
    prior_brier = float(np.mean((y - response_rate) ** 2))

    out = {
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "response_rate": response_rate,
        "prob_mean": float(np.mean(p)),
        "prob_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
        "prob_min": float(np.min(p)),
        "prob_max": float(np.max(p)),
        "frac_prob_lt_0_01": float(np.mean(p < 0.01)),
        "frac_prob_gt_0_99": float(np.mean(p > 0.99)),
        "frac_prob_extreme": float(np.mean((p < 0.01) | (p > 0.99))),
        "prior_brier": prior_brier,
        "brier": safe_metric(brier_score_loss, y, p),
        "log_loss": safe_metric(log_loss, y, p, labels=[0, 1]),
        "accuracy": safe_metric(accuracy_score, y, pred),
        "f1": safe_metric(f1_score, y, pred, zero_division=0),
        "precision": safe_metric(precision_score, y, pred, zero_division=0),
        "recall": safe_metric(recall_score, y, pred, zero_division=0),
        "ece_10bin": expected_calibration_error(y, p, 10),
    }

    if len(np.unique(y)) == 2:
        out["auc"] = safe_metric(roc_auc_score, y, p)
        out["average_precision"] = safe_metric(average_precision_score, y, p)
    else:
        out["auc"] = np.nan
        out["average_precision"] = np.nan

    # Brier improvement tests whether the model improves over the treatment prior.
    out["brier_improvement_vs_prior"] = out["prior_brier"] - out["brier"] if np.isfinite(out["brier"]) else np.nan
    majority_accuracy = max(response_rate, 1.0 - response_rate)
    out["majority_class_accuracy"] = majority_accuracy
    out["accuracy_improvement_vs_majority"] = out["accuracy"] - majority_accuracy if np.isfinite(out["accuracy"]) else np.nan

    return out




# =========================
# Deployable model training workflow
# =========================
# Main workflow: load canonical data, train per-treatment models, collect
# cross-validation predictions, calibrate probabilities, gate teacher use,
# serialize artifacts, and write indexes.

def main():
    """Train deployable calibrated treatment models and write model artifacts, CV outputs, and indexes."""

    args = parse_args()
    cfg = load_config(args.config)

    out_root = ensure_dir(resolve_path(cfg, cfg["output_root"]))
    data_dir = out_root / "data"
    canonical_path = data_dir / "training_table_canonical.tsv"

    if not canonical_path.exists():
        raise FileNotFoundError(f"Run 02 first. Missing: {canonical_path}")

    model_dir = ensure_dir(out_root / "models")
    cv_dir = ensure_dir(out_root / "cv")
    audit_dir = ensure_dir(out_root / "audit")

    case_col = cfg.get("case_col", "cases.case_id")
    requested_pca = int(cfg.get("pca_components", 50))
    n_splits_requested = int(cfg.get("cv_n_splits", 5))
    calibration_method = cfg.get("calibration_method", "sigmoid")

    min_rows = int(cfg.get("min_rows_per_drug", 30))
    min_cases = int(cfg.get("min_cases_per_drug", 30))
    min_class = int(cfg.get("min_rows_per_class", 8))



    # =========================
    # Teacher quality gates
    # =========================
    # Configurable gates decide whether a trained expression model is approved
    # for teacher_builder use.

    quality = cfg.get("quality", {})
    min_successful_folds = int(quality.get("min_successful_folds", 3))
    min_auc = float(quality.get("min_auc_for_teacher", 0.55))
    min_brier_improvement = float(quality.get("min_brier_improvement_for_teacher", 0.0))
    max_extreme_fraction = float(quality.get("max_extreme_fraction_for_teacher", 0.35))
    min_reliability = float(quality.get("min_reliability_weight_for_teacher", 0.05))

    print("Loading canonical training table...")
    df = read_table(canonical_path, sep="\t")

    # The canonical gene list defines the deployable feature space for every model.
    gene_cols = find_gene_columns(df, gene_prefix=cfg.get("gene_prefix", "ENSG"))
    if not gene_cols:
        raise ValueError("No gene columns found.")

    drug_keys = sorted(df["drug_key"].dropna().unique())
    print("Canonical drugs:", len(drug_keys))
    print("Gene columns:", len(gene_cols))

    model_index_rows = []
    skipped_rows = []
    fold_rows = []
    pred_frames = []



    # =========================
    # Treatment-specific training loop
    # =========================
    # One calibrated deployable model is attempted for each canonical treatment
    # with sufficient labeled response support.

    for i, drug_key in enumerate(drug_keys, start=1):
        sub = df[df["drug_key"] == drug_key].copy()
        drug = sub["canonical_drug"].iloc[0] if "canonical_drug" in sub.columns else drug_key

        n_rows = len(sub)
        n_cases = sub[case_col].nunique()
        class_counts = sub["y"].value_counts().to_dict()
        n_pos = int(class_counts.get(1, 0))
        n_neg = int(class_counts.get(0, 0))
        response_prior = float(sub["y"].mean()) if n_rows else np.nan

        # Skip treatments that cannot support a minimally balanced supervised model.
        if n_rows < min_rows or n_cases < min_cases or n_pos < min_class or n_neg < min_class:
            reason = []
            if n_rows < min_rows:
                reason.append("too_few_rows")
            if n_cases < min_cases:
                reason.append("too_few_cases")
            if n_pos < min_class or n_neg < min_class:
                reason.append("too_few_rows_in_one_class")
            skipped_rows.append({
                "drug": drug,
                "drug_key": drug_key,
                "reason": "|".join(reason),
                "n_rows": n_rows,
                "n_cases": n_cases,
                "n_responder": n_pos,
                "n_non_responder": n_neg,
            })
            continue

        print(f"[{i}/{len(drug_keys)}] {drug} rows={n_rows} cases={n_cases} pos={n_pos} neg={n_neg}")

        X = sub[gene_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        y = sub["y"].astype(int).to_numpy()
        groups = sub[case_col].astype(str).to_numpy()

        n_splits = min(n_splits_requested, n_cases)
        # Grouped folds prevent the same case from appearing in train and test partitions.
        gkf = GroupKFold(n_splits=n_splits)

        oof_prob = np.full(shape=len(sub), fill_value=np.nan, dtype=float)
        oof_fold = np.full(shape=len(sub), fill_value=-1, dtype=int)
        n_folds_ok = 0



        # =========================
        # Grouped cross-validation loop
        # =========================
        # GroupKFold keeps all rows from the same case together, reducing patient
        # leakage between training and test folds.

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Folds missing either class cannot provide meaningful discrimination metrics.
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                fold_rows.append({
                    "drug": drug,
                    "drug_key": drug_key,
                    "fold": fold_idx,
                    "status": "skipped_missing_class",
                    "n_train_rows": len(train_idx),
                    "n_test_rows": len(test_idx),
                    "train_responder": int((y_train == 1).sum()),
                    "train_non_responder": int((y_train == 0).sum()),
                    "test_responder": int((y_test == 1).sum()),
                    "test_non_responder": int((y_test == 0).sum()),
                })
                continue

            n_pca = choose_pca(len(train_idx), len(gene_cols), requested_pca)
            if n_pca is None:
                fold_rows.append({
                    "drug": drug,
                    "drug_key": drug_key,
                    "fold": fold_idx,
                    "status": "skipped_not_enough_pca",
                    "n_train_rows": len(train_idx),
                    "n_test_rows": len(test_idx),
                })
                continue

            # Fit only on the training fold, then score the held-out fold.
            model = build_pipeline(n_pca, cfg)
            model.fit(X.iloc[train_idx], y_train)

            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)

            oof_prob[test_idx] = prob
            oof_fold[test_idx] = fold_idx
            n_folds_ok += 1

            m = summarize_predictions(y_test, prob, pred)
            fold_row = {
                "drug": drug,
                "drug_key": drug_key,
                "fold": fold_idx,
                "status": "ok",
                "n_train_rows": len(train_idx),
                "n_test_rows": len(test_idx),
                "train_cases": int(len(set(groups[train_idx]))),
                "test_cases": int(len(set(groups[test_idx]))),
                "train_responder": int((y_train == 1).sum()),
                "train_non_responder": int((y_train == 0).sum()),
                "test_responder": int((y_test == 1).sum()),
                "test_non_responder": int((y_test == 0).sum()),
                "pca_components": n_pca,
            }
            fold_row.update(m)
            fold_rows.append(fold_row)

        # Only held-out predictions contribute to calibration and CV quality metrics.
        valid = np.isfinite(oof_prob)
        if valid.sum() < max(10, min_rows // 2) or len(np.unique(y[valid])) < 2:
            skipped_rows.append({
                "drug": drug,
                "drug_key": drug_key,
                "reason": "insufficient_valid_oof_predictions",
                "n_rows": n_rows,
                "n_cases": n_cases,
                "n_oof": int(valid.sum()),
                "n_successful_folds": n_folds_ok,
            })
            continue



        # =========================
        # Calibration and reliability scoring
        # =========================
        # Out-of-fold predictions are calibrated, summarized, and converted into a
        # bounded reliability weight for downstream teacher fusion.

        fitted_cal_method, calibrator = fit_calibrator(y[valid], oof_prob[valid], calibration_method)
        oof_prob_cal = np.full_like(oof_prob, np.nan)
        oof_prob_cal[valid] = apply_calibrator(oof_prob[valid], fitted_cal_method, calibrator)
        oof_pred_cal = (oof_prob_cal[valid] >= 0.5).astype(int)

        cv_metrics = summarize_predictions(y[valid], oof_prob_cal[valid], oof_pred_cal)
        rel = reliability_weight(
            auc=cv_metrics.get("auc", np.nan),
            brier_improvement=cv_metrics.get("brier_improvement_vs_prior", np.nan),
            prior_brier=cv_metrics.get("prior_brier", np.nan),
            ece=cv_metrics.get("ece_10bin", np.nan),
            n_rows=n_rows,
            n_folds_ok=n_folds_ok,
            n_folds_requested=n_splits,
        )

        # Approval flags are intentionally explicit so weak models remain visible.
        quality_flags = []
        if n_folds_ok < min_successful_folds:
            quality_flags.append("too_few_successful_folds")
        if not np.isfinite(cv_metrics.get("auc", np.nan)) or cv_metrics["auc"] < min_auc:
            quality_flags.append("weak_auc")
        if not np.isfinite(cv_metrics.get("brier_improvement_vs_prior", np.nan)) or cv_metrics["brier_improvement_vs_prior"] < min_brier_improvement:
            quality_flags.append("not_better_than_prior_brier")
        if cv_metrics.get("frac_prob_extreme", 1.0) > max_extreme_fraction:
            quality_flags.append("extreme_probabilities")
        if rel < min_reliability:
            quality_flags.append("low_reliability_weight")

        approved = len(quality_flags) == 0

        n_pca_final = choose_pca(n_rows, len(gene_cols), requested_pca)
        # After CV audit, refit the deployable model on all available rows for this treatment.
        final_model = build_pipeline(n_pca_final, cfg)
        final_model.fit(X, y)

        safe = safe_name(drug_key)
        model_path = model_dir / f"{safe}.joblib"



        # =========================
        # Deployable artifact payload
        # =========================
        # Each joblib artifact stores the fitted pipeline, calibrator, feature list,
        # prior, reliability, quality flags, and training support counts.

        artifact = {
            "model_version": "expression_response_model_v2",
            "drug": drug,
            "drug_key": drug_key,
            "case_col": case_col,
            "gene_columns": gene_cols,
            "base_model": final_model,
            "calibration_method": fitted_cal_method,
            "calibrator": calibrator,
            "response_prior": response_prior,
            "reliability_weight": rel,
            "approved_for_teacher": approved,
            "quality_flags": quality_flags,
            "cv_metrics": cv_metrics,
            "n_training_rows": n_rows,
            "n_training_cases": int(n_cases),
            "n_training_responders": int(n_pos),
            "n_training_non_responders": int(n_neg),
            "pca_components_final": int(n_pca_final),
        }
        # The saved artifact is what teacher scoring loads; no retraining should happen downstream.
        dump(artifact, model_path)

        pred_meta_cols = [
            c for c in [
                "episode_key",
                case_col,
                "cases.submitter_id",
                "project.project_id",
                "cases.primary_site",
                "cases.disease_type",
                "resolved_drug_original",
                "canonical_drug",
                "drug_key",
                "resolved_episode_binary_response",
                "y",
            ]
            if c in sub.columns
        ]
        pred_df = sub[pred_meta_cols].copy()
        pred_df["cv_fold"] = oof_fold
        pred_df["cv_prob_raw"] = oof_prob
        pred_df["cv_prob_calibrated"] = oof_prob_cal
        pred_df["cv_pred_calibrated"] = (pred_df["cv_prob_calibrated"] >= 0.5).astype("float")
        pred_frames.append(pred_df)

        idx = {
            "drug": drug,
            "drug_key": drug_key,
            "model_path": str(model_path),
            "approved_for_teacher": bool(approved),
            "quality_flags": "|".join(quality_flags) if quality_flags else "ok",
            "reliability_weight": rel,
            "response_prior": response_prior,
            "n_training_rows": n_rows,
            "n_training_cases": n_cases,
            "n_training_responders": n_pos,
            "n_training_non_responders": n_neg,
            "n_successful_folds": n_folds_ok,
            "n_folds_requested": n_splits,
            "calibration_method": fitted_cal_method,
            "pca_components_final": n_pca_final,
        }
        idx.update({f"cv_{k}": v for k, v in cv_metrics.items()})
        model_index_rows.append(idx)



    # =========================
    # Output table assembly
    # =========================
    # Model indexes, skipped-treatment reports, fold summaries, and predictions
    # are written as tabular audit products.

    model_index = pd.DataFrame(model_index_rows)
    skipped = pd.DataFrame(skipped_rows)
    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()

    model_index_path = out_root / "model_index.tsv"
    approved_path = out_root / "model_index_approved.tsv"
    skipped_path = out_root / "skipped_drugs.tsv"
    folds_path = cv_dir / "cv_fold_summary.tsv"
    preds_path = cv_dir / "cv_fold_predictions.tsv"

    # Index files are the contract consumed by audit and teacher-scoring steps.
    model_index.to_csv(model_index_path, sep="\t", index=False)
    if len(model_index):
        model_index[model_index["approved_for_teacher"] == True].to_csv(approved_path, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(approved_path, sep="\t", index=False)
    skipped.to_csv(skipped_path, sep="\t", index=False)
    folds.to_csv(folds_path, sep="\t", index=False)
    preds.to_csv(preds_path, sep="\t", index=False)



    # =========================
    # Training run summary
    # =========================
    # The compact JSON summary records model counts and key output paths.

    run_summary = {
        "n_drugs_with_models": int(len(model_index)),
        "n_approved_for_teacher": int(model_index["approved_for_teacher"].sum()) if len(model_index) else 0,
        "n_skipped": int(len(skipped)),
        "n_gene_columns": int(len(gene_cols)),
        "model_index": str(model_index_path),
        "approved_index": str(approved_path),
    }
    # The JSON summary gives a compact machine-readable run overview.
    write_json(run_summary, audit_dir / "training_run_summary.json")

    print("DONE")
    print("Models trained:", run_summary["n_drugs_with_models"])
    print("Approved for teacher:", run_summary["n_approved_for_teacher"])
    print("Skipped drugs:", run_summary["n_skipped"])
    print("Wrote:", model_index_path)
    print("Wrote:", approved_path)
    print("Wrote:", skipped_path)
    print("Wrote:", folds_path)
    print("Wrote:", preds_path)


if __name__ == "__main__":
    main()
