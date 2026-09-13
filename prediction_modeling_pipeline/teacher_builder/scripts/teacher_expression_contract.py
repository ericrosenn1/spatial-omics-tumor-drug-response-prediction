"""Reproduce the saved expression teacher contract without training new models.

The Visium transfer representation is log1p(CPM), using raw counts from QC-retained
spots. Training used log1p(TPM); this is a recorded cross-domain approximation,
not evidence of equivalent preprocessing or calibrated Visium outcomes.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


def normalized_gene_ids(values):
    # Ensembl PAR_Y is a distinct trained feature, not a version suffix.
    return pd.Index([re.sub(r"\.\d+(?=_|$)", "", str(x).strip()) for x in values])


def select_gene_ids(var, var_names):
    """Use Ensembl metadata rather than gene symbols; permit unexpressed genes."""
    values = var["gene_ids"] if "gene_ids" in var else var_names
    ids = normalized_gene_ids(values)
    if not ids.str.startswith("ENSG").any():
        raise ValueError("No Ensembl identifiers in var['gene_ids'] or var_names")
    return ids


def raw_counts_to_pseudobulk(X, gene_ids, ordered_genes):
    """Sum duplicate gene counts BEFORE library normalization and logarithms."""
    if X.shape[1] != len(gene_ids):
        raise ValueError("Expected spot-by-gene raw count matrix")
    from scipy.sparse import issparse
    raw_values = X.data if issparse(X) else np.asarray(X)
    if not np.isfinite(raw_values).all() or (raw_values < 0).any():
        raise ValueError("Raw matrix contains negative or nonfinite measurements")
    counts = np.asarray(X.sum(axis=0), dtype=float).ravel()
    if not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Raw gene counts must be finite, nonnegative, and nonempty")
    genes = normalized_gene_ids(gene_ids)
    aggregated = pd.Series(counts, index=genes).groupby(level=0, sort=False).sum()
    normalized_order = normalized_gene_ids(ordered_genes)
    if normalized_order.has_duplicates:
        raise ValueError("Ambiguous duplicate normalized genes in trained feature order")
    if not normalized_order.isin(aggregated.index).any():
        raise ValueError("No overlap between input gene identifiers and model genes")
    log_cpm = np.log1p(aggregated / aggregated.sum() * 1_000_000.0)
    # Absent genes are legitimate zeros under the original transfer contract.
    # Existing NA measurements are not converted to biological absence.
    aligned = log_cpm.reindex(normalized_order, fill_value=0.0)
    return dict(zip(ordered_genes, aligned.to_numpy()))


def read_canonical_pseudobulk(processed_path, ordered_genes):
    """Use canonical raw counts and the exact processed spot barcode set."""
    import anndata as ad
    processed_path = Path(processed_path)
    loaded_path = processed_path.with_name("01_loaded.h5ad")
    if not loaded_path.is_file():
        raise FileNotFoundError(f"Required raw-count handoff absent: {loaded_path}")
    loaded = ad.read_h5ad(loaded_path)
    processed = ad.read_h5ad(processed_path, backed="r")
    try:
        retained = pd.Index(processed.obs_names.astype(str))
        original = pd.Index(loaded.obs_names.astype(str))
        if retained.has_duplicates or original.has_duplicates:
            raise ValueError("Duplicate retained or original spot barcodes")
        if len(retained) == 0 or len(retained.difference(original)):
            raise ValueError("Processed spots do not match the canonical raw sample")
        # These fields connect reused SAMPLE_* names to external identities.
        for key in ("metadata_geo_accession", "metadata_dataset_id", "source_sample_dir"):
            if key in loaded.uns and key in processed.uns:
                if str(loaded.uns[key]) != str(processed.uns[key]):
                    raise ValueError(f"Raw/processed identity conflict for {key}")
        gene_ids = select_gene_ids(loaded.var, loaded.var_names)
        return raw_counts_to_pseudobulk(loaded.X[original.get_indexer(retained), :], gene_ids, ordered_genes)
    finally:
        processed.file.close()


def positive_class_probability(estimator, X):
    classes = list(estimator.classes_)
    if classes.count(1) != 1:
        raise ValueError(f"Artifact lacks unique responder class 1: {classes}")
    return np.asarray(estimator.predict_proba(X), dtype=float)[:, classes.index(1)]


def apply_saved_calibrator(raw, method, calibrator):
    """Match expression_model_v2_lib.apply_calibrator and training fit_calibrator."""
    p = np.clip(np.asarray(raw, dtype=float), 1e-6, 1.0 - 1e-6)
    if method == "identity":
        return p
    if calibrator is None:
        raise ValueError(f"Missing saved calibrator for method {method}")
    if method == "sigmoid":
        return positive_class_probability(calibrator, np.log(p / (1.0 - p)).reshape(-1, 1))
    if method == "isotonic":
        return np.asarray(calibrator.transform(p), dtype=float)
    raise ValueError(f"Unsupported recorded calibration method: {method}")


def score_expression_artifact(artifact, pseudobulk):
    """Return distinct raw/calibrated probabilities, preserving fitted preprocessing."""
    if not isinstance(artifact, dict) or "base_model" not in artifact:
        raise ValueError("Expected deployable expression artifact with base_model")
    order = list(artifact["gene_columns"])
    if len(order) != len(set(order)) or pseudobulk.columns.has_duplicates:
        raise ValueError("Duplicate expression feature names")
    missing = [g for g in order if g not in pseudobulk]
    X = pseudobulk.reindex(columns=order).apply(pd.to_numeric, errors="raise")
    if missing:
        X.loc[:, missing] = 0.0
    # Do not transform log values again or replace genuine NA: the saved pipeline
    # owns its imputer, variance filter, scaler, PCA, and logistic classifier.
    model = artifact["base_model"]
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != order:
        raise ValueError("Artifact gene order conflicts with fitted estimator")
    raw = positive_class_probability(model, X)
    calibrated = apply_saved_calibrator(raw, artifact.get("calibration_method", "identity"), artifact.get("calibrator"))
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise ValueError("Nonfinite expression artifact predictions")
    if ((calibrated < 0) | (calibrated > 1)).any():
        raise ValueError("Calibrated artifact output lies outside [0,1]")
    return raw, calibrated, {"missing_gene_count": len(missing), "model_gene_count": len(order)}
