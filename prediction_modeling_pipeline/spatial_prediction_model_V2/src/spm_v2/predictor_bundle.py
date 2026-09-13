"""Reloadable fitted spatial residual predictors, separate from alignment scores."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import csv
import hashlib
import json
import joblib
import numpy as np
import pandas as pd


def read_feature_table(path):
    """Reject duplicate file headers before pandas can auto-rename them."""
    path=Path(path)
    delimiter="\t" if path.suffix.lower()==".tsv" else ","
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        header=next(csv.reader(handle,delimiter=delimiter),None)
    if not header or len(header)!=len(set(header)):
        raise ValueError("Duplicate or missing raw feature-table header")
    return pd.read_csv(path,sep=delimiter,low_memory=False)


@dataclass
class SpatialPredictorBundle:
    drug_key: str
    ordered_features: list[str]
    pipeline: object
    treatment_prior: float | None
    training_sample_ids: list[str]
    feature_reference: object | None = None
    provenance: dict = field(default_factory=dict)
    support_status: str = "DEVELOPMENT_NOT_INDEPENDENTLY_VALIDATED"
    target_definition: str = "fused teacher probability minus its recorded treatment prior"
    version: str = "spatial-residual-predictor-v1"

    def feature_matrix(self, table, representation="saved_numeric", mode="external"):
        if representation not in {"saved_numeric", "raw_reference"}:
            raise ValueError("Unknown feature representation")
        if mode not in {"external", "training_reproduction", "heldout_evaluation_reproduction"}:
            raise ValueError("Unknown prediction mode")
        if table.columns.duplicated().any():
            raise ValueError("Duplicate input feature columns")
        if "sample_id" not in table or table.sample_id.isna().any() or table.sample_id.astype(str).str.strip().eq("").any() or table.sample_id.astype(str).duplicated().any():
            raise ValueError("Unique nonmissing sample_id required")
        reserved = self.provenance.get("reserved_cohort_sample_ids", self.training_sample_ids)
        collisions = sorted(set(table.sample_id.astype(str)) & set(reserved))
        if mode == "external" and collisions:
            raise ValueError(f"Train/transfer sample identifier collision: {collisions[:5]}; use verified external identities")
        if mode == "heldout_evaluation_reproduction":
            heldout = set(self.provenance.get("evaluation_holdout_sample_ids", []))
            if not heldout or not set(table.sample_id.astype(str)) <= heldout:
                raise ValueError("Input is not a recorded held-out evaluation sample set")
        work = table
        if representation == "raw_reference":
            if self.feature_reference is None:
                raise ValueError("Bundle has no verified fitted raw feature reference")
            work = self.feature_reference.transform(table)
        missing = [c for c in self.ordered_features if c not in work]
        if missing:
            raise ValueError(f"Missing required spatial feature columns: {missing}")
        # Present NA is handled only by the fitted training imputer. A missing
        # column is a schema failure, never biological-absence zero fill.
        X = work[self.ordered_features].apply(pd.to_numeric, errors="raise")
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isna().all(axis=1).any():
            raise ValueError("All required features are missing for an input sample")
        return X

    def predict(self, table, representation="saved_numeric", mode="external"):
        X = self.feature_matrix(table, representation, mode)
        prediction = np.asarray(self.pipeline.predict(X), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite fitted-model prediction")
        prior = np.nan if self.treatment_prior is None else self.treatment_prior
        reconstructed = prediction + prior
        return pd.DataFrame({
            "sample_id":table.sample_id.astype(str).to_numpy(),
            "drug_key":self.drug_key,
            "predicted_residual_vs_prior":prediction,
            "recorded_treatment_prior":prior,
            "predicted_fused_teacher_response_unclipped":reconstructed,
            "predicted_fused_teacher_response_display_0_1":np.clip(reconstructed, 0, 1),
            "response_reconstruction_defined":self.treatment_prior is not None,
            "n_required_features":len(self.ordered_features),
            "n_observed_features":X.notna().sum(axis=1).to_numpy(),
            "n_training_median_imputed_features":X.isna().sum(axis=1).to_numpy(),
            "feature_coverage_fraction":X.notna().mean(axis=1).to_numpy(),
            "support_status":self.support_status,
            "prediction_partition":("EXTERNAL_INFERENCE" if mode=="external" else
                "HELDOUT_EVALUATION_REPRODUCTION" if mode=="heldout_evaluation_reproduction" else
                self.provenance.get("training_prediction_partition", "ALL_DATA_FITTED_TRAINING_REPRODUCTION")),
            "quantity_definition":"Fitted spatial model prediction of teacher residual; reconstructed teacher score is not calibrated drug efficacy",
        })

    def contributions(self, table, representation="saved_numeric", mode="external"):
        import xgboost as xgb
        X=self.feature_matrix(table,representation,mode)
        values=self.pipeline.named_steps["imputer"].transform(X)
        contrib=self.pipeline.named_steps["model"].get_booster().predict(xgb.DMatrix(values), pred_contribs=True)
        predicted=self.pipeline.predict(X)
        if not np.allclose(contrib.sum(axis=1),predicted,rtol=1e-5,atol=1e-7):
            raise ValueError("Feature contribution sum does not reproduce fitted prediction")
        frame=pd.DataFrame(contrib,columns=self.ordered_features+["MODEL_BIAS"])
        frame.insert(0,"sample_id",table.sample_id.astype(str).to_numpy())
        result=frame.melt(id_vars="sample_id",var_name="feature_name",value_name="residual_contribution")
        result.insert(1,"drug_key",self.drug_key)
        return result


def save_bundle(bundle, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    joblib.dump(bundle,temporary,compress=3)
    temporary.replace(path)


def load_bundle(path):
    bundle=joblib.load(path)
    if not isinstance(bundle,SpatialPredictorBundle):
        raise ValueError("Not a spatial predictor bundle")
    return bundle


def load_bundle_directory(path):
    path=Path(path)
    manifest=pd.read_csv(path/"bundle_manifest.tsv",sep="\t")
    if manifest.drug_key.duplicated().any():
        raise ValueError("Duplicate treatment keys in bundle manifest")
    bundles=[]
    for r in manifest.itertuples():
        p=path/r.bundle_file
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r.sha256:
            raise ValueError(f"Bundle hash mismatch: {p}")
        b=load_bundle(p)
        if b.drug_key!=r.drug_key:
            raise ValueError("Manifest/bundle treatment identity conflict")
        bundles.append(b)
    return bundles
