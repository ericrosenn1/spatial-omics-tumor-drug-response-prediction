"""Frozen-reference interpretation scoring; never a calibrated efficacy probability."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require_unique(frame, keys, label):
    if frame.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate columns")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}: missing or duplicate identity in {keys}")
    if any(frame[k].astype(str).str.strip().eq("").any() for k in keys):
        raise ValueError(f"{label}: blank identity")


def fit_reference(training, features):
    """All-cohort deployment reference. No performance estimation is done here."""
    require_unique(training, ["sample_id"], "training reference")
    features = list(features)
    if len(features) != len(set(features)):
        raise ValueError("Duplicate ordered features")
    missing = set(features) - set(training)
    if missing:
        raise ValueError(f"Training reference missing required features: {sorted(missing)}")
    rows = []
    for feature in features:
        values = pd.to_numeric(training[feature], errors="raise")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Nonfinite reference: {feature}")
        if not values.notna().any():
            raise ValueError(f"Entirely missing training reference feature: {feature}")
        rows.append(dict(feature_name=feature, mean=float(values.mean()),
                         std=float(values.std(ddof=0)), median=float(values.median()),
                         minimum=float(values.min()), maximum=float(values.max()),
                         missing_rate=float(values.isna().mean())))
    return {"schema_version": 1, "quantity": "signed_spatial_alignment",
            "reference_scope": "all_cohort_deployment_only",
            "std_ddof": 0, "missing_value_policy": "neutral_training_mean_z0_with_coverage",
            "features": rows, "training_sample_ids": sorted(training.sample_id.astype(str)),
            "score_definition": "sum(z * signed_effect) / sum(abs(signed_effect))",
            "rescaled_definition": "sigmoid(2 * signed_spatial_alignment); uncalibrated interpretation score"}


def transform(reference, table, allow_training_ids=False):
    require_unique(table, ["sample_id"], "transfer feature table")
    features = [r["feature_name"] for r in reference["features"]]
    missing_columns = set(features) - set(table)
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}; provide explicit NA for unavailable measurements")
    collisions = set(table.sample_id.astype(str)) & set(reference["training_sample_ids"])
    if collisions and not allow_training_ids:
        raise ValueError(f"Train/transfer identifier collision: {sorted(collisions)}")
    scaled = {"sample_id": table["sample_id"]}
    coverage = {"sample_id": table["sample_id"]}
    for row in reference["features"]:
        feature = row["feature_name"]
        values = pd.to_numeric(table[feature], errors="raise")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite transfer value: {feature}")
        coverage[feature] = values.notna()
        # This is mean imputation in standardized space, not biological-absence zero-fill.
        scaled[feature] = ((values - row["mean"]) / row["std"]).fillna(0.0) if row["std"] > 0 else 0.0
    return pd.DataFrame(scaled, index=table.index), pd.DataFrame(coverage, index=table.index)


def score(reference, table, effects, allow_training_ids=False):
    require_unique(effects, ["drug_key", "feature_name"], "signed effects")
    scaled, observed = transform(reference, table, allow_training_ids)
    missing_effect_features = set(effects.feature_name) - set(scaled)
    if missing_effect_features:
        raise ValueError(f"Effects outside frozen reference: {sorted(missing_effect_features)}")
    rows = []
    for key, sub in effects.groupby("drug_key", sort=True):
        features = sub.feature_name.tolist()
        weights = pd.to_numeric(sub.signed_effect, errors="raise").to_numpy(float)
        if not np.isfinite(weights).all():
            raise ValueError("Undefined signed effect")
        denominator = np.abs(weights).sum()
        for i in range(len(scaled)):
            observed_weight = float(observed.iloc[i][features].to_numpy(float) @ np.abs(weights))
            value = float(scaled.iloc[i][features].to_numpy(float) @ weights / denominator) if denominator > 0 and observed_weight > 0 else np.nan
            status = ("UNSUPPORTED_ZERO_EFFECT_WEIGHT" if denominator <= 0 else
                      "UNSUPPORTED_NO_OBSERVED_EFFECT_WEIGHT" if observed_weight <= 0 else
                      "SCORED_PARTIAL_FEATURE_COVERAGE" if observed_weight < denominator else "SCORED")
            rows.append(dict(sample_id=str(scaled.iloc[i].sample_id), drug_key=str(key),
                             signed_spatial_alignment=value,
                             rescaled_alignment_score_0_1=float(1/(1+np.exp(-np.clip(2*value,-700,700)))) if np.isfinite(value) else np.nan,
                             observed_weight_fraction=observed_weight/denominator if denominator > 0 else 0.0,
                             score_status=status))
    return pd.DataFrame(rows)


def save_reference(reference, path):
    path = Path(path)
    payload = json.dumps(reference, sort_keys=True, indent=2, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def load_reference(path):
    reference = json.loads(Path(path).read_text())
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported alignment reference schema")
    return reference
