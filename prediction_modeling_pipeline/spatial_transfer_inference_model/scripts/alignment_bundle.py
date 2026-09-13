"""Export and apply a self-contained signed spatial interpretation atlas.

The resulting scores are associations, with no calibrated response probability.
Use V2 predictor bundles for fitted residual predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from _alignment_contract import fit_reference, load_reference, save_reference, score, transform
from _stim_utils import (load_pim_feature_dictionary, load_pim_signed_feature_effects,
                         load_pim_spatial_feature_pool, load_pim_treatment_cards, read_table)


def json_records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def export_atlas(pim_run_root, output_path, feature_reference_bundles=None):
    root = Path(pim_run_root).resolve()
    dictionary = load_pim_feature_dictionary(root)
    features = dictionary.feature_name.astype(str).tolist()
    training = load_pim_spatial_feature_pool(root, usecols=["sample_id"] + features)
    effects = load_pim_signed_feature_effects(root)
    cards = load_pim_treatment_cards(root)
    if set(effects.drug_key) != set(cards.drug_key):
        raise ValueError("Atlas cards and signed-effect treatment sets differ")
    reference = fit_reference(training, features)
    reference["effects"] = json_records(effects)
    reference["treatment_cards"] = json_records(cards)
    scopes = cards.get("source_estimator_validation_scope", pd.Series(["UNSPECIFIED_SOURCE_SCOPE"])).dropna().unique().tolist()
    if len(scopes) != 1:
        raise ValueError("Atlas cards have ambiguous source-estimator validation scopes")
    reference["source_estimator_validation_scope"] = scopes[0]
    reference["feature_dictionary"] = json_records(dictionary)
    reference["pim_run_root"] = str(root)
    reference["predictive_performance_status"] = "NOT_EVALUATED_FOR_THIS_ALIGNMENT_RULE"
    source_hashes = {}
    for rel in ["02_feature_and_treatment_dictionary/01_feature_dictionary/strict_spatial_feature_dictionary.tsv",
                "03_signed_spatial_effects/01_treatment_feature_effects/signed_treatment_feature_effects.tsv",
                "04_treatment_interpretation_cards/02_cards_tsv/treatment_interpretation_cards.tsv"]:
        path = root / rel
        source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference["source_sha256"] = source_hashes
    if feature_reference_bundles:
        predictor_reference = load_predictor_reference(feature_reference_bundles)
        if set(predictor_reference.training_ids) != set(reference["training_sample_ids"]):
            raise ValueError("Raw feature reference and atlas training identities differ")
        manifest = Path(feature_reference_bundles) / "bundle_manifest.tsv"
        reference["upstream_raw_feature_reference"] = {
            "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "training_sample_ids": sorted(predictor_reference.training_ids),
            "definition": "Saved spatial feature normalization only; no expression or histology model refit",
        }
    save_reference(reference, output_path)
    return reference


def load_predictor_reference(bundles_path):
    v2_src = Path(__file__).resolve().parents[2] / "spatial_prediction_model_V2/src"
    sys.path.insert(0, str(v2_src))
    from spm_v2.predictor_bundle import load_bundle_directory
    bundles = load_bundle_directory(bundles_path)
    if not bundles or bundles[0].feature_reference is None:
        raise ValueError("A saved fitted spatial feature reference is required")
    return bundles[0].feature_reference


def prepare_raw_features(bundle_path, bundles_path, raw_path, output_path, unavailable_manifest=None):
    """Apply frozen normalization to a source-bound raw spatial feature table."""
    reference = load_reference(bundle_path)
    binding = reference.get("upstream_raw_feature_reference")
    manifest_path = Path(bundles_path) / "bundle_manifest.tsv"
    if not binding or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != binding["bundle_manifest_sha256"]:
        raise ValueError("Atlas does not bind this exact saved raw feature reference")
    raw = read_table(raw_path)
    from _alignment_contract import require_unique
    require_unique(raw, ["sample_id"], "raw spatial feature input")
    if set(raw.sample_id.astype(str)) & set(reference["training_sample_ids"]):
        raise ValueError("Train/transfer identifier collision in raw spatial feature input")
    raw_sha = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    reviewed = pd.DataFrame()
    if unavailable_manifest:
        reviewed = read_table(unavailable_manifest, keep_default_na=False)
        required = {"feature_name", "reason", "source_evidence", "raw_input_sha256", "adapter_value"}
        if not required.issubset(reviewed.columns):
            raise ValueError("Unavailable-feature manifest lacks source-bound evidence columns")
        require_unique(reviewed, ["feature_name"], "unavailable-feature manifest")
        if not reviewed.raw_input_sha256.eq(raw_sha).all() or not reviewed.adapter_value.eq("NA").all():
            raise ValueError("Unavailable-feature manifest must match the exact input hash and explicit NA policy")
        if reviewed[["reason", "source_evidence"]].isna().any().any() or reviewed[["reason", "source_evidence"]].apply(lambda col: col.astype(str).str.strip().eq("")).any().any():
            raise ValueError("Unavailable-feature evidence must be nonempty")
    added = [f for f in reviewed.get("feature_name", []) if f not in raw]
    work = pd.concat([raw, pd.DataFrame(np.nan, index=raw.index, columns=added)], axis=1)
    fitted = load_predictor_reference(bundles_path)
    normalized = fitted.transform(work)
    features = [r["feature_name"] for r in reference["features"]]
    missing = set(features) - set(normalized)
    if missing:
        raise ValueError(f"Unreviewed missing atlas input columns: {sorted(missing)}")
    result = normalized[["sample_id"] + features]
    # This gate also validates numeric types, duplicates and train/transfer collisions.
    transform(reference, result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    output_path.with_suffix(".manifest.json").write_text(json.dumps({
        "raw_input": str(Path(raw_path).resolve()), "raw_input_sha256": raw_sha,
        "atlas_sha256": hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
        "fitted_reference_bundle_manifest_sha256": binding["bundle_manifest_sha256"],
        "unavailable_manifest_sha256": hashlib.sha256(Path(unavailable_manifest).read_bytes()).hexdigest() if unavailable_manifest else None,
        "added_explicit_na_columns": added, "zero_filled_columns": [],
        "samples": result.sample_id.astype(str).tolist(), "features": features,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "normalization_fit_on_new_samples": False, "expression_or_histology_retraining": False,
    }, indent=2))
    return result


def apply_atlas(bundle_path, table, output_root):
    reference = load_reference(bundle_path)
    effects = pd.DataFrame(reference["effects"])
    scores = score(reference, table, effects)
    scores["score_definition"] = reference["score_definition"]
    scores["model_prediction_status"] = "SEPARATE_V2_PREDICTOR_REQUIRED"
    scores["source_estimator_validation_scope"] = reference.get("source_estimator_validation_scope", "UNSPECIFIED_SOURCE_SCOPE")
    scores["treatment_label"] = scores["drug_key"]
    scores["treatment_identity_definition"] = "case-level recorded-treatment profile; component timing and simultaneous administration are not established"
    scaled, observed = transform(reference, table)
    long_rows = []
    for key, sub in effects.groupby("drug_key", sort=True):
        denominator = float(sub.signed_effect.abs().sum())
        for _, row in scaled.iterrows():
            index = row.name
            for effect in sub.to_dict("records"):
                feature = effect["feature_name"]
                contribution = float(row[feature] * effect["signed_effect"])
                long_rows.append({"sample_id": row.sample_id, **effect,
                                  "feature_observed": bool(observed.loc[index, feature]),
                                  "training_reference_z": float(row[feature]),
                                  "alignment_contribution": contribution / denominator if denominator else np.nan})
    contributions = pd.DataFrame(long_rows)
    expected_rows = table.sample_id.nunique() * len(reference["treatment_cards"])
    if len(scores) != expected_rows or scores.duplicated(["sample_id", "drug_key"]).any():
        raise ValueError("Sample/treatment score key set does not match bundle")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_root / "sample_treatment_alignment.tsv", sep="\t", index=False)
    contributions.to_csv(output_root / "feature_contributions.tsv", sep="\t", index=False)
    if "biological_theme" in contributions:
        contributions.groupby(["sample_id", "drug_key", "biological_theme"], dropna=False).alignment_contribution.sum(min_count=1).reset_index().to_csv(output_root / "theme_contributions.tsv", sep="\t", index=False)
    (output_root / "alignment_inference_manifest.json").write_text(json.dumps({
        "bundle_sha256": hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
        "samples": table.sample_id.astype(str).tolist(), "drug_keys": sorted(scores.drug_key.unique()),
        "expected_rows": expected_rows, "actual_rows": len(scores),
        "quantity": "signed spatial alignment and sigmoid-rescaled interpretation score",
        "missing_value_policy": reference["missing_value_policy"],
        "performance_status": reference["predictive_performance_status"],
    }, indent=2))
    return scores, contributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--pim-run-root", required=True)
    export.add_argument("--bundle", required=True)
    export.add_argument("--feature-reference-bundles")
    prepare = sub.add_parser("prepare-raw", help="Apply the saved spatial feature reference without cohort retraining")
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--feature-reference-bundles", required=True)
    prepare.add_argument("--raw-feature-table", required=True)
    prepare.add_argument("--unavailable-feature-manifest")
    prepare.add_argument("--output-feature-table", required=True)
    apply = sub.add_parser("score")
    apply.add_argument("--bundle", required=True)
    apply.add_argument("--feature-table", required=True)
    apply.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "export":
        reference = export_atlas(args.pim_run_root, args.bundle, args.feature_reference_bundles)
        print(f"Exported {len(reference['treatment_cards'])} treatment profiles and {len(reference['features'])} reference features to {args.bundle}")
    elif args.command == "prepare-raw":
        result = prepare_raw_features(args.bundle, args.feature_reference_bundles, args.raw_feature_table,
                                      args.output_feature_table, args.unavailable_feature_manifest)
        print(f"Applied saved feature reference to {len(result)} samples; output: {args.output_feature_table}")
    else:
        path = Path(args.feature_table)
        table = read_table(path)
        scores, _ = apply_atlas(args.bundle, table, args.output_root)
        print(f"Scored {len(scores)} sample-treatment alignments; output: {args.output_root}")


if __name__ == "__main__":
    main()
