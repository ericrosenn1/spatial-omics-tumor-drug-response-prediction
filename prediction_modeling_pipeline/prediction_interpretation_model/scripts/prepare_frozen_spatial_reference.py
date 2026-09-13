"""Export a manifest-bound interpretation reference using saved V2 preprocessing."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

V2_SRC = Path(__file__).resolve().parents[2] / "spatial_prediction_model_V2/src"
sys.path.insert(0, str(V2_SRC))
from spm_v2.predictor_bundle import load_bundle_directory, read_feature_table


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(bundles_path, raw_training_path, original_spatial_path, raw_external_path, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    training_output = output_root / "pim_frozen_training_spatial_reference.tsv"
    external_output = output_root / "pim_frozen_external_spatial_features.tsv"
    if training_output.exists() or external_output.exists():
        raise ValueError("Refusing to overwrite a previously prepared spatial reference")
    bundles = load_bundle_directory(bundles_path)
    reference = bundles[0].feature_reference
    if reference is None:
        raise ValueError("Saved predictor has no fitted feature reference")
    original = read_feature_table(original_spatial_path)
    raw = read_feature_table(raw_training_path)
    external = read_feature_table(raw_external_path)
    for frame, label in [(original, "original"), (raw, "raw training"), (external, "raw external")]:
        if "sample_id" not in frame or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Invalid {label} sample identity")
    training_ids = set(reference.training_ids)
    if set(original.sample_id) != training_ids or set(raw.sample_id) != training_ids:
        raise ValueError("Saved reference, original spatial table and raw cohort identity sets differ")
    if set(external.sample_id) & training_ids:
        raise ValueError("Training/transfer identifier collision")
    work = reference.transform(raw)
    missing = set(original.columns) - set(work.columns)
    if missing:
        raise ValueError(f"Fitted reference cannot reproduce original spatial schema: {sorted(missing)}")
    work = work.set_index("sample_id").loc[original.sample_id].reset_index()[original.columns]
    external_work = reference.transform(external)
    missing = set(original.columns) - set(external_work.columns)
    if any(not c.startswith(("motif_", "pair_")) for c in missing):
        raise ValueError(f"External raw adapter lacks required original spatial schema: {sorted(missing)}")
    unexported_columns = sorted(missing)
    # Step09 emits only motifs and pairs present in its per-slide tables. An
    # absent output column is unmeasured in this handoff, never a measured zero.
    for column in unexported_columns:
        external_work[column] = np.nan
    external_work = external_work[original.columns]
    comparisons = []
    for c in original.columns:
        if c == "sample_id":
            continue
        a = pd.to_numeric(original[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(work[c], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        comparisons.append({"feature_name": c, "na_changes": int(np.sum(np.isnan(a) != np.isnan(b))),
                            "old_observed_new_missing": int(np.sum(np.isfinite(a) & np.isnan(b))),
                            "max_abs_continuous_delta": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None})
    comparison = pd.DataFrame(comparisons)
    singles = pd.concat([reference.transform(external.iloc[[i]]).reindex(columns=original.columns) for i in range(len(external))], ignore_index=True)
    pd.testing.assert_frame_equal(external_work.reset_index(drop=True), singles, check_exact=False, rtol=1e-13, atol=1e-14)
    reordered = reference.transform(external.iloc[::-1, ::-1]).reindex(columns=original.columns).sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(external_work.sort_values("sample_id").reset_index(drop=True), reordered, check_exact=False, rtol=1e-13, atol=1e-14)
    delta = external_work.drop(columns="sample_id").apply(pd.to_numeric, errors="coerce").to_numpy(float) - singles.drop(columns="sample_id").apply(pd.to_numeric, errors="coerce").to_numpy(float)
    work.to_csv(training_output, sep="\t", index=False)
    external_work.to_csv(external_output, sep="\t", index=False)
    comparison.to_csv(output_root / "spatial_reference_comparison.tsv", sep="\t", index=False)
    manifest = {"status": "PASS", "schema": "full original V2 broad spatial feature columns in original order",
                "training_samples": len(work), "external_samples": len(external_work), "columns": len(work.columns),
                "reference_training_ids": reference.training_ids, "external_sample_ids": external.sample_id.tolist(),
                "na_changes": int(comparison.na_changes.sum()),
                "max_continuous_delta": float(comparison.max_abs_continuous_delta.max()),
                "missingness_change_features": comparison.loc[comparison.na_changes.gt(0)].to_dict("records"),
                "training_override": str(training_output.resolve()), "training_override_sha256": digest(training_output),
                "external_features": str(external_output.resolve()), "external_features_sha256": digest(external_output),
                "inputs": {"bundle_manifest": {"path": str(Path(bundles_path).resolve() / "bundle_manifest.tsv"), "sha256": digest(Path(bundles_path) / "bundle_manifest.tsv")},
                           "raw_training": {"path": str(Path(raw_training_path).resolve()), "sha256": digest(raw_training_path)},
                           "original_spatial": {"path": str(Path(original_spatial_path).resolve()), "sha256": digest(original_spatial_path)},
                           "raw_external": {"path": str(Path(raw_external_path).resolve()), "sha256": digest(raw_external_path)}},
                "same_saved_reference_used_for_training_and_external": True,
                "external_unexported_schema_columns_explicit_na": unexported_columns,
                "unexported_schema_policy": "Step09 motif/pair columns omitted by per-slide motif presence and pair availability; schema-padded NA only, not biological-absence zero. Source: spatial_feature_identification_pipeline/code/09_build_motif_tables.py build_motif_table/build_slide_summary.",
                "external_single_batch_and_reordering_equivalent": True,
                "numeric_equivalence_tolerance": {"rtol": 1e-13, "atol": 1e-14},
                "single_batch_max_abs_numeric_delta": float(np.nanmax(np.abs(delta))),
                "new_batch_reference_fit": False,
                "scope": "Development PIM signs and z reference must be recomputed from this derivative; estimator training/evaluation remain separate"}
    (output_root / "frozen_spatial_reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "training_samples", "external_samples", "columns", "na_changes", "max_continuous_delta", "training_override_sha256"]}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--raw-training", required=True)
    parser.add_argument("--original-spatial", required=True)
    parser.add_argument("--raw-external", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    prepare(args.bundles, args.raw_training, args.original_spatial, args.raw_external, args.output_root)


if __name__ == "__main__":
    main()
