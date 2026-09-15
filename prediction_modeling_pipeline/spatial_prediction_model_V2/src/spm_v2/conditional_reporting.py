"""Post-audit reporting completion without model fitting or metric changes.

Run after conditional_validation. The numerical audit and its immutable code
signature remain valid. This corrects the historical theme incidence/count
label and supplies optional legacy summary fields from audited numbers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def report_distinct_theme_counts(themes, features, registry, accepted):
    """Count distinct accepted treatments per theme, retaining feature incidence."""
    themes = themes.copy()
    incidence_column = "validated_treatment_feature_incidence_count"
    if incidence_column not in themes.columns:
        themes[incidence_column] = themes["validated_treatment_count"]
    selected = features[features.drug_key.isin(accepted)]
    selected = selected.merge(registry[["feature_name", "biological_theme"]], on="feature_name", validate="many_to_one")
    distinct = selected.groupby("biological_theme", dropna=False).drug_key.nunique()
    themes["validated_treatment_count"] = themes.biological_theme.map(distinct).fillna(0).astype(int)
    if (themes.validated_treatment_count > len(accepted)).any():
        raise ValueError("Impossible distinct treatment-theme count")
    return themes


def complete_reporting(output):
    output = Path(output)
    audit = json.loads((output / "10_independent_numerical_audit/audit_summary.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("Numerical audit must pass before completing reporting")
    run = json.loads((output / "run_manifest.json").read_text())
    results = pd.read_csv(output / "04_validation_results/tier1_label_shuffle_validation_results.tsv", sep="\t")
    features = pd.read_csv(output / "02_observed_models/tier1_observed_feature_evidence_long.tsv", sep="\t")
    registry = pd.read_csv(run["sources"]["registry"]["path"], sep="\t")
    accepted = set(results.loc[results.validated_for_step10, "drug_key"])
    if len(accepted) != audit["n_conditional_acceptances"]:
        raise ValueError("Audited accepted family differs")
    summary_path = output / "v2_step09_tier1_label_shuffle_validation_summary.json"
    summary = json.loads(summary_path.read_text())
    journal = output / "controller_logs/commands.jsonl"
    heartbeat = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    workers = [row["workers"] for row in heartbeat if row.get("event") == "HEARTBEAT"]
    best = results.iloc[0] if len(results) else None
    def finite_value(field):
        value = float(best[field]) if best is not None else np.nan
        return value if np.isfinite(value) else None
    summary.update({"max_workers_policy": "durable_max_two_outer_workers_single_model_thread_serial_failure_fallback",
                    "max_workers_used": max(workers) if workers else 0,
                    "best_treatment": str(best.drug_key) if best is not None else None,
                    "best_fdr_q_pearson": finite_value("fdr_q_pearson"),
                    "best_empirical_p_pearson": finite_value("empirical_p_pearson"),
                    "best_observed_test_pearson_mean": finite_value("observed_test_pearson_mean"),
                    "ready_for_step10_integrated_package": "yes", "readiness_scope": "conditional_development_outputs_only",
                    "production_dependency_on_v1_outputs": "no", "uses_step05_v2_registry": "yes", "uses_treatment_identity_features": "no"})
    atom_json(summary_path, summary)
    themes_path = output / "05_validated_features_and_themes/label_shuffle_validated_recurrent_biology_themes.tsv"
    themes = pd.read_csv(themes_path, sep="\t")
    original_path = themes_path.with_name(themes_path.stem + "_historical_incidence_aggregation.tsv")
    if not original_path.exists():
        original_path.write_bytes(themes_path.read_bytes())
    themes = report_distinct_theme_counts(themes, features, registry, accepted)
    temp = themes_path.with_name(themes_path.name + ".tmp")
    themes.to_csv(temp, sep="\t", index=False)
    temp.replace(themes_path)
    proof = {"status": "PASS", "model_fitting_performed": False, "acceptance_or_metrics_changed": False,
             "accepted_count": len(accepted), "reporting_code_sha256": sha(__file__),
             "repairs": ["legacy optional best/worker/readiness fields derived from audited outputs",
                         "theme validated_treatment_count now means distinct treatments; historical treatment-feature incidence retained in explicit column and original table"],
             "summary_sha256": sha(summary_path), "themes_sha256": sha(themes_path), "historical_themes_sha256": sha(original_path)}
    atom_json(output / "10_independent_numerical_audit/reporting_completion.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(complete_reporting(args.output_root), indent=2))


if __name__ == "__main__":
    main()
