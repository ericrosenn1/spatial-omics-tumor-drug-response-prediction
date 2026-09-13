import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spm_v2.conditional_reporting import complete_reporting


def test_distinct_theme_counts_and_idempotent_reporting(tmp_path):
    def js(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    def tab(name, rows):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    js("10_independent_numerical_audit/audit_summary.json", {"status": "PASS", "n_conditional_acceptances": 2})
    js("run_manifest.json", {"sources": {"registry": {"path": str(tmp_path / "registry.tsv")}}})
    js("v2_step09_tier1_label_shuffle_validation_summary.json", {"n_label_shuffle_validated_treatments": 2})
    tab("registry.tsv", [{"feature_name": "f1", "biological_theme": "immune"}, {"feature_name": "f2", "biological_theme": "immune"}])
    tab("04_validation_results/tier1_label_shuffle_validation_results.tsv", [
        {"drug_key": key, "validated_for_step10": True, "fdr_q_pearson": .01, "empirical_p_pearson": .001, "observed_test_pearson_mean": .8}
        for key in ["agent a", "agent b"]])
    tab("02_observed_models/tier1_observed_feature_evidence_long.tsv", [
        {"drug_key": "agent a", "feature_name": "f1"}, {"drug_key": "agent a", "feature_name": "f2"},
        {"drug_key": "agent b", "feature_name": "f1"}])
    tab("05_validated_features_and_themes/label_shuffle_validated_recurrent_biology_themes.tsv", [
        {"biological_theme": "immune", "validated_treatment_count": 3, "total_gain_importance": .7}])
    js("controller_logs/commands.jsonl", {"event": "HEARTBEAT", "workers": 2})
    first = complete_reporting(tmp_path)
    second = complete_reporting(tmp_path)
    assert first == second
    themes = pd.read_csv(tmp_path / "05_validated_features_and_themes/label_shuffle_validated_recurrent_biology_themes.tsv", sep="\t")
    assert themes.validated_treatment_count.tolist() == [2]
    assert themes.validated_treatment_feature_incidence_count.tolist() == [3]
    assert not first["acceptance_or_metrics_changed"]
