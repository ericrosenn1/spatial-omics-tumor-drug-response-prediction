D:\Adv_Omics_Fenyo\project\spatial_feature_identification_pipeline\docs\RUNBOOK.md

# Runbook

Dry run the orchestrator:
python run_pipeline.py --config configs/visium_cohort_clean.yaml --dry-run

Audit current outputs:
python tools/audit_pipeline_outputs.py --pipeline-root . --open

Run selected steps only when intentionally regenerating outputs:
python run_pipeline.py --config configs/visium_cohort_clean.yaml --start 06 --end 12
