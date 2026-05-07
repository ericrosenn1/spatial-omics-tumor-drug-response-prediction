# teacher_builder runbook

## Purpose

Operational notes for running the governed teacher_builder workflow.

## Standard full run

    cd "D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\teacher_builder"
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_full102.yaml -StartAt 1 -StopAt 6

## Smoke test

    .\run_teacher_builder_governed.ps1 -Config .\configs\visium_teacher_builder_governed_sample5.yaml -StartAt 1 -StopAt 6

## Main handoff files

- outputs\04_fused_teacher\fused_teacher_table.tsv
- outputs\05_prediction_ready_teacher\model_input_numeric.csv
- outputs\05_prediction_ready_teacher\visium_fused_teacher_table.tsv
- outputs\05_prediction_ready_teacher\prediction_ready_training_table.tsv
- outputs\06_teacher_qc\teacher_qc_decision.txt

## Notes

Do not move scripts\_backup_governed_20260505_072355 unless Step 03 is refactored, because the governed histology wrapper depends on that archived original scorer.

