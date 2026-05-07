# Spatial Prediction Model V2

Production residual spatial biology interpretation pipeline inside prediction_modeling_pipeline.

The pipeline starts from a teacher-builder handoff and tests whether spatial biology features explain treatment response residuals beyond treatment priors.

Pipeline steps 00 through 12 run validation, dataset construction, residual modeling, treatment-specific modeling, label-shuffle validation, publication packaging, and final QC.

Smoke run:
python .\scripts\00_run_spatial_prediction_model_v2.py --mode smoke --handoff-root <PATH_TO_TEACHER_BUILDER_HANDOFF> --max-workers 0 --open-output

Full run:
python .\scripts\00_run_spatial_prediction_model_v2.py --mode full --handoff-root <PATH_TO_TEACHER_BUILDER_HANDOFF> --max-workers 0 --full-step09-n-shuffles 100 --full-step09-n-repeats 5 --open-output

Generated outputs, logs, local data, handoffs, result figures, workbooks, ZIP packages, and backups should not be committed.
