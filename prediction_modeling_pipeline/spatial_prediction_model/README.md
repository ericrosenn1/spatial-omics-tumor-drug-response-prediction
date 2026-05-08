# spatial_prediction_model

This folder is the modeling stage after `teacher_builder`.

Input handoff comes from:

```text
prediction_modeling_pipeline/teacher_builder/outputs/05_prediction_ready_teacher/
    model_input_numeric.csv
    visium_fused_teacher_table.tsv
    feature_manifest.csv
    prediction_ready_training_table.tsv
```

Initial run mode is controlled by:

```text
configs/spatial_prediction_model.yaml
```

The scaffold starts in 10-sample mode:

```yaml
run_name: "output_run_10"
test_mode: true
test_n_samples: 10
limit_training_to_test_samples: true
prediction_sample_mode: "test_labeled_samples"
```

For the later full run, edit YAML only:

```yaml
run_name: "output_run_102"
output_root: ".../spatial_prediction_model/outputs/output_run_102"
run_scope: "full_102"
test_mode: false
limit_training_to_test_samples: false
prediction_sample_mode: "all_spatial_samples"
run_per_treatment_models: true
```

Planned steps:

```text
01_validate_prediction_inputs.py
02_build_spatial_modeling_dataset.py
03_train_global_spatial_response_model.py
04_train_per_treatment_models.py
05_explain_spatial_response_model.py
06_predict_all_sample_treatment_pairs.py
07_qc_spatial_prediction_outputs.py
```

The Python files are scaffold stubs. They write step contracts now and should be filled in one by one.