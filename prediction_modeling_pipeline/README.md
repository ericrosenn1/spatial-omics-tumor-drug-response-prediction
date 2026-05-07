# Prediction Modeling Pipeline

Reusable prediction modeling system for Visium based drug response interpretation.

## Main structure

model_training/
    histology_response_model/
    expression_response_model/

teacher_builder/
    visium_teacher_builder/

spatial_prediction_model/

data_manifests/
    cptac_histology_training_data/
    gdc_expression_training_data/

outputs/
configs/
docs/
logs/

## Conceptual flow

histology_response_model + expression_response_model
? visium_teacher_builder
? spatial_prediction_model

## Notes

Paths should be controlled by YAML configs.
Old hard coded paths may need updating after migration.
