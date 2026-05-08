FILEPATH: D:\Adv_Omics_Fenyo\project\prediction_modeling_pipeline\spatial_transfer_inference_model\docs\SPATIAL_TRANSFER_INFERENCE_MODEL_README.txt

SPATIAL TRANSFER INFERENCE MODEL README

Purpose:
Take one or more spatial_feature_identification_pipeline feature outputs, align them to the frozen V2/PIM strict spatial biology registry, apply signed treatment-feature effects, and generate a single-slide drug response interpretation table.

Implemented scripts:
00_run_spatial_transfer_inference_model.py
01_prepare_transfer_inputs.py
02_align_single_slide_features_to_v2.py
03_score_transfer_drug_response_alignment.py
04_make_single_slide_prediction_table.py
05_qc_and_package_transfer_outputs.py

Important caveat:
This is a research-use transfer inference layer. It reports spatial response alignment, not a calibrated clinical efficacy probability and not a treatment recommendation.

Smoke test:
The smoke test uses a V2-compatible feature row from the completed prediction_interpretation_model run to validate the adapter/scorer/table/QC package. It does not run raw Visium feature extraction.