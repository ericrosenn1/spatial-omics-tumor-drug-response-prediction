# Prediction interpretation model step order
Use 00_run_prediction_interpretation_model.py as the orchestrator.
1. 01_prepare_interpretation_inputs.py
2. 02_build_feature_and_treatment_dictionary.py
3. 03_compute_signed_spatial_effects.py
4. 04_build_treatment_interpretation_cards.py
5. 05_build_sample_level_interpretations.py
6. 06_build_mechanism_atlas.py
7. 07_make_final_outputs.py
8. 08_qc_and_package_final_outputs.py
Each step writes reports, summaries, manifests, and terminal output for auditability.
