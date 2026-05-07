# teacher_builder repository layout

## Active files

- README.md
- run_teacher_builder_governed.ps1
- configs\visium_teacher_builder_governed_full102.yaml
- configs\visium_teacher_builder_governed_sample5.yaml
- scripts\teacher_governance_lib.py
- scripts\01_validate_teacher_inputs.py through scripts\06_qc_teacher_outputs.py

## Local runtime dependencies

- scripts\_backup_governed_20260505_072355: original histology scorer called by Step 03 wrapper.
- outputs\_histology_v2_compat: histology model compatibility artifacts.

## Local-only folders

- outputs: generated teacher outputs and model artifacts; do not commit large generated outputs by default.
- local_archive: old runners, deprecated configs, old outputs, and patch backups.
- docs\documentation_polish: documentation-polish provenance reports.
- docs\cleanup_manifests: cleanup dry-run/apply manifests.

