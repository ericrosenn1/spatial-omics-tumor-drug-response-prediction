# Model-training repository layout

Active modules:

- expression_response_model_v2/
- histology_response_model_v2/

Each module uses:

- configs/ for active YAML configuration
- scripts/ for active pipeline source and reusable library modules
- outputs/ for canonical local outputs
- docs/ for status and provenance reports

The scripts/ folder name is intentionally retained.

local_archive/ stores deprecated workflows, patch backups, installers, dry-run outputs, and local provenance.
