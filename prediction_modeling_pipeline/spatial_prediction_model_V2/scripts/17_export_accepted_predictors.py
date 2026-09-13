"""Export the exact audited independent predictor route; optional deployment refit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spm_v2.accepted_predictor_export import export_accepted

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-all-data-deployment', action='store_true',
                   help='Optional separate all-cohort estimator/reference fit; retains fixed discovery-selected features')
    a = p.parse_args()
    print(json.dumps(export_accepted(a.evaluation_root, a.output, a.fit_all_data_deployment), indent=2))
