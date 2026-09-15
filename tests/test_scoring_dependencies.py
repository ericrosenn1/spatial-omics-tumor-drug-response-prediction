"""Exercise installed scoring APIs on tiny synthetic matrices, without network."""
import importlib.util
from pathlib import Path
import sys

import anndata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "spatial_feature_identification_pipeline/code"
sys.path.insert(0, str(CODE))


def test_enabled_ucell_and_gsva_return_finite_scores():
    spec = importlib.util.spec_from_file_location("scoring_dependency_test", CODE / "05_build_multi_axis_transcriptome_labels.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng(42)
    genes = [f"GENE{i}" for i in range(64)]
    expression = pd.DataFrame(rng.gamma(2, 2, size=(24, 64)), columns=genes, index=[f"spot{i}" for i in range(24)])
    signatures = {"program_a": genes[:12], "program_b": genes[12:24]}
    source = anndata.AnnData(expression)
    ucell, status = module.compute_ucell_scores(source, signatures, True, 1)
    assert status == "ok" and ucell.shape == (24, 2)
    assert np.isfinite(ucell.to_numpy()).all()
    gsva, status = module.compute_gsva_scores(expression, signatures, True, 2, 5000)
    assert status == "ok" and gsva.shape == (24, 2)
    assert np.isfinite(gsva.to_numpy(dtype=float)).all()
