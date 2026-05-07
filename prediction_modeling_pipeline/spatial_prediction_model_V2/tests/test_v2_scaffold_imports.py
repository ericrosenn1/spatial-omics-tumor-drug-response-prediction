from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import spm_v2
from spm_v2.io_utils import ensure_dir
from spm_v2.feature_governance import classify_feature, infer_theme
from spm_v2.validation import metric_safe


def test_imports():
    assert spm_v2.__version__
    assert classify_feature("hotspot__myeloid_macrophage_dist_to_tumor_mean")[0] == "include_biology"
    assert infer_theme("tryptophan kynurenine") == "tryptophan kynurenine immune suppression"
    assert metric_safe([1, 2, 3], [1, 2, 3])["pearson"] > 0.99
