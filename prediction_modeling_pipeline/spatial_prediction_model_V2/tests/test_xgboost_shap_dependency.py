"""Exercise the fitted SHAP interface in the pinned V2 numerical environment."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def test_xgboost_shap_is_finite_and_reconstructs_predictions(monkeypatch):
    """Installed SHAP must explain the actual model pipeline without fallback."""
    import shap

    monkeypatch.syspath_prepend(str(ROOT / "prediction_modeling_pipeline/spatial_prediction_model_V2/src"))
    from spm_v2.model_training import make_xgb_pipeline

    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(32, 4)), columns=["f0", "f1", "f2", "f3"])
    target = 2 * features["f0"] - features["f1"] + 0.1 * features["f2"]
    features.loc[[0, 5, 11], "f2"] = np.nan
    pipeline = make_xgb_pipeline(random_state=42, n_estimators=12)
    pipeline.fit(features, target)
    evaluation = features.iloc[:12]
    imputed = pd.DataFrame(pipeline.named_steps["imputer"].transform(evaluation), columns=features.columns)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    contributions = np.asarray(explainer.shap_values(imputed))

    assert contributions.shape == evaluation.shape
    assert np.isfinite(contributions).all()
    assert np.isfinite(explainer.expected_value).all()
    np.testing.assert_allclose(
        contributions.sum(axis=1) + explainer.expected_value,
        pipeline.predict(evaluation),
        rtol=1e-5,
        atol=1e-6,
    )
