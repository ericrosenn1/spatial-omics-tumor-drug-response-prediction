from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_xgb_regressor(
    random_state: int = 42,
    n_estimators: int = 250,
    max_depth: int = 2,
    learning_rate: float = 0.03,
    subsample: float = 0.85,
    colsample_bytree: float = 0.80,
    tree_method: str = "hist",
    n_jobs: int = 1,
):
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        tree_method=tree_method,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def make_xgb_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", make_xgb_regressor(**kwargs)),
    ])


def select_features_training_only(
    x_train: pd.DataFrame,
    y_train,
    feature_cols: list[str],
    max_features: int = 40,
    min_variance: float = 1e-12,
) -> list[str]:
    values = x_train[feature_cols].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)

    variances = values.var(axis=0, ddof=1)
    candidates = variances[variances > min_variance].index.tolist()
    if not candidates:
        candidates = feature_cols

    y_series = pd.Series(np.asarray(y_train, dtype=float), index=values.index)

    if y_series.std(ddof=1) <= 0:
        return variances.sort_values(ascending=False).head(max_features).index.tolist()

    x_rank = values[candidates].rank(axis=0)
    y_rank = y_series.rank()
    corr = x_rank.corrwith(y_rank).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = pd.DataFrame({
        "feature_name": corr.index,
        "abs_spearman_train": corr.values,
        "variance": variances.loc[corr.index].values,
    }).sort_values(["abs_spearman_train", "variance"], ascending=[False, False])

    return score["feature_name"].head(max_features).tolist()
