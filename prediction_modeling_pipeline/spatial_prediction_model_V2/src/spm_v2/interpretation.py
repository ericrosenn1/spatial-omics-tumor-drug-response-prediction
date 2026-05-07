from __future__ import annotations

import pandas as pd

from .feature_governance import infer_theme


def attach_biology_theme(df: pd.DataFrame, feature_col: str = "feature_name") -> pd.DataFrame:
    out = df.copy()
    if "biological_theme" not in out.columns and feature_col in out.columns:
        out["biological_theme"] = out[feature_col].map(infer_theme)
    return out


def summarize_theme_contribution(df: pd.DataFrame, score_col: str = "mean_abs_shap") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = attach_biology_theme(df)
    if score_col not in out.columns:
        numeric_candidates = [c for c in out.columns if "shap" in str(c).lower() or "score" in str(c).lower()]
        if not numeric_candidates:
            return pd.DataFrame()
        score_col = numeric_candidates[0]
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return (
        out.groupby("biological_theme", dropna=False)
        .agg(
            n_features=("feature_name", "count") if "feature_name" in out.columns else (score_col, "count"),
            total_score=(score_col, "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
        )
        .reset_index()
        .sort_values("total_score", ascending=False)
    )
