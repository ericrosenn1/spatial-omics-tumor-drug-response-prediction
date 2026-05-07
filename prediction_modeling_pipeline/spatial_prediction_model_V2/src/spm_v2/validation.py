from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_safe(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    out = {
        "n": int(len(y_true)),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }

    if len(y_true) == 0:
        return out

    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)

    if len(y_true) >= 3 and np.nanstd(y_true) > 0:
        try:
            out["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            pass

    if len(y_true) >= 3 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            try:
                out["pearson"] = float(pearsonr(y_true, y_pred)[0])
            except Exception:
                pass
            try:
                out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
            except Exception:
                pass

    return out


def summarize_repeated_split_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()

    rows = []

    for key, sub in metrics.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row["n_repeats"] = int(len(sub))

        for col in [
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_mae",
            "test_rmse",
            "baseline_test_rmse",
            "rmse_improvement_vs_baseline",
            "mae_improvement_vs_baseline",
        ]:
            if col not in sub.columns:
                continue

            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if len(vals.dropna()) else np.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals.dropna()) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else np.nan
            row[f"{col}_q025"] = float(vals.quantile(0.025)) if len(vals.dropna()) else np.nan
            row[f"{col}_q975"] = float(vals.quantile(0.975)) if len(vals.dropna()) else np.nan

        if "test_pearson" in sub.columns:
            vals = pd.to_numeric(sub["test_pearson"], errors="coerce")
            row["test_pearson_positive_fraction"] = float((vals > 0).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def bh_fdr(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)

    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        value = min(prev, p[idx] * n / rank)
        q[idx] = value
        prev = value

    return q
