from __future__ import annotations

import numpy as np
import pandas as pd


def require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")


def build_broad_residual_targets(teacher: pd.DataFrame, residual_col: str = "fused_residual_vs_prior") -> pd.DataFrame:
    require_column(teacher, "sample_id")
    require_column(teacher, residual_col)

    df = teacher.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df[residual_col] = pd.to_numeric(df[residual_col], errors="coerce")
    df = df.dropna(subset=[residual_col])

    rows = []
    for sample_id, sub in df.groupby("sample_id"):
        values = sub[residual_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        values_sorted = np.sort(values)
        top_n = min(5, len(values))
        top10_n = min(10, len(values))
        rows.append({
            "sample_id": sample_id,
            "n_treatments": int(len(values)),
            "mean_residual": float(np.mean(values)),
            "median_residual": float(np.median(values)),
            "residual_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "residual_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "positive_residual_fraction": float(np.mean(values > 0)),
            "strong_positive_fraction_005": float(np.mean(values > 0.05)),
            "strong_negative_fraction_m005": float(np.mean(values < -0.05)),
            "top5_mean_residual": float(np.mean(values_sorted[-top_n:])),
            "bottom5_mean_residual": float(np.mean(values_sorted[:top_n])),
            "top10_mean_residual": float(np.mean(values_sorted[-top10_n:])),
            "bottom10_mean_residual": float(np.mean(values_sorted[:top10_n])),
            "broad_resistance_score": float(-np.mean(values)),
        })

    return pd.DataFrame(rows)


def treatment_eligibility(
    teacher: pd.DataFrame,
    residual_col: str = "fused_residual_vs_prior",
    min_samples: int = 60,
    min_target_std: float = 0.02,
    min_target_range: float = 0.08,
    min_unique_targets: int = 10,
) -> pd.DataFrame:
    require_column(teacher, "sample_id")
    require_column(teacher, "drug_key")
    require_column(teacher, residual_col)

    rows = []
    for drug_key, sub in teacher.groupby("drug_key"):
        vals = pd.to_numeric(sub[residual_col], errors="coerce").dropna()
        n_samples = sub.loc[vals.index, "sample_id"].astype(str).nunique() if len(vals) else 0
        target_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        target_range = float(vals.max() - vals.min()) if len(vals) else 0.0
        unique_targets = int(vals.nunique()) if len(vals) else 0

        reasons = []
        if n_samples < min_samples:
            reasons.append("low_sample_count")
        if target_std < min_target_std:
            reasons.append("low_residual_std")
        if target_range < min_target_range:
            reasons.append("low_residual_range")
        if unique_targets < min_unique_targets:
            reasons.append("low_unique_targets")

        rows.append({
            "drug_key": str(drug_key),
            "n_samples": int(n_samples),
            "target_std": target_std,
            "target_range": target_range,
            "n_unique_targets": unique_targets,
            "eligible": len(reasons) == 0,
            "ineligibility_reasons": ";".join(reasons),
        })

    return pd.DataFrame(rows).sort_values(["eligible", "n_samples", "target_std"], ascending=[False, False, False])
