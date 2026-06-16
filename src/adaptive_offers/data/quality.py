"""Data-quality profiling for the processed base (Stage 1 quality report).

Produces a compact, reproducible quality profile (missingness, cardinality,
duplicates, target balance, numeric ranges) that backs both the EDA notebook
and ``reports/eda-quality-report.md``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def quality_report(df: pd.DataFrame, target: str = "subscribed") -> dict[str, Any]:
    """Return a JSON-serialisable data-quality profile of ``df``."""
    n = len(df)
    missing = df.isna().sum()
    report: dict[str, Any] = {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "n_duplicates": int(df.duplicated().sum()),
        "columns": {},
    }
    if target in df.columns:
        rate = float(df[target].mean())
        report["target"] = {
            "name": target,
            "positive_rate": round(rate, 4),
            "imbalance_ratio": round((1 - rate) / rate, 2) if rate > 0 else None,
        }

    for col in df.columns:
        s = df[col]
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing": int(missing[col]),
            "missing_pct": round(float(missing[col]) / n * 100, 2) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update(
                min=round(float(s.min()), 3),
                max=round(float(s.max()), 3),
                mean=round(float(s.mean()), 3),
                std=round(float(s.std()), 3),
            )
        else:
            top = s.value_counts(dropna=True).head(3)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        report["columns"][col] = info
    return report


def target_rate_by(df: pd.DataFrame, by: str, target: str = "subscribed") -> pd.DataFrame:
    """Subscription rate and volume grouped by a categorical column."""
    g = df.groupby(by)[target].agg(["mean", "count"]).rename(
        columns={"mean": "subscription_rate", "count": "n"}
    )
    return g.sort_values("subscription_rate", ascending=False).round(4)
