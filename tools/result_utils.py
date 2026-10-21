"""Shared validation for experiment results used by selection and reporting tools."""

from __future__ import annotations

import pandas as pd


def normalize_status(value: object) -> str:
    """Normalize a scalar status, including missing values read from a TSV."""
    return "" if pd.isna(value) else str(value).strip().lower()


def valid_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Copy scored, non-crash rows with finite metrics and normalized statuses.

    Search metrics in this repository are maximized. Failed runs and non-finite
    scores must be removed before ranking, even when no successful runs remain.
    """
    if "metric" not in df.columns:
        raise ValueError("results.tsv must contain a 'metric' column")

    view = df.copy()
    view["metric"] = pd.to_numeric(view["metric"], errors="coerce")
    statuses = view.get(
        "status", view.get("status_norm", pd.Series("", index=view.index))
    )
    view["status_norm"] = statuses.map(normalize_status)
    valid = view["metric"].notna() & ~view["metric"].isin([float("inf"), float("-inf")])
    valid &= view["status_norm"] != "crash"
    if "metric_name" in view.columns:
        valid &= view["metric_name"].map(normalize_status) != "error"
    return view.loc[valid].copy()
