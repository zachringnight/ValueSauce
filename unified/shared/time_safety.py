"""Time-safety utilities for leak-free feature engineering.

From: nil/nba_props/src/utils/time_safety.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def assert_no_future_data(
    df: pd.DataFrame,
    date_col: str,
    cutoff: datetime | pd.Timestamp,
    context: str = "",
):
    """Raise if any rows have dates after cutoff."""
    if df.empty:
        return
    max_date = pd.to_datetime(df[date_col]).max()
    if max_date > pd.Timestamp(cutoff):
        raise ValueError(
            f"Future data leak detected{' in ' + context if context else ''}: "
            f"max date {max_date} > cutoff {cutoff}"
        )


def strict_lag_rolling(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling mean with strict lag (shift 1 before rolling)."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def strict_lag_sum(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling sum with strict lag."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).sum()


def strict_lag_std(
    series: pd.Series,
    window: int,
    min_periods: int = 2,
) -> pd.Series:
    """Rolling std with strict lag."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).std()
