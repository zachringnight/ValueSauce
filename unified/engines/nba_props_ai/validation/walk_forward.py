"""Walk-forward OOS evaluation engine.

Splits a bet log into temporal folds and evaluates model accuracy
on out-of-sample data using only information available at bet time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    add_slice_columns,
    compute_all_sliced_metrics,
    compute_metrics_for_slice,
)


@dataclass
class WalkForwardFold:
    """One temporal fold of the walk-forward evaluation."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int


def build_temporal_folds(
    df: pd.DataFrame,
    *,
    min_train_rows: int = 80,
    n_folds: int = 5,
    time_col: str = "asof_utc",
) -> List[WalkForwardFold]:
    """Build expanding-window temporal folds.

    The training window starts at the beginning and expands forward.
    The test window is the next ~equal chunk after training.
    """
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    valid = ts.notna()
    if valid.sum() < min_train_rows + 10:
        return []

    sorted_idx = ts[valid].sort_values().index
    total = len(sorted_idx)

    # Use expanding window: train grows, test is fixed-size chunk
    test_size = max(total // (n_folds + 1), 10)
    folds: List[WalkForwardFold] = []

    for i in range(n_folds):
        train_end_pos = min_train_rows + i * test_size
        test_start_pos = train_end_pos
        test_end_pos = min(test_start_pos + test_size, total)

        if train_end_pos >= total or test_start_pos >= total:
            break

        train_idx = sorted_idx[:train_end_pos]
        test_idx = sorted_idx[test_start_pos:test_end_pos]

        if len(test_idx) < 5:
            break

        folds.append(WalkForwardFold(
            fold_id=i + 1,
            train_start=str(ts.loc[train_idx[0]]),
            train_end=str(ts.loc[train_idx[-1]]),
            test_start=str(ts.loc[test_idx[0]]),
            test_end=str(ts.loc[test_idx[-1]]),
            train_size=len(train_idx),
            test_size=len(test_idx),
        ))

    return folds


def run_walk_forward_oos(
    prepared_df: pd.DataFrame,
    *,
    min_train_rows: int = 80,
    n_folds: int = 5,
    slice_dims: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run walk-forward OOS evaluation.

    Returns a dict with:
      - folds: list of fold metadata
      - oos_metrics: DataFrame of metrics computed only on OOS (test) data
      - fold_metrics: list of per-fold metric DataFrames
      - pooled_oos: DataFrame of all OOS rows combined
    """
    folds = build_temporal_folds(prepared_df, min_train_rows=min_train_rows, n_folds=n_folds)
    if not folds:
        return {
            "folds": [],
            "oos_metrics": pd.DataFrame(),
            "fold_metrics": [],
            "pooled_oos": pd.DataFrame(),
        }

    ts = pd.to_datetime(prepared_df["asof_utc"], utc=True, errors="coerce")
    all_oos_rows = []
    fold_metrics_list = []

    for fold in folds:
        test_start = pd.Timestamp(fold.test_start)
        test_end = pd.Timestamp(fold.test_end)
        test_mask = (ts >= test_start) & (ts <= test_end)
        test_df = prepared_df[test_mask].copy()

        if len(test_df) == 0:
            continue

        all_oos_rows.append(test_df)

        # Metrics for this fold
        fold_sliced = compute_all_sliced_metrics(test_df, slice_dims=slice_dims)
        fold_sliced["fold_id"] = fold.fold_id
        fold_metrics_list.append(fold_sliced)

    # Pool all OOS rows (deduplicate by index)
    if all_oos_rows:
        pooled = pd.concat(all_oos_rows).drop_duplicates()
    else:
        pooled = pd.DataFrame()

    # OOS metrics on pooled data
    oos_metrics = compute_all_sliced_metrics(pooled, slice_dims=slice_dims) if len(pooled) > 0 else pd.DataFrame()

    return {
        "folds": [f.__dict__ for f in folds],
        "oos_metrics": oos_metrics,
        "fold_metrics": fold_metrics_list,
        "pooled_oos": pooled,
    }


def compare_vs_baseline(
    model_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    label_model: str = "model",
    label_baseline: str = "baseline",
) -> pd.DataFrame:
    """Side-by-side comparison of model vs baseline metrics.

    Merges on (slice_name, slice_value) and computes deltas.
    """
    if model_metrics.empty or baseline_metrics.empty:
        return pd.DataFrame()

    m = model_metrics.rename(columns={c: f"{c}_{label_model}" for c in model_metrics.columns
                                       if c not in ("slice_name", "slice_value")})
    b = baseline_metrics.rename(columns={c: f"{c}_{label_baseline}" for c in baseline_metrics.columns
                                          if c not in ("slice_name", "slice_value")})
    merged = m.merge(b, on=["slice_name", "slice_value"], how="outer")

    # Compute deltas for key metrics
    for metric in ["brier", "log_loss", "clv", "roi", "hit_rate", "cal_error"]:
        col_m = f"{metric}_{label_model}"
        col_b = f"{metric}_{label_baseline}"
        if col_m in merged.columns and col_b in merged.columns:
            merged[f"{metric}_delta"] = (
                pd.to_numeric(merged[col_m], errors="coerce").fillna(0)
                - pd.to_numeric(merged[col_b], errors="coerce").fillna(0)
            )

    return merged
