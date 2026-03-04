"""Metrics computation for RC validation.

Computes: minutes MAE, 3PA MAE/RMSE, O/U LogLoss, Brier, calibration error,
CLV, ROI, hit rate, drawdown, and sliced breakdowns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluation import (
    calibration_metrics_for_prob,
    payout_ratio_from_american,
    prepare_bet_log,
    settled_units,
    summarize_profitability,
)
from ..utils import american_to_implied_prob

# ---------------------------------------------------------------------------
# Slicing dimensions
# ---------------------------------------------------------------------------

SEASON_COL = "season"
MONTH_COL = "month"
MARKET_COL = "market"
ARCHETYPE_COL = "archetype"
LINE_BUCKET_COL = "line_bucket"
TRACKING_COL = "tracking_available"
REST_COL = "rest_bucket"
SPREAD_COL = "spread_bucket"
BET_TIME_COL = "bet_time_bucket"

LINE_BUCKET_EDGES = {
    "PTS": [0, 12.5, 18.5, 24.5, 30.5, 999],
    "REB": [0, 3.5, 5.5, 8.5, 999],
    "AST": [0, 2.5, 4.5, 7.5, 999],
    "FG3M": [0, 0.5, 1.5, 2.5, 3.5, 999],
}

SPREAD_BUCKET_EDGES = [-999, -10, -5, -1, 1, 5, 10, 999]
SPREAD_LABELS = ["fav10+", "fav5-10", "fav1-5", "pk", "dog1-5", "dog5-10", "dog10+"]


def _bucket(value: float, edges: Sequence[float], labels: Optional[Sequence[str]] = None) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            if labels and i < len(labels):
                return labels[i]
            return f"{edges[i]}-{edges[i+1]}"
    return "other"


def add_slice_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add slicing dimension columns to a prepared bet-log DataFrame."""
    out = df.copy()

    # Month
    if "asof_utc" in out.columns:
        ts = pd.to_datetime(out["asof_utc"], utc=True, errors="coerce")
        out[MONTH_COL] = ts.dt.strftime("%Y-%m")

    # Season (Oct→Jun mapping)
    if "asof_utc" in out.columns:
        ts = pd.to_datetime(out["asof_utc"], utc=True, errors="coerce")
        year = ts.dt.year
        month = ts.dt.month
        season_start = np.where(month >= 10, year, year - 1)
        season_end = season_start + 1
        out[SEASON_COL] = [f"{s}-{str(e)[-2:]}" for s, e in zip(season_start, season_end)]

    # Line bucket
    market_upper = out["market"].astype(str).str.upper()
    line_vals = pd.to_numeric(out["line"], errors="coerce").fillna(0) if "line" in out.columns else pd.Series(0, index=out.index)
    buckets = []
    for m, ln in zip(market_upper, line_vals):
        edges = LINE_BUCKET_EDGES.get(m, [0, 999])
        buckets.append(_bucket(float(ln), edges))
    out[LINE_BUCKET_COL] = buckets

    # Spread bucket
    spread_vals = pd.to_numeric(out["spread"], errors="coerce").fillna(0) if "spread" in out.columns else pd.Series(0, index=out.index)
    out[SPREAD_COL] = [_bucket(float(s), SPREAD_BUCKET_EDGES, SPREAD_LABELS) for s in spread_vals]

    # Tracking availability (flag column, default to unknown)
    if TRACKING_COL not in out.columns:
        out[TRACKING_COL] = "unknown"

    # Rest bucket
    rest_vals = pd.to_numeric(out["rest_days"], errors="coerce").fillna(-1).astype(int) if "rest_days" in out.columns else pd.Series(-1, index=out.index, dtype=int)
    out[REST_COL] = rest_vals.map(lambda r: f"{r}d" if 0 <= r <= 3 else "4+d" if r >= 4 else "unk")

    # Archetype (passthrough if present)
    if ARCHETYPE_COL not in out.columns:
        out[ARCHETYPE_COL] = "unknown"

    # Bet-time bucket
    if "time_to_tip_min" in out.columns:
        ttip = pd.to_numeric(out["time_to_tip_min"], errors="coerce").fillna(9999)
        out[BET_TIME_COL] = pd.cut(
            ttip,
            bins=[0, 30, 60, 120, 240, 9999],
            labels=["0-30m", "30-60m", "1-2h", "2-4h", "4h+"],
            right=True,
        ).astype(str)
    else:
        out[BET_TIME_COL] = "unknown"

    return out


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@dataclass
class MetricsRow:
    """One row of the metrics table."""
    slice_name: str
    slice_value: str
    n: int
    graded: int
    wins: int
    losses: int
    hit_rate: float
    roi: float
    units: float
    clv: float
    brier: float
    log_loss: float
    cal_error: float
    drawdown: float
    avg_edge: float
    expected_units: float


def _safe_log_loss(p_arr: np.ndarray, y_arr: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p_arr, eps, 1 - eps)
    ll = -(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p))
    return float(np.mean(ll))


def _safe_brier(p_arr: np.ndarray, y_arr: np.ndarray) -> float:
    return float(np.mean((p_arr - y_arr) ** 2))


def _max_drawdown(cumulative_units: np.ndarray) -> float:
    if len(cumulative_units) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative_units)
    dd = running_max - cumulative_units
    return float(np.max(dd)) if len(dd) else 0.0


def _clv(df: pd.DataFrame) -> float:
    """Closing Line Value: average edge of model prob vs closing implied prob."""
    if "model_prob_side" not in df.columns or "implied_prob" not in df.columns:
        return 0.0
    mp = pd.to_numeric(df["model_prob_side"], errors="coerce")
    ip = pd.to_numeric(df["implied_prob"], errors="coerce")
    valid = mp.notna() & ip.notna()
    if valid.sum() == 0:
        return 0.0
    return float((mp[valid] - ip[valid]).mean() * 100.0)


def compute_metrics_for_slice(df: pd.DataFrame, slice_name: str, slice_value: str) -> MetricsRow:
    """Compute all metrics for one dimension slice."""
    n = len(df)
    graded = df["is_graded"].sum() if "is_graded" in df.columns else 0
    wins = int(df["is_win"].sum()) if "is_win" in df.columns else 0
    losses = int(graded - wins)
    hit_rate = float(wins / graded) if graded > 0 else 0.0

    units_col = pd.to_numeric(df.get("units"), errors="coerce").fillna(0.0)
    total_units = float(units_col.sum())
    roi = float(total_units / graded * 100) if graded > 0 else 0.0

    clv_val = _clv(df)

    # Probability metrics (on graded bets only)
    graded_mask = df["is_graded"] if "is_graded" in df.columns else pd.Series(False, index=df.index)
    gdf = df[graded_mask]
    if len(gdf) > 0 and "model_prob_side" in gdf.columns:
        p = pd.to_numeric(gdf["model_prob_side"], errors="coerce").fillna(0.5).values
        y = gdf["is_win"].astype(float).values
        brier = _safe_brier(p, y)
        log_loss = _safe_log_loss(p, y)
    else:
        brier = 0.0
        log_loss = 0.0

    # Calibration error (mean absolute deviation of bin accuracy from bin probability)
    cal_error = 0.0
    if len(gdf) > 5 and "model_prob_side" in gdf.columns:
        calib = calibration_metrics_for_prob(gdf, "model_prob_side", bins=5)
        if calib and "reliability_bins" in calib:
            abs_errors = []
            for b in calib["reliability_bins"]:
                if b.get("count", 0) > 0:
                    abs_errors.append(abs(b["win_rate"] - b["p_mean"]))
            cal_error = float(np.mean(abs_errors)) if abs_errors else 0.0

    # Drawdown
    cum = np.cumsum(units_col.values)
    drawdown = _max_drawdown(cum)

    # Edge metrics
    edge_col = pd.to_numeric(df.get("edge_vs_market"), errors="coerce").fillna(0.0)
    avg_edge = float(edge_col.mean()) if len(df) > 0 else 0.0

    exp_units_col = pd.to_numeric(df.get("expected_units"), errors="coerce").fillna(0.0)
    expected_units = float(exp_units_col.sum())

    return MetricsRow(
        slice_name=slice_name,
        slice_value=slice_value,
        n=n,
        graded=int(graded),
        wins=wins,
        losses=losses,
        hit_rate=hit_rate,
        roi=roi,
        units=total_units,
        clv=clv_val,
        brier=brier,
        log_loss=log_loss,
        cal_error=cal_error,
        drawdown=drawdown,
        avg_edge=avg_edge,
        expected_units=expected_units,
    )


def compute_all_sliced_metrics(
    prepared_df: pd.DataFrame,
    slice_dims: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute metrics across all slice dimensions.

    Returns a DataFrame with one row per (dimension, value) pair.
    """
    if slice_dims is None:
        slice_dims = [
            SEASON_COL, MONTH_COL, MARKET_COL, ARCHETYPE_COL,
            LINE_BUCKET_COL, TRACKING_COL, REST_COL, SPREAD_COL,
            BET_TIME_COL,
        ]

    df = add_slice_columns(prepared_df)

    rows: List[MetricsRow] = []
    # Overall
    rows.append(compute_metrics_for_slice(df, "overall", "all"))

    for dim in slice_dims:
        if dim not in df.columns:
            continue
        for val in sorted(df[dim].dropna().unique()):
            sub = df[df[dim] == val]
            if len(sub) < 3:
                continue
            rows.append(compute_metrics_for_slice(sub, dim, str(val)))

    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# Projection-level error metrics (minutes, 3PA)
# ---------------------------------------------------------------------------

def minutes_mae(projected: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error for minutes projections."""
    return float(np.mean(np.abs(np.asarray(projected) - np.asarray(actual))))


def stat_mae_rmse(projected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """MAE and RMSE for any stat projection (e.g., 3PA)."""
    diff = np.asarray(projected) - np.asarray(actual)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse
