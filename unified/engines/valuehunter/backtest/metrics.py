"""Backtest metrics computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MinutesMetrics:
    mae_overall: float = 0.0
    mae_starters: float = 0.0  # >= 28 mpg
    mae_rotation: float = 0.0  # 18-28 mpg
    interval_coverage_80: float = 0.0  # % of actuals within p10-p90


@dataclass
class ThreePAMetrics:
    mae: float = 0.0
    rmse: float = 0.0
    count_calibration: float = 0.0  # chi-squared-like stat


@dataclass
class ThreePMMetrics:
    log_loss: float = 0.0
    brier_score: float = 0.0
    sharpness: float = 0.0  # avg |p - 0.5|


@dataclass
class BettingMetrics:
    clv_mean: float = 0.0
    roi: float = 0.0
    hit_rate: float = 0.0
    avg_edge: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    n_bets: int = 0


@dataclass
class SlicedMetrics:
    """Metrics broken down by slice."""
    by_archetype: dict = field(default_factory=dict)
    by_line_bucket: dict = field(default_factory=dict)  # 0.5, 1.5, 2.5, 3.5+
    by_tracking: dict = field(default_factory=dict)  # available vs fallback
    by_home_away: dict = field(default_factory=dict)
    by_rest: dict = field(default_factory=dict)
    by_b2b: dict = field(default_factory=dict)
    by_injury_load: dict = field(default_factory=dict)
    by_spread_bucket: dict = field(default_factory=dict)
    by_time_bucket: dict = field(default_factory=dict)


class BacktestMetrics:
    """Compute all required backtest metrics."""

    @staticmethod
    def compute_minutes_metrics(
        actual: np.ndarray,
        predicted_p50: np.ndarray,
        predicted_p10: np.ndarray,
        predicted_p90: np.ndarray,
        minutes_avg: Optional[np.ndarray] = None,
    ) -> MinutesMetrics:
        mae_overall = float(np.mean(np.abs(actual - predicted_p50)))

        # Starters: avg >= 28
        if minutes_avg is not None:
            starter_mask = minutes_avg >= 28
            rotation_mask = (minutes_avg >= 18) & (minutes_avg < 28)
            mae_starters = float(np.mean(np.abs(actual[starter_mask] - predicted_p50[starter_mask]))) if starter_mask.any() else 0.0
            mae_rotation = float(np.mean(np.abs(actual[rotation_mask] - predicted_p50[rotation_mask]))) if rotation_mask.any() else 0.0
        else:
            mae_starters = mae_overall
            mae_rotation = mae_overall

        # Interval coverage: fraction within p10-p90
        in_interval = (actual >= predicted_p10) & (actual <= predicted_p90)
        coverage = float(np.mean(in_interval))

        return MinutesMetrics(
            mae_overall=mae_overall,
            mae_starters=mae_starters,
            mae_rotation=mae_rotation,
            interval_coverage_80=coverage,
        )

    @staticmethod
    def compute_3pa_metrics(
        actual: np.ndarray, predicted: np.ndarray
    ) -> ThreePAMetrics:
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # Count calibration: bin predictions, compare mean actual vs mean predicted
        bins = [0, 2, 4, 6, 8, 12, 20]
        chi2 = 0.0
        for i in range(len(bins) - 1):
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
            if mask.any() and np.sum(mask) > 5:
                observed_mean = np.mean(actual[mask])
                expected_mean = np.mean(predicted[mask])
                if expected_mean > 0:
                    chi2 += (observed_mean - expected_mean) ** 2 / expected_mean

        return ThreePAMetrics(mae=mae, rmse=rmse, count_calibration=chi2)

    @staticmethod
    def compute_3pm_metrics(
        actual_over: np.ndarray,  # 1 if actual > line, 0 otherwise
        p_over: np.ndarray,
    ) -> ThreePMMetrics:
        eps = 1e-10
        p_clipped = np.clip(p_over, eps, 1 - eps)

        log_loss = -float(np.mean(
            actual_over * np.log(p_clipped) + (1 - actual_over) * np.log(1 - p_clipped)
        ))
        brier = float(np.mean((p_clipped - actual_over) ** 2))
        sharpness = float(np.mean(np.abs(p_clipped - 0.5)))

        return ThreePMMetrics(log_loss=log_loss, brier_score=brier, sharpness=sharpness)

    @staticmethod
    def compute_betting_metrics(
        edges: np.ndarray,
        results: np.ndarray,  # +1 for win, -1 for loss, 0 for push
        stakes: np.ndarray,
        odds_decimal: np.ndarray,
        clv_pts: Optional[np.ndarray] = None,
    ) -> BettingMetrics:
        n_bets = len(results)
        if n_bets == 0:
            return BettingMetrics()

        # PnL per bet
        pnl = np.where(
            results > 0,
            stakes * (odds_decimal - 1),
            np.where(results < 0, -stakes, 0.0),
        )

        roi = float(np.sum(pnl) / max(np.sum(stakes), 1e-10))
        hit_rate = float(np.mean(results > 0))
        avg_edge = float(np.mean(edges))
        turnover = float(np.sum(stakes))

        # Max drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        clv_mean = float(np.mean(clv_pts)) if clv_pts is not None else 0.0

        return BettingMetrics(
            clv_mean=clv_mean,
            roi=roi,
            hit_rate=hit_rate,
            avg_edge=avg_edge,
            max_drawdown=max_drawdown,
            turnover=turnover,
            n_bets=n_bets,
        )

    @staticmethod
    def compute_reliability_curve(
        actual_over: np.ndarray, p_over: np.ndarray, n_bins: int = 10
    ) -> list[dict]:
        """Compute reliability/calibration curve."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        curve = []
        for i in range(n_bins):
            mask = (p_over >= bin_edges[i]) & (p_over < bin_edges[i + 1])
            if mask.any():
                curve.append({
                    "bin_lower": float(bin_edges[i]),
                    "bin_upper": float(bin_edges[i + 1]),
                    "mean_predicted": float(np.mean(p_over[mask])),
                    "mean_actual": float(np.mean(actual_over[mask])),
                    "count": int(np.sum(mask)),
                })
        return curve

    @staticmethod
    def slice_by(
        data: dict,
        slice_column: np.ndarray,
        slice_values: list,
        metric_fn,
    ) -> dict:
        """Apply a metric function to slices of the data."""
        results = {}
        for val in slice_values:
            mask = slice_column == val
            if mask.any():
                sliced_data = {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray)}
                results[str(val)] = metric_fn(**sliced_data)
        return results
