"""Comprehensive metrics table generator for walk-forward validation results.

Converts raw ``WalkForwardResults`` into structured, human-readable tables
suitable for terminal display, report generation, and automated gate checks.

Each table method returns a ``list[dict]`` where every dict represents one
row in the table.  The ``format_as_text`` method renders any such table
as aligned columnar text for terminal output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..backtest.metrics import (
    BacktestMetrics,
    BettingMetrics,
    MinutesMetrics,
    ThreePAMetrics,
    ThreePMMetrics,
)
from .walk_forward import (
    ARCHETYPES,
    LINE_BUCKETS,
    REST_BUCKETS,
    SPREAD_BUCKETS,
    TIME_BUCKETS,
    WalkForwardResults,
)

logger = logging.getLogger(__name__)


class MetricsTableGenerator:
    """Generate comprehensive metrics tables from walk-forward results.

    Each public method produces a table as a ``list[dict]`` where every
    dict is a row.  The ``generate_all`` method returns all tables in a
    single ``dict[str, list[dict]]`` payload.

    Parameters
    ----------
    results : WalkForwardResults
        The walk-forward evaluation results to tabulate.
    """

    def __init__(self, results: WalkForwardResults) -> None:
        self.results = results
        self._predictions = results.predictions

    # ------------------------------------------------------------------ #
    # Public: generate all tables at once
    # ------------------------------------------------------------------ #

    def generate_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all metrics tables.

        Returns
        -------
        dict
            Keys are table names, values are ``list[dict]`` tables
            suitable for tabulation or serialisation.
        """
        logger.info(
            "Generating all metrics tables from %d predictions",
            len(self._predictions),
        )

        tables: Dict[str, List[Dict[str, Any]]] = {
            "minutes": self.minutes_table(),
            "three_pa": self.three_pa_table(),
            "three_pm": self.three_pm_table(),
            "betting": self.betting_table(),
            "vs_baselines": self.vs_baselines_table(),
            "calibration_curve": self.calibration_curve_data(),
        }

        return tables

    # ------------------------------------------------------------------ #
    # Minutes table
    # ------------------------------------------------------------------ #

    def minutes_table(self) -> List[Dict[str, Any]]:
        """Generate the minutes prediction metrics table.

        Columns: slice, mae, mae_starters, mae_rotation,
        interval_coverage_80, n_predictions.

        Rows: Overall, by archetype, by rest, by b2b, by home/away,
        by spread bucket.

        Threshold checks:
          - starters MAE <= 2.8
          - rotation MAE <= 3.5

        Returns
        -------
        list[dict]
            Table rows.
        """
        rows: List[Dict[str, Any]] = []

        # Overall
        overall = self._compute_minutes_slice(self._predictions, "Overall")
        rows.append(overall)

        # By archetype
        for archetype in ARCHETYPES:
            preds = [p for p in self._predictions if p.get("archetype") == archetype]
            if preds:
                rows.append(self._compute_minutes_slice(preds, f"archetype:{archetype}"))

        # By rest bucket
        for bucket_name, check_fn in REST_BUCKETS.items():
            preds = [p for p in self._predictions if check_fn(p.get("rest_days", 1))]
            if preds:
                rows.append(self._compute_minutes_slice(preds, f"rest:{bucket_name}"))

        # By b2b
        for b2b_val, label in [(True, "b2b:yes"), (False, "b2b:no")]:
            preds = [p for p in self._predictions if p.get("is_b2b") == b2b_val]
            if preds:
                rows.append(self._compute_minutes_slice(preds, label))

        # By home/away
        for home_val, label in [(True, "home"), (False, "away")]:
            preds = [p for p in self._predictions if p.get("is_home") == home_val]
            if preds:
                rows.append(self._compute_minutes_slice(preds, label))

        # By spread bucket
        for bucket_name, check_fn in SPREAD_BUCKETS.items():
            preds = [
                p for p in self._predictions
                if check_fn(p.get("spread", 0.0) if p.get("spread") is not None else 0.0)
            ]
            if preds:
                rows.append(self._compute_minutes_slice(preds, f"spread:{bucket_name}"))

        return rows

    def _compute_minutes_slice(
        self, predictions: List[Dict[str, Any]], slice_name: str
    ) -> Dict[str, Any]:
        """Compute minutes metrics for a single slice.

        Parameters
        ----------
        predictions : list[dict]
            Filtered predictions for this slice.
        slice_name : str
            Human-readable slice label.

        Returns
        -------
        dict
            Table row with columns: slice, mae, mae_starters,
            mae_rotation, interval_coverage_80, n_predictions.
        """
        valid = [p for p in predictions if p.get("actual_minutes") is not None]
        n = len(valid)

        if n == 0:
            return {
                "slice": slice_name,
                "mae": None,
                "mae_starters": None,
                "mae_rotation": None,
                "interval_coverage_80": None,
                "n_predictions": 0,
            }

        actual = np.array([p["actual_minutes"] for p in valid], dtype=np.float64)
        pred_p50 = np.array([p["predicted_minutes_p50"] for p in valid], dtype=np.float64)
        pred_p10 = np.array([p["predicted_minutes_p10"] for p in valid], dtype=np.float64)
        pred_p90 = np.array([p["predicted_minutes_p90"] for p in valid], dtype=np.float64)

        mae = float(np.mean(np.abs(actual - pred_p50)))

        # Starters: predicted_minutes_p50 >= 28
        starter_mask = pred_p50 >= 28
        mae_starters = float(np.mean(np.abs(actual[starter_mask] - pred_p50[starter_mask]))) if starter_mask.any() else None

        # Rotation: 18 <= predicted_minutes_p50 < 28
        rotation_mask = (pred_p50 >= 18) & (pred_p50 < 28)
        mae_rotation = float(np.mean(np.abs(actual[rotation_mask] - pred_p50[rotation_mask]))) if rotation_mask.any() else None

        # Interval coverage
        in_interval = (actual >= pred_p10) & (actual <= pred_p90)
        coverage = float(np.mean(in_interval))

        return {
            "slice": slice_name,
            "mae": round(mae, 3),
            "mae_starters": round(mae_starters, 3) if mae_starters is not None else None,
            "mae_rotation": round(mae_rotation, 3) if mae_rotation is not None else None,
            "interval_coverage_80": round(coverage, 4),
            "n_predictions": n,
        }

    # ------------------------------------------------------------------ #
    # 3PA table
    # ------------------------------------------------------------------ #

    def three_pa_table(self) -> List[Dict[str, Any]]:
        """Generate the 3PA prediction metrics table.

        Columns: slice, mae, rmse, count_calibration, n.

        Rows: Overall, by archetype, by line bucket, by tracking status.

        Thresholds:
          - High-volume (>=8 3PA/36): MAE <= 1.25
          - Standard (5-8 3PA/36): MAE <= 1.00

        Returns
        -------
        list[dict]
            Table rows.
        """
        rows: List[Dict[str, Any]] = []

        # Overall
        rows.append(self._compute_3pa_slice(self._predictions, "Overall"))

        # By archetype
        for archetype in ARCHETYPES:
            preds = [p for p in self._predictions if p.get("archetype") == archetype]
            if preds:
                rows.append(self._compute_3pa_slice(preds, f"archetype:{archetype}"))

        # By line bucket
        for bucket_name, check_fn in LINE_BUCKETS.items():
            preds = [p for p in self._predictions if check_fn(p.get("line", 0.0))]
            if preds:
                rows.append(self._compute_3pa_slice(preds, f"line:{bucket_name}"))

        # By tracking status
        for tracking_val, label in [(True, "tracking:available"), (False, "tracking:fallback")]:
            preds = [p for p in self._predictions if p.get("tracking_available") == tracking_val]
            if preds:
                rows.append(self._compute_3pa_slice(preds, label))

        # By volume bucket (high-volume vs standard)
        high_vol = [p for p in self._predictions if p.get("predicted_3pa_mean", 0) >= 8]
        if high_vol:
            rows.append(self._compute_3pa_slice(high_vol, "volume:high_>=8"))

        standard_vol = [
            p for p in self._predictions
            if 5 <= (p.get("predicted_3pa_mean", 0)) < 8
        ]
        if standard_vol:
            rows.append(self._compute_3pa_slice(standard_vol, "volume:standard_5-8"))

        return rows

    def _compute_3pa_slice(
        self, predictions: List[Dict[str, Any]], slice_name: str
    ) -> Dict[str, Any]:
        """Compute 3PA metrics for a single slice.

        Parameters
        ----------
        predictions : list[dict]
            Filtered predictions.
        slice_name : str
            Slice label.

        Returns
        -------
        dict
            Row with columns: slice, mae, rmse, count_calibration, n.
        """
        valid = [p for p in predictions if p.get("actual_3pa") is not None]
        n = len(valid)

        if n == 0:
            return {
                "slice": slice_name,
                "mae": None,
                "rmse": None,
                "count_calibration": None,
                "n": 0,
            }

        actual = np.array([p["actual_3pa"] for p in valid], dtype=np.float64)
        predicted = np.array([p["predicted_3pa_mean"] for p in valid], dtype=np.float64)

        tpa_metrics = BacktestMetrics.compute_3pa_metrics(actual=actual, predicted=predicted)

        return {
            "slice": slice_name,
            "mae": round(tpa_metrics.mae, 3),
            "rmse": round(tpa_metrics.rmse, 3),
            "count_calibration": round(tpa_metrics.count_calibration, 4),
            "n": n,
        }

    # ------------------------------------------------------------------ #
    # 3PM table
    # ------------------------------------------------------------------ #

    def three_pm_table(self) -> List[Dict[str, Any]]:
        """Generate the 3PM over/under probability metrics table.

        Columns: slice, log_loss, brier, calibration_error, sharpness, n.

        Rows: Overall, then by all slicing dimensions (line bucket,
        spread bucket, rest bucket, archetype, time bucket, tracking
        status, home/away, b2b).

        Returns
        -------
        list[dict]
            Table rows.
        """
        rows: List[Dict[str, Any]] = []

        # Overall
        rows.append(self._compute_3pm_slice(self._predictions, "Overall"))

        # By line bucket
        for bucket_name, check_fn in LINE_BUCKETS.items():
            preds = [p for p in self._predictions if check_fn(p.get("line", 0.0))]
            if preds:
                rows.append(self._compute_3pm_slice(preds, f"line:{bucket_name}"))

        # By spread bucket
        for bucket_name, check_fn in SPREAD_BUCKETS.items():
            preds = [
                p for p in self._predictions
                if check_fn(p.get("spread", 0.0) if p.get("spread") is not None else 0.0)
            ]
            if preds:
                rows.append(self._compute_3pm_slice(preds, f"spread:{bucket_name}"))

        # By rest bucket
        for bucket_name, check_fn in REST_BUCKETS.items():
            preds = [p for p in self._predictions if check_fn(p.get("rest_days", 1))]
            if preds:
                rows.append(self._compute_3pm_slice(preds, f"rest:{bucket_name}"))

        # By archetype
        for archetype in ARCHETYPES:
            preds = [p for p in self._predictions if p.get("archetype") == archetype]
            if preds:
                rows.append(self._compute_3pm_slice(preds, f"archetype:{archetype}"))

        # By time bucket
        for tb in TIME_BUCKETS:
            preds = [p for p in self._predictions if p.get("time_bucket") == tb]
            if preds:
                rows.append(self._compute_3pm_slice(preds, f"time:{tb}"))

        # By tracking status
        for tracking_val, label in [(True, "tracking:available"), (False, "tracking:fallback")]:
            preds = [p for p in self._predictions if p.get("tracking_available") == tracking_val]
            if preds:
                rows.append(self._compute_3pm_slice(preds, label))

        # By home/away
        for home_val, label in [(True, "home"), (False, "away")]:
            preds = [p for p in self._predictions if p.get("is_home") == home_val]
            if preds:
                rows.append(self._compute_3pm_slice(preds, label))

        # By b2b
        for b2b_val, label in [(True, "b2b:yes"), (False, "b2b:no")]:
            preds = [p for p in self._predictions if p.get("is_b2b") == b2b_val]
            if preds:
                rows.append(self._compute_3pm_slice(preds, label))

        return rows

    def _compute_3pm_slice(
        self, predictions: List[Dict[str, Any]], slice_name: str
    ) -> Dict[str, Any]:
        """Compute 3PM probability metrics for a single slice.

        Parameters
        ----------
        predictions : list[dict]
            Filtered predictions.
        slice_name : str
            Slice label.

        Returns
        -------
        dict
            Row with columns: slice, log_loss, brier, calibration_error,
            sharpness, n.
        """
        valid = [
            p for p in predictions
            if p.get("actual_over") is not None and p.get("sim_p_over") is not None
        ]
        n = len(valid)

        if n == 0:
            return {
                "slice": slice_name,
                "log_loss": None,
                "brier": None,
                "calibration_error": None,
                "sharpness": None,
                "n": 0,
            }

        actual_over = np.array([float(p["actual_over"]) for p in valid], dtype=np.float64)
        p_over = np.array([p["sim_p_over"] for p in valid], dtype=np.float64)

        pm_metrics = BacktestMetrics.compute_3pm_metrics(
            actual_over=actual_over, p_over=p_over
        )

        # Calibration error: mean absolute difference between predicted
        # and actual frequencies across bins
        calibration_error = self._compute_calibration_error(actual_over, p_over)

        return {
            "slice": slice_name,
            "log_loss": round(pm_metrics.log_loss, 4),
            "brier": round(pm_metrics.brier_score, 4),
            "calibration_error": round(calibration_error, 4),
            "sharpness": round(pm_metrics.sharpness, 4),
            "n": n,
        }

    @staticmethod
    def _compute_calibration_error(
        actual_over: np.ndarray,
        p_over: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute expected calibration error (ECE).

        Parameters
        ----------
        actual_over : np.ndarray
            Binary outcome array.
        p_over : np.ndarray
            Predicted probability array.
        n_bins : int
            Number of calibration bins.

        Returns
        -------
        float
            Weighted average absolute calibration error.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(actual_over)

        if total == 0:
            return 0.0

        for i in range(n_bins):
            mask = (p_over >= bin_edges[i]) & (p_over < bin_edges[i + 1])
            n_bin = int(np.sum(mask))
            if n_bin == 0:
                continue
            mean_pred = float(np.mean(p_over[mask]))
            mean_actual = float(np.mean(actual_over[mask]))
            ece += (n_bin / total) * abs(mean_pred - mean_actual)

        return ece

    # ------------------------------------------------------------------ #
    # Betting table
    # ------------------------------------------------------------------ #

    def betting_table(self) -> List[Dict[str, Any]]:
        """Generate the betting performance metrics table.

        Columns: slice, n_bets, hit_rate, avg_edge, clv_mean, roi,
        max_drawdown, turnover.

        Rows: Overall, by book, by line bucket, by archetype, by time
        bucket, by tracking status.

        Returns
        -------
        list[dict]
            Table rows.
        """
        rows: List[Dict[str, Any]] = []

        # Overall
        rows.append(self._compute_betting_slice(self._predictions, "Overall"))

        # By sportsbook (if available)
        sportsbooks = set()
        for p in self._predictions:
            sb = p.get("sportsbook")
            if sb:
                sportsbooks.add(sb)
        for sb in sorted(sportsbooks):
            preds = [p for p in self._predictions if p.get("sportsbook") == sb]
            if preds:
                rows.append(self._compute_betting_slice(preds, f"book:{sb}"))

        # By line bucket
        for bucket_name, check_fn in LINE_BUCKETS.items():
            preds = [p for p in self._predictions if check_fn(p.get("line", 0.0))]
            if preds:
                rows.append(self._compute_betting_slice(preds, f"line:{bucket_name}"))

        # By archetype
        for archetype in ARCHETYPES:
            preds = [p for p in self._predictions if p.get("archetype") == archetype]
            if preds:
                rows.append(self._compute_betting_slice(preds, f"archetype:{archetype}"))

        # By time bucket
        for tb in TIME_BUCKETS:
            preds = [p for p in self._predictions if p.get("time_bucket") == tb]
            if preds:
                rows.append(self._compute_betting_slice(preds, f"time:{tb}"))

        # By tracking status
        for tracking_val, label in [(True, "tracking:available"), (False, "tracking:fallback")]:
            preds = [p for p in self._predictions if p.get("tracking_available") == tracking_val]
            if preds:
                rows.append(self._compute_betting_slice(preds, label))

        return rows

    def _compute_betting_slice(
        self, predictions: List[Dict[str, Any]], slice_name: str
    ) -> Dict[str, Any]:
        """Compute betting metrics for a single slice.

        Parameters
        ----------
        predictions : list[dict]
            Filtered predictions.
        slice_name : str
            Slice label.

        Returns
        -------
        dict
            Row with columns: slice, n_bets, hit_rate, avg_edge,
            clv_mean, roi, max_drawdown, turnover.
        """
        bet_preds = [
            p for p in predictions
            if p.get("recommended_side") not in (None, "no_bet")
            and p.get("bet_result") is not None
        ]
        n_bets = len(bet_preds)

        if n_bets == 0:
            return {
                "slice": slice_name,
                "n_bets": 0,
                "hit_rate": None,
                "avg_edge": None,
                "clv_mean": None,
                "roi": None,
                "max_drawdown": None,
                "turnover": None,
            }

        edges = np.array([p["edge"] for p in bet_preds], dtype=np.float64)
        results = np.array(
            [
                1.0 if p["bet_result"] == "win"
                else (-1.0 if p["bet_result"] == "loss" else 0.0)
                for p in bet_preds
            ],
            dtype=np.float64,
        )
        stakes = np.ones(n_bets, dtype=np.float64)

        bet_odds: List[float] = []
        for p in bet_preds:
            if p["recommended_side"] == "over":
                bet_odds.append(self._american_to_decimal(float(p["odds_over"])))
            else:
                bet_odds.append(self._american_to_decimal(float(p["odds_under"])))
        odds_decimal = np.array(bet_odds, dtype=np.float64)

        clv_pts = np.array([p.get("clv_prob_pts", 0.0) for p in bet_preds], dtype=np.float64)

        bm = BacktestMetrics.compute_betting_metrics(
            edges=edges,
            results=results,
            stakes=stakes,
            odds_decimal=odds_decimal,
            clv_pts=clv_pts,
        )

        return {
            "slice": slice_name,
            "n_bets": bm.n_bets,
            "hit_rate": round(bm.hit_rate, 4),
            "avg_edge": round(bm.avg_edge, 4),
            "clv_mean": round(bm.clv_mean, 4),
            "roi": round(bm.roi, 4),
            "max_drawdown": round(bm.max_drawdown, 4),
            "turnover": round(bm.turnover, 2),
        }

    # ------------------------------------------------------------------ #
    # vs-baselines table
    # ------------------------------------------------------------------ #

    def vs_baselines_table(self) -> List[Dict[str, Any]]:
        """Generate the production vs. baselines comparison table.

        Columns: metric, production, rolling_avg, direct_3pm, bookmaker,
        production_beats_all (bool).

        Metrics compared: log_loss, brier, clv, roi.

        Returns
        -------
        list[dict]
            Table rows, one per metric.
        """
        prod_metrics = self.results.production_metrics
        bl_metrics = self.results.baseline_metrics

        # Production 3PM metrics
        prod_3pm = prod_metrics.get("three_pm", {})
        prod_betting = prod_metrics.get("betting", {})

        # Baselines 3PM metrics
        bl_rolling = bl_metrics.get("rolling_avg", {})
        bl_direct = bl_metrics.get("direct_3pm", {})
        bl_book = bl_metrics.get("bookmaker", {})

        metrics_to_compare = [
            ("log_loss", "lower_is_better"),
            ("brier", "lower_is_better"),
            ("clv", "higher_is_better"),
            ("roi", "higher_is_better"),
        ]

        rows: List[Dict[str, Any]] = []

        for metric_name, direction in metrics_to_compare:
            # Get production value
            if metric_name == "log_loss":
                prod_val = prod_3pm.get("log_loss")
            elif metric_name == "brier":
                prod_val = prod_3pm.get("brier_score")
            elif metric_name == "clv":
                prod_val = prod_betting.get("clv_mean")
            elif metric_name == "roi":
                prod_val = prod_betting.get("roi")
            else:
                prod_val = None

            # Get baseline values
            if metric_name in ("log_loss", "brier"):
                bl_key = "log_loss" if metric_name == "log_loss" else "brier_score"
                rolling_val = bl_rolling.get(bl_key)
                direct_val = bl_direct.get(bl_key)
                book_val = bl_book.get(bl_key)
            elif metric_name == "clv":
                # Baselines don't have CLV/ROI in the same way; set to None
                rolling_val = None
                direct_val = None
                book_val = None
            elif metric_name == "roi":
                rolling_val = None
                direct_val = None
                book_val = None
            else:
                rolling_val = None
                direct_val = None
                book_val = None

            # Determine if production beats all baselines
            production_beats_all = False
            if prod_val is not None:
                baseline_vals = [v for v in [rolling_val, direct_val, book_val] if v is not None]
                if baseline_vals:
                    if direction == "lower_is_better":
                        production_beats_all = all(prod_val <= bv for bv in baseline_vals)
                    else:
                        production_beats_all = all(prod_val >= bv for bv in baseline_vals)

            rows.append(
                {
                    "metric": metric_name,
                    "production": round(prod_val, 4) if prod_val is not None else None,
                    "rolling_avg": round(rolling_val, 4) if rolling_val is not None else None,
                    "direct_3pm": round(direct_val, 4) if direct_val is not None else None,
                    "bookmaker": round(book_val, 4) if book_val is not None else None,
                    "production_beats_all": production_beats_all,
                }
            )

        return rows

    # ------------------------------------------------------------------ #
    # Calibration curve data
    # ------------------------------------------------------------------ #

    def calibration_curve_data(self) -> List[Dict[str, Any]]:
        """Generate calibration (reliability) curve data.

        Uses ``BacktestMetrics.compute_reliability_curve`` to produce
        binned predicted-vs-actual frequencies.

        Returns
        -------
        list[dict]
            Calibration curve data points with ``bin_lower``,
            ``bin_upper``, ``mean_predicted``, ``mean_actual``, and
            ``count``.
        """
        valid = [
            p for p in self._predictions
            if p.get("actual_over") is not None and p.get("sim_p_over") is not None
        ]

        if not valid:
            logger.warning("No valid predictions for calibration curve")
            return []

        actual_over = np.array(
            [float(p["actual_over"]) for p in valid], dtype=np.float64
        )
        p_over = np.array(
            [p["sim_p_over"] for p in valid], dtype=np.float64
        )

        curve = BacktestMetrics.compute_reliability_curve(
            actual_over=actual_over, p_over=p_over, n_bins=10
        )

        return curve

    # ------------------------------------------------------------------ #
    # Text formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_as_text(table: List[Dict[str, Any]], title: str) -> str:
        """Format a table as aligned columnar text for terminal output.

        Parameters
        ----------
        table : list[dict]
            Table rows where each dict has the same keys.
        title : str
            Table title to display above the data.

        Returns
        -------
        str
            Formatted text string ready for printing.
        """
        if not table:
            return f"\n{title}\n{'=' * len(title)}\n(no data)\n"

        # Determine columns from the first row
        columns = list(table[0].keys())

        # Compute column widths: max of header length and formatted values
        col_widths: Dict[str, int] = {}
        for col in columns:
            max_val_len = max(
                len(_format_cell(row.get(col))) for row in table
            )
            col_widths[col] = max(len(col), max_val_len)

        # Build header
        header = "  ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "  ".join("-" * col_widths[col] for col in columns)

        # Build rows
        data_rows: List[str] = []
        for row in table:
            formatted = "  ".join(
                _format_cell(row.get(col)).ljust(col_widths[col])
                for col in columns
            )
            data_rows.append(formatted)

        # Assemble
        lines = [
            "",
            title,
            "=" * max(len(title), len(header)),
            header,
            separator,
        ]
        lines.extend(data_rows)
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _american_to_decimal(american: float) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return 1.0 + american / 100.0
        elif american < 0:
            return 1.0 + 100.0 / abs(american)
        else:
            return 1.0


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _format_cell(value: Any) -> str:
    """Format a single cell value for text display.

    Parameters
    ----------
    value : Any
        The cell value.

    Returns
    -------
    str
        Formatted string representation.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0.0:
            return f"{value:.6f}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)
