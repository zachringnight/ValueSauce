"""Tracking-rich vs fallback comparison for NBA 3PM Props Engine.

Splits walk-forward predictions into two groups -- games where NBA
tracking data was available versus games that relied on fallback
features -- and compares model performance across all key metrics.
Statistical significance tests flag meaningful discrepancies, and a
viability check determines whether the fallback path is production-
ready.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Significance threshold
# ---------------------------------------------------------------------------

SIGNIFICANCE_ALPHA = 0.05  # p-value threshold for paired t-tests


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackingComparisonResults:
    """Container for tracking-rich vs fallback comparison outputs."""

    tracking_rich: dict = field(default_factory=dict)
    # All metrics for the tracking-available subset.

    fallback: dict = field(default_factory=dict)
    # All metrics for the fallback subset.

    n_tracking: int = 0
    n_fallback: int = 0

    tracking_pct: float = 0.0
    # Percentage of predictions with tracking data available.

    deltas: dict = field(default_factory=dict)
    # metric_name -> (tracking_value - fallback_value)

    significant_differences: list[str] = field(default_factory=list)
    # Metric names with statistically significant differences.

    warnings: list[str] = field(default_factory=list)
    # Any concerning findings.

    summary_table: list[dict] = field(default_factory=list)
    # Formatted rows for display.


# ---------------------------------------------------------------------------
# Core comparison class
# ---------------------------------------------------------------------------

class TrackingComparison:
    """Compare model performance between tracking-rich and fallback games.

    Parameters
    ----------
    walk_forward_results :
        Walk-forward evaluation results (dict or object) containing
        individual predictions with tracking availability flags and
        actual outcomes.
    """

    def __init__(self, walk_forward_results: Any) -> None:
        self.wf_results = walk_forward_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> TrackingComparisonResults:
        """Execute the tracking vs fallback comparison.

        1. Split predictions into tracking_available=True vs False.
        2. Compute minutes, 3PA, 3PM, and betting metrics for each group.
        3. Run statistical significance tests where possible.
        4. Flag any concerning discrepancies.

        Returns
        -------
        TrackingComparisonResults
        """
        logger.info("Starting tracking-rich vs fallback comparison")

        predictions = self._extract_predictions()
        if not predictions:
            logger.warning("No predictions found in walk-forward results")
            return TrackingComparisonResults(
                warnings=["No predictions available for comparison"],
            )

        # Split by tracking availability
        tracking_preds = [
            p for p in predictions if p.get("tracking_available", False)
        ]
        fallback_preds = [
            p for p in predictions if not p.get("tracking_available", False)
        ]

        n_tracking = len(tracking_preds)
        n_fallback = len(fallback_preds)
        n_total = n_tracking + n_fallback
        tracking_pct = (n_tracking / n_total * 100.0) if n_total > 0 else 0.0

        logger.info(
            "Split: %d tracking-rich (%.1f%%), %d fallback (%.1f%%)",
            n_tracking,
            tracking_pct,
            n_fallback,
            100.0 - tracking_pct,
        )

        # Compute metrics for each group
        tracking_metrics = self._compute_group_metrics(tracking_preds)
        fallback_metrics = self._compute_group_metrics(fallback_preds)

        # Compute deltas (tracking - fallback)
        deltas = self._compute_deltas(tracking_metrics, fallback_metrics)

        # Statistical significance tests
        significant_differences = self._run_significance_tests(
            tracking_preds, fallback_preds
        )

        # Assemble warnings
        warnings = self._check_for_warnings(
            tracking_metrics,
            fallback_metrics,
            n_tracking,
            n_fallback,
            tracking_pct,
        )

        # Build summary table
        summary_table = self._build_summary_table(
            tracking_metrics,
            fallback_metrics,
            deltas,
            significant_differences,
            n_tracking,
            n_fallback,
        )

        results = TrackingComparisonResults(
            tracking_rich=tracking_metrics,
            fallback=fallback_metrics,
            n_tracking=n_tracking,
            n_fallback=n_fallback,
            tracking_pct=tracking_pct,
            deltas=deltas,
            significant_differences=significant_differences,
            warnings=warnings,
            summary_table=summary_table,
        )

        logger.info(
            "Tracking comparison complete: %d significant differences, %d warnings",
            len(significant_differences),
            len(warnings),
        )

        return results

    def generate_table(
        self, results: TrackingComparisonResults
    ) -> list[dict]:
        """Generate the formatted comparison table.

        Columns: metric, tracking_rich, fallback, delta, significant,
        n_tracking, n_fallback.

        Parameters
        ----------
        results : TrackingComparisonResults
            Output of :meth:`run`.

        Returns
        -------
        list[dict]
        """
        return results.summary_table

    def check_fallback_viability(
        self, results: TrackingComparisonResults
    ) -> tuple[bool, list[str]]:
        """Determine whether the fallback path is viable for production.

        Fallback is NOT viable if any of the following hold:
        - Fallback log_loss > tracking log_loss by more than 0.05.
        - Fallback CLV is negative while tracking CLV is positive.
        - Fallback calibration error > 0.05.
        - Fallback sample size (n) < 50.

        Parameters
        ----------
        results : TrackingComparisonResults
            Output of :meth:`run`.

        Returns
        -------
        tuple[bool, list[str]]
            ``(viable, reasons)``.  If ``viable`` is False, ``reasons``
            lists the failing criteria.
        """
        reasons: list[str] = []

        tracking = results.tracking_rich
        fallback = results.fallback

        # Check 1: log_loss gap
        tracking_ll = tracking.get("log_loss", 0.0)
        fallback_ll = fallback.get("log_loss", 0.0)
        ll_gap = fallback_ll - tracking_ll
        if ll_gap > 0.05:
            reasons.append(
                f"Fallback log_loss ({fallback_ll:.4f}) exceeds tracking "
                f"log_loss ({tracking_ll:.4f}) by {ll_gap:.4f} (> 0.05 threshold)"
            )

        # Check 2: CLV sign mismatch
        tracking_clv = tracking.get("clv", 0.0)
        fallback_clv = fallback.get("clv", 0.0)
        if fallback_clv < 0 and tracking_clv > 0:
            reasons.append(
                f"Fallback CLV is negative ({fallback_clv:.4f}) while tracking "
                f"CLV is positive ({tracking_clv:.4f})"
            )

        # Check 3: calibration error
        fallback_cal = fallback.get("calibration_error", 0.0)
        if fallback_cal > 0.05:
            reasons.append(
                f"Fallback calibration error ({fallback_cal:.4f}) exceeds "
                f"0.05 threshold"
            )

        # Check 4: insufficient sample size
        if results.n_fallback < 50:
            reasons.append(
                f"Fallback sample size ({results.n_fallback}) is below "
                f"minimum of 50"
            )

        viable = len(reasons) == 0

        if viable:
            logger.info("Fallback path is VIABLE for production use")
        else:
            logger.warning(
                "Fallback path is NOT VIABLE: %d failing criteria", len(reasons)
            )
            for reason in reasons:
                logger.warning("  - %s", reason)

        return viable, reasons

    def format_report(self, results: TrackingComparisonResults) -> str:
        """Produce a human-readable text report.

        Parameters
        ----------
        results : TrackingComparisonResults
            Output of :meth:`run`.

        Returns
        -------
        str
            Multi-line text report.
        """
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("TRACKING-RICH vs FALLBACK COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Sample sizes
        lines.append("SAMPLE SIZES:")
        lines.append("-" * 40)
        lines.append(f"  Tracking-rich: {results.n_tracking:>6d}")
        lines.append(f"  Fallback:      {results.n_fallback:>6d}")
        lines.append(
            f"  Tracking %%:    {results.tracking_pct:>6.1f}%"
        )
        lines.append("")

        # Metric comparison table
        lines.append("METRIC COMPARISON:")
        lines.append("-" * 90)
        header = (
            f"  {'Metric':<25s} {'Tracking':>10s} {'Fallback':>10s} "
            f"{'Delta':>10s} {'Sig?':>6s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for row in results.summary_table:
            sig_marker = "  *" if row.get("significant") else ""
            lines.append(
                f"  {row['metric']:<25s} "
                f"{row['tracking_rich']:>10.4f} "
                f"{row['fallback']:>10.4f} "
                f"{row['delta']:>10.4f} "
                f"{sig_marker:>6s}"
            )

        lines.append("")
        lines.append("  (* = statistically significant at alpha=0.05)")
        lines.append("")

        # Significant differences
        if results.significant_differences:
            lines.append("STATISTICALLY SIGNIFICANT DIFFERENCES:")
            lines.append("-" * 40)
            for metric_name in results.significant_differences:
                lines.append(f"  - {metric_name}")
            lines.append("")

        # Warnings
        if results.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for warning in results.warnings:
                lines.append(f"  [!] {warning}")
            lines.append("")

        # Fallback viability
        viable, reasons = self.check_fallback_viability(results)
        lines.append("FALLBACK VIABILITY ASSESSMENT:")
        lines.append("-" * 40)
        if viable:
            lines.append("  RESULT: VIABLE")
            lines.append(
                "  The fallback prediction path meets all quality thresholds "
                "for production use."
            )
        else:
            lines.append("  RESULT: NOT VIABLE")
            for reason in reasons:
                lines.append(f"  - {reason}")
        lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_predictions(self) -> list[dict]:
        """Extract individual prediction records from walk-forward results.

        Supports both dict-style and object-style walk-forward results.

        Returns
        -------
        list[dict]
            Prediction records, each containing at minimum:
            ``tracking_available``, ``actual_minutes``, ``predicted_minutes_p50``,
            ``predicted_minutes_p10``, ``predicted_minutes_p90``,
            ``actual_3pa``, ``predicted_3pa``, ``actual_3pm``, ``line``,
            ``p_over``, and optionally betting fields.
        """
        if isinstance(self.wf_results, dict):
            preds = self.wf_results.get("predictions", [])
        else:
            preds = getattr(self.wf_results, "predictions", [])

        return list(preds) if preds else []

    def _compute_group_metrics(self, predictions: list[dict]) -> dict:
        """Compute all metrics for a subset of predictions.

        Parameters
        ----------
        predictions : list[dict]
            Subset of prediction records (either tracking or fallback).

        Returns
        -------
        dict
            Flat dictionary of all computed metrics.
        """
        if not predictions:
            return {
                "minutes_mae": 0.0,
                "interval_coverage": 0.0,
                "3pa_mae": 0.0,
                "3pa_rmse": 0.0,
                "log_loss": 0.0,
                "brier": 0.0,
                "calibration_error": 0.0,
                "clv": 0.0,
                "roi": 0.0,
                "hit_rate": 0.0,
                "n_bets": 0,
            }

        metrics_calc = BacktestMetrics()
        result: dict[str, Any] = {}

        # ---- Minutes metrics ----
        minutes_actual = np.array([
            p.get("actual_minutes", p.get("actual", {}).get("minutes", 0.0))
            for p in predictions
        ], dtype=float)
        minutes_p50 = np.array([
            p.get("predicted_minutes_p50",
                   p.get("minutes_prediction", {}).get("p50", 0.0))
            for p in predictions
        ], dtype=float)
        minutes_p10 = np.array([
            p.get("predicted_minutes_p10",
                   p.get("minutes_prediction", {}).get("p10", 0.0))
            for p in predictions
        ], dtype=float)
        minutes_p90 = np.array([
            p.get("predicted_minutes_p90",
                   p.get("minutes_prediction", {}).get("p90", 0.0))
            for p in predictions
        ], dtype=float)

        if len(minutes_actual) > 0 and np.any(minutes_actual > 0):
            min_metrics = metrics_calc.compute_minutes_metrics(
                actual=minutes_actual,
                predicted_p50=minutes_p50,
                predicted_p10=minutes_p10,
                predicted_p90=minutes_p90,
            )
            result["minutes_mae"] = min_metrics.mae_overall
            result["interval_coverage"] = min_metrics.interval_coverage_80
        else:
            result["minutes_mae"] = 0.0
            result["interval_coverage"] = 0.0

        # ---- 3PA metrics ----
        tpa_actual = np.array([
            float(
                p.get("actual_3pa",
                       p.get("actual", {}).get("three_pa", 0))
            )
            for p in predictions
        ], dtype=float)
        tpa_predicted = np.array([
            float(
                p.get("predicted_3pa",
                       p.get("tpa_prediction", {}).get("expected_3pa", 0.0))
            )
            for p in predictions
        ], dtype=float)

        if len(tpa_actual) > 0:
            tpa_metrics = metrics_calc.compute_3pa_metrics(
                actual=tpa_actual, predicted=tpa_predicted
            )
            result["3pa_mae"] = tpa_metrics.mae
            result["3pa_rmse"] = tpa_metrics.rmse
        else:
            result["3pa_mae"] = 0.0
            result["3pa_rmse"] = 0.0

        # ---- 3PM metrics (log_loss, brier, calibration) ----
        actual_3pm_arr = np.array([
            float(
                p.get("actual_3pm",
                       p.get("actual", {}).get("three_pm", 0))
            )
            for p in predictions
        ], dtype=float)
        lines = np.array([
            float(p.get("line", 2.5))
            for p in predictions
        ], dtype=float)
        actual_over = (actual_3pm_arr > lines).astype(float)

        p_over = np.array([
            float(
                p.get("p_over",
                       p.get("mc_result", {}).get("p_over", 0.5))
            )
            for p in predictions
        ], dtype=float)

        if len(actual_over) > 0:
            tpm_metrics = metrics_calc.compute_3pm_metrics(
                actual_over=actual_over, p_over=p_over
            )
            result["log_loss"] = tpm_metrics.log_loss
            result["brier"] = tpm_metrics.brier_score

            # Calibration error: |mean(predicted) - mean(actual)|
            result["calibration_error"] = float(
                abs(np.mean(p_over) - np.mean(actual_over))
            )
        else:
            result["log_loss"] = 0.0
            result["brier"] = 0.0
            result["calibration_error"] = 0.0

        # ---- Betting metrics ----
        # Extract betting fields if available
        edges = []
        bet_results = []
        stakes = []
        odds_decimal = []
        clv_pts = []

        for p in predictions:
            edge = p.get("edge")
            bet_result = p.get("bet_result")
            stake = p.get("stake", p.get("stake_pct"))
            odds = p.get("odds_decimal")
            clv = p.get("clv_prob_pts", p.get("clv"))

            # Only include actual bets (not no_bet decisions)
            if (
                edge is not None
                and bet_result is not None
                and stake is not None
                and odds is not None
            ):
                edges.append(float(edge))
                # Convert bet_result to numeric: win=+1, loss=-1, push=0
                if isinstance(bet_result, (int, float)):
                    bet_results.append(float(bet_result))
                elif isinstance(bet_result, str):
                    if bet_result in ("win", "won"):
                        bet_results.append(1.0)
                    elif bet_result in ("loss", "lost"):
                        bet_results.append(-1.0)
                    else:
                        bet_results.append(0.0)
                else:
                    bet_results.append(0.0)

                stakes.append(float(stake))
                odds_decimal.append(float(odds))
                if clv is not None:
                    clv_pts.append(float(clv))

        if edges:
            betting_metrics = metrics_calc.compute_betting_metrics(
                edges=np.array(edges),
                results=np.array(bet_results),
                stakes=np.array(stakes),
                odds_decimal=np.array(odds_decimal),
                clv_pts=np.array(clv_pts) if clv_pts else None,
            )
            result["clv"] = betting_metrics.clv_mean
            result["roi"] = betting_metrics.roi
            result["hit_rate"] = betting_metrics.hit_rate
            result["n_bets"] = betting_metrics.n_bets
        else:
            result["clv"] = 0.0
            result["roi"] = 0.0
            result["hit_rate"] = 0.0
            result["n_bets"] = 0

        return result

    def _compute_deltas(
        self, tracking_metrics: dict, fallback_metrics: dict
    ) -> dict[str, float]:
        """Compute tracking - fallback delta for each metric.

        Parameters
        ----------
        tracking_metrics : dict
            Metrics from the tracking-available group.
        fallback_metrics : dict
            Metrics from the fallback group.

        Returns
        -------
        dict
            metric_name -> delta value.
        """
        all_metric_keys = set(tracking_metrics.keys()) | set(
            fallback_metrics.keys()
        )
        # Exclude non-numeric keys
        skip_keys = {"n_bets"}

        deltas: dict[str, float] = {}
        for key in sorted(all_metric_keys):
            if key in skip_keys:
                continue
            t_val = tracking_metrics.get(key, 0.0)
            f_val = fallback_metrics.get(key, 0.0)
            if isinstance(t_val, (int, float)) and isinstance(
                f_val, (int, float)
            ):
                deltas[key] = round(float(t_val) - float(f_val), 6)

        return deltas

    def _run_significance_tests(
        self,
        tracking_preds: list[dict],
        fallback_preds: list[dict],
    ) -> list[str]:
        """Run statistical significance tests between the two groups.

        Uses a Welch two-sample t-test on prediction errors for each
        applicable metric.  Falls back gracefully if scipy is not
        available.

        Parameters
        ----------
        tracking_preds : list[dict]
            Predictions from the tracking-available group.
        fallback_preds : list[dict]
            Predictions from the fallback group.

        Returns
        -------
        list[str]
            Names of metrics with statistically significant differences.
        """
        if not tracking_preds or not fallback_preds:
            return []

        significant: list[str] = []

        try:
            from scipy import stats as scipy_stats
        except ImportError:
            logger.warning(
                "scipy not available; skipping significance tests"
            )
            return []

        # Test minutes prediction errors
        tracking_min_errors = self._extract_errors(
            tracking_preds, "minutes"
        )
        fallback_min_errors = self._extract_errors(
            fallback_preds, "minutes"
        )
        if len(tracking_min_errors) >= 5 and len(fallback_min_errors) >= 5:
            stat, pval = scipy_stats.ttest_ind(
                tracking_min_errors, fallback_min_errors, equal_var=False
            )
            if pval < SIGNIFICANCE_ALPHA:
                significant.append("minutes_mae")
                logger.info(
                    "Significant difference in minutes_mae (p=%.4f)", pval
                )

        # Test 3PA prediction errors
        tracking_tpa_errors = self._extract_errors(
            tracking_preds, "3pa"
        )
        fallback_tpa_errors = self._extract_errors(
            fallback_preds, "3pa"
        )
        if len(tracking_tpa_errors) >= 5 and len(fallback_tpa_errors) >= 5:
            stat, pval = scipy_stats.ttest_ind(
                tracking_tpa_errors, fallback_tpa_errors, equal_var=False
            )
            if pval < SIGNIFICANCE_ALPHA:
                significant.append("3pa_mae")
                logger.info(
                    "Significant difference in 3pa_mae (p=%.4f)", pval
                )

        # Test 3PM log-loss (squared error as proxy for individual losses)
        tracking_3pm_errors = self._extract_errors(
            tracking_preds, "3pm_brier"
        )
        fallback_3pm_errors = self._extract_errors(
            fallback_preds, "3pm_brier"
        )
        if len(tracking_3pm_errors) >= 5 and len(fallback_3pm_errors) >= 5:
            stat, pval = scipy_stats.ttest_ind(
                tracking_3pm_errors, fallback_3pm_errors, equal_var=False
            )
            if pval < SIGNIFICANCE_ALPHA:
                significant.append("brier")
                logger.info(
                    "Significant difference in brier (p=%.4f)", pval
                )

        # Test CLV if available
        tracking_clv_vals = [
            float(p["clv_prob_pts"])
            for p in tracking_preds
            if p.get("clv_prob_pts") is not None
        ]
        fallback_clv_vals = [
            float(p["clv_prob_pts"])
            for p in fallback_preds
            if p.get("clv_prob_pts") is not None
        ]
        if len(tracking_clv_vals) >= 5 and len(fallback_clv_vals) >= 5:
            stat, pval = scipy_stats.ttest_ind(
                tracking_clv_vals, fallback_clv_vals, equal_var=False
            )
            if pval < SIGNIFICANCE_ALPHA:
                significant.append("clv")
                logger.info(
                    "Significant difference in clv (p=%.4f)", pval
                )

        return significant

    def _extract_errors(
        self, predictions: list[dict], error_type: str
    ) -> np.ndarray:
        """Extract per-prediction errors for significance testing.

        Parameters
        ----------
        predictions : list[dict]
            Prediction records.
        error_type : str
            One of ``"minutes"``, ``"3pa"``, ``"3pm_brier"``.

        Returns
        -------
        np.ndarray
            Array of individual prediction errors.
        """
        errors: list[float] = []

        for p in predictions:
            if error_type == "minutes":
                actual = p.get(
                    "actual_minutes",
                    p.get("actual", {}).get("minutes"),
                )
                predicted = p.get(
                    "predicted_minutes_p50",
                    p.get("minutes_prediction", {}).get("p50"),
                )
                if actual is not None and predicted is not None:
                    errors.append(abs(float(actual) - float(predicted)))

            elif error_type == "3pa":
                actual = p.get(
                    "actual_3pa",
                    p.get("actual", {}).get("three_pa"),
                )
                predicted = p.get(
                    "predicted_3pa",
                    p.get("tpa_prediction", {}).get("expected_3pa"),
                )
                if actual is not None and predicted is not None:
                    errors.append(abs(float(actual) - float(predicted)))

            elif error_type == "3pm_brier":
                actual_3pm = p.get(
                    "actual_3pm",
                    p.get("actual", {}).get("three_pm"),
                )
                line = p.get("line", 2.5)
                p_over = p.get(
                    "p_over",
                    p.get("mc_result", {}).get("p_over"),
                )
                if (
                    actual_3pm is not None
                    and p_over is not None
                    and line is not None
                ):
                    actual_over = 1.0 if float(actual_3pm) > float(line) else 0.0
                    # Brier component for this prediction
                    errors.append(
                        (float(p_over) - actual_over) ** 2
                    )

        return np.array(errors, dtype=float)

    def _check_for_warnings(
        self,
        tracking_metrics: dict,
        fallback_metrics: dict,
        n_tracking: int,
        n_fallback: int,
        tracking_pct: float,
    ) -> list[str]:
        """Check for concerning patterns and return warning messages.

        Parameters
        ----------
        tracking_metrics : dict
            Metrics from tracking-available group.
        fallback_metrics : dict
            Metrics from fallback group.
        n_tracking : int
            Number of tracking predictions.
        n_fallback : int
            Number of fallback predictions.
        tracking_pct : float
            Percentage of predictions with tracking data.

        Returns
        -------
        list[str]
            Warning messages.
        """
        warnings: list[str] = []

        # Warn if tracking coverage is too low
        if tracking_pct < 50.0:
            warnings.append(
                f"Low tracking data coverage: only {tracking_pct:.1f}% of "
                f"predictions have tracking data available"
            )

        # Warn if fallback sample is very small
        if 0 < n_fallback < 30:
            warnings.append(
                f"Very small fallback sample (n={n_fallback}); metrics "
                f"may be unreliable"
            )

        # Warn if tracking sample is very small
        if 0 < n_tracking < 30:
            warnings.append(
                f"Very small tracking sample (n={n_tracking}); metrics "
                f"may be unreliable"
            )

        # Warn if fallback log_loss is substantially worse
        t_ll = tracking_metrics.get("log_loss", 0.0)
        f_ll = fallback_metrics.get("log_loss", 0.0)
        if f_ll > 0 and t_ll > 0:
            ll_gap = f_ll - t_ll
            if ll_gap > 0.10:
                warnings.append(
                    f"Large log_loss gap: fallback ({f_ll:.4f}) is "
                    f"{ll_gap:.4f} worse than tracking ({t_ll:.4f})"
                )
            elif ll_gap > 0.05:
                warnings.append(
                    f"Moderate log_loss gap: fallback ({f_ll:.4f}) is "
                    f"{ll_gap:.4f} worse than tracking ({t_ll:.4f})"
                )

        # Warn if fallback calibration error is high
        f_cal = fallback_metrics.get("calibration_error", 0.0)
        if f_cal > 0.05:
            warnings.append(
                f"Fallback calibration error ({f_cal:.4f}) exceeds 0.05 "
                f"threshold"
            )

        # Warn if fallback CLV is negative while tracking is positive
        t_clv = tracking_metrics.get("clv", 0.0)
        f_clv = fallback_metrics.get("clv", 0.0)
        if f_clv < 0 and t_clv > 0:
            warnings.append(
                f"Fallback CLV is negative ({f_clv:.4f}) while tracking "
                f"CLV is positive ({t_clv:.4f}) -- fallback may be "
                f"destroying value"
            )

        # Warn if fallback ROI is significantly worse
        t_roi = tracking_metrics.get("roi", 0.0)
        f_roi = fallback_metrics.get("roi", 0.0)
        if t_roi > 0 and f_roi < -0.05:
            warnings.append(
                f"Fallback ROI ({f_roi:.4f}) is significantly negative "
                f"while tracking ROI ({t_roi:.4f}) is positive"
            )

        # Warn if minutes MAE gap is large
        t_min = tracking_metrics.get("minutes_mae", 0.0)
        f_min = fallback_metrics.get("minutes_mae", 0.0)
        if t_min > 0 and f_min > 0:
            pct_gap = (f_min - t_min) / t_min if t_min > 0 else 0.0
            if pct_gap > 0.25:
                warnings.append(
                    f"Fallback minutes MAE ({f_min:.2f}) is {pct_gap:.0%} "
                    f"worse than tracking ({t_min:.2f})"
                )

        return warnings

    def _build_summary_table(
        self,
        tracking_metrics: dict,
        fallback_metrics: dict,
        deltas: dict[str, float],
        significant_differences: list[str],
        n_tracking: int,
        n_fallback: int,
    ) -> list[dict]:
        """Build the formatted summary table.

        Parameters
        ----------
        tracking_metrics : dict
        fallback_metrics : dict
        deltas : dict
        significant_differences : list[str]
        n_tracking : int
        n_fallback : int

        Returns
        -------
        list[dict]
            One row per metric with columns: metric, tracking_rich,
            fallback, delta, significant, n_tracking, n_fallback.
        """
        # Define display order and labels
        metric_display_order = [
            ("minutes_mae", "Minutes MAE"),
            ("interval_coverage", "Interval Coverage (80%)"),
            ("3pa_mae", "3PA MAE"),
            ("3pa_rmse", "3PA RMSE"),
            ("log_loss", "Log Loss"),
            ("brier", "Brier Score"),
            ("calibration_error", "Calibration Error"),
            ("clv", "CLV (mean)"),
            ("roi", "ROI"),
            ("hit_rate", "Hit Rate"),
        ]

        table: list[dict] = []
        for metric_key, metric_label in metric_display_order:
            t_val = tracking_metrics.get(metric_key, 0.0)
            f_val = fallback_metrics.get(metric_key, 0.0)
            delta_val = deltas.get(metric_key, 0.0)
            is_sig = metric_key in significant_differences

            table.append({
                "metric": metric_label,
                "tracking_rich": float(t_val),
                "fallback": float(f_val),
                "delta": float(delta_val),
                "significant": is_sig,
                "n_tracking": n_tracking,
                "n_fallback": n_fallback,
            })

        # Add n_bets row (integer, not float)
        table.append({
            "metric": "N Bets",
            "tracking_rich": float(tracking_metrics.get("n_bets", 0)),
            "fallback": float(fallback_metrics.get("n_bets", 0)),
            "delta": float(
                tracking_metrics.get("n_bets", 0)
                - fallback_metrics.get("n_bets", 0)
            ),
            "significant": False,
            "n_tracking": n_tracking,
            "n_fallback": n_fallback,
        })

        return table
