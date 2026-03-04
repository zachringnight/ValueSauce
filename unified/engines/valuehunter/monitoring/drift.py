"""Drift detection and daily monitoring for NBA 3PM Props Engine.

Tracks prediction accuracy, data freshness, and model calibration
to detect degradation before it impacts betting decisions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "minutes_mae_drift_pct": 0.20,         # 20% increase over training MAE
    "three_pa_residual_drift_pct": 0.25,    # 25% increase over training residual
    "make_rate_calibration_drift": 0.05,     # 5 percentage-point calibration gap
    "tracking_missingness_rate": 0.40,       # 40% missing tracking data
    "injury_ingestion_delay_avg_sec": 1800,  # 30 minutes average delay
    "stale_odds_frequency": 0.15,            # 15% of odds snapshots are stale
    "rejection_rate": 0.90,                  # 90% rejection rate
    "suspension_rate": 0.05,                 # 5% suspension rate
    "min_clv_by_book": -0.02,               # negative CLV threshold per book
}


@dataclass
class TrainingBaseline:
    """Baseline metrics from training for comparison."""

    minutes_mae: float = 3.2
    three_pa_residual_std: float = 1.8
    make_rate_calibration_error: float = 0.02


class DriftMonitor:
    """Monitors daily model and data quality metrics for the NBA 3PM Props Engine.

    Compares recent prediction accuracy against training baselines and tracks
    data pipeline health indicators including odds staleness, injury report
    delays, and tracking data availability.
    """

    def __init__(
        self,
        training_baseline: Optional[TrainingBaseline] = None,
        thresholds: Optional[dict[str, float]] = None,
    ):
        self.baseline = training_baseline or TrainingBaseline()
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    def _compute_minutes_mae_drift(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Compare recent 7-day minutes MAE vs training MAE.

        Returns the recent MAE, the training baseline MAE, and the
        proportional drift.
        """
        end_date = target_date
        start_date = target_date - timedelta(days=7)

        rows = repository._fetchall(
            """
            SELECT bd.nba_player_id, bd.nba_game_id,
                   fs.feature_vector::json->>'projected_minutes' AS proj_min,
                   pg.minutes_played AS actual_min
            FROM bet_decisions bd
            JOIN feature_snapshots fs
              ON fs.feature_snapshot_id = bd.feature_snapshot_id
            JOIN player_games pg
              ON pg.nba_game_id = bd.nba_game_id
             AND pg.nba_player_id = bd.nba_player_id
            JOIN games g
              ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND pg.minutes_played IS NOT NULL
            """,
            (start_date, end_date),
        )

        if not rows:
            return {
                "minutes_mae_recent": None,
                "minutes_mae_training": self.baseline.minutes_mae,
                "minutes_mae_drift_pct": None,
            }

        errors = []
        for row in rows:
            proj = row.get("proj_min")
            actual = row.get("actual_min")
            if proj is not None and actual is not None:
                try:
                    errors.append(abs(float(proj) - float(actual)))
                except (ValueError, TypeError):
                    continue

        if not errors:
            return {
                "minutes_mae_recent": None,
                "minutes_mae_training": self.baseline.minutes_mae,
                "minutes_mae_drift_pct": None,
            }

        recent_mae = float(np.mean(errors))
        drift_pct = (recent_mae - self.baseline.minutes_mae) / self.baseline.minutes_mae

        return {
            "minutes_mae_recent": round(recent_mae, 4),
            "minutes_mae_training": self.baseline.minutes_mae,
            "minutes_mae_drift_pct": round(drift_pct, 4),
        }

    def _compute_three_pa_residual_drift(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Compare recent 7-day 3PA residuals vs training residual std."""
        end_date = target_date
        start_date = target_date - timedelta(days=7)

        rows = repository._fetchall(
            """
            SELECT bd.nba_player_id, bd.nba_game_id,
                   fs.feature_vector::json->>'projected_3pa' AS proj_3pa,
                   pg.three_pa AS actual_3pa
            FROM bet_decisions bd
            JOIN feature_snapshots fs
              ON fs.feature_snapshot_id = bd.feature_snapshot_id
            JOIN player_games pg
              ON pg.nba_game_id = bd.nba_game_id
             AND pg.nba_player_id = bd.nba_player_id
            JOIN games g
              ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND pg.three_pa IS NOT NULL
            """,
            (start_date, end_date),
        )

        if not rows:
            return {
                "three_pa_residual_std_recent": None,
                "three_pa_residual_std_training": self.baseline.three_pa_residual_std,
                "three_pa_residual_drift_pct": None,
            }

        residuals = []
        for row in rows:
            proj = row.get("proj_3pa")
            actual = row.get("actual_3pa")
            if proj is not None and actual is not None:
                try:
                    residuals.append(float(actual) - float(proj))
                except (ValueError, TypeError):
                    continue

        if not residuals:
            return {
                "three_pa_residual_std_recent": None,
                "three_pa_residual_std_training": self.baseline.three_pa_residual_std,
                "three_pa_residual_drift_pct": None,
            }

        recent_std = float(np.std(residuals))
        drift_pct = (
            (recent_std - self.baseline.three_pa_residual_std)
            / self.baseline.three_pa_residual_std
        )

        return {
            "three_pa_residual_std_recent": round(recent_std, 4),
            "three_pa_residual_std_training": self.baseline.three_pa_residual_std,
            "three_pa_residual_drift_pct": round(drift_pct, 4),
        }

    def _compute_make_rate_calibration_drift(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Compare model P(over) with observed hit rate over recent 7 days."""
        end_date = target_date
        start_date = target_date - timedelta(days=7)

        rows = repository._fetchall(
            """
            SELECT bd.model_p_over, bd.actual_3pm, bd.line
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND bd.actual_3pm IS NOT NULL
              AND bd.model_p_over IS NOT NULL
              AND bd.line IS NOT NULL
            """,
            (start_date, end_date),
        )

        if not rows:
            return {
                "make_rate_calibration_gap": None,
                "make_rate_calibration_training": self.baseline.make_rate_calibration_error,
            }

        # Bin predictions into deciles and compare predicted vs actual hit rate
        predicted_probs = []
        actual_outcomes = []
        for row in rows:
            p_over = float(row["model_p_over"])
            actual_3pm = float(row["actual_3pm"])
            line = float(row["line"])
            predicted_probs.append(p_over)
            actual_outcomes.append(1.0 if actual_3pm > line else 0.0)

        predicted_probs = np.array(predicted_probs)
        actual_outcomes = np.array(actual_outcomes)

        # Overall calibration gap: |mean(predicted) - mean(actual)|
        calibration_gap = abs(
            float(np.mean(predicted_probs)) - float(np.mean(actual_outcomes))
        )

        return {
            "make_rate_calibration_gap": round(calibration_gap, 4),
            "make_rate_calibration_training": self.baseline.make_rate_calibration_error,
        }

    def _compute_tracking_missingness(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Fraction of player-games missing tracking data on target_date."""
        rows = repository._fetchall(
            """
            SELECT pg.nba_game_id, pg.nba_player_id,
                   pt.tracking_available
            FROM player_games pg
            JOIN games g ON g.nba_game_id = pg.nba_game_id
            LEFT JOIN player_tracking pt
              ON pt.nba_game_id = pg.nba_game_id
             AND pt.nba_player_id = pg.nba_player_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )

        if not rows:
            return {"tracking_missingness_rate": None, "tracking_total_rows": 0}

        total = len(rows)
        missing = sum(
            1
            for r in rows
            if r.get("tracking_available") is None
            or r.get("tracking_available") is False
        )
        rate = missing / total if total > 0 else 0.0

        return {
            "tracking_missingness_rate": round(rate, 4),
            "tracking_total_rows": total,
            "tracking_missing_rows": missing,
        }

    def _compute_injury_ingestion_delay(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Average delay between injury report timestamp and game tipoff."""
        rows = repository._fetchall(
            """
            SELECT i.report_timestamp_utc, g.tipoff_time_utc
            FROM injury_snapshots i
            JOIN games g ON g.nba_game_id = i.nba_game_id
            WHERE g.game_date = %s
              AND i.report_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
            """,
            (target_date,),
        )

        if not rows:
            return {"injury_ingestion_delay_avg_sec": None, "injury_report_count": 0}

        delays = []
        for row in rows:
            report_ts = row["report_timestamp_utc"]
            tipoff_ts = row["tipoff_time_utc"]
            if isinstance(report_ts, datetime) and isinstance(tipoff_ts, datetime):
                delta = (tipoff_ts - report_ts).total_seconds()
                if delta > 0:
                    delays.append(delta)

        avg_delay = float(np.mean(delays)) if delays else None

        return {
            "injury_ingestion_delay_avg_sec": (
                round(avg_delay, 1) if avg_delay is not None else None
            ),
            "injury_report_count": len(rows),
        }

    def _compute_stale_odds_frequency(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Fraction of odds snapshots that are stale (>30 min before tipoff)."""
        rows = repository._fetchall(
            """
            SELECT op.snapshot_timestamp_utc, g.tipoff_time_utc
            FROM odds_props op
            JOIN games g ON g.nba_game_id = op.nba_game_id
            WHERE g.game_date = %s
              AND op.snapshot_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
            """,
            (target_date,),
        )

        if not rows:
            return {"stale_odds_frequency": None, "odds_snapshot_count": 0}

        stale_threshold_sec = self.thresholds.get(
            "injury_ingestion_delay_avg_sec", 1800
        )
        total = len(rows)
        stale = 0
        for row in rows:
            snapshot_ts = row["snapshot_timestamp_utc"]
            tipoff_ts = row["tipoff_time_utc"]
            if isinstance(snapshot_ts, datetime) and isinstance(tipoff_ts, datetime):
                age_sec = (tipoff_ts - snapshot_ts).total_seconds()
                if age_sec > stale_threshold_sec:
                    stale += 1

        rate = stale / total if total > 0 else 0.0

        return {
            "stale_odds_frequency": round(rate, 4),
            "odds_snapshot_count": total,
            "stale_odds_count": stale,
        }

    def _compute_clv_by_book(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Mean CLV (closing line value) grouped by sportsbook."""
        rows = repository._fetchall(
            """
            SELECT bd.sportsbook, bd.clv_prob_pts
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND bd.clv_prob_pts IS NOT NULL
              AND bd.recommended_side != 'no_bet'
            """,
            (target_date - timedelta(days=7), target_date),
        )

        clv_by_book: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            book = row.get("sportsbook", "unknown")
            clv_by_book[book].append(float(row["clv_prob_pts"]))

        return {
            book: round(float(np.mean(vals)), 4)
            for book, vals in clv_by_book.items()
        }

    def _compute_clv_by_time_bucket(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Mean CLV grouped by time-to-tipoff bucket.

        Buckets: '0-1h', '1-3h', '3-6h', '6h+'.
        """
        rows = repository._fetchall(
            """
            SELECT bd.clv_prob_pts, bd.decision_timestamp_utc,
                   g.tipoff_time_utc
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND bd.clv_prob_pts IS NOT NULL
              AND bd.recommended_side != 'no_bet'
              AND bd.decision_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
            """,
            (target_date - timedelta(days=7), target_date),
        )

        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            decision_ts = row["decision_timestamp_utc"]
            tipoff_ts = row["tipoff_time_utc"]
            if isinstance(decision_ts, datetime) and isinstance(tipoff_ts, datetime):
                hours_before = (tipoff_ts - decision_ts).total_seconds() / 3600.0
                if hours_before <= 1:
                    bucket = "0-1h"
                elif hours_before <= 3:
                    bucket = "1-3h"
                elif hours_before <= 6:
                    bucket = "3-6h"
                else:
                    bucket = "6h+"
                buckets[bucket].append(float(row["clv_prob_pts"]))

        return {
            bucket: round(float(np.mean(vals)), 4)
            for bucket, vals in buckets.items()
        }

    def _compute_rejection_and_suspension_rates(
        self, target_date: date, repository
    ) -> dict[str, float]:
        """Compute the rejection rate and suspension rate for the day.

        Rejection rate: fraction of evaluated props where recommended_side = 'no_bet'.
        Suspension rate: fraction of decisions that were later suspended/voided.
        """
        rows = repository._fetchall(
            """
            SELECT bd.recommended_side, bd.bet_result
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )

        if not rows:
            return {
                "rejection_rate": None,
                "suspension_rate": None,
                "total_evaluated": 0,
            }

        total = len(rows)
        rejections = sum(1 for r in rows if r.get("recommended_side") == "no_bet")
        suspensions = sum(
            1
            for r in rows
            if r.get("bet_result") in ("suspended", "voided", "cancelled")
        )

        return {
            "rejection_rate": round(rejections / total, 4) if total else 0.0,
            "suspension_rate": round(suspensions / total, 4) if total else 0.0,
            "total_evaluated": total,
        }

    def _compute_model_vs_market_by_archetype(
        self, target_date: date, repository
    ) -> dict[str, dict[str, float]]:
        """Model edge vs market by player archetype over the last 7 days.

        Returns a dict of archetype -> { mean_edge, count }.
        """
        rows = repository._fetchall(
            """
            SELECT
                fs.feature_vector::json->>'archetype' AS archetype,
                bd.edge_over, bd.edge_under, bd.recommended_side
            FROM bet_decisions bd
            JOIN feature_snapshots fs
              ON fs.feature_snapshot_id = bd.feature_snapshot_id
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date BETWEEN %s AND %s
              AND bd.recommended_side != 'no_bet'
            """,
            (target_date - timedelta(days=7), target_date),
        )

        archetype_edges: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            archetype = row.get("archetype") or "unknown"
            side = row.get("recommended_side")
            if side == "over" and row.get("edge_over") is not None:
                archetype_edges[archetype].append(float(row["edge_over"]))
            elif side == "under" and row.get("edge_under") is not None:
                archetype_edges[archetype].append(float(row["edge_under"]))

        return {
            archetype: {
                "mean_edge": round(float(np.mean(edges)), 4),
                "count": len(edges),
            }
            for archetype, edges in archetype_edges.items()
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_daily_report(self, target_date: date, repository) -> dict:
        """Gather all monitoring metrics for a given date.

        Parameters
        ----------
        target_date : date
            The date to compute the report for.
        repository : Repository
            Database access layer.

        Returns
        -------
        dict
            A dictionary containing all drift and health metrics.
        """
        logger.info("Computing daily drift report for %s", target_date)

        report: dict[str, Any] = {
            "report_date": target_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
        }

        # Model drift metrics
        report["minutes_mae"] = self._compute_minutes_mae_drift(
            target_date, repository
        )
        report["three_pa_residual"] = self._compute_three_pa_residual_drift(
            target_date, repository
        )
        report["make_rate_calibration"] = self._compute_make_rate_calibration_drift(
            target_date, repository
        )

        # Data quality metrics
        report["tracking_missingness"] = self._compute_tracking_missingness(
            target_date, repository
        )
        report["injury_ingestion_delay"] = self._compute_injury_ingestion_delay(
            target_date, repository
        )
        report["stale_odds"] = self._compute_stale_odds_frequency(
            target_date, repository
        )

        # CLV metrics
        report["clv_by_book"] = self._compute_clv_by_book(target_date, repository)
        report["clv_by_time_bucket"] = self._compute_clv_by_time_bucket(
            target_date, repository
        )

        # Decision metrics
        report["decision_rates"] = self._compute_rejection_and_suspension_rates(
            target_date, repository
        )

        # Archetype breakdown
        report["model_vs_market_by_archetype"] = (
            self._compute_model_vs_market_by_archetype(target_date, repository)
        )

        logger.info("Daily drift report complete for %s", target_date)
        return report

    def check_alerts(self, report: dict) -> list[str]:
        """Check a daily report against configured thresholds.

        Parameters
        ----------
        report : dict
            The report returned by :meth:`compute_daily_report`.

        Returns
        -------
        list[str]
            Human-readable alert messages for every threshold breach.
        """
        alerts: list[str] = []

        # Minutes MAE drift
        minutes = report.get("minutes_mae", {})
        drift_pct = minutes.get("minutes_mae_drift_pct")
        if drift_pct is not None and drift_pct > self.thresholds["minutes_mae_drift_pct"]:
            alerts.append(
                f"ALERT: Minutes MAE drifted {drift_pct:.1%} above training baseline "
                f"(recent={minutes.get('minutes_mae_recent')}, "
                f"training={minutes.get('minutes_mae_training')})"
            )

        # 3PA residual drift
        three_pa = report.get("three_pa_residual", {})
        drift_pct = three_pa.get("three_pa_residual_drift_pct")
        if (
            drift_pct is not None
            and drift_pct > self.thresholds["three_pa_residual_drift_pct"]
        ):
            alerts.append(
                f"ALERT: 3PA residual std drifted {drift_pct:.1%} above training "
                f"(recent={three_pa.get('three_pa_residual_std_recent')}, "
                f"training={three_pa.get('three_pa_residual_std_training')})"
            )

        # Make rate calibration
        make_rate = report.get("make_rate_calibration", {})
        cal_gap = make_rate.get("make_rate_calibration_gap")
        if (
            cal_gap is not None
            and cal_gap > self.thresholds["make_rate_calibration_drift"]
        ):
            alerts.append(
                f"ALERT: Make-rate calibration gap is {cal_gap:.4f}, "
                f"exceeding threshold {self.thresholds['make_rate_calibration_drift']}"
            )

        # Tracking missingness
        tracking = report.get("tracking_missingness", {})
        miss_rate = tracking.get("tracking_missingness_rate")
        if (
            miss_rate is not None
            and miss_rate > self.thresholds["tracking_missingness_rate"]
        ):
            alerts.append(
                f"ALERT: Tracking data missingness at {miss_rate:.1%} "
                f"({tracking.get('tracking_missing_rows')}/{tracking.get('tracking_total_rows')} rows)"
            )

        # Injury ingestion delay
        injury = report.get("injury_ingestion_delay", {})
        avg_delay = injury.get("injury_ingestion_delay_avg_sec")
        if (
            avg_delay is not None
            and avg_delay > self.thresholds["injury_ingestion_delay_avg_sec"]
        ):
            alerts.append(
                f"ALERT: Avg injury ingestion delay is {avg_delay:.0f}s "
                f"(threshold={self.thresholds['injury_ingestion_delay_avg_sec']}s)"
            )

        # Stale odds
        stale = report.get("stale_odds", {})
        stale_freq = stale.get("stale_odds_frequency")
        if (
            stale_freq is not None
            and stale_freq > self.thresholds["stale_odds_frequency"]
        ):
            alerts.append(
                f"ALERT: Stale odds frequency at {stale_freq:.1%} "
                f"({stale.get('stale_odds_count')}/{stale.get('odds_snapshot_count')} snapshots)"
            )

        # Rejection rate
        decisions = report.get("decision_rates", {})
        rej_rate = decisions.get("rejection_rate")
        if rej_rate is not None and rej_rate > self.thresholds["rejection_rate"]:
            alerts.append(
                f"ALERT: Rejection rate at {rej_rate:.1%} "
                f"(threshold={self.thresholds['rejection_rate']:.1%}, "
                f"total evaluated={decisions.get('total_evaluated')})"
            )

        # Suspension rate
        susp_rate = decisions.get("suspension_rate")
        if susp_rate is not None and susp_rate > self.thresholds["suspension_rate"]:
            alerts.append(
                f"ALERT: Suspension rate at {susp_rate:.1%} "
                f"(threshold={self.thresholds['suspension_rate']:.1%})"
            )

        # CLV per book - flag any book with negative mean CLV
        clv_by_book = report.get("clv_by_book", {})
        for book, mean_clv in clv_by_book.items():
            if mean_clv < self.thresholds["min_clv_by_book"]:
                alerts.append(
                    f"ALERT: Negative CLV for sportsbook '{book}': "
                    f"mean CLV = {mean_clv:.4f} "
                    f"(threshold={self.thresholds['min_clv_by_book']})"
                )

        if alerts:
            logger.warning(
                "Drift monitor raised %d alert(s) for %s",
                len(alerts),
                report.get("report_date", "unknown"),
            )
        else:
            logger.info(
                "No drift alerts for %s", report.get("report_date", "unknown")
            )

        return alerts
