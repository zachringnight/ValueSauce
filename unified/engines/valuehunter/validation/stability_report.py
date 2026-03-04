"""7-day unattended pipeline stability report for NBA 3PM Props Engine.

Verifies the pipeline ran cleanly for 7 consecutive days with no manual
intervention required.  Checks ingestion completeness, prediction
generation, null-field integrity, timing discipline, and optional drift
monitoring across each day in the window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DailyHealth:
    """Health summary for a single pipeline day."""

    date: str
    is_game_day: bool
    n_games: int
    n_players_scored: int
    n_decisions: int
    n_bets: int
    ingestion_complete: bool
    predictions_complete: bool
    no_null_fields: bool
    timing_ok: bool
    drift_alerts: list[str] = field(default_factory=list)
    status: str = "PASS"  # "PASS", "WARN", "FAIL"
    issues: list[str] = field(default_factory=list)


@dataclass
class StabilityResults:
    """Aggregated stability report across the full window."""

    daily_reports: list[DailyHealth] = field(default_factory=list)
    n_days: int = 0
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0
    consecutive_clean_days: int = 0
    meets_7day_requirement: bool = False
    no_manual_intervention: bool = True
    total_games: int = 0
    total_players_scored: int = 0
    total_decisions: int = 0
    total_bets: int = 0
    avg_decisions_per_game_day: float = 0.0
    drift_summary: dict = field(default_factory=dict)
    summary_table: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stability report
# ---------------------------------------------------------------------------

class StabilityReport:
    """Verify the pipeline ran cleanly for *n* consecutive days.

    Parameters
    ----------
    repository
        Database access layer (``nba_props.utils.db.Repository``).
    drift_monitor : optional
        An instance of ``nba_props.monitoring.drift.DriftMonitor``.
        When provided the report will include per-day drift checks.
    """

    def __init__(self, repository, drift_monitor=None):
        self.repository = repository
        self.drift_monitor = drift_monitor

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, end_date: date, n_days: int = 7) -> StabilityResults:
        """Run the stability report over the last *n_days*.

        Parameters
        ----------
        end_date : date
            The final date (inclusive) of the stability window.
        n_days : int
            Number of consecutive days to evaluate (default 7).

        Returns
        -------
        StabilityResults
        """
        logger.info(
            "Running stability report for %d days ending %s",
            n_days,
            end_date.isoformat(),
        )

        start_date = end_date - timedelta(days=n_days - 1)
        daily_reports: list[DailyHealth] = []

        for offset in range(n_days):
            target_date = start_date + timedelta(days=offset)
            daily = self._evaluate_day(target_date)
            daily_reports.append(daily)

        # Aggregate totals
        n_pass = sum(1 for d in daily_reports if d.status == "PASS")
        n_warn = sum(1 for d in daily_reports if d.status == "WARN")
        n_fail = sum(1 for d in daily_reports if d.status == "FAIL")

        consecutive_clean = self._max_consecutive_pass(daily_reports)
        game_days = [d for d in daily_reports if d.is_game_day]
        total_games = sum(d.n_games for d in daily_reports)
        total_players = sum(d.n_players_scored for d in daily_reports)
        total_decisions = sum(d.n_decisions for d in daily_reports)
        total_bets = sum(d.n_bets for d in daily_reports)
        avg_decisions = (
            total_decisions / len(game_days) if game_days else 0.0
        )

        # Build drift summary (max value observed for each metric)
        drift_summary = self._aggregate_drift(daily_reports)

        results = StabilityResults(
            daily_reports=daily_reports,
            n_days=n_days,
            n_pass=n_pass,
            n_warn=n_warn,
            n_fail=n_fail,
            consecutive_clean_days=consecutive_clean,
            meets_7day_requirement=consecutive_clean >= 7,
            no_manual_intervention=n_fail == 0,
            total_games=total_games,
            total_players_scored=total_players,
            total_decisions=total_decisions,
            total_bets=total_bets,
            avg_decisions_per_game_day=round(avg_decisions, 1),
            drift_summary=drift_summary,
            summary_table=self._build_summary_table(daily_reports),
        )

        logger.info(
            "Stability report complete: %d PASS / %d WARN / %d FAIL, "
            "consecutive clean = %d, meets requirement = %s",
            n_pass,
            n_warn,
            n_fail,
            consecutive_clean,
            results.meets_7day_requirement,
        )

        return results

    # ------------------------------------------------------------------
    # Per-day evaluation
    # ------------------------------------------------------------------

    def _evaluate_day(self, target_date: date) -> DailyHealth:
        """Evaluate pipeline health for a single day."""
        issues: list[str] = []
        drift_alerts: list[str] = []

        # a. Ingestion check
        ingestion_ok, ingestion_issues = self._check_ingestion(target_date)
        issues.extend(ingestion_issues)

        # Determine if this was a game day
        n_games = self._count_games(target_date)
        is_game_day = n_games > 0

        # Counts
        n_players_scored = self._count_players_scored(target_date)
        n_decisions = self._count_decisions(target_date)
        n_bets = self._count_bets(target_date)

        # b. Predictions check (only relevant on game days)
        if is_game_day:
            predictions_ok, pred_issues = self._check_predictions(target_date)
            issues.extend(pred_issues)
        else:
            predictions_ok = True

        # c. Null-field integrity
        null_ok, null_issues = self._check_null_fields(target_date)
        issues.extend(null_issues)

        # d. Timing discipline
        timing_ok, timing_issues = self._check_timing(target_date)
        issues.extend(timing_issues)

        # e. Drift monitoring (optional)
        if self.drift_monitor is not None and is_game_day:
            try:
                daily_drift_report = self.drift_monitor.compute_daily_report(
                    target_date, self.repository
                )
                drift_alerts = self.drift_monitor.check_alerts(daily_drift_report)
            except Exception as exc:
                logger.warning(
                    "Drift monitor failed for %s: %s", target_date, exc
                )
                drift_alerts = [f"Drift monitor error: {exc}"]

        # f. Determine status
        status = self._determine_status(
            is_game_day=is_game_day,
            ingestion_ok=ingestion_ok,
            predictions_ok=predictions_ok,
            null_ok=null_ok,
            timing_ok=timing_ok,
            drift_alerts=drift_alerts,
        )

        return DailyHealth(
            date=target_date.isoformat(),
            is_game_day=is_game_day,
            n_games=n_games,
            n_players_scored=n_players_scored,
            n_decisions=n_decisions,
            n_bets=n_bets,
            ingestion_complete=ingestion_ok,
            predictions_complete=predictions_ok,
            no_null_fields=null_ok,
            timing_ok=timing_ok,
            drift_alerts=drift_alerts,
            status=status,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Check helpers
    # ------------------------------------------------------------------

    def _check_ingestion(
        self, target_date: date
    ) -> tuple[bool, list[str]]:
        """Verify ingestion completeness for *target_date*.

        Checks that games, player_game rows, injury_reports, and
        odds_props were captured.

        Returns
        -------
        (complete, issues)
        """
        issues: list[str] = []

        # Games
        n_games = self._count_games(target_date)

        # Player games
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM player_games pg
            JOIN games g ON g.nba_game_id = pg.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        n_player_games = rows[0]["cnt"] if rows else 0
        if n_games > 0 and n_player_games == 0:
            issues.append(
                f"No player_game rows for {target_date} despite "
                f"{n_games} game(s)"
            )

        # Injury reports
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM injury_snapshots i
            JOIN games g ON g.nba_game_id = i.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        n_injury = rows[0]["cnt"] if rows else 0
        if n_games > 0 and n_injury == 0:
            issues.append(
                f"No injury_reports ingested for {target_date}"
            )

        # Odds props
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM odds_props op
            JOIN games g ON g.nba_game_id = op.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        n_odds = rows[0]["cnt"] if rows else 0
        if n_games > 0 and n_odds == 0:
            issues.append(
                f"No odds_props captured for {target_date}"
            )

        complete = len(issues) == 0
        return complete, issues

    def _check_predictions(
        self, target_date: date
    ) -> tuple[bool, list[str]]:
        """Verify predictions were generated for *target_date*.

        Checks feature_snapshots and bet_decisions tables.

        Returns
        -------
        (complete, issues)
        """
        issues: list[str] = []

        # Feature snapshots
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM feature_snapshots fs
            JOIN games g ON g.nba_game_id = fs.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        n_snapshots = rows[0]["cnt"] if rows else 0
        if n_snapshots == 0:
            issues.append(
                f"No feature_snapshots for {target_date}"
            )

        # Bet decisions
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        n_decisions = rows[0]["cnt"] if rows else 0
        if n_decisions == 0:
            issues.append(
                f"No bet_decisions for {target_date}"
            )

        # Orphaned feature snapshots (no corresponding decision)
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM feature_snapshots fs
            JOIN games g ON g.nba_game_id = fs.nba_game_id
            LEFT JOIN bet_decisions bd
              ON bd.feature_snapshot_id = fs.feature_snapshot_id
            WHERE g.game_date = %s
              AND bd.decision_id IS NULL
            """,
            (target_date,),
        )
        n_orphans = rows[0]["cnt"] if rows else 0
        if n_orphans > 0:
            issues.append(
                f"{n_orphans} orphaned feature_snapshots with no "
                f"corresponding bet_decision on {target_date}"
            )

        complete = len(issues) == 0
        return complete, issues

    def _check_null_fields(
        self, target_date: date
    ) -> tuple[bool, list[str]]:
        """Check for NULLs in critical decision fields.

        - No NULL model_p_over or model_p_under in decisions
        - No decisions with missing feature_snapshot_id

        Returns
        -------
        (ok, issues)
        """
        issues: list[str] = []

        # NULL model probabilities
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
              AND (bd.model_p_over IS NULL OR bd.model_p_under IS NULL)
            """,
            (target_date,),
        )
        n_null_probs = rows[0]["cnt"] if rows else 0
        if n_null_probs > 0:
            issues.append(
                f"{n_null_probs} decision(s) with NULL model_p_over "
                f"or model_p_under on {target_date}"
            )

        # Missing feature_snapshot_id
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
              AND bd.feature_snapshot_id IS NULL
            """,
            (target_date,),
        )
        n_missing_fs = rows[0]["cnt"] if rows else 0
        if n_missing_fs > 0:
            issues.append(
                f"{n_missing_fs} decision(s) with NULL "
                f"feature_snapshot_id on {target_date}"
            )

        ok = len(issues) == 0
        return ok, issues

    def _check_timing(
        self, target_date: date
    ) -> tuple[bool, list[str]]:
        """Ensure decisions were generated before tipoff.

        - decision_timestamp < tipoff
        - feature freeze timestamps within expected windows

        Returns
        -------
        (ok, issues)
        """
        issues: list[str] = []

        # Decisions after tipoff
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
              AND bd.decision_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
              AND bd.decision_timestamp_utc > g.tipoff_time_utc
            """,
            (target_date,),
        )
        n_late = rows[0]["cnt"] if rows else 0
        if n_late > 0:
            issues.append(
                f"{n_late} decision(s) generated AFTER tipoff on {target_date}"
            )

        # Feature freeze timestamps unreasonably early (> 24h before tipoff)
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM feature_snapshots fs
            JOIN games g ON g.nba_game_id = fs.nba_game_id
            WHERE g.game_date = %s
              AND fs.snapshot_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
              AND (
                  g.tipoff_time_utc - fs.snapshot_timestamp_utc
                  > INTERVAL '24 hours'
              )
            """,
            (target_date,),
        )
        n_early = rows[0]["cnt"] if rows else 0
        if n_early > 0:
            issues.append(
                f"{n_early} feature_snapshot(s) frozen > 24h before "
                f"tipoff on {target_date}"
            )

        # Feature freeze timestamps after tipoff
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM feature_snapshots fs
            JOIN games g ON g.nba_game_id = fs.nba_game_id
            WHERE g.game_date = %s
              AND fs.snapshot_timestamp_utc IS NOT NULL
              AND g.tipoff_time_utc IS NOT NULL
              AND fs.snapshot_timestamp_utc > g.tipoff_time_utc
            """,
            (target_date,),
        )
        n_post_tip = rows[0]["cnt"] if rows else 0
        if n_post_tip > 0:
            issues.append(
                f"{n_post_tip} feature_snapshot(s) frozen AFTER tipoff "
                f"on {target_date}"
            )

        ok = len(issues) == 0
        return ok, issues

    # ------------------------------------------------------------------
    # Count helpers
    # ------------------------------------------------------------------

    def _count_games(self, target_date: date) -> int:
        rows = self.repository._fetchall(
            "SELECT COUNT(*) AS cnt FROM games WHERE game_date = %s",
            (target_date,),
        )
        return rows[0]["cnt"] if rows else 0

    def _count_players_scored(self, target_date: date) -> int:
        rows = self.repository._fetchall(
            """
            SELECT COUNT(DISTINCT fs.nba_player_id) AS cnt
            FROM feature_snapshots fs
            JOIN games g ON g.nba_game_id = fs.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        return rows[0]["cnt"] if rows else 0

    def _count_decisions(self, target_date: date) -> int:
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
            """,
            (target_date,),
        )
        return rows[0]["cnt"] if rows else 0

    def _count_bets(self, target_date: date) -> int:
        rows = self.repository._fetchall(
            """
            SELECT COUNT(*) AS cnt
            FROM bet_decisions bd
            JOIN games g ON g.nba_game_id = bd.nba_game_id
            WHERE g.game_date = %s
              AND bd.recommended_side != 'no_bet'
            """,
            (target_date,),
        )
        return rows[0]["cnt"] if rows else 0

    # ------------------------------------------------------------------
    # Status determination
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_status(
        is_game_day: bool,
        ingestion_ok: bool,
        predictions_ok: bool,
        null_ok: bool,
        timing_ok: bool,
        drift_alerts: list[str],
    ) -> str:
        """Derive day-level status: PASS / WARN / FAIL.

        Rules:
        - FAIL if any critical check fails (ingestion, predictions on a
          game day, null fields, or timing).
        - WARN if drift alerts are present but no critical failures.
        - PASS otherwise.
        """
        if not ingestion_ok:
            return "FAIL"
        if is_game_day and not predictions_ok:
            return "FAIL"
        if not null_ok:
            return "FAIL"
        if not timing_ok:
            return "FAIL"
        if drift_alerts:
            return "WARN"
        return "PASS"

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_consecutive_pass(reports: list[DailyHealth]) -> int:
        """Return the longest run of consecutive PASS days."""
        max_run = 0
        current = 0
        for r in reports:
            if r.status == "PASS":
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    @staticmethod
    def _aggregate_drift(reports: list[DailyHealth]) -> dict:
        """Collect all unique drift alert strings across the window.

        Returns a dict keyed by a short metric label with the count of
        alerts observed across the window.
        """
        alert_counts: dict[str, int] = {}
        for r in reports:
            for alert in r.drift_alerts:
                # Use first word after "ALERT:" as category key
                key = alert.split(":")[0].strip() if ":" in alert else alert
                alert_counts[key] = alert_counts.get(key, 0) + 1
        return alert_counts

    # ------------------------------------------------------------------
    # Table / report generation
    # ------------------------------------------------------------------

    def generate_table(self) -> list[dict]:
        """Generate a summary table (requires ``run()`` to be called first).

        Returns a list-of-dicts suitable for rendering as a table.  Each
        row represents one day, with a final summary row appended.
        """
        # This is a convenience alias; the real work is in _build_summary_table.
        # If results have already been produced they are stored in the last
        # StabilityResults returned by run().  For standalone usage, callers
        # should use StabilityResults.summary_table directly.
        raise NotImplementedError(
            "Call run() first and access StabilityResults.summary_table"
        )

    @staticmethod
    def _build_summary_table(reports: list[DailyHealth]) -> list[dict]:
        """Build the per-day summary table with a trailing summary row."""
        rows: list[dict] = []
        for r in reports:
            rows.append({
                "date": r.date,
                "status": r.status,
                "games": r.n_games,
                "players_scored": r.n_players_scored,
                "decisions": r.n_decisions,
                "bets": r.n_bets,
                "ingestion": "OK" if r.ingestion_complete else "FAIL",
                "predictions": "OK" if r.predictions_complete else "FAIL",
                "timing": "OK" if r.timing_ok else "FAIL",
                "drift_alerts": len(r.drift_alerts),
            })

        # Summary row
        n_reports = len(reports)
        if n_reports > 0:
            rows.append({
                "date": "TOTAL",
                "status": "",
                "games": sum(r.n_games for r in reports),
                "players_scored": sum(r.n_players_scored for r in reports),
                "decisions": sum(r.n_decisions for r in reports),
                "bets": sum(r.n_bets for r in reports),
                "ingestion": sum(
                    1 for r in reports if r.ingestion_complete
                ),
                "predictions": sum(
                    1 for r in reports if r.predictions_complete
                ),
                "timing": sum(1 for r in reports if r.timing_ok),
                "drift_alerts": sum(
                    len(r.drift_alerts) for r in reports
                ),
            })

        return rows

    def format_report(self, results: StabilityResults) -> str:
        """Produce a human-readable text report.

        Parameters
        ----------
        results : StabilityResults
            The results object returned by :meth:`run`.

        Returns
        -------
        str
            Multi-line formatted report string.
        """
        lines: list[str] = []

        lines.append("=" * 72)
        lines.append("  7-DAY PIPELINE STABILITY REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"  Window          : {results.n_days} days")
        if results.daily_reports:
            lines.append(
                f"  Date range      : {results.daily_reports[0].date} "
                f"through {results.daily_reports[-1].date}"
            )
        lines.append(
            f"  Overall verdict : "
            f"{'PASS' if results.meets_7day_requirement else 'FAIL'}"
        )
        lines.append("")

        # Summary counts
        lines.append("  STATUS BREAKDOWN")
        lines.append(f"    PASS : {results.n_pass}")
        lines.append(f"    WARN : {results.n_warn}")
        lines.append(f"    FAIL : {results.n_fail}")
        lines.append(
            f"    Max consecutive clean : {results.consecutive_clean_days}"
        )
        lines.append(
            f"    No manual intervention : {results.no_manual_intervention}"
        )
        lines.append("")

        # Volume
        lines.append("  VOLUME")
        lines.append(f"    Total games          : {results.total_games}")
        lines.append(
            f"    Total players scored  : {results.total_players_scored}"
        )
        lines.append(f"    Total decisions      : {results.total_decisions}")
        lines.append(f"    Total bets           : {results.total_bets}")
        lines.append(
            f"    Avg decisions/game day: {results.avg_decisions_per_game_day}"
        )
        lines.append("")

        # Per-day table
        lines.append("  DAILY DETAIL")
        lines.append(
            "  {:<12} {:<6} {:>5} {:>8} {:>9} {:>5} {:>9} {:>11} {:>6} {:>6}".format(
                "Date", "Stat", "Games", "Players", "Decisions", "Bets",
                "Ingest", "Predict", "Time", "Drift",
            )
        )
        lines.append("  " + "-" * 88)

        for r in results.daily_reports:
            lines.append(
                "  {:<12} {:<6} {:>5} {:>8} {:>9} {:>5} {:>9} {:>11} {:>6} {:>6}".format(
                    r.date,
                    r.status,
                    r.n_games,
                    r.n_players_scored,
                    r.n_decisions,
                    r.n_bets,
                    "OK" if r.ingestion_complete else "FAIL",
                    "OK" if r.predictions_complete else "FAIL",
                    "OK" if r.timing_ok else "FAIL",
                    len(r.drift_alerts),
                )
            )
        lines.append("")

        # Issues
        any_issues = any(d.issues for d in results.daily_reports)
        if any_issues:
            lines.append("  ISSUES")
            for r in results.daily_reports:
                if r.issues:
                    for issue in r.issues:
                        lines.append(f"    [{r.date}] {issue}")
            lines.append("")

        # Drift summary
        if results.drift_summary:
            lines.append("  DRIFT SUMMARY")
            for metric, count in sorted(results.drift_summary.items()):
                lines.append(f"    {metric}: {count} alert(s)")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)
