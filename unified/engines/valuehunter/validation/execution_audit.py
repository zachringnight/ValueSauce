"""Execution realism audit for NBA 3PM Props Engine validation pack.

Verifies that no stale-line, suspended-line, or post-freeze information
leakage exists in any bet decision. Part of the release-candidate
validation suite (Section K6+).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import numpy as np

from ..backtest.leakage import LeakageDetector
from ..backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)

# Injury report deadline windows (local arena time)
DAY_BEFORE_DEADLINE_HOUR = 17   # 5:00 PM local, day before
B2B_DEADLINE_HOUR = 13          # 1:00 PM local, gameday (back-to-back)
GAMEDAY_WINDOW_START_HOUR = 11  # 11:00 AM local
GAMEDAY_WINDOW_END_HOUR = 13    # 1:00 PM local

DEFAULT_MAX_STALE_ODDS_MINUTES = 30


@dataclass
class ExecutionAuditResults:
    """Results container for the execution realism audit."""

    total_decisions_audited: int = 0
    total_leakage_violations: int = 0
    total_stale_odds: int = 0
    total_stale_injuries: int = 0
    total_suspended_lines: int = 0
    total_post_freeze_refs: int = 0
    leakage_details: list[dict] = field(default_factory=list)
    stale_odds_details: list[dict] = field(default_factory=list)
    stale_injury_details: list[dict] = field(default_factory=list)
    suspended_line_details: list[dict] = field(default_factory=list)
    passed: bool = True
    summary_table: list[dict] = field(default_factory=list)


class ExecutionRealismAudit:
    """
    Verify that no stale-line, suspended-line, or post-freeze information
    leakage exists in any bet decision.

    Runs the following checks on every bet decision in a date range:
    1. Feature timestamp vs freeze timestamp (LeakageDetector)
    2. Closing line leak detection
    3. Tracking imputation leak detection
    4. Same-day information leak detection
    5. Odds staleness (snapshot age > threshold)
    6. Injury staleness (outdated injury snapshots)
    7. Suspended line detection (line disappeared before fill)
    8. Post-freeze future data references
    """

    def __init__(self, repository, leakage_detector: LeakageDetector):
        """
        Args:
            repository: Data access layer providing bet decisions,
                feature snapshots, odds snapshots, and game info.
            leakage_detector: A LeakageDetector instance from
                nba_props.backtest.leakage.
        """
        self.repository = repository
        self.leakage_detector = leakage_detector
        self._results: Optional[ExecutionAuditResults] = None

    def run(
        self,
        start_date: str,
        end_date: str,
    ) -> ExecutionAuditResults:
        """
        Run the full execution realism audit over a date range.

        Steps:
        1. Fetch all bet decisions in the date range from DB.
        2. For each decision, fetch the corresponding feature snapshot.
        3. Run all leakage checks (feature timestamps, closing line,
           tracking imputation, same-day info).
        4. Check odds staleness (snapshot age > 30 min).
        5. Check injury staleness (outdated snapshots relative to
           injury report windows).
        6. Check for suspended lines.
        7. Verify no future data references post-freeze.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            ExecutionAuditResults with all violation details and
            pass/fail status.
        """
        logger.info(
            "Starting execution realism audit from %s to %s",
            start_date, end_date,
        )

        results = ExecutionAuditResults()

        bet_decisions = self.repository.get_bet_decisions_in_range(
            start_date, end_date
        )
        results.total_decisions_audited = len(bet_decisions)

        if not bet_decisions:
            logger.warning("No bet decisions found in range %s to %s", start_date, end_date)
            results.summary_table = self.generate_table()
            self._results = results
            return results

        logger.info("Auditing %d bet decisions", len(bet_decisions))

        # Group decisions by game_date for same-day leak checks
        decisions_by_date: dict[str, list[dict]] = {}
        for decision in bet_decisions:
            game_date = decision.get("game_date", "")
            decisions_by_date.setdefault(game_date, []).append(decision)

        for decision in bet_decisions:
            game_id = decision.get("game_id")
            player_id = decision.get("player_id")
            feature_snapshot_id = decision.get("feature_snapshot_id")
            decision_timestamp = decision.get("decision_timestamp")
            freeze_timestamp = decision.get("freeze_timestamp")

            # Fetch the feature snapshot associated with this decision
            feature_snapshot = self.repository.get_feature_snapshot(
                feature_snapshot_id
            )
            if feature_snapshot is None:
                logger.warning(
                    "No feature snapshot found for decision game_id=%s "
                    "player_id=%s snapshot_id=%s",
                    game_id, player_id, feature_snapshot_id,
                )
                continue

            # Fetch game info for injury window checks
            game_info = self.repository.get_game_info(game_id)

            # -----------------------------------------------------------
            # 1-4. Leakage checks via LeakageDetector
            # -----------------------------------------------------------
            freeze_dt = self._parse_timestamp(freeze_timestamp)
            if freeze_dt is not None:
                # Feature timestamps vs freeze timestamp
                ts_violations = self.leakage_detector.check_feature_timestamps(
                    feature_snapshot, freeze_dt
                )
                for v in ts_violations:
                    results.leakage_details.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "type": "feature_timestamp_vs_freeze",
                        "detail": v,
                    })

                # Closing line leak detection
                closing_violations = self.leakage_detector.check_closing_line_leak(
                    feature_snapshot
                )
                for v in closing_violations:
                    results.leakage_details.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "type": "closing_line_leak",
                        "detail": v,
                    })

                # Tracking imputation leak detection
                tracking_violations = self.leakage_detector.check_tracking_imputation(
                    feature_snapshot
                )
                for v in tracking_violations:
                    results.leakage_details.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "type": "tracking_imputation_leak",
                        "detail": v,
                    })

            # Same-day information leak detection
            game_date = decision.get("game_date", "")
            same_day_rows = decisions_by_date.get(game_date, [])
            if same_day_rows and freeze_dt is not None:
                same_day_snapshots = []
                for row_decision in same_day_rows:
                    row_snap_id = row_decision.get("feature_snapshot_id")
                    row_snap = self.repository.get_feature_snapshot(row_snap_id)
                    if row_snap is not None:
                        row_snap["freeze_timestamp_utc"] = row_decision.get(
                            "freeze_timestamp"
                        )
                        same_day_snapshots.append(row_snap)

                sameday_violations = self.leakage_detector.check_same_day_leak(
                    same_day_snapshots, game_date
                )
                for v in sameday_violations:
                    results.leakage_details.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "type": "same_day_info_leak",
                        "detail": v,
                    })

            # -----------------------------------------------------------
            # 5. Check odds staleness
            # -----------------------------------------------------------
            stale_odds_violations = self._check_odds_staleness(
                decision, feature_snapshot,
                max_stale_minutes=DEFAULT_MAX_STALE_ODDS_MINUTES,
            )
            results.stale_odds_details.extend(stale_odds_violations)

            # -----------------------------------------------------------
            # 6. Check injury staleness
            # -----------------------------------------------------------
            stale_injury_violations = self._check_injury_staleness(
                decision, feature_snapshot, game_info
            )
            results.stale_injury_details.extend(stale_injury_violations)

            # -----------------------------------------------------------
            # 7. Check suspended lines
            # -----------------------------------------------------------
            odds_snapshots = self.repository.get_odds_snapshots_for_game(
                player_id=player_id, game_id=game_id
            )
            suspended_violations = self._check_suspended_lines(
                decision, odds_snapshots
            )
            results.suspended_line_details.extend(suspended_violations)

            # -----------------------------------------------------------
            # 8. Verify no future data references (post-freeze)
            # -----------------------------------------------------------
            if freeze_dt is not None:
                post_freeze_violations = self._check_post_freeze_references(
                    decision, feature_snapshot, freeze_dt
                )
                for v in post_freeze_violations:
                    results.leakage_details.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "type": "post_freeze_reference",
                        "detail": v,
                    })

        # -----------------------------------------------------------
        # Aggregate totals
        # -----------------------------------------------------------
        results.total_leakage_violations = len(results.leakage_details)
        results.total_stale_odds = len(results.stale_odds_details)
        results.total_stale_injuries = len(results.stale_injury_details)
        results.total_suspended_lines = len(results.suspended_line_details)

        # Count post-freeze refs specifically
        results.total_post_freeze_refs = sum(
            1 for d in results.leakage_details
            if d.get("type") == "post_freeze_reference"
        )

        # Determine overall pass/fail
        total_violations = (
            results.total_leakage_violations
            + results.total_stale_odds
            + results.total_stale_injuries
            + results.total_suspended_lines
        )
        results.passed = total_violations == 0

        # Build summary table
        self._results = results
        results.summary_table = self.generate_table()

        logger.info(
            "Execution realism audit complete. Decisions=%d, "
            "Leakage=%d, StaleOdds=%d, StaleInjuries=%d, "
            "Suspended=%d, PostFreeze=%d, Passed=%s",
            results.total_decisions_audited,
            results.total_leakage_violations,
            results.total_stale_odds,
            results.total_stale_injuries,
            results.total_suspended_lines,
            results.total_post_freeze_refs,
            results.passed,
        )

        return results

    def _check_odds_staleness(
        self,
        decision: dict,
        feature_snapshot: dict,
        max_stale_minutes: int = 30,
    ) -> list[dict]:
        """
        Check whether the odds snapshot used in a decision is stale.

        An odds snapshot is considered stale if the time delta between
        the odds_snapshot_timestamp and the decision_timestamp exceeds
        max_stale_minutes.

        Args:
            decision: The bet decision record.
            feature_snapshot: The associated feature snapshot.
            max_stale_minutes: Maximum allowed age of odds snapshot
                in minutes (default 30).

        Returns:
            List of violation dicts. Each contains type, game_id,
            player_id, odds_age_minutes, and threshold.
        """
        violations = []

        odds_ts = (
            feature_snapshot.get("odds_snapshot_timestamp_utc")
            or decision.get("odds_snapshot_timestamp")
        )
        decision_ts = decision.get("decision_timestamp")

        if odds_ts is None or decision_ts is None:
            return violations

        odds_dt = self._parse_timestamp(odds_ts)
        decision_dt = self._parse_timestamp(decision_ts)

        if odds_dt is None or decision_dt is None:
            return violations

        delta = decision_dt - odds_dt
        age_minutes = delta.total_seconds() / 60.0

        if age_minutes > max_stale_minutes:
            violations.append({
                "type": "stale_odds",
                "game_id": decision.get("game_id"),
                "player_id": decision.get("player_id"),
                "odds_age_minutes": round(age_minutes, 2),
                "threshold": max_stale_minutes,
            })
            logger.warning(
                "Stale odds detected: game_id=%s player_id=%s "
                "age=%.1f min (threshold=%d min)",
                decision.get("game_id"),
                decision.get("player_id"),
                age_minutes,
                max_stale_minutes,
            )

        return violations

    def _check_injury_staleness(
        self,
        decision: dict,
        feature_snapshot: dict,
        game_info: Optional[dict],
    ) -> list[dict]:
        """
        Check whether the injury snapshot used in a decision is outdated
        relative to known injury report windows.

        Injury report windows (local arena time):
        - DAY_BEFORE_DEADLINE: 5:00 PM local (day before game)
        - B2B_DEADLINE: 1:00 PM local (gameday, back-to-back)
        - GAMEDAY_WINDOW: 11:00 AM - 1:00 PM local (gameday)

        If a gameday injury report was available but the decision used
        a day-before snapshot, it is flagged.

        Args:
            decision: The bet decision record.
            feature_snapshot: The associated feature snapshot.
            game_info: Game metadata including tipoff time and
                timezone offset.

        Returns:
            List of violation dicts with staleness details.
        """
        violations = []

        if game_info is None:
            return violations

        injury_ts = (
            feature_snapshot.get("injury_snapshot_timestamp_utc")
            or decision.get("injury_snapshot_timestamp")
        )
        decision_ts = decision.get("decision_timestamp")

        if injury_ts is None or decision_ts is None:
            return violations

        injury_dt = self._parse_timestamp(injury_ts)
        decision_dt = self._parse_timestamp(decision_ts)

        if injury_dt is None or decision_dt is None:
            return violations

        # Determine game date and local timezone offset
        game_date_str = game_info.get("game_date") or decision.get("game_date")
        local_tz_offset = game_info.get("local_tz_offset_hours", 0)

        if game_date_str is None:
            return violations

        try:
            game_date_dt = datetime.strptime(game_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return violations

        # Compute gameday injury window boundaries in UTC
        gameday_window_start_local = game_date_dt.replace(
            hour=GAMEDAY_WINDOW_START_HOUR, minute=0, second=0, microsecond=0
        )
        gameday_window_end_local = game_date_dt.replace(
            hour=GAMEDAY_WINDOW_END_HOUR, minute=0, second=0, microsecond=0
        )
        gameday_window_start_utc = gameday_window_start_local - timedelta(
            hours=local_tz_offset
        )
        gameday_window_end_utc = gameday_window_end_local - timedelta(
            hours=local_tz_offset
        )

        # Day-before deadline in UTC
        day_before = game_date_dt - timedelta(days=1)
        day_before_deadline_local = day_before.replace(
            hour=DAY_BEFORE_DEADLINE_HOUR, minute=0, second=0, microsecond=0
        )
        day_before_deadline_utc = day_before_deadline_local - timedelta(
            hours=local_tz_offset
        )

        # B2B deadline in UTC
        b2b_deadline_local = game_date_dt.replace(
            hour=B2B_DEADLINE_HOUR, minute=0, second=0, microsecond=0
        )
        b2b_deadline_utc = b2b_deadline_local - timedelta(
            hours=local_tz_offset
        )

        # If the decision was made after the gameday window opened,
        # the injury snapshot should be from the gameday window
        # (not from day-before).
        if decision_dt >= gameday_window_start_utc:
            # Decision is during or after the gameday injury window.
            # The injury snapshot should be at least as recent as the
            # gameday window start.
            if injury_dt < gameday_window_start_utc:
                violations.append({
                    "type": "stale_injury",
                    "game_id": decision.get("game_id"),
                    "player_id": decision.get("player_id"),
                    "injury_snapshot_timestamp": str(injury_dt),
                    "decision_timestamp": str(decision_dt),
                    "expected_window": "gameday",
                    "gameday_window_start_utc": str(gameday_window_start_utc),
                    "detail": (
                        "Decision made after gameday injury window opened "
                        "but injury snapshot is from day-before."
                    ),
                })
                logger.warning(
                    "Stale injury snapshot: game_id=%s player_id=%s "
                    "injury_ts=%s < gameday_window_start=%s",
                    decision.get("game_id"),
                    decision.get("player_id"),
                    injury_dt,
                    gameday_window_start_utc,
                )

        # If this is a B2B game and decision was after B2B deadline,
        # check that injury snapshot is at least from B2B deadline.
        is_b2b = game_info.get("is_b2b", False)
        if is_b2b and decision_dt >= b2b_deadline_utc:
            if injury_dt < b2b_deadline_utc:
                violations.append({
                    "type": "stale_injury_b2b",
                    "game_id": decision.get("game_id"),
                    "player_id": decision.get("player_id"),
                    "injury_snapshot_timestamp": str(injury_dt),
                    "decision_timestamp": str(decision_dt),
                    "expected_window": "b2b_deadline",
                    "b2b_deadline_utc": str(b2b_deadline_utc),
                    "detail": (
                        "B2B game: decision made after B2B deadline but "
                        "injury snapshot predates the B2B deadline."
                    ),
                })

        return violations

    def _check_suspended_lines(
        self,
        decision: dict,
        odds_snapshots: Optional[list[dict]],
    ) -> list[dict]:
        """
        Check whether a line was suspended between the odds snapshot
        used in the decision and game tip-off.

        If a line was available at snapshot time but is missing from
        a later snapshot before tip, it is flagged as potentially
        suspended.

        Args:
            decision: The bet decision record.
            odds_snapshots: All odds snapshots for this player/game,
                sorted chronologically.

        Returns:
            List of violation dicts for suspended lines.
        """
        violations = []

        if not odds_snapshots:
            return violations

        odds_ts = decision.get("odds_snapshot_timestamp")
        decision_ts = decision.get("decision_timestamp")

        if odds_ts is None or decision_ts is None:
            return violations

        odds_dt = self._parse_timestamp(odds_ts)
        decision_dt = self._parse_timestamp(decision_ts)

        if odds_dt is None or decision_dt is None:
            return violations

        # Sort snapshots by timestamp
        sorted_snapshots = sorted(
            odds_snapshots,
            key=lambda s: self._parse_timestamp(
                s.get("snapshot_timestamp_utc", "")
            ) or datetime.min,
        )

        # Find the snapshot used in the decision
        decision_line = decision.get("line")
        decision_sportsbook = decision.get("sportsbook")

        # Look at snapshots after the decision timestamp
        for snapshot in sorted_snapshots:
            snap_ts = self._parse_timestamp(
                snapshot.get("snapshot_timestamp_utc")
            )
            if snap_ts is None or snap_ts <= odds_dt:
                continue

            # This is a snapshot after the decision's odds snapshot
            snap_line = snapshot.get("line")
            snap_sportsbook = snapshot.get("sportsbook", decision_sportsbook)

            # If sportsbook matches and line is now missing, flag it
            if snap_sportsbook == decision_sportsbook:
                if snap_line is None:
                    violations.append({
                        "type": "suspended_line",
                        "game_id": decision.get("game_id"),
                        "player_id": decision.get("player_id"),
                        "sportsbook": decision_sportsbook,
                        "decision_line": decision_line,
                        "decision_odds_timestamp": str(odds_dt),
                        "suspended_at_timestamp": str(snap_ts),
                        "detail": (
                            f"Line {decision_line} at {decision_sportsbook} "
                            f"was available at {odds_dt} but missing at "
                            f"{snap_ts}."
                        ),
                    })
                    logger.warning(
                        "Suspended line detected: game_id=%s player_id=%s "
                        "sportsbook=%s line=%s available at %s, gone at %s",
                        decision.get("game_id"),
                        decision.get("player_id"),
                        decision_sportsbook,
                        decision_line,
                        odds_dt,
                        snap_ts,
                    )
                    # Only flag the first suspension event
                    break

        return violations

    def _check_post_freeze_references(
        self,
        decision: dict,
        feature_snapshot: dict,
        freeze_dt: datetime,
    ) -> list[str]:
        """
        Verify no future data references exist after the freeze timestamp.

        Checks:
        - All feature data timestamps <= freeze_timestamp
        - No actual game results leaked into features

        Args:
            decision: The bet decision record.
            feature_snapshot: The associated feature snapshot.
            freeze_dt: The freeze timestamp as a datetime.

        Returns:
            List of violation description strings.
        """
        violations = []

        features = feature_snapshot.get("feature_json", {})

        # Check for any embedded timestamp that exceeds freeze
        data_ts = features.get("_latest_data_timestamp")
        if data_ts:
            data_dt = self._parse_timestamp(data_ts)
            if data_dt is not None and data_dt > freeze_dt:
                violations.append(
                    f"Feature _latest_data_timestamp {data_ts} > "
                    f"freeze {freeze_dt}"
                )

        # Check for actual game results leaking into features
        result_leak_keys = [
            "actual_3pm", "actual_3pa", "actual_minutes",
            "game_result", "final_score", "actual_pts",
            "actual_fg3m", "actual_fg3a",
        ]
        for key in result_leak_keys:
            if key in features and features[key] is not None:
                violations.append(
                    f"Actual game result field '{key}' found in features "
                    f"with value {features[key]}. This indicates future "
                    f"data leakage."
                )

        # Check odds and injury snapshot timestamps vs freeze
        odds_ts = feature_snapshot.get("odds_snapshot_timestamp_utc")
        if odds_ts:
            odds_dt = self._parse_timestamp(odds_ts)
            if odds_dt is not None and odds_dt > freeze_dt:
                violations.append(
                    f"Odds snapshot timestamp {odds_ts} > freeze {freeze_dt}"
                )

        injury_ts = feature_snapshot.get("injury_snapshot_timestamp_utc")
        if injury_ts:
            injury_dt = self._parse_timestamp(injury_ts)
            if injury_dt is not None and injury_dt > freeze_dt:
                violations.append(
                    f"Injury snapshot timestamp {injury_ts} > "
                    f"freeze {freeze_dt}"
                )

        return violations

    def generate_table(self) -> list[dict]:
        """
        Generate a summary table of audit results.

        Columns: check_type, total_checked, violations, violation_rate,
        passed.

        Rows: leakage, stale_odds, stale_injuries, suspended_lines,
        post_freeze, TOTAL.

        Returns:
            List of dicts, one per row.
        """
        if self._results is None:
            return []

        r = self._results
        total_checked = r.total_decisions_audited

        # Count non-post-freeze leakage violations
        non_post_freeze_leakage = sum(
            1 for d in r.leakage_details
            if d.get("type") != "post_freeze_reference"
        )

        rows = [
            {
                "check_type": "leakage",
                "total_checked": total_checked,
                "violations": non_post_freeze_leakage,
                "violation_rate": (
                    non_post_freeze_leakage / total_checked
                    if total_checked > 0 else 0.0
                ),
                "passed": non_post_freeze_leakage == 0,
            },
            {
                "check_type": "stale_odds",
                "total_checked": total_checked,
                "violations": r.total_stale_odds,
                "violation_rate": (
                    r.total_stale_odds / total_checked
                    if total_checked > 0 else 0.0
                ),
                "passed": r.total_stale_odds == 0,
            },
            {
                "check_type": "stale_injuries",
                "total_checked": total_checked,
                "violations": r.total_stale_injuries,
                "violation_rate": (
                    r.total_stale_injuries / total_checked
                    if total_checked > 0 else 0.0
                ),
                "passed": r.total_stale_injuries == 0,
            },
            {
                "check_type": "suspended_lines",
                "total_checked": total_checked,
                "violations": r.total_suspended_lines,
                "violation_rate": (
                    r.total_suspended_lines / total_checked
                    if total_checked > 0 else 0.0
                ),
                "passed": r.total_suspended_lines == 0,
            },
            {
                "check_type": "post_freeze",
                "total_checked": total_checked,
                "violations": r.total_post_freeze_refs,
                "violation_rate": (
                    r.total_post_freeze_refs / total_checked
                    if total_checked > 0 else 0.0
                ),
                "passed": r.total_post_freeze_refs == 0,
            },
        ]

        total_violations = (
            non_post_freeze_leakage
            + r.total_stale_odds
            + r.total_stale_injuries
            + r.total_suspended_lines
            + r.total_post_freeze_refs
        )
        rows.append({
            "check_type": "TOTAL",
            "total_checked": total_checked,
            "violations": total_violations,
            "violation_rate": (
                total_violations / total_checked
                if total_checked > 0 else 0.0
            ),
            "passed": total_violations == 0,
        })

        return rows

    def format_report(self) -> str:
        """
        Format a human-readable audit report.

        Returns:
            Multi-line string report with summary table and
            violation details.
        """
        if self._results is None:
            return "No audit results available. Run the audit first."

        r = self._results
        lines = []
        lines.append("=" * 72)
        lines.append("EXECUTION REALISM AUDIT REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"Total decisions audited: {r.total_decisions_audited}")
        lines.append(f"Overall result: {'PASSED' if r.passed else 'FAILED'}")
        lines.append("")

        # Summary table
        lines.append("-" * 72)
        lines.append(
            f"{'Check Type':<20} {'Checked':>10} {'Violations':>12} "
            f"{'Rate':>10} {'Passed':>8}"
        )
        lines.append("-" * 72)

        table = self.generate_table()
        for row in table:
            check = row["check_type"]
            checked = row["total_checked"]
            violations = row["violations"]
            rate = row["violation_rate"]
            passed = "YES" if row["passed"] else "NO"
            lines.append(
                f"{check:<20} {checked:>10} {violations:>12} "
                f"{rate:>9.4f} {passed:>8}"
            )
        lines.append("-" * 72)
        lines.append("")

        # Violation details
        if r.leakage_details:
            lines.append("LEAKAGE VIOLATIONS:")
            for detail in r.leakage_details:
                lines.append(
                    f"  [{detail.get('type')}] game={detail.get('game_id')} "
                    f"player={detail.get('player_id')}: "
                    f"{detail.get('detail')}"
                )
            lines.append("")

        if r.stale_odds_details:
            lines.append("STALE ODDS VIOLATIONS:")
            for detail in r.stale_odds_details:
                lines.append(
                    f"  game={detail.get('game_id')} "
                    f"player={detail.get('player_id')}: "
                    f"age={detail.get('odds_age_minutes')} min "
                    f"(threshold={detail.get('threshold')} min)"
                )
            lines.append("")

        if r.stale_injury_details:
            lines.append("STALE INJURY VIOLATIONS:")
            for detail in r.stale_injury_details:
                lines.append(
                    f"  game={detail.get('game_id')} "
                    f"player={detail.get('player_id')}: "
                    f"{detail.get('detail')}"
                )
            lines.append("")

        if r.suspended_line_details:
            lines.append("SUSPENDED LINE VIOLATIONS:")
            for detail in r.suspended_line_details:
                lines.append(
                    f"  game={detail.get('game_id')} "
                    f"player={detail.get('player_id')}: "
                    f"{detail.get('detail')}"
                )
            lines.append("")

        if r.passed:
            lines.append("All checks passed. No execution realism issues found.")
        else:
            total = (
                r.total_leakage_violations + r.total_stale_odds
                + r.total_stale_injuries + r.total_suspended_lines
            )
            lines.append(
                f"AUDIT FAILED: {total} total violation(s) detected."
            )

        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def _parse_timestamp(ts) -> Optional[datetime]:
        """
        Parse a timestamp that may be a datetime, string, or None.

        Args:
            ts: Timestamp value (datetime, ISO string, or None).

        Returns:
            datetime or None if parsing fails.
        """
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass
            # Try common formats
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt)
                except (ValueError, TypeError):
                    continue
        return None
