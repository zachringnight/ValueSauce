"""100-paper-bet ledger for NBA 3PM Props Engine validation pack.

Maintains and validates a full ledger of at least 100 paper-traded bets
with complete logging and close capture. Part of the release-candidate
validation suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Optional

import numpy as np

from ..backtest.leakage import LeakageDetector
from ..backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)

# Critical fields that must be present for every bet record
CRITICAL_FIELDS = [
    "odds_snapshot_timestamp",
    "injury_snapshot_timestamp",
    "model_version",
    "tracking_available",
    "feature_snapshot_id",
]


@dataclass
class PaperLedgerResults:
    """Results container for the 100-paper-bet ledger."""

    ledger: list[dict] = field(default_factory=list)
    n_bets: int = 0
    n_settled: int = 0
    n_pending: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_pushes: int = 0
    hit_rate: float = 0.0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    avg_edge: float = 0.0
    avg_clv: float = 0.0
    max_drawdown: float = 0.0
    missing_fields_count: int = 0
    missing_fields_details: list[dict] = field(default_factory=list)
    meets_minimum: bool = False
    all_fields_complete: bool = True
    close_capture_rate: float = 0.0
    summary_table: list[dict] = field(default_factory=list)


class PaperBetLedger:
    """
    Maintain and validate a full ledger of at least 100 paper-traded
    bets with complete logging and close capture.

    Assembles every bet decision with its full context: decision fields,
    model state, snapshot references, closing line capture, actual
    results, and CLV calculation.

    Validates completeness of critical fields, closing line capture,
    and settlement status.
    """

    def __init__(self, repository):
        """
        Args:
            repository: Data access layer providing bet decisions,
                feature snapshots, odds snapshots, closing lines,
                and actual results.
        """
        self.repository = repository
        self._results: Optional[PaperLedgerResults] = None

    def build_ledger(
        self,
        start_date: str,
        end_date: str,
        min_bets: int = 100,
    ) -> PaperLedgerResults:
        """
        Build and validate the paper-bet ledger over a date range.

        Steps:
        1. Fetch all bet_decisions where recommended_side != 'no_bet'.
        2. Assemble the complete record for each bet.
        3. Validate completeness (critical fields, closing lines,
           settlement status).
        4. Compute ledger summary statistics.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            min_bets: Minimum number of bets required for the ledger
                to be considered valid (default 100).

        Returns:
            PaperLedgerResults with the full ledger and summary stats.
        """
        logger.info(
            "Building paper-bet ledger from %s to %s (min_bets=%d)",
            start_date, end_date, min_bets,
        )

        results = PaperLedgerResults()

        # ------------------------------------------------------------------
        # 1. Fetch all bet decisions with a real recommendation
        # ------------------------------------------------------------------
        bet_decisions = self.repository.get_bet_decisions_in_range(
            start_date, end_date
        )

        # Filter to only decisions where a bet was recommended
        active_decisions = [
            d for d in bet_decisions
            if d.get("recommended_side") not in (None, "no_bet", "")
        ]

        logger.info(
            "Found %d total decisions, %d with active recommendations",
            len(bet_decisions), len(active_decisions),
        )

        # ------------------------------------------------------------------
        # 2. Assemble complete records for each bet
        # ------------------------------------------------------------------
        ledger = []

        for decision in active_decisions:
            game_id = decision.get("game_id")
            player_id = decision.get("player_id")
            feature_snapshot_id = decision.get("feature_snapshot_id")

            # ---- Decision fields ----
            record = {
                "game_id": game_id,
                "player_id": player_id,
                "sportsbook": decision.get("sportsbook"),
                "line": decision.get("line"),
                "odds_over": decision.get("odds_over"),
                "odds_under": decision.get("odds_under"),
                "model_p_over": decision.get("model_p_over"),
                "model_p_under": decision.get("model_p_under"),
                "fair_odds_over": decision.get("fair_odds_over"),
                "fair_odds_under": decision.get("fair_odds_under"),
                "edge": decision.get("edge"),
                "recommended_side": decision.get("recommended_side"),
                "stake_pct": decision.get("stake_pct"),
                "decision_timestamp": decision.get("decision_timestamp"),
            }

            # ---- Model state ----
            record["model_version"] = decision.get("model_version")
            record["feature_snapshot_id"] = feature_snapshot_id
            record["tracking_available"] = decision.get("tracking_available")

            # ---- Snapshot references ----
            record["injury_snapshot_timestamp"] = decision.get(
                "injury_snapshot_timestamp"
            )
            record["odds_snapshot_timestamp"] = decision.get(
                "odds_snapshot_timestamp"
            )
            record["freeze_timestamp"] = decision.get("freeze_timestamp")

            # ---- Close capture ----
            closing_odds = self.repository.get_closing_odds(
                player_id=player_id, game_id=game_id
            )
            if closing_odds is not None:
                record["close_over_prob_novig"] = closing_odds.get(
                    "close_over_prob_novig"
                )
                record["close_under_prob_novig"] = closing_odds.get(
                    "close_under_prob_novig"
                )
            else:
                record["close_over_prob_novig"] = None
                record["close_under_prob_novig"] = None

            # ---- Result ----
            actual_results = self.repository.get_actual_results(
                player_id=player_id, game_id=game_id
            )

            if actual_results is not None:
                actual_3pm = actual_results.get("three_pm")
                record["actual_3pm"] = actual_3pm

                if actual_3pm is not None:
                    line = record["line"]
                    side = record["recommended_side"]
                    record["bet_result"] = self._determine_bet_result(
                        actual_3pm, line, side
                    )
                    record["pnl_units"] = self._compute_pnl(
                        record["bet_result"],
                        record["stake_pct"],
                        record.get("odds_over") if side == "over" else record.get("odds_under"),
                    )
                else:
                    record["actual_3pm"] = None
                    record["bet_result"] = None
                    record["pnl_units"] = None
            else:
                record["actual_3pm"] = None
                record["bet_result"] = None
                record["pnl_units"] = None

            # ---- CLV ----
            record["clv_prob_pts"] = self._compute_clv(record)

            ledger.append(record)

        # Sort by decision_timestamp
        ledger.sort(
            key=lambda r: r.get("decision_timestamp") or ""
        )

        results.ledger = ledger
        results.n_bets = len(ledger)

        # ------------------------------------------------------------------
        # 3. Validate completeness
        # ------------------------------------------------------------------
        missing_details = []
        for record in ledger:
            missing = self._check_critical_fields(record)
            if missing:
                missing_details.append({
                    "game_id": record.get("game_id"),
                    "player_id": record.get("player_id"),
                    "decision_timestamp": record.get("decision_timestamp"),
                    "missing_fields": missing,
                })

        results.missing_fields_count = len(missing_details)
        results.missing_fields_details = missing_details
        results.all_fields_complete = results.missing_fields_count == 0

        # Close capture rate
        bets_with_close = sum(
            1 for r in ledger
            if r.get("close_over_prob_novig") is not None
            or r.get("close_under_prob_novig") is not None
        )
        results.close_capture_rate = (
            bets_with_close / results.n_bets
            if results.n_bets > 0 else 0.0
        )

        # Check that every settled bet has actual_3pm and bet_result
        for record in ledger:
            if record.get("actual_3pm") is not None and record.get("bet_result") is None:
                logger.warning(
                    "Settled bet missing bet_result: game_id=%s player_id=%s",
                    record.get("game_id"), record.get("player_id"),
                )

        # ------------------------------------------------------------------
        # 4. Compute ledger summary statistics
        # ------------------------------------------------------------------
        settled = [r for r in ledger if r.get("bet_result") is not None]
        pending = [r for r in ledger if r.get("bet_result") is None]

        results.n_settled = len(settled)
        results.n_pending = len(pending)

        results.n_wins = sum(1 for r in settled if r["bet_result"] == "win")
        results.n_losses = sum(1 for r in settled if r["bet_result"] == "loss")
        results.n_pushes = sum(1 for r in settled if r["bet_result"] == "push")

        results.hit_rate = (
            results.n_wins / results.n_settled
            if results.n_settled > 0 else 0.0
        )

        # Total staked and PnL
        stakes = [
            r.get("stake_pct", 0.0) or 0.0
            for r in settled
        ]
        pnls = [
            r.get("pnl_units", 0.0) or 0.0
            for r in settled
        ]

        results.total_staked = sum(stakes)
        results.total_pnl = sum(pnls)
        results.roi = (
            results.total_pnl / results.total_staked
            if results.total_staked > 0 else 0.0
        )

        # Average edge
        edges = [
            r.get("edge", 0.0) or 0.0
            for r in ledger
        ]
        results.avg_edge = float(np.mean(edges)) if edges else 0.0

        # Average CLV
        clvs = [
            r.get("clv_prob_pts", 0.0) or 0.0
            for r in settled
            if r.get("clv_prob_pts") is not None
        ]
        results.avg_clv = float(np.mean(clvs)) if clvs else 0.0

        # Max drawdown
        if pnls:
            results.max_drawdown = self._compute_drawdown(pnls)
        else:
            results.max_drawdown = 0.0

        # Meets minimum
        results.meets_minimum = results.n_bets >= min_bets

        # Build summary table
        self._results = results
        results.summary_table = self.generate_table()

        logger.info(
            "Paper-bet ledger built. N=%d, Settled=%d, Pending=%d, "
            "Wins=%d, Losses=%d, Pushes=%d, HitRate=%.4f, ROI=%.4f, "
            "AvgCLV=%.4f, MeetsMinimum=%s, AllFieldsComplete=%s",
            results.n_bets, results.n_settled, results.n_pending,
            results.n_wins, results.n_losses, results.n_pushes,
            results.hit_rate, results.roi, results.avg_clv,
            results.meets_minimum, results.all_fields_complete,
        )

        return results

    def _compute_clv(self, bet_record: dict) -> float:
        """
        Compute closing line value in probability points.

        CLV = model_p_over_at_decision - close_novig_over (if bet is over)
        CLV = model_p_under_at_decision - close_novig_under (if bet is under)

        A positive CLV means the model probability was higher than the
        closing no-vig probability, indicating the bettor captured
        value that the market eventually priced away.

        Args:
            bet_record: The assembled bet record dict.

        Returns:
            CLV in probability points, or 0.0 if data is missing.
        """
        side = bet_record.get("recommended_side")
        if side is None:
            return 0.0

        if side == "over":
            model_prob = bet_record.get("model_p_over")
            close_prob = bet_record.get("close_over_prob_novig")
        elif side == "under":
            model_prob = bet_record.get("model_p_under")
            close_prob = bet_record.get("close_under_prob_novig")
        else:
            return 0.0

        if model_prob is None or close_prob is None:
            return 0.0

        return float(model_prob - close_prob)

    @staticmethod
    def _compute_drawdown(pnl_series: list[float]) -> float:
        """
        Compute the maximum drawdown from a PnL series.

        Standard max-drawdown calculation on cumulative PnL:
        the largest peak-to-trough decline.

        Args:
            pnl_series: List of per-bet PnL values in chronological
                order.

        Returns:
            Maximum drawdown as a positive float (0.0 if no drawdown).
        """
        if not pnl_series:
            return 0.0

        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        return max_dd

    @staticmethod
    def _check_critical_fields(record: dict) -> list[str]:
        """
        Check a bet record for missing critical fields.

        Critical fields: odds_snapshot_timestamp, injury_snapshot_timestamp,
        model_version, tracking_available, feature_snapshot_id.

        Args:
            record: A single bet record dict.

        Returns:
            List of missing field names (empty if all present).
        """
        missing = []
        for field_name in CRITICAL_FIELDS:
            if record.get(field_name) is None:
                missing.append(field_name)
        return missing

    @staticmethod
    def _determine_bet_result(
        actual_3pm: float,
        line: Optional[float],
        side: Optional[str],
    ) -> str:
        """
        Determine bet result (win/loss/push) from actual 3PM, line,
        and side.

        Args:
            actual_3pm: Actual 3-pointers made.
            line: The prop line.
            side: 'over' or 'under'.

        Returns:
            'win', 'loss', or 'push'.
        """
        if line is None or side is None:
            return "loss"

        if side == "over":
            if actual_3pm > line:
                return "win"
            elif actual_3pm < line:
                return "loss"
            else:
                return "push"
        else:  # under
            if actual_3pm < line:
                return "win"
            elif actual_3pm > line:
                return "loss"
            else:
                return "push"

    @staticmethod
    def _compute_pnl(
        bet_result: Optional[str],
        stake_pct: Optional[float],
        odds_decimal: Optional[float],
    ) -> float:
        """
        Compute PnL in units for a single bet.

        Args:
            bet_result: 'win', 'loss', or 'push'.
            stake_pct: Stake as a fraction of bankroll.
            odds_decimal: Decimal odds for the bet side.

        Returns:
            PnL in units.
        """
        if bet_result is None or stake_pct is None:
            return 0.0

        stake = stake_pct or 0.0
        odds = odds_decimal or 2.0  # Fallback to even money if missing

        if bet_result == "win":
            return stake * (odds - 1.0)
        elif bet_result == "loss":
            return -stake
        else:  # push
            return 0.0

    def generate_table(self) -> list[dict]:
        """
        Generate the full ledger table with all columns for each bet,
        sorted by decision_timestamp, plus a summary row at bottom.

        Returns:
            List of dicts. Each dict is a row in the table. The last
            row is the summary.
        """
        if self._results is None:
            return []

        r = self._results
        table = []

        for i, record in enumerate(r.ledger):
            row = {
                "row_num": i + 1,
                "game_id": record.get("game_id"),
                "player_id": record.get("player_id"),
                "sportsbook": record.get("sportsbook"),
                "line": record.get("line"),
                "odds_over": record.get("odds_over"),
                "odds_under": record.get("odds_under"),
                "model_p_over": record.get("model_p_over"),
                "model_p_under": record.get("model_p_under"),
                "fair_odds_over": record.get("fair_odds_over"),
                "fair_odds_under": record.get("fair_odds_under"),
                "edge": record.get("edge"),
                "recommended_side": record.get("recommended_side"),
                "stake_pct": record.get("stake_pct"),
                "decision_timestamp": record.get("decision_timestamp"),
                "model_version": record.get("model_version"),
                "feature_snapshot_id": record.get("feature_snapshot_id"),
                "tracking_available": record.get("tracking_available"),
                "injury_snapshot_timestamp": record.get(
                    "injury_snapshot_timestamp"
                ),
                "odds_snapshot_timestamp": record.get(
                    "odds_snapshot_timestamp"
                ),
                "freeze_timestamp": record.get("freeze_timestamp"),
                "close_over_prob_novig": record.get("close_over_prob_novig"),
                "close_under_prob_novig": record.get("close_under_prob_novig"),
                "actual_3pm": record.get("actual_3pm"),
                "bet_result": record.get("bet_result"),
                "pnl_units": record.get("pnl_units"),
                "clv_prob_pts": record.get("clv_prob_pts"),
            }
            table.append(row)

        # Summary row
        table.append({
            "row_num": "SUMMARY",
            "game_id": "",
            "player_id": "",
            "sportsbook": "",
            "line": "",
            "odds_over": "",
            "odds_under": "",
            "model_p_over": "",
            "model_p_under": "",
            "fair_odds_over": "",
            "fair_odds_under": "",
            "edge": r.avg_edge,
            "recommended_side": "",
            "stake_pct": r.total_staked,
            "decision_timestamp": "",
            "model_version": "",
            "feature_snapshot_id": "",
            "tracking_available": "",
            "injury_snapshot_timestamp": "",
            "odds_snapshot_timestamp": "",
            "freeze_timestamp": "",
            "close_over_prob_novig": "",
            "close_under_prob_novig": "",
            "actual_3pm": "",
            "bet_result": f"{r.n_wins}W-{r.n_losses}L-{r.n_pushes}P",
            "pnl_units": r.total_pnl,
            "clv_prob_pts": r.avg_clv,
        })

        return table

    def generate_summary(self) -> dict:
        """
        Generate a compact summary dictionary of ledger statistics.

        Returns:
            Dict with keys: n_bets, hit_rate, roi, avg_clv, avg_edge,
            max_drawdown, missing_fields, close_capture_rate.
        """
        if self._results is None:
            return {}

        r = self._results
        return {
            "n_bets": r.n_bets,
            "n_settled": r.n_settled,
            "n_pending": r.n_pending,
            "hit_rate": round(r.hit_rate, 4),
            "roi": round(r.roi, 4),
            "avg_clv": round(r.avg_clv, 4),
            "avg_edge": round(r.avg_edge, 4),
            "max_drawdown": round(r.max_drawdown, 4),
            "missing_fields": r.missing_fields_count,
            "close_capture_rate": round(r.close_capture_rate, 4),
            "meets_minimum": r.meets_minimum,
            "all_fields_complete": r.all_fields_complete,
        }

    def format_report(self) -> str:
        """
        Format a human-readable ledger report.

        Returns:
            Multi-line string report with summary stats and
            per-bet details.
        """
        if self._results is None:
            return "No ledger results available. Build the ledger first."

        r = self._results
        lines = []
        lines.append("=" * 80)
        lines.append("100-PAPER-BET LEDGER REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall status
        status_parts = []
        if r.meets_minimum:
            status_parts.append(f"Minimum bets: PASSED ({r.n_bets} >= 100)")
        else:
            status_parts.append(f"Minimum bets: FAILED ({r.n_bets} < 100)")

        if r.all_fields_complete:
            status_parts.append("Field completeness: PASSED (all critical fields present)")
        else:
            status_parts.append(
                f"Field completeness: FAILED ({r.missing_fields_count} "
                f"records with missing fields)"
            )

        status_parts.append(
            f"Close capture rate: {r.close_capture_rate:.1%}"
        )

        for part in status_parts:
            lines.append(f"  {part}")
        lines.append("")

        # Summary statistics
        lines.append("-" * 80)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"  Total bets:       {r.n_bets}")
        lines.append(f"  Settled:          {r.n_settled}")
        lines.append(f"  Pending:          {r.n_pending}")
        lines.append(f"  Wins:             {r.n_wins}")
        lines.append(f"  Losses:           {r.n_losses}")
        lines.append(f"  Pushes:           {r.n_pushes}")
        lines.append(f"  Hit rate:         {r.hit_rate:.4f}")
        lines.append(f"  Total staked:     {r.total_staked:.4f}")
        lines.append(f"  Total PnL:        {r.total_pnl:+.4f}")
        lines.append(f"  ROI:              {r.roi:+.4f}")
        lines.append(f"  Avg edge:         {r.avg_edge:.4f}")
        lines.append(f"  Avg CLV:          {r.avg_clv:+.4f}")
        lines.append(f"  Max drawdown:     {r.max_drawdown:.4f}")
        lines.append("")

        # Ledger detail (abbreviated for large ledgers)
        lines.append("-" * 80)
        lines.append("LEDGER DETAIL")
        lines.append("-" * 80)
        lines.append(
            f"{'#':>4} {'Game':>12} {'Player':>12} {'Side':>6} "
            f"{'Line':>5} {'Edge':>7} {'Stake':>7} {'Result':>7} "
            f"{'PnL':>8} {'CLV':>7}"
        )
        lines.append("-" * 80)

        for i, record in enumerate(r.ledger):
            game = str(record.get("game_id", ""))[:12]
            player = str(record.get("player_id", ""))[:12]
            side = record.get("recommended_side", "")[:6]
            line_val = record.get("line")
            edge_val = record.get("edge")
            stake_val = record.get("stake_pct")
            result_val = record.get("bet_result", "pending") or "pending"
            pnl_val = record.get("pnl_units")
            clv_val = record.get("clv_prob_pts")

            line_str = f"{line_val:.1f}" if line_val is not None else "  N/A"
            edge_str = f"{edge_val:+.4f}" if edge_val is not None else "   N/A"
            stake_str = f"{stake_val:.4f}" if stake_val is not None else "   N/A"
            pnl_str = f"{pnl_val:+.4f}" if pnl_val is not None else "    N/A"
            clv_str = f"{clv_val:+.4f}" if clv_val is not None else "   N/A"

            lines.append(
                f"{i + 1:>4} {game:>12} {player:>12} {side:>6} "
                f"{line_str:>5} {edge_str:>7} {stake_str:>7} "
                f"{result_val:>7} {pnl_str:>8} {clv_str:>7}"
            )

        lines.append("-" * 80)
        lines.append(
            f"     {'TOTAL':>12} {'':>12} {'':>6} {'':>5} "
            f"{r.avg_edge:+.4f} {r.total_staked:.4f} "
            f"{r.n_wins}W-{r.n_losses}L-{r.n_pushes}P "
            f"{r.total_pnl:+.4f} {r.avg_clv:+.4f}"
        )
        lines.append("")

        # Missing fields detail
        if r.missing_fields_details:
            lines.append("-" * 80)
            lines.append("MISSING FIELDS DETAIL")
            lines.append("-" * 80)
            for detail in r.missing_fields_details:
                lines.append(
                    f"  game={detail.get('game_id')} "
                    f"player={detail.get('player_id')}: "
                    f"missing={detail.get('missing_fields')}"
                )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)
