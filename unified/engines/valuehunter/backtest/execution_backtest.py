"""Execution backtest - does the edge survive real execution? (Section K2)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from .leakage import LeakageDetector, LeakageError
from .metrics import BacktestMetrics, BettingMetrics, SlicedMetrics

logger = logging.getLogger(__name__)

# Time buckets for multi-pass scoring
TIME_BUCKETS = [
    "D-1_1705_LOCAL",
    "B2B_1305_LOCAL",
    "GAMEDAY_1130_LOCAL",
    "T-90",
    "T-30",
    "T-5",
]


class ExecutionBacktest:
    """
    Execution backtest: evaluates whether model edge survives real execution.

    Purpose: "does the edge survive real execution?"

    Simulates the full betting workflow including:
    - Multi-pass scoring at realistic time buckets
    - Decision engine bet triggering
    - Fill simulation with latency
    - One ticket max per player x side x game
    - Duplicate state blocking
    """

    def __init__(
        self,
        repository,
        feature_builder,
        minutes_model,
        three_pa_model,
        make_rate_model,
        monte_carlo_simulator,
        decision_engine,
        leakage_detector: Optional[LeakageDetector] = None,
        latency_sec: float = 2.0,
    ):
        self.repository = repository
        self.feature_builder = feature_builder
        self.minutes_model = minutes_model
        self.three_pa_model = three_pa_model
        self.make_rate_model = make_rate_model
        self.monte_carlo_simulator = monte_carlo_simulator
        self.decision_engine = decision_engine
        self.leakage_detector = leakage_detector or LeakageDetector()
        self.latency_sec = latency_sec

    def run(self, start_date: str, end_date: str) -> dict:
        """
        Run the execution backtest over a date range.

        Simulates the full betting workflow at multiple time buckets,
        running the decision engine and simulating fills with latency.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Results dict with filled bets, K4 metrics, and comparison data.
        """
        logger.info(
            "Starting execution backtest from %s to %s (latency=%0.1fs)",
            start_date, end_date, self.latency_sec,
        )

        game_dates = self.repository.get_game_dates_in_range(start_date, end_date)
        logger.info("Found %d game dates in range", len(game_dates))

        all_filled_bets = []
        all_triggered = []
        all_missed = []
        tickets_placed: dict[str, set] = {}  # game_id -> set of (player_id, side)
        previous_states: dict[str, dict] = {}  # player_id -> last state hash

        total_scored = 0
        total_triggered = 0
        total_filled = 0
        total_duplicate_blocked = 0
        total_ticket_blocked = 0

        for game_date in game_dates:
            games = self.repository.get_games_on_date(game_date)

            for game in games:
                game_id = game.get("game_id")
                tipoff = game.get("tipoff_utc") or game.get("tipoff_datetime_utc")
                if tipoff is None:
                    logger.warning(
                        "Game %s has no tipoff time, skipping", game_id
                    )
                    continue

                if isinstance(tipoff, str):
                    tipoff = datetime.fromisoformat(tipoff)

                local_tz_offset = game.get("local_tz_offset_hours", 0)
                game_tickets = tickets_placed.setdefault(game_id, set())

                players = self.repository.get_eligible_players_for_game(game_id)

                for player in players:
                    player_id = player.get("player_id")
                    player_name = player.get("player_name", "Unknown")

                    for time_bucket in TIME_BUCKETS:
                        try:
                            freeze_timestamp = self._resolve_freeze_time(
                                tipoff, time_bucket, local_tz_offset, game_date
                            )
                            if freeze_timestamp is None:
                                continue

                            # Skip B2B bucket if not a back-to-back
                            if time_bucket == "B2B_1305_LOCAL" and not player.get(
                                "is_b2b", False
                            ):
                                continue

                            # Build features as-of freeze timestamp
                            feature_snapshot = self.feature_builder.build(
                                player_id=player_id,
                                game_id=game_id,
                                as_of=freeze_timestamp,
                            )

                            # Leakage check
                            passed, violations = self.leakage_detector.run_all_checks(
                                feature_snapshot, freeze_timestamp
                            )
                            if not passed:
                                raise LeakageError(
                                    f"Leakage at {time_bucket} for {player_name}: "
                                    + "; ".join(violations)
                                )

                            # Get odds snapshot
                            odds_snapshot = self.repository.get_odds_snapshot(
                                player_id=player_id,
                                game_id=game_id,
                                as_of=freeze_timestamp,
                            )

                            if not odds_snapshot or not odds_snapshot.get("line"):
                                continue

                            # Run models
                            minutes_pred = self.minutes_model.predict(feature_snapshot)
                            tpa_pred = self.three_pa_model.predict(feature_snapshot)
                            make_rate_pred = self.make_rate_model.predict(feature_snapshot)
                            mc_result = self.monte_carlo_simulator.simulate(
                                minutes_prediction=minutes_pred,
                                tpa_prediction=tpa_pred,
                                make_rate_prediction=make_rate_pred,
                            )

                            total_scored += 1

                            # Check for duplicate state (block unchanged)
                            state_key = f"{player_id}_{game_id}"
                            current_state = {
                                "p_over": mc_result.get("p_over"),
                                "line": odds_snapshot.get("line"),
                                "odds_over": odds_snapshot.get("odds_over_decimal"),
                                "odds_under": odds_snapshot.get("odds_under_decimal"),
                            }

                            prev_state = previous_states.get(state_key)
                            if prev_state and prev_state == current_state:
                                total_duplicate_blocked += 1
                                continue
                            previous_states[state_key] = current_state

                            # Run decision engine
                            decision = self.decision_engine.evaluate(
                                mc_result=mc_result,
                                odds_snapshot=odds_snapshot,
                                player_context=player,
                                feature_snapshot=feature_snapshot,
                            )

                            if not decision.get("trigger", False):
                                continue

                            total_triggered += 1
                            side = decision.get("side", "over")

                            # Check one ticket max per player x side x game
                            ticket_key = (player_id, side)
                            if ticket_key in game_tickets:
                                total_ticket_blocked += 1
                                logger.debug(
                                    "Ticket already placed for %s %s in game %s",
                                    player_name, side, game_id,
                                )
                                continue

                            # Simulate fill with latency
                            fill_result = self._simulate_fill(
                                odds_snapshot=odds_snapshot,
                                side=side,
                                latency_sec=self.latency_sec,
                                freeze_timestamp=freeze_timestamp,
                                player_id=player_id,
                                game_id=game_id,
                            )

                            if not fill_result.get("filled", False):
                                all_missed.append({
                                    "game_id": game_id,
                                    "player_id": player_id,
                                    "player_name": player_name,
                                    "time_bucket": time_bucket,
                                    "reason": fill_result.get("reason", "unknown"),
                                })
                                continue

                            # Mark ticket placed
                            game_tickets.add(ticket_key)
                            total_filled += 1

                            # Get actual results for grading
                            actual = self.repository.get_actual_results(
                                player_id=player_id, game_id=game_id
                            )

                            # Determine bet result
                            line = odds_snapshot.get("line", 2.5)
                            actual_3pm = actual.get("three_pm", 0) if actual else 0
                            if side == "over":
                                result = 1 if actual_3pm > line else (-1 if actual_3pm < line else 0)
                            else:
                                result = 1 if actual_3pm < line else (-1 if actual_3pm > line else 0)

                            # Get CLV if closing line is available
                            closing_odds = self.repository.get_closing_odds(
                                player_id=player_id, game_id=game_id
                            )
                            clv = self._compute_clv(
                                fill_result.get("fill_odds_decimal", 0),
                                closing_odds,
                                side,
                            )

                            filled_bet = {
                                "game_id": game_id,
                                "game_date": game_date,
                                "player_id": player_id,
                                "player_name": player_name,
                                "time_bucket": time_bucket,
                                "side": side,
                                "line": line,
                                "edge": decision.get("edge", 0.0),
                                "stake": decision.get("stake", 1.0),
                                "fill_odds_decimal": fill_result.get(
                                    "fill_odds_decimal", 0
                                ),
                                "result": result,
                                "actual_3pm": actual_3pm,
                                "clv": clv,
                                "mc_p_over": mc_result.get("p_over", 0.5),
                                "freeze_timestamp": str(freeze_timestamp),
                            }
                            all_filled_bets.append(filled_bet)
                            all_triggered.append(filled_bet)

                        except LeakageError:
                            raise
                        except Exception as e:
                            logger.error(
                                "Error at %s for %s in game %s: %s",
                                time_bucket, player_name, game_id, e,
                                exc_info=True,
                            )
                            continue

        logger.info(
            "Execution backtest complete. Scored=%d, Triggered=%d, "
            "Filled=%d, DupBlocked=%d, TicketBlocked=%d",
            total_scored, total_triggered, total_filled,
            total_duplicate_blocked, total_ticket_blocked,
        )

        # Compute K4 betting metrics for filled bets
        betting_metrics = BettingMetrics()
        sliced = SlicedMetrics()

        if all_filled_bets:
            edges = np.array([b["edge"] for b in all_filled_bets])
            results = np.array([b["result"] for b in all_filled_bets], dtype=float)
            stakes = np.array([b["stake"] for b in all_filled_bets])
            odds_decimal = np.array([b["fill_odds_decimal"] for b in all_filled_bets])
            clv_pts = np.array([b.get("clv", 0.0) for b in all_filled_bets])

            betting_metrics = BacktestMetrics.compute_betting_metrics(
                edges=edges,
                results=results,
                stakes=stakes,
                odds_decimal=odds_decimal,
                clv_pts=clv_pts,
            )

            # Slice by time bucket
            time_buckets_arr = np.array([b["time_bucket"] for b in all_filled_bets])
            sliced.by_time_bucket = BacktestMetrics.slice_by(
                {
                    "edges": edges,
                    "results": results,
                    "stakes": stakes,
                    "odds_decimal": odds_decimal,
                },
                time_buckets_arr,
                TIME_BUCKETS,
                BacktestMetrics.compute_betting_metrics,
            )

        results_dict = {
            "summary": {
                "start_date": start_date,
                "end_date": end_date,
                "latency_sec": self.latency_sec,
                "total_game_dates": len(game_dates),
                "total_scored": total_scored,
                "total_triggered": total_triggered,
                "total_filled": total_filled,
                "total_duplicate_blocked": total_duplicate_blocked,
                "total_ticket_blocked": total_ticket_blocked,
                "total_missed": len(all_missed),
            },
            "betting_metrics": betting_metrics,
            "sliced_metrics": sliced,
            "filled_bets": all_filled_bets,
            "missed_fills": all_missed,
        }

        logger.info(
            "Betting metrics: ROI=%.4f, HitRate=%.4f, CLV=%.4f, N=%d",
            betting_metrics.roi,
            betting_metrics.hit_rate,
            betting_metrics.clv_mean,
            betting_metrics.n_bets,
        )

        return results_dict

    def _resolve_freeze_time(
        self,
        tipoff: datetime,
        time_bucket: str,
        local_tz_offset_hours: float,
        game_date: str,
    ) -> Optional[datetime]:
        """
        Resolve a time bucket to a concrete freeze timestamp (UTC).

        Args:
            tipoff: Game tipoff time in UTC.
            time_bucket: One of TIME_BUCKETS.
            local_tz_offset_hours: Offset from UTC to local time.
            game_date: Game date string (YYYY-MM-DD).

        Returns:
            Freeze timestamp in UTC, or None if the bucket is not applicable.
        """
        try:
            game_date_dt = datetime.strptime(game_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

        if time_bucket == "T-90":
            return tipoff - timedelta(minutes=90)
        elif time_bucket == "T-30":
            return tipoff - timedelta(minutes=30)
        elif time_bucket == "T-5":
            return tipoff - timedelta(minutes=5)
        elif time_bucket == "D-1_1705_LOCAL":
            # Day before at 5:05 PM local time
            prev_day = game_date_dt - timedelta(days=1)
            local_time = prev_day.replace(hour=17, minute=5, second=0, microsecond=0)
            utc_time = local_time - timedelta(hours=local_tz_offset_hours)
            return utc_time
        elif time_bucket == "B2B_1305_LOCAL":
            # Game day at 1:05 PM local time (for B2B games)
            local_time = game_date_dt.replace(
                hour=13, minute=5, second=0, microsecond=0
            )
            utc_time = local_time - timedelta(hours=local_tz_offset_hours)
            return utc_time
        elif time_bucket == "GAMEDAY_1130_LOCAL":
            # Game day at 11:30 AM local time
            local_time = game_date_dt.replace(
                hour=11, minute=30, second=0, microsecond=0
            )
            utc_time = local_time - timedelta(hours=local_tz_offset_hours)
            return utc_time
        else:
            logger.warning("Unknown time bucket: %s", time_bucket)
            return None

    def _simulate_fill(
        self,
        odds_snapshot: dict,
        side: str,
        latency_sec: float,
        freeze_timestamp: datetime,
        player_id: str,
        game_id: str,
    ) -> dict:
        """
        Simulate filling a bet after latency.

        Returns the next available price after latency delay.
        May result in worse fill or missed fill.

        Args:
            odds_snapshot: Current odds snapshot.
            side: "over" or "under".
            latency_sec: Simulated latency in seconds.
            freeze_timestamp: Time of the scoring pass.
            player_id: Player ID.
            game_id: Game ID.

        Returns:
            Dict with 'filled', 'fill_odds_decimal', and optionally 'reason'.
        """
        fill_time = freeze_timestamp + timedelta(seconds=latency_sec)

        # Get the next available odds snapshot after latency
        next_odds = self.repository.get_odds_snapshot(
            player_id=player_id,
            game_id=game_id,
            as_of=fill_time,
        )

        if next_odds is None:
            return {
                "filled": False,
                "reason": "no_odds_available_after_latency",
            }

        # Check if the line has moved or odds have disappeared
        if side == "over":
            fill_odds = next_odds.get("odds_over_decimal")
        else:
            fill_odds = next_odds.get("odds_under_decimal")

        if fill_odds is None or fill_odds <= 1.0:
            return {
                "filled": False,
                "reason": "odds_disappeared_or_invalid",
            }

        # Check if line moved unfavorably
        original_line = odds_snapshot.get("line")
        new_line = next_odds.get("line")
        if original_line is not None and new_line is not None:
            if side == "over" and new_line > original_line + 0.5:
                return {
                    "filled": False,
                    "reason": f"line_moved_against: {original_line} -> {new_line}",
                }
            elif side == "under" and new_line < original_line - 0.5:
                return {
                    "filled": False,
                    "reason": f"line_moved_against: {original_line} -> {new_line}",
                }

        return {
            "filled": True,
            "fill_odds_decimal": fill_odds,
            "fill_time": str(fill_time),
            "original_odds": odds_snapshot.get(
                f"odds_{side}_decimal"
            ),
            "line_at_fill": new_line,
        }

    @staticmethod
    def _compute_clv(
        fill_odds_decimal: float,
        closing_odds: Optional[dict],
        side: str,
    ) -> float:
        """
        Compute closing line value in probability points.

        CLV = fill_implied_prob - closing_implied_prob
        Positive CLV means we got a better price than closing.

        Args:
            fill_odds_decimal: Decimal odds at fill.
            closing_odds: Dict with closing odds info.
            side: "over" or "under".

        Returns:
            CLV in probability points (positive = good).
        """
        if not closing_odds or fill_odds_decimal <= 1.0:
            return 0.0

        closing_key = f"closing_{side}_decimal"
        closing_decimal = closing_odds.get(closing_key, 0.0)

        if closing_decimal <= 1.0:
            return 0.0

        fill_implied = 1.0 / fill_odds_decimal
        closing_implied = 1.0 / closing_decimal

        # Positive CLV = we got better price (lower implied prob for our side)
        return closing_implied - fill_implied
