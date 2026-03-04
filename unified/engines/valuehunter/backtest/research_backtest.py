"""Research backtest - does the model predict well? (Section K1)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from .leakage import LeakageDetector, LeakageError
from .metrics import (
    BacktestMetrics,
    MinutesMetrics,
    SlicedMetrics,
    ThreePAMetrics,
    ThreePMMetrics,
)

logger = logging.getLogger(__name__)


class ResearchBacktest:
    """
    Research backtest: evaluates model prediction quality in isolation.

    Purpose: "does the model predict well?"

    Runs the full prediction pipeline on historical data with strict
    temporal discipline, comparing predictions to actual outcomes.
    """

    def __init__(
        self,
        repository,
        feature_builder,
        minutes_model,
        three_pa_model,
        make_rate_model,
        monte_carlo_simulator,
        leakage_detector: Optional[LeakageDetector] = None,
    ):
        self.repository = repository
        self.feature_builder = feature_builder
        self.minutes_model = minutes_model
        self.three_pa_model = three_pa_model
        self.make_rate_model = make_rate_model
        self.monte_carlo_simulator = monte_carlo_simulator
        self.leakage_detector = leakage_detector or LeakageDetector()

    def run(
        self,
        start_date: str,
        end_date: str,
        freeze_offset_minutes: int = -60,
        also_score_t30: bool = False,
    ) -> dict:
        """
        Run the research backtest over a date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            freeze_offset_minutes: Minutes before tipoff for feature freeze
                (negative means before tipoff). Default -60 (T-60).
            also_score_t30: If True, also score predictions at T-30.

        Returns:
            Comprehensive results dict with metrics, sliced metrics,
            reliability curves, and individual predictions.
        """
        logger.info(
            "Starting research backtest from %s to %s (freeze=%d min)",
            start_date, end_date, freeze_offset_minutes,
        )

        games = self.repository.get_games_in_range(start_date, end_date)
        logger.info("Found %d games in date range", len(games))

        # Collectors for aggregation
        all_predictions = []
        minutes_actuals = []
        minutes_p50 = []
        minutes_p10 = []
        minutes_p90 = []
        minutes_avg_list = []
        tpa_actuals = []
        tpa_predicted = []
        tpm_actual_over = []
        tpm_p_over = []
        archetypes = []
        line_buckets = []
        tracking_flags = []
        home_away_flags = []
        rest_days_list = []
        b2b_flags = []
        injury_load_list = []
        spread_buckets = []

        t30_predictions = [] if also_score_t30 else None

        skipped = 0
        leakage_failures = 0
        processed = 0

        for game in games:
            tipoff = game.get("tipoff_utc") or game.get("tipoff_datetime_utc")
            if tipoff is None:
                logger.warning("Game %s has no tipoff time, skipping", game.get("game_id"))
                skipped += 1
                continue

            if isinstance(tipoff, str):
                tipoff = datetime.fromisoformat(tipoff)

            freeze_timestamp = tipoff + timedelta(minutes=freeze_offset_minutes)
            game_date = game.get("game_date", tipoff.strftime("%Y-%m-%d"))
            game_id = game.get("game_id")

            players = self.repository.get_eligible_players_for_game(game_id)

            for player in players:
                player_id = player.get("player_id")
                player_name = player.get("player_name", "Unknown")

                # Check eligibility
                player_games = self.repository.get_player_recent_games(
                    player_id, as_of_date=game_date
                )
                season_stats = self.repository.get_player_season_stats(
                    player_id, as_of_date=game_date
                )

                if not self._check_eligibility(player_games, season_stats):
                    skipped += 1
                    continue

                try:
                    # Build feature snapshot as-of freeze timestamp
                    feature_snapshot = self.feature_builder.build(
                        player_id=player_id,
                        game_id=game_id,
                        as_of=freeze_timestamp,
                    )

                    # Get injury snapshot as-of freeze timestamp
                    injury_snapshot = self.repository.get_injury_snapshot(
                        as_of=freeze_timestamp
                    )
                    feature_snapshot["injury_snapshot_timestamp_utc"] = (
                        injury_snapshot.get("snapshot_timestamp_utc")
                    )

                    # Get best available odds snapshot as-of freeze timestamp
                    odds_snapshot = self.repository.get_odds_snapshot(
                        player_id=player_id,
                        game_id=game_id,
                        as_of=freeze_timestamp,
                    )
                    feature_snapshot["odds_snapshot_timestamp_utc"] = (
                        odds_snapshot.get("snapshot_timestamp_utc")
                    )

                    # Run leakage checks (hard fail)
                    passed, violations = self.leakage_detector.run_all_checks(
                        feature_snapshot, freeze_timestamp
                    )
                    if not passed:
                        leakage_failures += 1
                        raise LeakageError(
                            f"Leakage detected for {player_name} in game {game_id}: "
                            + "; ".join(violations)
                        )

                    # Run minutes model -> quantile predictions
                    minutes_prediction = self.minutes_model.predict(feature_snapshot)

                    # Run 3PA model -> distribution params
                    tpa_prediction = self.three_pa_model.predict(feature_snapshot)

                    # Run make-rate model -> probability estimates
                    make_rate_prediction = self.make_rate_model.predict(feature_snapshot)

                    # Run Monte Carlo simulation
                    mc_result = self.monte_carlo_simulator.simulate(
                        minutes_prediction=minutes_prediction,
                        tpa_prediction=tpa_prediction,
                        make_rate_prediction=make_rate_prediction,
                    )

                    # Get actual results
                    actual = self.repository.get_actual_results(
                        player_id=player_id, game_id=game_id
                    )

                    if actual is None:
                        logger.debug(
                            "No actual results for %s in game %s", player_name, game_id
                        )
                        skipped += 1
                        continue

                    # Collect data for metric computation
                    actual_minutes = actual.get("minutes", 0.0)
                    actual_3pa = actual.get("three_pa", 0)
                    actual_3pm = actual.get("three_pm", 0)
                    line = odds_snapshot.get("line", 2.5)

                    minutes_actuals.append(actual_minutes)
                    minutes_p50.append(minutes_prediction.get("p50", 0.0))
                    minutes_p10.append(minutes_prediction.get("p10", 0.0))
                    minutes_p90.append(minutes_prediction.get("p90", 0.0))
                    minutes_avg_list.append(
                        season_stats.get("minutes_avg", 0.0)
                        if season_stats else 0.0
                    )

                    tpa_actuals.append(actual_3pa)
                    tpa_predicted.append(tpa_prediction.get("expected_3pa", 0.0))

                    actual_over_flag = 1.0 if actual_3pm > line else 0.0
                    tpm_actual_over.append(actual_over_flag)
                    tpm_p_over.append(mc_result.get("p_over", 0.5))

                    # Slice dimensions
                    archetypes.append(
                        feature_snapshot.get("feature_json", {}).get("archetype", "unknown")
                    )
                    line_buckets.append(self._get_line_bucket(line))
                    tracking_flags.append(
                        "available" if feature_snapshot.get("tracking_available", False)
                        else "fallback"
                    )
                    home_away_flags.append(
                        "home" if player.get("is_home", False) else "away"
                    )
                    rest_days_list.append(player.get("rest_days", 1))
                    b2b_flags.append(
                        "b2b" if player.get("is_b2b", False) else "not_b2b"
                    )
                    injury_load_list.append(
                        feature_snapshot.get("feature_json", {}).get(
                            "team_injury_load", "normal"
                        )
                    )
                    spread_buckets.append(
                        self._get_spread_bucket(
                            odds_snapshot.get("spread", 0.0)
                        )
                    )

                    prediction_record = {
                        "game_id": game_id,
                        "game_date": game_date,
                        "player_id": player_id,
                        "player_name": player_name,
                        "freeze_timestamp": str(freeze_timestamp),
                        "minutes_prediction": minutes_prediction,
                        "tpa_prediction": tpa_prediction,
                        "make_rate_prediction": make_rate_prediction,
                        "mc_result": mc_result,
                        "actual": actual,
                        "line": line,
                        "odds_snapshot": odds_snapshot,
                    }
                    all_predictions.append(prediction_record)

                    # Optionally score at T-30
                    if also_score_t30:
                        t30_freeze = tipoff + timedelta(minutes=-30)
                        t30_record = self._score_at_time(
                            player_id, game_id, t30_freeze, actual
                        )
                        if t30_record is not None:
                            t30_predictions.append(t30_record)

                    processed += 1

                except LeakageError:
                    raise
                except Exception as e:
                    logger.error(
                        "Error processing %s in game %s: %s",
                        player_name, game_id, e,
                        exc_info=True,
                    )
                    skipped += 1
                    continue

        logger.info(
            "Processed %d predictions, skipped %d, leakage failures %d",
            processed, skipped, leakage_failures,
        )

        # Convert to numpy arrays for metric computation
        minutes_actuals_arr = np.array(minutes_actuals)
        minutes_p50_arr = np.array(minutes_p50)
        minutes_p10_arr = np.array(minutes_p10)
        minutes_p90_arr = np.array(minutes_p90)
        minutes_avg_arr = np.array(minutes_avg_list)
        tpa_actuals_arr = np.array(tpa_actuals, dtype=float)
        tpa_predicted_arr = np.array(tpa_predicted, dtype=float)
        tpm_actual_over_arr = np.array(tpm_actual_over)
        tpm_p_over_arr = np.array(tpm_p_over)

        # Compute K4 metrics
        metrics_calculator = BacktestMetrics()

        minutes_metrics = (
            metrics_calculator.compute_minutes_metrics(
                actual=minutes_actuals_arr,
                predicted_p50=minutes_p50_arr,
                predicted_p10=minutes_p10_arr,
                predicted_p90=minutes_p90_arr,
                minutes_avg=minutes_avg_arr,
            )
            if len(minutes_actuals_arr) > 0
            else MinutesMetrics()
        )

        tpa_metrics = (
            metrics_calculator.compute_3pa_metrics(
                actual=tpa_actuals_arr, predicted=tpa_predicted_arr
            )
            if len(tpa_actuals_arr) > 0
            else ThreePAMetrics()
        )

        tpm_metrics = (
            metrics_calculator.compute_3pm_metrics(
                actual_over=tpm_actual_over_arr, p_over=tpm_p_over_arr
            )
            if len(tpm_actual_over_arr) > 0
            else ThreePMMetrics()
        )

        reliability_curve = (
            metrics_calculator.compute_reliability_curve(
                actual_over=tpm_actual_over_arr, p_over=tpm_p_over_arr
            )
            if len(tpm_actual_over_arr) > 0
            else []
        )

        # Compute K5 sliced metrics
        sliced = SlicedMetrics()

        if len(tpm_actual_over_arr) > 0:
            tpm_data = {
                "actual_over": tpm_actual_over_arr,
                "p_over": tpm_p_over_arr,
            }

            archetypes_arr = np.array(archetypes)
            line_buckets_arr = np.array(line_buckets)
            tracking_arr = np.array(tracking_flags)
            home_away_arr = np.array(home_away_flags)
            rest_arr = np.array(rest_days_list)
            b2b_arr = np.array(b2b_flags)
            injury_arr = np.array(injury_load_list)
            spread_arr = np.array(spread_buckets)

            sliced.by_archetype = metrics_calculator.slice_by(
                tpm_data, archetypes_arr,
                list(set(archetypes)),
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_line_bucket = metrics_calculator.slice_by(
                tpm_data, line_buckets_arr,
                ["0.5", "1.5", "2.5", "3.5+"],
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_tracking = metrics_calculator.slice_by(
                tpm_data, tracking_arr,
                ["available", "fallback"],
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_home_away = metrics_calculator.slice_by(
                tpm_data, home_away_arr,
                ["home", "away"],
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_rest = metrics_calculator.slice_by(
                tpm_data, rest_arr,
                list(set(rest_days_list)),
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_b2b = metrics_calculator.slice_by(
                tpm_data, b2b_arr,
                ["b2b", "not_b2b"],
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_injury_load = metrics_calculator.slice_by(
                tpm_data, injury_arr,
                list(set(injury_load_list)),
                metrics_calculator.compute_3pm_metrics,
            )
            sliced.by_spread_bucket = metrics_calculator.slice_by(
                tpm_data, spread_arr,
                list(set(spread_buckets)),
                metrics_calculator.compute_3pm_metrics,
            )

        results = {
            "summary": {
                "start_date": start_date,
                "end_date": end_date,
                "freeze_offset_minutes": freeze_offset_minutes,
                "total_games": len(games),
                "total_predictions": processed,
                "total_skipped": skipped,
                "leakage_failures": leakage_failures,
            },
            "metrics": {
                "minutes": minutes_metrics,
                "three_pa": tpa_metrics,
                "three_pm": tpm_metrics,
            },
            "sliced_metrics": sliced,
            "reliability_curve": reliability_curve,
            "predictions": all_predictions,
        }

        if also_score_t30 and t30_predictions:
            results["t30_predictions"] = t30_predictions

        logger.info(
            "Research backtest complete. Minutes MAE=%.2f, 3PA MAE=%.2f, "
            "3PM LogLoss=%.4f, Brier=%.4f",
            minutes_metrics.mae_overall,
            tpa_metrics.mae,
            tpm_metrics.log_loss,
            tpm_metrics.brier_score,
        )

        return results

    def _score_at_time(
        self,
        player_id: str,
        game_id: str,
        freeze_timestamp: datetime,
        actual: dict,
    ) -> Optional[dict]:
        """Score a single prediction at a given freeze time."""
        try:
            feature_snapshot = self.feature_builder.build(
                player_id=player_id,
                game_id=game_id,
                as_of=freeze_timestamp,
            )

            passed, violations = self.leakage_detector.run_all_checks(
                feature_snapshot, freeze_timestamp
            )
            if not passed:
                logger.warning(
                    "Leakage at T-30 for player %s game %s: %s",
                    player_id, game_id, violations,
                )
                return None

            minutes_prediction = self.minutes_model.predict(feature_snapshot)
            tpa_prediction = self.three_pa_model.predict(feature_snapshot)
            make_rate_prediction = self.make_rate_model.predict(feature_snapshot)
            mc_result = self.monte_carlo_simulator.simulate(
                minutes_prediction=minutes_prediction,
                tpa_prediction=tpa_prediction,
                make_rate_prediction=make_rate_prediction,
            )

            return {
                "player_id": player_id,
                "game_id": game_id,
                "freeze_timestamp": str(freeze_timestamp),
                "mc_result": mc_result,
                "actual": actual,
            }
        except Exception as e:
            logger.error(
                "Error scoring at T-30 for player %s game %s: %s",
                player_id, game_id, e,
            )
            return None

    @staticmethod
    def _check_eligibility(
        player_games: Optional[list[dict]],
        season_stats: Optional[dict],
    ) -> bool:
        """
        Check if a player is eligible for the backtest.

        Criteria:
        - projected_minutes >= 18
        - trailing_10_game_avg_minutes >= 16 OR season_avg_minutes >= 16
        - trailing_20_game_3PA_per_36 >= 4.5 OR prop listed in >= 8 of last 15
        - Not on 10-day/two-way contract
        - Not on minutes limit
        """
        if season_stats is None:
            return False

        # Check projected minutes
        projected_minutes = season_stats.get("projected_minutes", 0.0)
        if projected_minutes < 18:
            return False

        # Check trailing minutes
        trailing_10_avg = season_stats.get("trailing_10_game_avg_minutes", 0.0)
        season_avg = season_stats.get("season_avg_minutes", 0.0)
        if trailing_10_avg < 16 and season_avg < 16:
            return False

        # Check 3PA volume
        trailing_20_3pa_per36 = season_stats.get("trailing_20_game_3pa_per_36", 0.0)
        prop_listing_count = season_stats.get("prop_listed_last_15", 0)
        if trailing_20_3pa_per36 < 4.5 and prop_listing_count < 8:
            return False

        # Check contract type
        contract_type = season_stats.get("contract_type", "standard")
        if contract_type in ("10-day", "two-way"):
            return False

        # Check minutes limit
        if season_stats.get("on_minutes_limit", False):
            return False

        return True

    @staticmethod
    def _get_line_bucket(line: float) -> str:
        """Categorize a prop line into buckets."""
        if line <= 0.5:
            return "0.5"
        elif line <= 1.5:
            return "1.5"
        elif line <= 2.5:
            return "2.5"
        else:
            return "3.5+"

    @staticmethod
    def _get_spread_bucket(spread: float) -> str:
        """Categorize a game spread into buckets."""
        abs_spread = abs(spread)
        if abs_spread <= 3.0:
            return "close"
        elif abs_spread <= 7.0:
            return "moderate"
        elif abs_spread <= 12.0:
            return "large"
        else:
            return "blowout"
