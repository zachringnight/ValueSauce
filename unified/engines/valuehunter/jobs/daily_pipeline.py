"""Daily pipeline - orchestrates the full decision-making workflow."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from ..config import Settings, get_settings
from ..features.builder import FeatureSnapshotBuilder
from ..models.minutes_model import MinutesModel
from ..models.three_pa_model import ThreePAModel
from ..models.make_rate_model import MakeRateModel
from ..pricing.monte_carlo import MonteCarloSimulator
from ..pricing.decision_engine import DecisionEngine
from ..utils.db import Repository

logger = logging.getLogger(__name__)


# Regular-rotation eligibility constants
MIN_PROJECTED_MINUTES = 18.0
MIN_TRAILING_10G_AVG_MINUTES = 16.0
MIN_SEASON_AVG_MINUTES = 16.0
MIN_TRAILING_20G_3PA_PER_36 = 4.5
MIN_PROP_APPEARANCES = 8
PROP_APPEARANCE_WINDOW = 15


class DailyPipeline:
    """Full daily pipeline: features -> models -> pricing -> decisions."""

    def __init__(
        self,
        repository: Repository,
        feature_builder: FeatureSnapshotBuilder,
        minutes_model: MinutesModel,
        three_pa_model: ThreePAModel,
        make_rate_model: MakeRateModel,
        simulator: MonteCarloSimulator,
        decision_engine: DecisionEngine,
        settings: Optional[Settings] = None,
    ):
        self.repo = repository
        self.feature_builder = feature_builder
        self.minutes_model = minutes_model
        self.three_pa_model = three_pa_model
        self.make_rate_model = make_rate_model
        self.simulator = simulator
        self.decision_engine = decision_engine
        self.settings = settings or get_settings()

    def run(self, target_date: date) -> list[dict]:
        """
        Run the full pipeline for a target date.

        Returns list of bet decisions (bets and passes).
        """
        now = datetime.now(timezone.utc)
        logger.info("Starting daily pipeline for %s at %s", target_date, now)

        # 1. Get today's games
        games = self._get_games(target_date)
        if not games:
            logger.info("No games found for %s", target_date)
            return []

        logger.info("Found %d games for %s", len(games), target_date)

        all_decisions = []
        pending_decisions = []

        for game in games:
            game_id = game["nba_game_id"]
            game_decisions = self._process_game(
                game, now, pending_decisions
            )
            all_decisions.extend(game_decisions)
            pending_decisions.extend(
                [d for d in game_decisions if d.get("recommended_side") != "no_bet"]
            )

        # Summary
        bets = [d for d in all_decisions if d.get("recommended_side") != "no_bet"]
        logger.info(
            "Pipeline complete: %d decisions, %d bets, %d passes",
            len(all_decisions),
            len(bets),
            len(all_decisions) - len(bets),
        )

        return all_decisions

    def _process_game(
        self, game: dict, freeze_time: datetime, pending_decisions: list
    ) -> list[dict]:
        """Process a single game: find eligible players and generate decisions."""
        game_id = game["nba_game_id"]
        home_abbr = game["home_team_abbr"]
        away_abbr = game["away_team_abbr"]
        decisions = []

        # Get players from both teams
        for team_abbr in [home_abbr, away_abbr]:
            players = self._get_eligible_players(game_id, team_abbr)
            opponent = away_abbr if team_abbr == home_abbr else home_abbr

            for player_id in players:
                try:
                    decision = self._process_player(
                        player_id, game, team_abbr, opponent,
                        freeze_time, pending_decisions,
                    )
                    if decision:
                        decisions.append(decision)
                except Exception as e:
                    logger.error(
                        "Error processing player %s in game %s: %s",
                        player_id, game_id, e,
                    )

        return decisions

    def _process_player(
        self,
        player_id: str,
        game: dict,
        team_abbr: str,
        opponent_abbr: str,
        freeze_time: datetime,
        pending_decisions: list,
    ) -> Optional[dict]:
        """Run the full model pipeline for a single player-game."""
        game_id = game["nba_game_id"]

        # Fetch historical data
        player_games = self.repo.get_player_games(player_id, limit=30)
        tracking_games = self.repo.get_player_tracking(player_id)
        opponent_shooting = self.repo.get_team_opponent_shooting(opponent_abbr)
        injury_snapshot = self.repo.get_injury_snapshot_as_of(
            player_id, game_id, freeze_time
        )
        odds_snapshots = self.repo.get_odds_snapshots(
            game_id, player_id, market="player_3pt_made"
        )

        if not odds_snapshots:
            logger.debug("No odds found for player %s game %s", player_id, game_id)
            return None

        # Get teammate injury statuses
        teammate_statuses = self._get_teammate_statuses(
            game_id, team_abbr, player_id, freeze_time
        )

        # Build game context
        game_context = {
            "spread": game.get("closing_spread_home", 0),
            "team_total": game.get("closing_total", 220) / 2,
            "is_home": team_abbr == game["home_team_abbr"],
            "opponent_abbr": opponent_abbr,
        }

        # Build feature snapshot
        snapshot = self.feature_builder.build_snapshot(
            player_id=player_id,
            game_id=game_id,
            freeze_timestamp=freeze_time,
            player_games=player_games,
            tracking_games=tracking_games,
            opponent_shooting=opponent_shooting,
            game_context=game_context,
            injury_snapshot=injury_snapshot,
            teammate_statuses=teammate_statuses,
        )

        # Store feature snapshot
        snapshot_id = self.repo.insert_feature_snapshot(snapshot)

        # Run minutes model
        import numpy as np
        features = snapshot.get("feature_json", {})
        minutes_features = np.array(
            [features.get(f, 0) for f in self.minutes_model.feature_names_]
            if hasattr(self.minutes_model, "feature_names_") else [0]
        ).reshape(1, -1)

        minutes_quantiles = self.minutes_model.predict_quantiles(minutes_features)

        # Run 3PA model
        three_pa_features = np.array(
            [features.get(f, 0) for f in self.three_pa_model.feature_names_]
            if hasattr(self.three_pa_model, "feature_names_") else [0]
        ).reshape(1, -1)

        three_pa_params = self.three_pa_model.predict_distribution_params(
            three_pa_features,
            minutes_exposure=np.array([minutes_quantiles["p50"][0]]),
        )

        # Run make-rate model
        make_features = np.array(
            [features.get(f, 0) for f in self.make_rate_model.feature_names_]
            if hasattr(self.make_rate_model, "feature_names_") else [0]
        ).reshape(1, -1)

        make_params = self.make_rate_model.predict_with_uncertainty(make_features)

        # Get best available odds
        best_odds = odds_snapshots[0]  # Most recent snapshot
        line = best_odds.get("line", 2.5)

        # Run Monte Carlo simulation
        sim_result = self.simulator.simulate(
            minutes_p10=float(minutes_quantiles["p10"][0]),
            minutes_p50=float(minutes_quantiles["p50"][0]),
            minutes_p90=float(minutes_quantiles["p90"][0]),
            three_pa_mean=float(three_pa_params["mean"][0]),
            three_pa_dispersion=float(three_pa_params["dispersion"][0]),
            make_prob_mean=float(make_params["mean"][0]),
            make_prob_uncertainty=float(make_params["uncertainty"][0]),
            line=line,
        )

        # Run decision engine for each book
        best_decision = None
        book_opportunities = []

        for odds in odds_snapshots:
            decision = self.decision_engine.evaluate_opportunity(
                simulation_result=sim_result,
                odds_prop=odds,
                game_context=game_context,
                feature_snapshot=snapshot,
            )
            book_opportunities.append(decision)

        # Line-shop across books
        if book_opportunities:
            best_decision = self.decision_engine.line_shop(book_opportunities)

            # Check correlation limits
            if best_decision.get("recommended_side") != "no_bet":
                can_bet = self.decision_engine.check_correlation_limits(
                    pending_decisions, best_decision
                )
                if not can_bet:
                    best_decision["recommended_side"] = "no_bet"
                    best_decision["exclusion_reason"] = "correlation_limit"

            # Store decision
            best_decision["feature_snapshot_id"] = snapshot_id
            best_decision["nba_game_id"] = game_id
            best_decision["nba_player_id"] = player_id
            self.repo.insert_bet_decision(best_decision)

        return best_decision

    def _get_games(self, target_date: date) -> list[dict]:
        """Fetch games for the target date from the database."""
        # Query games table for this date
        conn = self.repo.conn
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM games WHERE game_date = %s AND season_type = 'Regular Season'",
                (target_date,),
            )
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _get_eligible_players(self, game_id: str, team_abbr: str) -> list[str]:
        """Get list of eligible player IDs for a game/team."""
        player_games = self._get_team_recent_players(team_abbr)
        eligible = []

        for pid, games in player_games.items():
            if not games:
                continue

            season_avg_min = sum(g.get("minutes_played", 0) for g in games) / len(games)
            recent_10 = games[:10]
            trailing_10_avg = (
                sum(g.get("minutes_played", 0) for g in recent_10) / len(recent_10)
                if recent_10 else 0
            )

            # Minutes check
            if season_avg_min < MIN_SEASON_AVG_MINUTES and trailing_10_avg < MIN_TRAILING_10G_AVG_MINUTES:
                continue

            # 3PA volume check
            recent_20 = games[:20]
            total_3pa = sum(g.get("three_pa", 0) for g in recent_20)
            total_min = sum(g.get("minutes_played", 0) for g in recent_20)
            per_36 = (total_3pa / max(total_min, 1)) * 36 if total_min > 0 else 0

            if per_36 < MIN_TRAILING_20G_3PA_PER_36:
                # Check prop appearance fallback
                # Simplified: skip if low volume
                continue

            eligible.append(pid)

        return eligible

    def _get_team_recent_players(self, team_abbr: str) -> dict:
        """Get recent game logs grouped by player for a team."""
        conn = self.repo.conn
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT nba_player_id, minutes_played, three_pa, three_pm, started
                FROM player_game
                WHERE team_abbr = %s
                ORDER BY nba_game_id DESC
                LIMIT 500
                """,
                (team_abbr,),
            )
            rows = cur.fetchall()

        players: dict[str, list] = {}
        for row in rows:
            pid = str(row[0])
            if pid not in players:
                players[pid] = []
            players[pid].append({
                "minutes_played": row[1],
                "three_pa": row[2],
                "three_pm": row[3],
                "started": row[4],
            })

        return players

    def _get_teammate_statuses(
        self, game_id: str, team_abbr: str, exclude_player: str,
        as_of: datetime,
    ) -> list[dict]:
        """Get injury statuses for teammates."""
        conn = self.repo.conn
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (nba_player_id)
                    nba_player_id, status, reason_text
                FROM injury_reports
                WHERE nba_game_id = %s
                  AND team_abbr = %s
                  AND nba_player_id != %s
                  AND report_timestamp_utc <= %s
                ORDER BY nba_player_id, report_timestamp_utc DESC
                """,
                (game_id, team_abbr, exclude_player, as_of),
            )
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return [dict(zip(columns, row)) for row in cur.fetchall()]
