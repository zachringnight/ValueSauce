"""Feature builder — constructs leak-free feature tables for all three model layers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from engines.nil_props.features.as_of import AsOfEngine
from engines.nil_props.utils.time_safety import (
    assert_no_future_data,
    strict_lag_rolling,
    strict_lag_std,
    strict_lag_sum,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds feature tables for minutes, opportunity, and conversion models."""

    def __init__(self, session: Session):
        self.session = session
        self.as_of = AsOfEngine(session)

    def build_all_features(self, as_of_date: str | None = None) -> pd.DataFrame:
        """Build combined feature table for all completed games.

        If as_of_date is provided, only include games up to that date.
        """
        df = self._load_base_table(as_of_date)
        if df.empty:
            logger.warning("No base data to build features from")
            return df

        df = self._add_minutes_features(df)
        df = self._add_opportunity_features(df)
        df = self._add_conversion_features(df)
        df = self._add_context_features(df)
        df = self._add_opponent_features(df)
        df = self._add_injury_features(df)
        df = self._add_prior_features(df)

        return df

    def _load_base_table(self, as_of_date: str | None = None) -> pd.DataFrame:
        """Load player_game joined with game info."""
        date_filter = ""
        params = {}
        if as_of_date:
            date_filter = "AND g.game_date <= :asof"
            params["asof"] = as_of_date

        rows = self.session.execute(
            text(f"""SELECT pg.player_id, pg.game_id, pg.team_id,
                           pg.minutes, pg.started, pg.assists, pg.potential_assists,
                           pg.points, pg.rebounds, pg.turnovers, pg.steals, pg.blocks,
                           pg.fouls, pg.field_goals_made, pg.field_goals_attempted,
                           pg.free_throws_made, pg.free_throws_attempted,
                           pg.usage_rate, pg.touches, pg.passes_made,
                           pg.time_of_possession,
                           g.game_date, g.home_team_id, g.away_team_id,
                           g.scheduled_tip, g.status, g.season,
                           g.home_score, g.away_score,
                           p.position, p.full_name
                    FROM player_game pg
                    JOIN games g ON pg.game_id = g.game_id
                    JOIN players p ON pg.player_id = p.player_id
                    WHERE g.status = 'final'
                    {date_filter}
                    ORDER BY pg.player_id, g.game_date"""),
            params,
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r._mapping) for r in rows])
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["is_home"] = df["team_id"] == df["home_team_id"]
        return df

    def _add_minutes_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: Minutes model features (strictly lagged)."""
        grp = df.groupby("player_id")

        # Rolling minutes
        df["min_last_3"] = grp["minutes"].transform(
            lambda s: strict_lag_rolling(s, 3)
        )
        df["min_last_5"] = grp["minutes"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df["min_last_10"] = grp["minutes"].transform(
            lambda s: strict_lag_rolling(s, 10)
        )
        df["min_std_5"] = grp["minutes"].transform(
            lambda s: strict_lag_std(s, 5)
        )

        # Started last game
        df["started_last"] = grp["started"].transform(
            lambda s: s.shift(1)
        )

        # Starter rate (last 10)
        df["starter_rate_10"] = grp["started"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )

        # Foul rate (fouls per minute, rolling)
        df["_fpm"] = df["fouls"] / df["minutes"].clip(lower=1)
        df["foul_rate_5"] = grp["_fpm"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df.drop(columns=["_fpm"], inplace=True)

        # Rest days
        df["_prev_date"] = grp["game_date"].shift(1)
        df["rest_days"] = (df["game_date"] - df["_prev_date"]).dt.days
        df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)
        df.drop(columns=["_prev_date"], inplace=True)

        # Recent absences (games missed in last 14 days proxy)
        df["games_played_last_10"] = grp["minutes"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).count()
        )

        # Blowout proxy (point differential)
        df["point_diff"] = np.where(
            df["is_home"],
            df["home_score"] - df["away_score"],
            df["away_score"] - df["home_score"],
        ).astype(float)

        return df

    def _add_opportunity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 2: Opportunity model features."""
        grp = df.groupby("player_id")

        # Assists per minute
        df["ast_per_min"] = df["assists"] / df["minutes"].clip(lower=1)

        # Potential assists per minute
        df["past_per_min"] = df["potential_assists"] / df["minutes"].clip(lower=1)

        # Rolling opportunity rates (lagged)
        df["ast_per_min_last_5"] = grp["ast_per_min"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df["ast_per_min_last_10"] = grp["ast_per_min"].transform(
            lambda s: strict_lag_rolling(s, 10)
        )
        df["past_per_min_last_5"] = grp["past_per_min"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df["past_per_min_last_10"] = grp["past_per_min"].transform(
            lambda s: strict_lag_rolling(s, 10)
        )

        # Touches per minute
        df["touches_per_min"] = df["touches"] / df["minutes"].clip(lower=1)
        df["touches_per_min_last_5"] = grp["touches_per_min"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )

        # Passes per minute
        df["passes_per_min"] = df["passes_made"] / df["minutes"].clip(lower=1)
        df["passes_per_min_last_5"] = grp["passes_per_min"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )

        # Usage rate rolling
        df["usg_last_5"] = grp["usage_rate"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )

        # Rolling assists
        df["ast_last_3"] = grp["assists"].transform(
            lambda s: strict_lag_rolling(s, 3)
        )
        df["ast_last_5"] = grp["assists"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df["ast_last_10"] = grp["assists"].transform(
            lambda s: strict_lag_rolling(s, 10)
        )

        return df

    def _add_conversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 3: Conversion model features."""
        # Assist conversion rate (assists / potential_assists)
        df["ast_conversion"] = np.where(
            df["potential_assists"] > 0,
            df["assists"] / df["potential_assists"],
            np.nan,
        )

        grp = df.groupby("player_id")
        df["ast_conversion_last_5"] = grp["ast_conversion"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        df["ast_conversion_last_10"] = grp["ast_conversion"].transform(
            lambda s: strict_lag_rolling(s, 10)
        )

        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Game context features."""
        # Home/away already added
        df["is_home_int"] = df["is_home"].astype(int)

        # Use month directly as season context feature
        df["game_month"] = df["game_date"].dt.month

        # Position encoding
        pos_map = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4,
                    "G": 0.5, "F": 2.5, "G-F": 1, "F-G": 1.5, "F-C": 3.5}
        df["position_code"] = df["position"].map(pos_map).fillna(2)

        return df

    def _add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Opponent defensive features from team_game table."""
        # Load team game stats
        rows = self.session.execute(
            text("""SELECT tg.team_id, tg.game_id, tg.pace, tg.defensive_rating,
                           tg.assists as team_assists, g.game_date
                    FROM team_game tg
                    JOIN games g ON tg.game_id = g.game_id
                    WHERE g.status = 'final'
                    ORDER BY tg.team_id, g.game_date""")
        ).fetchall()

        if not rows:
            df["opp_pace_5"] = np.nan
            df["opp_drtg_5"] = np.nan
            df["opp_ast_allowed_5"] = np.nan
            df["team_pace_5"] = np.nan
            return df

        tg = pd.DataFrame([dict(r._mapping) for r in rows])
        tg["game_date"] = pd.to_datetime(tg["game_date"])

        # Compute rolling opponent stats
        tg_grp = tg.groupby("team_id")
        tg["pace_last_5"] = tg_grp["pace"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        tg["drtg_last_5"] = tg_grp["defensive_rating"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        tg["ast_allowed_last_5"] = tg_grp["team_assists"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )

        # Map opponent stats to each row
        opp_lookup = tg.set_index(["team_id", "game_id"])[
            ["pace_last_5", "drtg_last_5", "ast_allowed_last_5"]
        ].to_dict("index")

        def _get_opp_stat(row, stat):
            key = (row["opponent_team_id"], row["game_id"])
            if key in opp_lookup:
                return opp_lookup[key].get(stat)
            return np.nan

        # Determine opponent
        df["opponent_team_id"] = np.where(
            df["is_home"],
            df["away_team_id"],
            df["home_team_id"],
        )

        df["opp_pace_5"] = df.apply(lambda r: _get_opp_stat(r, "pace_last_5"), axis=1)
        df["opp_drtg_5"] = df.apply(lambda r: _get_opp_stat(r, "drtg_last_5"), axis=1)
        df["opp_ast_allowed_5"] = df.apply(
            lambda r: _get_opp_stat(r, "ast_allowed_last_5"), axis=1
        )

        # Team pace
        team_lookup = tg.set_index(["team_id", "game_id"])["pace_last_5"].to_dict()
        df["team_pace_5"] = df.apply(
            lambda r: team_lookup.get((r["team_id"], r["game_id"]), np.nan), axis=1
        )

        return df

    def _add_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add injury-context features."""
        # Load all injury reports
        rows = self.session.execute(
            text("""SELECT ir.player_id, ir.game_id, ir.status, ir.team_id,
                           p.position
                    FROM injury_reports ir
                    JOIN players p ON ir.player_id = p.player_id
                    WHERE ir.status IN ('Out', 'Doubtful')""")
        ).fetchall()

        if not rows:
            df["teammates_out"] = 0
            df["guard_teammates_out"] = 0
            return df

        inj_df = pd.DataFrame([dict(r._mapping) for r in rows])

        # Count teammates out per game/team
        team_game_out = (
            inj_df.groupby(["team_id", "game_id"])
            .size()
            .reset_index(name="teammates_out_total")
        )

        # Guard teammates out
        guard_inj = inj_df[inj_df["position"].isin(["PG", "SG", "G", "G-F"])]
        guard_out = (
            guard_inj.groupby(["team_id", "game_id"])
            .size()
            .reset_index(name="guard_teammates_out")
        )

        df = df.merge(team_game_out, on=["team_id", "game_id"], how="left")
        df = df.merge(guard_out, on=["team_id", "game_id"], how="left")
        df["teammates_out"] = df["teammates_out_total"].fillna(0).astype(int)
        df["guard_teammates_out"] = df["guard_teammates_out"].fillna(0).astype(int)
        df.drop(columns=["teammates_out_total"], inplace=True, errors="ignore")

        return df

    def _add_prior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prior-game count and positional-prior shrinkage features.

        When a player has few prior games, their rolling stats are unreliable.
        We shrink toward the position-group mean to stabilize projections.
        """
        grp = df.groupby("player_id")

        # Count of prior games available (strictly lagged)
        df["prior_game_count"] = grp.cumcount()  # 0-indexed, so game N has N prior

        # Positional assist priors: compute expanding position-group averages
        # Map to coarse position groups
        pos_group_map = {
            "PG": "guard", "SG": "guard", "G": "guard", "G-F": "guard",
            "SF": "wing", "F": "wing", "F-G": "wing",
            "PF": "big", "F-C": "big", "C": "big",
        }
        df["pos_group"] = df["position"].map(pos_group_map).fillna("wing")

        # Compute position-group average assists (lagged: exclude current row)
        pos_avg = df.groupby("pos_group")["assists"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        df["pos_group_avg_ast"] = pos_avg

        # Shrinkage: blend player's rolling avg with positional prior
        # Weight = prior_game_count / (prior_game_count + shrinkage_k)
        # With k=5: after 5 games, 50% player / 50% position; after 10, 67% player
        shrinkage_k = 5.0
        player_weight = df["prior_game_count"] / (df["prior_game_count"] + shrinkage_k)
        df["ast_shrunk_5"] = (
            player_weight * df["ast_last_5"].fillna(df["pos_group_avg_ast"])
            + (1 - player_weight) * df["pos_group_avg_ast"]
        )

        # Same for potential assists — use rolling count directly, not rate*minutes
        past_last_5 = grp["potential_assists"].transform(
            lambda s: strict_lag_rolling(s, 5)
        )
        pos_past_avg = df.groupby("pos_group")["potential_assists"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        df["pos_group_avg_past"] = pos_past_avg
        df["past_shrunk_5"] = (
            player_weight * past_last_5.fillna(pos_past_avg)
            + (1 - player_weight) * pos_past_avg
        )

        return df
