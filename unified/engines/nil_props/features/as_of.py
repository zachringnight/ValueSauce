"""As-of snapshot engine — reconstructs state at a given pregame timestamp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class AsOfState:
    """State snapshot at a given pregame timestamp."""

    timestamp: datetime
    player_id: str
    game_id: str
    team_id: str

    # Injury context
    injury_status: str | None = None
    injury_reason: str | None = None
    teammate_injuries: list[dict] = field(default_factory=list)

    # Odds context
    latest_odds: dict | None = None

    # Player history (already filtered to before timestamp)
    player_history: pd.DataFrame | None = None

    # Team context
    team_history: pd.DataFrame | None = None
    opponent_history: pd.DataFrame | None = None

    # Game context
    game_info: dict | None = None
    is_home: bool | None = None
    opponent_team_id: str | None = None


class AsOfEngine:
    """Reconstructs knowable state at a pregame timestamp."""

    def __init__(self, session: Session):
        self.session = session

    def get_state(
        self,
        player_id: str,
        game_id: str,
        as_of: datetime,
    ) -> AsOfState:
        """Build full as-of state for a player/game."""
        game_info = self._get_game_info(game_id)
        if not game_info:
            raise ValueError(f"Game not found: {game_id}")

        team_id = self._get_player_team(player_id, game_id)
        is_home = team_id == game_info["home_team_id"]
        opponent_id = game_info["away_team_id"] if is_home else game_info["home_team_id"]

        state = AsOfState(
            timestamp=as_of,
            player_id=player_id,
            game_id=game_id,
            team_id=team_id,
            is_home=is_home,
            opponent_team_id=opponent_id,
            game_info=game_info,
        )

        state.injury_status, state.injury_reason = self._get_injury_status(
            player_id, game_id, as_of
        )
        state.teammate_injuries = self._get_teammate_injuries(
            team_id, player_id, game_id, as_of
        )
        state.latest_odds = self._get_latest_odds(player_id, game_id, as_of)
        state.player_history = self._get_player_history(player_id, as_of)
        state.team_history = self._get_team_history(team_id, as_of)
        state.opponent_history = self._get_team_history(opponent_id, as_of)

        return state

    def _get_game_info(self, game_id: str) -> dict | None:
        row = self.session.execute(
            text("SELECT * FROM games WHERE game_id = :gid"),
            {"gid": game_id},
        ).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    def _get_player_team(self, player_id: str, game_id: str) -> str:
        # Check player_game first
        row = self.session.execute(
            text("""SELECT team_id FROM player_game
                    WHERE player_id = :pid AND game_id = :gid"""),
            {"pid": player_id, "gid": game_id},
        ).fetchone()
        if row:
            return row[0]
        # Fallback to player table
        row = self.session.execute(
            text("SELECT team_id FROM players WHERE player_id = :pid"),
            {"pid": player_id},
        ).fetchone()
        if row:
            return row[0]
        raise ValueError(f"Cannot determine team for player {player_id}")

    def _get_injury_status(
        self, player_id: str, game_id: str, as_of: datetime
    ) -> tuple[str | None, str | None]:
        row = self.session.execute(
            text("""SELECT status, reason FROM injury_reports
                    WHERE player_id = :pid
                      AND (game_id = :gid OR game_id IS NULL)
                      AND report_timestamp <= :asof
                    ORDER BY report_timestamp DESC LIMIT 1"""),
            {"pid": player_id, "gid": game_id, "asof": as_of},
        ).fetchone()
        if row:
            return row[0], row[1]
        return None, None

    def _get_teammate_injuries(
        self, team_id: str, player_id: str, game_id: str, as_of: datetime
    ) -> list[dict]:
        rows = self.session.execute(
            text("""SELECT ir.player_id, ir.status, ir.reason, p.full_name, p.position
                    FROM injury_reports ir
                    JOIN players p ON ir.player_id = p.player_id
                    WHERE ir.team_id = :tid
                      AND ir.player_id != :pid
                      AND (ir.game_id = :gid OR ir.game_id IS NULL)
                      AND ir.report_timestamp <= :asof
                      AND ir.status IN ('Out', 'Doubtful')
                    ORDER BY ir.report_timestamp DESC"""),
            {"tid": team_id, "pid": player_id, "gid": game_id, "asof": as_of},
        ).fetchall()
        # Deduplicate by player (keep latest)
        seen = set()
        results = []
        for row in rows:
            if row[0] not in seen:
                seen.add(row[0])
                results.append({
                    "player_id": row[0], "status": row[1], "reason": row[2],
                    "name": row[3], "position": row[4],
                })
        return results

    def _get_latest_odds(
        self, player_id: str, game_id: str, as_of: datetime
    ) -> dict | None:
        row = self.session.execute(
            text("""SELECT * FROM odds_props_snapshots
                    WHERE player_id = :pid AND game_id = :gid
                      AND snapshot_timestamp <= :asof
                    ORDER BY snapshot_timestamp DESC LIMIT 1"""),
            {"pid": player_id, "gid": game_id, "asof": as_of},
        ).fetchone()
        if row:
            return dict(row._mapping)
        return None

    def _get_player_history(
        self, player_id: str, as_of: datetime
    ) -> pd.DataFrame:
        """Get player game log strictly before as_of."""
        rows = self.session.execute(
            text("""SELECT pg.*, g.game_date, g.home_team_id, g.away_team_id
                    FROM player_game pg
                    JOIN games g ON pg.game_id = g.game_id
                    WHERE pg.player_id = :pid
                      AND g.game_date < DATE(:asof)
                      AND g.status = 'final'
                    ORDER BY g.game_date DESC"""),
            {"pid": player_id, "asof": as_of.isoformat()},
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r._mapping) for r in rows])

    def _get_team_history(
        self, team_id: str, as_of: datetime
    ) -> pd.DataFrame:
        rows = self.session.execute(
            text("""SELECT tg.*, g.game_date
                    FROM team_game tg
                    JOIN games g ON tg.game_id = g.game_id
                    WHERE tg.team_id = :tid
                      AND g.game_date < DATE(:asof)
                      AND g.status = 'final'
                    ORDER BY g.game_date DESC"""),
            {"tid": team_id, "asof": as_of.isoformat()},
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r._mapping) for r in rows])
