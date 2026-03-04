"""ID reconciliation and data quality validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    """Container for reconciliation results."""

    duplicate_player_games: list[dict] = field(default_factory=list)
    mismatched_teams: list[dict] = field(default_factory=list)
    orphan_odds: list[dict] = field(default_factory=list)
    orphan_injuries: list[dict] = field(default_factory=list)
    quarantined_rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            len(self.duplicate_player_games) == 0
            and len(self.mismatched_teams) == 0
            and len(self.orphan_odds) == 0
            and len(self.orphan_injuries) == 0
        )

    def summary(self) -> str:
        lines = [
            f"Duplicate player-game rows: {len(self.duplicate_player_games)}",
            f"Mismatched teams: {len(self.mismatched_teams)}",
            f"Orphan odds rows: {len(self.orphan_odds)}",
            f"Orphan injury rows: {len(self.orphan_injuries)}",
            f"Quarantined rows: {len(self.quarantined_rows)}",
            f"Warnings: {len(self.warnings)}",
        ]
        return "\n".join(lines)


class Reconciler:
    """Validates data quality and reconciles IDs across sources."""

    def __init__(self, session: Session):
        self.session = session

    def run_full_check(self) -> ReconciliationReport:
        report = ReconciliationReport()
        self._check_duplicate_player_games(report)
        self._check_team_consistency(report)
        self._check_orphan_odds(report)
        self._check_orphan_injuries(report)
        self._check_player_team_consistency(report)
        logger.info(f"Reconciliation report:\n{report.summary()}")
        return report

    def _check_duplicate_player_games(self, report: ReconciliationReport):
        """Check for duplicate player-game rows."""
        rows = self.session.execute(
            text("""SELECT player_id, game_id, COUNT(*) as cnt
                    FROM player_game
                    GROUP BY player_id, game_id
                    HAVING cnt > 1""")
        ).fetchall()
        for row in rows:
            report.duplicate_player_games.append({
                "player_id": row[0], "game_id": row[1], "count": row[2],
            })
            report.warnings.append(
                f"Duplicate player_game: player={row[0]} game={row[1]} count={row[2]}"
            )

    def _check_team_consistency(self, report: ReconciliationReport):
        """Check home/away team consistency between games and player_game."""
        rows = self.session.execute(
            text("""SELECT pg.player_id, pg.game_id, pg.team_id,
                           g.home_team_id, g.away_team_id
                    FROM player_game pg
                    JOIN games g ON pg.game_id = g.game_id
                    WHERE pg.team_id != g.home_team_id
                      AND pg.team_id != g.away_team_id""")
        ).fetchall()
        for row in rows:
            report.mismatched_teams.append({
                "player_id": row[0], "game_id": row[1],
                "player_team": row[2], "home": row[3], "away": row[4],
            })
            report.warnings.append(
                f"Team mismatch: player={row[0]} has team={row[2]} "
                f"but game has home={row[3]}, away={row[4]}"
            )

    def _check_orphan_odds(self, report: ReconciliationReport):
        """Check for odds rows referencing non-existent players or games."""
        rows = self.session.execute(
            text("""SELECT o.id, o.player_id, o.game_id
                    FROM odds_props_snapshots o
                    LEFT JOIN players p ON o.player_id = p.player_id
                    LEFT JOIN games g ON o.game_id = g.game_id
                    WHERE p.player_id IS NULL OR g.game_id IS NULL""")
        ).fetchall()
        for row in rows:
            report.orphan_odds.append({
                "odds_id": row[0], "player_id": row[1], "game_id": row[2],
            })

    def _check_orphan_injuries(self, report: ReconciliationReport):
        """Check for injury rows referencing non-existent players."""
        rows = self.session.execute(
            text("""SELECT ir.id, ir.player_id, ir.team_id
                    FROM injury_reports ir
                    LEFT JOIN players p ON ir.player_id = p.player_id
                    WHERE p.player_id IS NULL""")
        ).fetchall()
        for row in rows:
            report.orphan_injuries.append({
                "injury_id": row[0], "player_id": row[1], "team_id": row[2],
            })

    def _check_player_team_consistency(self, report: ReconciliationReport):
        """Warn if a player appears on multiple teams in the same season."""
        rows = self.session.execute(
            text("""SELECT pg.player_id, COUNT(DISTINCT pg.team_id) as team_count
                    FROM player_game pg
                    GROUP BY pg.player_id
                    HAVING team_count > 1""")
        ).fetchall()
        for row in rows:
            # This is valid (trades), but worth flagging
            report.warnings.append(
                f"Player {row[0]} appeared on {row[1]} different teams (possible trade)"
            )
