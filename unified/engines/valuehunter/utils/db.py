"""Database connection layer for NBA 3PM Props Engine.

Provides connection management, migration execution, and a Repository class
for all database operations using parameterized queries and ON CONFLICT upserts.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection(database_url: Optional[str] = None):
    """Return a new psycopg2 connection.

    Parameters
    ----------
    database_url : str, optional
        A PostgreSQL connection string.  Falls back to the ``DATABASE_URL``
        environment variable when *None*.

    Returns
    -------
    psycopg2.extensions.connection
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "No database URL provided.  Pass one explicitly or set the "
            "DATABASE_URL environment variable."
        )
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

def execute_migration(conn, migration_path: str) -> None:
    """Execute a single SQL migration file inside a transaction.

    Parameters
    ----------
    conn : psycopg2 connection
    migration_path : str
        Path to a ``.sql`` file.
    """
    path = pathlib.Path(migration_path)
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Applied migration: %s", path.name)


def run_all_migrations(conn, migrations_dir: str) -> None:
    """Run every ``.sql`` file inside *migrations_dir* in sorted order.

    Parameters
    ----------
    conn : psycopg2 connection
    migrations_dir : str
        Directory containing ``.sql`` migration files.
    """
    mdir = pathlib.Path(migrations_dir)
    if not mdir.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {mdir}")

    sql_files = sorted(mdir.glob("*.sql"))
    if not sql_files:
        logger.warning("No .sql files found in %s", mdir)
        return

    for sql_file in sql_files:
        logger.info("Running migration: %s", sql_file.name)
        execute_migration(conn, str(sql_file))

    logger.info("All %d migrations applied.", len(sql_files))


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class Repository:
    """Data-access layer wrapping common NBA props database operations.

    All mutations use parameterized queries to prevent SQL injection.
    Upserts rely on ``ON CONFLICT`` clauses that match the database schema's
    unique constraints.
    """

    def __init__(self, conn):
        self.conn = conn

    # -- helpers -------------------------------------------------------------

    def _execute(self, sql: str, params: tuple | dict = ()) -> None:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def _execute_returning(self, sql: str, params: tuple | dict = ()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def _fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def _fetchone(self, sql: str, params: tuple | dict = ()) -> dict | None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    # -- upserts -------------------------------------------------------------

    def upsert_game(self, game: dict) -> None:
        """Upsert a row into the ``games`` table.

        Expects keys matching the ``games`` columns.  The natural key is
        ``nba_game_id``.
        """
        sql = """
            INSERT INTO games (
                nba_game_id, season, season_type, game_date,
                tipoff_time_utc, home_team_abbr, away_team_abbr,
                sr_game_id, closing_spread_home, closing_total
            ) VALUES (
                %(nba_game_id)s, %(season)s, %(season_type)s, %(game_date)s,
                %(tipoff_time_utc)s, %(home_team_abbr)s, %(away_team_abbr)s,
                %(sr_game_id)s, %(closing_spread_home)s, %(closing_total)s
            )
            ON CONFLICT (nba_game_id) DO UPDATE SET
                season            = EXCLUDED.season,
                season_type       = EXCLUDED.season_type,
                game_date         = EXCLUDED.game_date,
                tipoff_time_utc   = EXCLUDED.tipoff_time_utc,
                home_team_abbr    = EXCLUDED.home_team_abbr,
                away_team_abbr    = EXCLUDED.away_team_abbr,
                sr_game_id        = EXCLUDED.sr_game_id,
                closing_spread_home = EXCLUDED.closing_spread_home,
                closing_total     = EXCLUDED.closing_total
        """
        self._execute(sql, game)
        self.conn.commit()

    def upsert_player_game(self, row: dict) -> None:
        """Upsert a row into the ``player_games`` table.

        Natural key: ``(nba_game_id, nba_player_id)``.
        """
        sql = """
            INSERT INTO player_games (
                nba_game_id, nba_player_id, team_abbr, opponent_abbr,
                is_home, started, minutes_played, three_pa, three_pm,
                fg3_pct, usage_rate, assist_rate, turnovers, personal_fouls,
                rest_days, is_back_to_back, is_3in4
            ) VALUES (
                %(nba_game_id)s, %(nba_player_id)s, %(team_abbr)s,
                %(opponent_abbr)s, %(is_home)s, %(started)s,
                %(minutes_played)s, %(three_pa)s, %(three_pm)s,
                %(fg3_pct)s, %(usage_rate)s, %(assist_rate)s,
                %(turnovers)s, %(personal_fouls)s, %(rest_days)s,
                %(is_back_to_back)s, %(is_3in4)s
            )
            ON CONFLICT (nba_game_id, nba_player_id) DO UPDATE SET
                team_abbr       = EXCLUDED.team_abbr,
                opponent_abbr   = EXCLUDED.opponent_abbr,
                is_home         = EXCLUDED.is_home,
                started         = EXCLUDED.started,
                minutes_played  = EXCLUDED.minutes_played,
                three_pa        = EXCLUDED.three_pa,
                three_pm        = EXCLUDED.three_pm,
                fg3_pct         = EXCLUDED.fg3_pct,
                usage_rate      = EXCLUDED.usage_rate,
                assist_rate     = EXCLUDED.assist_rate,
                turnovers       = EXCLUDED.turnovers,
                personal_fouls  = EXCLUDED.personal_fouls,
                rest_days       = EXCLUDED.rest_days,
                is_back_to_back = EXCLUDED.is_back_to_back,
                is_3in4         = EXCLUDED.is_3in4
        """
        self._execute(sql, row)
        self.conn.commit()

    def upsert_player_tracking(self, row: dict) -> None:
        """Upsert a row into the ``player_tracking`` table.

        Natural key: ``(nba_game_id, nba_player_id)``.
        """
        sql = """
            INSERT INTO player_tracking (
                nba_game_id, nba_player_id, tracking_available,
                tracking_provider, touches, passes_made, passes_received,
                time_of_possession_sec, avg_seconds_per_touch,
                avg_dribbles_per_touch, catch_shoot_fga, catch_shoot_fgm,
                catch_shoot_3pa, catch_shoot_3pm, pull_up_fga, pull_up_fgm,
                pull_up_3pa, pull_up_3pm, potential_assists, secondary_assists
            ) VALUES (
                %(nba_game_id)s, %(nba_player_id)s, %(tracking_available)s,
                %(tracking_provider)s, %(touches)s, %(passes_made)s,
                %(passes_received)s, %(time_of_possession_sec)s,
                %(avg_seconds_per_touch)s, %(avg_dribbles_per_touch)s,
                %(catch_shoot_fga)s, %(catch_shoot_fgm)s,
                %(catch_shoot_3pa)s, %(catch_shoot_3pm)s,
                %(pull_up_fga)s, %(pull_up_fgm)s,
                %(pull_up_3pa)s, %(pull_up_3pm)s,
                %(potential_assists)s, %(secondary_assists)s
            )
            ON CONFLICT (nba_game_id, nba_player_id) DO UPDATE SET
                tracking_available     = EXCLUDED.tracking_available,
                tracking_provider      = EXCLUDED.tracking_provider,
                touches                = EXCLUDED.touches,
                passes_made            = EXCLUDED.passes_made,
                passes_received        = EXCLUDED.passes_received,
                time_of_possession_sec = EXCLUDED.time_of_possession_sec,
                avg_seconds_per_touch  = EXCLUDED.avg_seconds_per_touch,
                avg_dribbles_per_touch = EXCLUDED.avg_dribbles_per_touch,
                catch_shoot_fga        = EXCLUDED.catch_shoot_fga,
                catch_shoot_fgm        = EXCLUDED.catch_shoot_fgm,
                catch_shoot_3pa        = EXCLUDED.catch_shoot_3pa,
                catch_shoot_3pm        = EXCLUDED.catch_shoot_3pm,
                pull_up_fga            = EXCLUDED.pull_up_fga,
                pull_up_fgm            = EXCLUDED.pull_up_fgm,
                pull_up_3pa            = EXCLUDED.pull_up_3pa,
                pull_up_3pm            = EXCLUDED.pull_up_3pm,
                potential_assists      = EXCLUDED.potential_assists,
                secondary_assists      = EXCLUDED.secondary_assists
        """
        self._execute(sql, row)
        self.conn.commit()

    def upsert_team_opponent_shooting(self, row: dict) -> None:
        """Upsert a row into the ``team_opponent_shooting`` table.

        Natural key: ``(nba_game_id, team_abbr)``.
        """
        sql = """
            INSERT INTO team_opponent_shooting (
                nba_game_id, team_abbr, opp_3pa, opp_3pm, opp_3p_pct,
                opp_corner_3pa, opp_corner_3pm, opp_corner_3p_pct,
                opp_above_break_3pa, opp_above_break_3pm,
                opp_above_break_3p_pct, defensive_rating, pace
            ) VALUES (
                %(nba_game_id)s, %(team_abbr)s, %(opp_3pa)s, %(opp_3pm)s,
                %(opp_3p_pct)s, %(opp_corner_3pa)s, %(opp_corner_3pm)s,
                %(opp_corner_3p_pct)s, %(opp_above_break_3pa)s,
                %(opp_above_break_3pm)s, %(opp_above_break_3p_pct)s,
                %(defensive_rating)s, %(pace)s
            )
            ON CONFLICT (nba_game_id, team_abbr) DO UPDATE SET
                opp_3pa                = EXCLUDED.opp_3pa,
                opp_3pm                = EXCLUDED.opp_3pm,
                opp_3p_pct             = EXCLUDED.opp_3p_pct,
                opp_corner_3pa         = EXCLUDED.opp_corner_3pa,
                opp_corner_3pm         = EXCLUDED.opp_corner_3pm,
                opp_corner_3p_pct      = EXCLUDED.opp_corner_3p_pct,
                opp_above_break_3pa    = EXCLUDED.opp_above_break_3pa,
                opp_above_break_3pm    = EXCLUDED.opp_above_break_3pm,
                opp_above_break_3p_pct = EXCLUDED.opp_above_break_3p_pct,
                defensive_rating       = EXCLUDED.defensive_rating,
                pace                   = EXCLUDED.pace
        """
        self._execute(sql, row)
        self.conn.commit()

    # -- inserts (append-only tables) ----------------------------------------

    def insert_injury_snapshot(self, snapshot: dict) -> int:
        """Insert a row into ``injury_snapshots`` and return its id."""
        sql = """
            INSERT INTO injury_snapshots (
                nba_game_id, nba_player_id, team_abbr,
                report_timestamp_utc, report_source, status,
                reason_text, report_url, report_hash,
                minutes_limit_flag
            ) VALUES (
                %(nba_game_id)s, %(nba_player_id)s, %(team_abbr)s,
                %(report_timestamp_utc)s, %(report_source)s, %(status)s,
                %(reason_text)s, %(report_url)s, %(report_hash)s,
                %(minutes_limit_flag)s
            )
            RETURNING injury_report_id
        """
        return self._execute_returning(sql, snapshot)

    def insert_odds_prop(self, prop: dict) -> int:
        """Insert a row into ``odds_props`` and return its id."""
        sql = """
            INSERT INTO odds_props (
                snapshot_timestamp_utc, sportsbook, market,
                nba_game_id, nba_player_id, line,
                over_price, under_price,
                over_implied_prob_raw, under_implied_prob_raw,
                over_implied_prob_novig, under_implied_prob_novig,
                hold_pct, is_closing_snapshot, source
            ) VALUES (
                %(snapshot_timestamp_utc)s, %(sportsbook)s, %(market)s,
                %(nba_game_id)s, %(nba_player_id)s, %(line)s,
                %(over_price)s, %(under_price)s,
                %(over_implied_prob_raw)s, %(under_implied_prob_raw)s,
                %(over_implied_prob_novig)s, %(under_implied_prob_novig)s,
                %(hold_pct)s, %(is_closing_snapshot)s, %(source)s
            )
            RETURNING odds_prop_id
        """
        return self._execute_returning(sql, prop)

    def insert_feature_snapshot(self, snapshot: dict) -> int:
        """Insert a row into ``feature_snapshots`` and return its id."""
        sql = """
            INSERT INTO feature_snapshots (
                nba_game_id, nba_player_id, snapshot_timestamp_utc,
                feature_vector, feature_version
            ) VALUES (
                %(nba_game_id)s, %(nba_player_id)s,
                %(snapshot_timestamp_utc)s,
                %(feature_vector)s, %(feature_version)s
            )
            RETURNING feature_snapshot_id
        """
        return self._execute_returning(sql, snapshot)

    def insert_model_run(self, run: dict) -> int:
        """Insert a row into ``model_runs`` and return its id."""
        sql = """
            INSERT INTO model_runs (
                model_name, model_version, run_timestamp_utc,
                parameters, notes
            ) VALUES (
                %(model_name)s, %(model_version)s, %(run_timestamp_utc)s,
                %(parameters)s, %(notes)s
            )
            RETURNING model_run_id
        """
        return self._execute_returning(sql, run)

    def insert_bet_decision(self, decision: dict) -> int:
        """Insert a row into ``bet_decisions`` and return its id."""
        sql = """
            INSERT INTO bet_decisions (
                feature_snapshot_id, model_run_id,
                nba_game_id, nba_player_id, sportsbook,
                line, odds_over, odds_under,
                model_p_over, model_p_under,
                fair_odds_over, fair_odds_under,
                edge_over, edge_under,
                recommended_side, stake_pct,
                decision_timestamp_utc, tracking_available,
                close_over_prob_novig, close_under_prob_novig,
                clv_prob_pts, actual_3pm, bet_result, pnl_units
            ) VALUES (
                %(feature_snapshot_id)s, %(model_run_id)s,
                %(nba_game_id)s, %(nba_player_id)s, %(sportsbook)s,
                %(line)s, %(odds_over)s, %(odds_under)s,
                %(model_p_over)s, %(model_p_under)s,
                %(fair_odds_over)s, %(fair_odds_under)s,
                %(edge_over)s, %(edge_under)s,
                %(recommended_side)s, %(stake_pct)s,
                %(decision_timestamp_utc)s, %(tracking_available)s,
                %(close_over_prob_novig)s, %(close_under_prob_novig)s,
                %(clv_prob_pts)s, %(actual_3pm)s,
                %(bet_result)s, %(pnl_units)s
            )
            RETURNING decision_id
        """
        return self._execute_returning(sql, decision)

    # -- reads ---------------------------------------------------------------

    def get_player_games(
        self,
        player_id: str,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Return player-game rows for *player_id*, newest first.

        Optionally filtered by *season* and capped at *limit* rows.
        """
        clauses = ["pg.nba_player_id = %s"]
        params: list = [player_id]

        if season is not None:
            clauses.append("g.season = %s")
            params.append(season)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT pg.*, g.game_date, g.season, g.season_type
            FROM player_games pg
            JOIN games g ON g.nba_game_id = pg.nba_game_id
            WHERE {where}
            ORDER BY g.game_date DESC
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        return self._fetchall(sql, tuple(params))

    def get_player_tracking(
        self,
        player_id: str,
        game_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return tracking rows for *player_id*.

        If *game_ids* is provided only those games are returned.
        """
        if game_ids:
            placeholders = ",".join(["%s"] * len(game_ids))
            sql = f"""
                SELECT * FROM player_tracking
                WHERE nba_player_id = %s
                  AND nba_game_id IN ({placeholders})
                ORDER BY nba_game_id
            """
            params = [player_id, *game_ids]
        else:
            sql = """
                SELECT * FROM player_tracking
                WHERE nba_player_id = %s
                ORDER BY nba_game_id
            """
            params = [player_id]

        return self._fetchall(sql, tuple(params))

    def get_injury_snapshot_as_of(
        self,
        player_id: str,
        game_id: str,
        as_of_utc,
    ) -> dict | None:
        """Return the most recent injury snapshot for a player/game
        recorded on or before *as_of_utc*, or *None*.
        """
        sql = """
            SELECT * FROM injury_snapshots
            WHERE nba_player_id = %s
              AND nba_game_id = %s
              AND report_timestamp_utc <= %s
            ORDER BY report_timestamp_utc DESC
            LIMIT 1
        """
        return self._fetchone(sql, (player_id, game_id, as_of_utc))

    def get_odds_snapshots(
        self,
        game_id: str,
        player_id: Optional[str] = None,
        market: str = "player_3pt_made",
    ) -> list[dict]:
        """Return odds prop snapshots for a game, optionally filtered by
        *player_id* and *market*.
        """
        clauses = ["nba_game_id = %s", "market = %s"]
        params: list = [game_id, market]

        if player_id is not None:
            clauses.append("nba_player_id = %s")
            params.append(player_id)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM odds_props
            WHERE {where}
            ORDER BY snapshot_timestamp_utc
        """
        return self._fetchall(sql, tuple(params))

    def get_closing_odds(
        self,
        game_id: str,
        player_id: str,
        market: str = "player_3pt_made",
    ) -> list[dict]:
        """Return the closing odds snapshot(s) for a player/game/market."""
        sql = """
            SELECT * FROM odds_props
            WHERE nba_game_id = %s
              AND nba_player_id = %s
              AND market = %s
              AND is_closing_snapshot = TRUE
            ORDER BY snapshot_timestamp_utc DESC
        """
        return self._fetchall(sql, (game_id, player_id, market))

    def get_team_opponent_shooting(
        self,
        team_abbr: str,
        game_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return opponent-shooting rows for *team_abbr*.

        If *game_ids* is provided only those games are returned.
        """
        if game_ids:
            placeholders = ",".join(["%s"] * len(game_ids))
            sql = f"""
                SELECT * FROM team_opponent_shooting
                WHERE team_abbr = %s
                  AND nba_game_id IN ({placeholders})
                ORDER BY nba_game_id
            """
            params = [team_abbr, *game_ids]
        else:
            sql = """
                SELECT * FROM team_opponent_shooting
                WHERE team_abbr = %s
                ORDER BY nba_game_id
            """
            params = [team_abbr]

        return self._fetchall(sql, tuple(params))
