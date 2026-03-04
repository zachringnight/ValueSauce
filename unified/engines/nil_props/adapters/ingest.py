"""Ingest pipeline — fetches, snapshots, normalizes, and stores data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from engines.nil_props.config.settings import Settings
from engines.nil_props.adapters.base import AdapterResult, BaseAdapter
from engines.nil_props.utils.odds import american_to_implied

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrates data ingestion from adapters to database."""

    def __init__(self, session: Session, settings: Settings, data_dir: Path | None = None):
        self.session = session
        self.settings = settings
        self.data_dir = data_dir or Path(settings.data_dir)

    def run_historical(self, adapter: BaseAdapter, season: str = "2024-25") -> int:
        """Full historical ingest from an adapter."""
        run_id = self._start_run(adapter.SOURCE_NAME)
        try:
            result = adapter.fetch_historical(season)
            adapter.persist_raw_snapshot(result, self.data_dir)
            validation = adapter.validate_records(result.records)
            logger.info(f"Validation: {validation}")

            stored = self._store_records(result)
            self._finish_run(run_id, result.record_count, stored)
            self.session.commit()
            return stored
        except Exception as e:
            self._fail_run(run_id, str(e))
            self.session.commit()
            raise

    def _start_run(self, source: str) -> int:
        self.session.execute(
            text("""INSERT INTO raw_ingest_runs (source, mode, started_at, status)
                    VALUES (:source, :mode, :started_at, 'running')"""),
            {"source": source, "mode": self.settings.source_mode.value,
             "started_at": datetime.utcnow()},
        )
        self.session.flush()
        row = self.session.execute(text("SELECT MAX(id) FROM raw_ingest_runs")).scalar()
        return row

    def _finish_run(self, run_id: int, fetched: int, stored: int):
        self.session.execute(
            text("""UPDATE raw_ingest_runs
                    SET status='success', finished_at=:now,
                        records_fetched=:fetched, records_stored=:stored
                    WHERE id=:id"""),
            {"id": run_id, "now": datetime.utcnow(), "fetched": fetched, "stored": stored},
        )

    def _fail_run(self, run_id: int, error: str):
        self.session.execute(
            text("""UPDATE raw_ingest_runs
                    SET status='failed', finished_at=:now, error_message=:error
                    WHERE id=:id"""),
            {"id": run_id, "now": datetime.utcnow(), "error": error},
        )

    def _store_records(self, result: AdapterResult) -> int:
        stored = 0
        for rec in result.records:
            rtype = rec.get("record_type")
            try:
                if rtype == "team":
                    self._upsert_team(rec)
                elif rtype == "player":
                    self._upsert_player(rec)
                elif rtype == "game":
                    self._upsert_game(rec)
                elif rtype == "player_game":
                    self._upsert_player_game(rec, result.source)
                elif rtype == "team_game":
                    self._upsert_team_game(rec, result.source)
                elif rtype == "injury":
                    self._insert_injury(rec)
                elif rtype == "odds_snapshot":
                    self._insert_odds_snapshot(rec)
                elif rtype == "odds_closing":
                    self._upsert_odds_closing(rec)
                else:
                    logger.warning(f"Unknown record type: {rtype}")
                    continue
                stored += 1
            except Exception as e:
                logger.error(f"Failed to store record {rtype}: {e}")
        return stored

    def _upsert_team(self, rec: dict):
        existing = self.session.execute(
            text("SELECT team_id FROM teams WHERE team_id = :id"),
            {"id": rec["team_id"]},
        ).fetchone()
        if existing:
            self.session.execute(
                text("""UPDATE teams SET full_name=:fn, abbreviation=:abbr,
                        conference=:conf, division=:div, updated_at=:now
                        WHERE team_id=:id"""),
                {"id": rec["team_id"], "fn": rec["full_name"], "abbr": rec["abbreviation"],
                 "conf": rec.get("conference"), "div": rec.get("division"),
                 "now": datetime.utcnow()},
            )
        else:
            self.session.execute(
                text("""INSERT INTO teams (team_id, full_name, abbreviation, conference, division, source)
                        VALUES (:id, :fn, :abbr, :conf, :div, :src)"""),
                {"id": rec["team_id"], "fn": rec["full_name"], "abbr": rec["abbreviation"],
                 "conf": rec.get("conference"), "div": rec.get("division"),
                 "src": rec.get("source", "mock")},
            )

    def _upsert_player(self, rec: dict):
        existing = self.session.execute(
            text("SELECT player_id FROM players WHERE player_id = :id"),
            {"id": rec["player_id"]},
        ).fetchone()
        if existing:
            self.session.execute(
                text("""UPDATE players SET full_name=:fn, team_id=:tid, position=:pos,
                        is_active=:active, updated_at=:now
                        WHERE player_id=:id"""),
                {"id": rec["player_id"], "fn": rec["full_name"], "tid": rec.get("team_id"),
                 "pos": rec.get("position"), "active": rec.get("is_active", True),
                 "now": datetime.utcnow()},
            )
        else:
            self.session.execute(
                text("""INSERT INTO players (player_id, full_name, first_name, last_name,
                        team_id, position, height_inches, weight_lbs, is_active, source)
                        VALUES (:id, :fn, :first, :last, :tid, :pos, :h, :w, :active, :src)"""),
                {"id": rec["player_id"], "fn": rec["full_name"],
                 "first": rec.get("first_name"), "last": rec.get("last_name"),
                 "tid": rec.get("team_id"), "pos": rec.get("position"),
                 "h": rec.get("height_inches"), "w": rec.get("weight_lbs"),
                 "active": rec.get("is_active", True), "src": rec.get("source", "mock")},
            )

    def _upsert_game(self, rec: dict):
        existing = self.session.execute(
            text("SELECT game_id FROM games WHERE game_id = :id"),
            {"id": rec["game_id"]},
        ).fetchone()
        if existing:
            self.session.execute(
                text("""UPDATE games SET status=:status, home_score=:hs, away_score=:as_,
                        updated_at=:now WHERE game_id=:id"""),
                {"id": rec["game_id"], "status": rec.get("status", "final"),
                 "hs": rec.get("home_score"), "as_": rec.get("away_score"),
                 "now": datetime.utcnow()},
            )
        else:
            self.session.execute(
                text("""INSERT INTO games (game_id, season, season_type, game_date,
                        home_team_id, away_team_id, scheduled_tip, status,
                        home_score, away_score, source)
                        VALUES (:id, :season, :stype, :gdate, :home, :away,
                                :tip, :status, :hs, :as_, :src)"""),
                {"id": rec["game_id"], "season": rec.get("season", "2024-25"),
                 "stype": rec.get("season_type", "regular"),
                 "gdate": rec["game_date"],
                 "home": rec["home_team_id"], "away": rec["away_team_id"],
                 "tip": rec.get("scheduled_tip"),
                 "status": rec.get("status", "final"),
                 "hs": rec.get("home_score"), "as_": rec.get("away_score"),
                 "src": rec.get("source", "mock")},
            )

    def _upsert_player_game(self, rec: dict, source: str):
        existing = self.session.execute(
            text("""SELECT id FROM player_game
                    WHERE player_id = :pid AND game_id = :gid"""),
            {"pid": rec["player_id"], "gid": rec["game_id"]},
        ).fetchone()
        if existing:
            return  # Don't overwrite existing stats
        self.session.execute(
            text("""INSERT INTO player_game (player_id, game_id, team_id, minutes, started,
                    assists, potential_assists, points, rebounds, turnovers, steals, blocks,
                    fouls, field_goals_made, field_goals_attempted,
                    free_throws_made, free_throws_attempted,
                    usage_rate, touches, passes_made, time_of_possession, source)
                    VALUES (:pid, :gid, :tid, :min, :started, :ast, :past, :pts, :reb,
                            :tov, :stl, :blk, :fouls, :fgm, :fga, :ftm, :fta,
                            :usg, :tch, :passes, :top, :src)"""),
            {"pid": rec["player_id"], "gid": rec["game_id"], "tid": rec["team_id"],
             "min": rec.get("minutes"), "started": rec.get("started"),
             "ast": rec.get("assists"), "past": rec.get("potential_assists"),
             "pts": rec.get("points"), "reb": rec.get("rebounds"),
             "tov": rec.get("turnovers"), "stl": rec.get("steals"),
             "blk": rec.get("blocks"), "fouls": rec.get("fouls"),
             "fgm": rec.get("field_goals_made"), "fga": rec.get("field_goals_attempted"),
             "ftm": rec.get("free_throws_made"), "fta": rec.get("free_throws_attempted"),
             "usg": rec.get("usage_rate"), "tch": rec.get("touches"),
             "passes": rec.get("passes_made"), "top": rec.get("time_of_possession"),
             "src": source},
        )

    def _upsert_team_game(self, rec: dict, source: str):
        existing = self.session.execute(
            text("""SELECT id FROM team_game
                    WHERE team_id = :tid AND game_id = :gid"""),
            {"tid": rec["team_id"], "gid": rec["game_id"]},
        ).fetchone()
        if existing:
            return
        self.session.execute(
            text("""INSERT INTO team_game (team_id, game_id, is_home, points, assists,
                    rebounds, turnovers, pace, offensive_rating, defensive_rating,
                    possessions, source)
                    VALUES (:tid, :gid, :home, :pts, :ast, :reb, :tov, :pace,
                            :ortg, :drtg, :poss, :src)"""),
            {"tid": rec["team_id"], "gid": rec["game_id"],
             "home": rec.get("is_home"), "pts": rec.get("points"),
             "ast": rec.get("assists"), "reb": rec.get("rebounds"),
             "tov": rec.get("turnovers"), "pace": rec.get("pace"),
             "ortg": rec.get("offensive_rating"), "drtg": rec.get("defensive_rating"),
             "poss": rec.get("possessions"), "src": source},
        )

    def _insert_injury(self, rec: dict):
        self.session.execute(
            text("""INSERT INTO injury_reports (player_id, team_id, game_id,
                    report_timestamp, status_timestamp, status, reason, source)
                    VALUES (:pid, :tid, :gid, :rts, :sts, :status, :reason, :src)"""),
            {"pid": rec["player_id"], "tid": rec["team_id"], "gid": rec.get("game_id"),
             "rts": rec["report_timestamp"], "sts": rec.get("status_timestamp"),
             "status": rec["status"], "reason": rec.get("reason"),
             "src": rec.get("source", "mock")},
        )

    def _insert_odds_snapshot(self, rec: dict):
        over_prob = None
        under_prob = None
        try:
            over_prob = american_to_implied(rec["over_price"])
            under_prob = american_to_implied(rec["under_price"])
        except (ValueError, KeyError):
            pass

        # Ensure sportsbook exists
        existing = self.session.execute(
            text("SELECT sportsbook_id FROM sportsbooks WHERE sportsbook_id = :id"),
            {"id": rec["sportsbook_id"]},
        ).fetchone()
        if not existing:
            self.session.execute(
                text("INSERT INTO sportsbooks (sportsbook_id, name) VALUES (:id, :name)"),
                {"id": rec["sportsbook_id"], "name": rec["sportsbook_id"]},
            )

        # Ensure market exists
        existing = self.session.execute(
            text("SELECT market_id FROM markets WHERE market_id = :id"),
            {"id": rec.get("market_id", "player_assists_ou")},
        ).fetchone()
        if not existing:
            self.session.execute(
                text("""INSERT INTO markets (market_id, market_type, description)
                        VALUES (:id, :type, :desc)"""),
                {"id": rec.get("market_id", "player_assists_ou"),
                 "type": "player_assists", "desc": "Player assists over/under"},
            )

        self.session.execute(
            text("""INSERT INTO odds_props_snapshots (snapshot_timestamp, sportsbook_id,
                    player_id, game_id, market_id, line, over_price, under_price,
                    over_prob, under_prob, source)
                    VALUES (:ts, :book, :pid, :gid, :mid, :line, :op, :up,
                            :oprob, :uprob, :src)"""),
            {"ts": rec["snapshot_timestamp"], "book": rec["sportsbook_id"],
             "pid": rec["player_id"], "gid": rec["game_id"],
             "mid": rec.get("market_id", "player_assists_ou"),
             "line": rec["line"], "op": rec["over_price"], "up": rec["under_price"],
             "oprob": over_prob, "uprob": under_prob,
             "src": rec.get("source", "mock")},
        )

    def _upsert_odds_closing(self, rec: dict):
        over_prob = None
        under_prob = None
        try:
            over_prob = american_to_implied(rec["over_price"])
            under_prob = american_to_implied(rec["under_price"])
        except (ValueError, KeyError):
            pass

        # Ensure sportsbook exists
        existing = self.session.execute(
            text("SELECT sportsbook_id FROM sportsbooks WHERE sportsbook_id = :id"),
            {"id": rec["sportsbook_id"]},
        ).fetchone()
        if not existing:
            self.session.execute(
                text("INSERT INTO sportsbooks (sportsbook_id, name) VALUES (:id, :name)"),
                {"id": rec["sportsbook_id"], "name": rec["sportsbook_id"]},
            )

        # Ensure market exists
        mid = rec.get("market_id", "player_assists_ou")
        existing = self.session.execute(
            text("SELECT market_id FROM markets WHERE market_id = :id"),
            {"id": mid},
        ).fetchone()
        if not existing:
            self.session.execute(
                text("""INSERT INTO markets (market_id, market_type, description)
                        VALUES (:id, :type, :desc)"""),
                {"id": mid, "type": "player_assists",
                 "desc": "Player assists over/under"},
            )

        self.session.execute(
            text("""INSERT OR REPLACE INTO odds_props_closing
                    (sportsbook_id, player_id, game_id, market_id, line,
                     over_price, under_price, over_prob, under_prob, source)
                    VALUES (:book, :pid, :gid, :mid, :line, :op, :up,
                            :oprob, :uprob, :src)"""),
            {"book": rec["sportsbook_id"], "pid": rec["player_id"],
             "gid": rec["game_id"],
             "mid": mid,
             "line": rec["line"], "op": rec["over_price"], "up": rec["under_price"],
             "oprob": over_prob, "uprob": under_prob,
             "src": rec.get("source", "mock")},
        )
