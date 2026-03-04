"""NBA Stats adapter — uses nba_api to fetch real NBA data from stats.nba.com."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from engines.nil_props.config.settings import Settings
from engines.nil_props.adapters.base import AdapterResult, BaseAdapter

logger = logging.getLogger(__name__)

# Rate limiting: NBA stats.nba.com is sensitive to rapid requests
_REQUEST_DELAY = 0.6  # seconds between requests


def _delay():
    time.sleep(_REQUEST_DELAY)


class NBAStatsAdapter(BaseAdapter):
    """Adapter that pulls real NBA game data from stats.nba.com via nba_api."""

    SOURCE_NAME = "nba_stats"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._headers = {
            "Host": "stats.nba.com",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nba.com/",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
        }

    def authenticate(self) -> bool:
        """No auth needed for stats.nba.com — always returns True."""
        return True

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        """Fetch all games and player stats for a full season."""
        raw = self._fetch_season_data(season)
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        """Fetch recent games since a given date."""
        season = kwargs.get("season", "2024-25")
        raw = self._fetch_season_data(
            season,
            date_from=since.strftime("%m/%d/%Y"),
        )
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def _fetch_season_data(
        self, season: str, date_from: str | None = None
    ) -> dict:
        """Fetch games, player stats, and team advanced stats for a season."""
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import players as nba_players
        from nba_api.stats.static import teams as nba_teams

        # 1. Get all teams
        all_teams = nba_teams.get_teams()
        logger.info(f"Loaded {len(all_teams)} NBA teams")

        # 2. Get all games for the season via LeagueGameFinder (team-level)
        logger.info(f"Fetching games for season {season}...")
        params = {
            "player_or_team_abbreviation": "T",
            "season_nullable": season,
            "season_type_nullable": "Regular Season",
        }
        if date_from:
            params["date_from_nullable"] = date_from

        gf = leaguegamefinder.LeagueGameFinder(
            **params,
            headers=self._headers,
            timeout=60,
        )
        _delay()
        games_df = gf.get_data_frames()[0]
        logger.info(f"Found {len(games_df)} team-game rows")

        # 3. Get unique game IDs
        game_ids = games_df["GAME_ID"].unique().tolist()
        logger.info(f"Found {len(game_ids)} unique games")

        # 4. Fetch box scores (traditional + advanced) for each game
        box_scores = []
        team_advanced = []
        batch_size = 20
        for i, gid in enumerate(game_ids):
            if i > 0 and i % batch_size == 0:
                logger.info(f"  Fetched {i}/{len(game_ids)} box scores...")

            try:
                trad, adv = self._fetch_box_score(gid)
                if trad is not None:
                    box_scores.append(trad)
                if adv is not None:
                    team_advanced.append(adv)
            except Exception as e:
                logger.warning(f"Box score fetch failed for {gid}: {e}")
                _delay()

        logger.info(f"Fetched {len(box_scores)} box scores")

        # 5. Get active players for position data
        active_players = [p for p in nba_players.get_players() if p["is_active"]]

        return {
            "teams": all_teams,
            "games_df": games_df.to_dict("records") if not games_df.empty else [],
            "box_scores": box_scores,
            "team_advanced": team_advanced,
            "players": active_players,
            "season": season,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _fetch_box_score(self, game_id: str) -> tuple:
        """Fetch traditional and advanced box score for one game."""
        from nba_api.stats.endpoints import (
            boxscoreadvancedv3,
            boxscoretraditionalv3,
        )

        _delay()
        trad = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game_id,
            headers=self._headers,
            timeout=60,
        )
        trad_data = trad.get_dict()

        _delay()
        adv = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=game_id,
            headers=self._headers,
            timeout=60,
        )
        adv_data = adv.get_dict()

        return (
            {"game_id": game_id, "data": trad_data},
            {"game_id": game_id, "data": adv_data},
        )

    def normalize_payload(self, raw: Any) -> list[dict]:
        """Normalize nba_api data into canonical records."""
        records = []

        # Teams
        for t in raw.get("teams", []):
            records.append({
                "record_type": "team",
                "team_id": str(t["id"]),
                "full_name": t["full_name"],
                "abbreviation": t["abbreviation"],
                "conference": "",
                "division": "",
                "source": self.SOURCE_NAME,
            })

        # Players (position info from static data)
        player_positions = {}
        for p in raw.get("players", []):
            player_positions[str(p["id"])] = p
            records.append({
                "record_type": "player",
                "player_id": str(p["id"]),
                "full_name": p["full_name"],
                "first_name": p.get("first_name", ""),
                "last_name": p.get("last_name", ""),
                "is_active": p.get("is_active", True),
                "source": self.SOURCE_NAME,
            })

        # Games from LeagueGameFinder results
        games_seen = set()
        for row in raw.get("games_df", []):
            gid = row["GAME_ID"]
            if gid in games_seen:
                continue
            games_seen.add(gid)

            matchup = row.get("MATCHUP", "")
            team_id = str(row.get("TEAM_ID", ""))
            # Parse matchup to get home/away
            # Format: "LAL vs. BOS" (home) or "LAL @ BOS" (away)
            is_home = " vs. " in matchup

            records.append({
                "record_type": "game",
                "game_id": gid,
                "game_date": row.get("GAME_DATE", ""),
                "season": raw.get("season", "2024-25"),
                "status": "final",
                "home_team_id": team_id if is_home else "",
                "away_team_id": "" if is_home else team_id,
                "source": self.SOURCE_NAME,
            })

        # Fix games: need both home/away from the game finder data
        self._fix_game_teams(records, raw.get("games_df", []))

        # Box scores → player_game records
        for bs in raw.get("box_scores", []):
            gid = bs["game_id"]
            data = bs.get("data", {})
            result_sets = data.get("boxScoreTraditional", {})

            # Player stats
            for player_row in result_sets.get("playerStats", []):
                pid = str(player_row.get("personId", ""))
                tid = str(player_row.get("teamId", ""))
                minutes_str = player_row.get("minutes", "PT0M0.0S")
                minutes = self._parse_minutes(minutes_str)

                # Update player position/team from box score
                pos = player_row.get("position", "")
                if pid and pos:
                    # Update or create player record with position
                    self._update_player_position(records, pid, tid, pos)

                records.append({
                    "record_type": "player_game",
                    "player_id": pid,
                    "game_id": gid,
                    "team_id": tid,
                    "minutes": minutes,
                    "started": 1 if player_row.get("starter") == "1" else 0,
                    "assists": player_row.get("assists", 0),
                    "points": player_row.get("points", 0),
                    "rebounds": player_row.get("reboundsTotal", 0),
                    "turnovers": player_row.get("turnovers", 0),
                    "steals": player_row.get("steals", 0),
                    "blocks": player_row.get("blocks", 0),
                    "fouls": player_row.get("foulsPersonal", 0),
                    "field_goals_made": player_row.get("fieldGoalsMade", 0),
                    "field_goals_attempted": player_row.get("fieldGoalsAttempted", 0),
                    "free_throws_made": player_row.get("freeThrowsMade", 0),
                    "free_throws_attempted": player_row.get("freeThrowsAttempted", 0),
                    "potential_assists": None,  # not in traditional box score
                    "touches": None,
                    "passes_made": None,
                    "usage_rate": None,
                    "time_of_possession": None,
                })

            # Team stats from traditional box → team_game
            for team_row in result_sets.get("teamStats", []):
                tid = str(team_row.get("teamId", ""))
                records.append({
                    "record_type": "team_game",
                    "team_id": tid,
                    "game_id": gid,
                    "points": team_row.get("points", 0),
                    "assists": team_row.get("assists", 0),
                    "rebounds": team_row.get("reboundsTotal", 0),
                    "turnovers": team_row.get("turnovers", 0),
                })

        # Advanced box scores → add pace/drtg to team_game records
        for adv in raw.get("team_advanced", []):
            gid = adv["game_id"]
            data = adv.get("data", {})
            result_sets = data.get("boxScoreAdvanced", {})

            for team_row in result_sets.get("teamStats", []):
                tid = str(team_row.get("teamId", ""))
                # Find matching team_game record and add pace/drtg
                for rec in records:
                    if (
                        rec.get("record_type") == "team_game"
                        and rec.get("team_id") == tid
                        and rec.get("game_id") == gid
                    ):
                        rec["pace"] = team_row.get("pace", 0)
                        rec["offensive_rating"] = team_row.get("offensiveRating", 0)
                        rec["defensive_rating"] = team_row.get("defensiveRating", 0)
                        rec["possessions"] = team_row.get("possessions", 0)
                        break

            # Also grab usage_rate from player advanced stats
            for player_row in result_sets.get("playerStats", []):
                pid = str(player_row.get("personId", ""))
                for rec in records:
                    if (
                        rec.get("record_type") == "player_game"
                        and rec.get("player_id") == pid
                        and rec.get("game_id") == gid
                    ):
                        rec["usage_rate"] = player_row.get("usagePercentage", 0)
                        break

        return records

    def _fix_game_teams(self, records: list[dict], games_data: list[dict]):
        """Fix game records to have both home and away team IDs."""
        # Group by game_id to find both teams
        from collections import defaultdict

        game_teams = defaultdict(dict)
        for row in games_data:
            gid = row["GAME_ID"]
            tid = str(row.get("TEAM_ID", ""))
            matchup = row.get("MATCHUP", "")
            if " vs. " in matchup:
                game_teams[gid]["home"] = tid
                game_teams[gid]["home_score"] = row.get("PTS")
            elif " @ " in matchup:
                game_teams[gid]["away"] = tid
                game_teams[gid]["away_score"] = row.get("PTS")
            game_teams[gid]["date"] = row.get("GAME_DATE", "")

        # Update game records
        for rec in records:
            if rec.get("record_type") == "game":
                gid = rec["game_id"]
                if gid in game_teams:
                    info = game_teams[gid]
                    rec["home_team_id"] = info.get("home", rec.get("home_team_id", ""))
                    rec["away_team_id"] = info.get("away", rec.get("away_team_id", ""))
                    rec["home_score"] = info.get("home_score")
                    rec["away_score"] = info.get("away_score")
                    rec["game_date"] = info.get("date", rec.get("game_date", ""))

    def _update_player_position(
        self, records: list[dict], player_id: str, team_id: str, position: str
    ):
        """Update or create player record with position and team."""
        for rec in records:
            if rec.get("record_type") == "player" and rec.get("player_id") == player_id:
                rec["position"] = position
                rec["team_id"] = team_id
                return

    @staticmethod
    def _parse_minutes(minutes_str: str) -> float:
        """Parse NBA API minutes format (PT32M15.0S or '32:15') to float."""
        if not minutes_str:
            return 0.0

        # Handle PT format (e.g., "PT32M15.0S")
        if isinstance(minutes_str, str) and minutes_str.startswith("PT"):
            try:
                s = minutes_str[2:]  # strip "PT"
                mins = 0.0
                secs = 0.0
                if "M" in s:
                    parts = s.split("M")
                    mins = float(parts[0])
                    s = parts[1]
                if "S" in s:
                    secs = float(s.replace("S", ""))
                return round(mins + secs / 60, 1)
            except (ValueError, IndexError):
                return 0.0

        # Handle "32:15" format
        if isinstance(minutes_str, str) and ":" in minutes_str:
            try:
                parts = minutes_str.split(":")
                return round(float(parts[0]) + float(parts[1]) / 60, 1)
            except (ValueError, IndexError):
                return 0.0

        # Direct numeric
        try:
            return float(minutes_str)
        except (ValueError, TypeError):
            return 0.0

    def _required_fields(self) -> list[str]:
        return ["record_type"]
