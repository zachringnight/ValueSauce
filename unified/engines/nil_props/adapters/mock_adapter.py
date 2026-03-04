"""Mock adapter that loads from sample_payloads/ for local dev."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from engines.nil_props.config.settings import Settings
from engines.nil_props.adapters.base import AdapterResult, BaseAdapter


class MockNBAAdapter(BaseAdapter):
    """Loads NBA game/player/team data from sample payloads."""

    SOURCE_NAME = "mock_nba"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.payload_dir = Path(settings.sample_payloads_dir)

    def authenticate(self) -> bool:
        return self.payload_dir.exists()

    def _load_json(self, filename: str) -> Any:
        path = self.payload_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Sample payload not found: {path}")
        with open(path) as f:
            return json.load(f)

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        """Load all sample data for a season."""
        games = self._load_json("nba_games.json")
        players = self._load_json("nba_players.json")
        teams = self._load_json("nba_teams.json")
        player_stats = self._load_json("player_game_stats.json")
        team_stats = self._load_json("team_game_stats.json")

        raw = {
            "games": games,
            "players": players,
            "teams": teams,
            "player_game_stats": player_stats,
            "team_game_stats": team_stats,
        }
        records = self.normalize_payload(raw)
        return AdapterResult(
            source=self.SOURCE_NAME,
            raw_payload=raw,
            records=records,
        )

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        """In mock mode, just returns all data."""
        return self.fetch_historical()

    def normalize_payload(self, raw: Any) -> list[dict]:
        """Normalize mock payload into canonical dicts."""
        records = []

        # Teams
        for t in raw.get("teams", []):
            records.append({
                "record_type": "team",
                "team_id": t["team_id"],
                "full_name": t["full_name"],
                "abbreviation": t["abbreviation"],
                "conference": t.get("conference"),
                "division": t.get("division"),
            })

        # Players
        for p in raw.get("players", []):
            records.append({
                "record_type": "player",
                "player_id": p["player_id"],
                "full_name": p["full_name"],
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "team_id": p.get("team_id"),
                "position": p.get("position"),
                "height_inches": p.get("height_inches"),
                "weight_lbs": p.get("weight_lbs"),
                "is_active": p.get("is_active", True),
            })

        # Games
        for g in raw.get("games", []):
            records.append({
                "record_type": "game",
                "game_id": g["game_id"],
                "season": g.get("season", "2024-25"),
                "game_date": g["game_date"],
                "home_team_id": g["home_team"],
                "away_team_id": g["away_team"],
                "scheduled_tip": g.get("scheduled_tip"),
                "status": g.get("status", "final"),
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
            })

        # Player game stats
        for s in raw.get("player_game_stats", []):
            records.append({
                "record_type": "player_game",
                "player_id": s["player_id"],
                "game_id": s["game_id"],
                "team_id": s["team_id"],
                "minutes": s.get("minutes"),
                "started": s.get("started"),
                "assists": s.get("assists"),
                "potential_assists": s.get("potential_assists"),
                "points": s.get("points"),
                "rebounds": s.get("rebounds"),
                "turnovers": s.get("turnovers"),
                "steals": s.get("steals"),
                "blocks": s.get("blocks"),
                "fouls": s.get("fouls"),
                "field_goals_made": s.get("fg_made"),
                "field_goals_attempted": s.get("fg_attempted"),
                "free_throws_made": s.get("ft_made"),
                "free_throws_attempted": s.get("ft_attempted"),
                "usage_rate": s.get("usage_rate"),
                "touches": s.get("touches"),
                "passes_made": s.get("passes_made"),
                "time_of_possession": s.get("time_of_possession"),
            })

        # Team game stats
        for s in raw.get("team_game_stats", []):
            records.append({
                "record_type": "team_game",
                "team_id": s["team_id"],
                "game_id": s["game_id"],
                "is_home": s.get("is_home"),
                "points": s.get("points"),
                "assists": s.get("assists"),
                "rebounds": s.get("rebounds"),
                "turnovers": s.get("turnovers"),
                "pace": s.get("pace"),
                "offensive_rating": s.get("offensive_rating"),
                "defensive_rating": s.get("defensive_rating"),
                "possessions": s.get("possessions"),
            })

        return records

    def _required_fields(self) -> list[str]:
        return ["record_type"]


class MockInjuryAdapter(BaseAdapter):
    """Loads injury reports from sample payloads."""

    SOURCE_NAME = "mock_injury"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.payload_dir = Path(settings.sample_payloads_dir)

    def authenticate(self) -> bool:
        return self.payload_dir.exists()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        path = self.payload_dir / "injury_reports.json"
        if not path.exists():
            return AdapterResult(self.SOURCE_NAME, [], [])
        with open(path) as f:
            raw = json.load(f)
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        return self.fetch_historical()

    def normalize_payload(self, raw: Any) -> list[dict]:
        records = []
        for r in raw:
            records.append({
                "record_type": "injury",
                "player_id": r["player_id"],
                "team_id": r["team_id"],
                "game_id": r.get("game_id"),
                "report_timestamp": r["report_timestamp"],
                "status_timestamp": r.get("status_timestamp"),
                "status": r["status"],
                "reason": r.get("reason"),
                "source": r.get("source", "mock"),
            })
        return records

    def _required_fields(self) -> list[str]:
        return ["player_id", "status", "report_timestamp"]


class MockOddsAdapter(BaseAdapter):
    """Loads odds data from sample payloads."""

    SOURCE_NAME = "mock_odds"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.payload_dir = Path(settings.sample_payloads_dir)

    def authenticate(self) -> bool:
        return self.payload_dir.exists()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        path = self.payload_dir / "odds_player_assists.json"
        if not path.exists():
            return AdapterResult(self.SOURCE_NAME, [], [])
        with open(path) as f:
            raw = json.load(f)
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        return self.fetch_historical()

    def normalize_payload(self, raw: Any) -> list[dict]:
        records = []
        for r in raw:
            records.append({
                "record_type": "odds_snapshot" if not r.get("is_closing") else "odds_closing",
                "snapshot_timestamp": r["snapshot_timestamp"],
                "sportsbook_id": r["sportsbook_id"],
                "player_id": r["player_id"],
                "game_id": r["game_id"],
                "market_id": r.get("market_id", "player_assists_ou"),
                "line": r["line"],
                "over_price": r["over_price"],
                "under_price": r["under_price"],
            })
        return records

    def _required_fields(self) -> list[str]:
        return ["player_id", "game_id", "line", "over_price", "under_price"]
