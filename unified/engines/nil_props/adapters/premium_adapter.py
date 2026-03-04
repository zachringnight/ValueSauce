"""Premium API adapters (Tier A) — require API keys."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from engines.nil_props.config.settings import Settings
from engines.nil_props.adapters.base import AdapterResult, BaseAdapter

logger = logging.getLogger(__name__)


class PremiumNBAAdapter(BaseAdapter):
    """Premium NBA core feed adapter (e.g., Sportradar, Stats Perform)."""

    SOURCE_NAME = "premium_nba"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.api_key = settings.premium_nba_api_key
        self.base_url = settings.premium_nba_base_url
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def authenticate(self) -> bool:
        if not self.api_key or not self.base_url:
            self.logger.warning("Premium NBA adapter: no credentials configured")
            return False
        try:
            resp = self._session.get(f"{self.base_url}/status", timeout=10)
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Premium NBA auth failed: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        if not self.authenticate():
            raise ConnectionError("Premium NBA adapter not authenticated")
        raw = self._api_get(f"seasons/{season}/games", params={"include": "stats,players"})
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        if not self.authenticate():
            raise ConnectionError("Premium NBA adapter not authenticated")
        raw = self._api_get("games/recent", params={"since": since.isoformat()})
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def normalize_payload(self, raw: Any) -> list[dict]:
        """Normalize premium API response. Structure depends on actual vendor."""
        records = []
        # This normalization is vendor-specific. The structure below is a template
        # that would be adapted to the actual API response format.
        for game in raw.get("games", []):
            records.append({
                "record_type": "game",
                "game_id": game.get("id"),
                "season": game.get("season"),
                "game_date": game.get("date"),
                "home_team_id": game.get("home", {}).get("id"),
                "away_team_id": game.get("away", {}).get("id"),
                "status": game.get("status"),
                "home_score": game.get("home", {}).get("score"),
                "away_score": game.get("away", {}).get("score"),
            })
            for side in ["home", "away"]:
                team_data = game.get(side, {})
                for player in team_data.get("players", []):
                    records.append({
                        "record_type": "player_game",
                        "player_id": player.get("id"),
                        "game_id": game.get("id"),
                        "team_id": team_data.get("id"),
                        "minutes": player.get("minutes"),
                        "started": player.get("started"),
                        "assists": player.get("assists"),
                        "potential_assists": player.get("potential_assists"),
                        "points": player.get("points"),
                        "rebounds": player.get("rebounds"),
                        "turnovers": player.get("turnovers"),
                    })
        return records

    def _required_fields(self) -> list[str]:
        return ["record_type", "game_id"]


class PremiumOddsAdapter(BaseAdapter):
    """Premium historical odds adapter."""

    SOURCE_NAME = "premium_odds"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.api_key = settings.premium_odds_api_key
        self.base_url = settings.premium_odds_base_url
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["api-key"] = self.api_key

    def authenticate(self) -> bool:
        if not self.api_key or not self.base_url:
            self.logger.warning("Premium odds adapter: no credentials configured")
            return False
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        if not self.authenticate():
            raise ConnectionError("Premium odds adapter not authenticated")
        raw = self._api_get(
            "odds/historical",
            params={"sport": "basketball_nba", "market": "player_assists", "season": season},
        )
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        if not self.authenticate():
            raise ConnectionError("Premium odds adapter not authenticated")
        raw = self._api_get(
            "odds/recent",
            params={"sport": "basketball_nba", "market": "player_assists",
                     "since": since.isoformat()},
        )
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def normalize_payload(self, raw: Any) -> list[dict]:
        records = []
        for event in raw.get("events", []):
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "player_assists":
                        continue
                    for outcome in market.get("outcomes", []):
                        records.append({
                            "record_type": "odds_snapshot",
                            "snapshot_timestamp": event.get("timestamp"),
                            "sportsbook_id": book.get("key"),
                            "player_id": outcome.get("player_id"),
                            "game_id": event.get("game_id"),
                            "market_id": "player_assists_ou",
                            "line": outcome.get("point"),
                            "over_price": outcome.get("over_price"),
                            "under_price": outcome.get("under_price"),
                        })
        return records

    def _required_fields(self) -> list[str]:
        return ["player_id", "game_id", "line", "over_price", "under_price"]
