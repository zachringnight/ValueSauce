"""The Odds API adapter — fetches NBA player assist props from the-odds-api.com."""

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

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_nba"
_REQUEST_DELAY = 1.0  # seconds between requests to avoid 429


class TheOddsAPIAdapter(BaseAdapter):
    """Fetches NBA player assist over/under odds from The Odds API (V4)."""

    SOURCE_NAME = "theoddsapi"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.api_key = settings.the_odds_api_key
        self._session = requests.Session()

    def authenticate(self) -> bool:
        if not self.api_key:
            self.logger.warning("The Odds API: no API key configured")
            return False
        # Test with free events endpoint (no credit cost)
        try:
            resp = self._session.get(
                f"{_BASE_URL}/sports/{_SPORT}/events",
                params={"apiKey": self.api_key},
                timeout=15,
            )
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"The Odds API auth failed: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _api_get(self, url: str, params: dict | None = None) -> dict | list:
        params = params or {}
        params["apiKey"] = self.api_key
        time.sleep(_REQUEST_DELAY)
        resp = self._session.get(url, params=params, timeout=30)
        # Log remaining credits
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining:
            logger.info(f"Odds API credits: {remaining} remaining, {used} used")
        resp.raise_for_status()
        return resp.json()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        """Fetch current NBA player assist odds for all upcoming events."""
        if not self.authenticate():
            raise ConnectionError("The Odds API not authenticated")
        raw = self._fetch_all_events()
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        """Fetch current NBA player assist odds (same as historical for live API)."""
        if not self.authenticate():
            raise ConnectionError("The Odds API not authenticated")
        raw = self._fetch_all_events()
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def _fetch_all_events(self) -> dict:
        """Fetch events and their player assist odds."""
        # Step 1: Get all NBA events (FREE — no credit cost)
        events = self._api_get(f"{_BASE_URL}/sports/{_SPORT}/events")
        logger.info(f"Found {len(events)} NBA events")

        # Step 2: For each event, fetch player_assists odds (1 credit each)
        all_event_odds = []
        for event in events:
            event_id = event.get("id")
            try:
                event_odds = self._api_get(
                    f"{_BASE_URL}/sports/{_SPORT}/events/{event_id}/odds",
                    params={
                        "regions": "us",
                        "markets": "player_assists",
                        "oddsFormat": "american",
                        "bookmakers": "draftkings,fanduel",
                    },
                )
                all_event_odds.append(event_odds)
            except Exception as e:
                logger.warning(f"Failed to fetch odds for event {event_id}: {e}")

        return {"events": all_event_odds}

    def normalize_payload(self, raw: Any) -> list[dict]:
        """Normalize The Odds API response into canonical records.

        Response structure per event:
        {
          "id": "abc123",
          "commence_time": "2025-01-15T00:10:00Z",
          "home_team": "Los Angeles Lakers",
          "away_team": "Boston Celtics",
          "bookmakers": [{
            "key": "draftkings",
            "markets": [{
              "key": "player_assists",
              "outcomes": [
                {"name": "Over", "description": "LeBron James", "price": -115, "point": 8.5},
                {"name": "Under", "description": "LeBron James", "price": -105, "point": 8.5},
              ]
            }]
          }]
        }
        """
        records = []

        for event in raw.get("events", []):
            event_id = event.get("id", "")
            commence_time = event.get("commence_time", "")

            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker.get("key", "")

                for market in bookmaker.get("markets", []):
                    if market.get("key") != "player_assists":
                        continue

                    # Group outcomes by player (Over/Under pairs)
                    player_lines = {}
                    for outcome in market.get("outcomes", []):
                        player_name = outcome.get("description", "")
                        side = outcome.get("name", "").lower()
                        price = outcome.get("price")
                        point = outcome.get("point")

                        if player_name not in player_lines:
                            player_lines[player_name] = {
                                "line": point,
                                "over_price": None,
                                "under_price": None,
                            }

                        if side == "over":
                            player_lines[player_name]["over_price"] = price
                            player_lines[player_name]["line"] = point
                        elif side == "under":
                            player_lines[player_name]["under_price"] = price

                    # Create records
                    for player_name, pdata in player_lines.items():
                        if pdata["over_price"] is None or pdata["line"] is None:
                            continue

                        # Normalize player name to ID format
                        player_id = self._name_to_id(player_name)

                        records.append({
                            "record_type": "odds_snapshot",
                            "snapshot_timestamp": commence_time,
                            "sportsbook_id": book_key,
                            "player_id": player_id,
                            "game_id": event_id,
                            "market_id": "player_assists_ou",
                            "line": float(pdata["line"]),
                            "over_price": float(pdata["over_price"]),
                            "under_price": float(pdata["under_price"])
                            if pdata["under_price"] is not None
                            else float(pdata["over_price"]) * -1,
                        })

        logger.info(f"Normalized {len(records)} odds records")
        return records

    @staticmethod
    def _name_to_id(name: str) -> str:
        """Convert player name to a normalized ID for matching.

        Input: "LeBron James"
        Output: "lebron_james"
        """
        return name.lower().replace(" ", "_").replace(".", "").replace("'", "")

    def _required_fields(self) -> list[str]:
        return ["player_id", "game_id", "line", "over_price"]
