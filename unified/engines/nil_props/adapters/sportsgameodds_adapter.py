"""SportsGameOdds adapter — fetches NBA player assist props from sportsgameodds.com."""

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

_BASE_URL = "https://api.sportsgameodds.com/v2"
_REQUEST_DELAY = 0.3  # seconds between requests


class SportsgameoddsAdapter(BaseAdapter):
    """Fetches NBA player assist over/under odds from SportsGameOdds API."""

    SOURCE_NAME = "sportsgameodds"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.api_key = settings.sportsgameodds_api_key
        self._session = requests.Session()

    def authenticate(self) -> bool:
        if not self.api_key:
            self.logger.warning("SportsGameOdds adapter: no API key configured")
            return False
        try:
            resp = self._api_get("account/usage")
            return resp.get("success", False)
        except Exception as e:
            self.logger.error(f"SportsGameOdds auth failed: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{_BASE_URL}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        time.sleep(_REQUEST_DELAY)
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_historical(self, season: str = "2024-25", **kwargs) -> AdapterResult:
        """Fetch historical NBA assist odds for completed games."""
        if not self.authenticate():
            raise ConnectionError("SportsGameOdds adapter not authenticated")

        all_events = []
        cursor = None

        # Paginate through finalized NBA events with assist odds
        while True:
            params = {
                "leagueID": "NBA",
                "oddIDs": "assists-PLAYER_ID-game-ou-over",
                "includeOpposingOddIDs": "true",
                "finalized": "true",
                "limit": "50",
                "bookmakerID": "draftkings,fanduel",
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_get("events", params)
            events = data.get("data", [])
            all_events.extend(events)

            cursor = data.get("nextCursor")
            if not cursor or not events:
                break

            logger.info(f"Fetched {len(all_events)} events so far...")

        logger.info(f"Total historical events fetched: {len(all_events)}")
        raw = {"events": all_events, "source": "historical"}
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        """Fetch current/upcoming NBA assist odds (pre-game lines)."""
        if not self.authenticate():
            raise ConnectionError("SportsGameOdds adapter not authenticated")

        all_events = []
        cursor = None

        while True:
            params = {
                "leagueID": "NBA",
                "oddIDs": "assists-PLAYER_ID-game-ou-over",
                "includeOpposingOddIDs": "true",
                "oddsAvailable": "true",
                "limit": "50",
                "startsAfter": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakerID": "draftkings,fanduel",
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_get("events", params)
            events = data.get("data", [])
            all_events.extend(events)

            cursor = data.get("nextCursor")
            if not cursor or not events:
                break

        raw = {"events": all_events, "source": "incremental"}
        records = self.normalize_payload(raw)
        return AdapterResult(self.SOURCE_NAME, raw, records)

    def normalize_payload(self, raw: Any) -> list[dict]:
        """Normalize SportsGameOdds response into canonical records.

        The API returns odds keyed by oddID like:
            "assists-LEBRON_JAMES_NBA-game-ou-over": {
                "fairOdds": "-110",
                "fairOverUnder": "8.5",
                "byBookmaker": { "draftkings": {"odds": "-110", "overUnder": "8.5"}, ... }
            }
        """
        records = []

        for event in raw.get("events", []):
            event_id = event.get("eventID", "")
            status_info = event.get("status", {})
            starts_at = status_info.get("startsAt", "")
            is_finalized = status_info.get("ended", False)

            # Extract team info
            teams = event.get("teams", {})
            home_team = teams.get("home", {})
            away_team = teams.get("away", {})

            odds = event.get("odds", {})
            if not odds:
                continue

            # Group odds by player: find over/under pairs
            player_odds = {}  # player_id -> {line, over_price, under_price, ...}

            for odd_id, odd_data in odds.items():
                # Parse oddID: assists-PLAYER_ID-game-ou-over/under
                parts = odd_id.split("-")
                if len(parts) < 5:
                    continue
                stat_id = parts[0]
                if stat_id != "assists":
                    continue

                # Player ID is everything between stat and period
                # Format: assists-FIRST_LAST_NBA-game-ou-side
                side = parts[-1]  # "over" or "under"
                # Reconstruct player ID (may contain hyphens in name)
                player_key = "-".join(parts[1:-3])

                if player_key not in player_odds:
                    player_odds[player_key] = {
                        "player_id_raw": player_key,
                        "line": None,
                        "over_price": None,
                        "under_price": None,
                        "score": None,
                    }

                # Get line and price from byBookmaker (prefer DraftKings, then FanDuel)
                by_book = odd_data.get("byBookmaker", {})
                fair_ou = odd_data.get("fairOverUnder")
                score = odd_data.get("score")

                if score is not None:
                    player_odds[player_key]["score"] = score

                for book_id in ["draftkings", "fanduel"]:
                    book_data = by_book.get(book_id, {})
                    if not book_data.get("available"):
                        continue

                    line = book_data.get("overUnder") or fair_ou
                    price = book_data.get("odds")

                    if line is not None:
                        try:
                            player_odds[player_key]["line"] = float(line)
                        except (ValueError, TypeError):
                            pass

                    if price is not None and side in ("over", "under"):
                        try:
                            price_val = float(price)
                            if side == "over":
                                player_odds[player_key]["over_price"] = price_val
                                player_odds[player_key]["over_book"] = book_id
                            else:
                                player_odds[player_key]["under_price"] = price_val
                                player_odds[player_key]["under_book"] = book_id
                        except (ValueError, TypeError):
                            pass

            # Create records from grouped odds
            snapshot_time = starts_at or datetime.utcnow().isoformat()

            for player_key, pdata in player_odds.items():
                if pdata["line"] is None or pdata["over_price"] is None:
                    continue

                # Determine record type based on game status
                record_type = "odds_closing" if is_finalized else "odds_snapshot"

                # Use the book that provided the over price
                book_id = pdata.get("over_book", "draftkings")

                records.append({
                    "record_type": record_type,
                    "snapshot_timestamp": snapshot_time,
                    "sportsbook_id": book_id,
                    "player_id": self._normalize_player_id(player_key),
                    "game_id": event_id,
                    "market_id": "player_assists_ou",
                    "line": pdata["line"],
                    "over_price": pdata["over_price"],
                    "under_price": pdata.get("under_price", pdata["over_price"] * -1),
                })

        return records

    @staticmethod
    def _normalize_player_id(raw_id: str) -> str:
        """Convert SportsGameOdds player ID format to a normalized form.

        Input: "LEBRON_JAMES_NBA" or similar
        Output: "lebron_james" (lowercase, strip league suffix)
        """
        # Strip league suffix
        for suffix in ("_NBA", "_WNBA", "_NFL", "_MLB"):
            if raw_id.endswith(suffix):
                raw_id = raw_id[: -len(suffix)]
                break
        return raw_id.lower()

    def _required_fields(self) -> list[str]:
        return ["player_id", "game_id", "line", "over_price"]
