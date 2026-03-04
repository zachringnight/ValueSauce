"""Unified The Odds API client.

Based on: ValueHunter/src/nba_props/ingestion/odds_api.py
All markets supported (player_points, player_rebounds, player_assists, player_threes).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

HISTORY_START = "2023-05-03"
SNAPSHOT_INTERVAL_MIN = 5

_DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
_DEFAULT_TIMEOUT = 15
_DEFAULT_RATE_LIMIT = 0.25
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0

ALL_PLAYER_MARKETS = (
    "player_points", "player_rebounds", "player_assists", "player_threes",
)

MARKET_KEY_TO_MODEL = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
}


class OddsAPIRequestError(Exception):
    """Raised when a request to The Odds API fails after retries."""


class OddsAPIClient:
    """Unified HTTP client for The Odds API.

    Supports live, historical, and per-event prop snapshots.
    Quota monitoring via response headers.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit = rate_limit

        self._session = requests.Session()
        self._last_request_ts: float = 0.0

    def get_upcoming_events(
        self,
        sport: str = "basketball_nba",
    ) -> list[dict[str, Any]]:
        """Fetch upcoming NBA events (games)."""
        path = f"/sports/{sport}/events"
        data = self._request(path)
        if isinstance(data, list):
            return data
        return data.get("events", data.get("data", []))

    def get_event_odds(
        self,
        event_id: str,
        markets: tuple[str, ...] = ALL_PLAYER_MARKETS,
        bookmakers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch current odds for a specific event across all player prop markets."""
        params: dict[str, Any] = {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        path = f"/sports/basketball_nba/events/{event_id}/odds"
        return self._request(path, params=params)

    def get_historical_event_odds(
        self,
        event_id: str,
        date: str,
        markets: tuple[str, ...] = ALL_PLAYER_MARKETS,
    ) -> dict[str, Any]:
        """Fetch historical odds snapshot for an event."""
        params: dict[str, Any] = {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": date,
        }
        path = f"/historical/sports/basketball_nba/events/{event_id}/odds"
        return self._request(path, params=params)

    def get_player_prop_snapshots(
        self,
        event_id: str,
        player_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Flatten nested bookmaker/market structure into per-outcome records."""
        odds = self.get_event_odds(event_id)
        snapshots: list[dict[str, Any]] = []

        for bookmaker in odds.get("bookmakers", []):
            bk_key = bookmaker.get("key", "")
            bk_title = bookmaker.get("title", "")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")

                for outcome in market.get("outcomes", []):
                    description = outcome.get("description", "")
                    if player_name and player_name.lower() not in description.lower():
                        continue
                    snapshots.append({
                        "bookmaker": bk_key,
                        "bookmaker_title": bk_title,
                        "market": market_key,
                        "player": description,
                        "name": outcome.get("name"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                    })

        return snapshots

    def extract_novig_probs(
        self,
        over_price: int | float,
        under_price: int | float,
    ) -> tuple[float, float]:
        """Remove vig from a two-way market and return fair probabilities."""
        over_implied = self._american_to_implied(over_price)
        under_implied = self._american_to_implied(under_price)
        total = over_implied + under_implied
        if total == 0:
            return 0.5, 0.5
        return over_implied / total, under_implied / total

    def compute_hold_pct(
        self,
        over_price: int | float,
        under_price: int | float,
    ) -> float:
        """Compute the bookmaker hold (overround) percentage."""
        over_implied = self._american_to_implied(over_price)
        under_implied = self._american_to_implied(under_price)
        total = over_implied + under_implied
        if total == 0:
            return 0.0
        return total - 1.0

    @staticmethod
    def _american_to_implied(american_odds: int | float) -> float:
        """Convert American odds to implied probability."""
        odds = float(american_odds)
        if odds == 0:
            return 0.0
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return 100.0 / (odds + 100.0)

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute GET with rate limiting, retry, and quota logging."""
        url = f"{self.base_url}{path}"
        query: dict[str, Any] = {"apiKey": self.api_key}
        if params:
            query.update(params)

        for attempt in range(1, _MAX_RETRIES + 1):
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                resp = self._session.get(url, params=query, timeout=self.timeout)
                self._last_request_ts = time.monotonic()

                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    logger.debug("Odds API quota: %s remaining, %s used", remaining, used)

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (401, 403):
                    raise OddsAPIRequestError(
                        f"Authentication failed (HTTP {status}). Check your API key."
                    ) from exc
                if status == 429:
                    backoff = (_BACKOFF_FACTOR ** attempt) * 2
                    logger.info("Rate limited by Odds API, sleeping %.1fs", backoff)
                    time.sleep(backoff)
                    continue
                if attempt == _MAX_RETRIES:
                    raise OddsAPIRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: HTTP {status} from {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise OddsAPIRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise OddsAPIRequestError(f"Unexpected retry exhaustion for {path}")  # pragma: no cover
