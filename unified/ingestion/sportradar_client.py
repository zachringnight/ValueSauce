"""Unified Sportradar API client.

Directly from: ValueHunter/src/nba_props/ingestion/sportradar.py
Schedules, boxscores, change logs, player profiles, and ID mapping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.sportradar.us/nba/trial/v8/en"
_DEFAULT_TIMEOUT = 20
_DEFAULT_RATE_LIMIT = 1.0
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


class SportradarRequestError(Exception):
    """Raised when a Sportradar API request fails after retries."""


class SportradarClient:
    """HTTP client for the Sportradar NBA API."""

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

    def get_daily_schedule(self, date: str) -> list[dict[str, Any]]:
        """Fetch NBA schedule for YYYY-MM-DD."""
        date_path = date.replace("-", "/")
        path = f"/games/{date_path}/schedule.json"
        data = self._request(path)
        games = data.get("games", [])
        logger.info("Fetched %d games for %s.", len(games), date)
        return games

    def get_game_boxscore(self, sr_game_id: str) -> dict[str, Any]:
        """Fetch full boxscore for a Sportradar game UUID."""
        path = f"/games/{sr_game_id}/boxscore.json"
        return self._request(path)

    def get_daily_change_log(self, date: str) -> list[dict[str, Any]]:
        """Fetch change log for incremental syncs."""
        date_path = date.replace("-", "/")
        path = f"/league/{date_path}/changes.json"
        data = self._request(path)
        return data.get("changes", data.get("changelog", []))

    def get_player_profile(self, sr_player_id: str) -> dict[str, Any]:
        """Fetch player profile by Sportradar UUID."""
        path = f"/players/{sr_player_id}/profile.json"
        return self._request(path)

    def map_to_nba_ids(self, sr_game_id: str) -> dict[str, Any]:
        """Build Sportradar → NBA ID mappings for a game."""
        path = f"/games/{sr_game_id}/summary.json"
        data = self._request(path)

        mapping: dict[str, Any] = {
            "sr_game_id": sr_game_id,
            "game": {
                "sr_id": sr_game_id,
                "nba_id": data.get("id") or data.get("reference"),
            },
            "home_team": self._extract_team_ids(data.get("home", {})),
            "away_team": self._extract_team_ids(data.get("away", {})),
            "players": [],
        }

        for side in ("home", "away"):
            team_block = data.get(side, {})
            for player in team_block.get("players", []):
                mapping["players"].append({
                    "sr_player_id": player.get("id"),
                    "full_name": player.get("full_name"),
                    "nba_player_id": self._find_reference(
                        player.get("reference", player.get("references")),
                        source="nba",
                    ),
                    "team_side": side,
                })

        return mapping

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute GET with rate limiting and retry."""
        url = f"{self.base_url}{path}"
        query: dict[str, Any] = {"api_key": self.api_key}
        if params:
            query.update(params)

        for attempt in range(1, _MAX_RETRIES + 1):
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                resp = self._session.get(url, params=query, timeout=self.timeout)
                self._last_request_ts = time.monotonic()
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if attempt == _MAX_RETRIES:
                    raise SportradarRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: HTTP {status} from {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise SportradarRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise SportradarRequestError(f"Unexpected retry exhaustion for {path}")  # pragma: no cover

    @staticmethod
    def _extract_team_ids(team_block: dict[str, Any]) -> dict[str, Any]:
        return {
            "sr_team_id": team_block.get("id"),
            "name": team_block.get("name"),
            "alias": team_block.get("alias"),
            "nba_team_id": SportradarClient._find_reference(
                team_block.get("reference", team_block.get("references")),
                source="nba",
            ),
        }

    @staticmethod
    def _find_reference(references: Any, source: str = "nba") -> str | None:
        if references is None:
            return None
        if isinstance(references, str):
            return references
        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict) and ref.get("origin", "").lower() == source:
                    return ref.get("id")
            return None
        if isinstance(references, dict):
            return references.get(source) or references.get("id")
        return None
