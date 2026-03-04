"""Sportradar API client for schedules, boxscores, and ID mapping.

Sportradar uses its own internal IDs (``sr_game_id``, ``sr_player_id``)
that need to be mapped to canonical NBA IDs.  This client handles the
mapping transparently.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = "https://api.sportradar.us/nba/trial/v8/en"
_DEFAULT_TIMEOUT = 20
_DEFAULT_RATE_LIMIT = 1.0  # Sportradar trial tier: 1 req/sec
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SportradarRequestError(Exception):
    """Raised when a Sportradar API request fails after retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SportradarClient:
    """HTTP client for the Sportradar NBA API.

    Parameters
    ----------
    api_key:
        Sportradar API key.
    base_url:
        Root URL for the Sportradar NBA API.
    timeout:
        Per-request timeout in seconds.
    rate_limit:
        Minimum seconds between consecutive requests.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_schedule(self, date: str) -> list[dict[str, Any]]:
        """Fetch the NBA schedule for a given date.

        Parameters
        ----------
        date:
            Date string in ``YYYY-MM-DD`` format (or ``YYYY/MM/DD``).

        Returns
        -------
        list[dict]
            One dict per game scheduled for that date.
        """
        # Sportradar expects YYYY/MM/DD in the path
        date_path = date.replace("-", "/")
        path = f"/games/{date_path}/schedule.json"
        data = self._request(path)

        games: list[dict[str, Any]] = data.get("games", [])
        logger.info(
            "Fetched %d games for schedule date %s.", len(games), date
        )
        return games

    def get_game_boxscore(self, sr_game_id: str) -> dict[str, Any]:
        """Fetch the full boxscore for a Sportradar game.

        Parameters
        ----------
        sr_game_id:
            Sportradar game UUID.

        Returns
        -------
        dict
            Full boxscore payload including team and player stats.
        """
        path = f"/games/{sr_game_id}/boxscore.json"
        return self._request(path)

    def get_daily_change_log(self, date: str) -> list[dict[str, Any]]:
        """Fetch the change log for a given date.

        The change log lists resources that were updated (scores finalised,
        stats corrected, etc.) and is useful for incremental syncs.

        Parameters
        ----------
        date:
            Date string in ``YYYY-MM-DD`` format.

        Returns
        -------
        list[dict]
            Change log entries for the date.
        """
        date_path = date.replace("-", "/")
        path = f"/league/{date_path}/changes.json"
        data = self._request(path)
        return data.get("changes", data.get("changelog", []))

    def get_player_profile(self, sr_player_id: str) -> dict[str, Any]:
        """Fetch a player's profile from Sportradar.

        Parameters
        ----------
        sr_player_id:
            Sportradar player UUID.

        Returns
        -------
        dict
            Player profile including biographical info and reference IDs.
        """
        path = f"/players/{sr_player_id}/profile.json"
        return self._request(path)

    def map_to_nba_ids(self, sr_game_id: str) -> dict[str, Any]:
        """Build a canonical ID mapping for a Sportradar game.

        Fetches the game summary and extracts Sportradar-to-NBA ID
        correspondences for the game itself and all participating players.

        Parameters
        ----------
        sr_game_id:
            Sportradar game UUID.

        Returns
        -------
        dict
            Mapping with keys ``game``, ``home_team``, ``away_team``,
            and ``players``.  Each value is a dict containing both the
            Sportradar UUID and any available NBA reference ID.
        """
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
                player_map = {
                    "sr_player_id": player.get("id"),
                    "full_name": player.get("full_name"),
                    "nba_player_id": self._find_reference(
                        player.get("reference", player.get("references")),
                        source="nba",
                    ),
                    "team_side": side,
                }
                mapping["players"].append(player_map)

        return mapping

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a GET request with rate limiting and retry logic.

        Parameters
        ----------
        path:
            URL path appended to ``base_url``.
        params:
            Optional query parameters (the API key is injected automatically).

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        SportradarRequestError
            After exhausting all retry attempts.
        """
        url = f"{self.base_url}{path}"
        query: dict[str, Any] = {"api_key": self.api_key}
        if params:
            query.update(params)

        for attempt in range(1, _MAX_RETRIES + 1):
            # Rate limit
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                logger.debug(
                    "Sportradar request attempt=%d path=%s", attempt, path
                )
                resp = self._session.get(
                    url,
                    params=query,
                    timeout=self.timeout,
                )
                self._last_request_ts = time.monotonic()

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                logger.warning(
                    "Sportradar HTTP %s on attempt %d/%d for %s",
                    status,
                    attempt,
                    _MAX_RETRIES,
                    path,
                )
                if attempt == _MAX_RETRIES:
                    raise SportradarRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: "
                        f"HTTP {status} from {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Sportradar request error on attempt %d/%d for %s: %s",
                    attempt,
                    _MAX_RETRIES,
                    path,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    raise SportradarRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise SportradarRequestError(  # pragma: no cover
            f"Unexpected retry exhaustion for {path}"
        )

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_team_ids(team_block: dict[str, Any]) -> dict[str, Any]:
        """Pull Sportradar and NBA IDs from a team block."""
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
    def _find_reference(
        references: Any,
        source: str = "nba",
    ) -> str | None:
        """Search a Sportradar ``references`` list for a given source ID.

        References can appear as a list of dicts with ``origin`` / ``id``
        keys, or sometimes as a simple string.
        """
        if references is None:
            return None

        if isinstance(references, str):
            return references

        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict):
                    if ref.get("origin", "").lower() == source:
                        return ref.get("id")
            return None

        if isinstance(references, dict):
            return references.get(source) or references.get("id")

        return None
