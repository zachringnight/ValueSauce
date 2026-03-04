"""NBA Stats API client for fetching player/team game logs and tracking data.

stats.nba.com requires specific browser-like headers and is aggressively
rate-limited.  This client handles both concerns transparently.
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
_DEFAULT_BASE_URL = "https://stats.nba.com/stats"
_DEFAULT_RATE_LIMIT = 0.6  # seconds between requests
_DEFAULT_TIMEOUT = 30  # seconds

_BROWSER_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class NBAStatsRequestError(Exception):
    """Raised when a request to the NBA Stats API fails after retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class NBAStatsClient:
    """HTTP client for the NBA Stats API (stats.nba.com).

    Parameters
    ----------
    base_url:
        Root URL for the stats API.  Defaults to ``https://stats.nba.com/stats``.
    rate_limit:
        Minimum seconds between consecutive requests.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(_BROWSER_HEADERS)
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_player_game_logs(
        self,
        season: str,
        season_type: str = "Regular Season",
        player_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch player-level game logs from the ``leaguegamelog`` endpoint.

        Parameters
        ----------
        season:
            NBA season string, e.g. ``"2025-26"``.
        season_type:
            One of ``"Regular Season"``, ``"Playoffs"``, ``"Pre Season"``.
        player_id:
            Optional NBA player ID to filter results.
        date_from:
            Optional start date ``MM/DD/YYYY``.
        date_to:
            Optional end date ``MM/DD/YYYY``.

        Returns
        -------
        list[dict]
            One dict per game with column names as keys.
        """
        params: dict[str, Any] = {
            "Counter": 0,
            "DateFrom": date_from or "",
            "DateTo": date_to or "",
            "Direction": "DESC",
            "LeagueID": "00",
            "PlayerOrTeam": "P",
            "Season": season,
            "SeasonType": season_type,
            "Sorter": "DATE",
        }
        if player_id is not None:
            params["PlayerID"] = player_id

        data = self._request("leaguegamelog", params)
        return self._result_set_to_dicts(data)

    def get_team_game_logs(
        self,
        season: str,
        team_id: int | None = None,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch team-level game logs from the ``leaguegamelog`` endpoint."""
        params: dict[str, Any] = {
            "Counter": 0,
            "DateFrom": "",
            "DateTo": "",
            "Direction": "DESC",
            "LeagueID": "00",
            "PlayerOrTeam": "T",
            "Season": season,
            "SeasonType": season_type,
            "Sorter": "DATE",
        }
        if team_id is not None:
            params["TeamID"] = team_id

        data = self._request("leaguegamelog", params)
        return self._result_set_to_dicts(data)

    def get_tracking_catch_and_shoot(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch catch-and-shoot tracking data from ``playerdashptshots``."""
        params: dict[str, Any] = {
            "DateFrom": "",
            "DateTo": "",
            "GameSegment": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PerMode": "Totals",
            "Period": 0,
            "PlayerID": 0,
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "TeamID": 0,
            "VsConference": "",
            "VsDivision": "",
            "GeneralRange": "Catch and Shoot",
        }
        data = self._request("playerdashptshots", params)
        return self._result_set_to_dicts(data)

    def get_tracking_pull_up(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch pull-up shooting tracking data from ``playerdashptshots``."""
        params: dict[str, Any] = {
            "DateFrom": "",
            "DateTo": "",
            "GameSegment": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PerMode": "Totals",
            "Period": 0,
            "PlayerID": 0,
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "TeamID": 0,
            "VsConference": "",
            "VsDivision": "",
            "GeneralRange": "Pullups",
        }
        data = self._request("playerdashptshots", params)
        return self._result_set_to_dicts(data)

    def get_tracking_touches(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch player touch-tracking data from ``playerdashpttotalreboundchancemisc``.

        Uses the ``playerdashtouchesummary`` family of endpoints which expose
        touches, time of possession, and related metrics.
        """
        params: dict[str, Any] = {
            "DateFrom": "",
            "DateTo": "",
            "GameSegment": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PerMode": "Totals",
            "Period": 0,
            "PlayerID": 0,
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "TeamID": 0,
            "VsConference": "",
            "VsDivision": "",
        }
        data = self._request("playerdashptpass", params)
        return self._result_set_to_dicts(data)

    def get_tracking_passing(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch player passing-tracking data from ``playerdashptpass``."""
        params: dict[str, Any] = {
            "DateFrom": "",
            "DateTo": "",
            "GameSegment": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PerMode": "Totals",
            "Period": 0,
            "PlayerID": 0,
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "TeamID": 0,
            "VsConference": "",
            "VsDivision": "",
        }
        data = self._request("playerdashptpass", params)
        return self._result_set_to_dicts(data)

    def get_opponent_shooting(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch team opponent shooting dashboard from ``leaguedashteamstats``.

        Returns opponent field-goal data broken down by shot distance/area,
        useful for measuring defensive three-point allowance rates.
        """
        params: dict[str, Any] = {
            "Conference": "",
            "DateFrom": "",
            "DateTo": "",
            "Division": "",
            "GameScope": "",
            "GameSegment": "",
            "Height": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "MeasureType": "Opponent",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PORound": 0,
            "PaceAdjust": "N",
            "PerMode": "PerGame",
            "Period": 0,
            "PlayerExperience": "",
            "PlayerPosition": "",
            "PlusMinus": "N",
            "Rank": "N",
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "ShotClockRange": "",
            "StarterBench": "",
            "TeamID": 0,
            "TwoWay": 0,
            "VsConference": "",
            "VsDivision": "",
            "Weight": "",
        }
        data = self._request("leaguedashteamstats", params)
        return self._result_set_to_dicts(data)

    def get_opponent_shooting_closest_defender(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Fetch opponent shooting stats by closest-defender distance.

        Uses the ``leaguedashplayershotlocations`` or similar endpoint
        that returns opponent FG% by defender proximity buckets.
        """
        params: dict[str, Any] = {
            "CloseDefDistRange": "",
            "College": "",
            "Conference": "",
            "Country": "",
            "DateFrom": "",
            "DateTo": "",
            "Division": "",
            "DraftPick": "",
            "DraftYear": "",
            "DribbleRange": "",
            "GameScope": "",
            "GameSegment": "",
            "GeneralRange": "",
            "Height": "",
            "LastNGames": 0,
            "LeagueID": "00",
            "Location": "",
            "Month": 0,
            "OpponentTeamID": 0,
            "Outcome": "",
            "PORound": 0,
            "PerMode": "Totals",
            "Period": 0,
            "PlayerExperience": "",
            "PlayerPosition": "",
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "ShotClockRange": "",
            "ShotDistRange": "",
            "StarterBench": "",
            "TeamID": 0,
            "TouchTimeRange": "",
            "VsConference": "",
            "VsDivision": "",
            "Weight": "",
        }
        data = self._request("leaguedashoppptshot", params)
        return self._result_set_to_dicts(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a GET request with rate limiting, retries, and back-off.

        Parameters
        ----------
        endpoint:
            API endpoint name (appended to ``base_url``).
        params:
            Query parameters for the request.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        NBAStatsRequestError
            After exhausting all retry attempts.
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(1, _MAX_RETRIES + 1):
            # Rate-limit: wait until enough time has elapsed since the last call
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                logger.debug(
                    "NBA Stats request attempt=%d endpoint=%s",
                    attempt,
                    endpoint,
                )
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                self._last_request_ts = time.monotonic()

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                logger.warning(
                    "NBA Stats HTTP %s on attempt %d/%d for %s",
                    status,
                    attempt,
                    _MAX_RETRIES,
                    endpoint,
                )
                if attempt == _MAX_RETRIES:
                    raise NBAStatsRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: "
                        f"HTTP {status} from {endpoint}"
                    ) from exc
                backoff = _BACKOFF_FACTOR ** attempt
                time.sleep(backoff)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "NBA Stats request error on attempt %d/%d for %s: %s",
                    attempt,
                    _MAX_RETRIES,
                    endpoint,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    raise NBAStatsRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {endpoint}"
                    ) from exc
                backoff = _BACKOFF_FACTOR ** attempt
                time.sleep(backoff)

        # Unreachable, but keeps mypy happy
        raise NBAStatsRequestError(f"Unexpected retry exhaustion for {endpoint}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _result_set_to_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert an NBA Stats ``resultSets`` response into a list of dicts.

        The API returns data in a columnar format::

            {
                "resultSets": [
                    {
                        "headers": ["COL_A", "COL_B", ...],
                        "rowSet": [[val_a, val_b, ...], ...]
                    }
                ]
            }

        This method flattens the first result set into row-oriented dicts.
        """
        try:
            result_set = data["resultSets"][0]
            headers = result_set["headers"]
            rows = result_set["rowSet"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected NBA Stats response structure: %s", exc)
            return []

        return [dict(zip(headers, row)) for row in rows]
