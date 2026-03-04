"""Unified NBA Stats API client.

Merges ValueHunter's robust retry/rate-limit architecture with
NBA_Props_AI's extended endpoint catalog. All engines read from this
single client so stats.nba.com is only hit once per endpoint+params.

Based on: ValueHunter/src/nba_props/ingestion/nba_stats.py
Extended with: NBA_Props_AI/core_best_v3/nba_props/nba_data.py endpoint catalog
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
    """Unified HTTP client for the NBA Stats API (stats.nba.com).

    Combines ValueHunter's retry/rate-limit with NBA_Props_AI's endpoint catalog.
    Session reuse, dynamic rate limiting (0.6s), centralized retry (3 attempts).
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
    # Player/Team Game Logs (from ValueHunter)
    # ------------------------------------------------------------------

    def get_player_game_logs(
        self,
        season: str,
        season_type: str = "Regular Season",
        player_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch player-level game logs from ``leaguegamelog``."""
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
        """Fetch team-level game logs from ``leaguegamelog``."""
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

    # ------------------------------------------------------------------
    # Tracking Data (from ValueHunter)
    # ------------------------------------------------------------------

    def get_tracking_catch_and_shoot(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Catch-and-shoot tracking from ``playerdashptshots``."""
        params = self._tracking_base_params(season, season_type)
        params["GeneralRange"] = "Catch and Shoot"
        data = self._request("playerdashptshots", params)
        return self._result_set_to_dicts(data)

    def get_tracking_pull_up(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Pull-up shooting tracking from ``playerdashptshots``."""
        params = self._tracking_base_params(season, season_type)
        params["GeneralRange"] = "Pullups"
        data = self._request("playerdashptshots", params)
        return self._result_set_to_dicts(data)

    def get_tracking_touches(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Player touch-tracking from ``playerdashptpass``."""
        params = self._tracking_base_params(season, season_type)
        data = self._request("playerdashptpass", params)
        return self._result_set_to_dicts(data)

    def get_tracking_passing(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Player passing-tracking from ``playerdashptpass``."""
        params = self._tracking_base_params(season, season_type)
        data = self._request("playerdashptpass", params)
        return self._result_set_to_dicts(data)

    # ------------------------------------------------------------------
    # Opponent/Defensive Data (from ValueHunter)
    # ------------------------------------------------------------------

    def get_opponent_shooting(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Team opponent shooting dashboard from ``leaguedashteamstats``."""
        params: dict[str, Any] = {
            "Conference": "", "DateFrom": "", "DateTo": "",
            "Division": "", "GameScope": "", "GameSegment": "",
            "Height": "", "LastNGames": 0, "LeagueID": "00",
            "Location": "", "MeasureType": "Opponent", "Month": 0,
            "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
            "PaceAdjust": "N", "PerMode": "PerGame", "Period": 0,
            "PlayerExperience": "", "PlayerPosition": "",
            "PlusMinus": "N", "Rank": "N", "Season": season,
            "SeasonSegment": "", "SeasonType": season_type,
            "ShotClockRange": "", "StarterBench": "", "TeamID": 0,
            "TwoWay": 0, "VsConference": "", "VsDivision": "", "Weight": "",
        }
        data = self._request("leaguedashteamstats", params)
        return self._result_set_to_dicts(data)

    def get_opponent_shooting_closest_defender(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Opponent FG% by closest-defender distance from ``leaguedashoppptshot``."""
        params: dict[str, Any] = {
            "CloseDefDistRange": "", "College": "", "Conference": "",
            "Country": "", "DateFrom": "", "DateTo": "", "Division": "",
            "DraftPick": "", "DraftYear": "", "DribbleRange": "",
            "GameScope": "", "GameSegment": "", "GeneralRange": "",
            "Height": "", "LastNGames": 0, "LeagueID": "00",
            "Location": "", "Month": 0, "OpponentTeamID": 0,
            "Outcome": "", "PORound": 0, "PerMode": "Totals",
            "Period": 0, "PlayerExperience": "", "PlayerPosition": "",
            "Season": season, "SeasonSegment": "", "SeasonType": season_type,
            "ShotClockRange": "", "ShotDistRange": "", "StarterBench": "",
            "TeamID": 0, "TouchTimeRange": "", "VsConference": "",
            "VsDivision": "", "Weight": "",
        }
        data = self._request("leaguedashoppptshot", params)
        return self._result_set_to_dicts(data)

    # ------------------------------------------------------------------
    # Extended Endpoints (from NBA_Props_AI nba_data.py catalog)
    # ------------------------------------------------------------------

    def get_boxscore_traditional(
        self,
        game_id: str,
    ) -> list[dict[str, Any]]:
        """BoxScoreTraditionalV3 for a single game."""
        params = {"GameID": game_id, "LeagueID": "00"}
        data = self._request("boxscoretraditionalv3", params)
        return self._result_set_to_dicts(data)

    def get_boxscore_advanced(
        self,
        game_id: str,
    ) -> list[dict[str, Any]]:
        """BoxScoreAdvancedV3 for a single game."""
        params = {"GameID": game_id, "LeagueID": "00"}
        data = self._request("boxscoreadvancedv3", params)
        return self._result_set_to_dicts(data)

    def get_league_game_finder(
        self,
        season: str,
        team_id: int | None = None,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """LeagueGameFinder — flexible game search."""
        params: dict[str, Any] = {
            "LeagueID": "00",
            "Season": season,
            "SeasonType": season_type,
        }
        if team_id is not None:
            params["TeamID"] = team_id
        data = self._request("leaguegamefinder", params)
        return self._result_set_to_dicts(data)

    def get_common_player_info(
        self,
        player_id: int,
    ) -> list[dict[str, Any]]:
        """CommonPlayerInfo for a specific player."""
        params = {"PlayerID": player_id, "LeagueID": "00"}
        data = self._request("commonplayerinfo", params)
        return self._result_set_to_dicts(data)

    def get_common_team_roster(
        self,
        team_id: int,
        season: str,
    ) -> list[dict[str, Any]]:
        """CommonTeamRoster for a specific team/season."""
        params = {"TeamID": team_id, "Season": season, "LeagueID": "00"}
        data = self._request("commonteamroster", params)
        return self._result_set_to_dicts(data)

    def get_scoreboard(
        self,
        game_date: str,
    ) -> list[dict[str, Any]]:
        """Scoreboard for a specific date (MM/DD/YYYY)."""
        params = {"GameDate": game_date, "LeagueID": "00", "DayOffset": 0}
        data = self._request("scoreboardv2", params)
        return self._result_set_to_dicts(data)

    def get_player_dashboard_by_game_splits(
        self,
        player_id: int,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Player dashboard split by game."""
        params: dict[str, Any] = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": season_type,
            "MeasureType": "Base",
            "PerMode": "PerGame",
            "LeagueID": "00",
        }
        data = self._request("playerdashboardbygamesplits", params)
        return self._result_set_to_dicts(data)

    def raw_request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a raw request to any NBA Stats endpoint.

        Use this for endpoints not yet wrapped with a dedicated method.
        """
        return self._request(endpoint, params)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tracking_base_params(
        self, season: str, season_type: str
    ) -> dict[str, Any]:
        """Common params for tracking endpoints."""
        return {
            "DateFrom": "", "DateTo": "", "GameSegment": "",
            "LastNGames": 0, "LeagueID": "00", "Location": "",
            "Month": 0, "OpponentTeamID": 0, "Outcome": "",
            "PerMode": "Totals", "Period": 0, "PlayerID": 0,
            "Season": season, "SeasonSegment": "",
            "SeasonType": season_type, "TeamID": 0,
            "VsConference": "", "VsDivision": "",
        }

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a GET with rate limiting, retries, and backoff."""
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(1, _MAX_RETRIES + 1):
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                logger.debug(
                    "NBA Stats request attempt=%d endpoint=%s",
                    attempt, endpoint,
                )
                resp = self._session.get(
                    url, params=params, timeout=self.timeout,
                )
                self._last_request_ts = time.monotonic()
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                logger.warning(
                    "NBA Stats HTTP %s on attempt %d/%d for %s",
                    status, attempt, _MAX_RETRIES, endpoint,
                )
                if attempt == _MAX_RETRIES:
                    raise NBAStatsRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: "
                        f"HTTP {status} from {endpoint}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "NBA Stats request error on attempt %d/%d for %s: %s",
                    attempt, _MAX_RETRIES, endpoint, exc,
                )
                if attempt == _MAX_RETRIES:
                    raise NBAStatsRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {endpoint}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise NBAStatsRequestError(  # pragma: no cover
            f"Unexpected retry exhaustion for {endpoint}"
        )

    @staticmethod
    def _result_set_to_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert NBA Stats ``resultSets`` columnar response to row dicts."""
        try:
            result_set = data["resultSets"][0]
            headers = result_set["headers"]
            rows = result_set["rowSet"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected NBA Stats response structure: %s", exc)
            return []
        return [dict(zip(headers, row)) for row in rows]
