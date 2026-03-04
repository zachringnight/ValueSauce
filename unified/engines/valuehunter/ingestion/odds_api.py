"""The Odds API client for fetching player prop lines and computing no-vig probabilities.

This client focuses on the ``player_threes`` market but is structured to
support any player-prop market exposed by The Odds API.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORY_START = "2023-05-03"  # Earliest date with historical odds available
SNAPSHOT_INTERVAL_MIN = 5     # Minutes between consecutive line snapshots

_DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
_DEFAULT_TIMEOUT = 15
_DEFAULT_RATE_LIMIT = 0.25  # seconds between requests
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class OddsAPIRequestError(Exception):
    """Raised when a request to The Odds API fails after retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class OddsAPIClient:
    """HTTP client for The Odds API (https://the-odds-api.com).

    Parameters
    ----------
    api_key:
        API key for The Odds API.
    base_url:
        Root URL for the API.
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

    def get_upcoming_events(
        self,
        sport: str = "basketball_nba",
    ) -> list[dict[str, Any]]:
        """Fetch upcoming NBA events (games).

        Parameters
        ----------
        sport:
            Sport key recognised by The Odds API.

        Returns
        -------
        list[dict]
            One dict per upcoming event with ``id``, ``home_team``,
            ``away_team``, ``commence_time``, etc.
        """
        path = f"/sports/{sport}/events"
        data = self._request(path)

        if isinstance(data, list):
            return data

        # Some endpoints nest events under a key
        return data.get("events", data.get("data", []))

    def get_event_odds(
        self,
        event_id: str,
        markets: tuple[str, ...] = ("player_threes",),
        bookmakers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch current odds for a specific event.

        Parameters
        ----------
        event_id:
            The Odds API event identifier.
        markets:
            Tuple of market keys to request.
        bookmakers:
            Optional list of bookmaker keys to filter on.

        Returns
        -------
        dict
            Full odds payload for the event including bookmaker lines.
        """
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
        markets: tuple[str, ...] = ("player_threes",),
    ) -> dict[str, Any]:
        """Fetch historical odds for an event as of a given date.

        Parameters
        ----------
        event_id:
            The Odds API event identifier.
        date:
            ISO-8601 date(time) string, e.g. ``"2024-01-15T19:00:00Z"``.
        markets:
            Market keys to request.

        Returns
        -------
        dict
            Historical odds snapshot.
        """
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
        """Fetch all prop snapshots for an event, optionally filtered by player.

        This is a convenience wrapper around ``get_event_odds`` that
        flattens the nested bookmaker/market structure into a list of
        per-outcome records.

        Parameters
        ----------
        event_id:
            The Odds API event identifier.
        player_name:
            If provided, only return outcomes matching this player name
            (case-insensitive substring match).

        Returns
        -------
        list[dict]
            Flat list of outcome dicts with ``bookmaker``, ``market``,
            ``player``, ``point``, ``price``, ``name`` (Over/Under).
        """
        odds = self.get_event_odds(event_id)
        snapshots: list[dict[str, Any]] = []

        for bookmaker in odds.get("bookmakers", []):
            bk_key = bookmaker.get("key", "")
            bk_title = bookmaker.get("title", "")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")

                for outcome in market.get("outcomes", []):
                    description = outcome.get("description", "")

                    # Filter by player name if requested
                    if player_name and player_name.lower() not in description.lower():
                        continue

                    snapshots.append(
                        {
                            "bookmaker": bk_key,
                            "bookmaker_title": bk_title,
                            "market": market_key,
                            "player": description,
                            "name": outcome.get("name"),  # Over / Under
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        }
                    )

        return snapshots

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------

    def extract_novig_probs(
        self,
        over_price: int | float,
        under_price: int | float,
    ) -> tuple[float, float]:
        """Remove the vig from a two-way market and return fair probabilities.

        Parameters
        ----------
        over_price:
            American odds for the Over.
        under_price:
            American odds for the Under.

        Returns
        -------
        tuple[float, float]
            ``(over_prob, under_prob)`` that sum to 1.0.
        """
        over_implied = self._american_to_implied(over_price)
        under_implied = self._american_to_implied(under_price)

        total = over_implied + under_implied
        if total == 0:
            logger.warning(
                "Implied probabilities sum to zero (over=%s, under=%s). "
                "Returning equal split.",
                over_price,
                under_price,
            )
            return 0.5, 0.5

        return over_implied / total, under_implied / total

    def compute_hold_pct(
        self,
        over_price: int | float,
        under_price: int | float,
    ) -> float:
        """Compute the bookmaker hold (overround) percentage.

        Parameters
        ----------
        over_price:
            American odds for the Over.
        under_price:
            American odds for the Under.

        Returns
        -------
        float
            The hold percentage expressed as a decimal (e.g. 0.045 = 4.5%).
        """
        over_implied = self._american_to_implied(over_price)
        under_implied = self._american_to_implied(under_price)
        total = over_implied + under_implied

        if total == 0:
            return 0.0

        return total - 1.0

    @staticmethod
    def _american_to_implied(american_odds: int | float) -> float:
        """Convert American odds to an implied probability.

        Parameters
        ----------
        american_odds:
            American-format odds (e.g. ``-110``, ``+150``).

        Returns
        -------
        float
            Implied probability in ``[0, 1]``.

        Examples
        --------
        >>> OddsAPIClient._american_to_implied(-110)
        0.5238095238095238
        >>> OddsAPIClient._american_to_implied(150)
        0.4
        """
        odds = float(american_odds)
        if odds == 0:
            return 0.0

        if odds < 0:
            # Favourite: implied = |odds| / (|odds| + 100)
            return abs(odds) / (abs(odds) + 100.0)
        else:
            # Underdog: implied = 100 / (odds + 100)
            return 100.0 / (odds + 100.0)

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
        OddsAPIRequestError
            After exhausting all retry attempts.
        """
        url = f"{self.base_url}{path}"
        query: dict[str, Any] = {"apiKey": self.api_key}
        if params:
            query.update(params)

        for attempt in range(1, _MAX_RETRIES + 1):
            # Rate limit
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                logger.debug(
                    "Odds API request attempt=%d path=%s", attempt, path
                )
                resp = self._session.get(
                    url,
                    params=query,
                    timeout=self.timeout,
                )
                self._last_request_ts = time.monotonic()

                # Log remaining API quota from response headers
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    logger.debug(
                        "Odds API quota: %s remaining, %s used",
                        remaining,
                        used,
                    )

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                logger.warning(
                    "Odds API HTTP %s on attempt %d/%d for %s",
                    status,
                    attempt,
                    _MAX_RETRIES,
                    path,
                )

                # 401/403 = bad key, no point retrying
                if status in (401, 403):
                    raise OddsAPIRequestError(
                        f"Authentication failed (HTTP {status}). "
                        "Check your API key."
                    ) from exc

                # 429 = rate limited, back off harder
                if status == 429:
                    backoff = (_BACKOFF_FACTOR ** attempt) * 2
                    logger.info(
                        "Rate limited by Odds API, sleeping %.1fs", backoff
                    )
                    time.sleep(backoff)
                    continue

                if attempt == _MAX_RETRIES:
                    raise OddsAPIRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: "
                        f"HTTP {status} from {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Odds API request error on attempt %d/%d for %s: %s",
                    attempt,
                    _MAX_RETRIES,
                    path,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    raise OddsAPIRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise OddsAPIRequestError(  # pragma: no cover
            f"Unexpected retry exhaustion for {path}"
        )
