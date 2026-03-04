"""TheRundown API client — NEW odds data source.

Provides NBA events and player prop markets.
API docs: https://docs.therundown.io/introduction
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://therundown-therundown-v1.p.rapidapi.com"
_DEFAULT_TIMEOUT = 15
_DEFAULT_RATE_LIMIT = 0.5
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0

# NBA sport ID for TheRundown
NBA_SPORT_ID = 4


class TheRundownRequestError(Exception):
    """Raised when a TheRundown API request fails after retries."""


class TheRundownClient:
    """HTTP client for TheRundown API.

    Parameters
    ----------
    api_key:
        RapidAPI key for TheRundown.
    base_url:
        Root URL for the API.
    timeout:
        Per-request timeout in seconds.
    rate_limit:
        Minimum seconds between requests.
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
        self._session.headers.update({
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
        })
        self._last_request_ts: float = 0.0

    def get_nba_events(
        self,
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch NBA events/games, optionally filtered by date.

        Parameters
        ----------
        date:
            Date in YYYY-MM-DD format. If None, returns today's events.

        Returns
        -------
        list[dict]
            One dict per NBA event.
        """
        path = f"/sports/{NBA_SPORT_ID}/events"
        params: dict[str, Any] = {}
        if date:
            path = f"/sports/{NBA_SPORT_ID}/events/{date}"

        data = self._request(path, params)

        if isinstance(data, list):
            return data
        return data.get("events", data.get("data", []))

    def get_event_markets(
        self,
        event_id: str,
    ) -> dict[str, Any]:
        """Fetch player prop markets for a specific event.

        Parameters
        ----------
        event_id:
            TheRundown event identifier.

        Returns
        -------
        dict
            Full markets payload including player props.
        """
        path = f"/events/{event_id}/markets"
        return self._request(path)

    def get_nba_events_with_props(
        self,
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch NBA events and enrich each with player prop markets.

        Convenience method that combines get_nba_events + get_event_markets.
        """
        events = self.get_nba_events(date=date)
        enriched = []
        for event in events:
            event_id = event.get("event_id") or event.get("id")
            if not event_id:
                enriched.append(event)
                continue
            try:
                markets = self.get_event_markets(str(event_id))
                event["markets"] = markets
            except TheRundownRequestError:
                logger.warning("Failed to fetch markets for event %s", event_id)
                event["markets"] = {}
            enriched.append(event)
        return enriched

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute GET with rate limiting and retry."""
        url = f"{self.base_url}{path}"
        query = params or {}

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
                logger.warning(
                    "TheRundown HTTP %s on attempt %d/%d for %s",
                    status, attempt, _MAX_RETRIES, path,
                )
                if status in (401, 403):
                    raise TheRundownRequestError(
                        f"Authentication failed (HTTP {status}). Check API key."
                    ) from exc
                if attempt == _MAX_RETRIES:
                    raise TheRundownRequestError(
                        f"Failed after {_MAX_RETRIES} attempts: HTTP {status} from {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise TheRundownRequestError(
                        f"Failed after {_MAX_RETRIES} attempts for {path}"
                    ) from exc
                time.sleep(_BACKOFF_FACTOR ** attempt)

        raise TheRundownRequestError(f"Unexpected retry exhaustion for {path}")  # pragma: no cover
