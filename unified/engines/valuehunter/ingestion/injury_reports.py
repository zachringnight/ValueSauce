"""Injury report ingester for the official NBA injury report.

Fetches, parses, and snapshots injury data so that models can look up
the *exact* injury status that was known before a given tip-off.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injury-window constants (local arena time, 24-h format)
# ---------------------------------------------------------------------------
DAY_BEFORE_DEADLINE = "17:00"  # 5 PM local - day-before submission deadline
B2B_DEADLINE = "13:00"         # 1 PM local - back-to-back early deadline
GAMEDAY_WINDOW = ("11:00", "13:00")  # Standard game-day update window
EARLY_TIP_WINDOW = ("08:00", "10:00")  # Early-tip (matinee) window

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_NBA_INJURY_URL = "https://official.nba.com/injury-report/"
_NBA_INJURY_JSON_URL = (
    "https://cdn.nba.com/static/json/liveData/injuries/injuries_all.json"
)
_DEFAULT_TIMEOUT = 15

_BROWSER_HEADERS = {
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class InjuryReportError(Exception):
    """Raised when fetching or parsing the injury report fails."""


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------
class InjuryReportIngester:
    """Fetches, snapshots, and queries NBA injury reports.

    Parameters
    ----------
    injury_url:
        URL for the JSON feed of current injury data.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        injury_url: str = _NBA_INJURY_JSON_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.injury_url = injury_url
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_BROWSER_HEADERS)

        # In-memory snapshot store keyed by (player_id, game_id).
        # Production deployments should back this with a database.
        self._snapshots: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_current_report(self) -> list[dict[str, Any]]:
        """Fetch the current NBA injury report.

        Attempts the JSON feed first; falls back to scraping the HTML page
        if the JSON endpoint is unavailable.

        Returns
        -------
        list[dict]
            One dict per player entry with at minimum the keys:
            ``player_name``, ``player_id``, ``team``, ``status``,
            ``reason``, ``game_date``, ``game_id``.
        """
        try:
            return self._fetch_json_feed()
        except Exception:
            logger.info(
                "JSON injury feed unavailable, falling back to HTML scrape."
            )

        return self._fetch_html_report()

    def create_snapshot(
        self,
        reports: list[dict[str, Any]],
        game_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create immutable snapshots of injury reports.

        Each snapshot includes a UTC timestamp and a content hash so that
        duplicate reports can be detected cheaply.

        Parameters
        ----------
        reports:
            List of injury report dicts (as returned by ``fetch_current_report``).
        game_id:
            Optional game ID to associate with every snapshot entry.

        Returns
        -------
        list[dict]
            The newly created snapshot entries.
        """
        now_utc = datetime.now(timezone.utc)
        created: list[dict[str, Any]] = []

        for report in reports:
            content_hash = self._compute_report_hash(report)

            snapshot: dict[str, Any] = {
                "snapshot_utc": now_utc.isoformat(),
                "content_hash": content_hash,
                "game_id": game_id or report.get("game_id"),
                "player_id": report.get("player_id"),
                "player_name": report.get("player_name"),
                "team": report.get("team"),
                "status": report.get("status"),
                "reason": report.get("reason"),
                "game_date": report.get("game_date"),
                "raw": report,
            }

            # De-duplicate: skip if we already have an identical hash for this
            # player + game combination.
            duplicate = any(
                s["content_hash"] == content_hash
                and s["player_id"] == snapshot["player_id"]
                and s["game_id"] == snapshot["game_id"]
                for s in self._snapshots
            )
            if duplicate:
                logger.debug(
                    "Skipping duplicate snapshot for player_id=%s game_id=%s",
                    snapshot["player_id"],
                    snapshot["game_id"],
                )
                continue

            self._snapshots.append(snapshot)
            created.append(snapshot)

        logger.info("Created %d new injury snapshots.", len(created))
        return created

    def get_latest_snapshot(
        self,
        player_id: int | str,
        game_id: str,
        as_of_utc: datetime | str,
    ) -> dict[str, Any] | None:
        """Return the most recent injury snapshot for a player before *as_of_utc*.

        Parameters
        ----------
        player_id:
            NBA player ID.
        game_id:
            NBA game ID.
        as_of_utc:
            The cutoff timestamp.  Only snapshots taken *before* this moment
            are considered.  Accepts an ISO-format string or a ``datetime``.

        Returns
        -------
        dict or None
            The snapshot dict, or ``None`` if no matching snapshot exists.
        """
        if isinstance(as_of_utc, str):
            as_of_utc = datetime.fromisoformat(as_of_utc)

        # Ensure timezone awareness
        if as_of_utc.tzinfo is None:
            as_of_utc = as_of_utc.replace(tzinfo=timezone.utc)

        player_id_str = str(player_id)
        best: dict[str, Any] | None = None
        best_ts: datetime | None = None

        for snap in self._snapshots:
            if str(snap.get("player_id")) != player_id_str:
                continue
            if snap.get("game_id") != game_id:
                continue

            snap_ts = datetime.fromisoformat(snap["snapshot_utc"])
            if snap_ts.tzinfo is None:
                snap_ts = snap_ts.replace(tzinfo=timezone.utc)

            if snap_ts >= as_of_utc:
                continue

            if best_ts is None or snap_ts > best_ts:
                best = snap
                best_ts = snap_ts

        return best

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_report_hash(report_dict: dict[str, Any]) -> str:
        """Return a SHA-256 hex digest of the report's content.

        The dict is serialised with sorted keys so that the hash is
        deterministic regardless of insertion order.
        """
        serialised = json.dumps(report_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal fetchers
    # ------------------------------------------------------------------

    def _fetch_json_feed(self) -> list[dict[str, Any]]:
        """Fetch and normalise the NBA CDN JSON injury feed."""
        resp = self._session.get(self.injury_url, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        # The CDN feed nests entries under "league" -> "standard"
        raw_entries: list[dict[str, Any]] = []
        try:
            league = payload.get("league") or payload
            # Handle different possible structures
            if isinstance(league, dict):
                entries = league.get("standard") or league.get("injuries") or []
                if isinstance(entries, list):
                    raw_entries = entries
                else:
                    # Entries might be nested under teams
                    for team_block in entries.values() if isinstance(entries, dict) else []:
                        if isinstance(team_block, list):
                            raw_entries.extend(team_block)
            elif isinstance(league, list):
                raw_entries = league
        except Exception as exc:
            raise InjuryReportError(
                f"Could not parse JSON injury feed: {exc}"
            ) from exc

        return [self._normalise_entry(e) for e in raw_entries]

    def _fetch_html_report(self) -> list[dict[str, Any]]:
        """Scrape the official NBA injury report HTML page.

        This is a best-effort fallback.  The HTML structure may change
        without notice; prefer the JSON feed.
        """
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        except ImportError as exc:
            raise InjuryReportError(
                "beautifulsoup4 is required for HTML fallback parsing. "
                "Install it with: pip install beautifulsoup4"
            ) from exc

        resp = self._session.get(
            _NBA_INJURY_URL,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            raise InjuryReportError("No injury table found on NBA injury page.")

        rows = table.find_all("tr")  # type: ignore[union-attr]
        if not rows:
            return []

        # First row is usually the header
        header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        results: list[dict[str, Any]] = []

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < len(header_cells):
                continue
            entry = dict(zip(header_cells, cells))
            results.append(self._normalise_entry(entry))

        return results

    @staticmethod
    def _normalise_entry(raw: dict[str, Any]) -> dict[str, Any]:
        """Map varied field names from the source into a canonical schema."""
        # The JSON and HTML feeds use different key names; normalise them.
        def _get(keys: list[str], default: Any = None) -> Any:
            for k in keys:
                if k in raw:
                    return raw[k]
            return default

        return {
            "player_name": _get(["playerName", "Player", "player_name", "Name"]),
            "player_id": _get(["personId", "playerId", "player_id"]),
            "team": _get([
                "teamTricode", "team", "Team", "teamAbbreviation",
            ]),
            "status": _get([
                "injuryStatus", "status", "Status", "Current Status",
                "Game Status",
            ]),
            "reason": _get([
                "reason", "Reason", "injuryDescription", "Category",
            ]),
            "game_date": _get(["gameDate", "game_date", "Game Date", "Date"]),
            "game_id": _get(["gameId", "game_id"]),
        }
