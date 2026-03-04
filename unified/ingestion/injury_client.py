"""Unified injury report client.

Merges ValueHunter's snapshot/dedup model with NBA_Props_AI's
multi-source aggregation (ESPN + Rotowire).

Based on:
- ValueHunter/src/nba_props/ingestion/injury_reports.py (snapshot model)
- NBA_Props_AI/core_best_v3/nba_props/injuries.py (ESPN + Rotowire sources)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Injury-window constants (local arena time, 24-h format)
DAY_BEFORE_DEADLINE = "17:00"
B2B_DEADLINE = "13:00"
GAMEDAY_WINDOW = ("11:00", "13:00")
EARLY_TIP_WINDOW = ("08:00", "10:00")

# URLs
_NBA_INJURY_JSON_URL = (
    "https://cdn.nba.com/static/json/liveData/injuries/injuries_all.json"
)
_NBA_INJURY_HTML_URL = "https://official.nba.com/injury-report/"
_ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_ROTOWIRE_URL = "https://www.rotowire.com/basketball/tables/injury-report.php"

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

STATUS_MAP = {
    "out": "OUT",
    "o": "OUT",
    "doubtful": "DOUBTFUL",
    "d": "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "q": "QUESTIONABLE",
    "probable": "PROBABLE",
    "p": "PROBABLE",
    "available": "AVAILABLE",
    "active": "AVAILABLE",
    "day-to-day": "QUESTIONABLE",
    "gtd": "QUESTIONABLE",
}


class InjuryReportError(Exception):
    """Raised when fetching or parsing the injury report fails."""


class InjuryClient:
    """Unified injury client with multi-source support and snapshot dedup.

    Sources:
    1. NBA CDN JSON feed (primary)
    2. ESPN API (fallback)
    3. Rotowire HTML scrape (fallback)
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_BROWSER_HEADERS)
        self._snapshots: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Multi-source fetch
    # ------------------------------------------------------------------

    def fetch_injuries(
        self,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        """Fetch injury reports from available sources.

        Parameters
        ----------
        mode:
            "auto" (try all sources in order), "espn", "rotowire", or "nba_cdn".
        """
        if mode == "espn":
            return self._fetch_espn()
        elif mode == "rotowire":
            return self._fetch_rotowire()
        elif mode == "nba_cdn":
            return self._fetch_nba_cdn()

        # Auto mode: try in order
        for source_fn in [self._fetch_nba_cdn, self._fetch_espn, self._fetch_rotowire]:
            try:
                result = source_fn()
                if result:
                    return result
            except Exception as exc:
                logger.info("Injury source %s failed: %s", source_fn.__name__, exc)
                continue

        logger.warning("All injury sources failed, returning empty list.")
        return []

    # ------------------------------------------------------------------
    # Snapshot management (from ValueHunter)
    # ------------------------------------------------------------------

    def create_snapshot(
        self,
        reports: list[dict[str, Any]],
        game_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create immutable snapshots with deduplication."""
        now_utc = datetime.now(timezone.utc)
        created: list[dict[str, Any]] = []

        for report in reports:
            content_hash = self._compute_report_hash(report)
            snapshot = {
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

            duplicate = any(
                s["content_hash"] == content_hash
                and s["player_id"] == snapshot["player_id"]
                and s["game_id"] == snapshot["game_id"]
                for s in self._snapshots
            )
            if duplicate:
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
        """Return most recent snapshot for player before as_of_utc."""
        if isinstance(as_of_utc, str):
            as_of_utc = datetime.fromisoformat(as_of_utc)
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
    # Source: NBA CDN JSON
    # ------------------------------------------------------------------

    def _fetch_nba_cdn(self) -> list[dict[str, Any]]:
        resp = self._session.get(_NBA_INJURY_JSON_URL, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        raw_entries: list[dict[str, Any]] = []
        league = payload.get("league") or payload
        if isinstance(league, dict):
            entries = league.get("standard") or league.get("injuries") or []
            if isinstance(entries, list):
                raw_entries = entries
        elif isinstance(league, list):
            raw_entries = league

        return [self._normalise_entry(e, "nba_cdn") for e in raw_entries]

    # ------------------------------------------------------------------
    # Source: ESPN API (from NBA_Props_AI)
    # ------------------------------------------------------------------

    def _fetch_espn(self) -> list[dict[str, Any]]:
        resp = self._session.get(_ESPN_API_URL, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        results: list[dict[str, Any]] = []
        for team_block in payload.get("items", []):
            team_name = team_block.get("team", {}).get("abbreviation", "")
            for athlete in team_block.get("injuries", []):
                status_raw = athlete.get("status", "")
                results.append({
                    "player_name": athlete.get("athlete", {}).get("displayName"),
                    "player_id": athlete.get("athlete", {}).get("id"),
                    "team": team_name,
                    "status": STATUS_MAP.get(status_raw.lower(), "UNKNOWN"),
                    "status_raw": status_raw,
                    "reason": athlete.get("type", {}).get("description", ""),
                    "game_date": None,
                    "game_id": None,
                    "source": "espn",
                })
        return results

    # ------------------------------------------------------------------
    # Source: Rotowire HTML (from NBA_Props_AI)
    # ------------------------------------------------------------------

    def _fetch_rotowire(self) -> list[dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("beautifulsoup4 required for Rotowire scraping")
            return []

        resp = self._session.get(_ROTOWIRE_URL, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return []

        rows = table.find_all("tr")
        if not rows:
            return []

        header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        results: list[dict[str, Any]] = []

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < len(header_cells):
                continue
            entry = dict(zip(header_cells, cells))
            results.append(self._normalise_entry(entry, "rotowire"))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_entry(raw: dict[str, Any], source: str = "unknown") -> dict[str, Any]:
        """Map varied field names to canonical schema."""
        def _get(keys: list[str], default: Any = None) -> Any:
            for k in keys:
                if k in raw:
                    return raw[k]
            return default

        status_raw = _get(["injuryStatus", "status", "Status", "Current Status", "Game Status"]) or ""
        return {
            "player_name": _get(["playerName", "Player", "player_name", "Name"]),
            "player_id": _get(["personId", "playerId", "player_id"]),
            "team": _get(["teamTricode", "team", "Team", "teamAbbreviation"]),
            "status": STATUS_MAP.get(str(status_raw).lower(), "UNKNOWN"),
            "status_raw": status_raw,
            "reason": _get(["reason", "Reason", "injuryDescription", "Category"]),
            "game_date": _get(["gameDate", "game_date", "Game Date", "Date"]),
            "game_id": _get(["gameId", "game_id"]),
            "source": source,
        }

    @staticmethod
    def _compute_report_hash(report_dict: dict[str, Any]) -> str:
        serialised = json.dumps(report_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
