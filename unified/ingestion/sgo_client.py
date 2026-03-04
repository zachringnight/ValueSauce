"""Unified SportsGameOdds API client.

Merges NBA_Props_AI's multi-market parsing (PTS/REB/AST/FG3M) + consensus fallback
with nil's pagination and adapter patterns.

Based on: NBA_Props_AI/core_best_v3/nba_props/sgo_api.py
Extended with: nil/nba_props/src/adapters/sportsgameodds_adapter.py (pagination)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SGO_API_BASE = "https://api.sportsgameodds.com/v2"

STAT_TO_MODEL_MARKET = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threePointersMade": "FG3M",
}

DEFAULT_BOOKMAKER_PRIORITY = [
    "draftkings", "fanduel", "betmgm", "caesars",
    "espnbet", "bet365", "prizepicks",
]

_REQUEST_DELAY = 0.3
_MAX_RETRIES = 3


class SportsGameOddsAPIError(Exception):
    pass


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _parse_iso_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _parse_american(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().upper()
    if not s:
        return None
    if s.startswith("+"):
        s = s[1:]
    try:
        return int(s)
    except Exception:
        return None


def _parse_line(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _choose_bookmaker_rank(book_key: str, priority: Iterable[str]) -> int:
    book_key = str(book_key or "").lower()
    pri = [str(x).lower() for x in priority]
    try:
        return pri.index(book_key)
    except ValueError:
        return len(pri) + 1


def _american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def _calc_vig(over_price: Optional[int], under_price: Optional[int]) -> Optional[float]:
    if over_price is None or under_price is None:
        return None
    try:
        return float(_american_to_implied_prob(int(over_price)) + _american_to_implied_prob(int(under_price)) - 1.0)
    except Exception:
        return None


class SGOClient:
    """Unified SportsGameOdds client with pagination and multi-market support."""

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        self._last_request_ts: float = 0.0

    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Rate-limited GET with retry."""
        url = f"{SGO_API_BASE}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        for attempt in range(1, _MAX_RETRIES + 1):
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < _REQUEST_DELAY:
                time.sleep(_REQUEST_DELAY - elapsed)

            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                self._last_request_ts = time.monotonic()
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise SportsGameOddsAPIError(
                        f"SGO request failed after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)

        raise SportsGameOddsAPIError("Unexpected retry exhaustion")  # pragma: no cover

    def authenticate(self) -> bool:
        """Test API key by hitting account/usage."""
        if not self.api_key:
            return False
        try:
            resp = self._api_get("account/usage")
            return resp.get("success", False)
        except Exception:
            return False

    def fetch_nba_events(self) -> List[Dict[str, Any]]:
        """Fetch all NBA events with odds (paginated)."""
        params = {
            "leagueID": "NBA",
            "oddsPresent": "true",
            "oddsAvailable": "true",
        }
        return self._paginate("events", params)

    def fetch_historical_events(
        self,
        bookmaker_ids: str = "draftkings,fanduel",
    ) -> List[Dict[str, Any]]:
        """Fetch finalized NBA events with odds (paginated)."""
        params = {
            "leagueID": "NBA",
            "finalized": "true",
            "oddsPresent": "true",
            "limit": "50",
            "bookmakerID": bookmaker_ids,
        }
        return self._paginate("events", params)

    def fetch_incremental_events(
        self,
        since: datetime,
        bookmaker_ids: str = "draftkings,fanduel",
    ) -> List[Dict[str, Any]]:
        """Fetch upcoming NBA events since a given time (paginated)."""
        params = {
            "leagueID": "NBA",
            "oddsAvailable": "true",
            "limit": "50",
            "startsAfter": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmakerID": bookmaker_ids,
        }
        return self._paginate("events", params)

    def _paginate(self, endpoint: str, params: dict) -> List[Dict[str, Any]]:
        """Auto-paginate through cursor-based results."""
        all_results: List[Dict[str, Any]] = []
        cursor = None

        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._api_get(endpoint, params)

            if not isinstance(data, dict) or data.get("success") is not True:
                raise SportsGameOddsAPIError(
                    f"Unexpected payload: {str(data)[:300]}"
                )

            events = data.get("data", [])
            if not isinstance(events, list):
                break
            all_results.extend(events)

            cursor = data.get("nextCursor")
            if not cursor or not events:
                break

            logger.info("SGO: fetched %d events so far...", len(all_results))

        return all_results

    def filter_events_for_local_date(
        self,
        events: List[Dict[str, Any]],
        local_date: str,
        timezone_name: str = "America/New_York",
    ) -> List[Dict[str, Any]]:
        """Filter events by local date."""
        from zoneinfo import ZoneInfo

        import pandas as pd

        tz = ZoneInfo(timezone_name)
        want = pd.to_datetime(local_date).date()
        out: List[Dict[str, Any]] = []
        for ev in events:
            status = ev.get("status") or {}
            dt = _parse_iso_utc(status.get("startsAt"))
            if dt is None:
                continue
            if dt.astimezone(tz).date() == want:
                out.append(ev)
        return out

    def parse_event_player_props(
        self,
        event_payload: Dict[str, Any],
        *,
        team_name_to_abbr: Dict[str, str],
        bookmaker_priority: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract player props from an event, grouped by over/under pairs.

        Selects best bookmaker per priority. Falls back to consensus line.
        Supports all markets in STAT_TO_MODEL_MARKET.
        """
        priority = bookmaker_priority or list(DEFAULT_BOOKMAKER_PRIORITY)
        away_abbr, home_abbr = self._event_team_abbr(event_payload, team_name_to_abbr)
        base_game_key = f"{away_abbr}@{home_abbr}"
        event_id = str(event_payload.get("eventID") or "")
        status = event_payload.get("status") or {}
        commence_time = str(status.get("startsAt") or "")
        teams = event_payload.get("teams") or {}
        away_team_id = str((teams.get("away") or {}).get("teamID") or "").strip()
        home_team_id = str((teams.get("home") or {}).get("teamID") or "").strip()

        players = event_payload.get("players") or {}
        odds = event_payload.get("odds") or {}
        grouped: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}

        for odd in (odds.values() if isinstance(odds, dict) else []):
            if not isinstance(odd, dict):
                continue
            stat_id = str(odd.get("statID") or "")
            if stat_id not in STAT_TO_MODEL_MARKET:
                continue
            if str(odd.get("periodID") or "").lower() != "game":
                continue
            if str(odd.get("betTypeID") or "").lower() != "ou":
                continue
            side = str(odd.get("sideID") or "").lower()
            if side not in {"over", "under"}:
                continue

            player_id = str(odd.get("playerID") or odd.get("statEntityID") or "").strip()
            if not player_id:
                continue
            player_entry = players.get(player_id) if isinstance(players, dict) else None
            if not isinstance(player_entry, dict):
                continue
            player_name = re.sub(r"\s+", " ", ((player_entry or {}).get("name") or player_id).strip())
            player_team_id = str((player_entry or {}).get("teamID") or "").strip()
            if not player_team_id:
                continue

            team_hint = None
            if player_team_id == away_team_id:
                team_hint = away_abbr
            elif player_team_id == home_team_id:
                team_hint = home_abbr

            base_line = _parse_line(odd.get("bookOverUnder"))
            base_odds = _parse_american(odd.get("bookOdds"))
            by_book = odd.get("byBookmaker") or {}
            used_bookmaker = False

            if isinstance(by_book, dict):
                for book, quote in by_book.items():
                    if not isinstance(quote, dict):
                        continue
                    if quote.get("available") is False:
                        continue
                    line = _parse_line(quote.get("overUnder"))
                    price = _parse_american(quote.get("odds"))
                    if line is None or price is None:
                        continue
                    key = (player_id, stat_id, str(book).lower(), float(line))
                    slot = grouped.setdefault(key, {
                        "player_id": player_id,
                        "player": player_name,
                        "team_hint": team_hint,
                        "market": STAT_TO_MODEL_MARKET[stat_id],
                        "bookmaker": str(book).lower(),
                        "line": float(line),
                        "over_price": None,
                        "under_price": None,
                    })
                    if side == "over":
                        slot["over_price"] = price
                    else:
                        slot["under_price"] = price
                    used_bookmaker = True

            # Consensus fallback
            if not used_bookmaker and base_line is not None and base_odds is not None:
                key = (player_id, stat_id, "consensus", float(base_line))
                slot = grouped.setdefault(key, {
                    "player_id": player_id,
                    "player": player_name,
                    "team_hint": team_hint,
                    "market": STAT_TO_MODEL_MARKET[stat_id],
                    "bookmaker": "consensus",
                    "line": float(base_line),
                    "over_price": None,
                    "under_price": None,
                })
                if side == "over":
                    slot["over_price"] = base_odds
                else:
                    slot["under_price"] = base_odds

        # Select best bookmaker per player+market
        by_player_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for item in grouped.values():
            if item.get("over_price") is None:
                continue
            item["vig"] = _calc_vig(item.get("over_price"), item.get("under_price"))
            key = (str(item["player"]), str(item["market"]))
            by_player_market.setdefault(key, []).append(item)

        selected: List[Dict[str, Any]] = []
        for _, candidates in by_player_market.items():
            best = sorted(candidates, key=lambda x: (
                _choose_bookmaker_rank(x.get("bookmaker"), priority),
                0 if x.get("under_price") is not None else 1,
                abs(float(x["vig"])) if x.get("vig") is not None else 999.0,
            ))[0]
            selected.append({
                "event_id": event_id,
                "commence_time": commence_time,
                "event_start_utc": commence_time,
                "game_key_base": base_game_key,
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "player": best["player"],
                "market": best["market"],
                "line": float(best["line"]),
                "odds_over": int(best["over_price"]),
                "odds_under": int(best["under_price"]) if best.get("under_price") is not None else None,
                "bookmaker": best.get("bookmaker"),
                "team_hint": best.get("team_hint"),
            })
        return selected

    @staticmethod
    def _event_team_abbr(
        event_payload: Dict[str, Any],
        team_name_to_abbr: Dict[str, str],
    ) -> Tuple[str, str]:
        """Map SGO team names to NBA abbreviations."""
        teams = event_payload.get("teams") or {}
        away_team = teams.get("away") or {}
        home_team = teams.get("home") or {}
        away_names = away_team.get("names") or {}
        home_names = home_team.get("names") or {}

        def _to_abbr(candidates: list) -> str:
            for item in candidates:
                s = str(item or "").strip().upper()
                if len(s) in {2, 3, 4} and s.isalpha():
                    if len(s) == 3:
                        return s
                key = _norm_token(item or "")
                if key and key in team_name_to_abbr:
                    return team_name_to_abbr[key]
            return ""

        away_abbr = _to_abbr([away_names.get("short"), away_names.get("medium"), away_names.get("long")])
        home_abbr = _to_abbr([home_names.get("short"), home_names.get("medium"), home_names.get("long")])
        if not away_abbr or not home_abbr:
            raise SportsGameOddsAPIError("Could not map SGO event teams to NBA abbreviations.")
        return away_abbr, home_abbr
