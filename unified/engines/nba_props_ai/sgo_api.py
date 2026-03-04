from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from .utils import american_to_implied_prob, norm_name

SGO_API_BASE = "https://api.sportsgameodds.com/v2"

STAT_TO_MODEL_MARKET = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threePointersMade": "FG3M",
}

DEFAULT_BOOKMAKER_PRIORITY = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "espnbet",
    "bet365",
    "prizepicks",
]


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


def _calc_vig(over_price: Optional[int], under_price: Optional[int]) -> Optional[float]:
    if over_price is None or under_price is None:
        return None
    try:
        return float(american_to_implied_prob(int(over_price)) + american_to_implied_prob(int(under_price)) - 1.0)
    except Exception:
        return None


def fetch_nba_events(api_key: str, *, timeout_seconds: int = 30) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "leagueID": "NBA",
        "oddsPresent": "true",
        "oddsAvailable": "true",
    }
    resp = requests.get(f"{SGO_API_BASE}/events", params=params, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise SportsGameOddsAPIError(f"Events request failed ({resp.status_code}): {resp.text[:300]}")
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise SportsGameOddsAPIError(f"Unexpected events payload: {str(payload)[:300]}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise SportsGameOddsAPIError("Missing events data array.")
    return data


def _event_team_abbr(event_payload: Dict[str, Any], team_name_to_abbr: Dict[str, str]) -> Tuple[str, str]:
    teams = event_payload.get("teams") or {}
    away_team = teams.get("away") or {}
    home_team = teams.get("home") or {}
    away_names = away_team.get("names") or {}
    home_names = home_team.get("names") or {}
    away_candidates = [
        away_names.get("short"),
        away_names.get("medium"),
        away_names.get("long"),
    ]
    home_candidates = [
        home_names.get("short"),
        home_names.get("medium"),
        home_names.get("long"),
    ]

    def _to_abbr(candidates: List[Any]) -> str:
        for item in candidates:
            s = str(item or "").strip().upper()
            if len(s) in {2, 3, 4} and s.isalpha():
                if len(s) == 3:
                    return s
            key = _norm_token(item or "")
            if key and key in team_name_to_abbr:
                return team_name_to_abbr[key]
        return ""

    away_abbr = _to_abbr(away_candidates)
    home_abbr = _to_abbr(home_candidates)
    if not away_abbr or not home_abbr:
        raise SportsGameOddsAPIError("Could not map SGO event teams to NBA abbreviations.")
    return away_abbr, home_abbr


def filter_events_for_local_date(events: List[Dict[str, Any]], *, local_date: str, timezone_name: str) -> List[Dict[str, Any]]:
    from zoneinfo import ZoneInfo

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
    event_payload: Dict[str, Any],
    *,
    team_name_to_abbr: Dict[str, str],
    bookmaker_priority: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    priority = bookmaker_priority or list(DEFAULT_BOOKMAKER_PRIORITY)
    away_abbr, home_abbr = _event_team_abbr(event_payload, team_name_to_abbr)
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
        player_name = norm_name((player_entry or {}).get("name") or player_id)
        player_team_id = str((player_entry or {}).get("teamID") or "").strip()
        if not player_team_id:
            continue
        team_hint = None
        if player_team_id and away_team_id and player_team_id == away_team_id:
            team_hint = away_abbr
        elif player_team_id and home_team_id and player_team_id == home_team_id:
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
                slot = grouped.setdefault(
                    key,
                    {
                        "player_id": player_id,
                        "player": player_name,
                        "team_hint": team_hint,
                        "market": STAT_TO_MODEL_MARKET[stat_id],
                        "bookmaker": str(book).lower(),
                        "line": float(line),
                        "over_price": None,
                        "under_price": None,
                    },
                )
                if side == "over":
                    slot["over_price"] = price
                else:
                    slot["under_price"] = price
                used_bookmaker = True

        if not used_bookmaker and base_line is not None and base_odds is not None:
            key = (player_id, stat_id, "consensus", float(base_line))
            slot = grouped.setdefault(
                key,
                {
                    "player_id": player_id,
                    "player": player_name,
                    "team_hint": team_hint,
                    "market": STAT_TO_MODEL_MARKET[stat_id],
                    "bookmaker": "consensus",
                    "line": float(base_line),
                    "over_price": None,
                    "under_price": None,
                },
            )
            if side == "over":
                slot["over_price"] = base_odds
            else:
                slot["under_price"] = base_odds

    by_player_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in grouped.values():
        if item.get("over_price") is None:
            continue
        item["vig"] = _calc_vig(item.get("over_price"), item.get("under_price"))
        key = (str(item["player"]), str(item["market"]))
        by_player_market.setdefault(key, []).append(item)

    selected: List[Dict[str, Any]] = []
    for _, candidates in by_player_market.items():
        best = sorted(
            candidates,
            key=lambda x: (
                _choose_bookmaker_rank(x.get("bookmaker"), priority),
                0 if x.get("under_price") is not None else 1,
                abs(float(x["vig"])) if x.get("vig") is not None else 999.0,
            ),
        )[0]
        selected.append(
            {
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
            }
        )
    return selected
