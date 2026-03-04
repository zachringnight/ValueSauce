from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from .utils import american_to_implied_prob, norm_name

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NBA_SPORT_KEY = "basketball_nba"

MARKET_KEY_TO_MODEL = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
}

DEFAULT_MARKETS = ["player_points", "player_rebounds", "player_assists", "player_threes"]

DEFAULT_BOOKMAKER_PRIORITY = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "espnbet",
    "bet365",
]

TEAM_ABBR_ALIAS = {
    "LAK": "LAL",
    "LAL": "LAL",
    "LAC": "LAC",
    "LOS": "LAL",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
}


class OddsAPIError(Exception):
    pass


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


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _extract_player_and_hint(description: str) -> Tuple[str, Optional[str]]:
    raw = norm_name(str(description or ""))
    if not raw:
        return "", None
    # Example: "LeBron James (LAL)"
    m = re.match(r"^(.*?)\s*\(([A-Za-z]{2,4})\)\s*$", raw)
    if m:
        player = norm_name(m.group(1))
        hint = TEAM_ABBR_ALIAS.get(m.group(2).upper(), m.group(2).upper())
        return player, hint
    return raw, None


def _choose_bookmaker_rank(book_key: str, priority: Iterable[str]) -> int:
    book_key = str(book_key or "").lower()
    pri = [str(x).lower() for x in priority]
    try:
        return pri.index(book_key)
    except ValueError:
        return len(pri) + 1


def _candidate_sort_key(candidate: Dict[str, Any], priority: Iterable[str]) -> Tuple[Any, ...]:
    rank = _choose_bookmaker_rank(candidate.get("bookmaker"), priority)
    has_under = candidate.get("under_price") is not None
    vig = candidate.get("vig")
    vig_abs = abs(float(vig)) if vig is not None else 999.0
    last_update = candidate.get("last_update") or ""
    # Prefer known book, paired over/under lines, tighter vig, freshest update
    return (rank, 0 if has_under else 1, vig_abs, -len(str(last_update)))


def _calc_vig(over_price: Optional[int], under_price: Optional[int]) -> Optional[float]:
    if over_price is None or under_price is None:
        return None
    try:
        return float(american_to_implied_prob(int(over_price)) + american_to_implied_prob(int(under_price)) - 1.0)
    except Exception:
        return None


def fetch_nba_events(api_key: str, *, timeout_seconds: int = 30) -> List[Dict[str, Any]]:
    url = f"{ODDS_API_BASE}/sports/{NBA_SPORT_KEY}/events"
    params = {"apiKey": api_key, "dateFormat": "iso"}
    resp = requests.get(url, params=params, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise OddsAPIError(f"Events request failed ({resp.status_code}): {resp.text[:300]}")
    payload = resp.json()
    if not isinstance(payload, list):
        raise OddsAPIError("Unexpected events payload shape.")
    return payload


def fetch_event_odds(
    api_key: str,
    event_id: str,
    *,
    regions: str = "us",
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
    odds_format: str = "american",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    mkts = markets or list(DEFAULT_MARKETS)
    url = f"{ODDS_API_BASE}/sports/{NBA_SPORT_KEY}/events/{event_id}/odds"
    params: Dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(mkts),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    resp = requests.get(url, params=params, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise OddsAPIError(f"Event odds request failed ({resp.status_code}) for event {event_id}: {resp.text[:300]}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise OddsAPIError(f"Unexpected event odds payload for event {event_id}")
    return payload


def _event_team_abbr(event_payload: Dict[str, Any], team_name_to_abbr: Dict[str, str]) -> Tuple[str, str]:
    away_team = str(event_payload.get("away_team") or "").strip()
    home_team = str(event_payload.get("home_team") or "").strip()
    away_abbr = team_name_to_abbr.get(_norm_token(away_team), "")
    home_abbr = team_name_to_abbr.get(_norm_token(home_team), "")
    if not away_abbr or not home_abbr:
        raise OddsAPIError(f"Could not map event team names to abbreviations: away='{away_team}', home='{home_team}'")
    return away_abbr, home_abbr


def parse_event_player_props(
    event_payload: Dict[str, Any],
    *,
    team_name_to_abbr: Dict[str, str],
    bookmaker_priority: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    priority = bookmaker_priority or list(DEFAULT_BOOKMAKER_PRIORITY)
    away_abbr, home_abbr = _event_team_abbr(event_payload, team_name_to_abbr)
    base_game_key = f"{away_abbr}@{home_abbr}"
    commence_time = str(event_payload.get("commence_time") or "")
    event_id = str(event_payload.get("id") or "")

    grouped: Dict[Tuple[str, str, str, float, Optional[str]], Dict[str, Any]] = {}

    for book in event_payload.get("bookmakers", []) or []:
        book_key = str(book.get("key") or "").lower()
        for market in book.get("markets", []) or []:
            market_key = str(market.get("key") or "")
            if market_key not in MARKET_KEY_TO_MODEL:
                continue
            market_update = market.get("last_update")
            for outcome in market.get("outcomes", []) or []:
                side_name = str(outcome.get("name") or "").strip().lower()
                if side_name not in {"over", "under"}:
                    continue
                player_raw, desc_hint = _extract_player_and_hint(outcome.get("description"))
                if not player_raw:
                    continue
                team_hint = desc_hint
                participant = str(outcome.get("participant") or "").strip().upper()
                if not team_hint and participant:
                    team_hint = TEAM_ABBR_ALIAS.get(participant, participant if len(participant) <= 4 else None)
                point = outcome.get("point")
                price = outcome.get("price")
                if point is None or price is None:
                    continue
                try:
                    point = float(point)
                    price = int(price)
                except Exception:
                    continue

                key = (book_key, market_key, player_raw, point, team_hint)
                slot = grouped.setdefault(
                    key,
                    {
                        "bookmaker": book_key,
                        "market_key": market_key,
                        "player": player_raw,
                        "point": point,
                        "team_hint": team_hint,
                        "over_price": None,
                        "under_price": None,
                        "last_update": market_update,
                    },
                )
                if side_name == "over":
                    slot["over_price"] = price
                else:
                    slot["under_price"] = price

    by_player_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in grouped.values():
        if item.get("over_price") is None:
            continue
        item["vig"] = _calc_vig(item.get("over_price"), item.get("under_price"))
        out_key = (str(item["player"]), str(item["market_key"]))
        by_player_market.setdefault(out_key, []).append(item)

    selected: List[Dict[str, Any]] = []
    for (player, market_key), candidates in by_player_market.items():
        best = sorted(candidates, key=lambda x: _candidate_sort_key(x, priority))[0]
        selected.append(
            {
                "event_id": event_id,
                "commence_time": commence_time,
                "event_start_utc": commence_time,
                "game_key_base": base_game_key,
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "player": player,
                "market": MARKET_KEY_TO_MODEL[market_key],
                "line": float(best["point"]),
                "odds_over": int(best["over_price"]),
                "odds_under": int(best["under_price"]) if best.get("under_price") is not None else None,
                "bookmaker": best.get("bookmaker"),
                "team_hint": best.get("team_hint"),
            }
        )
    return selected


def build_team_name_to_abbr_map() -> Dict[str, str]:
    from nba_api.stats.static import teams as nba_teams

    mapping: Dict[str, str] = {}
    for t in nba_teams.get_teams():
        abbr = str(t.get("abbreviation") or "").upper()
        for raw in [t.get("full_name"), f"{t.get('city')} {t.get('nickname')}", t.get("nickname"), t.get("city")]:
            key = _norm_token(raw or "")
            if key:
                mapping[key] = abbr
        # Common aliases
        if abbr == "LAL":
            mapping[_norm_token("LA Lakers")] = "LAL"
        if abbr == "LAC":
            mapping[_norm_token("LA Clippers")] = "LAC"
    return mapping


def filter_events_for_local_date(events: List[Dict[str, Any]], *, local_date: str, timezone_name: str) -> List[Dict[str, Any]]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone_name)
    want = pd.to_datetime(local_date).date()
    out: List[Dict[str, Any]] = []
    for ev in events:
        dt = _parse_iso_utc(ev.get("commence_time"))
        if dt is None:
            continue
        if dt.astimezone(tz).date() == want:
            out.append(ev)
    return out
