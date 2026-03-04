from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import re
from html import unescape
from typing import Any, Dict, List, Optional, Tuple
import requests

from .utils import name_key, norm_name

ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ROTOWIRE_INJURY_URL = "https://www.rotowire.com/basketball/tables/injury-report.php"
ROTOWIRE_MINUTES_URL = "https://www.rotowire.com/basketball/ajax/get-projected-minutes.php"

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (NBAPropsAI/1.0)"}

TEAM_ABBR_TO_NBA_NAME = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "BRK": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NO": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "UTH": "Utah Jazz",
    "WAS": "Washington Wizards",
}

ESPN_TO_NBA_TEAM = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}

@dataclass(frozen=True)
class InjuryItem:
    team: str
    player: str
    status_raw: str
    status: str
    short_comment: str
    long_comment: str
    date: str

@dataclass(frozen=True)
class MinutesProjectionItem:
    team_abbr: str
    player: str
    projected_minutes: float
    floor_minutes: Optional[float]
    ceiling_minutes: Optional[float]
    std_minutes: Optional[float]
    status_raw: str
    status_hint: str

def _strip_html(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", unescape(str(text or "")))).strip()

def _rotowire_headers() -> Dict[str, str]:
    headers = dict(REQUEST_HEADERS)
    cookie = str(os.getenv("ROTOWIRE_COOKIE", "")).strip()
    if cookie:
        headers["Cookie"] = cookie
    return headers

def _parse_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out

def map_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return "UNKNOWN"
    normalized = re.sub(r"[^a-z0-9]+", " ", s).strip()
    tokens = normalized.split()
    if (
        "out for season" in normalized
        or "suspension" in normalized
        or "inactive" in normalized
        or "ofs" in tokens
        or normalized == "out"
        or normalized.startswith("out ")
    ):
        return "OUT"
    if "doubtful" in s:
        return "DOUBTFUL"
    if any(x in s for x in {"questionable", "day-to-day", "day to day", "gtd", "game time decision"}):
        return "QUESTIONABLE"
    if "probable" in s:
        return "PROBABLE"
    if s in {"available", "active", "no"}:
        return "AVAILABLE"
    if "available" in s or "active" in s or "will play" in s or "expected to play" in s:
        return "AVAILABLE"
    return "UNKNOWN"

def fetch_espn_injuries(timeout_s: int = 15) -> Tuple[datetime, List[InjuryItem]]:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    r = requests.get(ESPN_INJURY_URL, headers=REQUEST_HEADERS, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    out: List[InjuryItem] = []
    for team_entry in data.get("injuries", []):
        team_name = ESPN_TO_NBA_TEAM.get(team_entry.get("displayName", "Unknown"), team_entry.get("displayName", "Unknown"))
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {}) or {}
            player = norm_name(athlete.get("displayName", "Unknown"))
            status_raw = (inj.get("status") or "Unknown").strip()
            out.append(InjuryItem(
                team=team_name,
                player=player,
                status_raw=status_raw,
                status=map_status(status_raw),
                short_comment=(inj.get("shortComment") or "").strip(),
                long_comment=(inj.get("longComment") or "").strip(),
                date=(inj.get("date") or "")[:10],
            ))
    return ts, out

def fetch_rotowire_injuries(timeout_s: int = 15) -> Tuple[datetime, List[InjuryItem]]:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    params = {"team": "ALL", "pos": "ALL", "ajax": "1"}
    r = requests.get(ROTOWIRE_INJURY_URL, params=params, headers=_rotowire_headers(), timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Rotowire injuries payload.")

    out: List[InjuryItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        player = norm_name(str(row.get("player") or "").strip())
        if not player:
            first = str(row.get("firstname") or "").strip()
            last = str(row.get("lastname") or "").strip()
            player = norm_name(f"{first} {last}".strip())
        if not player:
            continue

        team_abbr = str(row.get("team") or "").strip().upper()
        team_name = TEAM_ABBR_TO_NBA_NAME.get(team_abbr, team_abbr or "Unknown")
        status_raw = _strip_html(row.get("status") or "")
        out.append(
            InjuryItem(
                team=team_name,
                player=player,
                status_raw=status_raw,
                status=map_status(status_raw),
                short_comment=_strip_html(row.get("injury") or ""),
                long_comment="",
                date=_strip_html(row.get("rDate") or ""),
            )
        )
    return ts, out

def fetch_rotowire_team_projected_minutes(
    team_abbr: str,
    timeout_s: int = 15,
) -> Tuple[datetime, List[MinutesProjectionItem], Optional[str]]:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    team = str(team_abbr or "").strip().upper()
    if not team:
        return ts, [], "missing_team"

    r = requests.get(
        ROTOWIRE_MINUTES_URL,
        params={"team": team},
        headers=_rotowire_headers(),
        timeout=timeout_s,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        err = str(payload.get("error") or "").strip() or "unknown_error"
        return ts, [], err
    if not isinstance(payload, list):
        return ts, [], "unexpected_payload"

    out: List[MinutesProjectionItem] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        player = norm_name(str(row.get("name") or "").strip())
        if not player:
            first = str(row.get("firstname") or "").strip()
            last = str(row.get("lastname") or "").strip()
            player = norm_name(f"{first} {last}".strip())
        if not player:
            continue
        proj = _parse_float(row.get("proj"))
        if proj is None:
            continue
        status_raw = _strip_html(row.get("inj") or "")
        out.append(
            MinutesProjectionItem(
                team_abbr=team,
                player=player,
                projected_minutes=float(max(proj, 0.0)),
                floor_minutes=_parse_float(row.get("min")),
                ceiling_minutes=_parse_float(row.get("max")),
                std_minutes=_parse_float(row.get("stdev")),
                status_raw=status_raw,
                status_hint=map_status(status_raw),
            )
        )
    return ts, out, None

def fetch_rotowire_projected_minutes(
    team_abbrs: List[str],
    timeout_s: int = 15,
) -> Tuple[datetime, Dict[str, Dict[str, MinutesProjectionItem]], Dict[str, str]]:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    by_team: Dict[str, Dict[str, MinutesProjectionItem]] = {}
    errors: Dict[str, str] = {}
    unique_teams = sorted({str(x or "").strip().upper() for x in team_abbrs if str(x or "").strip()})

    for team in unique_teams:
        _, items, err = fetch_rotowire_team_projected_minutes(team, timeout_s=timeout_s)
        if err:
            errors[team] = err
            by_team[team] = {}
            continue
        team_map: Dict[str, MinutesProjectionItem] = {}
        for item in items:
            key = name_key(item.player)
            if not key:
                continue
            # Keep first row deterministically; payload is already one row per player in normal cases.
            team_map.setdefault(key, item)
        by_team[team] = team_map
    return ts, by_team, errors

def _merge_injuries_by_priority(groups: List[Tuple[str, List[InjuryItem]]]) -> List[InjuryItem]:
    merged: Dict[Tuple[str, str], InjuryItem] = {}
    for _, rows in groups:
        for row in rows:
            key = (str(row.team), name_key(row.player))
            if not key[1]:
                continue
            merged.setdefault(key, row)
    return list(merged.values())

def fetch_injuries_with_fallback(
    source: str = "auto",
    timeout_s: int = 15,
) -> Tuple[datetime, List[InjuryItem], Dict[str, Any]]:
    mode = str(source or "auto").strip().lower()
    if mode not in {"auto", "rotowire", "espn"}:
        raise ValueError(f"Unsupported injury source: {source}")

    provider_rows: List[Tuple[str, List[InjuryItem]]] = []
    provider_meta: List[Dict[str, Any]] = []
    timestamps: List[datetime] = []

    def _record(provider: str, fn) -> None:
        try:
            pts, rows = fn(timeout_s=timeout_s)
            provider_rows.append((provider, rows))
            provider_meta.append({"provider": provider, "ok": True, "count": len(rows), "error": None})
            timestamps.append(pts)
        except Exception as exc:
            provider_meta.append({"provider": provider, "ok": False, "count": 0, "error": str(exc)})

    if mode in {"auto", "rotowire"}:
        _record("rotowire", fetch_rotowire_injuries)
    if mode in {"auto", "espn"}:
        _record("espn", fetch_espn_injuries)

    if mode == "rotowire":
        ordered_groups = [g for g in provider_rows if g[0] == "rotowire"]
    elif mode == "espn":
        ordered_groups = [g for g in provider_rows if g[0] == "espn"]
    else:
        # Auto mode prioritizes Rotowire rows, then backfills from ESPN.
        ordered_groups = sorted(provider_rows, key=lambda x: 0 if x[0] == "rotowire" else 1)

    merged = _merge_injuries_by_priority(ordered_groups)
    ts = max(timestamps) if timestamps else datetime.now(timezone.utc).replace(microsecond=0)

    primary = "none"
    for row in provider_meta:
        if row.get("ok") and int(row.get("count") or 0) > 0:
            primary = str(row.get("provider"))
            break

    snapshot = {
        "source_mode": mode,
        "primary_source": primary,
        "providers": provider_meta,
        "count": len(merged),
    }
    return ts, merged, snapshot

def build_team_injury_maps(
    injuries: List[InjuryItem],
    team_roster_by_name: Dict[str, int],
    opp_roster_by_name: Dict[str, int],
) -> Tuple[Dict[str, str], Dict[int, str]]:
    # Returns by_name, by_player_id
    by_name: Dict[str, str] = {}
    by_id: Dict[int, str] = {}

    # Try exact roster mapping first
    normalized_team = {name_key(k): v for k, v in team_roster_by_name.items()}
    normalized_opp = {name_key(k): v for k, v in opp_roster_by_name.items()}
    merged = {**normalized_team, **normalized_opp}

    for item in injuries:
        name = name_key(item.player)
        if not name:
            continue
        by_name[name] = item.status
        pid = merged.get(name)
        if pid is not None:
            by_id[int(pid)] = item.status
    return by_name, by_id
