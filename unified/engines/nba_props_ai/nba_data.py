from __future__ import annotations

import os
import signal
import socket
import time
import json
from io import StringIO
from typing import Dict, Optional, Tuple, List

import pandas as pd

from .cache import SQLiteCache
from .utils import parse_minutes_value

API_DELAY = 0.6


def _endpoint_timeout_seconds() -> float:
    raw = str(os.getenv("NBA_PROPS_ENDPOINT_TIMEOUT_SECONDS", "20")).strip()
    try:
        timeout_s = float(raw)
    except Exception:
        timeout_s = 20.0
    # Set <=0 to disable timeout wrapper.
    return max(timeout_s, 0.0)


def _cache_only_endpoints_enabled() -> bool:
    raw = str(os.getenv("NBA_PROPS_CACHE_ONLY_ENDPOINTS", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _run_endpoint_call(fetch_fn, *, label: str = ""):
    timeout_s = float(_endpoint_timeout_seconds())
    if timeout_s <= 0.0 or not hasattr(signal, "SIGALRM"):
        return fetch_fn()

    class _EndpointTimeout(Exception):
        pass

    def _handle_timeout(signum, frame):  # pragma: no cover - signal handler
        raise _EndpointTimeout(f"{label or 'endpoint'} timeout after {timeout_s:.1f}s")

    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_socket_timeout = socket.getdefaulttimeout()
    applied_socket_timeout = False
    # Guard against C-level SSL poll hangs by forcing a socket default timeout
    # while the endpoint call is running.
    if prev_socket_timeout is None or float(prev_socket_timeout) > timeout_s:
        socket.setdefaulttimeout(timeout_s)
        applied_socket_timeout = True
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fetch_fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)
        if applied_socket_timeout:
            socket.setdefaulttimeout(prev_socket_timeout)


def _decode_cached_data(data_json: str):
    try:
        return json.loads(data_json)
    except Exception:
        return data_json


def _read_cached_frame(data_json: str) -> pd.DataFrame:
    payload = _decode_cached_data(data_json)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    if not text:
        return pd.DataFrame()
    return pd.read_json(StringIO(text))


def _read_cached_frame_map(data_json: str) -> Dict[str, pd.DataFrame]:
    payload = _decode_cached_data(data_json)
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for key, value in payload.items():
        text = value if isinstance(value, str) else json.dumps(value)
        out[str(key)] = pd.read_json(StringIO(text)) if text else pd.DataFrame()
    return out


def _sleep():
    time.sleep(API_DELAY)

def _import_endpoints():
    from nba_api.stats.endpoints import (
        playergamelog,
        leaguedashteamstats,
        playerdashboardbyshootingsplits,
        playerdashptshots,
        commonteamroster,
        leaguedashplayerstats,
        teamplayeronoffdetails,
    )
    try:
        from nba_api.stats.endpoints import commonplayerinfo
    except Exception:
        commonplayerinfo = None
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv3
    except Exception:
        boxscoretraditionalv3 = None
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv2
    except Exception:
        boxscoretraditionalv2 = None
    try:
        from nba_api.stats.endpoints import boxscoreplayertrackv3
    except Exception:
        boxscoreplayertrackv3 = None
    try:
        from nba_api.stats.endpoints import boxscoreplayertrackv2
    except Exception:
        boxscoreplayertrackv2 = None
    try:
        from nba_api.stats.endpoints import boxscoreusagev3
    except Exception:
        boxscoreusagev3 = None
    try:
        from nba_api.stats.endpoints import boxscoreusagev2
    except Exception:
        boxscoreusagev2 = None
    try:
        from nba_api.stats.endpoints import boxscorescoringv3
    except Exception:
        boxscorescoringv3 = None
    try:
        from nba_api.stats.endpoints import boxscorescoringv2
    except Exception:
        boxscorescoringv2 = None
    try:
        from nba_api.stats.endpoints import boxscorematchupsv3
    except Exception:
        boxscorematchupsv3 = None
    try:
        from nba_api.stats.endpoints import gamerotation
    except Exception:
        gamerotation = None
    try:
        from nba_api.stats.endpoints import leaguedashlineups
    except Exception:
        leaguedashlineups = None
    try:
        from nba_api.stats.endpoints import teamdashlineups
    except Exception:
        teamdashlineups = None
    try:
        from nba_api.stats.endpoints import leagueseasonmatchups
    except Exception:
        leagueseasonmatchups = None
    try:
        from nba_api.stats.endpoints import matchupsrollup
    except Exception:
        matchupsrollup = None
    try:
        from nba_api.stats.endpoints import leaguedashptdefend
    except Exception:
        leaguedashptdefend = None
    try:
        from nba_api.stats.endpoints import leaguedashoppptshot
    except Exception:
        leaguedashoppptshot = None
    try:
        from nba_api.stats.endpoints import leaguedashptstats
    except Exception:
        leaguedashptstats = None
    try:
        from nba_api.stats.endpoints import assisttracker
    except Exception:
        assisttracker = None
    try:
        from nba_api.stats.endpoints import playerdashptshotdefend
    except Exception:
        playerdashptshotdefend = None
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except Exception:
        scoreboardv2 = None
    try:
        from nba_api.stats.endpoints import boxscoresummaryv2
    except Exception:
        boxscoresummaryv2 = None
    try:
        from nba_api.stats.endpoints import playbyplayv3
    except Exception:
        playbyplayv3 = None
    try:
        from nba_api.stats.endpoints import winprobabilitypbp
    except Exception:
        winprobabilitypbp = None
    try:
        from nba_api.stats.endpoints import teamestimatedmetrics
    except Exception:
        teamestimatedmetrics = None
    try:
        from nba_api.stats.endpoints import playerestimatedmetrics
    except Exception:
        playerestimatedmetrics = None
    try:
        from nba_api.stats.endpoints import playerdashptpass
    except Exception:
        playerdashptpass = None
    try:
        from nba_api.stats.endpoints import playerdashptreb
    except Exception:
        playerdashptreb = None
    try:
        from nba_api.stats.endpoints import synergyplaytypes
    except Exception:
        synergyplaytypes = None
    try:
        from nba_api.stats.endpoints import leaguehustlestatsteam
    except Exception:
        leaguehustlestatsteam = None
    try:
        from nba_api.stats.endpoints import leaguehustlestatsplayer
    except Exception:
        leaguehustlestatsplayer = None
    try:
        from nba_api.stats.endpoints import boxscorehustlev2
    except Exception:
        boxscorehustlev2 = None
    try:
        from nba_api.stats.endpoints import hustlestatsboxscore
    except Exception:
        hustlestatsboxscore = None
    try:
        from nba_api.stats.endpoints import teamplayeronoffsummary
    except Exception:
        teamplayeronoffsummary = None
    try:
        from nba_api.stats.endpoints import teamdashptshots
    except Exception:
        teamdashptshots = None
    try:
        from nba_api.stats.endpoints import leaguedashplayerptshot
    except Exception:
        leaguedashplayerptshot = None
    try:
        from nba_api.stats.endpoints import leaguedashteamptshot
    except Exception:
        leaguedashteamptshot = None
    try:
        from nba_api.stats.endpoints import leaguedashplayershotlocations
    except Exception:
        leaguedashplayershotlocations = None
    try:
        from nba_api.stats.endpoints import leaguedashptteamdefend
    except Exception:
        leaguedashptteamdefend = None
    try:
        from nba_api.stats.endpoints import leaguedashteamclutch
    except Exception:
        leaguedashteamclutch = None
    try:
        from nba_api.stats.endpoints import leaguedashplayerclutch
    except Exception:
        leaguedashplayerclutch = None
    try:
        from nba_api.stats.endpoints import playerdashboardbyclutch
    except Exception:
        playerdashboardbyclutch = None
    try:
        from nba_api.stats.endpoints import leaguedashteamshotlocations
    except Exception:
        leaguedashteamshotlocations = None
    try:
        from nba_api.stats.endpoints import leaguelineupviz
    except Exception:
        leaguelineupviz = None
    try:
        from nba_api.stats.endpoints import boxscoreadvancedv3
    except Exception:
        boxscoreadvancedv3 = None
    try:
        from nba_api.stats.endpoints import boxscoreadvancedv2
    except Exception:
        boxscoreadvancedv2 = None
    try:
        from nba_api.stats.endpoints import boxscorefourfactorsv3
    except Exception:
        boxscorefourfactorsv3 = None
    try:
        from nba_api.stats.endpoints import boxscorefourfactorsv2
    except Exception:
        boxscorefourfactorsv2 = None
    try:
        from nba_api.stats.endpoints import boxscoremiscv3
    except Exception:
        boxscoremiscv3 = None
    try:
        from nba_api.stats.endpoints import boxscoremiscv2
    except Exception:
        boxscoremiscv2 = None
    try:
        from nba_api.stats.endpoints import boxscoredefensivev2
    except Exception:
        boxscoredefensivev2 = None
    try:
        from nba_api.stats.endpoints import glalumboxscoresimilarityscore
    except Exception:
        glalumboxscoresimilarityscore = None
    try:
        from nba_api.stats.endpoints import shotchartlineupdetail
    except Exception:
        shotchartlineupdetail = None
    try:
        from nba_api.stats.endpoints import shotchartdetail
    except Exception:
        shotchartdetail = None
    try:
        from nba_api.stats.endpoints import shotchartleaguewide
    except Exception:
        shotchartleaguewide = None
    from nba_api.stats.library.http import NBAStatsHTTP
    from nba_api.stats.static import players, teams
    return {
        "playergamelog": playergamelog,
        "leaguedashteamstats": leaguedashteamstats,
        "playerdashboardbyshootingsplits": playerdashboardbyshootingsplits,
        "playerdashptshots": playerdashptshots,
        "commonteamroster": commonteamroster,
        "leaguedashplayerstats": leaguedashplayerstats,
        "teamplayeronoffdetails": teamplayeronoffdetails,
        "commonplayerinfo": commonplayerinfo,
        "boxscoretraditionalv3": boxscoretraditionalv3,
        "boxscoretraditionalv2": boxscoretraditionalv2,
        "boxscoreplayertrackv3": boxscoreplayertrackv3,
        "boxscoreplayertrackv2": boxscoreplayertrackv2,
        "boxscoreusagev3": boxscoreusagev3,
        "boxscoreusagev2": boxscoreusagev2,
        "boxscorescoringv3": boxscorescoringv3,
        "boxscorescoringv2": boxscorescoringv2,
        "boxscorematchupsv3": boxscorematchupsv3,
        "gamerotation": gamerotation,
        "leaguedashlineups": leaguedashlineups,
        "teamdashlineups": teamdashlineups,
        "leagueseasonmatchups": leagueseasonmatchups,
        "matchupsrollup": matchupsrollup,
        "leaguedashptdefend": leaguedashptdefend,
        "leaguedashoppptshot": leaguedashoppptshot,
        "leaguedashptstats": leaguedashptstats,
        "assisttracker": assisttracker,
        "playerdashptshotdefend": playerdashptshotdefend,
        "scoreboardv2": scoreboardv2,
        "boxscoresummaryv2": boxscoresummaryv2,
        "playbyplayv3": playbyplayv3,
        "winprobabilitypbp": winprobabilitypbp,
        "teamestimatedmetrics": teamestimatedmetrics,
        "playerestimatedmetrics": playerestimatedmetrics,
        "playerdashptpass": playerdashptpass,
        "playerdashptreb": playerdashptreb,
        "synergyplaytypes": synergyplaytypes,
        "leaguehustlestatsteam": leaguehustlestatsteam,
        "leaguehustlestatsplayer": leaguehustlestatsplayer,
        "boxscorehustlev2": boxscorehustlev2,
        "hustlestatsboxscore": hustlestatsboxscore,
        "teamplayeronoffsummary": teamplayeronoffsummary,
        "teamdashptshots": teamdashptshots,
        "leaguedashplayerptshot": leaguedashplayerptshot,
        "leaguedashteamptshot": leaguedashteamptshot,
        "leaguedashplayershotlocations": leaguedashplayershotlocations,
        "leaguedashptteamdefend": leaguedashptteamdefend,
        "leaguedashteamclutch": leaguedashteamclutch,
        "leaguedashplayerclutch": leaguedashplayerclutch,
        "playerdashboardbyclutch": playerdashboardbyclutch,
        "leaguedashteamshotlocations": leaguedashteamshotlocations,
        "leaguelineupviz": leaguelineupviz,
        "boxscoreadvancedv3": boxscoreadvancedv3,
        "boxscoreadvancedv2": boxscoreadvancedv2,
        "boxscorefourfactorsv3": boxscorefourfactorsv3,
        "boxscorefourfactorsv2": boxscorefourfactorsv2,
        "boxscoremiscv3": boxscoremiscv3,
        "boxscoremiscv2": boxscoremiscv2,
        "boxscoredefensivev2": boxscoredefensivev2,
        "glalumboxscoresimilarityscore": glalumboxscoresimilarityscore,
        "shotchartlineupdetail": shotchartlineupdetail,
        "shotchartdetail": shotchartdetail,
        "shotchartleaguewide": shotchartleaguewide,
        "NBAStatsHTTP": NBAStatsHTTP,
        "players": players,
        "teams": teams,
    }

def find_player(full_name: str) -> dict:
    eps = _import_endpoints()
    results = eps["players"].find_players_by_full_name(full_name)
    if not results:
        all_p = eps["players"].get_active_players()
        n = full_name.lower()
        results = [p for p in all_p if n in p["full_name"].lower()]
    if not results:
        raise ValueError(f"Player not found: {full_name}")
    return results[0]

def find_player_id(full_name: str) -> int:
    return int(find_player(full_name)["id"])

def get_all_teams() -> List[dict]:
    eps = _import_endpoints()
    return eps["teams"].get_teams()

def find_team_by_abbr(abbr: str) -> dict:
    abbr = abbr.upper()
    for t in get_all_teams():
        if t["abbreviation"].upper() == abbr:
            return t
    raise ValueError(f"Team not found: {abbr}")

def team_name_from_abbr(abbr: str) -> str:
    return find_team_by_abbr(abbr)["full_name"]

def get_common_team_roster(team_id: int, season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"team_id": team_id, "season": season}
    hit = cache.get("commonteamroster", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: eps["commonteamroster"].CommonTeamRoster(team_id=team_id, season=season).get_data_frames(),
                label="commonteamroster",
            )
            df = dfs[0] if len(dfs) else pd.DataFrame()
            cache.set("commonteamroster", params, df.to_json())
            return df
        except Exception as exc:
            last_exc = exc
            # brief local backoff for transient stats.nba.com/network timeouts
            time.sleep(1.0 + attempt * 0.5)
    if last_exc is not None:
        print(
            f"[nba_data] warning: commonteamroster fetch failed for team_id={team_id} season={season}: {last_exc}"
        )
    fallback = _fallback_roster_from_league_stats(team_id=team_id, season=season, cache=cache)
    if len(fallback):
        print(
            "[nba_data] info: commonteamroster fallback from leaguedashplayerstats "
            f"for team_id={team_id} season={season} rows={len(fallback)}"
        )
        cache.set("commonteamroster", params, fallback.to_json())
        return fallback
    return pd.DataFrame()


def _fallback_roster_from_league_stats(team_id: int, season: str, cache: SQLiteCache) -> pd.DataFrame:
    try:
        players = get_league_player_stats(season=season, cache=cache)
    except Exception:
        return pd.DataFrame()
    if len(players) == 0:
        return pd.DataFrame()

    team_col = next((c for c in ["TEAM_ID", "TEAMID"] if c in players.columns), "")
    name_col = next((c for c in ["PLAYER_NAME", "PLAYER"] if c in players.columns), "")
    id_col = next((c for c in ["PLAYER_ID", "PERSON_ID"] if c in players.columns), "")
    min_col = "MIN" if "MIN" in players.columns else ""
    if not team_col or not name_col or not id_col:
        return pd.DataFrame()

    team_ids = pd.to_numeric(players[team_col], errors="coerce")
    team_rows = players[team_ids == int(team_id)].copy()
    if len(team_rows) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "PLAYER": team_rows[name_col].astype(str).str.strip(),
            "PLAYER_ID": pd.to_numeric(team_rows[id_col], errors="coerce"),
            "MIN": pd.to_numeric(team_rows[min_col], errors="coerce") if min_col else pd.NA,
        }
    )
    out = out.dropna(subset=["PLAYER", "PLAYER_ID"]).copy()
    out = out[out["PLAYER"].astype(str).str.len() > 0].copy()
    if len(out) == 0:
        return pd.DataFrame()
    if "MIN" in out.columns:
        out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")
        out = out.sort_values(["MIN", "PLAYER"], ascending=[False, True], kind="mergesort")
    else:
        out = out.sort_values(["PLAYER"], ascending=[True], kind="mergesort")
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    out = out.drop_duplicates(subset=["PLAYER_ID"], keep="first")
    return out.reset_index(drop=True)

def get_roster_name_to_id(team_id: int, season: str, cache: SQLiteCache) -> Dict[str, int]:
    df = get_common_team_roster(team_id, season, cache)
    mapping = {}
    if len(df) == 0:
        return mapping
    player_col = "PLAYER" if "PLAYER" in df.columns else "PLAYER_NAME"
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PERSON_ID"
    for _, row in df.iterrows():
        mapping[str(row[player_col]).strip()] = int(row[id_col])
    return mapping

def get_team_advanced(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season, "measure": "Advanced"}
    hit = cache.get("leaguedashteamstats_adv", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["leaguedashteamstats"].LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames(),
        label="leaguedashteamstats_adv",
    )
    df = dfs[0] if len(dfs) else pd.DataFrame()
    cache.set("leaguedashteamstats_adv", params, df.to_json())
    return df

def get_team_opponent_pergame(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season, "measure": "Opponent"}
    hit = cache.get("leaguedashteamstats_opp", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["leaguedashteamstats"].LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
        ).get_data_frames(),
        label="leaguedashteamstats_opp",
    )
    df = dfs[0] if len(dfs) else pd.DataFrame()
    cache.set("leaguedashteamstats_opp", params, df.to_json())
    return df


def get_team_base_pergame(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season, "measure": "Base"}
    hit = cache.get("leaguedashteamstats_base", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["leaguedashteamstats"].LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        ).get_data_frames(),
        label="leaguedashteamstats_base",
    )
    df = dfs[0] if len(dfs) else pd.DataFrame()
    cache.set("leaguedashteamstats_base", params, df.to_json())
    return df

def get_league_player_stats(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season}
    hit = cache.get("leaguedashplayerstats", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["leaguedashplayerstats"].LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
        ).get_data_frames(),
        label="leaguedashplayerstats",
    )
    df = dfs[0] if len(dfs) else pd.DataFrame()
    cache.set("leaguedashplayerstats", params, df.to_json())
    return df


def get_common_player_info(player_id: int, cache: SQLiteCache) -> pd.DataFrame:
    params = {"player_id": int(player_id)}
    hit = cache.get("commonplayerinfo", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    endpoint = eps.get("commonplayerinfo")
    if endpoint is None:
        df = pd.DataFrame()
        cache.set("commonplayerinfo", params, df.to_json())
        return df
    try:
        _sleep()
        dfs = _run_endpoint_call(
            lambda: endpoint.CommonPlayerInfo(player_id=player_id).get_data_frames(),
            label="commonplayerinfo",
        )
        df = dfs[0] if len(dfs) else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    cache.set("commonplayerinfo", params, df.to_json())
    return df


def get_player_primary_position(player_id: int, cache: SQLiteCache) -> str:
    df = get_common_player_info(player_id, cache)
    if len(df) == 0:
        return ""
    pos_col = next((c for c in ["POSITION", "Pos", "POSITION_NAME", "PLAYER_POSITION"] if c in df.columns), None)
    if not pos_col:
        return ""
    raw = str(df.iloc[0][pos_col] or "").upper()
    for token in ["G", "F", "C"]:
        if token in raw:
            return token
    return ""


def get_team_opponent_pergame_by_position(season: str, position_abbr: str, cache: SQLiteCache) -> pd.DataFrame:
    pos = str(position_abbr or "").strip().upper()
    if pos not in {"G", "F", "C"}:
        pos = ""
    params = {"season": season, "measure": "Opponent", "position": pos}
    hit = cache.get("leaguedashteamstats_opp_position", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    try:
        _sleep()
        dfs = _run_endpoint_call(
            lambda: eps["leaguedashteamstats"].LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Opponent",
                per_mode_detailed="PerGame",
                player_position_abbreviation_nullable=pos,
            ).get_data_frames(),
            label="leaguedashteamstats_opp_position",
        )
        df = dfs[0] if len(dfs) else pd.DataFrame()
    except Exception as exc:
        print(
            "[nba_data] warning: position opponent split fetch failed "
            f"for season={season} position={pos or 'ALL'}: {exc}"
        )
        df = pd.DataFrame()
    cache.set("leaguedashteamstats_opp_position", params, df.to_json())
    return df

def get_player_gamelog(player_id: int, season: str, cache: SQLiteCache, n_games: int = 82) -> pd.DataFrame:
    params = {"player_id": player_id, "season": season}
    hit = cache.get("playergamelog", params)
    if hit:
        df = _read_cached_frame(hit.data_json)
        return df.head(n_games).copy()
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["playergamelog"].PlayerGameLog(player_id=player_id, season=season).get_data_frames(),
        label="playergamelog",
    )
    df = dfs[0] if len(dfs) else pd.DataFrame()
    if len(df) == 0:
        cache.set("playergamelog", params, df.to_json())
        return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE_PARSED"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE_PARSED", ascending=False).reset_index(drop=True)
        if "GAME_DATE_PARSED" in df.columns:
            df["DAYS_REST"] = df["GAME_DATE_PARSED"].diff(-1).dt.days.fillna(2).clip(lower=0)
    if "MATCHUP" in df.columns:
        df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.")
        df["OPP_ABBR"] = df["MATCHUP"].astype(str).apply(
            lambda x: x.split("vs. ")[-1].strip() if "vs." in x else (x.split("@ ")[-1].strip() if "@ " in x else "")
        )
    if "MIN" in df.columns:
        df["MIN"] = df["MIN"].apply(parse_minutes_value)
    cache.set("playergamelog", params, df.to_json())
    return df.head(n_games).copy()

def get_boxscore_traditional(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": game_id}
    hit = cache.get("boxscoretraditional", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    df = pd.DataFrame()
    if eps["boxscoretraditionalv3"] is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: eps["boxscoretraditionalv3"].BoxScoreTraditionalV3(
                    game_id=game_id, start_period=1, end_period=10, start_range=0, end_range=0, range_type=0
                ),
                label="boxscoretraditionalv3",
            )
            dsets = resp.get_dict()
            # V3 nested schema
            player_stats = dsets.get("boxScoreTraditional", {}).get("playerStats") or dsets.get("boxScoreTraditionalV3", {}).get("playerStats")
            if player_stats:
                df = pd.DataFrame(player_stats)
            _record_endpoint_call(cache, "boxscoretraditionalv3", ok=True)
        except Exception:
            _record_endpoint_call(cache, "boxscoretraditionalv3", ok=False)
            df = pd.DataFrame()
    if len(df) == 0 and eps["boxscoretraditionalv2"] is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: eps["boxscoretraditionalv2"].BoxScoreTraditionalV2(
                    game_id=game_id, start_period=1, end_period=10, start_range=0, end_range=0, range_type=0
                ),
                label="boxscoretraditionalv2",
            )
            dfs = _run_endpoint_call(lambda: resp.get_data_frames(), label="boxscoretraditionalv2")
            df = dfs[0] if len(dfs) else pd.DataFrame()
            _record_endpoint_call(cache, "boxscoretraditionalv2", ok=True)
        except Exception:
            _record_endpoint_call(cache, "boxscoretraditionalv2", ok=False)
            df = pd.DataFrame()
    cache.set("boxscoretraditional", params, df.to_json())
    return df

def _extract_v3_nested_frame(resp, root_candidates: List[str]) -> pd.DataFrame:
    try:
        payload = resp.get_dict()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    for root_key in root_candidates:
        root = payload.get(root_key)
        if isinstance(root, dict):
            rows = root.get("playerStats")
            if isinstance(rows, list):
                return pd.DataFrame(rows)
            rows = root.get("rows")
            if isinstance(rows, list):
                return pd.DataFrame(rows)
        if isinstance(root, list):
            return pd.DataFrame(root)
    return pd.DataFrame()


def _extract_v3_player_team_frames(resp, root_candidates: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    try:
        payload = resp.get_dict()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    for root_key in root_candidates:
        root = payload.get(root_key)
        if not isinstance(root, dict):
            continue
        p_rows = root.get("playerStats")
        t_rows = root.get("teamStats")
        if isinstance(p_rows, list) and len(p_rows):
            out["player"] = pd.DataFrame(p_rows)
        if isinstance(t_rows, list) and len(t_rows):
            out["team"] = pd.DataFrame(t_rows)
        if len(out["player"]) or len(out["team"]):
            return out

    try:
        dfs = resp.get_data_frames()
    except Exception:
        dfs = []
    if len(dfs):
        out["player"] = dfs[0]
    if len(dfs) > 1:
        out["team"] = dfs[1]
    return out


def _safe_first_data_frame(resp) -> pd.DataFrame:
    try:
        dfs = resp.get_data_frames()
    except Exception:
        dfs = []
    return dfs[0] if len(dfs) else pd.DataFrame()


def _frame_from_result_set_payload(data: dict) -> pd.DataFrame:
    result_sets = data.get("resultSets")
    if isinstance(result_sets, list):
        rs = result_sets[0] if result_sets else {}
    elif isinstance(result_sets, dict):
        rs = result_sets
    else:
        rs = {}

    headers = rs.get("headers") or rs.get("Headers") or []
    rows = rs.get("rowSet") or rs.get("RowSet") or rs.get("rows") or []

    # Some endpoints return header metadata objects instead of plain strings.
    if isinstance(headers, list) and headers and isinstance(headers[0], dict):
        normalized_headers = []
        for i, h in enumerate(headers):
            if not isinstance(h, dict):
                normalized_headers.append(str(h))
                continue
            normalized_headers.append(
                str(
                    h.get("name")
                    or h.get("columnName")
                    or h.get("column")
                    or h.get("label")
                    or i
                )
            )
        headers = normalized_headers

    if isinstance(rows, list):
        try:
            return pd.DataFrame(rows, columns=headers if isinstance(headers, list) and headers else None)
        except Exception:
            return pd.DataFrame(rows)
    return pd.DataFrame()


def _safe_http_frame(resp) -> pd.DataFrame:
    # First try nba_api's own parser (most stable across endpoint schema shifts).
    df = _safe_first_data_frame(resp)
    if len(df):
        return df
    # Fallback to manual payload parsing.
    try:
        data = resp.get_dict()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return _frame_from_result_set_payload(data)


def _record_endpoint_call(
    cache: SQLiteCache,
    endpoint: str,
    *,
    ok: bool,
    error: Optional[Exception] = None,
) -> None:
    try:
        cache.record_endpoint_call(
            str(endpoint),
            ok=bool(ok),
            error=None if ok else str(error or ""),
        )
    except Exception:
        pass

def get_boxscore_player_track(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": game_id}
    hit = cache.get("boxscoreplayertrackv3", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint_v3 = eps.get("boxscoreplayertrackv3")
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScorePlayerTrackV3(game_id=game_id),
                label="boxscoreplayertrackv3",
            )
            df = _extract_v3_nested_frame(resp, ["boxScorePlayerTrack", "boxScorePlayerTrackV3"])
            if len(df) == 0:
                df = _safe_first_data_frame(resp)
            _record_endpoint_call(cache, "boxscoreplayertrackv3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoreplayertrackv3", ok=False, error=exc)
            df = pd.DataFrame()
    if len(df) == 0:
        endpoint_v2 = eps.get("boxscoreplayertrackv2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScorePlayerTrackV2(game_id=game_id).get_data_frames(),
                    label="boxscoreplayertrackv2",
                )
                df = dfs[0] if len(dfs) else pd.DataFrame()
                _record_endpoint_call(cache, "boxscoreplayertrackv2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscoreplayertrackv2", ok=False, error=exc)
                df = pd.DataFrame()
    cache.set("boxscoreplayertrackv3", params, df.to_json())
    return df

def get_boxscore_usage(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": game_id}
    hit = cache.get("boxscoreusagev3", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint_v3 = eps.get("boxscoreusagev3")
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScoreUsageV3(game_id=game_id),
                label="boxscoreusagev3",
            )
            df = _extract_v3_nested_frame(resp, ["boxScoreUsage", "boxScoreUsageV3"])
            if len(df) == 0:
                df = _safe_first_data_frame(resp)
            _record_endpoint_call(cache, "boxscoreusagev3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoreusagev3", ok=False, error=exc)
            df = pd.DataFrame()
    if len(df) == 0:
        endpoint_v2 = eps.get("boxscoreusagev2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScoreUsageV2(game_id=game_id).get_data_frames(),
                    label="boxscoreusagev2",
                )
                df = dfs[0] if len(dfs) else pd.DataFrame()
                _record_endpoint_call(cache, "boxscoreusagev2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscoreusagev2", ok=False, error=exc)
                df = pd.DataFrame()
    cache.set("boxscoreusagev3", params, df.to_json())
    return df

def get_boxscore_scoring(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": game_id}
    hit = cache.get("boxscorescoringv3", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint_v3 = eps.get("boxscorescoringv3")
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScoreScoringV3(game_id=game_id),
                label="boxscorescoringv3",
            )
            df = _extract_v3_nested_frame(resp, ["boxScoreScoring", "boxScoreScoringV3"])
            if len(df) == 0:
                df = _safe_first_data_frame(resp)
            _record_endpoint_call(cache, "boxscorescoringv3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscorescoringv3", ok=False, error=exc)
            df = pd.DataFrame()
    if len(df) == 0:
        endpoint_v2 = eps.get("boxscorescoringv2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScoreScoringV2(game_id=game_id).get_data_frames(),
                    label="boxscorescoringv2",
                )
                df = dfs[0] if len(dfs) else pd.DataFrame()
                _record_endpoint_call(cache, "boxscorescoringv2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscorescoringv2", ok=False, error=exc)
                df = pd.DataFrame()
    cache.set("boxscorescoringv3", params, df.to_json())
    return df

def get_boxscore_matchups(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": game_id}
    hit = cache.get("boxscorematchupsv3", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint = eps.get("boxscorematchupsv3")
    if endpoint is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint.BoxScoreMatchupsV3(game_id=game_id),
                label="boxscorematchupsv3",
            )
            df = _extract_v3_nested_frame(resp, ["boxScoreMatchups", "boxScoreMatchupsV3"])
            if len(df) == 0:
                df = _safe_first_data_frame(resp)
            _record_endpoint_call(cache, "boxscorematchupsv3", ok=True)
        except Exception:
            _record_endpoint_call(cache, "boxscorematchupsv3", ok=False)
            df = pd.DataFrame()
    cache.set("boxscorematchupsv3", params, df.to_json())
    return df

def get_game_rotation(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": game_id}
    hit = cache.get("gamerotation", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    data: Dict[str, pd.DataFrame] = {"home": pd.DataFrame(), "away": pd.DataFrame()}
    endpoint = eps.get("gamerotation")
    if endpoint is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint.GameRotation(game_id=game_id),
                label="gamerotation",
            )
            dfs = _run_endpoint_call(lambda: resp.get_data_frames(), label="gamerotation")
            if len(dfs) >= 2:
                data = {"home": dfs[0], "away": dfs[1]}
            elif len(dfs) == 1:
                data = {"home": dfs[0], "away": pd.DataFrame()}
            _record_endpoint_call(cache, "gamerotation", ok=True)
        except Exception:
            _record_endpoint_call(cache, "gamerotation", ok=False)
            data = {"home": pd.DataFrame(), "away": pd.DataFrame()}
    cache.set("gamerotation", params, {k: v.to_json() for k, v in data.items()})
    return data

def enrich_rebound_columns(game_log: pd.DataFrame, player_id: int, cache: SQLiteCache) -> pd.DataFrame:
    df = game_log.copy()
    if "OREB" in df.columns and "DREB" in df.columns:
        return df
    if "Game_ID" not in df.columns:
        return df
    oreb_vals = []
    dreb_vals = []
    for gid in df["Game_ID"].astype(str).tolist():
        bs = get_boxscore_traditional(gid, cache)
        oreb = None
        dreb = None
        if len(bs):
            cols = {c.lower(): c for c in bs.columns}
            if "player_id" in cols:
                sub = bs[bs[cols["player_id"]].astype(str) == str(player_id)]
            elif "personid" in cols:
                sub = bs[bs[cols["personid"]].astype(str) == str(player_id)]
            else:
                sub = pd.DataFrame()
            if len(sub):
                r = sub.iloc[0]
                # v3 camelcase or v2 uppercase
                for cand in ["reboundsOffensive", "OREB", "oreb"]:
                    if cand in bs.columns:
                        oreb = r[cand]
                        break
                for cand in ["reboundsDefensive", "DREB", "dreb"]:
                    if cand in bs.columns:
                        dreb = r[cand]
                        break
        oreb_vals.append(float(oreb) if oreb is not None else None)
        dreb_vals.append(float(dreb) if dreb is not None else None)
    if "OREB" not in df.columns:
        df["OREB"] = oreb_vals
    if "DREB" not in df.columns:
        df["DREB"] = dreb_vals
    return df

def get_player_zone_splits(player_id: int, season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"player_id": player_id, "season": season}
    hit = cache.get("player_zone_splits", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["playerdashboardbyshootingsplits"].PlayerDashboardByShootingSplits(
            player_id=player_id, season=season, per_mode_detailed="PerGame"
        ).get_data_frames(),
        label="playerdashboardbyshootingsplits",
    )
    zone_df = dfs[3] if len(dfs) > 3 else pd.DataFrame()
    cache.set("player_zone_splits", params, zone_df.to_json())
    return zone_df

def get_player_pt_shots(player_id: int, team_id: int, season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"player_id": player_id, "team_id": team_id, "season": season}
    hit = cache.get("player_pt_shots", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    eps = _import_endpoints()
    _sleep()
    dfs = _run_endpoint_call(
        lambda: eps["playerdashptshots"].PlayerDashPtShots(
            player_id=player_id, team_id=team_id, season=season, per_mode_simple="PerGame"
        ).get_data_frames(),
        label="playerdashptshots",
    )
    shot_type_df = dfs[1] if len(dfs) > 1 else pd.DataFrame()
    cache.set("player_pt_shots", params, shot_type_df.to_json())
    return shot_type_df

def get_opponent_shot_locations(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season, "measure": "Opponent", "dist": "By Zone"}
    hit = cache.get("leaguedashteamshotlocations_opp", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint = eps.get("leaguedashteamshotlocations")
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.LeagueDashTeamShotLocations(
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame",
                    measure_type_simple="Opponent",
                    distance_range="By Zone",
                ).get_data_frames(),
                label="leaguedashteamshotlocations_opp",
            )
            df = dfs[0] if len(dfs) else pd.DataFrame()
            _record_endpoint_call(cache, "leaguedashteamshotlocations", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "leaguedashteamshotlocations", ok=False, error=exc)
            df = pd.DataFrame()
    if len(df) == 0:
        _sleep()
        nba_http = eps["NBAStatsHTTP"]()
        resp = _run_endpoint_call(
            lambda: nba_http.send_api_request(
                endpoint="leaguedashteamshotlocations",
                parameters={
                    "LeagueID": "00",
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "PerMode": "PerGame",
                    "MeasureType": "Opponent",
                    "DistanceRange": "By Zone",
                    "PaceAdjust": "N",
                    "PlusMinus": "N",
                    "Rank": "N",
                    "TeamID": 0,
                    "OpponentTeamID": 0,
                    "LastNGames": 0,
                    "Location": "",
                    "Outcome": "",
                    "Month": 0,
                    "SeasonSegment": "",
                    "DateFrom": "",
                    "DateTo": "",
                    "VsConference": "",
                    "VsDivision": "",
                    "Conference": "",
                    "Division": "",
                    "GameScope": "",
                    "GameSegment": "",
                    "Period": 0,
                    "ShotClockRange": "",
                    "ShotDistRange": "",
                    "StarterBench": "",
                    "PlayerExperience": "",
                    "PlayerPosition": "",
                    "TouchTimeRange": "",
                    "CloseDefDistRange": "",
                    "DribbleRange": "",
                },
                timeout=max(5, int(_endpoint_timeout_seconds() or 20)),
            ),
            label="leaguedashteamshotlocations_http",
        )
        df = _safe_http_frame(resp)
    cache.set("leaguedashteamshotlocations_opp", params, df.to_json())
    return df

def get_opponent_ptshot_defense(season: str, general_range: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season, "general_range": general_range}
    hit = cache.get("leaguedashoppptshot", params)
    if hit:
        return _read_cached_frame(hit.data_json)
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    eps = _import_endpoints()
    df = pd.DataFrame()
    endpoint = eps.get("leaguedashoppptshot")
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.LeagueDashOppPtShot(
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_simple="PerGame",
                    general_range_nullable=str(general_range),
                    team_id_nullable="0",
                    opponent_team_id_nullable="0",
                ).get_data_frames(),
                label="leaguedashoppptshot",
            )
            df = dfs[0] if len(dfs) else pd.DataFrame()
            _record_endpoint_call(cache, "leaguedashoppptshot", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "leaguedashoppptshot", ok=False, error=exc)
            df = pd.DataFrame()
    if len(df) == 0:
        _sleep()
        nba_http = eps["NBAStatsHTTP"]()
        resp = _run_endpoint_call(
            lambda: nba_http.send_api_request(
                endpoint="leaguedashoppptshot",
                parameters={
                    "LeagueID": "00",
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "PerMode": "PerGame",
                    "GeneralRange": general_range,
                    "CloseDefDistRange": "",
                    "Conference": "",
                    "DateFrom": "",
                    "DateTo": "",
                    "Division": "",
                    "DribbleRange": "",
                    "GameSegment": "",
                    "LastNGames": 0,
                    "Location": "",
                    "Month": 0,
                    "OpponentTeamID": 0,
                    "Outcome": "",
                    "PORound": 0,
                    "Period": 0,
                    "SeasonSegment": "",
                    "ShotClockRange": "",
                    "ShotDistRange": "",
                    "TeamID": 0,
                    "TouchTimeRange": "",
                    "VsConference": "",
                    "VsDivision": "",
                },
                timeout=max(5, int(_endpoint_timeout_seconds() or 20)),
            ),
            label="leaguedashoppptshot_http",
        )
        df = _safe_http_frame(resp)
    cache.set("leaguedashoppptshot", params, df.to_json())
    return df

def get_team_onoff(team_id: int, season: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"team_id": team_id, "season": season}
    hit = cache.get("teamplayeronoffdetails", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    if _cache_only_endpoints_enabled():
        return {"overall": pd.DataFrame(), "on": pd.DataFrame(), "off": pd.DataFrame()}
    eps = _import_endpoints()
    data = {"overall": pd.DataFrame(), "on": pd.DataFrame(), "off": pd.DataFrame()}
    try:
        _sleep()
        dfs = _run_endpoint_call(
            lambda: eps["teamplayeronoffdetails"].TeamPlayerOnOffDetails(
                team_id=team_id,
                season=season,
                per_mode_detailed="PerGame",
            ).get_data_frames(),
            label="teamplayeronoffdetails",
        )
        if len(dfs):
            data["overall"] = dfs[0]
        if len(dfs) > 1:
            data["on"] = dfs[1]
        if len(dfs) > 2:
            data["off"] = dfs[2]
        _record_endpoint_call(cache, "teamplayeronoffdetails", ok=True)
    except Exception as exc:
        _record_endpoint_call(cache, "teamplayeronoffdetails", ok=False, error=exc)
        print(f"[nba_data] warning: teamplayeronoffdetails fetch failed for team_id={team_id}: {exc}")
    cache.set("teamplayeronoffdetails", params, {k: v.to_json() for k, v in data.items()})
    return data


def _cached_frame_fetch(
    cache: SQLiteCache,
    cache_key: str,
    params: Dict[str, object],
    fetch_fn,
) -> pd.DataFrame:
    hit = cache.get(cache_key, params)
    if hit:
        return _read_cached_frame(hit.data_json)
    if _cache_only_endpoints_enabled():
        return pd.DataFrame()
    try:
        df = _run_endpoint_call(fetch_fn, label=cache_key)
        _record_endpoint_call(cache, cache_key, ok=True)
    except Exception as exc:
        print(f"[nba_data] warning: {cache_key} fetch failed params={params}: {exc}")
        _record_endpoint_call(cache, cache_key, ok=False, error=exc)
        df = pd.DataFrame()
    cache.set(cache_key, params, df.to_json())
    return df


def get_team_dash_lineups(
    team_id: int,
    season: str,
    cache: SQLiteCache,
    *,
    group_quantity: int = 5,
    last_n_games: int = 25,
) -> pd.DataFrame:
    params = {
        "team_id": int(team_id),
        "season": season,
        "group_quantity": int(group_quantity),
        "last_n_games": int(last_n_games),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("teamdashlineups")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.TeamDashLineups(
            team_id=team_id,
            season=season,
            group_quantity=str(group_quantity),
            last_n_games=str(last_n_games),
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "teamdashlineups", params, _fetch)


def get_league_dash_lineups(
    season: str,
    cache: SQLiteCache,
    *,
    group_quantity: int = 5,
    last_n_games: int = 25,
) -> pd.DataFrame:
    params = {
        "season": season,
        "group_quantity": int(group_quantity),
        "last_n_games": int(last_n_games),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashlineups")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashLineups(
            season=season,
            group_quantity=str(group_quantity),
            last_n_games=str(last_n_games),
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashlineups", params, _fetch)


def get_league_season_matchups(
    season: str,
    cache: SQLiteCache,
    *,
    off_player_id: Optional[int] = None,
    def_player_id: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "season": season,
        "off_player_id": int(off_player_id) if off_player_id is not None else "",
        "def_player_id": int(def_player_id) if def_player_id is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leagueseasonmatchups")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueSeasonMatchups(
            season=season,
            season_type_playoffs="Regular Season",
            per_mode_simple="PerGame",
            off_player_id_nullable=str(params["off_player_id"]) if params["off_player_id"] != "" else "",
            def_player_id_nullable=str(params["def_player_id"]) if params["def_player_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leagueseasonmatchups", params, _fetch)


def get_matchups_rollup(
    season: str,
    cache: SQLiteCache,
    *,
    off_player_id: Optional[int] = None,
    def_player_id: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "season": season,
        "off_player_id": int(off_player_id) if off_player_id is not None else "",
        "def_player_id": int(def_player_id) if def_player_id is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("matchupsrollup")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.MatchupsRollup(
            season=season,
            season_type_playoffs="Regular Season",
            per_mode_simple="PerGame",
            off_player_id_nullable=str(params["off_player_id"]) if params["off_player_id"] != "" else "",
            def_player_id_nullable=str(params["def_player_id"]) if params["def_player_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "matchupsrollup", params, _fetch)


def get_league_dash_pt_defend(
    season: str,
    cache: SQLiteCache,
    *,
    defense_category: str = "Overall",
) -> pd.DataFrame:
    params = {"season": season, "defense_category": str(defense_category)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashptdefend")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPtDefend(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            defense_category=str(defense_category),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashptdefend", params, _fetch)


def get_player_dash_pt_shot_defend(
    team_id: int,
    player_id: int,
    season: str,
    cache: SQLiteCache,
) -> pd.DataFrame:
    params = {"team_id": int(team_id), "player_id": int(player_id), "season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("playerdashptshotdefend")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.PlayerDashPtShotDefend(
            team_id=int(team_id),
            player_id=int(player_id),
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "playerdashptshotdefend", params, _fetch)


def get_player_dash_pt_pass(
    team_id: int,
    player_id: int,
    season: str,
    cache: SQLiteCache,
) -> pd.DataFrame:
    params = {"team_id": int(team_id), "player_id": int(player_id), "season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("playerdashptpass")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.PlayerDashPtPass(
            team_id=int(team_id),
            player_id=int(player_id),
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "playerdashptpass", params, _fetch)


def get_player_dash_pt_reb(
    team_id: int,
    player_id: int,
    season: str,
    cache: SQLiteCache,
) -> pd.DataFrame:
    params = {"team_id": int(team_id), "player_id": int(player_id), "season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("playerdashptreb")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.PlayerDashPtReb(
            team_id=int(team_id),
            player_id=int(player_id),
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "playerdashptreb", params, _fetch)


def get_league_hustle_team_stats(
    season: str,
    cache: SQLiteCache,
    *,
    per_mode_time: str = "PerGame",
) -> pd.DataFrame:
    params = {"season": season, "per_mode_time": str(per_mode_time)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguehustlestatsteam")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueHustleStatsTeam(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_time=str(per_mode_time),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguehustlestatsteam", params, _fetch)


def get_league_hustle_player_stats(
    season: str,
    cache: SQLiteCache,
    *,
    per_mode_time: str = "PerGame",
) -> pd.DataFrame:
    params = {"season": season, "per_mode_time": str(per_mode_time)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguehustlestatsplayer")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueHustleStatsPlayer(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_time=str(per_mode_time),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguehustlestatsplayer", params, _fetch)


def get_boxscore_hustle(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscorehustlev2", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("boxscorehustlev2")
    data = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.BoxScoreHustleV2(game_id=str(game_id)).get_data_frames(),
                label="boxscorehustlev2",
            )
            if len(dfs):
                data["player"] = dfs[0]
            if len(dfs) > 1:
                data["team"] = dfs[1]
            _record_endpoint_call(cache, "boxscorehustlev2", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscorehustlev2", ok=False, error=exc)
            print(f"[nba_data] warning: boxscorehustlev2 fetch failed for game_id={game_id}: {exc}")
    cache.set("boxscorehustlev2", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_team_player_onoff_summary(team_id: int, season: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"team_id": int(team_id), "season": season}
    hit = cache.get("teamplayeronoffsummary", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("teamplayeronoffsummary")
    data = {
        "overall": pd.DataFrame(),
        "off": pd.DataFrame(),
        "on": pd.DataFrame(),
    }
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.TeamPlayerOnOffSummary(
                    team_id=int(team_id),
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame",
                ).get_data_frames(),
                label="teamplayeronoffsummary",
            )
            if len(dfs):
                data["overall"] = dfs[0]
            if len(dfs) > 1:
                data["off"] = dfs[1]
            if len(dfs) > 2:
                data["on"] = dfs[2]
            _record_endpoint_call(cache, "teamplayeronoffsummary", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "teamplayeronoffsummary", ok=False, error=exc)
            print(f"[nba_data] warning: teamplayeronoffsummary fetch failed for team_id={team_id}: {exc}")
    cache.set("teamplayeronoffsummary", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_team_dash_pt_shots(
    team_id: int,
    season: str,
    cache: SQLiteCache,
    *,
    last_n_games: int = 25,
) -> Dict[str, pd.DataFrame]:
    params = {"team_id": int(team_id), "season": season, "last_n_games": int(last_n_games)}
    hit = cache.get("teamdashptshots", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("teamdashptshots")
    names = [
        "closest_def_10ft",
        "closest_def",
        "dribble",
        "general",
        "shot_clock",
        "touch_time",
    ]
    data: Dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in names}
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.TeamDashPtShots(
                    team_id=int(team_id),
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_simple="PerGame",
                    last_n_games=str(int(last_n_games)),
                ).get_data_frames(),
                label="teamdashptshots",
            )
            for i, df in enumerate(dfs[: len(names)]):
                data[names[i]] = df
            _record_endpoint_call(cache, "teamdashptshots", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "teamdashptshots", ok=False, error=exc)
            print(f"[nba_data] warning: teamdashptshots fetch failed for team_id={team_id}: {exc}")
    cache.set("teamdashptshots", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_league_dash_player_pt_shot(
    season: str,
    cache: SQLiteCache,
    *,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None,
    general_range: str = "",
) -> pd.DataFrame:
    params = {
        "season": season,
        "player_id": int(player_id) if player_id is not None else "",
        "team_id": int(team_id) if team_id is not None else "",
        "general_range": str(general_range),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashplayerptshot")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPlayerPtShot(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
            general_range_nullable=str(general_range),
        ).get_data_frames()
        df = dfs[0] if len(dfs) else pd.DataFrame()
        if len(df) and params["player_id"] != "":
            pid_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else ("playerId" if "playerId" in df.columns else None)
            if pid_col:
                df = df[df[pid_col].astype(str) == str(params["player_id"])]
        return df

    return _cached_frame_fetch(cache, "leaguedashplayerptshot", params, _fetch)


def get_league_dash_team_pt_shot(
    season: str,
    cache: SQLiteCache,
    *,
    team_id: Optional[int] = None,
    general_range: str = "",
) -> pd.DataFrame:
    params = {
        "season": season,
        "team_id": int(team_id) if team_id is not None else "",
        "general_range": str(general_range),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashteamptshot")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashTeamPtShot(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
            general_range_nullable=str(general_range),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashteamptshot", params, _fetch)


def get_league_dash_player_shot_locations(
    season: str,
    cache: SQLiteCache,
    *,
    team_id: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "season": season,
        "team_id": int(team_id) if team_id is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashplayershotlocations")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPlayerShotLocations(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            distance_range="By Zone",
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashplayershotlocations", params, _fetch)


def get_league_dash_pt_team_defend(
    season: str,
    cache: SQLiteCache,
    *,
    defense_category: str = "Overall",
) -> pd.DataFrame:
    params = {"season": season, "defense_category": str(defense_category)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashptteamdefend")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPtTeamDefend(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            defense_category=str(defense_category),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashptteamdefend", params, _fetch)


def get_boxscore_advanced_v3(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscoreadvancedv3", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    endpoint_v3 = eps.get("boxscoreadvancedv3")
    data = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScoreAdvancedV3(game_id=str(game_id)),
                label="boxscoreadvancedv3",
            )
            data = _extract_v3_player_team_frames(resp, ["boxScoreAdvanced", "boxScoreAdvancedV3"])
            _record_endpoint_call(cache, "boxscoreadvancedv3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoreadvancedv3", ok=False, error=exc)
            print(f"[nba_data] warning: boxscoreadvancedv3 fetch failed for game_id={game_id}: {exc}")
    if len(data["player"]) == 0 and len(data["team"]) == 0:
        endpoint_v2 = eps.get("boxscoreadvancedv2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScoreAdvancedV2(game_id=str(game_id)).get_data_frames(),
                    label="boxscoreadvancedv2",
                )
                if len(dfs):
                    data["player"] = dfs[0]
                if len(dfs) > 1:
                    data["team"] = dfs[1]
                _record_endpoint_call(cache, "boxscoreadvancedv2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscoreadvancedv2", ok=False, error=exc)
    cache.set("boxscoreadvancedv3", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_boxscore_four_factors_v3(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscorefourfactorsv3", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    endpoint_v3 = eps.get("boxscorefourfactorsv3")
    data = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScoreFourFactorsV3(game_id=str(game_id)),
                label="boxscorefourfactorsv3",
            )
            data = _extract_v3_player_team_frames(resp, ["boxScoreFourFactors", "boxScoreFourFactorsV3"])
            _record_endpoint_call(cache, "boxscorefourfactorsv3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscorefourfactorsv3", ok=False, error=exc)
            print(f"[nba_data] warning: boxscorefourfactorsv3 fetch failed for game_id={game_id}: {exc}")
    if len(data["player"]) == 0 and len(data["team"]) == 0:
        endpoint_v2 = eps.get("boxscorefourfactorsv2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScoreFourFactorsV2(game_id=str(game_id)).get_data_frames(),
                    label="boxscorefourfactorsv2",
                )
                if len(dfs):
                    data["player"] = dfs[0]
                if len(dfs) > 1:
                    data["team"] = dfs[1]
                _record_endpoint_call(cache, "boxscorefourfactorsv2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscorefourfactorsv2", ok=False, error=exc)
    cache.set("boxscorefourfactorsv3", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_boxscore_misc_v3(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscoremiscv3", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    endpoint_v3 = eps.get("boxscoremiscv3")
    data = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    if endpoint_v3 is not None:
        try:
            _sleep()
            resp = _run_endpoint_call(
                lambda: endpoint_v3.BoxScoreMiscV3(game_id=str(game_id)),
                label="boxscoremiscv3",
            )
            data = _extract_v3_player_team_frames(resp, ["boxScoreMisc", "boxScoreMiscV3"])
            _record_endpoint_call(cache, "boxscoremiscv3", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoremiscv3", ok=False, error=exc)
            print(f"[nba_data] warning: boxscoremiscv3 fetch failed for game_id={game_id}: {exc}")
    if len(data["player"]) == 0 and len(data["team"]) == 0:
        endpoint_v2 = eps.get("boxscoremiscv2")
        if endpoint_v2 is not None:
            try:
                _sleep()
                dfs = _run_endpoint_call(
                    lambda: endpoint_v2.BoxScoreMiscV2(game_id=str(game_id)).get_data_frames(),
                    label="boxscoremiscv2",
                )
                if len(dfs):
                    data["player"] = dfs[0]
                if len(dfs) > 1:
                    data["team"] = dfs[1]
                _record_endpoint_call(cache, "boxscoremiscv2", ok=True)
            except Exception as exc:
                _record_endpoint_call(cache, "boxscoremiscv2", ok=False, error=exc)
    cache.set("boxscoremiscv3", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_shot_chart_lineup_detail(
    season: str,
    cache: SQLiteCache,
    *,
    team_id: int,
    group_id: Optional[int] = None,
    last_n_games: int = 10,
    context_measure_detailed: str = "FGA",
) -> Dict[str, pd.DataFrame]:
    params = {
        "season": season,
        "team_id": int(team_id),
        "group_id": int(group_id) if group_id is not None else int(team_id),
        "last_n_games": int(last_n_games),
        "context_measure_detailed": str(context_measure_detailed),
    }
    hit = cache.get("shotchartlineupdetail", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("shotchartlineupdetail")
    data = {"shots": pd.DataFrame(), "league_avg": pd.DataFrame()}
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.ShotChartLineupDetail(
                    season=season,
                    season_type_all_star="Regular Season",
                    team_id_nullable=str(int(team_id)),
                    group_id=int(params["group_id"]),
                    context_measure_detailed=str(context_measure_detailed),
                    last_n_games_nullable=str(int(last_n_games)),
                ).get_data_frames(),
                label="shotchartlineupdetail",
            )
            if len(dfs):
                data["shots"] = dfs[0]
            if len(dfs) > 1:
                data["league_avg"] = dfs[1]
            _record_endpoint_call(cache, "shotchartlineupdetail", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "shotchartlineupdetail", ok=False, error=exc)
            print(f"[nba_data] warning: shotchartlineupdetail fetch failed for team_id={team_id}: {exc}")
    cache.set("shotchartlineupdetail", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_team_estimated_metrics(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("teamestimatedmetrics")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.TeamEstimatedMetrics(
            season=season,
            season_type="Regular Season",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "teamestimatedmetrics", params, _fetch)


def get_player_estimated_metrics(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("playerestimatedmetrics")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.PlayerEstimatedMetrics(
            season=season,
            season_type="Regular Season",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "playerestimatedmetrics", params, _fetch)


def get_synergy_playtypes(
    season: str,
    cache: SQLiteCache,
    *,
    player_or_team: str = "P",
    play_type: str = "",
) -> pd.DataFrame:
    player_or_team = str(player_or_team).strip().upper() if str(player_or_team).strip() else "P"
    params = {"season": season, "player_or_team": player_or_team, "play_type": str(play_type)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("synergyplaytypes")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.SynergyPlayTypes(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            player_or_team_abbreviation=player_or_team,
            play_type_nullable=str(play_type),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "synergyplaytypes", params, _fetch)


def get_scoreboard_v2(game_date: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_date": str(game_date)}
    hit = cache.get("scoreboardv2", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("scoreboardv2")
    data: Dict[str, pd.DataFrame] = {"game_header": pd.DataFrame(), "line_score": pd.DataFrame()}
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.ScoreboardV2(game_date=str(game_date), day_offset="0").get_data_frames(),
                label="scoreboardv2",
            )
            if len(dfs):
                data["game_header"] = dfs[0]
            if len(dfs) > 1:
                data["line_score"] = dfs[1]
            _record_endpoint_call(cache, "scoreboardv2", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "scoreboardv2", ok=False, error=exc)
            print(f"[nba_data] warning: scoreboardv2 fetch failed for game_date={game_date}: {exc}")
            data = {"game_header": pd.DataFrame(), "line_score": pd.DataFrame()}
    cache.set("scoreboardv2", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_boxscore_summary_v2(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscoresummaryv2", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("boxscoresummaryv2")
    names = [
        "game_summary",
        "other_stats",
        "officials",
        "inactive_players",
        "game_info",
        "line_score",
        "last_meeting",
        "season_series",
        "available_video",
    ]
    data: Dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in names}
    if endpoint is not None:
        try:
            _sleep()
            dfs = _run_endpoint_call(
                lambda: endpoint.BoxScoreSummaryV2(game_id=str(game_id)).get_data_frames(),
                label="boxscoresummaryv2",
            )
            for i, df in enumerate(dfs[: len(names)]):
                data[names[i]] = df
            _record_endpoint_call(cache, "boxscoresummaryv2", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoresummaryv2", ok=False, error=exc)
            print(f"[nba_data] warning: boxscoresummaryv2 fetch failed for game_id={game_id}: {exc}")
    cache.set("boxscoresummaryv2", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_playbyplay_v3(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": str(game_id)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("playbyplayv3")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        resp = endpoint.PlayByPlayV3(game_id=str(game_id), start_period="0", end_period="0")
        dfs = resp.get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "playbyplayv3", params, _fetch)


def get_win_probability_pbp(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": str(game_id)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("winprobabilitypbp")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        resp = endpoint.WinProbabilityPBP(game_id=str(game_id), run_type="each second")
        dfs = resp.get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "winprobabilitypbp", params, _fetch)


def get_assist_tracker(
    season: str,
    cache: SQLiteCache,
    *,
    team_id: Optional[int] = None,
    per_mode_simple: str = "PerGame",
) -> pd.DataFrame:
    params = {
        "season": season,
        "team_id": int(team_id) if team_id is not None else "",
        "per_mode_simple": str(per_mode_simple),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("assisttracker")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.AssistTracker(
            season_nullable=season,
            season_type_all_star_nullable="Regular Season",
            per_mode_simple_nullable=str(per_mode_simple),
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "assisttracker", params, _fetch)


def get_boxscore_defensive_v2(game_id: str, cache: SQLiteCache) -> Dict[str, pd.DataFrame]:
    params = {"game_id": str(game_id)}
    hit = cache.get("boxscoredefensivev2", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    endpoint = eps.get("boxscoredefensivev2")
    data = {"player": pd.DataFrame(), "team": pd.DataFrame()}
    if endpoint is not None:
        try:
            _sleep()
            dfs = endpoint.BoxScoreDefensiveV2(game_id=str(game_id)).get_data_frames()
            if len(dfs):
                data["player"] = dfs[0]
            if len(dfs) > 1:
                data["team"] = dfs[1]
            _record_endpoint_call(cache, "boxscoredefensivev2", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "boxscoredefensivev2", ok=False, error=exc)
            print(f"[nba_data] warning: boxscoredefensivev2 fetch failed for game_id={game_id}: {exc}")
    cache.set("boxscoredefensivev2", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_hustle_stats_boxscore(game_id: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"game_id": str(game_id)}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("hustlestatsboxscore")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.HustleStatsBoxScore(game_id=str(game_id)).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "hustlestatsboxscore", params, _fetch)


def get_gl_alum_boxscore_similarity_score(
    person1_id: int,
    person2_id: int,
    cache: SQLiteCache,
    *,
    season_year: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "person1_id": int(person1_id),
        "person2_id": int(person2_id),
        "season_year": int(season_year) if season_year is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("glalumboxscoresimilarityscore")
        if endpoint is None:
            return pd.DataFrame()
        kwargs = {
            "person1_id": int(person1_id),
            "person2_id": int(person2_id),
        }
        if params["season_year"] != "":
            kwargs["person1_season_year"] = int(params["season_year"])
            kwargs["person2_season_year"] = int(params["season_year"])
        _sleep()
        dfs = endpoint.GLAlumBoxScoreSimilarityScore(**kwargs).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "glalumboxscoresimilarityscore", params, _fetch)


def get_league_dash_opp_pt_shot(
    season: str,
    cache: SQLiteCache,
    *,
    general_range: str = "",
    per_mode_simple: str = "PerGame",
) -> pd.DataFrame:
    params = {
        "season": season,
        "general_range": str(general_range),
        "per_mode_simple": str(per_mode_simple),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashoppptshot")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashOppPtShot(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple=str(per_mode_simple),
            general_range_nullable=str(general_range),
            team_id_nullable="0",
            opponent_team_id_nullable="0",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashoppptshot_endpoint", params, _fetch)


def get_league_dash_pt_stats(
    season: str,
    cache: SQLiteCache,
    *,
    player_or_team: str = "Team",
    pt_measure_type: str = "SpeedDistance",
    team_id: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "season": season,
        "player_or_team": str(player_or_team),
        "pt_measure_type": str(pt_measure_type),
        "team_id": int(team_id) if team_id is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashptstats")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPtStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            player_or_team=str(player_or_team),
            pt_measure_type=str(pt_measure_type),
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashptstats", params, _fetch)


def get_league_dash_team_clutch(
    season: str,
    cache: SQLiteCache,
    *,
    clutch_time: str = "Last 5 Minutes",
    ahead_behind: str = "Ahead or Behind",
    point_diff: str = "5",
) -> pd.DataFrame:
    params = {
        "season": season,
        "clutch_time": str(clutch_time),
        "ahead_behind": str(ahead_behind),
        "point_diff": str(point_diff),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashteamclutch")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashTeamClutch(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            clutch_time=str(clutch_time),
            ahead_behind=str(ahead_behind),
            point_diff=str(point_diff),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashteamclutch", params, _fetch)


def get_league_dash_player_clutch(
    season: str,
    cache: SQLiteCache,
    *,
    clutch_time: str = "Last 5 Minutes",
    ahead_behind: str = "Ahead or Behind",
    point_diff: str = "5",
) -> pd.DataFrame:
    params = {
        "season": season,
        "clutch_time": str(clutch_time),
        "ahead_behind": str(ahead_behind),
        "point_diff": str(point_diff),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashplayerclutch")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashPlayerClutch(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            clutch_time=str(clutch_time),
            ahead_behind=str(ahead_behind),
            point_diff=str(point_diff),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashplayerclutch", params, _fetch)


def get_player_dashboard_by_clutch(
    player_id: int,
    season: str,
    cache: SQLiteCache,
) -> Dict[str, pd.DataFrame]:
    params = {"player_id": int(player_id), "season": season}
    hit = cache.get("playerdashboardbyclutch", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)
    eps = _import_endpoints()
    endpoint = eps.get("playerdashboardbyclutch")
    names = [
        "overall",
        "shot_clock",
        "dribble",
        "defender_distance",
        "shot_type",
        "assist",
    ]
    data: Dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in names}
    if endpoint is not None:
        try:
            _sleep()
            dfs = endpoint.PlayerDashboardByClutch(
                player_id=int(player_id),
                season=season,
                season_type_playoffs="Regular Season",
                per_mode_detailed="PerGame",
            ).get_data_frames()
            for i, df in enumerate(dfs[: len(names)]):
                data[names[i]] = df
            _record_endpoint_call(cache, "playerdashboardbyclutch", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "playerdashboardbyclutch", ok=False, error=exc)
            print(f"[nba_data] warning: playerdashboardbyclutch fetch failed for player_id={player_id}: {exc}")
    cache.set("playerdashboardbyclutch", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_league_dash_team_shot_locations(
    season: str,
    cache: SQLiteCache,
    *,
    measure_type_simple: str = "Opponent",
    distance_range: str = "By Zone",
) -> pd.DataFrame:
    params = {
        "season": season,
        "measure_type_simple": str(measure_type_simple),
        "distance_range": str(distance_range),
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguedashteamshotlocations")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueDashTeamShotLocations(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_simple=str(measure_type_simple),
            distance_range=str(distance_range),
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguedashteamshotlocations", params, _fetch)


def get_league_lineup_viz(
    season: str,
    cache: SQLiteCache,
    *,
    minutes_min: int = 15,
    group_quantity: int = 5,
    team_id: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "season": season,
        "minutes_min": int(minutes_min),
        "group_quantity": int(group_quantity),
        "team_id": int(team_id) if team_id is not None else "",
    }

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("leaguelineupviz")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.LeagueLineupViz(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            minutes_min=str(int(minutes_min)),
            group_quantity=str(int(group_quantity)),
            team_id_nullable=str(params["team_id"]) if params["team_id"] != "" else "",
        ).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "leaguelineupviz", params, _fetch)


def get_shot_chart_detail(
    season: str,
    cache: SQLiteCache,
    *,
    team_id: int,
    player_id: int = 0,
    context_measure_simple: str = "PTS",
    last_n_games: int = 20,
) -> Dict[str, pd.DataFrame]:
    params = {
        "season": season,
        "team_id": int(team_id),
        "player_id": int(player_id),
        "context_measure_simple": str(context_measure_simple),
        "last_n_games": int(last_n_games),
    }
    hit = cache.get("shotchartdetail", params)
    if hit:
        return _read_cached_frame_map(hit.data_json)

    eps = _import_endpoints()
    endpoint = eps.get("shotchartdetail")
    data = {"shots": pd.DataFrame(), "league_avg": pd.DataFrame()}
    if endpoint is not None:
        try:
            _sleep()
            dfs = endpoint.ShotChartDetail(
                team_id=int(team_id),
                player_id=int(player_id),
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple=str(context_measure_simple),
                last_n_games=str(int(last_n_games)),
            ).get_data_frames()
            if len(dfs):
                data["shots"] = dfs[0]
            if len(dfs) > 1:
                data["league_avg"] = dfs[1]
            _record_endpoint_call(cache, "shotchartdetail", ok=True)
        except Exception as exc:
            _record_endpoint_call(cache, "shotchartdetail", ok=False, error=exc)
            print(f"[nba_data] warning: shotchartdetail fetch failed for team_id={team_id} player_id={player_id}: {exc}")
    cache.set("shotchartdetail", params, {k: v.to_json() for k, v in data.items()})
    return data


def get_shot_chart_league_wide(season: str, cache: SQLiteCache) -> pd.DataFrame:
    params = {"season": season}

    def _fetch():
        eps = _import_endpoints()
        endpoint = eps.get("shotchartleaguewide")
        if endpoint is None:
            return pd.DataFrame()
        _sleep()
        dfs = endpoint.ShotChartLeagueWide(season=season).get_data_frames()
        return dfs[0] if len(dfs) else pd.DataFrame()

    return _cached_frame_fetch(cache, "shotchartleaguewide", params, _fetch)
