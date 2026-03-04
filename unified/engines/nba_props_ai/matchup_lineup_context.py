from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import SQLiteCache
from .historical_fg3_context import get_runtime_historical_fg3_factors
from .nba_data import (
    get_assist_tracker,
    get_boxscore_advanced_v3,
    get_boxscore_defensive_v2,
    get_boxscore_four_factors_v3,
    get_boxscore_hustle,
    get_boxscore_matchups,
    get_boxscore_misc_v3,
    get_boxscore_summary_v2,
    get_game_rotation,
    get_league_dash_lineups,
    get_opponent_ptshot_defense,
    get_league_dash_player_pt_shot,
    get_league_dash_player_shot_locations,
    get_league_dash_player_clutch,
    get_league_dash_pt_stats,
    get_league_dash_pt_defend,
    get_league_dash_team_clutch,
    get_league_dash_team_shot_locations,
    get_league_dash_pt_team_defend,
    get_league_dash_team_pt_shot,
    get_league_lineup_viz,
    get_league_hustle_player_stats,
    get_league_hustle_team_stats,
    get_league_season_matchups,
    get_matchups_rollup,
    get_playbyplay_v3,
    get_player_dash_pt_pass,
    get_player_dash_pt_reb,
    get_player_dash_pt_shot_defend,
    get_player_dashboard_by_clutch,
    get_player_estimated_metrics,
    get_player_primary_position,
    get_scoreboard_v2,
    get_shot_chart_detail,
    get_shot_chart_league_wide,
    get_shot_chart_lineup_detail,
    get_synergy_playtypes,
    get_team_dash_pt_shots,
    get_team_dash_lineups,
    get_team_estimated_metrics,
    get_team_onoff,
    get_team_player_onoff_summary,
    get_team_opponent_pergame_by_position,
    get_win_probability_pbp,
)
from .utils import name_key, safe_clip

_ENDPOINT_MEMO: Dict[Tuple[Any, ...], Any] = {}


def _memoized(key: Tuple[Any, ...], builder):
    if key in _ENDPOINT_MEMO:
        return _ENDPOINT_MEMO[key]
    value = builder()
    _ENDPOINT_MEMO[key] = value
    return value


def _memo_key(cache: SQLiteCache, *parts: Any) -> Tuple[Any, ...]:
    return ("cache_path", getattr(cache, "path", ""), *parts)


def _normalize_col_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    mapping = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in mapping:
            return mapping[key]
    return None


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(x)


def _recent_weights(n: int, decay: float = 0.92) -> np.ndarray:
    w = np.array([decay ** i for i in range(max(n, 1))], dtype=float)
    s = float(w.sum())
    if s <= 0:
        return np.ones(max(n, 1), dtype=float)
    return w / s


def _blend_to_neutral(value: float, strength: float) -> float:
    s = float(safe_clip(strength, 0.0, 1.0))
    return float(1.0 + (float(value) - 1.0) * s)


def _endpoint_reliability_score(cache: SQLiteCache, endpoint_names: List[str]) -> float:
    if not endpoint_names:
        return 0.60
    scores: List[float] = []
    for name in endpoint_names:
        try:
            health = cache.get_endpoint_health(str(name))
        except Exception:
            health = None
        if health is None:
            scores.append(0.58)
            continue
        calls = int(getattr(health, "calls", 0) or 0)
        rate = float(getattr(health, "success_rate", 0.0) or 0.0)
        sample_weight = float(safe_clip(calls / 20.0, 0.15, 1.0))
        # Blend with a conservative prior to avoid overconfidence on tiny samples.
        stabilized = float(safe_clip(0.60 + (rate - 0.60) * sample_weight, 0.30, 1.0))
        scores.append(stabilized)
    return float(safe_clip(float(np.mean(np.array(scores, dtype=float))), 0.30, 1.0))


def _context_signal_coverage(out: Dict[str, Any]) -> float:
    keys = [k for k, v in out.items() if isinstance(v, (int, float)) and str(k).endswith("_factor")]
    if not keys:
        return 0.50
    active = 0
    for key in keys:
        val = float(out[key])
        if abs(val - 1.0) >= 0.015:
            active += 1
    activation = float(active / max(len(keys), 1))
    starter_prob = float(out.get("starter_prob", 0.5) or 0.5)
    rotation_certainty = float(out.get("rotation_certainty", 0.6) or 0.6)
    starter_conf = float(safe_clip(abs(starter_prob - 0.5) * 2.0, 0.0, 1.0))
    rotation_conf = float(safe_clip((rotation_certainty - 0.2) / 0.8, 0.0, 1.0))
    return float(safe_clip(0.40 + activation * 0.35 + starter_conf * 0.10 + rotation_conf * 0.15, 0.35, 1.0))


def _minutes_starter_profile(game_log: pd.DataFrame) -> Dict[str, float]:
    out = {
        "minutes_p10": 0.0,
        "minutes_p90": 0.0,
        "starter_prob": 0.5,
        "rotation_certainty": 0.6,
        "lineup_minutes_factor": 1.0,
        "lineup_minutes_var_factor": 1.0,
    }
    if len(game_log) == 0 or "MIN" not in game_log.columns:
        return out
    mins = pd.to_numeric(game_log["MIN"], errors="coerce").dropna().head(12).astype(float).values
    if len(mins) == 0:
        return out
    weights = _recent_weights(len(mins))
    mu = float(np.average(mins, weights=weights))
    stdev = float(np.sqrt(np.average((mins - mu) ** 2, weights=weights)))
    out["minutes_p10"] = float(np.percentile(mins, 10))
    out["minutes_p90"] = float(np.percentile(mins, 90))

    starter_indicator = None
    starter_col = _find_col(game_log, ["START_POSITION", "STARTER", "isStarter"])
    if starter_col:
        raw = game_log[starter_col].head(len(mins))
        if raw.dtype == object or pd.api.types.is_string_dtype(raw):
            starter_indicator = raw.astype(str).str.strip().str.upper().isin({"G", "F", "C", "1", "Y", "YES", "TRUE", "T"})
        else:
            starter_indicator = pd.to_numeric(raw, errors="coerce").fillna(0).astype(float) > 0
    if starter_indicator is None:
        starter_indicator = pd.Series(mins >= 24.0)

    starter_vals = starter_indicator.astype(float).values[: len(mins)]
    out["starter_prob"] = float(np.average(starter_vals, weights=weights))
    cv = stdev / max(mu, 1e-3)
    rotation_certainty = safe_clip(1.0 - cv * 0.85, 0.25, 1.0)
    out["rotation_certainty"] = float(rotation_certainty)
    out["lineup_minutes_factor"] = float(
        safe_clip(
            1.0 + (out["starter_prob"] - 0.5) * 0.10 + (rotation_certainty - 0.60) * 0.08,
            0.90,
            1.12,
        )
    )
    out["lineup_minutes_var_factor"] = float(safe_clip(1.18 - rotation_certainty * 0.30, 0.88, 1.24))
    return out


def _row_by_name(df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    name_col = _find_col(df, ["VS_PLAYER_NAME", "PLAYER_NAME", "PLAYER", "vsPlayerName"])
    if not name_col:
        return None
    target = name_key(player_name)
    if not target:
        return None
    names = df[name_col].astype(str)
    exact = names.map(name_key) == target
    if exact.any():
        return df[exact].iloc[0]
    partial = names.str.lower().str.contains(str(player_name).lower(), na=False, regex=False)
    if partial.any():
        return df[partial].iloc[0]
    return None


def _stat_minute_rate(row: pd.Series, stat_col: Optional[str], min_col: Optional[str]) -> Optional[float]:
    if stat_col is None or min_col is None:
        return None
    stat = _to_float(row.get(stat_col))
    mins = _to_float(row.get(min_col))
    if stat is None or mins is None or mins <= 0:
        return None
    return float(stat / mins)


def _target_absorption_factor(
    *,
    onoff_data: Dict[str, pd.DataFrame],
    target_player: str,
    absent_player: str,
    stat_candidates: List[str],
) -> float:
    on_df = onoff_data.get("on", pd.DataFrame())
    off_df = onoff_data.get("off", pd.DataFrame())
    if len(on_df) == 0 or len(off_df) == 0:
        return 1.0

    absent_on = _row_by_name(on_df, absent_player)
    absent_off = _row_by_name(off_df, absent_player)
    target_on = _row_by_name(on_df, target_player)
    if absent_on is None or absent_off is None:
        return 1.0

    stat_col = _find_col(on_df, stat_candidates)
    min_col = _find_col(on_df, ["MIN", "minutes", "Minutes"])
    on_rate = _stat_minute_rate(absent_on, stat_col, min_col)
    off_rate = _stat_minute_rate(absent_off, stat_col, min_col)
    if on_rate is None or off_rate is None or on_rate <= 0:
        return 1.0
    team_shift = safe_clip(off_rate / on_rate, 0.82, 1.20)
    if target_on is None or stat_col is None:
        return float(safe_clip(1.0 + (team_shift - 1.0) * 0.18, 0.92, 1.08))

    target_stat = _to_float(target_on.get(stat_col))
    absent_stat = _to_float(absent_on.get(stat_col))
    if target_stat is None or absent_stat is None or absent_stat <= 0:
        absorb = 0.16
    else:
        absorb = safe_clip((target_stat / absent_stat) * 1.05, 0.07, 0.35)
    return float(safe_clip(1.0 + (team_shift - 1.0) * absorb, 0.90, 1.14))


def _onoff_factors(
    cache: SQLiteCache,
    season: str,
    team_id: int,
    target_player: str,
    team_out: List[str],
) -> Dict[str, float]:
    out = {
        "onoff_usage_factor": 1.0,
        "onoff_ast_factor": 1.0,
        "onoff_reb_factor": 1.0,
        "onoff_fg3a_factor": 1.0,
        "lineup_team_pace_factor": 1.0,
    }
    if not team_out:
        return out
    try:
        data = get_team_onoff(team_id, season, cache)
    except Exception:
        return out
    if not isinstance(data, dict):
        return out

    on_df = data.get("on", pd.DataFrame())
    off_df = data.get("off", pd.DataFrame())
    pace_col = _find_col(on_df, ["PACE", "pace"])
    min_col = _find_col(on_df, ["MIN", "minutes", "Minutes"])

    for absent in team_out:
        out["onoff_usage_factor"] *= _target_absorption_factor(
            onoff_data=data,
            target_player=target_player,
            absent_player=absent,
            stat_candidates=["PTS", "points"],
        )
        out["onoff_ast_factor"] *= _target_absorption_factor(
            onoff_data=data,
            target_player=target_player,
            absent_player=absent,
            stat_candidates=["AST", "assists"],
        )
        out["onoff_reb_factor"] *= _target_absorption_factor(
            onoff_data=data,
            target_player=target_player,
            absent_player=absent,
            stat_candidates=["REB", "rebounds"],
        )
        out["onoff_fg3a_factor"] *= _target_absorption_factor(
            onoff_data=data,
            target_player=target_player,
            absent_player=absent,
            stat_candidates=["FG3A", "FG3M", "threePointersMade", "threePointersAttempted"],
        )

        absent_on = _row_by_name(on_df, absent)
        absent_off = _row_by_name(off_df, absent)
        if absent_on is not None and absent_off is not None and pace_col and min_col:
            on_rate = _stat_minute_rate(absent_on, pace_col, min_col)
            off_rate = _stat_minute_rate(absent_off, pace_col, min_col)
            if on_rate is not None and off_rate is not None and on_rate > 0:
                pace_ratio = safe_clip(off_rate / on_rate, 0.90, 1.10)
                out["lineup_team_pace_factor"] *= safe_clip(1.0 + (pace_ratio - 1.0) * 0.65, 0.94, 1.06)

    for key in list(out.keys()):
        lo, hi = (0.90, 1.16) if "pace" not in key else (0.92, 1.08)
        out[key] = float(safe_clip(out[key], lo, hi))
    return out


def _lineup_opponent_pace_factor(
    cache: SQLiteCache,
    season: str,
    opp_team_id: int,
    opponent_out: List[str],
) -> float:
    if not opponent_out:
        return 1.0
    try:
        data = get_team_onoff(opp_team_id, season, cache)
    except Exception:
        return 1.0
    on_df = data.get("on", pd.DataFrame())
    off_df = data.get("off", pd.DataFrame())
    pace_col = _find_col(on_df, ["PACE", "pace"])
    min_col = _find_col(on_df, ["MIN", "minutes", "Minutes"])
    if pace_col is None or min_col is None:
        return 1.0
    fac = 1.0
    for absent in opponent_out:
        absent_on = _row_by_name(on_df, absent)
        absent_off = _row_by_name(off_df, absent)
        if absent_on is None or absent_off is None:
            continue
        on_rate = _stat_minute_rate(absent_on, pace_col, min_col)
        off_rate = _stat_minute_rate(absent_off, pace_col, min_col)
        if on_rate is None or off_rate is None or on_rate <= 0:
            continue
        ratio = safe_clip(off_rate / on_rate, 0.90, 1.10)
        fac *= safe_clip(1.0 + (ratio - 1.0) * 0.65, 0.94, 1.06)
    return float(safe_clip(fac, 0.92, 1.08))


def _position_defense_factors(
    cache: SQLiteCache,
    season: str,
    opp_name: str,
    opp_abbr: str,
    position_bucket: str,
) -> Dict[str, float]:
    out = {"pts": 1.0, "fg3m": 1.0, "ast": 1.0, "reb": 1.0}
    if not position_bucket:
        return out
    df = get_team_opponent_pergame_by_position(season, position_bucket, cache)
    if len(df) == 0:
        return out
    team_name_col = _find_col(df, ["TEAM_NAME", "teamName"])
    team_abbr_col = _find_col(df, ["TEAM_ABBREVIATION", "teamAbbreviation"])
    row = pd.DataFrame()
    if team_name_col:
        row = df[df[team_name_col].astype(str).str.lower() == str(opp_name).lower()]
    if len(row) == 0 and team_abbr_col:
        row = df[df[team_abbr_col].astype(str).str.upper() == str(opp_abbr).upper()]
    if len(row) == 0:
        return out
    r = row.iloc[0]

    def _factor(col_candidates: List[str], shrink: float, lo: float, hi: float) -> float:
        col = _find_col(df, col_candidates)
        if not col:
            return 1.0
        lg = _to_float(df[col].mean())
        val = _to_float(r.get(col))
        if lg is None or val is None or lg <= 0:
            return 1.0
        ratio = val / lg
        return float(safe_clip(1.0 + (ratio - 1.0) * shrink, lo, hi))

    out["pts"] = _factor(["OPP_PTS", "PTS"], shrink=0.32, lo=0.90, hi=1.10)
    out["fg3m"] = _factor(["OPP_FG3M", "FG3M"], shrink=0.35, lo=0.88, hi=1.12)
    out["ast"] = _factor(["OPP_AST", "AST"], shrink=0.32, lo=0.90, hi=1.10)
    out["reb"] = _factor(["OPP_REB", "REB"], shrink=0.30, lo=0.90, hi=1.10)
    return out


def _primary_defender_factors(
    cache: SQLiteCache,
    player_id: int,
    opp_abbr: str,
    game_log: pd.DataFrame,
    league_players: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pts": 1.0,
        "fg3m": 1.0,
        "ast": 1.0,
        "reb": 1.0,
        "primary_defender_id": "",
        "primary_defender_share": 0.0,
        "defender_entropy": 0.0,
    }
    if len(game_log) == 0:
        return out
    opp_col = _find_col(game_log, ["OPP_ABBR"])
    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    if not opp_col or not gid_col:
        return out

    sample = game_log[game_log[opp_col].astype(str).str.upper() == str(opp_abbr).upper()].head(6)
    if len(sample) == 0:
        return out

    defender_weights: Dict[str, float] = {}
    defender_name_map: Dict[str, str] = {}
    for gid in sample[gid_col].astype(str).tolist()[:4]:
        if not gid:
            continue
        mdf = get_boxscore_matchups(gid, cache)
        if len(mdf) == 0:
            continue
        off_col = _find_col(mdf, ["OFF_PLAYER_ID", "offPersonId", "offPlayerId"])
        def_col = _find_col(mdf, ["DEF_PLAYER_ID", "defPersonId", "defPlayerId"])
        w_col = _find_col(mdf, ["MATCHUP_MIN", "matchupMinutes", "minutes", "partialPossessions"])
        def_name_col = _find_col(mdf, ["DEF_PLAYER_NAME", "defPlayerName"])
        if not off_col or not def_col or not w_col:
            continue
        rows = mdf[mdf[off_col].astype(str) == str(player_id)].copy()
        if len(rows) == 0:
            continue
        rows["_w"] = pd.to_numeric(rows[w_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        for _, row in rows.iterrows():
            did = str(row[def_col]).strip()
            if not did:
                continue
            defender_weights[did] = defender_weights.get(did, 0.0) + float(row["_w"])
            if def_name_col and did not in defender_name_map:
                defender_name_map[did] = str(row.get(def_name_col) or "").strip()

    if not defender_weights:
        return out

    total = float(sum(defender_weights.values()))
    if total <= 0:
        return out
    ranked = sorted(defender_weights.items(), key=lambda x: x[1], reverse=True)
    top_id, top_w = ranked[0]
    share = float(top_w / total)
    out["primary_defender_id"] = top_id
    out["primary_defender_share"] = share
    if top_id in defender_name_map and defender_name_map[top_id]:
        out["primary_defender_name"] = defender_name_map[top_id]

    if len(league_players) == 0:
        return out
    pid_col = _find_col(league_players, ["PLAYER_ID", "PERSON_ID"])
    if not pid_col:
        return out
    stl_col = _find_col(league_players, ["STL"])
    blk_col = _find_col(league_players, ["BLK"])
    reb_col = _find_col(league_players, ["REB"])
    if not stl_col and not blk_col and not reb_col:
        return out

    # Probabilistic assignment: blend top recent defenders by matchup-share.
    weighted_pts = 0.0
    weighted_fg3 = 0.0
    weighted_ast = 0.0
    weighted_reb = 0.0
    entropy = 0.0
    normalized = [(str(pid), float(w / total)) for pid, w in ranked[:5] if total > 0]
    for did, p in normalized:
        p = float(safe_clip(p, 0.0, 1.0))
        if p <= 0:
            continue
        drow = league_players[league_players[pid_col].astype(str) == str(did)]
        if len(drow) == 0:
            continue
        d = drow.iloc[0]
        stl = float(_to_float(d.get(stl_col)) or 0.0) if stl_col else 0.0
        blk = float(_to_float(d.get(blk_col)) or 0.0) if blk_col else 0.0
        reb = float(_to_float(d.get(reb_col)) or 0.0) if reb_col else 0.0
        weighted_pts += p * max((stl + blk) - 1.2, 0.0)
        weighted_fg3 += p * max(stl - 0.8, 0.0)
        weighted_ast += p * max(stl - 0.7, 0.0)
        weighted_reb += p * max(reb - 7.0, 0.0)
        entropy += (-p * np.log(max(p, 1e-9)))

    # Low entropy = sticky matchup; high entropy = frequent switches.
    # Convert to [0, 1], where 1 means concentrated/sticky.
    max_entropy = np.log(max(len(normalized), 1))
    stickiness = 1.0 if max_entropy <= 1e-6 else float(safe_clip(1.0 - (entropy / max_entropy), 0.0, 1.0))
    blend = float(safe_clip(0.55 + stickiness * 0.45, 0.55, 1.0))
    out["defender_entropy"] = float(entropy)
    out["pts"] = float(safe_clip(1.0 - blend * weighted_pts * 0.03, 0.92, 1.04))
    out["fg3m"] = float(safe_clip(1.0 - blend * weighted_fg3 * 0.03, 0.92, 1.04))
    out["ast"] = float(safe_clip(1.0 - blend * weighted_ast * 0.04, 0.90, 1.04))
    out["reb"] = float(safe_clip(1.0 - blend * weighted_reb * 0.015, 0.94, 1.04))
    return out


def _rebound_ecosystem_factor(
    team_base_df: pd.DataFrame,
    team_name: str,
    opp_name: str,
    position_bucket: str,
) -> float:
    if len(team_base_df) == 0:
        return 1.0
    name_col = _find_col(team_base_df, ["TEAM_NAME", "teamName"])
    if not name_col:
        return 1.0
    fga_col = _find_col(team_base_df, ["FGA"])
    fgm_col = _find_col(team_base_df, ["FGM"])
    fg3a_col = _find_col(team_base_df, ["FG3A"])
    if not fga_col or not fgm_col:
        return 1.0

    row_team = team_base_df[team_base_df[name_col].astype(str).str.lower() == str(team_name).lower()]
    row_opp = team_base_df[team_base_df[name_col].astype(str).str.lower() == str(opp_name).lower()]
    if len(row_team) == 0 or len(row_opp) == 0:
        return 1.0

    miss_series = (pd.to_numeric(team_base_df[fga_col], errors="coerce") - pd.to_numeric(team_base_df[fgm_col], errors="coerce")).clip(lower=0.0)
    lg_miss = _to_float(miss_series.mean())
    if lg_miss is None or lg_miss <= 0:
        return 1.0

    team_miss = _to_float(row_team.iloc[0][fga_col]) - _to_float(row_team.iloc[0][fgm_col])
    opp_miss = _to_float(row_opp.iloc[0][fga_col]) - _to_float(row_opp.iloc[0][fgm_col])
    if team_miss is None or opp_miss is None:
        return 1.0
    combined_ratio = (max(team_miss, 0.0) + max(opp_miss, 0.0)) / (2.0 * lg_miss)

    long_reb_adj = 0.0
    if fg3a_col:
        fg3a_series = pd.to_numeric(team_base_df[fg3a_col], errors="coerce")
        fga_series = pd.to_numeric(team_base_df[fga_col], errors="coerce")
        lg_3pa_rate = _to_float((fg3a_series / fga_series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).mean())
        opp_3pa_rate = _to_float(
            (pd.to_numeric(row_opp.iloc[0][fg3a_col], errors="coerce") / max(_to_float(row_opp.iloc[0][fga_col]) or 0.0, 1e-6))
        )
        if lg_3pa_rate and lg_3pa_rate > 0 and opp_3pa_rate is not None:
            ratio = opp_3pa_rate / lg_3pa_rate
            if position_bucket == "G":
                long_reb_adj = (ratio - 1.0) * 0.07
            elif position_bucket == "C":
                long_reb_adj = (ratio - 1.0) * -0.05
            else:
                long_reb_adj = (ratio - 1.0) * 0.02

    return float(safe_clip(1.0 + (combined_ratio - 1.0) * 0.30 + long_reb_adj, 0.90, 1.14))


def _fg3m_team_traditional_factor(
    team_base_df: pd.DataFrame,
    *,
    team_name: str,
    opp_name: str,
) -> float:
    if len(team_base_df) == 0:
        return 1.0
    name_col = _find_col(team_base_df, ["TEAM_NAME", "teamName"])
    fga_col = _find_col(team_base_df, ["FGA"])
    fg3a_col = _find_col(team_base_df, ["FG3A"])
    if not name_col or not fga_col or not fg3a_col:
        return 1.0

    row_team = team_base_df[team_base_df[name_col].astype(str).str.lower() == str(team_name).lower()]
    row_opp = team_base_df[team_base_df[name_col].astype(str).str.lower() == str(opp_name).lower()]
    if len(row_team) == 0 or len(row_opp) == 0:
        return 1.0

    fga_series = pd.to_numeric(team_base_df[fga_col], errors="coerce").replace(0, np.nan)
    fg3a_series = pd.to_numeric(team_base_df[fg3a_col], errors="coerce")
    lg_3pa_rate = _to_float((fg3a_series / fga_series).replace([np.inf, -np.inf], np.nan).mean())
    team_3pa_rate = _to_float(
        pd.to_numeric(row_team.iloc[0][fg3a_col], errors="coerce")
        / max(_to_float(row_team.iloc[0][fga_col]) or 0.0, 1e-6)
    )
    opp_3pa_rate = _to_float(
        pd.to_numeric(row_opp.iloc[0][fg3a_col], errors="coerce")
        / max(_to_float(row_opp.iloc[0][fga_col]) or 0.0, 1e-6)
    )
    if not lg_3pa_rate or lg_3pa_rate <= 0 or team_3pa_rate is None or opp_3pa_rate is None:
        return 1.0

    combined_ratio = ((team_3pa_rate / lg_3pa_rate) * (opp_3pa_rate / lg_3pa_rate)) ** 0.5
    return float(safe_clip(1.0 + (combined_ratio - 1.0) * 0.20, 0.90, 1.12))


def _foul_risk_factors(game_log: pd.DataFrame, opp_fta_factor: float) -> Dict[str, float]:
    out = {"foul_minutes_factor": 1.0, "foul_var_factor": 1.0, "foul_risk_score": 1.0}
    pf_col = _find_col(game_log, ["PF"])
    min_col = _find_col(game_log, ["MIN"])
    if not pf_col or not min_col or len(game_log) == 0:
        return out
    df = game_log.head(20).copy()
    pf = pd.to_numeric(df[pf_col], errors="coerce")
    mins = pd.to_numeric(df[min_col], errors="coerce")
    rate = (pf / mins.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rate) == 0:
        return out
    w = _recent_weights(len(rate))
    pf_pm = float(np.average(rate.values, weights=w))
    risk_score = (pf_pm / 0.105) * max(float(opp_fta_factor), 0.85)
    pressure = max(risk_score - 1.0, 0.0)
    out["foul_risk_score"] = float(risk_score)
    out["foul_minutes_factor"] = float(safe_clip(1.0 - pressure * 0.08, 0.92, 1.0))
    out["foul_var_factor"] = float(safe_clip(1.0 + pressure * 0.25, 1.0, 1.25))
    return out


def _blowout_risk_factors(spread: Optional[float], starter_prob: float) -> Dict[str, float]:
    out = {"blowout_minutes_factor": 1.0, "blowout_var_factor": 1.0}
    if spread is None:
        return out
    s = abs(float(spread))
    if s < 7.0:
        return out
    minutes_down = min((s - 7.0) * 0.010, 0.06)
    var_up = min((s - 7.0) * 0.030, 0.16)
    if starter_prob < 0.45:
        minutes_down *= 0.60
        var_up *= 0.70
    elif starter_prob > 0.70:
        minutes_down *= 1.10
    out["blowout_minutes_factor"] = float(safe_clip(1.0 - minutes_down, 0.92, 1.0))
    out["blowout_var_factor"] = float(safe_clip(1.0 + var_up, 1.0, 1.20))
    return out


def _row_by_team_id_or_name(df: pd.DataFrame, team_id: int, team_name: str, team_abbr: str) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    team_id_col = _find_col(df, ["TEAM_ID", "teamId", "TEAMID", "PLAYER_LAST_TEAM_ID", "D_TEAM_ID"])
    if team_id_col:
        rows = df[df[team_id_col].astype(str) == str(team_id)]
        if len(rows):
            return rows.iloc[0]
    team_name_col = _find_col(df, ["TEAM_NAME", "teamName"])
    if team_name_col:
        rows = df[df[team_name_col].astype(str).str.lower() == str(team_name).lower()]
        if len(rows):
            return rows.iloc[0]
    team_abbr_col = _find_col(df, ["TEAM_ABBREVIATION", "teamAbbreviation"])
    if team_abbr_col:
        rows = df[df[team_abbr_col].astype(str).str.upper() == str(team_abbr).upper()]
        if len(rows):
            return rows.iloc[0]
    return None


def _row_by_player_id(df: pd.DataFrame, player_id: int) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    pid_col = _find_col(df, ["PLAYER_ID", "PERSON_ID", "personId", "playerId", "PLAYER"])
    if not pid_col:
        return None
    rows = df[df[pid_col].astype(str) == str(player_id)]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def _opponent_pressure_hustle_factors(
    cache: SQLiteCache,
    season: str,
    *,
    opp_team_id: int,
    opp_name: str,
    opp_abbr: str,
    game_log: pd.DataFrame,
) -> Dict[str, float]:
    out = {
        "opp_pressure_pts_factor": 1.0,
        "opp_pressure_ast_factor": 1.0,
        "opp_pressure_fg3m_factor": 1.0,
    }
    hustle = _memoized(_memo_key(cache, "leaguehustlestatsteam", season), lambda: get_league_hustle_team_stats(season=season, cache=cache))
    if len(hustle) == 0:
        return out
    row = _row_by_team_id_or_name(hustle, opp_team_id, opp_name, opp_abbr)
    if row is None:
        return out

    g_col = _find_col(hustle, ["G", "GP"])
    cont_col = _find_col(hustle, ["CONTESTED_SHOTS", "contestedShots"])
    defl_col = _find_col(hustle, ["DEFLECTIONS", "deflections"])
    charge_col = _find_col(hustle, ["CHARGES_DRAWN", "chargesDrawn"])

    def _per_game(col: Optional[str]) -> Optional[float]:
        if not col:
            return None
        val = _to_float(row.get(col))
        if val is None:
            return None
        games = _to_float(row.get(g_col)) if g_col else None
        if games and games > 0 and val > 20:
            return float(val / games)
        return float(val)

    c = _per_game(cont_col)
    d = _per_game(defl_col)
    ch = _per_game(charge_col)
    if c is None and d is None and ch is None:
        return out

    def _league_mean(col: Optional[str]) -> Optional[float]:
        if not col:
            return None
        vals = pd.to_numeric(hustle[col], errors="coerce").dropna()
        if len(vals) == 0:
            return None
        if g_col and vals.mean() > 20:
            gvals = pd.to_numeric(hustle[g_col], errors="coerce").replace(0, np.nan).dropna()
            if len(gvals):
                return float((vals / gvals.reindex(vals.index, fill_value=np.nan)).replace([np.inf, -np.inf], np.nan).dropna().mean())
        return float(vals.mean())

    lg_c = _league_mean(cont_col)
    lg_d = _league_mean(defl_col)
    lg_ch = _league_mean(charge_col)
    ratios: List[Tuple[float, float]] = []
    if c is not None and lg_c and lg_c > 0:
        ratios.append((c / lg_c, 0.45))
    if d is not None and lg_d and lg_d > 0:
        ratios.append((d / lg_d, 0.35))
    if ch is not None and lg_ch and lg_ch > 0:
        ratios.append((ch / lg_ch, 0.20))
    if not ratios:
        return out
    total_w = float(sum(w for _, w in ratios))
    pressure_idx = float(sum(v * w for v, w in ratios) / max(total_w, 1e-6))

    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    opp_col = _find_col(game_log, ["OPP_ABBR"])
    recency_adj = 1.0
    if gid_col and opp_col and len(game_log):
        sample = game_log[game_log[opp_col].astype(str).str.upper() == str(opp_abbr).upper()].head(4)
        rec_cont: List[float] = []
        rec_defl: List[float] = []
        for gid in sample[gid_col].astype(str).tolist():
            box = _memoized(_memo_key(cache, "boxscorehustlev2", gid), lambda gid=gid: get_boxscore_hustle(gid, cache))
            tdf = box.get("team", pd.DataFrame()) if isinstance(box, dict) else pd.DataFrame()
            if len(tdf) == 0:
                continue
            trow = _row_by_team_id_or_name(tdf, opp_team_id, opp_name, opp_abbr)
            if trow is None:
                continue
            tc = _to_float(trow.get(_find_col(tdf, ["contestedShots", "CONTESTED_SHOTS"])))
            td = _to_float(trow.get(_find_col(tdf, ["deflections", "DEFLECTIONS"])))
            if tc is not None:
                rec_cont.append(float(tc))
            if td is not None:
                rec_defl.append(float(td))
        if rec_cont and c and c > 0:
            recency_adj *= float(safe_clip(1.0 + ((float(np.mean(rec_cont)) / c) - 1.0) * 0.15, 0.94, 1.08))
        if rec_defl and d and d > 0:
            recency_adj *= float(safe_clip(1.0 + ((float(np.mean(rec_defl)) / d) - 1.0) * 0.15, 0.94, 1.08))

    pressure_idx = float(safe_clip(pressure_idx * recency_adj, 0.88, 1.16))
    out["opp_pressure_pts_factor"] = float(safe_clip(1.0 + ((1.0 / pressure_idx) - 1.0) * 0.45, 0.88, 1.08))
    out["opp_pressure_ast_factor"] = float(safe_clip(1.0 + ((1.0 / pressure_idx) - 1.0) * 0.52, 0.86, 1.10))
    out["opp_pressure_fg3m_factor"] = float(safe_clip(1.0 + ((1.0 / pressure_idx) - 1.0) * 0.35, 0.88, 1.10))
    return out


def _defender_discipline_factors(
    cache: SQLiteCache,
    season: str,
    *,
    primary_defender_id: Optional[str],
    primary_defender_share: float,
    league_players: pd.DataFrame,
) -> Dict[str, float]:
    out = {
        "defender_discipline_pts_factor": 1.0,
        "defender_discipline_ast_factor": 1.0,
        "defender_discipline_fg3m_factor": 1.0,
        "defender_discipline_fta_factor": 1.0,
    }
    if not primary_defender_id:
        return out
    try:
        did = int(str(primary_defender_id))
    except Exception:
        return out

    hustle = _memoized(_memo_key(cache, "leaguehustlestatsplayer", season), lambda: get_league_hustle_player_stats(season=season, cache=cache))
    drow = _row_by_player_id(hustle, did)
    if drow is None:
        return out

    g_col = _find_col(hustle, ["G", "GP"])
    cont_col = _find_col(hustle, ["CONTESTED_SHOTS", "contestedShots"])
    defl_col = _find_col(hustle, ["DEFLECTIONS", "deflections"])
    charge_col = _find_col(hustle, ["CHARGES_DRAWN", "chargesDrawn"])

    def _pg(col: Optional[str]) -> Optional[float]:
        if not col:
            return None
        v = _to_float(drow.get(col))
        if v is None:
            return None
        g = _to_float(drow.get(g_col)) if g_col else None
        if g and g > 0 and v > 20:
            return float(v / g)
        return float(v)

    c = _pg(cont_col) or 0.0
    d = _pg(defl_col) or 0.0
    ch = _pg(charge_col) or 0.0
    lg_c = float(pd.to_numeric(hustle[cont_col], errors="coerce").mean()) if cont_col else 0.0
    lg_d = float(pd.to_numeric(hustle[defl_col], errors="coerce").mean()) if defl_col else 0.0
    lg_ch = float(pd.to_numeric(hustle[charge_col], errors="coerce").mean()) if charge_col else 0.0
    if g_col and len(hustle):
        gvals = pd.to_numeric(hustle[g_col], errors="coerce").replace(0, np.nan)
        if cont_col and lg_c > 20 and len(gvals.dropna()):
            lg_c = float((pd.to_numeric(hustle[cont_col], errors="coerce") / gvals).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)
        if defl_col and lg_d > 20 and len(gvals.dropna()):
            lg_d = float((pd.to_numeric(hustle[defl_col], errors="coerce") / gvals).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)
        if charge_col and lg_ch > 20 and len(gvals.dropna()):
            lg_ch = float((pd.to_numeric(hustle[charge_col], errors="coerce") / gvals).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)

    stock = 0.0
    if len(league_players):
        prow = _row_by_player_id(league_players, did)
        if prow is not None:
            stl_col = _find_col(league_players, ["STL"])
            blk_col = _find_col(league_players, ["BLK"])
            stl = _to_float(prow.get(stl_col)) if stl_col else None
            blk = _to_float(prow.get(blk_col)) if blk_col else None
            stock = float(max((stl or 0.0) + (blk or 0.0) - 2.2, 0.0))

    p_share = float(safe_clip(primary_defender_share, 0.0, 1.0))
    wt = float(safe_clip(p_share * 1.25, 0.0, 1.0))
    disrupt_idx = 1.0
    if lg_c > 0:
        disrupt_idx *= 1.0 + ((c / lg_c) - 1.0) * 0.35
    if lg_d > 0:
        disrupt_idx *= 1.0 + ((d / lg_d) - 1.0) * 0.35
    disrupt_idx = float(safe_clip(disrupt_idx, 0.86, 1.18))
    charge_idx = float(safe_clip((ch / lg_ch) if lg_ch > 0 else 1.0, 0.80, 1.30))

    raw_pts = float(safe_clip(1.0 + ((1.0 / disrupt_idx) - 1.0) * 0.35 - stock * 0.015, 0.88, 1.08))
    raw_ast = float(safe_clip(1.0 + ((1.0 / disrupt_idx) - 1.0) * 0.45 - stock * 0.020, 0.86, 1.10))
    raw_fg3 = float(safe_clip(1.0 + ((1.0 / disrupt_idx) - 1.0) * 0.30 - stock * 0.012, 0.88, 1.10))
    raw_fta = float(safe_clip(1.0 + ((1.0 / charge_idx) - 1.0) * 0.22 - stock * 0.010, 0.88, 1.08))

    out["defender_discipline_pts_factor"] = float(safe_clip(1.0 + (raw_pts - 1.0) * wt, 0.90, 1.08))
    out["defender_discipline_ast_factor"] = float(safe_clip(1.0 + (raw_ast - 1.0) * wt, 0.88, 1.10))
    out["defender_discipline_fg3m_factor"] = float(safe_clip(1.0 + (raw_fg3 - 1.0) * wt, 0.90, 1.10))
    out["defender_discipline_fta_factor"] = float(safe_clip(1.0 + (raw_fta - 1.0) * wt, 0.90, 1.08))
    return out


def _lineup_interaction_matrix_factors(
    cache: SQLiteCache,
    season: str,
    *,
    team_id: int,
) -> Dict[str, float]:
    out = {
        "lineup_combo_off_factor": 1.0,
        "lineup_combo_pace_factor": 1.0,
        "lineup_combo_stability_factor": 1.0,
        "lineup_combo_var_factor": 1.0,
    }
    qty_weights = {2: 0.25, 3: 0.30, 5: 0.45}
    off_blend: List[Tuple[float, float]] = []
    pace_blend: List[Tuple[float, float]] = []
    stability_blend: List[Tuple[float, float]] = []
    for qty in [2, 3, 5]:
        team_df = _memoized(
            _memo_key(cache, "teamdashlineups", season, int(team_id), qty, 25),
            lambda qty=qty: get_team_dash_lineups(team_id=team_id, season=season, cache=cache, group_quantity=qty, last_n_games=25),
        )
        league_df = _memoized(
            _memo_key(cache, "leaguedashlineups", season, qty, 25),
            lambda qty=qty: get_league_dash_lineups(season=season, cache=cache, group_quantity=qty, last_n_games=25),
        )
        if len(team_df) == 0:
            continue
        mins_col = _find_col(team_df, ["MIN", "minutes"])
        off_col = _find_col(team_df, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])
        pace_col = _find_col(team_df, ["PACE", "estimatedPace"])
        if not mins_col:
            continue
        work = team_df.copy()
        work["_w"] = pd.to_numeric(work[mins_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        work = work.sort_values("_w", ascending=False).head(12)
        total_w = float(work["_w"].sum())
        if total_w <= 0:
            continue
        top_share = float(work["_w"].max() / total_w)
        stability_blend.append((float(safe_clip(top_share, 0.20, 1.00)), qty_weights[qty]))

        if off_col and len(league_df):
            lg_off_col = _find_col(league_df, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])
            if lg_off_col:
                team_off = float((pd.to_numeric(work[off_col], errors="coerce").fillna(0.0) * work["_w"]).sum() / total_w)
                lg_off = _to_float(pd.to_numeric(league_df[lg_off_col], errors="coerce").mean())
                if lg_off and lg_off > 0:
                    off_blend.append((team_off / lg_off, qty_weights[qty]))

        if pace_col and len(league_df):
            lg_pace_col = _find_col(league_df, ["PACE", "estimatedPace"])
            if lg_pace_col:
                team_pace = float((pd.to_numeric(work[pace_col], errors="coerce").fillna(0.0) * work["_w"]).sum() / total_w)
                lg_pace = _to_float(pd.to_numeric(league_df[lg_pace_col], errors="coerce").mean())
                if lg_pace and lg_pace > 0:
                    pace_blend.append((team_pace / lg_pace, qty_weights[qty]))

    if off_blend:
        wsum = float(sum(w for _, w in off_blend))
        ratio = float(sum(v * w for v, w in off_blend) / max(wsum, 1e-6))
        out["lineup_combo_off_factor"] = float(safe_clip(1.0 + (ratio - 1.0) * 0.28, 0.90, 1.12))
    if pace_blend:
        wsum = float(sum(w for _, w in pace_blend))
        ratio = float(sum(v * w for v, w in pace_blend) / max(wsum, 1e-6))
        out["lineup_combo_pace_factor"] = float(safe_clip(1.0 + (ratio - 1.0) * 0.25, 0.92, 1.10))
    if stability_blend:
        wsum = float(sum(w for _, w in stability_blend))
        stab = float(sum(v * w for v, w in stability_blend) / max(wsum, 1e-6))
        out["lineup_combo_stability_factor"] = float(safe_clip(0.94 + stab * 0.12, 0.92, 1.06))
        out["lineup_combo_var_factor"] = float(safe_clip(1.16 - stab * 0.24, 0.90, 1.14))
    return out


def _opponent_lineup_weakness_factors(
    cache: SQLiteCache,
    season: str,
    *,
    opp_team_id: int,
    opp_name: str,
    opp_abbr: str,
) -> Dict[str, float]:
    out = {
        "opp_lineup_pts_factor": 1.0,
        "opp_lineup_fg3m_factor": 1.0,
        "opp_lineup_ast_factor": 1.0,
        "opp_lineup_reb_factor": 1.0,
    }
    league_df = _memoized(
        _memo_key(cache, "leaguedashlineups", season, 5, 20),
        lambda: get_league_dash_lineups(season=season, cache=cache, group_quantity=5, last_n_games=20),
    )
    if len(league_df) == 0:
        return out
    team_col = _find_col(league_df, ["TEAM_ID", "teamId"])
    team_name_col = _find_col(league_df, ["TEAM_NAME", "teamName"])
    team_abbr_col = _find_col(league_df, ["TEAM_ABBREVIATION", "teamAbbreviation"])
    mins_col = _find_col(league_df, ["MIN", "minutes"])
    def_col = _find_col(league_df, ["DEF_RATING", "DEFRTG", "DEFENSIVE_RATING"])
    pace_col = _find_col(league_df, ["PACE", "estimatedPace"])
    ast_col = _find_col(league_df, ["AST_PCT", "assistPercentage"])
    reb_col = _find_col(league_df, ["REB_PCT", "reboundPercentage", "DREB_PCT", "defensiveReboundPercentage"])
    fg3a_col = _find_col(league_df, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])

    rows = pd.DataFrame()
    if team_col:
        rows = league_df[league_df[team_col].astype(str) == str(opp_team_id)]
    if len(rows) == 0 and team_name_col:
        rows = league_df[league_df[team_name_col].astype(str).str.lower() == str(opp_name).lower()]
    if len(rows) == 0 and team_abbr_col:
        rows = league_df[league_df[team_abbr_col].astype(str).str.upper() == str(opp_abbr).upper()]
    if len(rows) == 0:
        return out

    if not mins_col:
        rows["_w"] = 1.0
        all_w = pd.Series(np.ones(len(league_df)))
    else:
        rows = rows.copy()
        rows["_w"] = pd.to_numeric(rows[mins_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        all_w = pd.to_numeric(league_df[mins_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    wsum = float(rows["_w"].sum())
    if wsum <= 0:
        return out

    def _weighted_mean(df: pd.DataFrame, col: Optional[str], weights: pd.Series) -> Optional[float]:
        if not col or col not in df.columns:
            return None
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna() & weights.notna()
        if not mask.any():
            return None
        vw = weights[mask].astype(float)
        total = float(vw.sum())
        if total <= 0:
            return None
        return float((vals[mask].astype(float) * vw).sum() / total)

    team_def = _weighted_mean(rows, def_col, rows["_w"])
    lg_def = _weighted_mean(league_df, def_col, all_w) if def_col else None
    if team_def and lg_def and lg_def > 0:
        out["opp_lineup_pts_factor"] = float(safe_clip(1.0 + ((team_def / lg_def) - 1.0) * 0.28, 0.90, 1.12))

    team_fg3a = _weighted_mean(rows, fg3a_col, rows["_w"]) if fg3a_col else None
    lg_fg3a = _weighted_mean(league_df, fg3a_col, all_w) if fg3a_col else None
    if team_fg3a and lg_fg3a and lg_fg3a > 0:
        out["opp_lineup_fg3m_factor"] = float(safe_clip(1.0 + ((team_fg3a / lg_fg3a) - 1.0) * 0.24, 0.90, 1.12))

    team_ast = _weighted_mean(rows, ast_col, rows["_w"]) if ast_col else None
    lg_ast = _weighted_mean(league_df, ast_col, all_w) if ast_col else None
    if team_ast and lg_ast and lg_ast > 0:
        out["opp_lineup_ast_factor"] = float(safe_clip(1.0 + ((team_ast / lg_ast) - 1.0) * 0.22, 0.90, 1.12))

    team_reb = _weighted_mean(rows, reb_col, rows["_w"]) if reb_col else None
    lg_reb = _weighted_mean(league_df, reb_col, all_w) if reb_col else None
    if team_reb and lg_reb and lg_reb > 0:
        out["opp_lineup_reb_factor"] = float(safe_clip(1.0 + ((team_reb / lg_reb) - 1.0) * 0.20, 0.90, 1.10))

    if pace_col:
        team_pace = _weighted_mean(rows, pace_col, rows["_w"])
        lg_pace = _weighted_mean(league_df, pace_col, all_w)
        if team_pace and lg_pace and lg_pace > 0:
            pace_adj = float(safe_clip(1.0 + ((team_pace / lg_pace) - 1.0) * 0.10, 0.96, 1.06))
            out["opp_lineup_pts_factor"] = float(safe_clip(out["opp_lineup_pts_factor"] * pace_adj, 0.90, 1.14))
            out["opp_lineup_fg3m_factor"] = float(safe_clip(out["opp_lineup_fg3m_factor"] * pace_adj, 0.90, 1.14))
            out["opp_lineup_ast_factor"] = float(safe_clip(out["opp_lineup_ast_factor"] * pace_adj, 0.90, 1.14))
            out["opp_lineup_reb_factor"] = float(safe_clip(out["opp_lineup_reb_factor"] * pace_adj, 0.90, 1.12))
    return out


def _lineup_vs_lineup_expectation_factors(
    cache: SQLiteCache,
    season: str,
    *,
    team_id: int,
    opp_team_id: int,
) -> Dict[str, float]:
    out = {
        "lineup_vs_lineup_pts_factor": 1.0,
        "lineup_vs_lineup_fg3m_factor": 1.0,
        "lineup_vs_lineup_ast_factor": 1.0,
        "lineup_vs_lineup_reb_factor": 1.0,
        "lineup_vs_lineup_pace_factor": 1.0,
        "lineup_vs_lineup_coverage": 0.0,
    }
    team_lineups = _memoized(
        _memo_key(cache, "teamdashlineups", season, int(team_id), "vsv", 5, 20),
        lambda: get_team_dash_lineups(team_id=team_id, season=season, cache=cache, group_quantity=5, last_n_games=20),
    )
    opp_lineups = _memoized(
        _memo_key(cache, "teamdashlineups", season, int(opp_team_id), "vsv", 5, 20),
        lambda: get_team_dash_lineups(team_id=opp_team_id, season=season, cache=cache, group_quantity=5, last_n_games=20),
    )
    league_lineups = _memoized(
        _memo_key(cache, "leaguedashlineups", season, "vsv", 5, 20),
        lambda: get_league_dash_lineups(season=season, cache=cache, group_quantity=5, last_n_games=20),
    )
    if len(team_lineups) == 0 or len(opp_lineups) == 0:
        return out
    mins_col_t = _find_col(team_lineups, ["MIN", "minutes"])
    mins_col_o = _find_col(opp_lineups, ["MIN", "minutes"])
    if not mins_col_t or not mins_col_o:
        return out

    tdf = team_lineups.copy()
    odf = opp_lineups.copy()
    tdf["_w"] = pd.to_numeric(tdf[mins_col_t], errors="coerce").fillna(0.0).clip(lower=0.0)
    odf["_w"] = pd.to_numeric(odf[mins_col_o], errors="coerce").fillna(0.0).clip(lower=0.0)
    tdf = tdf.sort_values("_w", ascending=False).head(8)
    odf = odf.sort_values("_w", ascending=False).head(8)
    tw = float(tdf["_w"].sum())
    ow = float(odf["_w"].sum())
    if tw <= 0 or ow <= 0:
        return out
    tdf["_p"] = tdf["_w"] / tw
    odf["_p"] = odf["_w"] / ow
    out["lineup_vs_lineup_coverage"] = float(
        safe_clip(min(tw / 220.0, 1.0) * min(ow / 220.0, 1.0), 0.0, 1.0)
    )

    off_col = _find_col(tdf, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])
    def_col = _find_col(odf, ["DEF_RATING", "DEFRTG", "DEFENSIVE_RATING"])
    pace_col_t = _find_col(tdf, ["PACE", "estimatedPace"])
    pace_col_o = _find_col(odf, ["PACE", "estimatedPace"])
    fg3a_col_o = _find_col(odf, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])
    ast_col_o = _find_col(odf, ["AST_PCT", "assistPercentage"])
    reb_col_o = _find_col(odf, ["REB_PCT", "reboundPercentage", "DREB_PCT", "defensiveReboundPercentage"])

    lg_off_col = _find_col(league_lineups, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])
    lg_def_col = _find_col(league_lineups, ["DEF_RATING", "DEFRTG", "DEFENSIVE_RATING"])
    lg_pace_col = _find_col(league_lineups, ["PACE", "estimatedPace"])
    lg_fg3a_col = _find_col(league_lineups, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])
    lg_ast_col = _find_col(league_lineups, ["AST_PCT", "assistPercentage"])
    lg_reb_col = _find_col(league_lineups, ["REB_PCT", "reboundPercentage", "DREB_PCT", "defensiveReboundPercentage"])

    lg_off = _to_float(pd.to_numeric(league_lineups[lg_off_col], errors="coerce").mean()) if lg_off_col else None
    lg_def = _to_float(pd.to_numeric(league_lineups[lg_def_col], errors="coerce").mean()) if lg_def_col else None
    lg_pace = _to_float(pd.to_numeric(league_lineups[lg_pace_col], errors="coerce").mean()) if lg_pace_col else None
    lg_fg3a = _to_float(pd.to_numeric(league_lineups[lg_fg3a_col], errors="coerce").mean()) if lg_fg3a_col else None
    lg_ast = _to_float(pd.to_numeric(league_lineups[lg_ast_col], errors="coerce").mean()) if lg_ast_col else None
    lg_reb = _to_float(pd.to_numeric(league_lineups[lg_reb_col], errors="coerce").mean()) if lg_reb_col else None

    pts_env = 0.0
    fg3_env = 0.0
    ast_env = 0.0
    reb_env = 0.0
    pace_env = 0.0
    total_pair_w = 0.0

    for _, tr in tdf.iterrows():
        tp = float(tr["_p"])
        t_off = _to_float(tr.get(off_col)) if off_col else None
        t_pace = _to_float(tr.get(pace_col_t)) if pace_col_t else None
        for _, orow in odf.iterrows():
            op = float(orow["_p"])
            w = tp * op
            if w <= 0:
                continue
            o_def = _to_float(orow.get(def_col)) if def_col else None
            o_pace = _to_float(orow.get(pace_col_o)) if pace_col_o else None
            o_fg3 = _to_float(orow.get(fg3a_col_o)) if fg3a_col_o else None
            o_ast = _to_float(orow.get(ast_col_o)) if ast_col_o else None
            o_reb = _to_float(orow.get(reb_col_o)) if reb_col_o else None

            off_factor = 1.0
            if t_off is not None and lg_off and lg_off > 0:
                off_factor = float(safe_clip(t_off / lg_off, 0.85, 1.15))
            def_factor = 1.0
            if o_def is not None and lg_def and lg_def > 0:
                # Lower defensive rating suppresses opponent outputs.
                def_factor = float(safe_clip(lg_def / o_def, 0.86, 1.14))
            pace_factor = 1.0
            if t_pace is not None and o_pace is not None and lg_pace and lg_pace > 0:
                pace_factor = float(safe_clip(((t_pace + o_pace) / 2.0) / lg_pace, 0.90, 1.12))
            fg3_factor = 1.0
            if o_fg3 is not None and lg_fg3a and lg_fg3a > 0:
                fg3_factor = float(safe_clip(o_fg3 / lg_fg3a, 0.88, 1.14))
            ast_factor = 1.0
            if o_ast is not None and lg_ast and lg_ast > 0:
                ast_factor = float(safe_clip(o_ast / lg_ast, 0.88, 1.14))
            reb_factor = 1.0
            if o_reb is not None and lg_reb and lg_reb > 0:
                # Higher defensive reb% tends to reduce opponent rebound chance.
                reb_factor = float(safe_clip(1.0 + ((lg_reb / o_reb) - 1.0) * 0.65, 0.88, 1.14))

            pts_env += w * off_factor * def_factor * pace_factor
            fg3_env += w * off_factor * fg3_factor * pace_factor
            ast_env += w * off_factor * ast_factor * pace_factor
            reb_env += w * reb_factor
            pace_env += w * pace_factor
            total_pair_w += w

    if total_pair_w <= 0:
        return out
    pts_env /= total_pair_w
    fg3_env /= total_pair_w
    ast_env /= total_pair_w
    reb_env /= total_pair_w
    pace_env /= total_pair_w

    out["lineup_vs_lineup_pts_factor"] = float(safe_clip(1.0 + (pts_env - 1.0) * 0.30, 0.88, 1.14))
    out["lineup_vs_lineup_fg3m_factor"] = float(safe_clip(1.0 + (fg3_env - 1.0) * 0.28, 0.88, 1.14))
    out["lineup_vs_lineup_ast_factor"] = float(safe_clip(1.0 + (ast_env - 1.0) * 0.30, 0.86, 1.15))
    out["lineup_vs_lineup_reb_factor"] = float(safe_clip(1.0 + (reb_env - 1.0) * 0.24, 0.88, 1.12))
    out["lineup_vs_lineup_pace_factor"] = float(safe_clip(1.0 + (pace_env - 1.0) * 0.25, 0.90, 1.12))
    return out


def _lineup_dash_factors(
    cache: SQLiteCache,
    season: str,
    team_id: int,
) -> Dict[str, float]:
    out = {
        "lineup_dash_pace_factor": 1.0,
        "lineup_dash_off_factor": 1.0,
        "lineup_dash_stability": 0.6,
    }
    team_df = _memoized(
        _memo_key(cache, "teamdashlineups", season, int(team_id)),
        lambda: get_team_dash_lineups(team_id=team_id, season=season, cache=cache, group_quantity=5, last_n_games=25),
    )
    league_df = _memoized(
        _memo_key(cache, "leaguedashlineups", season),
        lambda: get_league_dash_lineups(season=season, cache=cache, group_quantity=5, last_n_games=25),
    )
    if len(team_df) == 0:
        return out

    mins_col = _find_col(team_df, ["MIN", "minutes"])
    off_col = _find_col(team_df, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])
    pace_col = _find_col(team_df, ["PACE"])
    if not mins_col:
        return out
    work = team_df.copy()
    work["_w"] = pd.to_numeric(work[mins_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.sort_values("_w", ascending=False).head(8)
    total_w = float(work["_w"].sum())
    if total_w <= 0:
        return out
    max_share = float(work["_w"].max() / total_w)
    out["lineup_dash_stability"] = float(safe_clip(max_share, 0.2, 1.0))

    if off_col:
        team_off = float((pd.to_numeric(work[off_col], errors="coerce").fillna(0.0) * work["_w"]).sum() / total_w)
        lg_off = _to_float(pd.to_numeric(league_df[_find_col(league_df, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"])], errors="coerce").mean()) if len(league_df) and _find_col(league_df, ["OFF_RATING", "OFFRTG", "OFFENSIVE_RATING"]) else None
        if lg_off and lg_off > 0:
            out["lineup_dash_off_factor"] = float(safe_clip(1.0 + ((team_off / lg_off) - 1.0) * 0.30, 0.90, 1.12))

    if pace_col:
        team_pace = float((pd.to_numeric(work[pace_col], errors="coerce").fillna(0.0) * work["_w"]).sum() / total_w)
        lg_pace_col = _find_col(league_df, ["PACE"])
        lg_pace = _to_float(pd.to_numeric(league_df[lg_pace_col], errors="coerce").mean()) if len(league_df) and lg_pace_col else None
        if lg_pace and lg_pace > 0:
            out["lineup_dash_pace_factor"] = float(safe_clip(1.0 + ((team_pace / lg_pace) - 1.0) * 0.30, 0.92, 1.10))
    return out


def _advanced_metric_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
    team_id: int,
    opp_team_id: int,
    team_name: str,
    opp_name: str,
    opp_abbr: str,
) -> Dict[str, float]:
    out = {
        "team_est_pace_factor": 1.0,
        "opp_est_def_factor": 1.0,
        "player_est_impact_factor": 1.0,
    }
    team_est = _memoized(_memo_key(cache, "teamestimatedmetrics", season), lambda: get_team_estimated_metrics(season=season, cache=cache))
    player_est = _memoized(_memo_key(cache, "playerestimatedmetrics", season), lambda: get_player_estimated_metrics(season=season, cache=cache))

    if len(team_est):
        team_row = _row_by_team_id_or_name(team_est, team_id, team_name, "")
        opp_row = _row_by_team_id_or_name(team_est, opp_team_id, opp_name, opp_abbr)
        pace_col = _find_col(team_est, ["E_PACE", "PACE"])
        def_col = _find_col(team_est, ["E_DEF_RATING", "DEF_RATING", "DEFRTG"])
        if team_row is not None and opp_row is not None and pace_col:
            team_pace = _to_float(team_row.get(pace_col))
            opp_pace = _to_float(opp_row.get(pace_col))
            lg_pace = _to_float(pd.to_numeric(team_est[pace_col], errors="coerce").mean())
            if team_pace and opp_pace and lg_pace and lg_pace > 0:
                avg_pace = (team_pace + opp_pace) / 2.0
                out["team_est_pace_factor"] = float(safe_clip(1.0 + ((avg_pace / lg_pace) - 1.0) * 0.30, 0.92, 1.10))
        if opp_row is not None and def_col:
            opp_def = _to_float(opp_row.get(def_col))
            lg_def = _to_float(pd.to_numeric(team_est[def_col], errors="coerce").mean())
            if opp_def and lg_def and lg_def > 0:
                # Lower defensive rating means tougher defense.
                rel = lg_def / opp_def
                out["opp_est_def_factor"] = float(safe_clip(1.0 + (rel - 1.0) * 0.25, 0.90, 1.10))

    if len(player_est):
        pid_col = _find_col(player_est, ["PLAYER_ID", "PERSON_ID", "playerId"])
        off_col = _find_col(player_est, ["E_OFF_RATING", "OFF_RATING", "OFFRTG"])
        if pid_col and off_col:
            row = player_est[player_est[pid_col].astype(str) == str(player_id)]
            if len(row):
                player_off = _to_float(row.iloc[0][off_col])
                lg_off = _to_float(pd.to_numeric(player_est[off_col], errors="coerce").mean())
                if player_off and lg_off and lg_off > 0:
                    out["player_est_impact_factor"] = float(safe_clip(1.0 + ((player_off / lg_off) - 1.0) * 0.25, 0.90, 1.12))
    return out


def _season_matchup_factors(
    cache: SQLiteCache,
    season: str,
    *,
    off_player_id: int,
    opp_team_id: int,
) -> Dict[str, float]:
    out = {"season_matchup_factor": 1.0, "season_matchup_defender_share": 0.0}
    season_matchups = _memoized(
        _memo_key(cache, "leagueseasonmatchups", season, int(off_player_id)),
        lambda: get_league_season_matchups(season=season, cache=cache, off_player_id=off_player_id),
    )
    rollup = _memoized(
        _memo_key(cache, "matchupsrollup", season, int(off_player_id)),
        lambda: get_matchups_rollup(season=season, cache=cache, off_player_id=off_player_id),
    )
    for df in [rollup, season_matchups]:
        if len(df) == 0:
            continue
        def_team_col = _find_col(df, ["DEF_TEAM_ID", "defTeamId", "D_TEAM_ID", "TEAM_ID"])
        if def_team_col:
            df = df[df[def_team_col].astype(str) == str(opp_team_id)]
        if len(df) == 0:
            continue
        poss_col = _find_col(df, ["POSS", "possessions", "MATCHUP_MIN", "matchupMinutes"])
        pts_col = _find_col(df, ["PTS", "POINTS", "PLAYER_PTS", "FGM"])
        def_col = _find_col(df, ["DEF_PLAYER_ID", "defPlayerId"])
        if poss_col and pts_col:
            poss = pd.to_numeric(df[poss_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            pts = pd.to_numeric(df[pts_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            tot_poss = float(poss.sum())
            if tot_poss > 0:
                eff = float(pts.sum() / tot_poss)
                # Relative to rough baseline scoring efficiency per possession.
                out["season_matchup_factor"] = float(safe_clip(1.0 + ((eff / 0.52) - 1.0) * 0.18, 0.90, 1.10))
        if def_col and poss_col:
            tmp = df.copy()
            tmp["_w"] = pd.to_numeric(tmp[poss_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            wsum = float(tmp["_w"].sum())
            if wsum > 0:
                by_def = tmp.groupby(def_col, dropna=False)["_w"].sum()
                out["season_matchup_defender_share"] = float(safe_clip(float(by_def.max() / wsum), 0.0, 1.0))
        break
    return out


def _pt_defense_endpoint_factors(
    cache: SQLiteCache,
    season: str,
    *,
    opp_team_id: int,
    primary_defender_id: Optional[str],
) -> Dict[str, float]:
    out = {"pt_defend_pts_factor": 1.0, "pt_defend_fg3m_factor": 1.0}
    league_ptd = _memoized(
        _memo_key(cache, "leaguedashptdefend", season, "Overall"),
        lambda: get_league_dash_pt_defend(season=season, cache=cache, defense_category="Overall"),
    )
    if len(league_ptd):
        team_col = _find_col(league_ptd, ["TEAM_ID", "teamId", "PLAYER_LAST_TEAM_ID", "D_TEAM_ID"])
        pct_col = _find_col(league_ptd, ["D_FG_PCT", "defendedFgPct", "FG_PCT", "OPP_FG_PCT"])
        freq_col = _find_col(league_ptd, ["FREQ", "frequency", "PCT_PLUSMINUS"])
        if team_col and pct_col:
            rows = league_ptd[league_ptd[team_col].astype(str) == str(opp_team_id)]
            if len(rows):
                pct = _to_float(pd.to_numeric(rows[pct_col], errors="coerce").mean())
                lg = _to_float(pd.to_numeric(league_ptd[pct_col], errors="coerce").mean())
                if pct and lg and lg > 0:
                    suppress = lg / pct
                    out["pt_defend_pts_factor"] = float(safe_clip(1.0 + (suppress - 1.0) * 0.22, 0.90, 1.10))
                if freq_col:
                    freq = _to_float(pd.to_numeric(rows[freq_col], errors="coerce").mean())
                    lgf = _to_float(pd.to_numeric(league_ptd[freq_col], errors="coerce").mean())
                    if freq and lgf and lgf > 0:
                        out["pt_defend_fg3m_factor"] = float(safe_clip(1.0 + ((freq / lgf) - 1.0) * 0.14, 0.92, 1.10))

    if primary_defender_id:
        try:
            primary_id = int(str(primary_defender_id))
        except Exception:
            primary_id = None
        if primary_id is not None:
            ptd = _memoized(
                _memo_key(cache, "playerdashptshotdefend", season, int(opp_team_id), int(primary_id)),
                lambda: get_player_dash_pt_shot_defend(team_id=opp_team_id, player_id=primary_id, season=season, cache=cache),
            )
            if len(ptd):
                pct_col = _find_col(ptd, ["D_FG_PCT", "defendedFgPct", "FG_PCT"])
                if pct_col:
                    val = _to_float(pd.to_numeric(ptd[pct_col], errors="coerce").mean())
                    if val is not None:
                        out["pt_defend_pts_factor"] *= float(safe_clip(1.0 + ((0.45 / max(val, 1e-6)) - 1.0) * 0.12, 0.94, 1.07))
                        out["pt_defend_fg3m_factor"] *= float(safe_clip(1.0 + ((0.35 / max(val, 1e-6)) - 1.0) * 0.10, 0.94, 1.07))
    out["pt_defend_pts_factor"] = float(safe_clip(out["pt_defend_pts_factor"], 0.88, 1.12))
    out["pt_defend_fg3m_factor"] = float(safe_clip(out["pt_defend_fg3m_factor"], 0.88, 1.12))
    return out


def _player_tracking_micro_factors(
    cache: SQLiteCache,
    season: str,
    *,
    team_id: int,
    player_id: int,
) -> Dict[str, float]:
    out = {"pt_pass_ast_factor": 1.0, "pt_reb_factor": 1.0}
    pt_pass = _memoized(
        _memo_key(cache, "playerdashptpass", season, int(team_id), int(player_id)),
        lambda: get_player_dash_pt_pass(team_id=team_id, player_id=player_id, season=season, cache=cache),
    )
    if len(pt_pass):
        ast_col = _find_col(pt_pass, ["AST", "ASSISTS", "potentialAssists"])
        if ast_col:
            ast = _to_float(pd.to_numeric(pt_pass[ast_col], errors="coerce").mean())
            if ast is not None:
                out["pt_pass_ast_factor"] = float(safe_clip(1.0 + ((ast / 6.0) - 1.0) * 0.16, 0.90, 1.14))

    pt_reb = _memoized(
        _memo_key(cache, "playerdashptreb", season, int(team_id), int(player_id)),
        lambda: get_player_dash_pt_reb(team_id=team_id, player_id=player_id, season=season, cache=cache),
    )
    if len(pt_reb):
        reb_col = _find_col(pt_reb, ["REB", "REB_CHANCES", "reboundChances", "TOTAL_REB"])
        if reb_col:
            reb = _to_float(pd.to_numeric(pt_reb[reb_col], errors="coerce").mean())
            if reb is not None:
                out["pt_reb_factor"] = float(safe_clip(1.0 + ((reb / 9.0) - 1.0) * 0.16, 0.90, 1.14))
    return out


def _extra_endpoint_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
    team_id: int,
    opp_team_id: int,
    game_log: pd.DataFrame,
    opp_abbr: str,
) -> Dict[str, float]:
    out = {
        "assist_tracker_ast_factor": 1.0,
        "pt_stats_pace_factor": 1.0,
        "team_clutch_pts_factor": 1.0,
        "team_clutch_ast_factor": 1.0,
        "lineup_viz_opp_def_factor": 1.0,
        "shot_chart_detail_fg3_factor": 1.0,
        "boxscore_defensive_opp_factor": 1.0,
        "extra_endpoint_coverage": 0.0,
    }
    signals = 0

    assist = _memoized(
        _memo_key(cache, "assisttracker", season, int(team_id)),
        lambda: get_assist_tracker(season=season, cache=cache, team_id=team_id, per_mode_simple="PerGame"),
    )
    if len(assist):
        row = _row_by_player_id(assist, player_id)
        ast_created_col = _find_col(assist, ["AST_POINTS_CREATED", "AST_PTS_CREATED", "AST"])
        if row is not None and ast_created_col:
            val = _to_float(row.get(ast_created_col))
            lg = _to_float(pd.to_numeric(assist[ast_created_col], errors="coerce").mean())
            if val is not None and lg and lg > 0:
                ratio = val / lg
                out["assist_tracker_ast_factor"] = float(
                    safe_clip(1.0 + (ratio - 1.0) * 0.16, 0.90, 1.14)
                )
                signals += 1

    pt_stats = _memoized(
        _memo_key(cache, "leaguedashptstats", season, int(team_id), "Team", "SpeedDistance"),
        lambda: get_league_dash_pt_stats(
            season=season,
            cache=cache,
            team_id=team_id,
            player_or_team="Team",
            pt_measure_type="SpeedDistance",
        ),
    )
    if len(pt_stats):
        row = _row_by_team_id_or_name(pt_stats, team_id, "", "")
        speed_col = _find_col(pt_stats, ["AVG_SPEED", "SPEED"])
        if row is not None and speed_col:
            val = _to_float(row.get(speed_col))
            lg = _to_float(pd.to_numeric(pt_stats[speed_col], errors="coerce").mean())
            if val is not None and lg and lg > 0:
                out["pt_stats_pace_factor"] = float(
                    safe_clip(1.0 + ((val / lg) - 1.0) * 0.22, 0.90, 1.10)
                )
                signals += 1

    team_clutch = _memoized(
        _memo_key(cache, "leaguedashteamclutch", season),
        lambda: get_league_dash_team_clutch(season=season, cache=cache),
    )
    player_clutch = _memoized(
        _memo_key(cache, "leaguedashplayerclutch", season),
        lambda: get_league_dash_player_clutch(season=season, cache=cache),
    )
    if len(team_clutch):
        trow = _row_by_team_id_or_name(team_clutch, team_id, "", "")
    else:
        trow = None
    if len(player_clutch):
        prow = _row_by_player_id(player_clutch, player_id)
    else:
        prow = None
    if trow is not None and prow is not None:
        pts_col = _find_col(player_clutch, ["PTS", "POINTS"])
        ast_col = _find_col(player_clutch, ["AST", "ASSISTS"])
        if pts_col:
            p_pts = _to_float(prow.get(pts_col))
            lg_pts = _to_float(pd.to_numeric(player_clutch[pts_col], errors="coerce").mean())
            if p_pts is not None and lg_pts and lg_pts > 0:
                out["team_clutch_pts_factor"] = float(
                    safe_clip(1.0 + ((p_pts / lg_pts) - 1.0) * 0.18, 0.90, 1.14)
                )
                signals += 1
        if ast_col:
            p_ast = _to_float(prow.get(ast_col))
            lg_ast = _to_float(pd.to_numeric(player_clutch[ast_col], errors="coerce").mean())
            if p_ast is not None and lg_ast and lg_ast > 0:
                out["team_clutch_ast_factor"] = float(
                    safe_clip(1.0 + ((p_ast / lg_ast) - 1.0) * 0.18, 0.88, 1.16)
                )
                signals += 1
    pdc = _memoized(
        _memo_key(cache, "playerdashboardbyclutch", season, int(player_id)),
        lambda: get_player_dashboard_by_clutch(player_id=player_id, season=season, cache=cache),
    )
    pdc_overall = pdc.get("overall", pd.DataFrame()) if isinstance(pdc, dict) else pd.DataFrame()
    if len(pdc_overall):
        pts_col = _find_col(pdc_overall, ["PTS", "POINTS"])
        ast_col = _find_col(pdc_overall, ["AST", "ASSISTS"])
        if pts_col:
            val = _to_float(pd.to_numeric(pdc_overall[pts_col], errors="coerce").mean())
            if val is not None:
                out["team_clutch_pts_factor"] = float(
                    safe_clip(out["team_clutch_pts_factor"] * (1.0 + ((val / 6.0) - 1.0) * 0.06), 0.88, 1.16)
                )
                signals += 1
        if ast_col:
            val = _to_float(pd.to_numeric(pdc_overall[ast_col], errors="coerce").mean())
            if val is not None:
                out["team_clutch_ast_factor"] = float(
                    safe_clip(out["team_clutch_ast_factor"] * (1.0 + ((val / 1.5) - 1.0) * 0.05), 0.86, 1.18)
                )
                signals += 1

    lineup_viz_opp = _memoized(
        _memo_key(cache, "leaguelineupviz", season, int(opp_team_id)),
        lambda: get_league_lineup_viz(
            season=season,
            cache=cache,
            minutes_min=10,
            group_quantity=5,
            team_id=opp_team_id,
        ),
    )
    lineup_viz_all = _memoized(
        _memo_key(cache, "leaguelineupviz", season, "all"),
        lambda: get_league_lineup_viz(
            season=season,
            cache=cache,
            minutes_min=10,
            group_quantity=5,
            team_id=None,
        ),
    )
    if len(lineup_viz_opp) and len(lineup_viz_all):
        def_col = _find_col(lineup_viz_opp, ["DEF_RATING", "DEFRTG", "DEFENSIVE_RATING"])
        if def_col:
            val = _to_float(pd.to_numeric(lineup_viz_opp[def_col], errors="coerce").mean())
            lg_col = _find_col(lineup_viz_all, ["DEF_RATING", "DEFRTG", "DEFENSIVE_RATING"])
            lg = _to_float(pd.to_numeric(lineup_viz_all[lg_col], errors="coerce").mean()) if lg_col else None
            if val is not None and lg and lg > 0:
                out["lineup_viz_opp_def_factor"] = float(
                    safe_clip(1.0 + ((lg / val) - 1.0) * 0.18, 0.90, 1.10)
                )
                signals += 1

    shot_detail = _memoized(
        _memo_key(cache, "shotchartdetail", season, int(team_id), int(player_id)),
        lambda: get_shot_chart_detail(
            season=season,
            cache=cache,
            team_id=team_id,
            player_id=player_id,
            context_measure_simple="FGA",
            last_n_games=25,
        ),
    )
    shot_lg = _memoized(
        _memo_key(cache, "shotchartleaguewide", season),
        lambda: get_shot_chart_league_wide(season=season, cache=cache),
    )
    shots = shot_detail.get("shots", pd.DataFrame()) if isinstance(shot_detail, dict) else pd.DataFrame()
    opp_zone = _memoized(
        _memo_key(cache, "leaguedashteamshotlocations", season, int(opp_team_id)),
        lambda: get_league_dash_team_shot_locations(
            season=season,
            cache=cache,
            measure_type_simple="Opponent",
            distance_range="By Zone",
        ),
    )
    if len(shots) and len(shot_lg):
        zone_col = _find_col(shots, ["SHOT_ZONE_BASIC", "shotZoneBasic"])
        att_col = _find_col(shots, ["SHOT_ATTEMPTED_FLAG", "SHOT_ATTEMPTS", "FGA"])
        lg_zone_col = _find_col(shot_lg, ["SHOT_ZONE_BASIC", "shotZoneBasic"])
        lg_att_col = _find_col(shot_lg, ["SHOT_ATTEMPTED_FLAG", "SHOT_ATTEMPTS", "FGA"])
        if zone_col and att_col and lg_zone_col and lg_att_col:
            shot_zone = shots[zone_col].astype(str).str.lower()
            lg_zone = shot_lg[lg_zone_col].astype(str).str.lower()
            p_3 = pd.to_numeric(shots.loc[shot_zone.str.contains("3"), att_col], errors="coerce").fillna(0.0).sum()
            p_t = pd.to_numeric(shots[att_col], errors="coerce").fillna(0.0).sum()
            lg_3 = pd.to_numeric(shot_lg.loc[lg_zone.str.contains("3"), lg_att_col], errors="coerce").fillna(0.0).sum()
            lg_t = pd.to_numeric(shot_lg[lg_att_col], errors="coerce").fillna(0.0).sum()
            if p_t > 0 and lg_t > 0 and lg_3 > 0:
                p_rate = float(p_3 / p_t)
                lg_rate = float(lg_3 / lg_t)
                out["shot_chart_detail_fg3_factor"] = float(
                    safe_clip(1.0 + ((p_rate / lg_rate) - 1.0) * 0.14, 0.90, 1.12)
                )
                signals += 1
    if len(opp_zone):
        fg3_col = _find_col(opp_zone, ["FG3A", "Above the Break 3 FGA", "Corner 3 FGA"])
        if fg3_col:
            row = _row_by_team_id_or_name(opp_zone, opp_team_id, "", "")
            val = _to_float(row.get(fg3_col)) if row is not None else None
            lg = _to_float(pd.to_numeric(opp_zone[fg3_col], errors="coerce").mean())
            if val is not None and lg and lg > 0:
                out["shot_chart_detail_fg3_factor"] = float(
                    safe_clip(out["shot_chart_detail_fg3_factor"] * (1.0 + ((val / lg) - 1.0) * 0.08), 0.88, 1.14)
                )
                signals += 1

    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    opp_col = _find_col(game_log, ["OPP_ABBR"])
    if gid_col and opp_col and len(game_log):
        recent_vs = game_log[game_log[opp_col].astype(str).str.upper() == str(opp_abbr).upper()].head(1)
        if len(recent_vs):
            gid = str(recent_vs.iloc[0][gid_col])
            box_def = _memoized(
                _memo_key(cache, "boxscoredefensivev2", gid),
                lambda: get_boxscore_defensive_v2(gid, cache),
            )
            tdf = box_def.get("team", pd.DataFrame()) if isinstance(box_def, dict) else pd.DataFrame()
            if len(tdf):
                row = _row_by_team_id_or_name(tdf, opp_team_id, "", "")
                def_col = _find_col(tdf, ["DEF_RATING", "DEFRTG", "defensiveRating"])
                if row is not None and def_col:
                    val = _to_float(row.get(def_col))
                    lg = _to_float(pd.to_numeric(tdf[def_col], errors="coerce").mean())
                    if val is not None and lg and lg > 0:
                        out["boxscore_defensive_opp_factor"] = float(
                            safe_clip(1.0 + ((lg / val) - 1.0) * 0.15, 0.92, 1.10)
                        )
                        signals += 1

    out["extra_endpoint_coverage"] = float(safe_clip(signals / 9.0, 0.0, 1.0))
    return out


def _game_day_availability_factors(
    cache: SQLiteCache,
    *,
    game_date_local: Optional[str],
    team_id: int,
    opp_team_id: int,
    player_id: int,
) -> Dict[str, float]:
    out = {
        "game_day_minutes_factor": 1.0,
        "game_day_usage_factor": 1.0,
        "game_day_opp_factor": 1.0,
    }
    if not str(game_date_local or "").strip():
        return out
    scoreboard = _memoized(_memo_key(cache, "scoreboardv2", str(game_date_local)), lambda: get_scoreboard_v2(str(game_date_local), cache))
    gh = scoreboard.get("game_header", pd.DataFrame()) if isinstance(scoreboard, dict) else pd.DataFrame()
    if len(gh) == 0:
        return out
    home_col = _find_col(gh, ["HOME_TEAM_ID", "homeTeamId"])
    away_col = _find_col(gh, ["VISITOR_TEAM_ID", "AWAY_TEAM_ID", "awayTeamId", "visitorTeamId"])
    gid_col = _find_col(gh, ["GAME_ID", "gameId"])
    if not home_col or not away_col or not gid_col:
        return out
    mask = (
        (gh[home_col].astype(str) == str(team_id)) & (gh[away_col].astype(str) == str(opp_team_id))
    ) | (
        (gh[home_col].astype(str) == str(opp_team_id)) & (gh[away_col].astype(str) == str(team_id))
    )
    if not mask.any():
        return out
    game_id = str(gh[mask].iloc[0][gid_col])
    if not game_id:
        return out
    summary = _memoized(_memo_key(cache, "boxscoresummaryv2", game_id), lambda: get_boxscore_summary_v2(game_id, cache))
    inactive = summary.get("inactive_players", pd.DataFrame()) if isinstance(summary, dict) else pd.DataFrame()
    if len(inactive) == 0:
        return out
    pid_col = _find_col(inactive, ["PLAYER_ID", "PERSON_ID"])
    tid_col = _find_col(inactive, ["TEAM_ID", "teamId"])
    if pid_col and (inactive[pid_col].astype(str) == str(player_id)).any():
        out["game_day_minutes_factor"] = 0.70
        out["game_day_usage_factor"] = 0.82
    if tid_col:
        team_inactive = int((inactive[tid_col].astype(str) == str(team_id)).sum())
        opp_inactive = int((inactive[tid_col].astype(str) == str(opp_team_id)).sum())
        if team_inactive >= 2:
            out["game_day_usage_factor"] *= float(safe_clip(1.0 + (team_inactive - 1) * 0.03, 1.0, 1.10))
        if opp_inactive >= 2:
            out["game_day_opp_factor"] *= float(safe_clip(1.0 + (opp_inactive - 1) * 0.03, 1.0, 1.10))
    out["game_day_minutes_factor"] = float(safe_clip(out["game_day_minutes_factor"], 0.70, 1.04))
    out["game_day_usage_factor"] = float(safe_clip(out["game_day_usage_factor"], 0.82, 1.12))
    out["game_day_opp_factor"] = float(safe_clip(out["game_day_opp_factor"], 0.92, 1.12))
    return out


def _pbp_volatility_factors(
    cache: SQLiteCache,
    game_log: pd.DataFrame,
) -> Dict[str, float]:
    out = {"pbp_blowout_minutes_factor": 1.0, "pbp_blowout_var_factor": 1.0}
    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    if not gid_col or len(game_log) == 0:
        return out
    game_ids = [str(x) for x in game_log[gid_col].head(6).tolist() if str(x).strip()]
    if not game_ids:
        return out
    blowout_scores: List[float] = []
    for gid in game_ids[:4]:
        wp = _memoized(_memo_key(cache, "winprobabilitypbp", gid), lambda gid=gid: get_win_probability_pbp(gid, cache))
        if len(wp) == 0:
            _memoized(_memo_key(cache, "playbyplayv3", gid), lambda gid=gid: get_playbyplay_v3(gid, cache))
            continue
        home_wp_col = _find_col(wp, ["HOME_PCT", "HOME_WIN_PCT", "homeWinProbability", "HOME_POSS_WIN_PCT"])
        if not home_wp_col:
            continue
        series = pd.to_numeric(wp[home_wp_col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        blow = float((series.sub(0.5).abs() >= 0.35).mean())
        blowout_scores.append(blow)
    if blowout_scores:
        b = float(np.mean(blowout_scores))
        out["pbp_blowout_minutes_factor"] = float(safe_clip(1.0 - b * 0.05, 0.92, 1.0))
        out["pbp_blowout_var_factor"] = float(safe_clip(1.0 + b * 0.20, 1.0, 1.18))
    return out


def _synergy_playtype_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
) -> Dict[str, float]:
    out = {"synergy_pts_factor": 1.0, "synergy_fg3m_factor": 1.0, "synergy_ast_factor": 1.0}
    df = _memoized(_memo_key(cache, "synergyplaytypes", season, "P"), lambda: get_synergy_playtypes(season=season, cache=cache, player_or_team="P"))
    if len(df) == 0:
        return out
    pid_col = _find_col(df, ["PLAYER_ID", "PERSON_ID", "playerId"])
    if not pid_col:
        return out
    rows = df[df[pid_col].astype(str) == str(player_id)]
    if len(rows) == 0:
        return out
    play_col = _find_col(rows, ["PLAY_TYPE", "PLAYTYPE"])
    ppp_col = _find_col(rows, ["PPP", "POINTS_PER_POSSESSION"])
    freq_col = _find_col(rows, ["POSS_PCT", "PERCENTILE", "frequency"])
    if not play_col or not ppp_col:
        return out
    rows = rows.copy()
    rows["_ppp"] = pd.to_numeric(rows[ppp_col], errors="coerce").fillna(0.0)
    rows["_freq"] = pd.to_numeric(rows[freq_col], errors="coerce").fillna(0.0) if freq_col else 1.0
    rows["_w"] = rows["_freq"].clip(lower=0.0)
    if float(rows["_w"].sum()) <= 0:
        rows["_w"] = 1.0
    base = float((rows["_ppp"] * rows["_w"]).sum() / rows["_w"].sum())

    def _blend(play_tokens: List[str], shrink: float, lo: float, hi: float) -> float:
        mask = rows[play_col].astype(str).str.lower().apply(lambda s: any(tok in s for tok in play_tokens))
        sub = rows[mask]
        if len(sub) == 0:
            return 1.0
        val = float((sub["_ppp"] * sub["_w"]).sum() / max(float(sub["_w"].sum()), 1e-6))
        if base <= 0:
            return 1.0
        return float(safe_clip(1.0 + ((val / base) - 1.0) * shrink, lo, hi))

    out["synergy_pts_factor"] = _blend(["isolation", "post", "cut", "transition"], shrink=0.20, lo=0.92, hi=1.10)
    out["synergy_fg3m_factor"] = _blend(["spotup", "handoff", "offscreen"], shrink=0.20, lo=0.90, hi=1.12)
    out["synergy_ast_factor"] = _blend(["pickandroll", "handoff"], shrink=0.20, lo=0.90, hi=1.12)
    return out


def _onoff_summary_scenario_factors(
    cache: SQLiteCache,
    season: str,
    *,
    team_id: int,
    target_player: str,
    team_out: List[str],
) -> Dict[str, float]:
    out = {
        "onoff_summary_usage_factor": 1.0,
        "onoff_summary_ast_factor": 1.0,
        "onoff_summary_reb_factor": 1.0,
        "onoff_summary_pace_factor": 1.0,
        "onoff_summary_pts_market_factor": 1.0,
    }
    if not team_out:
        return out
    data = _memoized(
        _memo_key(cache, "teamplayeronoffsummary", season, int(team_id)),
        lambda: get_team_player_onoff_summary(team_id=team_id, season=season, cache=cache),
    )
    on_df = data.get("on", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    off_df = data.get("off", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    if len(on_df) == 0 or len(off_df) == 0:
        return out

    offrt_col = _find_col(on_df, ["OFF_RATING", "offensiveRating", "OFFRTG"])
    pace_col = _find_col(on_df, ["PACE", "pace"])
    net_col = _find_col(on_df, ["NET_RATING", "netRating"])
    min_col = _find_col(on_df, ["MIN", "minutes"])
    target_row = _row_by_name(on_df, target_player)
    target_weight = 0.20
    if target_row is not None and min_col:
        target_mins = _to_float(target_row.get(min_col))
        if target_mins is not None:
            target_weight = float(safe_clip(target_mins / 36.0, 0.12, 0.35))

    for absent in team_out:
        absent_on = _row_by_name(on_df, absent)
        absent_off = _row_by_name(off_df, absent)
        if absent_on is None or absent_off is None:
            continue
        if offrt_col:
            on_offrt = _to_float(absent_on.get(offrt_col))
            off_offrt = _to_float(absent_off.get(offrt_col))
            if on_offrt and off_offrt and on_offrt > 0:
                ratio = float(safe_clip(off_offrt / on_offrt, 0.84, 1.18))
                out["onoff_summary_usage_factor"] *= float(safe_clip(1.0 + (1.0 - ratio) * (target_weight + 0.10), 0.92, 1.10))
                out["onoff_summary_ast_factor"] *= float(safe_clip(1.0 + (1.0 - ratio) * (target_weight + 0.12), 0.90, 1.12))
                out["onoff_summary_pts_market_factor"] *= float(safe_clip(1.0 + (ratio - 1.0) * 0.22, 0.90, 1.10))
        if pace_col:
            on_pace = _to_float(absent_on.get(pace_col))
            off_pace = _to_float(absent_off.get(pace_col))
            if on_pace and off_pace and on_pace > 0:
                pr = float(safe_clip(off_pace / on_pace, 0.90, 1.10))
                out["onoff_summary_pace_factor"] *= float(safe_clip(1.0 + (pr - 1.0) * 0.55, 0.94, 1.06))
        if net_col:
            on_net = _to_float(absent_on.get(net_col))
            off_net = _to_float(absent_off.get(net_col))
            if on_net is not None and off_net is not None:
                delta = float(off_net - on_net)
                out["onoff_summary_reb_factor"] *= float(safe_clip(1.0 + (delta / 25.0) * 0.08, 0.92, 1.08))

    out["onoff_summary_usage_factor"] = float(safe_clip(out["onoff_summary_usage_factor"], 0.90, 1.12))
    out["onoff_summary_ast_factor"] = float(safe_clip(out["onoff_summary_ast_factor"], 0.88, 1.14))
    out["onoff_summary_reb_factor"] = float(safe_clip(out["onoff_summary_reb_factor"], 0.90, 1.10))
    out["onoff_summary_pace_factor"] = float(safe_clip(out["onoff_summary_pace_factor"], 0.92, 1.08))
    out["onoff_summary_pts_market_factor"] = float(safe_clip(out["onoff_summary_pts_market_factor"], 0.90, 1.10))
    return out


def _shot_creation_style_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
    team_id: int,
    opp_team_id: int,
    opp_name: str,
    opp_abbr: str,
) -> Dict[str, float]:
    out = {
        "shot_style_pts_factor": 1.0,
        "shot_style_fg3m_factor": 1.0,
        "shot_style_ast_factor": 1.0,
    }
    player_pt = _memoized(
        _memo_key(cache, "leaguedashplayerptshot", season, int(team_id), int(player_id), ""),
        lambda: get_league_dash_player_pt_shot(season=season, cache=cache, player_id=player_id, team_id=team_id),
    )
    team_pt = _memoized(
        _memo_key(cache, "leaguedashteamptshot", season, int(opp_team_id), ""),
        lambda: get_league_dash_team_pt_shot(season=season, cache=cache, team_id=opp_team_id),
    )
    team_def = _memoized(
        _memo_key(cache, "leaguedashptteamdefend", season, "Overall"),
        lambda: get_league_dash_pt_team_defend(season=season, cache=cache, defense_category="Overall"),
    )
    team_dash = _memoized(
        _memo_key(cache, "teamdashptshots", season, int(opp_team_id), 25),
        lambda: get_team_dash_pt_shots(team_id=opp_team_id, season=season, cache=cache, last_n_games=25),
    )

    prow = _row_by_player_id(player_pt, player_id)
    trow = _row_by_team_id_or_name(team_pt, opp_team_id, opp_name, opp_abbr)
    drow = _row_by_team_id_or_name(team_def, opp_team_id, opp_name, opp_abbr)
    if prow is None and drow is None and trow is None:
        return out

    p_3freq_col = _find_col(player_pt, ["FG3A_FREQUENCY", "FG3A_FREQ", "fg3aFrequency"])
    p_efg_col = _find_col(player_pt, ["EFG_PCT", "effectiveFieldGoalPercentage"])
    t_3freq_col = _find_col(team_pt, ["FG3A_FREQUENCY", "FG3A_FREQ", "fg3aFrequency"])
    d_pct_col = _find_col(team_def, ["D_FG_PCT", "defendedFgPct", "FG_PCT"])

    p_3freq = _to_float(prow.get(p_3freq_col)) if prow is not None and p_3freq_col else None
    p_efg = _to_float(prow.get(p_efg_col)) if prow is not None and p_efg_col else None
    t_3freq = _to_float(trow.get(t_3freq_col)) if trow is not None and t_3freq_col else None
    d_pct = _to_float(drow.get(d_pct_col)) if drow is not None and d_pct_col else None

    lg_p3 = _to_float(pd.to_numeric(player_pt[p_3freq_col], errors="coerce").mean()) if p_3freq_col and len(player_pt) else None
    lg_pefg = _to_float(pd.to_numeric(player_pt[p_efg_col], errors="coerce").mean()) if p_efg_col and len(player_pt) else None
    lg_t3 = _to_float(pd.to_numeric(team_pt[t_3freq_col], errors="coerce").mean()) if t_3freq_col and len(team_pt) else None
    lg_dpct = _to_float(pd.to_numeric(team_def[d_pct_col], errors="coerce").mean()) if d_pct_col and len(team_def) else None

    touch_long_adj = 0.0
    if isinstance(team_dash, dict):
        touch_df = team_dash.get("touch_time", pd.DataFrame())
        range_col = _find_col(touch_df, ["TOUCH_TIME_RANGE", "touchTimeRange"])
        freq_col = _find_col(touch_df, ["FGA_FREQUENCY", "fgaFrequency"])
        if len(touch_df) and range_col and freq_col:
            tmp = touch_df.copy()
            tmp["_f"] = pd.to_numeric(tmp[freq_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            total = float(tmp["_f"].sum())
            if total > 0:
                long_share = float(
                    tmp[tmp[range_col].astype(str).str.contains("6", case=False, na=False)]["_f"].sum() / total
                )
                quick_share = float(
                    tmp[tmp[range_col].astype(str).str.contains("<\\s*2|0-2|2 seconds", case=False, na=False, regex=True)]["_f"].sum() / total
                )
                touch_long_adj = (long_share - quick_share) * 0.10

    fg3_ratio = 1.0
    if p_3freq and lg_p3 and lg_p3 > 0:
        fg3_ratio *= p_3freq / lg_p3
    if t_3freq and lg_t3 and lg_t3 > 0:
        fg3_ratio *= t_3freq / lg_t3
    if d_pct and lg_dpct and lg_dpct > 0:
        fg3_ratio *= lg_dpct / d_pct
    out["shot_style_fg3m_factor"] = float(safe_clip(1.0 + (fg3_ratio - 1.0) * 0.22 + touch_long_adj, 0.90, 1.12))

    pts_ratio = 1.0
    if p_efg and lg_pefg and lg_pefg > 0:
        pts_ratio *= p_efg / lg_pefg
    if d_pct and lg_dpct and lg_dpct > 0:
        pts_ratio *= lg_dpct / d_pct
    out["shot_style_pts_factor"] = float(safe_clip(1.0 + (pts_ratio - 1.0) * 0.18 + touch_long_adj * 0.5, 0.90, 1.10))

    out["shot_style_ast_factor"] = float(
        safe_clip(
            1.0
            + ((out["shot_style_pts_factor"] - 1.0) * 0.30)
            + touch_long_adj * 0.8,
            0.90,
            1.10,
        )
    )
    return out


def _fg3m_endpoint_focus_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
    team_id: int,
    team_name: str,
    opp_team_id: int,
    opp_name: str,
    opp_abbr: str,
    team_base_df: pd.DataFrame,
) -> Dict[str, float]:
    out = {
        "fg3m_catch_shoot_factor": 1.0,
        "fg3m_pullup_factor": 1.0,
        "fg3m_transition_factor": 1.0,
        "fg3m_zone_matchup_factor": 1.0,
        "fg3m_team_traditional_factor": 1.0,
        "fg3m_lineup_traditional_factor": 1.0,
    }

    def _ratio(value: Optional[float], baseline: Optional[float], *, lo: float = 0.72, hi: float = 1.38) -> float:
        if value is None or baseline is None or baseline <= 0:
            return 1.0
        return float(safe_clip(value / baseline, lo, hi))

    def _mean(df: pd.DataFrame, col: Optional[str]) -> Optional[float]:
        if not col or col not in df.columns:
            return None
        return _to_float(pd.to_numeric(df[col], errors="coerce").mean())

    def _ptshot_factor(
        *,
        general_range: str,
        shrink: float,
        lo: float = 0.88,
        hi: float = 1.14,
    ) -> float:
        player_df = _memoized(
            _memo_key(cache, "leaguedashplayerptshot", season, int(team_id), int(player_id), general_range),
            lambda gr=general_range: get_league_dash_player_pt_shot(
                season=season,
                cache=cache,
                player_id=player_id,
                team_id=team_id,
                general_range=gr,
            ),
        )
        opp_df = _memoized(
            _memo_key(cache, "leaguedashoppptshot", season, general_range),
            lambda gr=general_range: get_opponent_ptshot_defense(season=season, general_range=gr, cache=cache),
        )
        prow = _row_by_player_id(player_df, player_id)
        orow = _row_by_team_id_or_name(opp_df, opp_team_id, opp_name, opp_abbr)
        if prow is None and orow is None:
            return 1.0

        p_3freq_col = _find_col(player_df, ["FG3A_FREQUENCY", "FG3A_FREQ", "fg3aFrequency", "FG3A"])
        p_3pct_col = _find_col(player_df, ["FG3_PCT", "FG3M_PCT", "FG3M_PERCENT", "FG_PCT", "EFG_PCT"])
        o_3freq_col = _find_col(opp_df, ["FG3A_FREQUENCY", "FG3A_FREQ", "fg3aFrequency", "FG3A"])
        o_3pct_col = _find_col(opp_df, ["FG3_PCT", "FG3M_PCT", "FG3M_PERCENT", "FG_PCT"])

        p_3freq = _to_float(prow.get(p_3freq_col)) if prow is not None and p_3freq_col else None
        p_3pct = _to_float(prow.get(p_3pct_col)) if prow is not None and p_3pct_col else None
        o_3freq = _to_float(orow.get(o_3freq_col)) if orow is not None and o_3freq_col else None
        o_3pct = _to_float(orow.get(o_3pct_col)) if orow is not None and o_3pct_col else None

        lg_p_3freq = _mean(player_df, p_3freq_col)
        lg_p_3pct = _mean(player_df, p_3pct_col)
        lg_o_3freq = _mean(opp_df, o_3freq_col)
        lg_o_3pct = _mean(opp_df, o_3pct_col)

        raw_ratio = (
            _ratio(p_3freq, lg_p_3freq, lo=0.75, hi=1.35)
            * _ratio(p_3pct, lg_p_3pct, lo=0.80, hi=1.25)
            * _ratio(o_3freq, lg_o_3freq, lo=0.78, hi=1.30)
            * _ratio(o_3pct, lg_o_3pct, lo=0.80, hi=1.25)
        )
        return float(safe_clip(1.0 + (raw_ratio - 1.0) * shrink, lo, hi))

    out["fg3m_catch_shoot_factor"] = _ptshot_factor(general_range="Catch and Shoot", shrink=0.24)
    out["fg3m_pullup_factor"] = _ptshot_factor(general_range="Pullups", shrink=0.18)
    out["fg3m_transition_factor"] = _ptshot_factor(general_range="Transition", shrink=0.14, lo=0.90, hi=1.12)

    zone_df = _memoized(
        _memo_key(cache, "leaguedashteamshotlocations", season, "Opponent", "By Zone"),
        lambda: get_league_dash_team_shot_locations(
            season=season,
            cache=cache,
            measure_type_simple="Opponent",
            distance_range="By Zone",
        ),
    )
    zrow = _row_by_team_id_or_name(zone_df, opp_team_id, opp_name, opp_abbr)
    if zrow is not None and len(zone_df):
        cols = list(zone_df.columns)

        def _zone_cols(zone_tag: str, stat_tag: str) -> List[str]:
            z = str(zone_tag).lower()
            s = str(stat_tag).lower()
            return [c for c in cols if z in str(c).lower() and s in str(c).lower()]

        ab3_fga_cols = _zone_cols("above the break 3", "fga")
        corner_fga_cols = _zone_cols("corner 3", "fga")
        ab3_pct_cols = _zone_cols("above the break 3", "fg")
        corner_pct_cols = _zone_cols("corner 3", "fg")

        def _row_sum(row: pd.Series, use_cols: List[str]) -> Optional[float]:
            if not use_cols:
                return None
            vals = pd.to_numeric(row[use_cols], errors="coerce").dropna()
            if len(vals) == 0:
                return None
            return float(vals.sum())

        def _mean_sum(df: pd.DataFrame, use_cols: List[str]) -> Optional[float]:
            if not use_cols:
                return None
            vals = pd.to_numeric(df[use_cols].sum(axis=1), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                return None
            return float(vals.mean())

        def _row_mean(row: pd.Series, use_cols: List[str]) -> Optional[float]:
            if not use_cols:
                return None
            vals = pd.to_numeric(row[use_cols], errors="coerce").dropna()
            if len(vals) == 0:
                return None
            return float(vals.mean())

        def _mean_mean(df: pd.DataFrame, use_cols: List[str]) -> Optional[float]:
            if not use_cols:
                return None
            vals = pd.to_numeric(df[use_cols].mean(axis=1), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                return None
            return float(vals.mean())

        row_3pa = (_row_sum(zrow, ab3_fga_cols) or 0.0) + (_row_sum(zrow, corner_fga_cols) or 0.0)
        lg_3pa = (_mean_sum(zone_df, ab3_fga_cols) or 0.0) + (_mean_sum(zone_df, corner_fga_cols) or 0.0)
        row_pct_vals = [
            v
            for v in [
                _row_mean(zrow, ab3_pct_cols),
                _row_mean(zrow, corner_pct_cols),
            ]
            if v is not None
        ]
        lg_pct_vals = [
            v
            for v in [
                _mean_mean(zone_df, ab3_pct_cols),
                _mean_mean(zone_df, corner_pct_cols),
            ]
            if v is not None
        ]
        row_3pct = float(np.mean(np.array(row_pct_vals, dtype=float))) if row_pct_vals else None
        lg_3pct = float(np.mean(np.array(lg_pct_vals, dtype=float))) if lg_pct_vals else None

        zone_ratio = _ratio(row_3pa, lg_3pa, lo=0.78, hi=1.32) * _ratio(row_3pct, lg_3pct, lo=0.82, hi=1.22)
        out["fg3m_zone_matchup_factor"] = float(safe_clip(1.0 + (zone_ratio - 1.0) * 0.20, 0.88, 1.14))

    out["fg3m_team_traditional_factor"] = _fg3m_team_traditional_factor(
        team_base_df=team_base_df,
        team_name=team_name,
        opp_name=opp_name,
    )

    team_lineups = _memoized(
        _memo_key(cache, "teamdashlineups", season, int(team_id), "fg3_focus", 5, 25),
        lambda: get_team_dash_lineups(team_id=team_id, season=season, cache=cache, group_quantity=5, last_n_games=25),
    )
    opp_lineups = _memoized(
        _memo_key(cache, "teamdashlineups", season, int(opp_team_id), "fg3_focus", 5, 25),
        lambda: get_team_dash_lineups(team_id=opp_team_id, season=season, cache=cache, group_quantity=5, last_n_games=25),
    )
    league_lineups = _memoized(
        _memo_key(cache, "leaguedashlineups", season, "fg3_focus", 5, 25),
        lambda: get_league_dash_lineups(season=season, cache=cache, group_quantity=5, last_n_games=25),
    )
    fg3a_col_t = _find_col(team_lineups, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])
    fg3a_col_o = _find_col(opp_lineups, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])
    fg3a_col_l = _find_col(league_lineups, ["FG3A", "FG3A_FREQUENCY", "threePointFieldGoalsAttempted"])
    mins_col_t = _find_col(team_lineups, ["MIN", "minutes"])
    mins_col_o = _find_col(opp_lineups, ["MIN", "minutes"])
    mins_col_l = _find_col(league_lineups, ["MIN", "minutes"])

    def _weighted(df: pd.DataFrame, value_col: Optional[str], weight_col: Optional[str]) -> Optional[float]:
        if len(df) == 0 or not value_col or value_col not in df.columns:
            return None
        vals = pd.to_numeric(df[value_col], errors="coerce")
        if weight_col and weight_col in df.columns:
            w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            w = pd.Series(np.ones(len(df)), index=df.index, dtype=float)
        mask = vals.notna() & w.notna()
        if not mask.any():
            return None
        vv = vals[mask].astype(float)
        ww = w[mask].astype(float)
        total = float(ww.sum())
        if total <= 0:
            return None
        return float((vv * ww).sum() / total)

    t_fg3 = _weighted(team_lineups, fg3a_col_t, mins_col_t)
    o_fg3 = _weighted(opp_lineups, fg3a_col_o, mins_col_o)
    lg_fg3 = _weighted(league_lineups, fg3a_col_l, mins_col_l)
    if t_fg3 is not None and o_fg3 is not None and lg_fg3 and lg_fg3 > 0:
        lineup_ratio = ((t_fg3 / lg_fg3) * (o_fg3 / lg_fg3)) ** 0.5
        out["fg3m_lineup_traditional_factor"] = float(safe_clip(1.0 + (lineup_ratio - 1.0) * 0.22, 0.90, 1.14))

    return out


def _spatial_lineup_shotmap_factors(
    cache: SQLiteCache,
    season: str,
    *,
    player_id: int,
    team_id: int,
    opp_team_id: int,
) -> Dict[str, float]:
    out = {
        "spatial_pts_factor": 1.0,
        "spatial_fg3m_factor": 1.0,
        "spatial_reb_factor": 1.0,
    }
    loc_df = _memoized(
        _memo_key(cache, "leaguedashplayershotlocations", season, int(team_id)),
        lambda: get_league_dash_player_shot_locations(season=season, cache=cache, team_id=team_id),
    )
    line_chart = _memoized(
        _memo_key(cache, "shotchartlineupdetail", season, int(opp_team_id), int(opp_team_id), 10, "FGA"),
        lambda: get_shot_chart_lineup_detail(
            season=season,
            cache=cache,
            team_id=opp_team_id,
            group_id=opp_team_id,
            last_n_games=10,
            context_measure_detailed="FGA",
        ),
    )
    prow = _row_by_player_id(loc_df, player_id)
    player_corner_share = 0.0
    player_ab3_share = 0.0
    player_rim_share = 0.0
    if prow is not None:
        cols = list(loc_df.columns)
        fga_by_zone: Dict[str, float] = {}
        for zone in ["Restricted Area", "Left Corner 3", "Right Corner 3", "Above the Break 3"]:
            zone_col = next((c for c in cols if str(zone).lower() in str(c).lower() and "fga" in str(c).lower()), None)
            if zone_col:
                fga_by_zone[zone] = float(_to_float(prow.get(zone_col)) or 0.0)
        if not fga_by_zone:
            vals = pd.to_numeric(pd.Series(prow.values), errors="coerce").dropna().astype(float).values
            if len(vals) >= 21:
                tail = vals[-21:]
                fga_by_zone = {
                    "Restricted Area": float(max(tail[1], 0.0)),
                    "Left Corner 3": float(max(tail[10], 0.0)),
                    "Right Corner 3": float(max(tail[13], 0.0)),
                    "Above the Break 3": float(max(tail[16], 0.0)),
                }
        total = float(sum(v for v in fga_by_zone.values() if v > 0))
        if total > 0:
            player_corner_share = float((fga_by_zone.get("Left Corner 3", 0.0) + fga_by_zone.get("Right Corner 3", 0.0)) / total)
            player_ab3_share = float(fga_by_zone.get("Above the Break 3", 0.0) / total)
            player_rim_share = float(fga_by_zone.get("Restricted Area", 0.0) / total)

    shots = line_chart.get("shots", pd.DataFrame()) if isinstance(line_chart, dict) else pd.DataFrame()
    if len(shots) == 0:
        return out
    zone_col = _find_col(shots, ["SHOT_ZONE_BASIC", "shotZoneBasic"])
    dist_col = _find_col(shots, ["SHOT_DISTANCE", "shotDistance"])
    if not zone_col:
        return out

    zones = shots[zone_col].astype(str).str.lower()
    total = float(len(zones))
    if total <= 0:
        return out
    corner_rate = float(zones.str.contains("corner 3").mean())
    ab3_rate = float(zones.str.contains("above the break 3").mean())
    rim_rate = float(zones.str.contains("restricted area").mean())
    perim_rate = corner_rate + ab3_rate
    avg_dist = float(pd.to_numeric(shots[dist_col], errors="coerce").dropna().mean()) if dist_col else 0.0

    player_perim = player_corner_share + player_ab3_share
    if player_perim > 0:
        out["spatial_fg3m_factor"] = float(safe_clip(1.0 + (((player_perim * max(perim_rate, 1e-6)) / 0.15) - 1.0) * 0.16, 0.90, 1.12))
    if player_rim_share > 0:
        out["spatial_pts_factor"] = float(safe_clip(1.0 + (((player_rim_share * max(rim_rate, 1e-6)) / 0.08) - 1.0) * 0.14, 0.90, 1.10))
    out["spatial_reb_factor"] = float(
        safe_clip(
            1.0 + ((perim_rate / 0.35) - 1.0) * 0.10 + ((avg_dist / 14.0) - 1.0) * 0.06,
            0.92,
            1.12,
        )
    )
    return out


def _recent_defensive_form_factors(
    cache: SQLiteCache,
    game_log: pd.DataFrame,
    *,
    opp_abbr: str,
    opp_team_id: int,
) -> Dict[str, float]:
    out = {
        "recent_def_pts_factor": 1.0,
        "recent_def_fg3m_factor": 1.0,
        "recent_def_ast_factor": 1.0,
        "recent_def_reb_factor": 1.0,
    }
    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    opp_col = _find_col(game_log, ["OPP_ABBR"])
    if not gid_col or not opp_col or len(game_log) == 0:
        return out
    sample = game_log[game_log[opp_col].astype(str).str.upper() == str(opp_abbr).upper()].head(6)
    if len(sample) == 0:
        return out

    def_vals: List[float] = []
    efg_vals: List[float] = []
    tov_vals: List[float] = []
    oreb_vals: List[float] = []
    paint_vals: List[float] = []
    for gid in sample[gid_col].astype(str).tolist():
        adv = _memoized(_memo_key(cache, "boxscoreadvancedv3", gid), lambda gid=gid: get_boxscore_advanced_v3(gid, cache))
        ff = _memoized(_memo_key(cache, "boxscorefourfactorsv3", gid), lambda gid=gid: get_boxscore_four_factors_v3(gid, cache))
        misc = _memoized(_memo_key(cache, "boxscoremiscv3", gid), lambda gid=gid: get_boxscore_misc_v3(gid, cache))
        team_adv = adv.get("team", pd.DataFrame()) if isinstance(adv, dict) else pd.DataFrame()
        team_ff = ff.get("team", pd.DataFrame()) if isinstance(ff, dict) else pd.DataFrame()
        team_misc = misc.get("team", pd.DataFrame()) if isinstance(misc, dict) else pd.DataFrame()

        arow = _row_by_team_id_or_name(team_adv, opp_team_id, "", "")
        frow = _row_by_team_id_or_name(team_ff, opp_team_id, "", "")
        mrow = _row_by_team_id_or_name(team_misc, opp_team_id, "", "")

        if arow is not None:
            def_col = _find_col(team_adv, ["defensiveRating", "DEF_RATING", "defRating"])
            v = _to_float(arow.get(def_col)) if def_col else None
            if v is not None:
                def_vals.append(float(v))
        if frow is not None:
            efg_col = _find_col(team_ff, ["oppEffectiveFieldGoalPercentage", "OPP_EFG_PCT"])
            tov_col = _find_col(team_ff, ["oppTeamTurnoverPercentage", "OPP_TOV_PCT"])
            oreb_col = _find_col(team_ff, ["oppOffensiveReboundPercentage", "OPP_OREB_PCT"])
            efg = _to_float(frow.get(efg_col)) if efg_col else None
            tov = _to_float(frow.get(tov_col)) if tov_col else None
            oreb = _to_float(frow.get(oreb_col)) if oreb_col else None
            if efg is not None:
                efg_vals.append(float(efg))
            if tov is not None:
                tov_vals.append(float(tov))
            if oreb is not None:
                oreb_vals.append(float(oreb))
        if mrow is not None:
            paint_col = _find_col(team_misc, ["oppPointsPaint", "OPP_PTS_PAINT"])
            paint = _to_float(mrow.get(paint_col)) if paint_col else None
            if paint is not None:
                paint_vals.append(float(paint))

    if not any([def_vals, efg_vals, tov_vals, oreb_vals, paint_vals]):
        return out
    w = _recent_weights(max(len(def_vals), len(efg_vals), len(tov_vals), len(oreb_vals), len(paint_vals)))

    def _wmean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        ww = _recent_weights(len(arr))
        return float(np.average(arr, weights=ww))

    def_rt = _wmean(def_vals)
    opp_efg = _wmean(efg_vals)
    opp_tov = _wmean(tov_vals)
    opp_oreb = _wmean(oreb_vals)
    opp_paint = _wmean(paint_vals)

    permissive = 1.0
    if def_rt is not None:
        permissive *= float(safe_clip(def_rt / 112.0, 0.88, 1.14))
    if opp_efg is not None:
        permissive *= float(safe_clip(opp_efg / 0.54, 0.88, 1.14))
    if opp_paint is not None:
        permissive *= float(safe_clip(opp_paint / 48.0, 0.88, 1.14))
    out["recent_def_pts_factor"] = float(safe_clip(1.0 + (permissive - 1.0) * 0.22, 0.88, 1.12))
    out["recent_def_fg3m_factor"] = float(safe_clip(1.0 + (permissive - 1.0) * 0.20, 0.88, 1.12))

    if opp_tov is not None:
        ast_ratio = float(safe_clip(0.14 / max(opp_tov, 1e-6), 0.86, 1.14))
        out["recent_def_ast_factor"] = float(safe_clip(1.0 + (ast_ratio - 1.0) * 0.22, 0.88, 1.12))
    if opp_oreb is not None:
        reb_ratio = float(safe_clip(opp_oreb / 0.28, 0.86, 1.14))
        out["recent_def_reb_factor"] = float(safe_clip(1.0 + (reb_ratio - 1.0) * 0.18, 0.90, 1.10))
    return out


def _matchup_persistence_switch_factors(
    cache: SQLiteCache,
    game_log: pd.DataFrame,
    *,
    player_id: int,
    opp_abbr: str,
) -> Dict[str, float]:
    out = {
        "matchup_persistence_pts_factor": 1.0,
        "matchup_persistence_fg3m_factor": 1.0,
        "matchup_persistence_ast_factor": 1.0,
        "matchup_switch_var_factor": 1.0,
    }
    gid_col = _find_col(game_log, ["GAME_ID", "Game_ID"])
    opp_col = _find_col(game_log, ["OPP_ABBR"])
    if not gid_col or not opp_col or len(game_log) == 0:
        return out
    sample = game_log[game_log[opp_col].astype(str).str.upper() == str(opp_abbr).upper()].head(6)
    if len(sample) == 0:
        return out

    top_shares: List[float] = []
    defender_counts: List[float] = []
    stint_counts: List[float] = []
    for gid in sample[gid_col].astype(str).tolist():
        mdf = _memoized(_memo_key(cache, "boxscorematchupsv3", gid), lambda gid=gid: get_boxscore_matchups(gid, cache))
        if len(mdf):
            off_col = _find_col(mdf, ["OFF_PLAYER_ID", "offPersonId", "offPlayerId"])
            def_col = _find_col(mdf, ["DEF_PLAYER_ID", "defPersonId", "defPlayerId"])
            w_col = _find_col(mdf, ["MATCHUP_MIN", "matchupMinutes", "minutes", "partialPossessions", "POSS"])
            if off_col and def_col and w_col:
                rows = mdf[mdf[off_col].astype(str) == str(player_id)].copy()
                if len(rows):
                    rows["_w"] = pd.to_numeric(rows[w_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                    tot = float(rows["_w"].sum())
                    if tot > 0:
                        by_def = rows.groupby(def_col, dropna=False)["_w"].sum().sort_values(ascending=False)
                        top_shares.append(float(by_def.iloc[0] / tot))
                        defenders = int((by_def >= max(2.5, 0.12 * tot)).sum())
                        defender_counts.append(float(max(defenders, 1)))
        rot = _memoized(_memo_key(cache, "gamerotation", gid), lambda gid=gid: get_game_rotation(gid, cache))
        rdf = pd.concat([rot.get("home", pd.DataFrame()), rot.get("away", pd.DataFrame())], ignore_index=True) if isinstance(rot, dict) else pd.DataFrame()
        if len(rdf):
            pid_col = _find_col(rdf, ["PERSON_ID", "PLAYER_ID", "playerId", "personId"])
            if pid_col:
                stints = int((rdf[pid_col].astype(str) == str(player_id)).sum())
                if stints > 0:
                    stint_counts.append(float(stints))

    if not top_shares:
        return out
    persistence = float(np.mean(top_shares))
    switch_count = float(np.mean(defender_counts)) if defender_counts else 2.0
    stints = float(np.mean(stint_counts)) if stint_counts else 6.0

    switchiness = float(safe_clip(switch_count / 2.4, 0.80, 1.30))
    persist_adj = float(safe_clip(1.0 + (persistence - 0.58) * -0.15, 0.92, 1.08))
    stint_adj = float(safe_clip(1.0 + ((stints / 6.0) - 1.0) * 0.10, 0.92, 1.10))

    out["matchup_persistence_pts_factor"] = float(
        safe_clip(1.0 + (persist_adj - 1.0) * 0.40 + (switchiness - 1.0) * -0.20, 0.90, 1.08)
    )
    out["matchup_persistence_fg3m_factor"] = float(
        safe_clip(1.0 + (persist_adj - 1.0) * 0.30 + (switchiness - 1.0) * -0.15, 0.90, 1.10)
    )
    out["matchup_persistence_ast_factor"] = float(
        safe_clip(1.0 + (persist_adj - 1.0) * 0.45 + (switchiness - 1.0) * -0.26, 0.88, 1.10)
    )
    out["matchup_switch_var_factor"] = float(
        safe_clip(1.0 + (switchiness - 1.0) * 0.18 + (stint_adj - 1.0) * 0.12, 0.90, 1.20)
    )
    return out


def build_matchup_lineup_context(
    *,
    cache: SQLiteCache,
    season: str,
    player_id: int,
    player_name: str,
    team_id: int,
    opp_team_id: int,
    team_name: str,
    opp_name: str,
    opp_abbr: str,
    game_log: pd.DataFrame,
    team_out: List[str],
    opponent_out: List[str],
    spread: Optional[float],
    team_base_df: pd.DataFrame,
    league_players: pd.DataFrame,
    opp_fta_factor: float,
    game_date_local: Optional[str] = None,
) -> Dict[str, Any]:
    minutes_profile = _minutes_starter_profile(game_log)
    position_bucket = get_player_primary_position(player_id, cache) or "F"
    onoff = _onoff_factors(cache, season, team_id, player_name, team_out)
    opp_pace = _lineup_opponent_pace_factor(cache, season, opp_team_id, opponent_out)
    lineup_dash = _lineup_dash_factors(cache, season, team_id)
    advanced = _advanced_metric_factors(
        cache,
        season,
        player_id=player_id,
        team_id=team_id,
        opp_team_id=opp_team_id,
        team_name=team_name,
        opp_name=opp_name,
        opp_abbr=opp_abbr,
    )
    season_matchup = _season_matchup_factors(cache, season, off_player_id=player_id, opp_team_id=opp_team_id)
    position_def = _position_defense_factors(cache, season, opp_name, opp_abbr, position_bucket)
    primary_def = _primary_defender_factors(cache, player_id, opp_abbr, game_log, league_players)
    hist_fg3 = get_runtime_historical_fg3_factors(
        player_id=player_id,
        opp_team_id=opp_team_id,
        primary_defender_id=str(primary_def.get("primary_defender_id") or ""),
        primary_defender_share=float(primary_def.get("primary_defender_share") or 0.0),
    )
    defender_discipline = _defender_discipline_factors(
        cache,
        season,
        primary_defender_id=str(primary_def.get("primary_defender_id") or ""),
        primary_defender_share=float(primary_def.get("primary_defender_share") or 0.0),
        league_players=league_players,
    )
    pt_def = _pt_defense_endpoint_factors(
        cache,
        season,
        opp_team_id=opp_team_id,
        primary_defender_id=str(primary_def.get("primary_defender_id") or ""),
    )
    hustle_pressure = _opponent_pressure_hustle_factors(
        cache,
        season,
        opp_team_id=opp_team_id,
        opp_name=opp_name,
        opp_abbr=opp_abbr,
        game_log=game_log,
    )
    lineup_combo = _lineup_interaction_matrix_factors(cache, season, team_id=team_id)
    lineup_vs_lineup = _lineup_vs_lineup_expectation_factors(
        cache,
        season,
        team_id=team_id,
        opp_team_id=opp_team_id,
    )
    opp_lineup_weak = _opponent_lineup_weakness_factors(
        cache,
        season,
        opp_team_id=opp_team_id,
        opp_name=opp_name,
        opp_abbr=opp_abbr,
    )
    onoff_summary = _onoff_summary_scenario_factors(
        cache,
        season,
        team_id=team_id,
        target_player=player_name,
        team_out=team_out,
    )
    shot_style = _shot_creation_style_factors(
        cache,
        season,
        player_id=player_id,
        team_id=team_id,
        opp_team_id=opp_team_id,
        opp_name=opp_name,
        opp_abbr=opp_abbr,
    )
    fg3m_focus = _fg3m_endpoint_focus_factors(
        cache,
        season,
        player_id=player_id,
        team_id=team_id,
        team_name=team_name,
        opp_team_id=opp_team_id,
        opp_name=opp_name,
        opp_abbr=opp_abbr,
        team_base_df=team_base_df,
    )
    spatial = _spatial_lineup_shotmap_factors(
        cache,
        season,
        player_id=player_id,
        team_id=team_id,
        opp_team_id=opp_team_id,
    )
    tracking = _player_tracking_micro_factors(cache, season, team_id=team_id, player_id=player_id)
    extra = _extra_endpoint_factors(
        cache,
        season,
        player_id=player_id,
        team_id=team_id,
        opp_team_id=opp_team_id,
        game_log=game_log,
        opp_abbr=opp_abbr,
    )
    game_day = _game_day_availability_factors(
        cache,
        game_date_local=game_date_local,
        team_id=team_id,
        opp_team_id=opp_team_id,
        player_id=player_id,
    )
    pbp = _pbp_volatility_factors(cache, game_log)
    synergy = _synergy_playtype_factors(cache, season, player_id=player_id)
    recent_def = _recent_defensive_form_factors(cache, game_log, opp_abbr=opp_abbr, opp_team_id=opp_team_id)
    matchup_switch = _matchup_persistence_switch_factors(cache, game_log, player_id=player_id, opp_abbr=opp_abbr)
    reb_env = _rebound_ecosystem_factor(team_base_df, team_name, opp_name, position_bucket)
    foul = _foul_risk_factors(game_log, opp_fta_factor)
    blowout = _blowout_risk_factors(spread, minutes_profile["starter_prob"])

    lineup_pace_factor = float(
        safe_clip(
            (
                float(onoff["lineup_team_pace_factor"])
                + float(opp_pace)
                + float(lineup_dash["lineup_dash_pace_factor"])
                + float(advanced["team_est_pace_factor"])
                + float(lineup_combo["lineup_combo_pace_factor"])
                + float(onoff_summary["onoff_summary_pace_factor"])
                + float(lineup_vs_lineup["lineup_vs_lineup_pace_factor"])
                + float(extra["pt_stats_pace_factor"])
            )
            / 8.0,
            0.90,
            1.10,
        )
    )
    lineup_minutes_factor = float(
        safe_clip(
            float(minutes_profile["lineup_minutes_factor"])
            * float(game_day["game_day_minutes_factor"])
            * float(lineup_combo["lineup_combo_stability_factor"]),
            0.84,
            1.14,
        )
    )
    lineup_minutes_var_factor = float(
        safe_clip(
            float(minutes_profile["lineup_minutes_var_factor"])
            * float(pbp["pbp_blowout_var_factor"])
            * float(lineup_combo["lineup_combo_var_factor"])
            * float(matchup_switch["matchup_switch_var_factor"]),
            0.86,
            1.28,
        )
    )

    onoff_usage_factor = float(
        safe_clip(
            float(onoff["onoff_usage_factor"])
            * float(lineup_dash["lineup_dash_off_factor"])
            * float(lineup_combo["lineup_combo_off_factor"])
            * float(advanced["player_est_impact_factor"])
            * float(game_day["game_day_usage_factor"])
            * float(onoff_summary["onoff_summary_usage_factor"])
            * float(lineup_vs_lineup["lineup_vs_lineup_pts_factor"])
            * float(extra["team_clutch_pts_factor"]),
            0.84,
            1.24,
        )
    )
    onoff_ast_factor = float(
        safe_clip(
            float(onoff["onoff_ast_factor"])
            * float(tracking["pt_pass_ast_factor"])
            * float(lineup_dash["lineup_dash_off_factor"])
            * float(lineup_combo["lineup_combo_off_factor"])
            * float(advanced["player_est_impact_factor"])
            * float(game_day["game_day_usage_factor"])
            * float(onoff_summary["onoff_summary_ast_factor"])
            * float(shot_style["shot_style_ast_factor"])
            * float(lineup_vs_lineup["lineup_vs_lineup_ast_factor"])
            * float(extra["assist_tracker_ast_factor"])
            * float(extra["team_clutch_ast_factor"]),
            0.82,
            1.26,
        )
    )
    onoff_reb_factor = float(
        safe_clip(
            float(onoff["onoff_reb_factor"])
            * float(tracking["pt_reb_factor"])
            * float(advanced["player_est_impact_factor"])
            * float(onoff_summary["onoff_summary_reb_factor"])
            * float(spatial["spatial_reb_factor"])
            * float(lineup_vs_lineup["lineup_vs_lineup_reb_factor"]),
            0.84,
            1.24,
        )
    )
    onoff_fg3a_factor = float(
        safe_clip(
            float(onoff["onoff_fg3a_factor"])
            * float(lineup_dash["lineup_dash_off_factor"])
            * float(lineup_combo["lineup_combo_off_factor"])
            * float(advanced["player_est_impact_factor"])
            * float(game_day["game_day_usage_factor"])
            * float(shot_style["shot_style_fg3m_factor"])
            * float(fg3m_focus["fg3m_catch_shoot_factor"])
            * float(fg3m_focus["fg3m_pullup_factor"])
            * float(fg3m_focus["fg3m_transition_factor"])
            * float(fg3m_focus["fg3m_team_traditional_factor"])
            * float(fg3m_focus["fg3m_lineup_traditional_factor"])
            * float(lineup_vs_lineup["lineup_vs_lineup_fg3m_factor"])
            * float(hist_fg3["hist_fg3a_volume_factor"])
            * float(extra["shot_chart_detail_fg3_factor"]),
            0.84,
            1.24,
        )
    )

    blowout_minutes_factor = float(
        safe_clip(float(blowout["blowout_minutes_factor"]) * float(pbp["pbp_blowout_minutes_factor"]), 0.90, 1.02)
    )
    blowout_var_factor = float(
        safe_clip(float(blowout["blowout_var_factor"]) * float(pbp["pbp_blowout_var_factor"]), 1.0, 1.24)
    )

    out: Dict[str, Any] = {
        "position_bucket": position_bucket,
        "minutes_p10": float(minutes_profile["minutes_p10"]),
        "minutes_p90": float(minutes_profile["minutes_p90"]),
        "starter_prob": float(minutes_profile["starter_prob"]),
        "rotation_certainty": float(minutes_profile["rotation_certainty"]),
        "lineup_minutes_factor": lineup_minutes_factor,
        "lineup_minutes_var_factor": lineup_minutes_var_factor,
        "onoff_usage_factor": onoff_usage_factor,
        "onoff_ast_factor": onoff_ast_factor,
        "onoff_reb_factor": onoff_reb_factor,
        "onoff_fg3a_factor": onoff_fg3a_factor,
        "lineup_pace_factor": float(lineup_pace_factor),
        "lineup_dash_pace_factor": float(lineup_dash["lineup_dash_pace_factor"]),
        "lineup_dash_off_factor": float(lineup_dash["lineup_dash_off_factor"]),
        "lineup_dash_stability": float(lineup_dash["lineup_dash_stability"]),
        "lineup_combo_off_factor": float(lineup_combo["lineup_combo_off_factor"]),
        "lineup_combo_pace_factor": float(lineup_combo["lineup_combo_pace_factor"]),
        "lineup_combo_stability_factor": float(lineup_combo["lineup_combo_stability_factor"]),
        "lineup_combo_var_factor": float(lineup_combo["lineup_combo_var_factor"]),
        "lineup_vs_lineup_pts_factor": float(lineup_vs_lineup["lineup_vs_lineup_pts_factor"]),
        "lineup_vs_lineup_fg3m_factor": float(lineup_vs_lineup["lineup_vs_lineup_fg3m_factor"]),
        "lineup_vs_lineup_ast_factor": float(lineup_vs_lineup["lineup_vs_lineup_ast_factor"]),
        "lineup_vs_lineup_reb_factor": float(lineup_vs_lineup["lineup_vs_lineup_reb_factor"]),
        "lineup_vs_lineup_pace_factor": float(lineup_vs_lineup["lineup_vs_lineup_pace_factor"]),
        "lineup_vs_lineup_coverage": float(lineup_vs_lineup["lineup_vs_lineup_coverage"]),
        "assist_tracker_ast_factor": float(extra["assist_tracker_ast_factor"]),
        "pt_stats_pace_factor": float(extra["pt_stats_pace_factor"]),
        "team_clutch_pts_factor": float(extra["team_clutch_pts_factor"]),
        "team_clutch_ast_factor": float(extra["team_clutch_ast_factor"]),
        "lineup_viz_opp_def_factor": float(extra["lineup_viz_opp_def_factor"]),
        "shot_chart_detail_fg3_factor": float(extra["shot_chart_detail_fg3_factor"]),
        "boxscore_defensive_opp_factor": float(extra["boxscore_defensive_opp_factor"]),
        "extra_endpoint_coverage": float(extra["extra_endpoint_coverage"]),
        "team_est_pace_factor": float(advanced["team_est_pace_factor"]),
        "opp_est_def_factor": float(advanced["opp_est_def_factor"]),
        "player_est_impact_factor": float(advanced["player_est_impact_factor"]),
        "season_matchup_factor": float(season_matchup["season_matchup_factor"]),
        "season_matchup_defender_share": float(season_matchup["season_matchup_defender_share"]),
        "pt_defend_pts_factor": float(pt_def["pt_defend_pts_factor"]),
        "pt_defend_fg3m_factor": float(pt_def["pt_defend_fg3m_factor"]),
        "opp_pressure_pts_factor": float(hustle_pressure["opp_pressure_pts_factor"]),
        "opp_pressure_ast_factor": float(hustle_pressure["opp_pressure_ast_factor"]),
        "opp_pressure_fg3m_factor": float(hustle_pressure["opp_pressure_fg3m_factor"]),
        "defender_discipline_pts_factor": float(defender_discipline["defender_discipline_pts_factor"]),
        "defender_discipline_ast_factor": float(defender_discipline["defender_discipline_ast_factor"]),
        "defender_discipline_fg3m_factor": float(defender_discipline["defender_discipline_fg3m_factor"]),
        "defender_discipline_fta_factor": float(defender_discipline["defender_discipline_fta_factor"]),
        "opp_lineup_pts_factor": float(opp_lineup_weak["opp_lineup_pts_factor"]),
        "opp_lineup_fg3m_factor": float(opp_lineup_weak["opp_lineup_fg3m_factor"]),
        "opp_lineup_ast_factor": float(opp_lineup_weak["opp_lineup_ast_factor"]),
        "opp_lineup_reb_factor": float(opp_lineup_weak["opp_lineup_reb_factor"]),
        "onoff_summary_usage_factor": float(onoff_summary["onoff_summary_usage_factor"]),
        "onoff_summary_ast_factor": float(onoff_summary["onoff_summary_ast_factor"]),
        "onoff_summary_reb_factor": float(onoff_summary["onoff_summary_reb_factor"]),
        "onoff_summary_pace_factor": float(onoff_summary["onoff_summary_pace_factor"]),
        "onoff_summary_pts_market_factor": float(onoff_summary["onoff_summary_pts_market_factor"]),
        "shot_style_pts_factor": float(shot_style["shot_style_pts_factor"]),
        "shot_style_fg3m_factor": float(shot_style["shot_style_fg3m_factor"]),
        "shot_style_ast_factor": float(shot_style["shot_style_ast_factor"]),
        "fg3m_catch_shoot_factor": float(fg3m_focus["fg3m_catch_shoot_factor"]),
        "fg3m_pullup_factor": float(fg3m_focus["fg3m_pullup_factor"]),
        "fg3m_transition_factor": float(fg3m_focus["fg3m_transition_factor"]),
        "fg3m_zone_matchup_factor": float(fg3m_focus["fg3m_zone_matchup_factor"]),
        "fg3m_team_traditional_factor": float(fg3m_focus["fg3m_team_traditional_factor"]),
        "fg3m_lineup_traditional_factor": float(fg3m_focus["fg3m_lineup_traditional_factor"]),
        "spatial_pts_factor": float(spatial["spatial_pts_factor"]),
        "spatial_fg3m_factor": float(spatial["spatial_fg3m_factor"]),
        "spatial_reb_factor": float(spatial["spatial_reb_factor"]),
        "recent_def_pts_factor": float(recent_def["recent_def_pts_factor"]),
        "recent_def_fg3m_factor": float(recent_def["recent_def_fg3m_factor"]),
        "recent_def_ast_factor": float(recent_def["recent_def_ast_factor"]),
        "recent_def_reb_factor": float(recent_def["recent_def_reb_factor"]),
        "hist_fg3a_volume_factor": float(hist_fg3["hist_fg3a_volume_factor"]),
        "hist_fg3m_matchup_factor": float(hist_fg3["hist_fg3m_matchup_factor"]),
        "hist_fg3m_defender_factor": float(hist_fg3["hist_fg3m_defender_factor"]),
        "hist_fg3m_team_allowed_factor": float(hist_fg3["hist_fg3m_team_allowed_factor"]),
        "hist_fg3m_zone_mix_factor": float(hist_fg3["hist_fg3m_zone_mix_factor"]),
        "hist_fg3m_factor": float(hist_fg3["hist_fg3m_factor"]),
        "hist_fg3_sample_weight": float(hist_fg3.get("hist_fg3_sample_weight") or 0.0),
        "hist_fg3_matchup_3pa": float(hist_fg3.get("hist_fg3_matchup_3pa") or 0.0),
        "hist_fg3_team_3pa": float(hist_fg3.get("hist_fg3_team_3pa") or 0.0),
        "matchup_persistence_pts_factor": float(matchup_switch["matchup_persistence_pts_factor"]),
        "matchup_persistence_fg3m_factor": float(matchup_switch["matchup_persistence_fg3m_factor"]),
        "matchup_persistence_ast_factor": float(matchup_switch["matchup_persistence_ast_factor"]),
        "matchup_switch_var_factor": float(matchup_switch["matchup_switch_var_factor"]),
        "pt_pass_ast_factor": float(tracking["pt_pass_ast_factor"]),
        "pt_reb_factor": float(tracking["pt_reb_factor"]),
        "game_day_minutes_factor": float(game_day["game_day_minutes_factor"]),
        "game_day_usage_factor": float(game_day["game_day_usage_factor"]),
        "game_day_opp_factor": float(game_day["game_day_opp_factor"]),
        "pbp_blowout_minutes_factor": float(pbp["pbp_blowout_minutes_factor"]),
        "pbp_blowout_var_factor": float(pbp["pbp_blowout_var_factor"]),
        "synergy_pts_factor": float(synergy["synergy_pts_factor"]),
        "synergy_fg3m_factor": float(synergy["synergy_fg3m_factor"]),
        "synergy_ast_factor": float(synergy["synergy_ast_factor"]),
        "position_defense_pts_factor": float(position_def["pts"]),
        "position_defense_fg3m_factor": float(position_def["fg3m"]),
        "position_defense_ast_factor": float(position_def["ast"]),
        "position_defense_reb_factor": float(position_def["reb"]),
        "primary_defender_pts_factor": float(primary_def["pts"]),
        "primary_defender_fg3m_factor": float(primary_def["fg3m"]),
        "primary_defender_ast_factor": float(primary_def["ast"]),
        "primary_defender_reb_factor": float(primary_def["reb"]),
        "primary_defender_share": float(primary_def.get("primary_defender_share") or 0.0),
        "primary_defender_entropy": float(primary_def.get("defender_entropy") or 0.0),
        "rebound_environment_factor": float(reb_env),
        "foul_minutes_factor": float(foul["foul_minutes_factor"]),
        "foul_var_factor": float(foul["foul_var_factor"]),
        "foul_risk_score": float(foul["foul_risk_score"]),
        "blowout_minutes_factor": blowout_minutes_factor,
        "blowout_var_factor": blowout_var_factor,
    }
    if primary_def.get("primary_defender_id"):
        out["primary_defender_id"] = str(primary_def["primary_defender_id"])
    if primary_def.get("primary_defender_name"):
        out["primary_defender_name"] = str(primary_def["primary_defender_name"])

    reliability_endpoints = [
        "teamplayeronoffdetails",
        "teamdashlineups",
        "leaguedashlineups",
        "teamestimatedmetrics",
        "playerestimatedmetrics",
        "leagueseasonmatchups",
        "matchupsrollup",
        "leaguedashptdefend",
        "playerdashptshotdefend",
        "leaguehustlestatsteam",
        "leaguehustlestatsplayer",
        "boxscorehustlev2",
        "teamplayeronoffsummary",
        "teamdashptshots",
        "leaguedashplayerptshot",
        "leaguedashoppptshot",
        "leaguedashteamptshot",
        "leaguedashplayershotlocations",
        "leaguedashteamshotlocations",
        "leaguedashptteamdefend",
        "assisttracker",
        "leaguedashptstats",
        "leaguedashteamclutch",
        "leaguedashplayerclutch",
        "playerdashboardbyclutch",
        "leaguelineupviz",
        "shotchartdetail",
        "shotchartleaguewide",
        "boxscoredefensivev2",
        "playbyplayv3",
        "winprobabilitypbp",
        "scoreboardv2",
        "synergyplaytypes",
        "gamerotation",
        "boxscorematchupsv3",
    ]
    endpoint_reliability = _endpoint_reliability_score(cache, reliability_endpoints)
    signal_coverage = _context_signal_coverage(out)
    context_quality = float(
        safe_clip(endpoint_reliability * 0.58 + signal_coverage * 0.42, 0.35, 1.0)
    )
    context_strength = float(safe_clip(0.50 + context_quality * 0.50, 0.50, 1.0))

    for key, value in list(out.items()):
        if not isinstance(value, (int, float)):
            continue
        if not str(key).endswith("_factor"):
            continue
        out[key] = float(_blend_to_neutral(float(value), context_strength))

    out["endpoint_reliability"] = float(endpoint_reliability)
    out["context_signal_coverage"] = float(signal_coverage)
    out["lineup_context_quality"] = float(context_quality)
    out["lineup_context_strength"] = float(context_strength)

    out["pts_market_factor"] = float(
        safe_clip(
            out["position_defense_pts_factor"]
            * out["primary_defender_pts_factor"]
            * out["defender_discipline_pts_factor"]
            * out["opp_est_def_factor"]
            * out["pt_defend_pts_factor"]
            * out["opp_pressure_pts_factor"]
            * out["opp_lineup_pts_factor"]
            * out["season_matchup_factor"]
            * out["synergy_pts_factor"]
            * out["shot_style_pts_factor"]
            * out["spatial_pts_factor"]
            * out["recent_def_pts_factor"]
            * out["matchup_persistence_pts_factor"]
            * out["defender_discipline_fta_factor"]
            * out["onoff_summary_pts_market_factor"]
            * out["lineup_vs_lineup_pts_factor"]
            * out["team_clutch_pts_factor"]
            * out["lineup_viz_opp_def_factor"]
            * out["boxscore_defensive_opp_factor"]
            * out["game_day_opp_factor"],
            0.78,
            1.26,
        )
    )
    out["fg3m_market_factor"] = float(
        safe_clip(
            out["position_defense_fg3m_factor"]
            * out["primary_defender_fg3m_factor"]
            * out["defender_discipline_fg3m_factor"]
            * out["opp_est_def_factor"]
            * out["pt_defend_fg3m_factor"]
            * out["opp_pressure_fg3m_factor"]
            * out["opp_lineup_fg3m_factor"]
            * out["season_matchup_factor"]
            * out["synergy_fg3m_factor"]
            * out["shot_style_fg3m_factor"]
            * out["fg3m_zone_matchup_factor"]
            * out["spatial_fg3m_factor"]
            * out["recent_def_fg3m_factor"]
            * out["matchup_persistence_fg3m_factor"]
            * out["hist_fg3m_factor"]
            * out["lineup_vs_lineup_fg3m_factor"]
            * out["shot_chart_detail_fg3_factor"]
            * out["lineup_viz_opp_def_factor"]
            * out["game_day_opp_factor"],
            0.78,
            1.26,
        )
    )
    out["ast_market_factor"] = float(
        safe_clip(
            out["position_defense_ast_factor"]
            * out["primary_defender_ast_factor"]
            * out["defender_discipline_ast_factor"]
            * out["opp_est_def_factor"]
            * out["opp_pressure_ast_factor"]
            * out["opp_lineup_ast_factor"]
            * out["season_matchup_factor"]
            * out["synergy_ast_factor"]
            * out["shot_style_ast_factor"]
            * out["recent_def_ast_factor"]
            * out["matchup_persistence_ast_factor"]
            * out["pt_pass_ast_factor"]
            * out["lineup_vs_lineup_ast_factor"]
            * out["assist_tracker_ast_factor"]
            * out["team_clutch_ast_factor"]
            * out["lineup_viz_opp_def_factor"]
            * out["game_day_opp_factor"],
            0.78,
            1.28,
        )
    )
    out["reb_market_factor"] = float(
        safe_clip(
            out["position_defense_reb_factor"]
            * out["primary_defender_reb_factor"]
            * out["rebound_environment_factor"]
            * out["opp_lineup_reb_factor"]
            * out["spatial_reb_factor"]
            * out["recent_def_reb_factor"]
            * out["lineup_vs_lineup_reb_factor"]
            * out["boxscore_defensive_opp_factor"],
            0.82,
            1.22,
        )
    )
    out["reb_market_factor"] = float(
        safe_clip(
            out["reb_market_factor"]
            * out["opp_est_def_factor"]
            * out["pt_reb_factor"]
            * out["game_day_opp_factor"],
            0.80,
            1.24,
        )
    )
    return out
