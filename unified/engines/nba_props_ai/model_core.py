from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import SQLiteCache
from .feature_gate import apply_factor_gate
from .matchup_lineup_context import build_matchup_lineup_context
from .nba_data import (
    find_player,
    find_player_id,
    find_team_by_abbr,
    team_name_from_abbr,
    get_player_gamelog,
    get_team_advanced,
    get_team_opponent_pergame,
    get_team_base_pergame,
    get_player_zone_splits,
    get_player_pt_shots,
    get_opponent_shot_locations,
    get_opponent_ptshot_defense,
    get_league_player_stats,
    enrich_rebound_columns,
    get_boxscore_player_track,
    get_boxscore_usage,
    get_boxscore_scoring,
    get_boxscore_matchups,
    get_game_rotation,
)
from .on_off_context import estimate_usage_boost
from .splits_context import PlayerSplitsContext
from .utils import eb_shrink_rate, safe_clip

def recency_weights(n: int, decay: float = 0.93) -> np.ndarray:
    w = np.array([decay ** i for i in range(n)], dtype=float)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / max(n, 1)

def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    mu = float(np.average(x, weights=w))
    var = float(np.average((x - mu) ** 2, weights=w))
    return mu, max(var, 1e-8)

def _normalize_col_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    mapping = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in mapping:
            return mapping[key]
    return None

def _coerce_float(v: object) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out

def _player_row(df: pd.DataFrame, player_id: int) -> Optional[pd.Series]:
    if len(df) == 0:
        return None
    pid_col = _find_col(df, ["PLAYER_ID", "PERSON_ID", "personId", "personid", "playerId"])
    if not pid_col:
        return None
    rows = df[df[pid_col].astype(str) == str(player_id)]
    if len(rows) == 0:
        return None
    return rows.iloc[0]

def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return float(num / den)

def _single_game_role_features(player_id: int, game_id: str, cache: SQLiteCache) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # BoxScorePlayerTrackV3
    track_df = get_boxscore_player_track(game_id, cache)
    row = _player_row(track_df, player_id)
    if row is not None:
        min_col = _find_col(track_df, ["MIN", "minutes", "minutesCalculated", "minutesPlayed"])
        touches_col = _find_col(track_df, ["TOUCHES", "touches"])
        p_ast_col = _find_col(track_df, ["POTENTIAL_AST", "potentialAssists", "passesLeadingToScore"])
        reb_total_col = _find_col(track_df, ["REB_CHANCES", "reboundChances", "reboundChancesTotal"])
        reb_oreb_col = _find_col(track_df, ["OREB_CHANCES", "reboundChancesOffensive", "offensiveReboundChances"])
        reb_dreb_col = _find_col(track_df, ["DREB_CHANCES", "reboundChancesDefensive", "defensiveReboundChances"])

        mins = _coerce_float(row[min_col]) if min_col else None
        touches = _coerce_float(row[touches_col]) if touches_col else None
        potential_ast = _coerce_float(row[p_ast_col]) if p_ast_col else None
        reb_chances = _coerce_float(row[reb_total_col]) if reb_total_col else None
        if reb_chances is None:
            r1 = _coerce_float(row[reb_oreb_col]) if reb_oreb_col else None
            r2 = _coerce_float(row[reb_dreb_col]) if reb_dreb_col else None
            if r1 is not None or r2 is not None:
                reb_chances = float((r1 or 0.0) + (r2 or 0.0))

        tpm = _safe_ratio(touches, mins)
        papm = _safe_ratio(potential_ast, mins)
        rcpm = _safe_ratio(reb_chances, mins)
        if tpm is not None:
            out["touches_pm"] = tpm
        if papm is not None:
            out["potential_ast_pm"] = papm
        if rcpm is not None:
            out["reb_chances_pm"] = rcpm

    # BoxScoreUsageV3
    usage_df = get_boxscore_usage(game_id, cache)
    row = _player_row(usage_df, player_id)
    if row is not None:
        usage_col = _find_col(usage_df, ["USG_PCT", "usagePercentage", "usage"])
        usage = _coerce_float(row[usage_col]) if usage_col else None
        if usage is not None:
            if usage <= 1.0:
                usage *= 100.0
            out["usage_pct"] = usage

    # BoxScoreScoringV3
    scoring_df = get_boxscore_scoring(game_id, cache)
    row = _player_row(scoring_df, player_id)
    if row is not None:
        unast_pts_col = _find_col(scoring_df, ["PCT_UAST_PTS", "percentagePointsUnassisted"])
        unast_fg_col = _find_col(scoring_df, ["PCT_UAST_FGM", "percentageFieldGoalsUnassisted"])
        unast_pts = _coerce_float(row[unast_pts_col]) if unast_pts_col else None
        unast_fg = _coerce_float(row[unast_fg_col]) if unast_fg_col else None
        chosen = unast_pts if unast_pts is not None else unast_fg
        if chosen is not None:
            if chosen > 1.0:
                chosen /= 100.0
            out["unassisted_share"] = safe_clip(chosen, 0.0, 1.0)

    # BoxScoreMatchupsV3
    matchup_df = get_boxscore_matchups(game_id, cache)
    if len(matchup_df):
        off_col = _find_col(matchup_df, ["OFF_PLAYER_ID", "offPersonId", "offPlayerId", "offplayerid"])
        def_col = _find_col(matchup_df, ["DEF_PLAYER_ID", "defPersonId", "defPlayerId", "defplayerid"])
        mins_col = _find_col(matchup_df, ["MATCHUP_MIN", "matchupMinutes", "minutes", "partialPossessions"])
        if off_col and def_col and mins_col:
            rows = matchup_df[matchup_df[off_col].astype(str) == str(player_id)].copy()
            if len(rows):
                rows["_w"] = pd.to_numeric(rows[mins_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                total = float(rows["_w"].sum())
                if total > 0:
                    by_def = rows.groupby(def_col, dropna=False)["_w"].sum()
                    out["defender_concentration"] = float(by_def.max() / total)

    # GameRotation
    rot = get_game_rotation(game_id, cache)
    rot_df = pd.concat([rot.get("home", pd.DataFrame()), rot.get("away", pd.DataFrame())], ignore_index=True)
    if len(rot_df):
        pid_col = _find_col(rot_df, ["PERSON_ID", "PLAYER_ID", "personId", "playerId"])
        if pid_col:
            rows = rot_df[rot_df[pid_col].astype(str) == str(player_id)]
            if len(rows):
                out["stints"] = float(len(rows))

    return out

def _recent_role_context(player_id: int, game_log: pd.DataFrame, cache: SQLiteCache, n_games: int = 5) -> Dict[str, float]:
    out = {
        "touch_fac": 1.0,
        "usage_fac": 1.0,
        "ast_creation_fac": 1.0,
        "reb_chance_fac": 1.0,
        "self_create_fac": 1.0,
        "minutes_var_fac": 1.0,
        "pts_role_fac": 1.0,
        "fg3m_role_fac": 1.0,
        "ast_role_fac": 1.0,
        "reb_role_fac": 1.0,
        "role_samples": 0.0,
    }
    gid_col = _find_col(game_log, ["Game_ID", "GAME_ID"])
    if not gid_col:
        return out
    game_ids = [str(x) for x in game_log[gid_col].head(max(n_games, 1)).tolist() if str(x).strip()]
    if not game_ids:
        return out

    series: Dict[str, List[float]] = {}
    for gid in game_ids:
        feat = _single_game_role_features(player_id, gid, cache)
        if not feat:
            continue
        for k, v in feat.items():
            series.setdefault(k, []).append(float(v))
    if not series:
        return out

    out["role_samples"] = float(max(len(v) for v in series.values()))
    for key, vals in series.items():
        w = recency_weights(len(vals))
        out[key] = float(np.average(np.array(vals, dtype=float), weights=w))

    touch_pm = out.get("touches_pm")
    if touch_pm is not None:
        out["touch_fac"] = safe_clip(1.0 + (((touch_pm / 2.00) - 1.0) * 0.12), 0.92, 1.10)
    usage_pct = out.get("usage_pct")
    if usage_pct is not None:
        out["usage_fac"] = safe_clip(1.0 + (((usage_pct / 22.0) - 1.0) * 0.22), 0.90, 1.14)
    ast_pm = out.get("potential_ast_pm")
    if ast_pm is not None:
        out["ast_creation_fac"] = safe_clip(1.0 + (((ast_pm / 0.22) - 1.0) * 0.20), 0.88, 1.18)
    reb_pm = out.get("reb_chances_pm")
    if reb_pm is not None:
        out["reb_chance_fac"] = safe_clip(1.0 + (((reb_pm / 0.35) - 1.0) * 0.22), 0.88, 1.18)
    self_create = out.get("unassisted_share")
    if self_create is not None:
        out["self_create_fac"] = safe_clip(1.0 + (((self_create / 0.45) - 1.0) * 0.10), 0.94, 1.08)

    var_fac = 1.0
    stints = out.get("stints")
    if stints is not None:
        if stints >= 9:
            var_fac *= 1.12
        elif stints <= 4:
            var_fac *= 0.95
    def_conc = out.get("defender_concentration")
    if def_conc is not None:
        if def_conc < 0.45:
            var_fac *= 1.06
        elif def_conc > 0.75:
            var_fac *= 0.96
    out["minutes_var_fac"] = safe_clip(var_fac, 0.90, 1.20)

    out["pts_role_fac"] = safe_clip(0.45 * out["usage_fac"] + 0.35 * out["touch_fac"] + 0.20 * out["self_create_fac"], 0.90, 1.14)
    out["fg3m_role_fac"] = safe_clip(0.55 * out["usage_fac"] + 0.45 * out["touch_fac"], 0.90, 1.13)
    out["ast_role_fac"] = safe_clip(0.55 * out["ast_creation_fac"] + 0.25 * out["usage_fac"] + 0.20 * out["touch_fac"], 0.88, 1.18)
    out["reb_role_fac"] = safe_clip(0.65 * out["reb_chance_fac"] + 0.35 * out["touch_fac"], 0.90, 1.15)
    return out

def _pace_factor(team_a_name: str, team_b_name: str, adv_df: pd.DataFrame) -> float:
    lg = float(adv_df["PACE"].mean()) if "PACE" in adv_df.columns else 100.0
    a = adv_df[adv_df["TEAM_NAME"] == team_a_name]
    b = adv_df[adv_df["TEAM_NAME"] == team_b_name]
    if len(a) == 0 or len(b) == 0 or lg <= 0:
        return 1.0
    exp = (float(a.iloc[0]["PACE"]) + float(b.iloc[0]["PACE"])) / 2.0
    return exp / lg

def _count_from_min_rate(mu_min: float, var_min: float, mu_rate: float, var_rate: float, scale: float = 1.0) -> Tuple[float, float]:
    mu = mu_min * mu_rate * scale
    var = (mu_min ** 2) * var_rate + (mu_rate ** 2) * var_min + var_min * var_rate
    var *= scale ** 2
    return float(mu), float(max(var, 1e-8))

def _rate_model(game_log: pd.DataFrame, num_col: str, den_col: str = "MIN", n_use: int = 25) -> Tuple[float, float]:
    df = game_log.head(n_use).copy()
    if num_col not in df.columns or den_col not in df.columns:
        return 0.0, 1e-8
    num = df[num_col].astype(float).values
    den = df[den_col].astype(float).values
    den = np.where(den <= 0.1, np.nan, den)
    rate = num / den
    rate = rate[~np.isnan(rate)]
    if len(rate) == 0:
        return 0.0, 1e-8
    w = recency_weights(len(rate))
    mu, var = _weighted_mean_var(rate, w)
    return max(mu, 0.0), max(var, 1e-8)


def _ewma_recent(values: np.ndarray, span: int) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    # game_log is most-recent first; EWMA should roll oldest -> newest
    seq = arr[::-1]
    alpha = 2.0 / float(max(span, 1) + 1.0)
    ewma = float(seq[0])
    for val in seq[1:]:
        ewma = float(alpha * float(val) + (1.0 - alpha) * ewma)
    return ewma


def _slope_recent(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return 0.0
    seq = arr[::-1]
    x = np.arange(len(seq), dtype=float)
    try:
        slope = float(np.polyfit(x, seq, 1)[0])
    except Exception:
        slope = 0.0
    if not np.isfinite(slope):
        return 0.0
    return slope


def _market_trend_context(game_log: pd.DataFrame, market: str) -> Dict[str, float]:
    out = {
        "minutes_trend_factor": 1.0,
        "minutes_var_trend_factor": 1.0,
        "market_trend_factor": 1.0,
        "minutes_trend_samples": 0.0,
        "market_trend_samples": 0.0,
        "minutes_recent": 0.0,
        "minutes_base": 0.0,
        "market_rate_recent": 0.0,
        "market_rate_base": 0.0,
    }
    if len(game_log) == 0:
        return out

    min_col = "MIN" if "MIN" in game_log.columns else _find_col(game_log, ["MIN", "minutes"])
    if min_col:
        mins = pd.to_numeric(game_log[min_col], errors="coerce").dropna().head(14).astype(float).values
    else:
        mins = np.array([], dtype=float)
    if len(mins):
        recent = _ewma_recent(mins, span=3)
        base = _ewma_recent(mins, span=9)
        slope = _slope_recent(mins)
        if recent is not None and base is not None and base > 0:
            reliability = min(len(mins) / 12.0, 1.0)
            ratio = recent / base
            slope_norm = slope / max(base, 1.0)
            shift = (ratio - 1.0) * 0.40 + slope_norm * 0.25
            vol = abs(ratio - 1.0) * 0.25 + abs(slope_norm) * 0.35
            out["minutes_trend_factor"] = float(
                safe_clip(1.0 + shift * reliability, 0.90, 1.10)
            )
            out["minutes_var_trend_factor"] = float(
                safe_clip(1.0 + vol * reliability, 0.95, 1.22)
            )
            out["minutes_trend_samples"] = float(len(mins))
            out["minutes_recent"] = float(recent)
            out["minutes_base"] = float(base)

    stat_col = {
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "FG3M": "FG3M",
    }.get(str(market).upper())
    if not stat_col or stat_col not in game_log.columns or len(mins) == 0:
        return out

    use = game_log[[stat_col, min_col]].copy() if min_col else pd.DataFrame(columns=[stat_col, "MIN"])
    if len(use) == 0:
        return out
    num = pd.to_numeric(use[stat_col], errors="coerce")
    den = pd.to_numeric(use[min_col], errors="coerce")
    rate = (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().head(16)
    if len(rate) < 4:
        return out
    arr = rate.astype(float).values
    recent_rate = _ewma_recent(arr, span=3)
    base_rate = _ewma_recent(arr, span=8)
    slope_rate = _slope_recent(arr)
    if recent_rate is None or base_rate is None or base_rate <= 0:
        return out
    reliability = min(len(arr) / 12.0, 1.0)
    ratio = recent_rate / base_rate
    slope_norm = slope_rate / max(base_rate, 1e-6)
    shift = (ratio - 1.0) * 0.45 + slope_norm * 0.22
    out["market_trend_factor"] = float(safe_clip(1.0 + shift * reliability, 0.90, 1.14))
    out["market_trend_samples"] = float(len(arr))
    out["market_rate_recent"] = float(recent_rate)
    out["market_rate_base"] = float(base_rate)
    return out

def _minutes_model(game_log: pd.DataFrame, override: Optional[Tuple[str, float]] = None, spread: Optional[float] = None) -> Tuple[float, float, List[str]]:
    flags: List[str] = []
    mins = game_log.head(20)["MIN"].astype(float).values
    w = recency_weights(len(mins))
    mu, var = _weighted_mean_var(mins, w)
    recent_mu = float(np.mean(mins[:5])) if len(mins) >= 5 else mu

    # Role-based variance floors
    if mu >= 32:
        var = max(var, 5.0)
    elif mu >= 24:
        var = max(var, 7.0)
    else:
        var = max(var, 9.0)

    # Recent volatility matters
    if abs(recent_mu - mu) >= 4.0:
        var *= 1.15
        flags.append("minutes_recently_volatile")

    # Blowout heuristic from spread (negative = player's team favored)
    if spread is not None:
        if spread <= -10 and mu >= 30:
            mu *= 0.97
            var *= 1.15
            flags.append("blowout_risk_favorite")
        elif spread >= 10:
            var *= 1.10
            flags.append("blowout_risk_underdog")

    if override:
        kind, val = override
        flags.append(f"minutes_override:{kind}:{val:g}")
        if kind == "target":
            mu = float(val)
            var = max(var * 0.30, 2.25)
        elif kind == "cap":
            mu = min(mu, float(val))
            var = max(var * 0.50, 2.25)
    return float(mu), float(var), flags

def _compute_rest_days(game_log: pd.DataFrame, game_date_local: Optional[str]) -> Optional[int]:
    if "GAME_DATE_PARSED" not in game_log.columns or len(game_log) == 0:
        return None
    last_game = pd.to_datetime(game_log.iloc[0]["GAME_DATE_PARSED"], errors="coerce")
    if pd.isna(last_game):
        return None
    if game_date_local:
        try:
            target = pd.to_datetime(game_date_local)
            return max(int((target.normalize() - last_game.normalize()).days), 0)
        except Exception:
            pass
    if "DAYS_REST" in game_log.columns:
        try:
            return int(float(game_log["DAYS_REST"].iloc[0]))
        except Exception:
            return None
    return None

def _player_shot_profile(player_id: int, team_id: int, season: str, cache: SQLiteCache) -> Dict[str, float]:
    params = {"player_id": player_id, "team_id": team_id, "season": season}
    import json
    hit = cache.get("player_shot_profile", params)
    if hit:
        return json.loads(hit.data_json)

    zone_df = get_player_zone_splits(player_id, season, cache)
    pt_df = get_player_pt_shots(player_id, team_id, season, cache)

    out = {
        "corner3_share": 0.0,
        "ab3_share": 0.0,
        "rim_share": 0.0,
        "cs_share": 0.0,
        "pullup_share": 0.0,
    }

    # Zone shares
    total_fga = 0.0
    corner_fga = 0.0
    ab3_fga = 0.0
    rim_fga = 0.0
    if len(zone_df):
        gv_col = _find_col(zone_df, ["GROUP_VALUE"])
        fga_col = _find_col(zone_df, ["FGA"])
        if gv_col and fga_col:
            for _, row in zone_df.iterrows():
                zone = str(row[gv_col])
                fga = float(row[fga_col] or 0)
                total_fga += fga
                z = zone.lower()
                if "corner 3" in z:
                    corner_fga += fga
                elif "above the break 3" in z:
                    ab3_fga += fga
                elif "restricted area" in z:
                    rim_fga += fga
    if total_fga > 0:
        out["corner3_share"] = corner_fga / total_fga
        out["ab3_share"] = ab3_fga / total_fga
        out["rim_share"] = rim_fga / total_fga

    # Tracking shares
    total_track_fga = 0.0
    cs_fga = 0.0
    pullup_fga = 0.0
    if len(pt_df):
        type_col = _find_col(pt_df, ["SHOT_TYPE"])
        fga_col = _find_col(pt_df, ["FGA"])
        if type_col and fga_col:
            for _, row in pt_df.iterrows():
                stype = str(row[type_col]).lower()
                fga = float(row[fga_col] or 0)
                total_track_fga += fga
                if "catch" in stype:
                    cs_fga += fga
                elif "pull" in stype:
                    pullup_fga += fga
    if total_track_fga > 0:
        out["cs_share"] = cs_fga / total_track_fga
        out["pullup_share"] = pullup_fga / total_track_fga

    cache.set("player_shot_profile", params, out)
    return out

def _player_efficiencies(game_log: pd.DataFrame) -> Dict[str, float]:
    df = game_log.head(35).copy()
    fg3a = df["FG3A"].astype(float) if "FG3A" in df.columns else pd.Series([0]*len(df))
    fg3m = df["FG3M"].astype(float) if "FG3M" in df.columns else pd.Series([0]*len(df))
    fga = df["FGA"].astype(float) if "FGA" in df.columns else pd.Series([0]*len(df))
    fgm = df["FGM"].astype(float) if "FGM" in df.columns else pd.Series([0]*len(df))
    fta = df["FTA"].astype(float) if "FTA" in df.columns else pd.Series([0]*len(df))
    ftm = df["FTM"].astype(float) if "FTM" in df.columns else pd.Series([0]*len(df))

    total_3a = float(fg3a.sum())
    total_3m = float(fg3m.sum())
    total_2a = float((fga - fg3a).clip(lower=0).sum())
    total_2m = float((fgm - fg3m).clip(lower=0).sum())
    total_fta = float(fta.sum())
    total_ftm = float(ftm.sum())

    p3 = eb_shrink_rate((total_3m / total_3a) if total_3a > 0 else 0.36, total_3a, 0.36, 120.0)
    p2 = eb_shrink_rate((total_2m / total_2a) if total_2a > 0 else 0.52, total_2a, 0.52, 160.0)
    ft = eb_shrink_rate((total_ftm / total_fta) if total_fta > 0 else 0.78, total_fta, 0.78, 80.0)

    return {
        "p2": safe_clip(p2, 0.40, 0.65),
        "p3": safe_clip(p3, 0.25, 0.48),
        "ft": safe_clip(ft, 0.60, 0.92),
        "n_2a": total_2a,
        "n_3a": total_3a,
        "n_fta": total_fta,
    }

def _opp_zone_factor(shotloc_df: pd.DataFrame, opp_team_name: str, metric: str, shrink: float = 0.45) -> float:
    # metric examples: "Corner 3 FGA", "Above the Break 3 FG%", "Restricted Area FGA"
    col = _find_col(shotloc_df, [metric])
    if not col:
        return 1.0
    row = shotloc_df[shotloc_df["TEAM_NAME"] == opp_team_name]
    if len(row) == 0:
        return 1.0
    lg = float(shotloc_df[col].mean())
    if lg <= 0:
        return 1.0
    raw = float(row.iloc[0][col]) / lg
    return safe_clip(1.0 + (raw - 1.0) * shrink, 0.80, 1.25)

def _opp_ptshot_factor(df: pd.DataFrame, opp_team_name: str, col_name: str = "FGA", shrink: float = 0.25) -> float:
    if len(df) == 0 or "TEAM_NAME" not in df.columns or col_name not in df.columns:
        return 1.0
    row = df[df["TEAM_NAME"] == opp_team_name]
    if len(row) == 0:
        return 1.0
    lg = float(df[col_name].mean())
    if lg <= 0:
        return 1.0
    raw = float(row.iloc[0][col_name]) / lg
    return safe_clip(1.0 + (raw - 1.0) * shrink, 0.85, 1.15)

def _team_total_factor(team_base_df: pd.DataFrame, team_name: str, vegas_team_total: Optional[float]) -> float:
    if vegas_team_total is None or "TEAM_NAME" not in team_base_df.columns:
        return 1.0
    row = team_base_df[team_base_df["TEAM_NAME"] == team_name]
    if len(row) == 0 or "PTS" not in row.columns:
        return 1.0
    season_ppg = float(row.iloc[0]["PTS"])
    if season_ppg <= 0:
        return 1.0
    raw = vegas_team_total / season_ppg
    return safe_clip(1.0 + (raw - 1.0) * 0.45, 0.90, 1.10)

def _opponent_absence_factor(league_df: pd.DataFrame, opponent_out: List[str], market: str) -> float:
    if not opponent_out or len(league_df) == 0 or "PLAYER_NAME" not in league_df.columns:
        return 1.0
    factor = 1.0
    for name in opponent_out:
        rows = league_df[league_df["PLAYER_NAME"].astype(str).str.lower() == name.lower()]
        if len(rows) == 0:
            rows = league_df[league_df["PLAYER_NAME"].astype(str).str.lower().str.contains(name.lower(), regex=False)]
        if len(rows) == 0:
            continue
        r = rows.sort_values("MIN", ascending=False).iloc[0]
        mpg = float(r.get("MIN", 0) or 0)
        blk = float(r.get("BLK", 0) or 0)
        stl = float(r.get("STL", 0) or 0)
        reb = float(r.get("REB", 0) or 0)
        weight = safe_clip(mpg / 32.0, 0.0, 1.0)
        if market in {"PTS", "FG3M"}:
            if blk >= 1.4:
                factor += 0.03 * weight
            if stl >= 1.2:
                factor += 0.02 * weight
        if market == "REB" and reb >= 8.0:
            factor += 0.03 * weight
        if market == "AST" and stl >= 1.2:
            factor += 0.01 * weight
    return safe_clip(factor, 0.95, 1.12)


def _vacated_opportunity_factors(league_df: pd.DataFrame, team_out: List[str]) -> Dict[str, float]:
    out = {
        "usage_factor": 1.0,
        "fg3a_factor": 1.0,
        "fta_factor": 1.0,
        "ast_factor": 1.0,
        "reb_factor": 1.0,
        "vacated_usage_proxy": 0.0,
        "vacated_minutes": 0.0,
    }
    if len(league_df) == 0 or not team_out:
        return out
    name_col = _find_col(league_df, ["PLAYER_NAME", "PLAYER"])
    min_col = _find_col(league_df, ["MIN", "minutes"])
    fga_col = _find_col(league_df, ["FGA"])
    fta_col = _find_col(league_df, ["FTA"])
    tov_col = _find_col(league_df, ["TOV", "TO"])
    ast_col = _find_col(league_df, ["AST"])
    reb_col = _find_col(league_df, ["REB"])
    fg3a_col = _find_col(league_df, ["FG3A"])
    if not name_col:
        return out

    v_usage = 0.0
    v_ast = 0.0
    v_reb = 0.0
    v_fg3a = 0.0
    v_fta = 0.0
    v_min = 0.0
    names = league_df[name_col].astype(str)
    for name in team_out:
        row = league_df[names.str.lower() == str(name).lower()]
        if len(row) == 0:
            row = league_df[names.str.lower().str.contains(str(name).lower(), regex=False)]
        if len(row) == 0:
            continue
        r = row.sort_values(min_col, ascending=False).iloc[0] if min_col else row.iloc[0]
        mins = float(r.get(min_col, 0.0) or 0.0) if min_col else 0.0
        weight = safe_clip(mins / 34.0, 0.0, 1.2)
        fga = float(r.get(fga_col, 0.0) or 0.0) if fga_col else 0.0
        fta = float(r.get(fta_col, 0.0) or 0.0) if fta_col else 0.0
        tov = float(r.get(tov_col, 0.0) or 0.0) if tov_col else 0.0
        ast = float(r.get(ast_col, 0.0) or 0.0) if ast_col else 0.0
        reb = float(r.get(reb_col, 0.0) or 0.0) if reb_col else 0.0
        fg3a = float(r.get(fg3a_col, 0.0) or 0.0) if fg3a_col else 0.0

        usage_proxy = fga + 0.44 * fta + tov
        v_usage += usage_proxy * weight
        v_ast += ast * weight
        v_reb += reb * weight
        v_fg3a += fg3a * weight
        v_fta += fta * weight
        v_min += mins * weight

    out["vacated_usage_proxy"] = float(v_usage)
    out["vacated_minutes"] = float(v_min)

    usage_shift = safe_clip(v_usage / 22.0, 0.0, 1.8)
    ast_shift = safe_clip(v_ast / 8.0, 0.0, 1.8)
    reb_shift = safe_clip(v_reb / 10.0, 0.0, 1.8)
    fg3a_shift = safe_clip(v_fg3a / 6.0, 0.0, 1.8)
    fta_shift = safe_clip(v_fta / 6.0, 0.0, 1.8)

    out["usage_factor"] = float(safe_clip(1.0 + usage_shift * 0.06, 1.0, 1.12))
    out["ast_factor"] = float(safe_clip(1.0 + ast_shift * 0.05, 1.0, 1.10))
    out["reb_factor"] = float(safe_clip(1.0 + reb_shift * 0.04, 1.0, 1.08))
    out["fg3a_factor"] = float(safe_clip(1.0 + fg3a_shift * 0.05, 1.0, 1.10))
    out["fta_factor"] = float(safe_clip(1.0 + fta_shift * 0.05, 1.0, 1.10))
    return out


def _starter_rest_combo_factors(
    league_df: pd.DataFrame,
    team_id: int,
    target_player: str,
    team_out: List[str],
) -> Dict[str, float]:
    out = {
        "minutes_factor": 1.0,
        "minutes_var_factor": 1.0,
        "pts_usage_factor": 1.0,
        "ast_usage_factor": 1.0,
        "reb_usage_factor": 1.0,
        "fg3a_usage_factor": 1.0,
        "combo_concentration": 1.0,
        "starter_absent_count": 0.0,
    }
    if len(league_df) == 0 or not team_out:
        return out

    name_col = _find_col(league_df, ["PLAYER_NAME", "PLAYER"])
    team_col = _find_col(league_df, ["TEAM_ID", "teamId"])
    min_col = _find_col(league_df, ["MIN", "minutes"])
    fga_col = _find_col(league_df, ["FGA"])
    fta_col = _find_col(league_df, ["FTA"])
    tov_col = _find_col(league_df, ["TOV", "TO"])
    ast_col = _find_col(league_df, ["AST"])
    reb_col = _find_col(league_df, ["REB"])
    fg3a_col = _find_col(league_df, ["FG3A"])
    if not name_col or not min_col:
        return out

    rows = league_df.copy()
    if team_col:
        rows = rows[rows[team_col].astype(str) == str(int(team_id))]
    if len(rows) == 0:
        return out

    names = rows[name_col].astype(str)
    target_rows = rows[names.str.lower() == str(target_player).lower()]
    if len(target_rows) == 0:
        target_rows = rows[names.str.lower().str.contains(str(target_player).lower(), regex=False)]
    if len(target_rows) == 0:
        return out
    target_row = target_rows.sort_values(min_col, ascending=False).iloc[0]
    target_min = float(target_row.get(min_col, 0.0) or 0.0)
    target_weight = safe_clip(target_min / 34.0, 0.08, 0.55)

    team_sorted = rows.sort_values(min_col, ascending=False).head(10)
    top5 = team_sorted.head(5)
    top10_total = float(pd.to_numeric(team_sorted[min_col], errors="coerce").fillna(0.0).sum())
    top5_total = float(pd.to_numeric(top5[min_col], errors="coerce").fillna(0.0).sum())
    if top10_total > 0:
        concentration = safe_clip(top5_total / top10_total, 0.48, 0.82)
    else:
        concentration = 0.62
    # 1.0 means league-neutral concentration; >1 means tighter combos.
    combo_conc = float(safe_clip(1.0 + (concentration - 0.62) * 0.90, 0.90, 1.12))
    out["combo_concentration"] = combo_conc

    v_usage = 0.0
    v_ast = 0.0
    v_reb = 0.0
    v_fg3a = 0.0
    v_min = 0.0
    absent_count = 0

    for absent in team_out:
        absent_rows = rows[names.str.lower() == str(absent).lower()]
        if len(absent_rows) == 0:
            absent_rows = rows[names.str.lower().str.contains(str(absent).lower(), regex=False)]
        if len(absent_rows) == 0:
            continue
        r = absent_rows.sort_values(min_col, ascending=False).iloc[0]
        mins = float(r.get(min_col, 0.0) or 0.0)
        starter_score = safe_clip((mins - 20.0) / 14.0, 0.0, 1.2)
        if starter_score <= 0.0:
            continue

        fga = float(r.get(fga_col, 0.0) or 0.0) if fga_col else 0.0
        fta = float(r.get(fta_col, 0.0) or 0.0) if fta_col else 0.0
        tov = float(r.get(tov_col, 0.0) or 0.0) if tov_col else 0.0
        ast = float(r.get(ast_col, 0.0) or 0.0) if ast_col else 0.0
        reb = float(r.get(reb_col, 0.0) or 0.0) if reb_col else 0.0
        fg3a = float(r.get(fg3a_col, 0.0) or 0.0) if fg3a_col else 0.0

        usage_proxy = fga + 0.44 * fta + tov
        v_usage += usage_proxy * starter_score
        v_ast += ast * starter_score
        v_reb += reb * starter_score
        v_fg3a += fg3a * starter_score
        v_min += mins * starter_score
        absent_count += 1

    if absent_count == 0:
        return out
    out["starter_absent_count"] = float(absent_count)

    absorb = safe_clip(0.08 + target_weight * 0.26 + (combo_conc - 1.0) * 0.20, 0.08, 0.40)
    min_shift = safe_clip(v_min / 42.0, 0.0, 1.8)
    usage_shift = safe_clip(v_usage / 24.0, 0.0, 1.8)
    ast_shift = safe_clip(v_ast / 8.0, 0.0, 1.8)
    reb_shift = safe_clip(v_reb / 10.0, 0.0, 1.8)
    fg3a_shift = safe_clip(v_fg3a / 6.0, 0.0, 1.8)

    out["minutes_factor"] = float(safe_clip(1.0 + min_shift * absorb * 0.11, 1.0, 1.14))
    out["minutes_var_factor"] = float(safe_clip(1.0 + min_shift * (0.06 + (combo_conc - 1.0) * 0.20), 1.0, 1.20))
    out["pts_usage_factor"] = float(safe_clip(1.0 + usage_shift * absorb * 0.10, 1.0, 1.16))
    out["ast_usage_factor"] = float(safe_clip(1.0 + ast_shift * absorb * 0.12, 1.0, 1.18))
    out["reb_usage_factor"] = float(safe_clip(1.0 + reb_shift * absorb * 0.09, 1.0, 1.14))
    out["fg3a_usage_factor"] = float(safe_clip(1.0 + fg3a_shift * absorb * 0.11, 1.0, 1.16))
    return out

def _game_overlap_deltas(cache: SQLiteCache, season: str, player_name: str, teammate_name: str, base_log: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int]]:
    # Returns per-minute deltas and sample sizes
    try:
        tid = find_player_id(teammate_name)
    except Exception:
        return {}, {}
    tm_log = get_player_gamelog(tid, season, cache, n_games=82)
    if "Game_ID" not in base_log.columns or "Game_ID" not in tm_log.columns:
        return {}, {}
    player_games = set(base_log["Game_ID"].astype(str).values)
    tm_games = set(tm_log["Game_ID"].astype(str).values)
    games_with = player_games & tm_games
    games_without = player_games - tm_games
    if len(games_with) < 6 or len(games_without) < 4:
        return {}, {"with": len(games_with), "without": len(games_without)}
    with_df = base_log[base_log["Game_ID"].astype(str).isin(games_with)]
    without_df = base_log[base_log["Game_ID"].astype(str).isin(games_without)]
    out: Dict[str, float] = {}
    for stat in ["MIN", "FGA", "FG3A", "FTA", "AST", "REB", "OREB", "DREB", "PTS", "FG3M"]:
        if stat not in with_df.columns or stat not in without_df.columns:
            continue
        if stat == "MIN":
            delta = float(without_df["MIN"].mean() - with_df["MIN"].mean())
            shrink = min(len(games_without)/12.0, 1.0) * 0.65
            out["MIN"] = delta * shrink
        else:
            r_with = (with_df[stat].astype(float) / with_df["MIN"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
            r_wo = (without_df[stat].astype(float) / without_df["MIN"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
            if len(r_with) < 4 or len(r_wo) < 3:
                continue
            delta = float(r_wo.mean() - r_with.mean())
            shrink = min(len(r_wo)/12.0, 1.0) * 0.65
            out[f"{stat}_PM"] = delta * shrink
    return out, {"with": len(games_with), "without": len(games_without)}

def project_market(
    cache: SQLiteCache,
    season: str,
    player_name: str,
    player_team_abbr: str,
    opp_abbr: str,
    is_home: bool,
    market: str,
    team_out: List[str],
    opponent_out: List[str],
    minutes_override: Optional[Tuple[str, float]] = None,
    game_date_local: Optional[str] = None,
    spread: Optional[float] = None,
    vegas_team_total: Optional[float] = None,
    feature_gate: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, str, List[str], Dict[str, float]]:
    drivers: List[str] = []
    meta: Dict[str, float] = {}

    def _feature_flag_on(key: str) -> bool:
        if feature_gate is None:
            return False
        raw = feature_gate.get(key, 0.0)
        try:
            return float(raw) >= 0.5
        except Exception:
            return False

    disable_matchup_context = _feature_flag_on("disable_matchup_context")
    disable_player_splits_context = _feature_flag_on("disable_player_splits_context")
    disable_shot_context = _feature_flag_on("disable_shot_context")

    player = find_player(player_name)
    player_id = int(player["id"])
    team = find_team_by_abbr(player_team_abbr)
    opp = find_team_by_abbr(opp_abbr)
    team_name = team["full_name"]
    opp_name = opp["full_name"]
    team_id = int(team["id"])

    game_log = get_player_gamelog(player_id, season, cache, n_games=82)
    if len(game_log) == 0:
        raise ValueError(f"No game log found for {player_name} in {season}")
    game_log = enrich_rebound_columns(game_log, player_id, cache)

    rest_days = _compute_rest_days(game_log, game_date_local)
    if disable_player_splits_context:
        class _NeutralContext:
            @staticmethod
            def composite_factor(*args, **kwargs) -> float:
                return 1.0
        context = _NeutralContext()
        drivers.append("splits_ctx:disabled")
    else:
        context = PlayerSplitsContext(player_id, season, cache)

    adv_df = get_team_advanced(season, cache)
    opp_df = get_team_opponent_pergame(season, cache)
    team_base_df = get_team_base_pergame(season, cache)
    pace_fac = _pace_factor(team_name, opp_name, adv_df)
    team_total_fac = _team_total_factor(team_base_df, team_name, vegas_team_total)
    league_players = get_league_player_stats(season, cache)
    trend_ctx = _market_trend_context(game_log, market)

    def _gate(value: float, key: str) -> float:
        return apply_factor_gate(float(value), feature_gate, key)

    pace_fac = _gate(pace_fac, "pace_factor")
    team_total_fac = _gate(team_total_fac, "team_total_factor")
    trend_minutes_fac = _gate(trend_ctx["minutes_trend_factor"], "minutes_trend_factor")
    trend_minutes_var_fac = _gate(trend_ctx["minutes_var_trend_factor"], "minutes_var_trend_factor")
    trend_market_fac = _gate(trend_ctx["market_trend_factor"], "market_trend_factor")

    mu_min, var_min, min_flags = _minutes_model(game_log, override=minutes_override, spread=spread)
    drivers.extend(min_flags)
    role_ctx = _recent_role_context(player_id, game_log, cache, n_games=5)
    var_min *= _gate(role_ctx["minutes_var_fac"], "role_minutes_var_factor")
    mu_min *= trend_minutes_fac
    var_min *= trend_minutes_var_fac
    if role_ctx.get("role_samples", 0) > 0:
        drivers.append(
            "role_ctx:"
            f"samples={int(role_ctx['role_samples'])},"
            f"touch={role_ctx['touch_fac']:.3f},"
            f"usage={role_ctx['usage_fac']:.3f},"
            f"ast_create={role_ctx['ast_creation_fac']:.3f},"
            f"reb_chance={role_ctx['reb_chance_fac']:.3f},"
            f"min_var={role_ctx['minutes_var_fac']:.3f}"
        )
    if trend_ctx.get("minutes_trend_samples", 0) > 0 or trend_ctx.get("market_trend_samples", 0) > 0:
        drivers.append(
            "trend_ctx:"
            f"min={trend_minutes_fac:.3f},"
            f"min_var={trend_minutes_var_fac:.3f},"
            f"market={trend_market_fac:.3f},"
            f"mins={int(trend_ctx.get('minutes_trend_samples', 0))},"
            f"rate={int(trend_ctx.get('market_trend_samples', 0))}"
        )
        meta.update(
            {
                "trend_minutes_factor": float(trend_minutes_fac),
                "trend_minutes_var_factor": float(trend_minutes_var_fac),
                "trend_market_factor": float(trend_market_fac),
                "trend_minutes_recent": float(trend_ctx.get("minutes_recent", 0.0)),
                "trend_minutes_base": float(trend_ctx.get("minutes_base", 0.0)),
                "trend_market_rate_recent": float(trend_ctx.get("market_rate_recent", 0.0)),
                "trend_market_rate_base": float(trend_ctx.get("market_rate_base", 0.0)),
            }
        )

    # direct game-overlap deltas + on/off backup multipliers
    deltas: Dict[str, float] = {}
    usage_factors: Dict[str, float] = {"PTS": 1.0, "REB": 1.0, "AST": 1.0, "FG3M": 1.0, "FGA": 1.0, "FG3A": 1.0, "FTA": 1.0}
    if team_out:
        for tm in team_out:
            d, sample = _game_overlap_deltas(cache, season, player_name, tm, game_log)
            if d:
                if "MIN" in d:
                    mu_min = max(0.0, mu_min + safe_clip(d["MIN"], -6.0, 6.0))
                for k, v in d.items():
                    if k != "MIN":
                        deltas[k] = deltas.get(k, 0.0) + v
                drivers.append(f"onoff_overlap:{tm}:with={sample.get('with',0)},without={sample.get('without',0)}")
            # backup multiplier from team on/off
            for st in ["PTS", "REB", "AST", "FG3M", "FGA", "FG3A", "FTA"]:
                boost = estimate_usage_boost(cache, team_id, season, player_name, tm, st if st in {"PTS","REB","AST","FG3M"} else "PTS")
                # only mild partial use
                usage_factors[st] *= 1.0 + (boost - 1.0) * 0.35
        for k in usage_factors:
            usage_factors[k] = safe_clip(usage_factors[k], 0.90, 1.15)

    vacated = _vacated_opportunity_factors(league_players, team_out)
    vac_usage = _gate(vacated["usage_factor"], "vacated_usage_factor")
    vac_ast = _gate(vacated["ast_factor"], "vacated_ast_factor")
    vac_reb = _gate(vacated["reb_factor"], "vacated_reb_factor")
    vac_fg3a = _gate(vacated["fg3a_factor"], "vacated_fg3a_factor")
    vac_fta = _gate(vacated["fta_factor"], "vacated_fta_factor")
    usage_factors["PTS"] *= vac_usage
    usage_factors["FGA"] *= vac_usage
    usage_factors["AST"] *= vac_ast
    usage_factors["REB"] *= vac_reb
    usage_factors["FG3A"] *= vac_fg3a
    usage_factors["FG3M"] *= vac_fg3a
    usage_factors["FTA"] *= vac_fta
    if team_out and vacated["vacated_usage_proxy"] > 0:
        drivers.append(
            "vacated:"
            f"usage={vacated['vacated_usage_proxy']:.1f},"
            f"min={vacated['vacated_minutes']:.1f},"
            f"u_fac={vac_usage:.3f},"
            f"ast_fac={vac_ast:.3f},"
            f"reb_fac={vac_reb:.3f}"
        )
        meta.update(
            {
                "vacated_usage_proxy": float(vacated["vacated_usage_proxy"]),
                "vacated_minutes": float(vacated["vacated_minutes"]),
                "vacated_usage_factor": float(vac_usage),
                "vacated_ast_factor": float(vac_ast),
                "vacated_reb_factor": float(vac_reb),
                "vacated_fg3a_factor": float(vac_fg3a),
                "vacated_fta_factor": float(vac_fta),
            }
        )

    starter_rest = _starter_rest_combo_factors(
        league_df=league_players,
        team_id=team_id,
        target_player=player_name,
        team_out=team_out,
    )
    if float(starter_rest.get("starter_absent_count", 0.0)) > 0:
        sr_min = _gate(starter_rest["minutes_factor"], "starter_rest_minutes_factor")
        sr_var = _gate(starter_rest["minutes_var_factor"], "starter_rest_minutes_var_factor")
        sr_pts = _gate(starter_rest["pts_usage_factor"], "starter_rest_pts_factor")
        sr_ast = _gate(starter_rest["ast_usage_factor"], "starter_rest_ast_factor")
        sr_reb = _gate(starter_rest["reb_usage_factor"], "starter_rest_reb_factor")
        sr_fg3a = _gate(starter_rest["fg3a_usage_factor"], "starter_rest_fg3a_factor")

        # Blend these at partial strength to avoid double counting with on/off + vacated factors.
        mu_min *= 1.0 + (sr_min - 1.0) * 0.45
        var_min *= 1.0 + (sr_var - 1.0) * 0.50
        usage_factors["PTS"] *= 1.0 + (sr_pts - 1.0) * 0.45
        usage_factors["FGA"] *= 1.0 + (sr_pts - 1.0) * 0.45
        usage_factors["AST"] *= 1.0 + (sr_ast - 1.0) * 0.45
        usage_factors["REB"] *= 1.0 + (sr_reb - 1.0) * 0.45
        usage_factors["FG3A"] *= 1.0 + (sr_fg3a - 1.0) * 0.45
        usage_factors["FG3M"] *= 1.0 + (sr_fg3a - 1.0) * 0.45
        for key in usage_factors:
            usage_factors[key] = safe_clip(usage_factors[key], 0.88, 1.20)
        drivers.append(
            "starter_rest:"
            f"count={int(starter_rest['starter_absent_count'])},"
            f"combo={starter_rest['combo_concentration']:.3f},"
            f"min={sr_min:.3f},"
            f"pts={sr_pts:.3f},"
            f"ast={sr_ast:.3f},"
            f"reb={sr_reb:.3f},"
            f"fg3={sr_fg3a:.3f}"
        )
        meta.update(
            {
                "starter_rest_absent_count": float(starter_rest["starter_absent_count"]),
                "starter_rest_combo_concentration": float(starter_rest["combo_concentration"]),
                "starter_rest_minutes_factor": float(sr_min),
                "starter_rest_minutes_var_factor": float(sr_var),
                "starter_rest_pts_factor": float(sr_pts),
                "starter_rest_ast_factor": float(sr_ast),
                "starter_rest_reb_factor": float(sr_reb),
                "starter_rest_fg3a_factor": float(sr_fg3a),
            }
        )

    if disable_shot_context:
        shotloc = pd.DataFrame()
        cs_df = pd.DataFrame()
        pull_df = pd.DataFrame()
        lt10_df = pd.DataFrame()
        profile = {
            "corner3_share": 0.20,
            "ab3_share": 0.56,
            "rim_share": 0.28,
            "cs_share": 0.58,
            "pullup_share": 0.24,
        }
        drivers.append("shot_ctx:disabled")
    else:
        shotloc = get_opponent_shot_locations(season, cache)
        cs_df = get_opponent_ptshot_defense(season, "Catch and Shoot", cache)
        pull_df = get_opponent_ptshot_defense(season, "Pullups", cache)
        lt10_df = get_opponent_ptshot_defense(season, "Less Than 10 ft", cache)
        profile = _player_shot_profile(player_id, team_id, season, cache)
    eff = _player_efficiencies(game_log)

    corner_fga_fac = _opp_zone_factor(shotloc, opp_name, "Corner 3 FGA", 0.55)
    ab3_fga_fac = _opp_zone_factor(shotloc, opp_name, "Above the Break 3 FGA", 0.55)
    corner_pct_fac = _opp_zone_factor(shotloc, opp_name, "Corner 3 FG%", 0.45)
    ab3_pct_fac = _opp_zone_factor(shotloc, opp_name, "Above the Break 3 FG%", 0.45)
    rim_fga_fac = _opp_zone_factor(shotloc, opp_name, "Restricted Area FGA", 0.35)
    rim_pct_fac = _opp_zone_factor(shotloc, opp_name, "Restricted Area FG%", 0.18)

    cs_fga_fac = _opp_ptshot_factor(cs_df, opp_name, "FGA", 0.25)
    pull_fga_fac = _opp_ptshot_factor(pull_df, opp_name, "FGA", 0.25)
    paint_fga_fac = _opp_ptshot_factor(lt10_df, opp_name, "FGA", 0.20)

    # direct opponent factors
    def _opp_allowed(col: str, shrink: float = 0.35):
        if col not in opp_df.columns:
            return 1.0
        row = opp_df[opp_df["TEAM_NAME"] == opp_name]
        if len(row) == 0:
            return 1.0
        lg = float(opp_df[col].mean())
        if lg <= 0:
            return 1.0
        raw = float(row.iloc[0][col]) / lg
        return safe_clip(1.0 + (raw - 1.0) * shrink, 0.85, 1.15)
    opp_ast_fac = _opp_allowed("OPP_AST")
    opp_reb_fac = _opp_allowed("OPP_REB")
    opp_oreb_fac = _opp_allowed("OPP_OREB")
    opp_dreb_fac = _opp_allowed("OPP_DREB")
    opp_fta_fac = _opp_allowed("OPP_FTA")
    opp_fg3m_fac = _opp_allowed("OPP_FG3M")

    matchup_ctx: Dict[str, float] = {
        "lineup_minutes_factor": 1.0,
        "lineup_minutes_var_factor": 1.0,
        "foul_minutes_factor": 1.0,
        "foul_var_factor": 1.0,
        "blowout_minutes_factor": 1.0,
        "blowout_var_factor": 1.0,
        "lineup_pace_factor": 1.0,
        "onoff_usage_factor": 1.0,
        "onoff_ast_factor": 1.0,
        "onoff_reb_factor": 1.0,
        "onoff_fg3a_factor": 1.0,
        "pts_market_factor": 1.0,
        "fg3m_market_factor": 1.0,
        "ast_market_factor": 1.0,
        "reb_market_factor": 1.0,
    }
    try:
        if disable_matchup_context:
            raw_ctx = None
            drivers.append("lineup_context:disabled")
        else:
            raw_ctx = build_matchup_lineup_context(
                cache=cache,
                season=season,
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                opp_team_id=int(opp["id"]),
                team_name=team_name,
                opp_name=opp_name,
                opp_abbr=opp_abbr,
                game_log=game_log,
                team_out=team_out,
                opponent_out=opponent_out,
                spread=spread,
                team_base_df=team_base_df,
                league_players=league_players,
                opp_fta_factor=opp_fta_fac,
                game_date_local=game_date_local,
            )
        if isinstance(raw_ctx, dict):
            matchup_ctx.update({k: float(v) for k, v in raw_ctx.items() if isinstance(v, (int, float))})
            ctx_strength = float(raw_ctx.get("lineup_context_strength", 1.0) or 1.0)
            if ctx_strength < 0.999:
                for key, value in list(matchup_ctx.items()):
                    if not str(key).endswith("_factor"):
                        continue
                    matchup_ctx[key] = float(
                        safe_clip(1.0 + (float(value) - 1.0) * ctx_strength, 0.76, 1.32)
                    )
            if raw_ctx.get("primary_defender_name"):
                drivers.append(
                    "primary_def:"
                    f"{raw_ctx.get('primary_defender_name')}({float(raw_ctx.get('primary_defender_share') or 0.0):.2f})"
                )
            drivers.append(
                "lineup:"
                f"starter={float(raw_ctx.get('starter_prob', 0.5)):.2f},"
                f"rot={float(raw_ctx.get('rotation_certainty', 0.6)):.2f},"
                f"min_p10={float(raw_ctx.get('minutes_p10', 0.0)):.1f},"
                f"min_p90={float(raw_ctx.get('minutes_p90', 0.0)):.1f},"
                f"pace={matchup_ctx['lineup_pace_factor']:.3f}"
            )
            drivers.append(
                "lineup_onoff:"
                f"usage={matchup_ctx['onoff_usage_factor']:.3f},"
                f"ast={matchup_ctx['onoff_ast_factor']:.3f},"
                f"reb={matchup_ctx['onoff_reb_factor']:.3f},"
                f"fg3={matchup_ctx['onoff_fg3a_factor']:.3f}"
            )
            drivers.append(
                "opp_adv:"
                f"def={matchup_ctx.get('opp_est_def_factor',1.0):.3f},"
                f"pt_def_pts={matchup_ctx.get('pt_defend_pts_factor',1.0):.3f},"
                f"pt_def_3={matchup_ctx.get('pt_defend_fg3m_factor',1.0):.3f},"
                f"season_match={matchup_ctx.get('season_matchup_factor',1.0):.3f}"
            )
            drivers.append(
                "lineup_dash:"
                f"pace={matchup_ctx.get('lineup_dash_pace_factor',1.0):.3f},"
                f"off={matchup_ctx.get('lineup_dash_off_factor',1.0):.3f},"
                f"stable={matchup_ctx.get('lineup_dash_stability',0.6):.3f}"
            )
            drivers.append(
                "adv_matchup:"
                f"hustle={matchup_ctx.get('opp_pressure_pts_factor',1.0):.3f},"
                f"opp_lineup={matchup_ctx.get('opp_lineup_pts_factor',1.0):.3f},"
                f"shot_style={matchup_ctx.get('shot_style_pts_factor',1.0):.3f},"
                f"recent_def={matchup_ctx.get('recent_def_pts_factor',1.0):.3f},"
                f"persist={matchup_ctx.get('matchup_persistence_pts_factor',1.0):.3f}"
            )
            drivers.append(
                "hist_fg3:"
                f"vol={matchup_ctx.get('hist_fg3a_volume_factor',1.0):.3f},"
                f"match={matchup_ctx.get('hist_fg3m_matchup_factor',1.0):.3f},"
                f"def={matchup_ctx.get('hist_fg3m_defender_factor',1.0):.3f},"
                f"team={matchup_ctx.get('hist_fg3m_team_allowed_factor',1.0):.3f},"
                f"zone={matchup_ctx.get('hist_fg3m_zone_mix_factor',1.0):.3f},"
                f"sw={matchup_ctx.get('hist_fg3_sample_weight',0.0):.2f}"
            )
            drivers.append(
                "lineup_matrix:"
                f"combo_off={matchup_ctx.get('lineup_combo_off_factor',1.0):.3f},"
                f"combo_pace={matchup_ctx.get('lineup_combo_pace_factor',1.0):.3f},"
                f"combo_var={matchup_ctx.get('lineup_combo_var_factor',1.0):.3f},"
                f"switch_var={matchup_ctx.get('matchup_switch_var_factor',1.0):.3f}"
            )
            drivers.append(
                "lineup_vsv:"
                f"pts={matchup_ctx.get('lineup_vs_lineup_pts_factor',1.0):.3f},"
                f"fg3={matchup_ctx.get('lineup_vs_lineup_fg3m_factor',1.0):.3f},"
                f"ast={matchup_ctx.get('lineup_vs_lineup_ast_factor',1.0):.3f},"
                f"reb={matchup_ctx.get('lineup_vs_lineup_reb_factor',1.0):.3f},"
                f"pace={matchup_ctx.get('lineup_vs_lineup_pace_factor',1.0):.3f},"
                f"cov={matchup_ctx.get('lineup_vs_lineup_coverage',0.0):.2f}"
            )
            drivers.append(
                "extra_eps:"
                f"asttrk={matchup_ctx.get('assist_tracker_ast_factor',1.0):.3f},"
                f"cl_pts={matchup_ctx.get('team_clutch_pts_factor',1.0):.3f},"
                f"cl_ast={matchup_ctx.get('team_clutch_ast_factor',1.0):.3f},"
                f"viz={matchup_ctx.get('lineup_viz_opp_def_factor',1.0):.3f},"
                f"shot={matchup_ctx.get('shot_chart_detail_fg3_factor',1.0):.3f},"
                f"cov={matchup_ctx.get('extra_endpoint_coverage',0.0):.2f}"
            )
            drivers.append(
                "ctx_quality:"
                f"rel={matchup_ctx.get('endpoint_reliability',0.0):.2f},"
                f"cov={matchup_ctx.get('context_signal_coverage',0.0):.2f},"
                f"q={matchup_ctx.get('lineup_context_quality',0.0):.2f},"
                f"s={matchup_ctx.get('lineup_context_strength',1.0):.2f}"
            )
            drivers.append(
                "risk:"
                f"foul_min={matchup_ctx['foul_minutes_factor']:.3f},"
                f"foul_var={matchup_ctx['foul_var_factor']:.3f},"
                f"blowout_min={matchup_ctx['blowout_minutes_factor']:.3f},"
                f"blowout_var={matchup_ctx['blowout_var_factor']:.3f}"
            )
            meta.update(
                {
                    "minutes_p10": float(raw_ctx.get("minutes_p10", 0.0)),
                    "minutes_p90": float(raw_ctx.get("minutes_p90", 0.0)),
                    "starter_prob": float(raw_ctx.get("starter_prob", 0.5)),
                    "rotation_certainty": float(raw_ctx.get("rotation_certainty", 0.6)),
                    "position_bucket": str(raw_ctx.get("position_bucket", "")),
                    "opp_est_def_factor": float(raw_ctx.get("opp_est_def_factor", 1.0)),
                    "season_matchup_factor": float(raw_ctx.get("season_matchup_factor", 1.0)),
                    "lineup_dash_stability": float(raw_ctx.get("lineup_dash_stability", 0.6)),
                    "opp_pressure_pts_factor": float(raw_ctx.get("opp_pressure_pts_factor", 1.0)),
                    "opp_lineup_pts_factor": float(raw_ctx.get("opp_lineup_pts_factor", 1.0)),
                    "shot_style_pts_factor": float(raw_ctx.get("shot_style_pts_factor", 1.0)),
                    "recent_def_pts_factor": float(raw_ctx.get("recent_def_pts_factor", 1.0)),
                    "matchup_persistence_pts_factor": float(raw_ctx.get("matchup_persistence_pts_factor", 1.0)),
                    "lineup_combo_off_factor": float(raw_ctx.get("lineup_combo_off_factor", 1.0)),
                    "lineup_combo_pace_factor": float(raw_ctx.get("lineup_combo_pace_factor", 1.0)),
                    "lineup_combo_var_factor": float(raw_ctx.get("lineup_combo_var_factor", 1.0)),
                    "matchup_switch_var_factor": float(raw_ctx.get("matchup_switch_var_factor", 1.0)),
                    "lineup_vs_lineup_pts_factor": float(raw_ctx.get("lineup_vs_lineup_pts_factor", 1.0)),
                    "lineup_vs_lineup_fg3m_factor": float(raw_ctx.get("lineup_vs_lineup_fg3m_factor", 1.0)),
                    "lineup_vs_lineup_ast_factor": float(raw_ctx.get("lineup_vs_lineup_ast_factor", 1.0)),
                    "lineup_vs_lineup_reb_factor": float(raw_ctx.get("lineup_vs_lineup_reb_factor", 1.0)),
                    "lineup_vs_lineup_pace_factor": float(raw_ctx.get("lineup_vs_lineup_pace_factor", 1.0)),
                    "lineup_vs_lineup_coverage": float(raw_ctx.get("lineup_vs_lineup_coverage", 0.0)),
                    "assist_tracker_ast_factor": float(raw_ctx.get("assist_tracker_ast_factor", 1.0)),
                    "team_clutch_pts_factor": float(raw_ctx.get("team_clutch_pts_factor", 1.0)),
                    "team_clutch_ast_factor": float(raw_ctx.get("team_clutch_ast_factor", 1.0)),
                    "lineup_viz_opp_def_factor": float(raw_ctx.get("lineup_viz_opp_def_factor", 1.0)),
                    "shot_chart_detail_fg3_factor": float(raw_ctx.get("shot_chart_detail_fg3_factor", 1.0)),
                    "boxscore_defensive_opp_factor": float(raw_ctx.get("boxscore_defensive_opp_factor", 1.0)),
                    "hist_fg3a_volume_factor": float(raw_ctx.get("hist_fg3a_volume_factor", 1.0)),
                    "hist_fg3m_matchup_factor": float(raw_ctx.get("hist_fg3m_matchup_factor", 1.0)),
                    "hist_fg3m_defender_factor": float(raw_ctx.get("hist_fg3m_defender_factor", 1.0)),
                    "hist_fg3m_team_allowed_factor": float(raw_ctx.get("hist_fg3m_team_allowed_factor", 1.0)),
                    "hist_fg3m_zone_mix_factor": float(raw_ctx.get("hist_fg3m_zone_mix_factor", 1.0)),
                    "hist_fg3m_factor": float(raw_ctx.get("hist_fg3m_factor", 1.0)),
                    "hist_fg3_sample_weight": float(raw_ctx.get("hist_fg3_sample_weight", 0.0)),
                    "extra_endpoint_coverage": float(raw_ctx.get("extra_endpoint_coverage", 0.0)),
                    "endpoint_reliability": float(raw_ctx.get("endpoint_reliability", 0.0)),
                    "context_signal_coverage": float(raw_ctx.get("context_signal_coverage", 0.0)),
                    "lineup_context_quality": float(raw_ctx.get("lineup_context_quality", 0.0)),
                    "lineup_context_strength": float(raw_ctx.get("lineup_context_strength", 1.0)),
                }
            )
    except Exception as exc:
        drivers.append(f"lineup_context_unavailable:{type(exc).__name__}")

    mu_min *= (
        _gate(matchup_ctx["lineup_minutes_factor"], "lineup_minutes_factor")
        * _gate(matchup_ctx["foul_minutes_factor"], "foul_minutes_factor")
        * _gate(matchup_ctx["blowout_minutes_factor"], "blowout_minutes_factor")
    )
    var_min *= (
        _gate(matchup_ctx["lineup_minutes_var_factor"], "lineup_minutes_var_factor")
        * _gate(matchup_ctx["foul_var_factor"], "foul_var_factor")
        * _gate(matchup_ctx["blowout_var_factor"], "blowout_var_factor")
    )
    pace_fac *= _gate(matchup_ctx["lineup_pace_factor"], "lineup_pace_factor")
    usage_factors["PTS"] *= _gate(matchup_ctx["onoff_usage_factor"], "onoff_usage_factor")
    usage_factors["FGA"] *= _gate(matchup_ctx["onoff_usage_factor"], "onoff_usage_factor")
    usage_factors["FTA"] *= _gate(matchup_ctx["onoff_usage_factor"], "onoff_usage_factor")
    usage_factors["AST"] *= _gate(matchup_ctx["onoff_ast_factor"], "onoff_ast_factor")
    usage_factors["REB"] *= _gate(matchup_ctx["onoff_reb_factor"], "onoff_reb_factor")
    usage_factors["FG3A"] *= _gate(matchup_ctx["onoff_fg3a_factor"], "onoff_fg3a_factor")
    usage_factors["FG3M"] *= _gate(matchup_ctx["onoff_fg3a_factor"], "onoff_fg3a_factor")
    for key in usage_factors:
        usage_factors[key] = safe_clip(usage_factors[key], 0.88, 1.20)

    # Context factors
    fga_ctx = context.composite_factor("FGA", is_home=is_home, rest_days=rest_days)
    fg3m_ctx = context.composite_factor("FG3M", is_home=is_home, rest_days=rest_days)
    fta_ctx = context.composite_factor("FTA", is_home=is_home, rest_days=rest_days)
    reb_ctx = context.composite_factor("REB", is_home=is_home, rest_days=rest_days)
    ast_ctx = context.composite_factor("AST", is_home=is_home, rest_days=rest_days)

    opp_abs_fac = _gate(_opponent_absence_factor(league_players, opponent_out, market), "opponent_absence_factor")
    drivers.append(f"pace={pace_fac:.3f}")
    drivers.append(f"context:fga={fga_ctx:.3f},fg3m={fg3m_ctx:.3f},fta={fta_ctx:.3f},reb={reb_ctx:.3f},ast={ast_ctx:.3f}")
    drivers.append(f"player_profile:corner={profile['corner3_share']:.3f},ab3={profile['ab3_share']:.3f},rim={profile['rim_share']:.3f},cs={profile['cs_share']:.3f},pull={profile['pullup_share']:.3f}")
    drivers.append(f"opp_scheme:corner_fga={corner_fga_fac:.3f},ab3_fga={ab3_fga_fac:.3f},rim_fga={rim_fga_fac:.3f},cs_fga={cs_fga_fac:.3f},pull_fga={pull_fga_fac:.3f}")

    if market == "FG3M":
        mu_3pa_pm, var_3pa_pm = _rate_model(game_log, "FG3A", "MIN", 30)
        mu_3pa_pm = max(0.0, mu_3pa_pm + deltas.get("FG3A_PM", 0.0))
        p3 = eff["p3"]

        zone_3pa_fac = (
            profile["corner3_share"] * corner_fga_fac +
            profile["ab3_share"] * ab3_fga_fac
        )
        if (profile["corner3_share"] + profile["ab3_share"]) < 0.15:
            zone_3pa_fac = 0.5 * corner_fga_fac + 0.5 * ab3_fga_fac
        shottype_fac = max(0.2, profile["cs_share"]) * cs_fga_fac + max(0.2, profile["pullup_share"]) * pull_fga_fac
        att_fac = safe_clip(0.55 * zone_3pa_fac + 0.25 * shottype_fac + 0.20 * opp_fg3m_fac, 0.80, 1.25)
        pct_fac = safe_clip(profile["corner3_share"] * corner_pct_fac + max(profile["ab3_share"], 0.1) * ab3_pct_fac, 0.88, 1.12)
        if (profile["corner3_share"] + profile["ab3_share"]) < 0.15:
            pct_fac = safe_clip(0.35 * corner_pct_fac + 0.65 * ab3_pct_fac, 0.88, 1.12)

        scale = (
            pace_fac
            * team_total_fac
            * fg3m_ctx
            * usage_factors["FG3A"]
            * att_fac
            * opp_abs_fac
            * _gate(role_ctx["fg3m_role_fac"], "role_fg3m_factor")
            * _gate(matchup_ctx["fg3m_market_factor"], "fg3m_market_factor")
            * trend_market_fac
        )
        mu_3pa, var_3pa = _count_from_min_rate(mu_min, var_min, mu_3pa_pm, var_3pa_pm, scale=scale)
        mu = mu_3pa * p3 * pct_fac * usage_factors["FG3M"]
        var = (p3 * pct_fac) ** 2 * var_3pa + mu * (1 - p3) * 0.90
        sigma = math.sqrt(max(var, mu + 0.25))
        drivers.append(f"fg3m:p3={p3:.3f},att_fac={att_fac:.3f},pct_fac={pct_fac:.3f},team_total_fac={team_total_fac:.3f},opp_abs={opp_abs_fac:.3f}")
        meta.update({"p3": p3, "att_fac": att_fac, "pct_fac": pct_fac})
        return float(mu), float(sigma), "neg_binom", drivers, meta

    if market == "PTS":
        mu_fga_pm, var_fga_pm = _rate_model(game_log, "FGA", "MIN", 35)
        mu_fg3a_pm, var_fg3a_pm = _rate_model(game_log, "FG3A", "MIN", 35)
        mu_fta_pm, var_fta_pm = _rate_model(game_log, "FTA", "MIN", 35)

        mu_fga_pm = max(0.0, mu_fga_pm + deltas.get("FGA_PM", 0.0))
        mu_fg3a_pm = max(0.0, mu_fg3a_pm + deltas.get("FG3A_PM", 0.0))
        mu_fta_pm = max(0.0, mu_fta_pm + deltas.get("FTA_PM", 0.0))
        mu_2pa_pm = max(mu_fga_pm - mu_fg3a_pm, 0.0)
        # FG3A is a subset of FGA so they are positively correlated.
        # Var(FGA - FG3A) = Var(FGA) + Var(FG3A) - 2·Cov(FGA,FG3A)
        # Estimated correlation ≈ 0.55 between per-minute FGA and FG3A rates.
        cov_est = 0.55 * math.sqrt(max(var_fga_pm * var_fg3a_pm, 0.0))
        var_2pa_pm = max(var_fga_pm + var_fg3a_pm - 2.0 * cov_est, var_fga_pm * 0.40)

        att3_zone = profile["corner3_share"] * corner_fga_fac + max(profile["ab3_share"], 0.1) * ab3_fga_fac
        if (profile["corner3_share"] + profile["ab3_share"]) < 0.15:
            att3_zone = 0.35 * corner_fga_fac + 0.65 * ab3_fga_fac
        att3_shottype = max(profile["cs_share"], 0.1) * cs_fga_fac + max(profile["pullup_share"], 0.1) * pull_fga_fac
        att3_fac = safe_clip(0.60 * att3_zone + 0.20 * att3_shottype + 0.20 * opp_fg3m_fac, 0.80, 1.22)
        pct3_fac = safe_clip(profile["corner3_share"] * corner_pct_fac + max(profile["ab3_share"], 0.1) * ab3_pct_fac, 0.90, 1.10)
        if (profile["corner3_share"] + profile["ab3_share"]) < 0.15:
            pct3_fac = safe_clip(0.35 * corner_pct_fac + 0.65 * ab3_pct_fac, 0.90, 1.10)

        two_vol_fac = safe_clip(profile["rim_share"] * rim_fga_fac + (1.0 - profile["rim_share"]) * 1.0 + 0.10 * paint_fga_fac, 0.90, 1.12)
        p2_fac = safe_clip(profile["rim_share"] * rim_pct_fac + (1.0 - profile["rim_share"]) * 1.0, 0.94, 1.08)

        scale_common = (
            pace_fac
            * team_total_fac
            * usage_factors["FGA"]
            * fga_ctx
            * opp_abs_fac
            * _gate(role_ctx["pts_role_fac"], "role_pts_factor")
            * _gate(matchup_ctx["pts_market_factor"], "pts_market_factor")
            * trend_market_fac
        )
        mu_3pa, var_3pa = _count_from_min_rate(mu_min, var_min, mu_fg3a_pm, var_fg3a_pm, scale=scale_common * att3_fac)
        mu_2pa, var_2pa = _count_from_min_rate(mu_min, var_min, mu_2pa_pm, var_2pa_pm, scale=scale_common * two_vol_fac)
        mu_fta, var_fta = _count_from_min_rate(
            mu_min,
            var_min,
            mu_fta_pm,
            var_fta_pm,
            scale=pace_fac
            * team_total_fac
            * usage_factors["FTA"]
            * fta_ctx
            * opp_fta_fac
            * opp_abs_fac
            * _gate(role_ctx["pts_role_fac"], "role_pts_factor")
            * trend_market_fac,
        )

        mu_3m = mu_3pa * eff["p3"] * pct3_fac
        mu_2m = mu_2pa * eff["p2"] * p2_fac
        mu_ftm = mu_fta * eff["ft"]

        mu = 3.0 * mu_3m + 2.0 * mu_2m + mu_ftm
        var_3m = (eff["p3"] * pct3_fac) ** 2 * var_3pa + mu_3m * (1 - eff["p3"]) * 0.90
        var_2m = (eff["p2"] * p2_fac) ** 2 * var_2pa + mu_2m * (1 - eff["p2"]) * 0.90
        var_ftm = (eff["ft"]) ** 2 * var_fta + mu_ftm * (1 - eff["ft"]) * 0.65
        # Cross-component covariance: 3PM, 2PM, and FTM all correlate
        # positively through shared minutes/usage fluctuations.
        cov_3m_2m = 0.30 * math.sqrt(max(var_3m * var_2m, 0.0))
        cov_3m_ft = 0.20 * math.sqrt(max(var_3m * var_ftm, 0.0))
        cov_2m_ft = 0.25 * math.sqrt(max(var_2m * var_ftm, 0.0))
        var = (
            9.0 * var_3m + 4.0 * var_2m + var_ftm
            + 2.0 * 3.0 * 2.0 * cov_3m_2m
            + 2.0 * 3.0 * 1.0 * cov_3m_ft
            + 2.0 * 2.0 * 1.0 * cov_2m_ft
        )
        # Variance floor: PTS has high game-to-game variance (CV ~ 0.30-0.35).
        # Without this floor the model can become overconfident for stable players.
        var = max(var, 2.5 * mu + 4.0)
        sigma = math.sqrt(var)
        drivers.append(f"pts:p2={eff['p2']:.3f}*{p2_fac:.3f},p3={eff['p3']:.3f}*{pct3_fac:.3f},ft={eff['ft']:.3f},att3_fac={att3_fac:.3f},two_vol={two_vol_fac:.3f},team_total={team_total_fac:.3f},opp_abs={opp_abs_fac:.3f}")
        meta.update({"p2": eff["p2"], "p3": eff["p3"], "ft": eff["ft"]})
        return float(mu), float(sigma), "normal", drivers, meta

    if market == "REB":
        reb_split_ready = "OREB" in game_log.columns and "DREB" in game_log.columns and game_log["OREB"].notna().sum() >= 8 and game_log["DREB"].notna().sum() >= 8
        if reb_split_ready:
            mu_oreb_pm, var_oreb_pm = _rate_model(game_log.fillna(0), "OREB", "MIN", 35)
            mu_dreb_pm, var_dreb_pm = _rate_model(game_log.fillna(0), "DREB", "MIN", 35)
            mu_oreb_pm = max(0.0, mu_oreb_pm + deltas.get("OREB_PM", 0.0))
            mu_dreb_pm = max(0.0, mu_dreb_pm + deltas.get("DREB_PM", 0.0))
            mu_oreb, var_oreb = _count_from_min_rate(
                mu_min,
                var_min,
                mu_oreb_pm,
                var_oreb_pm,
                scale=pace_fac
                * reb_ctx
                * opp_oreb_fac
                * usage_factors["REB"]
                * opp_abs_fac
                * _gate(role_ctx["reb_role_fac"], "role_reb_factor")
                * _gate(matchup_ctx["reb_market_factor"], "reb_market_factor")
                * trend_market_fac,
            )
            mu_dreb, var_dreb = _count_from_min_rate(
                mu_min,
                var_min,
                mu_dreb_pm,
                var_dreb_pm,
                scale=pace_fac
                * reb_ctx
                * opp_dreb_fac
                * usage_factors["REB"]
                * opp_abs_fac
                * _gate(role_ctx["reb_role_fac"], "role_reb_factor")
                * _gate(matchup_ctx["reb_market_factor"], "reb_market_factor")
                * trend_market_fac,
            )
            mu = mu_oreb + mu_dreb
            var = var_oreb + var_dreb
            drivers.append(f"reb:split_model,oreb_fac={opp_oreb_fac:.3f},dreb_fac={opp_dreb_fac:.3f},opp_abs={opp_abs_fac:.3f}")
        else:
            mu_reb_pm, var_reb_pm = _rate_model(game_log, "REB", "MIN", 35)
            mu_reb_pm = max(0.0, mu_reb_pm + deltas.get("REB_PM", 0.0))
            mu, var = _count_from_min_rate(
                mu_min,
                var_min,
                mu_reb_pm,
                var_reb_pm,
                scale=pace_fac
                * reb_ctx
                * opp_reb_fac
                * usage_factors["REB"]
                * opp_abs_fac
                * _gate(role_ctx["reb_role_fac"], "role_reb_factor")
                * _gate(matchup_ctx["reb_market_factor"], "reb_market_factor")
                * trend_market_fac,
            )
            drivers.append("reb:reb_only_fallback")
        sigma = math.sqrt(max(var, mu + 0.25))
        return float(mu), float(sigma), "neg_binom", drivers, meta

    if market == "AST":
        mu_ast_pm, var_ast_pm = _rate_model(game_log, "AST", "MIN", 35)
        mu_ast_pm = max(0.0, mu_ast_pm + deltas.get("AST_PM", 0.0))
        mu, var = _count_from_min_rate(
            mu_min,
            var_min,
            mu_ast_pm,
            var_ast_pm,
            scale=pace_fac
            * ast_ctx
            * opp_ast_fac
            * team_total_fac
            * usage_factors["AST"]
            * opp_abs_fac
            * _gate(role_ctx["ast_role_fac"], "role_ast_factor")
            * _gate(matchup_ctx["ast_market_factor"], "ast_market_factor")
            * trend_market_fac,
        )
        sigma = math.sqrt(max(var, mu + 0.25))
        drivers.append(f"ast:opp_ast={opp_ast_fac:.3f},team_total={team_total_fac:.3f},opp_abs={opp_abs_fac:.3f}")
        return float(mu), float(sigma), "neg_binom", drivers, meta

    raise ValueError(f"Unsupported market: {market}")
