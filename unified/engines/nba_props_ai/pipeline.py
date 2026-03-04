from __future__ import annotations

import json
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .cache import SQLiteCache
from .injuries import (
    MinutesProjectionItem,
    build_team_injury_maps,
    fetch_injuries_with_fallback,
    fetch_rotowire_projected_minutes,
)
from .math_dists import normal_over_prob, negbin_over_prob
from .model_core import project_market
from .models import DistributionSpec, GameState, MinutesOverride, ProjectionResult, PropRequest, Scenario
from .nba_data import find_team_by_abbr, get_roster_name_to_id
from .utils import american_to_implied_prob, implied_prob_to_american, name_key, norm_name, parse_minutes_value, safe_clip

class PipelineError(Exception):
    pass

STATUS_PRIORS = {
    "PROBABLE": (0.85, 0.15),
    "QUESTIONABLE": (0.55, 0.45),
    "DOUBTFUL": (0.20, 0.80),
}
TEAM_MINUTES_BUDGET = 240.0
ROTATION_CORE_PLAYERS = 9
MIN_PLAYERS_FOR_BUDGET_ALLOCATION = 8

def _roster_id_to_name(roster_by_name: Dict[str, int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, pid in roster_by_name.items():
        out.setdefault(int(pid), name)
    return out

def _roster_player_id(roster_by_name: Dict[str, int], player_name_or_key: str) -> Optional[int]:
    key = name_key(player_name_or_key)
    if not key:
        return None
    for roster_name, pid in roster_by_name.items():
        if name_key(roster_name) == key:
            return int(pid)
    return None

def _display_name_from_key(roster_by_name: Dict[str, int], key: str) -> str:
    target = name_key(key)
    for roster_name in roster_by_name.keys():
        if name_key(roster_name) == target:
            return norm_name(roster_name)
    return key

def _display_name_from_game(gs: GameState, key: str) -> str:
    team_name = _display_name_from_key(gs.team_roster_by_name, key)
    if team_name != key:
        return team_name
    return _display_name_from_key(gs.opp_roster_by_name, key)

def _lookup_minutes_override(gs: GameState, player_name_or_key: str):
    key = name_key(player_name_or_key)
    if not key:
        return None
    for name, override in gs.overrides.minutes_overrides.items():
        if name_key(name) == key:
            return override
    return None

def _apply_minutes_overrides_from_feed(
    gs: GameState,
    minutes_by_team: Dict[str, Dict[str, MinutesProjectionItem]],
) -> int:
    existing_keys = {name_key(x) for x in gs.overrides.minutes_overrides.keys() if name_key(x)}
    applied = 0
    roster_sets = [
        (str(gs.player_team_abbr).upper(), gs.team_roster_by_name),
        (str(gs.opponent_abbr).upper(), gs.opp_roster_by_name),
    ]
    for team_abbr, roster in roster_sets:
        team_minutes = minutes_by_team.get(team_abbr, {})
        if not team_minutes:
            continue
        for roster_name in roster.keys():
            key = name_key(roster_name)
            if not key or key in existing_keys:
                continue
            item = team_minutes.get(key)
            if item is None:
                continue
            gs.overrides.minutes_overrides[norm_name(roster_name)] = MinutesOverride(
                kind="target",
                value=float(max(item.projected_minutes, 0.0)),
            )
            existing_keys.add(key)
            applied += 1
    return applied

def _merge_status_hints_from_minutes(
    gs: GameState,
    injury_by_name: Dict[str, str],
    minutes_by_team: Dict[str, Dict[str, MinutesProjectionItem]],
) -> Tuple[Dict[str, str], int]:
    merged = dict(injury_by_name or {})
    injected = 0
    roster_sets = [
        (str(gs.player_team_abbr).upper(), gs.team_roster_by_name),
        (str(gs.opponent_abbr).upper(), gs.opp_roster_by_name),
    ]
    for team_abbr, roster in roster_sets:
        team_minutes = minutes_by_team.get(team_abbr, {})
        if not team_minutes:
            continue
        for roster_name in roster.keys():
            key = name_key(roster_name)
            if not key:
                continue
            item = team_minutes.get(key)
            if item is None:
                continue
            status_hint = str(item.status_hint or "").upper()
            if status_hint in {"UNKNOWN", ""}:
                continue
            if str(merged.get(roster_name) or "").strip():
                continue
            merged[roster_name] = status_hint
            injected += 1
    return merged, injected

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
    try:
        return pd.read_json(StringIO(text))
    except Exception:
        return pd.DataFrame()

def _normalize_col_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if len(df) == 0:
        return None
    mapping = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in mapping:
            return mapping[key]
    return None

def _load_league_minutes_map(cache: SQLiteCache, season: str) -> Dict[int, float]:
    try:
        hit = cache.get("leaguedashplayerstats", {"season": season})
    except Exception:
        return {}
    if hit is None:
        return {}
    df = _read_cached_frame(hit.data_json)
    if len(df) == 0:
        return {}
    pid_col = _find_first_col(df, ["PLAYER_ID", "PERSON_ID", "playerId"])
    min_col = _find_first_col(df, ["MIN", "minutes"])
    if not pid_col or not min_col:
        return {}
    out: Dict[int, float] = {}
    for _, row in df.iterrows():
        try:
            pid = int(row[pid_col])
            mins = float(row[min_col])
        except Exception:
            continue
        if mins <= 0:
            continue
        out[pid] = mins
    return out

def _status_minutes_multiplier(status: str) -> float:
    key = str(status or "").upper()
    if key == "AVAILABLE":
        return 1.00
    if key == "PROBABLE":
        return 0.95
    if key == "QUESTIONABLE":
        return 0.72
    if key == "DOUBTFUL":
        return 0.40
    if key == "OUT":
        return 0.00
    return 0.88

def _status_minutes_cap(status: str) -> float:
    key = str(status or "").upper()
    if key == "DOUBTFUL":
        return 16.0
    if key == "QUESTIONABLE":
        return 28.0
    if key == "PROBABLE":
        return 36.0
    if key == "AVAILABLE":
        return 38.0
    if key == "OUT":
        return 0.0
    return 34.0

def _load_cached_recent_minutes_for_player(
    cache: SQLiteCache,
    season: str,
    player_id: int,
) -> Optional[float]:
    try:
        hit = cache.get("playergamelog", {"player_id": int(player_id), "season": season})
    except Exception:
        return None
    if hit is None:
        return None
    df = _read_cached_frame(hit.data_json)
    if len(df) == 0:
        return None
    min_col = _find_first_col(df, ["MIN", "minutes"])
    if not min_col:
        return None
    vals: List[float] = []
    for raw in df[min_col].head(12).tolist():
        mins = float(parse_minutes_value(raw))
        if mins > 0:
            vals.append(mins)
    if len(vals) < 3:
        return None
    weights = [0.92 ** i for i in range(len(vals))]
    denom = float(sum(weights))
    if denom <= 0:
        return None
    return float(sum(v * w for v, w in zip(vals, weights)) / denom)

def _apply_nonrotowire_minutes_fallback_overrides(
    gs: GameState,
    injury_by_name: Dict[str, str],
    injury_by_id: Dict[int, str],
    league_minutes_map: Dict[int, float],
    cache: SQLiteCache,
    season: str,
    recent_minutes_cache: Dict[int, Optional[float]],
) -> Dict[str, int]:
    if not league_minutes_map and not recent_minutes_cache:
        # Keep going; recent map can be lazily populated even if currently empty.
        pass
    by_name_key = {name_key(k): str(v) for k, v in (injury_by_name or {}).items() if name_key(k)}
    by_id_key = {int(k): str(v) for k, v in (injury_by_id or {}).items()}
    existing_keys = {name_key(x) for x in gs.overrides.minutes_overrides.keys() if name_key(x)}
    applied = 0
    league_only = 0
    recent_only = 0
    blended = 0
    applied_keys: set[str] = set()
    baseline_pre_status: Dict[int, float] = {}
    roster_sets = [gs.team_roster_by_name, gs.opp_roster_by_name]

    for roster in roster_sets:
        for roster_name, pid in roster.items():
            key = name_key(roster_name)
            if not key or key in existing_keys:
                continue
            pid_int = int(pid)
            league_base = league_minutes_map.get(pid_int)
            if pid_int not in recent_minutes_cache:
                recent_minutes_cache[pid_int] = _load_cached_recent_minutes_for_player(
                    cache=cache,
                    season=season,
                    player_id=pid_int,
                )
            recent_base = recent_minutes_cache.get(pid_int)

            source = ""
            if league_base is not None and recent_base is not None:
                base = 0.62 * float(recent_base) + 0.38 * float(league_base)
                source = "blend"
            elif recent_base is not None:
                base = float(recent_base)
                source = "recent"
            elif league_base is not None:
                base = float(league_base)
                source = "league"
            else:
                base = None
            if base is None:
                continue
            baseline_pre_status[pid_int] = float(max(base, 0.0))
            status = str(injury_by_id.get(pid_int) or by_name_key.get(key) or "AVAILABLE").upper()
            target = float(base) * float(_status_minutes_multiplier(status))
            target = float(max(min(target, _status_minutes_cap(status)), 0.0))
            gs.overrides.minutes_overrides[norm_name(roster_name)] = MinutesOverride(
                kind="target",
                value=target,
            )
            existing_keys.add(key)
            applied_keys.add(key)
            applied += 1
            if source == "blend":
                blended += 1
            elif source == "recent":
                recent_only += 1
            elif source == "league":
                league_only += 1

    def _status_of(roster_name: str, pid: int) -> str:
        k = name_key(roster_name)
        return str(by_id_key.get(int(pid)) or by_name_key.get(k) or "AVAILABLE").upper()

    redistributed_minutes = 0.0
    redistributed_players = 0
    for roster in roster_sets:
        # Only redistribute among players whose minutes were fallback-derived in this pass.
        team_players = [(str(name), int(pid)) for name, pid in roster.items() if name_key(name) in applied_keys]
        if len(team_players) < 3:
            continue

        lost_pool = 0.0
        candidates: List[Tuple[str, int]] = []
        candidate_weights: Dict[int, float] = {}
        candidate_caps: Dict[int, float] = {}
        candidate_current: Dict[int, float] = {}
        candidate_initial: Dict[int, float] = {}

        for roster_name, pid in team_players:
            k = name_key(roster_name)
            ov = gs.overrides.minutes_overrides.get(norm_name(roster_name))
            if ov is None or ov.kind != "target":
                continue
            cur = float(max(ov.value, 0.0))
            status = _status_of(roster_name, pid)
            baseline = float(max(baseline_pre_status.get(int(pid), cur), cur))
            starter_like = safe_clip((baseline - 24.0) / 10.0, 0.0, 1.2)
            if status in {"OUT", "DOUBTFUL"} and starter_like > 0:
                status_weight = 1.0 if status == "OUT" else 0.65
                lost_pool += baseline * starter_like * status_weight
                continue
            if status in {"OUT", "DOUBTFUL"}:
                continue
            cap = float(max(_status_minutes_cap(status), 0.0))
            room = max(cap - cur, 0.0)
            if room <= 0:
                continue
            # Prefer rotation players and ball-dominant minute anchors.
            w = max(cur, 8.0) * (1.0 + starter_like * 0.55)
            candidate_weights[pid] = float(w)
            candidate_caps[pid] = float(room)
            candidate_current[pid] = float(cur)
            candidate_initial[pid] = float(cur)
            candidates.append((roster_name, pid))

        if lost_pool <= 0 or not candidates:
            continue
        # Redistribute only a portion to avoid over-reaction and preserve uncertainty.
        pool = float(min(lost_pool * 0.74, 28.0))
        touched: set[int] = set()
        for _ in range(24):
            if pool <= 1e-6:
                break
            open_ids = [pid for _, pid in candidates if candidate_caps.get(pid, 0.0) > 1e-6]
            if not open_ids:
                break
            wsum = float(sum(candidate_weights.get(pid, 0.0) for pid in open_ids))
            if wsum <= 0:
                break
            progressed = 0.0
            for pid in open_ids:
                room = float(candidate_caps.get(pid, 0.0))
                if room <= 0:
                    continue
                add = min(pool * (candidate_weights.get(pid, 0.0) / wsum), room)
                if add <= 0:
                    continue
                candidate_current[pid] = float(candidate_current.get(pid, 0.0) + add)
                candidate_caps[pid] = float(max(room - add, 0.0))
                pool -= add
                progressed += add
                touched.add(int(pid))
            if progressed <= 1e-6:
                break

        if touched:
            pid_to_name = {int(pid): str(name) for name, pid in team_players}
            for pid in touched:
                roster_name = pid_to_name.get(int(pid), "")
                if not roster_name:
                    continue
                gs.overrides.minutes_overrides[norm_name(roster_name)] = MinutesOverride(
                    kind="target",
                    value=float(max(candidate_current.get(int(pid), 0.0), 0.0)),
                )
            redistributed_minutes += float(
                sum(
                    max(candidate_current.get(int(pid), 0.0) - candidate_initial.get(int(pid), 0.0), 0.0)
                    for pid in touched
                )
            )
            redistributed_players += int(len(touched))
    return {
        "applied": int(applied),
        "league_only": int(league_only),
        "recent_only": int(recent_only),
        "blended": int(blended),
        "redistributed_players": int(redistributed_players),
        "redistributed_minutes": float(redistributed_minutes),
    }


def _status_token(raw: object) -> str:
    token = str(raw or "").strip().upper()
    if token in {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE"}:
        return token
    return "AVAILABLE"


def _allocate_roster_minutes_to_budget(
    players: List[Dict[str, object]],
    *,
    manual_locked_name_keys: set[str],
) -> Tuple[Dict[int, float], Dict[str, float]]:
    allocations: Dict[int, float] = {}
    fixed_target: Dict[int, float] = {}
    weights: Dict[int, float] = {}
    caps: Dict[int, float] = {}
    floors: Dict[int, float] = {}

    for p in players:
        pid = int(p["player_id"])
        status = _status_token(p.get("status"))
        allocations[pid] = 0.0
        if status == "OUT":
            caps[pid] = 0.0
            continue
        cap = float(_status_minutes_cap(status))
        ov = p.get("override")
        if isinstance(ov, MinutesOverride) and ov.kind == "cap":
            cap = min(cap, float(max(ov.value, 0.0)))
        caps[pid] = float(max(cap, 0.0))

    ordered = sorted(players, key=lambda x: float(x.get("raw_minutes", 0.0) or 0.0), reverse=True)
    for rank, p in enumerate(ordered):
        pid = int(p["player_id"])
        status = _status_token(p.get("status"))
        if status == "OUT":
            continue
        name_key_val = name_key(str(p.get("player_name", "")))
        ov = p.get("override")
        if isinstance(ov, MinutesOverride) and ov.kind == "target" and name_key_val in manual_locked_name_keys:
            fixed_target[pid] = float(max(min(ov.value, caps.get(pid, TEAM_MINUTES_BUDGET)), 0.0))
            continue

        raw = float(max(float(p.get("raw_minutes", 0.0) or 0.0), 0.0))
        source = str(p.get("source", "prior")).strip().lower()
        w = max(raw, 0.1)
        rank_decay = max(0.25, 1.0 - 0.065 * float(rank))
        w *= rank_decay
        if source == "league":
            w *= 0.82
        elif source == "recent":
            w *= 1.0
        elif source == "prior":
            w *= 0.60
        if rank >= ROTATION_CORE_PLAYERS:
            w *= 0.30
        if rank >= ROTATION_CORE_PLAYERS + 2:
            w *= 0.10
        if status == "QUESTIONABLE":
            w *= 0.70
        elif status == "DOUBTFUL":
            w *= 0.35
        weights[pid] = float(max(w, 0.01))

        cap = float(caps.get(pid, 0.0))
        if rank >= ROTATION_CORE_PLAYERS:
            cap = min(cap, 20.0)
        if rank >= ROTATION_CORE_PLAYERS + 2:
            cap = min(cap, 14.0)
        caps[pid] = float(max(cap, 0.0))

        floor = 0.0
        if rank < 5:
            floor = 8.0
        elif rank < ROTATION_CORE_PLAYERS:
            floor = 4.0
        elif rank < ROTATION_CORE_PLAYERS + 2:
            floor = 1.0
        if source == "prior" and rank >= ROTATION_CORE_PLAYERS:
            floor = 0.0
        floors[pid] = float(max(min(floor, caps.get(pid, 0.0)), 0.0))

    fixed_total = float(sum(fixed_target.values()))
    if fixed_total > TEAM_MINUTES_BUDGET and fixed_total > 0:
        scale = TEAM_MINUTES_BUDGET / fixed_total
        fixed_target = {pid: float(val * scale) for pid, val in fixed_target.items()}
        fixed_total = float(sum(fixed_target.values()))
    remaining = float(max(TEAM_MINUTES_BUDGET - fixed_total, 0.0))

    floor_total = float(sum(floors.values()))
    floor_scale = 1.0
    if floor_total > 0 and floor_total > remaining:
        floor_scale = remaining / floor_total
    for pid, floor in floors.items():
        allocations[pid] = float(max(allocations.get(pid, 0.0), floor * floor_scale))
    remaining_after_floors = float(
        max(remaining - sum(allocations.get(pid, 0.0) for pid in floors.keys()), 0.0)
    )

    open_ids = [
        pid
        for pid in weights.keys()
        if pid not in fixed_target and allocations.get(pid, 0.0) < caps.get(pid, 0.0) - 1e-6
    ]
    for _ in range(28):
        if remaining_after_floors <= 1e-6 or not open_ids:
            break
        wsum = float(sum(weights.get(pid, 0.0) for pid in open_ids))
        if wsum <= 0:
            break
        progressed = 0.0
        for pid in list(open_ids):
            room = float(max(caps.get(pid, 0.0) - allocations.get(pid, 0.0), 0.0))
            if room <= 0:
                continue
            add = min(remaining_after_floors * (weights.get(pid, 0.0) / wsum), room)
            if add <= 0:
                continue
            allocations[pid] = float(allocations.get(pid, 0.0) + add)
            progressed += add
            remaining_after_floors -= add
        open_ids = [
            pid
            for pid in open_ids
            if allocations.get(pid, 0.0) < caps.get(pid, 0.0) - 1e-6
        ]
        if progressed <= 1e-8:
            break

    for pid, val in fixed_target.items():
        allocations[pid] = float(val)
    for p in players:
        pid = int(p["player_id"])
        if _status_token(p.get("status")) == "OUT":
            allocations[pid] = 0.0

    total = float(sum(allocations.values()))
    delta = float(TEAM_MINUTES_BUDGET - total)
    if abs(delta) >= 0.05:
        candidates = [int(p["player_id"]) for p in ordered if _status_token(p.get("status")) != "OUT"]
        for pid in candidates:
            cap = float(caps.get(pid, _status_minutes_cap("AVAILABLE")))
            cur = float(allocations.get(pid, 0.0))
            if delta > 0 and cur < cap:
                add = min(delta, cap - cur)
                allocations[pid] = float(cur + add)
                delta -= add
            elif delta < 0 and cur > 0:
                sub = min(-delta, cur)
                allocations[pid] = float(cur - sub)
                delta += sub
            if abs(delta) < 0.05:
                break

    final_total = float(sum(allocations.values()))
    diag = {
        "players": float(len(players)),
        "fixed_players": float(len(fixed_target)),
        "weighted_players": float(len(weights)),
        "minutes_total": final_total,
        "delta_vs_240": float(final_total - TEAM_MINUTES_BUDGET),
    }
    return {int(pid): float(max(val, 0.0)) for pid, val in allocations.items()}, diag


def _enforce_team_minutes_budget_overrides(
    gs: GameState,
    *,
    injury_by_name: Dict[str, str],
    injury_by_id: Dict[int, str],
    league_minutes_map: Dict[int, float],
    cache: SQLiteCache,
    season: str,
    recent_minutes_cache: Dict[int, Optional[float]],
    manual_locked_name_keys: set[str],
) -> Dict[str, float]:
    by_name_key = {name_key(k): _status_token(v) for k, v in (injury_by_name or {}).items() if name_key(k)}
    by_id_key = {int(k): _status_token(v) for k, v in (injury_by_id or {}).items()}
    summary = {
        "teams": 0.0,
        "players": 0.0,
        "total_abs_delta_before": 0.0,
        "total_abs_delta_after": 0.0,
    }

    for roster in [gs.team_roster_by_name, gs.opp_roster_by_name]:
        if not roster:
            continue
        players: List[Dict[str, object]] = []
        for roster_name, pid in roster.items():
            pid_int = int(pid)
            key = name_key(roster_name)
            status = _status_token(by_id_key.get(pid_int) or by_name_key.get(key) or "AVAILABLE")
            override = _lookup_minutes_override(gs, roster_name)
            if pid_int not in recent_minutes_cache:
                recent_minutes_cache[pid_int] = _load_cached_recent_minutes_for_player(
                    cache=cache,
                    season=season,
                    player_id=pid_int,
                )
            recent_base = recent_minutes_cache.get(pid_int)
            league_base = league_minutes_map.get(pid_int)
            if isinstance(override, MinutesOverride) and override.kind == "target":
                raw = float(max(override.value, 0.0))
                source = "override"
            elif recent_base is not None and league_base is not None:
                raw = float(max(0.62 * float(recent_base) + 0.38 * float(league_base), 0.0))
                source = "blend"
            elif recent_base is not None:
                raw = float(max(recent_base, 0.0))
                source = "recent"
            elif league_base is not None:
                raw = float(max(league_base, 0.0))
                source = "league"
            else:
                raw = 14.0
                source = "prior"
            raw = float(raw * _status_minutes_multiplier(status))
            players.append(
                {
                    "player_name": str(roster_name),
                    "player_id": pid_int,
                    "status": status,
                    "override": override,
                    "raw_minutes": raw,
                    "source": source,
                }
            )

        if not players:
            continue

        non_out_players = [p for p in players if _status_token(p.get("status")) != "OUT"]
        should_allocate_budget = bool(len(non_out_players) >= MIN_PLAYERS_FOR_BUDGET_ALLOCATION)

        if not should_allocate_budget:
            for p in players:
                name = norm_name(str(p.get("player_name", "")))
                status = _status_token(p.get("status"))
                ov = p.get("override")
                cap = float(_status_minutes_cap(status))
                if isinstance(ov, MinutesOverride) and ov.kind == "cap":
                    cap = min(cap, float(max(ov.value, 0.0)))
                if isinstance(ov, MinutesOverride) and ov.kind == "target" and name_key(name) in manual_locked_name_keys:
                    target = float(max(min(ov.value, cap), 0.0))
                else:
                    target = float(max(min(float(p.get("raw_minutes", 0.0) or 0.0), cap), 0.0))
                if status == "OUT":
                    target = 0.0
                gs.overrides.minutes_overrides[name] = MinutesOverride(kind="target", value=target)
            continue

        total_before = float(sum(float(p.get("raw_minutes", 0.0) or 0.0) for p in players))
        alloc, diag = _allocate_roster_minutes_to_budget(
            players,
            manual_locked_name_keys=manual_locked_name_keys,
        )
        summary["teams"] += 1.0
        summary["players"] += float(len(players))
        summary["total_abs_delta_before"] += float(abs(total_before - TEAM_MINUTES_BUDGET))
        summary["total_abs_delta_after"] += float(abs(float(diag.get("minutes_total", 0.0)) - TEAM_MINUTES_BUDGET))

        for p in players:
            name = norm_name(str(p.get("player_name", "")))
            pid_int = int(p.get("player_id", 0) or 0)
            status = _status_token(p.get("status"))
            target = 0.0 if status == "OUT" else float(max(alloc.get(pid_int, 0.0), 0.0))
            gs.overrides.minutes_overrides[name] = MinutesOverride(kind="target", value=target)

    return summary

def _resolve_player_id(gs: GameState, player_name: str) -> Optional[int]:
    pid = _roster_player_id(gs.team_roster_by_name, player_name)
    if pid is None:
        pid = _roster_player_id(gs.opp_roster_by_name, player_name)
    return pid

def hydrate_game_state(gs: GameState, season: str, cache: SQLiteCache) -> None:
    team = find_team_by_abbr(gs.player_team_abbr)
    opp = find_team_by_abbr(gs.opponent_abbr)
    gs.player_team_id = int(team["id"])
    gs.opponent_team_id = int(opp["id"])
    gs.player_team_name = team["full_name"]
    gs.opponent_team_name = opp["full_name"]
    gs.team_roster_by_name = get_roster_name_to_id(gs.player_team_id, season, cache)
    gs.opp_roster_by_name = get_roster_name_to_id(gs.opponent_team_id, season, cache)

def resolve_statuses(gs: GameState, injury_by_name: Optional[Dict[str, str]], injury_by_id: Optional[Dict[int, str]]) -> None:
    resolved_name: Dict[str, str] = {}
    resolved_id: Dict[int, str] = {}
    unmatched_injury_names: set[str] = set()
    valid_player_ids = set(int(pid) for pid in gs.team_roster_by_name.values()) | set(int(pid) for pid in gs.opp_roster_by_name.values())
    id_to_name = _roster_id_to_name(gs.team_roster_by_name)
    id_to_name.update(_roster_id_to_name(gs.opp_roster_by_name))

    # Overrides win
    for name in gs.overrides.team_out | gs.overrides.opp_out:
        nn = name_key(name)
        if not nn:
            continue
        resolved_name[nn] = "OUT"
        pid = _resolve_player_id(gs, nn)
        if pid is not None:
            resolved_id[int(pid)] = "OUT"
    for name in gs.overrides.team_in | gs.overrides.opp_in:
        nn = name_key(name)
        if not nn:
            continue
        if resolved_name.get(nn) == "OUT":
            raise PipelineError(f"{gs.game_key}: override conflict for {nn}")
        resolved_name[nn] = "AVAILABLE"
        pid = _resolve_player_id(gs, nn)
        if pid is not None:
            resolved_id[int(pid)] = "AVAILABLE"

    if injury_by_name:
        for name, st in injury_by_name.items():
            nn = name_key(name)
            pid = _resolve_player_id(gs, nn)
            if nn and nn not in resolved_name and pid is not None:
                resolved_name[nn] = st
            elif nn and pid is None:
                unmatched_injury_names.add(norm_name(name))
    if injury_by_id:
        for pid, st in injury_by_id.items():
            pid_int = int(pid)
            if pid_int in valid_player_ids and pid_int not in resolved_id:
                resolved_id[pid_int] = st
                mapped_name = id_to_name.get(pid_int)
                mapped_key = name_key(mapped_name) if mapped_name else ""
                if mapped_key and mapped_key not in resolved_name:
                    resolved_name[mapped_key] = st

    gs.resolved_status_by_name = resolved_name
    gs.resolved_status_by_player_id = resolved_id
    gs.unmapped_injury_names = sorted(unmatched_injury_names)

def build_scenarios(gs: GameState, max_players: int = 2) -> None:
    # unresolved GTDs not overridden
    unresolved: List[Tuple[str, str, str]] = []  # (name, side, status)
    team_ids = set(int(pid) for pid in gs.team_roster_by_name.values())
    for name, st in gs.resolved_status_by_name.items():
        if st not in STATUS_PRIORS:
            continue
        pid = _resolve_player_id(gs, name)
        if pid is None:
            continue
        side = "team" if pid in team_ids else "opp"
        unresolved.append((name, side, st))

    # Prioritize QUESTIONABLE/DOUBTFUL over PROBABLE; cap scenario explosion
    priority = {"QUESTIONABLE": 0, "DOUBTFUL": 1, "PROBABLE": 2}
    unresolved = sorted(unresolved, key=lambda x: priority.get(x[2], 9))[:max_players]

    if not unresolved:
        gs.scenarios = [Scenario(name="base", prob=1.0,
                                 team_out={name_key(x) for x in gs.overrides.team_out if name_key(x)},
                                 team_in={name_key(x) for x in gs.overrides.team_in if name_key(x)},
                                 opp_out={name_key(x) for x in gs.overrides.opp_out if name_key(x)},
                                 opp_in={name_key(x) for x in gs.overrides.opp_in if name_key(x)})]
        return

    scenarios = [Scenario(name="base", prob=1.0,
                          team_out={name_key(x) for x in gs.overrides.team_out if name_key(x)},
                          team_in={name_key(x) for x in gs.overrides.team_in if name_key(x)},
                          opp_out={name_key(x) for x in gs.overrides.opp_out if name_key(x)},
                          opp_in={name_key(x) for x in gs.overrides.opp_in if name_key(x)})]
    for name, side, st in unresolved:
        p_in, p_out = STATUS_PRIORS[st]
        display_name = _display_name_from_game(gs, name)
        nxt: List[Scenario] = []
        for sc in scenarios:
            # IN
            if side == "team":
                nxt.append(Scenario(
                    name=sc.name + f"|{display_name}:IN",
                    prob=sc.prob * p_in,
                    team_out=set(sc.team_out) - {name},
                    team_in=set(sc.team_in) | {name},
                    opp_out=set(sc.opp_out),
                    opp_in=set(sc.opp_in),
                ))
                nxt.append(Scenario(
                    name=sc.name + f"|{display_name}:OUT",
                    prob=sc.prob * p_out,
                    team_out=set(sc.team_out) | {name},
                    team_in=set(sc.team_in) - {name},
                    opp_out=set(sc.opp_out),
                    opp_in=set(sc.opp_in),
                ))
            else:
                nxt.append(Scenario(
                    name=sc.name + f"|{display_name}:IN",
                    prob=sc.prob * p_in,
                    team_out=set(sc.team_out),
                    team_in=set(sc.team_in),
                    opp_out=set(sc.opp_out) - {name},
                    opp_in=set(sc.opp_in) | {name},
                ))
                nxt.append(Scenario(
                    name=sc.name + f"|{display_name}:OUT",
                    prob=sc.prob * p_out,
                    team_out=set(sc.team_out),
                    team_in=set(sc.team_in),
                    opp_out=set(sc.opp_out) | {name},
                    opp_in=set(sc.opp_in) - {name},
                ))
        scenarios = nxt

    total = sum(s.prob for s in scenarios) or 1.0
    gs.scenarios = [Scenario(name=s.name, prob=s.prob/total, team_out=s.team_out, team_in=s.team_in, opp_out=s.opp_out, opp_in=s.opp_in) for s in scenarios]

def _combine_scenarios(details: List[Dict[str, float]]) -> Tuple[float, float]:
    mu = sum(d["prob"] * d["mu"] for d in details)
    second = sum(d["prob"] * (d["var"] + d["mu"] ** 2) for d in details)
    var = max(second - mu ** 2, 1e-8)
    return mu, var


def _market_prob_and_edge(prob: float, odds_american: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    if odds_american is None:
        return None, None
    market_prob = american_to_implied_prob(int(odds_american))
    edge_cents = (float(prob) - float(market_prob)) * 100.0
    return float(market_prob), float(edge_cents)


def _pick_recommended_side(
    edge_over: Optional[float],
    edge_under: Optional[float],
) -> Optional[str]:
    if edge_over is None and edge_under is None:
        return None
    if edge_over is None:
        return "under"
    if edge_under is None:
        return "over"
    return "over" if float(edge_over) >= float(edge_under) else "under"

def project_prop(
    cache: SQLiteCache,
    season: str,
    gs: GameState,
    req: PropRequest,
    feature_gate: Optional[Dict[str, float]] = None,
) -> ProjectionResult:
    flags: List[str] = []
    drivers: List[str] = []

    player_name = norm_name(req.player_name)
    player_key = name_key(player_name)
    player_id = _resolve_player_id(gs, player_key)
    mo = _lookup_minutes_override(gs, player_key)
    minutes_override = (mo.kind, mo.value) if mo else None
    if mo:
        flags.append(f"minutes_override:{mo.kind}:{mo.value:g}")
    if gs.unmapped_injury_names:
        flags.append(f"injury_unmapped_names:{len(gs.unmapped_injury_names)}")

    status = gs.resolved_status_by_player_id.get(player_id) if player_id is not None else None
    if status is None:
        status = gs.resolved_status_by_name.get(player_key)
    if status == "OUT":
        mu = 0.0
        sigma = 0.01
        raw_p_over = 0.0
        p_over = 0.0
        p_under = 1.0
        dist = DistributionSpec(name="normal", params={"mu": mu, "sigma": sigma})
        odds_over = req.odds_over_american if req.odds_over_american is not None else req.odds_american
        odds_under = req.odds_under_american
        fair_over = implied_prob_to_american(1e-6)
        fair_under = implied_prob_to_american(1.0 - 1e-6)
        market_prob_over, edge_cents_over = _market_prob_and_edge(p_over, odds_over)
        market_prob_under, edge_cents_under = _market_prob_and_edge(p_under, odds_under)
        recommended_side = _pick_recommended_side(edge_cents_over, edge_cents_under)
        recommended_odds = odds_over if recommended_side == "over" else odds_under if recommended_side == "under" else None
        model_prob_side = p_over if recommended_side == "over" else p_under if recommended_side == "under" else None
        market_prob_side = market_prob_over if recommended_side == "over" else market_prob_under if recommended_side == "under" else None
        edge_cents_side = edge_cents_over if recommended_side == "over" else edge_cents_under if recommended_side == "under" else None
        eligible = bool(recommended_side and (recommended_odds is not None))
        return ProjectionResult(
            game_key=req.game_key, asof_utc=gs.asof_utc,
            player_name=player_name, market=req.market, line=req.line, odds_american=odds_over,
            mu=mu, sigma=sigma, dist=dist,
            raw_p_over=raw_p_over, p_over=p_over, p_under=p_under,
            fair_over_odds=fair_over, edge_cents_over=edge_cents_over,
            flags=flags + ["player_out"], drivers=["Player marked OUT via override/feed."], scenario_details=[],
            odds_over_american=odds_over,
            odds_under_american=odds_under,
            bookmaker=req.bookmaker,
            event_id=req.event_id,
            event_start_utc=req.event_start_utc,
            fair_under_odds=fair_under,
            market_prob_over=market_prob_over,
            market_prob_under=market_prob_under,
            edge_cents_under=edge_cents_under,
            recommended_side=recommended_side,
            recommended_odds_american=recommended_odds,
            model_prob_side=model_prob_side,
            market_prob_side=market_prob_side,
            edge_cents_side=edge_cents_side,
            eligible_for_recommendation=eligible,
        )

    scenario_details: List[Dict[str, float]] = []
    dist_name = "normal"
    for sc in gs.scenarios or [Scenario(name="base", prob=1.0,
                                        team_out={name_key(x) for x in gs.overrides.team_out if name_key(x)},
                                        team_in={name_key(x) for x in gs.overrides.team_in if name_key(x)},
                                        opp_out={name_key(x) for x in gs.overrides.opp_out if name_key(x)},
                                        opp_in={name_key(x) for x in gs.overrides.opp_in if name_key(x)})]:
        if player_key in sc.team_out:
            mu_s, sigma_s, dist_name_s = 0.0, 0.01, "normal"
            drv = ["player_out_in_scenario"]
        else:
            team_out_names = [_display_name_from_key(gs.team_roster_by_name, x) for x in sorted(sc.team_out - {player_key})]
            opp_out_names = [_display_name_from_key(gs.opp_roster_by_name, x) for x in sorted(sc.opp_out)]
            mu_s, sigma_s, dist_name_s, drv, meta = project_market(
                cache=cache,
                season=season,
                player_name=player_name,
                player_team_abbr=gs.player_team_abbr,
                opp_abbr=gs.opponent_abbr,
                is_home=gs.is_home,
                market=req.market,
                team_out=team_out_names,
                opponent_out=opp_out_names,
                minutes_override=minutes_override,
                game_date_local=gs.game_date_local,
                spread=gs.spread,
                vegas_team_total=gs.vegas_team_total,
                feature_gate=feature_gate,
            )
        dist_name = dist_name_s
        scenario_details.append({
            "prob": float(sc.prob),
            "mu": float(mu_s),
            "var": float(max(sigma_s ** 2, 1e-8)),
        })
        if len(drivers) < 30:
            drivers.extend([f"{sc.name}:{d}" for d in drv[:8]])

    mu, var = _combine_scenarios(scenario_details)
    sigma = var ** 0.5
    if len(scenario_details) > 1:
        flags.append("injury_scenarios_present")

    if dist_name == "neg_binom":
        raw_p_over = negbin_over_prob(mu, var, req.line)
        dist = DistributionSpec(name="neg_binom", params={"mu": mu, "var": var})
    else:
        raw_p_over = normal_over_prob(mu, sigma, req.line)
        dist = DistributionSpec(name="normal", params={"mu": mu, "sigma": sigma})

    # no fitted calibrator yet; preserve field
    p_over = raw_p_over
    p_under = 1.0 - p_over

    odds_over = req.odds_over_american if req.odds_over_american is not None else req.odds_american
    odds_under = req.odds_under_american
    fair_over = implied_prob_to_american(max(1e-6, min(1 - 1e-6, p_over)))
    fair_under = implied_prob_to_american(max(1e-6, min(1 - 1e-6, p_under)))
    market_prob_over, edge_cents_over = _market_prob_and_edge(p_over, odds_over)
    market_prob_under, edge_cents_under = _market_prob_and_edge(p_under, odds_under)
    recommended_side = _pick_recommended_side(edge_cents_over, edge_cents_under)
    recommended_odds = odds_over if recommended_side == "over" else odds_under if recommended_side == "under" else None
    model_prob_side = p_over if recommended_side == "over" else p_under if recommended_side == "under" else None
    market_prob_side = market_prob_over if recommended_side == "over" else market_prob_under if recommended_side == "under" else None
    edge_cents_side = edge_cents_over if recommended_side == "over" else edge_cents_under if recommended_side == "under" else None
    eligible = bool(recommended_side and (recommended_odds is not None))

    return ProjectionResult(
        game_key=req.game_key, asof_utc=gs.asof_utc,
        player_name=player_name, market=req.market, line=req.line, odds_american=odds_over,
        mu=float(mu), sigma=float(sigma), dist=dist,
        raw_p_over=float(raw_p_over), p_over=float(p_over), p_under=float(p_under),
        fair_over_odds=fair_over, edge_cents_over=edge_cents_over,
        flags=flags, drivers=drivers[:40], scenario_details=scenario_details,
        odds_over_american=odds_over,
        odds_under_american=odds_under,
        bookmaker=req.bookmaker,
        event_id=req.event_id,
        event_start_utc=req.event_start_utc,
        fair_under_odds=fair_under,
        market_prob_over=market_prob_over,
        market_prob_under=market_prob_under,
        edge_cents_under=edge_cents_under,
        recommended_side=recommended_side,
        recommended_odds_american=recommended_odds,
        model_prob_side=model_prob_side,
        market_prob_side=market_prob_side,
        edge_cents_side=edge_cents_side,
        eligible_for_recommendation=eligible,
    )

def run_pipeline(
    games: Dict[str, GameState],
    props: List[PropRequest],
    season: str,
    refresh: str = "none",
    only_game_key: Optional[str] = None,
    cache_db: str = "nba_props_cache.sqlite",
    feature_gate: Optional[Dict[str, float]] = None,
    injury_source: str = "auto",
    minutes_source: str = "auto",
):
    cache = SQLiteCache(cache_db)

    injury_snapshot = None
    injuries = None
    minutes_by_team: Dict[str, Dict[str, MinutesProjectionItem]] = {}
    minutes_override_applied = 0
    minutes_status_hints = 0
    league_fallback_applied = 0
    recent_fallback_applied = 0
    blended_fallback_applied = 0
    redistributed_fallback_players = 0
    redistributed_fallback_minutes = 0.0
    minute_budget_teams = 0.0
    minute_budget_players = 0.0
    minute_budget_abs_delta_before = 0.0
    minute_budget_abs_delta_after = 0.0
    league_minutes_map: Dict[int, float] = {}
    recent_minutes_cache: Dict[int, Optional[float]] = {}

    selected_games = [
        gs for gk, gs in games.items()
        if not only_game_key or gk == only_game_key
    ]
    team_abbrs = sorted({
        str(gs.player_team_abbr).upper()
        for gs in selected_games
    } | {
        str(gs.opponent_abbr).upper()
        for gs in selected_games
    })

    if refresh in {"injuries", "all"}:
        ts, injuries, source_meta = fetch_injuries_with_fallback(source=injury_source)
        injury_snapshot = {"asof_utc": ts.isoformat(), **source_meta}
        if minutes_source == "auto":
            league_minutes_map = _load_league_minutes_map(cache, season)
        if minutes_source in {"auto", "rotowire"} and team_abbrs:
            try:
                mts, minutes_by_team, team_errors = fetch_rotowire_projected_minutes(team_abbrs)
                minute_count = sum(len(x) for x in minutes_by_team.values())
                injury_snapshot["minutes_projection"] = {
                    "asof_utc": mts.isoformat(),
                    "source": "rotowire" if minutes_source == "rotowire" else "rotowire+cache_fallback",
                    "teams_requested": team_abbrs,
                    "teams_with_data": sorted([t for t in team_abbrs if len(minutes_by_team.get(t, {})) > 0]),
                    "team_errors": team_errors,
                    "count": int(minute_count),
                    "league_fallback_player_pool": int(len(league_minutes_map)),
                }
            except Exception as exc:
                injury_snapshot["minutes_projection"] = {
                    "asof_utc": ts.isoformat(),
                    "source": "rotowire" if minutes_source == "rotowire" else "rotowire+cache_fallback",
                    "teams_requested": team_abbrs,
                    "teams_with_data": [],
                    "team_errors": {"_global": str(exc)},
                    "count": 0,
                    "league_fallback_player_pool": int(len(league_minutes_map)),
                }

    for gk, gs in games.items():
        if only_game_key and gk != only_game_key:
            continue
        hydrate_game_state(gs, season, cache)
        manual_locked_name_keys = {
            name_key(name)
            for name in gs.overrides.minutes_overrides.keys()
            if name_key(name)
        }
        if injuries is not None:
            by_name, by_id = build_team_injury_maps(injuries, gs.team_roster_by_name, gs.opp_roster_by_name)
        else:
            by_name, by_id = {}, {}
        if minutes_by_team:
            by_name, status_count = _merge_status_hints_from_minutes(gs, by_name, minutes_by_team)
            minutes_status_hints += int(status_count)
            minutes_override_applied += int(_apply_minutes_overrides_from_feed(gs, minutes_by_team))
        if minutes_source == "auto":
            fallback_counts = _apply_nonrotowire_minutes_fallback_overrides(
                    gs,
                    injury_by_name=by_name,
                    injury_by_id=by_id,
                    league_minutes_map=league_minutes_map,
                    cache=cache,
                    season=season,
                    recent_minutes_cache=recent_minutes_cache,
            )
            league_fallback_applied += int(fallback_counts.get("league_only", 0))
            recent_fallback_applied += int(fallback_counts.get("recent_only", 0))
            blended_fallback_applied += int(fallback_counts.get("blended", 0))
            redistributed_fallback_players += int(fallback_counts.get("redistributed_players", 0))
            redistributed_fallback_minutes += float(fallback_counts.get("redistributed_minutes", 0.0))

        budget_summary = _enforce_team_minutes_budget_overrides(
            gs,
            injury_by_name=by_name,
            injury_by_id=by_id,
            league_minutes_map=league_minutes_map,
            cache=cache,
            season=season,
            recent_minutes_cache=recent_minutes_cache,
            manual_locked_name_keys=manual_locked_name_keys,
        )
        minute_budget_teams += float(budget_summary.get("teams", 0.0))
        minute_budget_players += float(budget_summary.get("players", 0.0))
        minute_budget_abs_delta_before += float(budget_summary.get("total_abs_delta_before", 0.0))
        minute_budget_abs_delta_after += float(budget_summary.get("total_abs_delta_after", 0.0))

        resolve_statuses(gs, by_name, by_id)
        build_scenarios(gs)

    if injury_snapshot is not None and "minutes_projection" in injury_snapshot:
        injury_snapshot["minutes_projection"]["status_hints_applied"] = int(minutes_status_hints)
        injury_snapshot["minutes_projection"]["minutes_overrides_applied"] = int(minutes_override_applied)
        injury_snapshot["minutes_projection"]["league_fallback_overrides_applied"] = int(league_fallback_applied)
        injury_snapshot["minutes_projection"]["recent_fallback_overrides_applied"] = int(recent_fallback_applied)
        injury_snapshot["minutes_projection"]["blended_fallback_overrides_applied"] = int(blended_fallback_applied)
        injury_snapshot["minutes_projection"]["redistributed_fallback_players"] = int(redistributed_fallback_players)
        injury_snapshot["minutes_projection"]["redistributed_fallback_minutes"] = float(redistributed_fallback_minutes)
        injury_snapshot["minutes_projection"]["minute_budget_allocator_teams"] = int(minute_budget_teams)
        injury_snapshot["minutes_projection"]["minute_budget_allocator_players"] = int(minute_budget_players)
        injury_snapshot["minutes_projection"]["minute_budget_abs_delta_before"] = float(minute_budget_abs_delta_before)
        injury_snapshot["minutes_projection"]["minute_budget_abs_delta_after"] = float(minute_budget_abs_delta_after)

    results: List[ProjectionResult] = []
    attempted = 0
    skipped = 0
    for pr in props:
        if only_game_key and pr.game_key != only_game_key:
            continue
        attempted += 1
        gs = games.get(pr.game_key)
        if not gs:
            raise PipelineError(f"Missing GameState for {pr.game_key}")
        try:
            results.append(project_prop(cache, season, gs, pr, feature_gate=feature_gate))
        except Exception as exc:
            skipped += 1
            print(
                "[pipeline] warning: skipping prop after projection error "
                f"game={pr.game_key} player={pr.player_name} market={pr.market}: {exc}"
            )
            continue

    if attempted > 0 and len(results) == 0:
        raise PipelineError("All props failed projection.")
    if skipped > 0:
        print(f"[pipeline] skipped props due to projection errors: {skipped}/{attempted}")

    return results, injury_snapshot
