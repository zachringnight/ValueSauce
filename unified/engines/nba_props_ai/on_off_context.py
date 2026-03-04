from __future__ import annotations

from typing import Dict, Tuple

from .cache import SQLiteCache
from .nba_data import get_team_onoff
from .utils import safe_clip

def estimate_usage_boost(
    cache: SQLiteCache,
    team_id: int,
    season: str,
    target_player: str,
    absent_player: str,
    stat: str,
) -> float:
    # Uses TeamPlayerOnOffDetails as a backup / mild factor, not a primary source of truth.
    try:
        data = get_team_onoff(team_id, season, cache)
    except Exception:
        return 1.0
    on_df = data.get("on")
    off_df = data.get("off")
    if on_df is None or off_df is None or len(on_df) == 0 or len(off_df) == 0:
        return 1.0
    stat_upper = stat.upper()

    def _row(df, player_name: str):
        mask = df["VS_PLAYER_NAME"].astype(str).str.lower().str.contains(player_name.lower(), na=False)
        rows = df[mask]
        if len(rows) == 0:
            parts = player_name.lower().split()
            for part in parts:
                rows = df[df["VS_PLAYER_NAME"].astype(str).str.lower().str.contains(part, na=False)]
                if len(rows):
                    break
        return rows.iloc[0] if len(rows) else None

    absent_on = _row(on_df, absent_player)
    absent_off = _row(off_df, absent_player)
    target_on = _row(on_df, target_player)
    if absent_on is None or absent_off is None or target_on is None:
        return 1.0
    if stat_upper not in absent_on.index or stat_upper not in absent_off.index or stat_upper not in target_on.index:
        return 1.0

    on_min = float(absent_on.get("MIN", 0) or 0)
    off_min = float(absent_off.get("MIN", 0) or 0)
    if on_min <= 0 or off_min <= 0:
        return 1.0

    on_rate = float(absent_on[stat_upper]) / on_min
    off_rate = float(absent_off[stat_upper]) / off_min
    if on_rate <= 0:
        return 1.0

    target_share = float(target_on[stat_upper]) / max(float(absent_on[stat_upper]), 1e-6)
    target_share = safe_clip(target_share, 0.05, 0.35)

    rate_delta_pct = (off_rate - on_rate) / on_rate
    absorption = min(target_share * 1.15, 0.30)
    factor = 1.0 + rate_delta_pct * absorption
    return safe_clip(factor, 0.92, 1.12)
