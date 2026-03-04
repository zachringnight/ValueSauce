from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


GENERAL_RANGES: List[str] = ["Catch and Shoot", "Pullups", "Transition"]
RANGE_KEY = {
    "Catch and Shoot": "catch",
    "Pullups": "pullup",
    "Transition": "transition",
}


def _norm_col(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    mapping = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(str(cand))
        if key in mapping:
            return mapping[key]
    return None


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pct_rank(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return vals.rank(method="average", pct=True).astype(float)


def _safe_col(df: pd.DataFrame, candidates: Iterable[str], default: object = np.nan) -> pd.Series:
    col = _find_col(df, candidates)
    if not col:
        return pd.Series(default, index=df.index)
    return df[col]


def _parse_player_range(df: pd.DataFrame, general_range: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "player_id": _to_num(_safe_col(df, ["PLAYER_ID", "playerId"])),
            "player_name": _safe_col(df, ["PLAYER_NAME", "playerName"]).fillna("").astype(str).str.strip(),
            "team_abbr": _safe_col(
                df,
                ["PLAYER_LAST_TEAM_ABBREVIATION", "TEAM_ABBREVIATION", "teamAbbreviation", "TEAM"],
                "",
            )
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper(),
            "gp": _to_num(_safe_col(df, ["GP"])),
            "fga_frequency": _to_num(_safe_col(df, ["FGA_FREQUENCY"])),
            "fg3a_frequency": _to_num(_safe_col(df, ["FG3A_FREQUENCY"])),
            "fg3m": _to_num(_safe_col(df, ["FG3M"])),
            "fg3a": _to_num(_safe_col(df, ["FG3A"])),
            "fg3_pct": _to_num(_safe_col(df, ["FG3_PCT"])),
            "general_range": str(general_range),
        }
    )
    out = out[out["player_name"] != ""].copy()
    out = out.drop_duplicates(["player_name", "team_abbr"], keep="first")
    return out.reset_index(drop=True)


def _parse_team_defense_range(df: pd.DataFrame, general_range: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "team_id": _to_num(_safe_col(df, ["TEAM_ID", "teamId"])),
            "team_name": _safe_col(df, ["TEAM_NAME", "teamName"]).fillna("").astype(str).str.strip(),
            "team_abbr": _safe_col(df, ["TEAM_ABBREVIATION", "teamAbbreviation", "TEAM"], "")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper(),
            "gp": _to_num(_safe_col(df, ["GP"])),
            "fga_frequency": _to_num(_safe_col(df, ["FGA_FREQUENCY"])),
            "fg3a_frequency": _to_num(_safe_col(df, ["FG3A_FREQUENCY"])),
            "fg3m_allowed": _to_num(_safe_col(df, ["FG3M"])),
            "fg3a_allowed": _to_num(_safe_col(df, ["FG3A"])),
            "fg3_pct_allowed": _to_num(_safe_col(df, ["FG3_PCT"])),
            "general_range": str(general_range),
        }
    )
    out = out[(out["team_abbr"] != "") | (out["team_name"] != "")].copy()
    out = out.drop_duplicates(["team_abbr", "team_name"], keep="first")
    return out.reset_index(drop=True)


def _playtype_label(row: pd.Series) -> str:
    catch_share = float(row.get("catch_share_3pa") or 0.0)
    pull_share = float(row.get("pullup_share_3pa") or 0.0)
    transition_share = float(row.get("transition_share_3pa") or 0.0)
    catch_3pa = float(row.get("catch_fg3a") or 0.0)
    pull_3pa = float(row.get("pullup_fg3a") or 0.0)
    transition_3pa = float(row.get("transition_fg3a") or 0.0)

    if catch_share >= 0.56 and catch_3pa >= 1.0:
        return "Catch-and-Shoot Specialist"
    if pull_share >= 0.34 and pull_3pa >= 1.5:
        return "Off-Dribble Creator"
    if transition_share >= 0.18 and transition_3pa >= 0.8:
        return "Transition 3 Specialist"
    if catch_share >= 0.42 and pull_share >= 0.22 and transition_share >= 0.10:
        return "Hybrid Three-Level Shooter"
    return "Balanced 3PT Profile"


def build_player_playtype_profiles(player_range_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parsed: Dict[str, pd.DataFrame] = {}
    for general_range in GENERAL_RANGES:
        frame = player_range_frames.get(general_range, pd.DataFrame())
        item = _parse_player_range(frame, general_range=general_range)
        if len(item):
            parsed[general_range] = item
    if not parsed:
        return pd.DataFrame()

    metrics = ["gp", "fga_frequency", "fg3a_frequency", "fg3m", "fg3a", "fg3_pct"]
    merge_keys = ["player_id", "player_name", "team_abbr"]
    out: Optional[pd.DataFrame] = None
    for general_range in GENERAL_RANGES:
        item = parsed.get(general_range, pd.DataFrame())
        if len(item) == 0:
            continue
        key = RANGE_KEY[general_range]
        rename_map = {m: f"{key}_{m}" for m in metrics}
        view = item[merge_keys + metrics].rename(columns=rename_map)
        if out is None:
            out = view.copy()
        else:
            out = out.merge(view, on=merge_keys, how="outer")

    if out is None or len(out) == 0:
        return pd.DataFrame()

    out["fg3a_total_pg"] = (
        out.get("catch_fg3a", 0).fillna(0.0)
        + out.get("pullup_fg3a", 0).fillna(0.0)
        + out.get("transition_fg3a", 0).fillna(0.0)
    )
    out["fg3m_total_pg"] = (
        out.get("catch_fg3m", 0).fillna(0.0)
        + out.get("pullup_fg3m", 0).fillna(0.0)
        + out.get("transition_fg3m", 0).fillna(0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fg3_pct_weighted"] = np.where(
            out["fg3a_total_pg"] > 1e-9,
            out["fg3m_total_pg"] / out["fg3a_total_pg"],
            np.nan,
        )
        out["catch_share_3pa"] = out.get("catch_fg3a", 0).fillna(0.0) / out["fg3a_total_pg"].replace(0.0, np.nan)
        out["pullup_share_3pa"] = out.get("pullup_fg3a", 0).fillna(0.0) / out["fg3a_total_pg"].replace(0.0, np.nan)
        out["transition_share_3pa"] = out.get("transition_fg3a", 0).fillna(0.0) / out["fg3a_total_pg"].replace(0.0, np.nan)

    out["catch_fg3_pctile"] = _pct_rank(out.get("catch_fg3_pct", pd.Series(np.nan, index=out.index)))
    out["pullup_fg3_pctile"] = _pct_rank(out.get("pullup_fg3_pct", pd.Series(np.nan, index=out.index)))
    out["transition_fg3_pctile"] = _pct_rank(out.get("transition_fg3_pct", pd.Series(np.nan, index=out.index)))
    out["catch_fg3a_pctile"] = _pct_rank(out.get("catch_fg3a", pd.Series(np.nan, index=out.index)))
    out["pullup_fg3a_pctile"] = _pct_rank(out.get("pullup_fg3a", pd.Series(np.nan, index=out.index)))
    out["transition_fg3a_pctile"] = _pct_rank(out.get("transition_fg3a", pd.Series(np.nan, index=out.index)))

    out["catch_shoot_strength"] = (
        out["catch_share_3pa"].fillna(0.0) * 0.45
        + out["catch_fg3_pctile"].fillna(0.0) * 0.35
        + out["catch_fg3a_pctile"].fillna(0.0) * 0.20
    )
    out["off_dribble_strength"] = (
        out["pullup_share_3pa"].fillna(0.0) * 0.45
        + out["pullup_fg3_pctile"].fillna(0.0) * 0.30
        + out["pullup_fg3a_pctile"].fillna(0.0) * 0.25
    )
    out["transition_3_strength"] = (
        out["transition_share_3pa"].fillna(0.0) * 0.50
        + out["transition_fg3_pctile"].fillna(0.0) * 0.30
        + out["transition_fg3a_pctile"].fillna(0.0) * 0.20
    )
    out["fg3_playtype_label"] = out.apply(_playtype_label, axis=1)
    out = out.sort_values(
        ["fg3a_total_pg", "fg3_pct_weighted", "off_dribble_strength"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def build_team_defense_playtype_profiles(team_range_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parsed: Dict[str, pd.DataFrame] = {}
    for general_range in GENERAL_RANGES:
        frame = team_range_frames.get(general_range, pd.DataFrame())
        item = _parse_team_defense_range(frame, general_range=general_range)
        if len(item):
            parsed[general_range] = item
    if not parsed:
        return pd.DataFrame()

    metrics = ["gp", "fga_frequency", "fg3a_frequency", "fg3m_allowed", "fg3a_allowed", "fg3_pct_allowed"]
    merge_keys = ["team_id", "team_name", "team_abbr"]
    out: Optional[pd.DataFrame] = None
    for general_range in GENERAL_RANGES:
        item = parsed.get(general_range, pd.DataFrame())
        if len(item) == 0:
            continue
        key = RANGE_KEY[general_range]
        rename_map = {m: f"{key}_{m}" for m in metrics}
        view = item[merge_keys + metrics].rename(columns=rename_map)
        if out is None:
            out = view.copy()
        else:
            out = out.merge(view, on=merge_keys, how="outer")

    if out is None or len(out) == 0:
        return pd.DataFrame()

    out["catch_pctile_fg3_pct_allowed"] = _pct_rank(out.get("catch_fg3_pct_allowed", pd.Series(np.nan, index=out.index)))
    out["catch_pctile_fg3a_freq_allowed"] = _pct_rank(
        out.get("catch_fg3a_frequency", pd.Series(np.nan, index=out.index))
    )
    out["pullup_pctile_fg3_pct_allowed"] = _pct_rank(
        out.get("pullup_fg3_pct_allowed", pd.Series(np.nan, index=out.index))
    )
    out["pullup_pctile_fg3a_freq_allowed"] = _pct_rank(
        out.get("pullup_fg3a_frequency", pd.Series(np.nan, index=out.index))
    )
    out["transition_pctile_fg3_pct_allowed"] = _pct_rank(
        out.get("transition_fg3_pct_allowed", pd.Series(np.nan, index=out.index))
    )
    out["transition_pctile_fg3a_freq_allowed"] = _pct_rank(
        out.get("transition_fg3a_frequency", pd.Series(np.nan, index=out.index))
    )

    out["vuln_catch_shoot"] = (
        out["catch_pctile_fg3_pct_allowed"].fillna(0.0) * 0.55
        + out["catch_pctile_fg3a_freq_allowed"].fillna(0.0) * 0.45
    )
    out["vuln_off_dribble"] = (
        out["pullup_pctile_fg3_pct_allowed"].fillna(0.0) * 0.60
        + out["pullup_pctile_fg3a_freq_allowed"].fillna(0.0) * 0.40
    )
    out["vuln_transition_3"] = (
        out["transition_pctile_fg3_pct_allowed"].fillna(0.0) * 0.60
        + out["transition_pctile_fg3a_freq_allowed"].fillna(0.0) * 0.40
    )

    vuln_cols = ["vuln_catch_shoot", "vuln_off_dribble", "vuln_transition_3"]
    label_map = {
        "vuln_catch_shoot": "Catch-and-Shoot",
        "vuln_off_dribble": "Off-Dribble",
        "vuln_transition_3": "Transition 3s",
    }
    out["weakest_against_playtype"] = out[vuln_cols].idxmax(axis=1).map(label_map)
    out["strongest_against_playtype"] = out[vuln_cols].idxmin(axis=1).map(label_map)
    out = out.sort_values(
        ["vuln_catch_shoot", "vuln_off_dribble", "vuln_transition_3"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return out
