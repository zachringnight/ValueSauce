from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import name_key, safe_clip


def _norm_col(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    mapping = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in mapping:
            return mapping[key]
    return None


def _parse_percentish(value) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return float("nan")


def _to_numeric_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    col = _find_col(df, candidates)
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[col].map(_parse_percentish).astype(float)


def _pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).astype(float)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mu = values.mean()
    sigma = values.std(ddof=0)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-9:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return ((values - mu) / sigma).astype(float)


def _weighted_score(df: pd.DataFrame, parts: Dict[str, float]) -> pd.Series:
    total_weight = sum(float(w) for w in parts.values() if float(w) > 0)
    if total_weight <= 0:
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.Series(0.0, index=df.index, dtype=float)
    weight_used = pd.Series(0.0, index=df.index, dtype=float)
    for col, weight in parts.items():
        w = float(weight)
        if w <= 0:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        out.loc[mask] += vals.loc[mask] * w
        weight_used.loc[mask] += w
    with np.errstate(invalid="ignore"):
        out = out / weight_used.replace(0.0, np.nan)
    return out.astype(float)


def _ensure_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


TEAM_ABBR_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "PHO": "PHX",
    "UT": "UTA",
    "WSH": "WAS",
    "BKO": "BKN",
    "CHO": "CHA",
}


def _build_team_alias_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        from nba_api.stats.static import teams as nba_teams

        for team in nba_teams.get_teams():
            abbr = str(team.get("abbreviation") or "").upper()
            city = str(team.get("city") or "").strip()
            nickname = str(team.get("nickname") or "").strip()
            full_name = str(team.get("full_name") or "").strip()
            for raw in [abbr, city, nickname, full_name, f"{city} {nickname}".strip()]:
                key = _norm_col(raw)
                if key:
                    out[key] = abbr
            if abbr == "LAL":
                out[_norm_col("LA Lakers")] = "LAL"
            if abbr == "LAC":
                out[_norm_col("LA Clippers")] = "LAC"
    except Exception:
        # Fallback aliases if nba_api static import is unavailable.
        pass

    fallback_pairs = {
        "ATLANTA": "ATL",
        "BOSTON": "BOS",
        "BROOKLYN": "BKN",
        "CHARLOTTE": "CHA",
        "CHICAGO": "CHI",
        "CLEVELAND": "CLE",
        "DALLAS": "DAL",
        "DENVER": "DEN",
        "DETROIT": "DET",
        "GOLDEN STATE": "GSW",
        "HOUSTON": "HOU",
        "INDIANA": "IND",
        "LA CLIPPERS": "LAC",
        "LA LAKERS": "LAL",
        "LOS ANGELES CLIPPERS": "LAC",
        "LOS ANGELES LAKERS": "LAL",
        "MEMPHIS": "MEM",
        "MIAMI": "MIA",
        "MILWAUKEE": "MIL",
        "MINNESOTA": "MIN",
        "NEW ORLEANS": "NOP",
        "NEW YORK": "NYK",
        "OKLAHOMA CITY": "OKC",
        "ORLANDO": "ORL",
        "PHILADELPHIA": "PHI",
        "PHOENIX": "PHX",
        "PORTLAND": "POR",
        "SACRAMENTO": "SAC",
        "SAN ANTONIO": "SAS",
        "TORONTO": "TOR",
        "UTAH": "UTA",
        "WASHINGTON": "WAS",
    }
    for raw, abbr in fallback_pairs.items():
        out[_norm_col(raw)] = abbr
        out[_norm_col(abbr)] = abbr
    for raw, abbr in TEAM_ABBR_ALIASES.items():
        out[_norm_col(raw)] = abbr
    return out


def _canonical_team_abbr(value: object, alias_map: Dict[str, str]) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    key = _norm_col(text)
    if key in alias_map:
        return alias_map[key]
    if len(text) <= 4 and text.isalpha():
        return TEAM_ABBR_ALIASES.get(text, text)
    if " " in text:
        tail_key = _norm_col(text.split()[-1])
        if tail_key in alias_map:
            return alias_map[tail_key]
    return text


def _team_vulnerability_archetype(row: pd.Series) -> str:
    corner = float(row.get("vuln_vs_corner_spotup") or 0.0)
    above_break = float(row.get("vuln_vs_above_break_movement") or 0.0)
    creator = float(row.get("vuln_vs_pullup_creator") or 0.0)
    volume = float(row.get("vuln_vs_hybrid_volume") or 0.0)
    if corner >= 0.70 and above_break <= 0.55:
        return "Corner-3 Leak"
    if above_break >= 0.70 and corner <= 0.55:
        return "Above-Break Leak"
    if creator >= 0.70:
        return "Creator-3 Leak"
    if volume >= 0.70:
        return "Volume-3 Leak"
    if max(corner, above_break, creator, volume) <= 0.40:
        return "Universal Perimeter Resistance"
    return "Mixed 3PT Vulnerability"


def _shooter_location_type(row: pd.Series) -> str:
    corner_bias = float(row.get("corner_bias_pctile") or 0.0)
    if corner_bias >= 0.72:
        return "Corner-Oriented Shooter"
    if corner_bias <= 0.30:
        return "Above-the-Break Shooter"
    return "Balanced-Zone Shooter"


def _shooter_play_type(row: pd.Series) -> str:
    assisted_pctile = float(row.get("assisted_three_pctile") or 0.0)
    self_creation = float(row.get("self_creation_pctile") or 0.0)
    non_corner_acc = float(row.get("non_corner_accuracy_pctile") or 0.0)
    corner_bias = float(row.get("corner_bias_pctile") or 0.0)
    shotmaking = float(row.get("shotmaking_pctile") or 0.0)
    assisted_abs = pd.to_numeric(pd.Series([row.get("assisted_three_pct")]), errors="coerce").iloc[0]
    has_assisted_abs = np.isfinite(float(assisted_abs)) if assisted_abs is not None else False

    if ((has_assisted_abs and float(assisted_abs) >= 80.0) or assisted_pctile >= 0.75) and corner_bias >= 0.60:
        return "Spot-Up Corner Specialist"
    if ((has_assisted_abs and float(assisted_abs) >= 78.0) or assisted_pctile >= 0.70) and non_corner_acc >= 0.58 and corner_bias < 0.55:
        return "Movement Shooter"
    if ((has_assisted_abs and float(assisted_abs) <= 65.0) or self_creation >= 0.70) and non_corner_acc >= 0.55:
        return "Pull-Up Creator"
    if ((has_assisted_abs and float(assisted_abs) <= 72.0) or self_creation >= 0.62) and shotmaking >= 0.62 and non_corner_acc >= 0.50:
        return "On-Ball Shotmaker"
    if shotmaking >= 0.70:
        return "Hybrid Volume Shooter"
    return "Balanced Floor Spacer"


def _team_vulnerability_for_shooter(row: pd.Series) -> float:
    style = str(row.get("shooter_play_type") or "")
    corner = float(row.get("vuln_vs_corner_spotup") or np.nan)
    above_break = float(row.get("vuln_vs_above_break_movement") or np.nan)
    creator = float(row.get("vuln_vs_pullup_creator") or np.nan)
    volume = float(row.get("vuln_vs_hybrid_volume") or np.nan)

    if style == "Spot-Up Corner Specialist":
        return corner
    if style == "Movement Shooter":
        return float(np.nanmean([above_break, volume]))
    if style in {"Pull-Up Creator", "On-Ball Shotmaker"}:
        return creator
    if style == "Hybrid Volume Shooter":
        return volume
    return float(np.nanmean([corner, above_break]))


def _player_archetype(row: pd.Series) -> str:
    shot = float(row.get("shotmaking_pctile") or 0.0)
    creation = float(row.get("self_creation_pctile") or 0.0)
    defense = float(row.get("defensive_deterrence_pctile") or 0.0)
    minutes = float(row.get("minutes_role_pctile") or 0.0)
    corner_bias = float(row.get("corner_bias_pctile") or 0.0)
    assisted_three = float(row.get("assisted_three_pct") or np.nan)

    if shot >= 0.80 and creation >= 0.65:
        return "Shot-Creator Bomber"
    if shot >= 0.75 and defense >= 0.60:
        return "Two-Way 3&D Anchor"
    if shot >= 0.70 and corner_bias >= 0.65 and np.isfinite(assisted_three) and assisted_three >= 75.0:
        return "Corner Spacer Specialist"
    if shot >= 0.62 and minutes >= 0.62:
        return "High-Volume Floor Spacer"
    if defense >= 0.72 and shot < 0.62:
        return "Defensive Suppressor"
    if shot <= 0.35 and minutes <= 0.40:
        return "Low-Volume Specialist"
    return "Balanced Perimeter Contributor"


def _team_defense_archetype(row: pd.Series) -> str:
    leak = float(row.get("defense_perimeter_leak_index") or 0.0)
    freq = float(row.get("def_all_three_freq_pctile") or 0.0)
    acc = float(row.get("def_all_three_acc_pctile") or 0.0)

    if leak <= 0.25 and freq <= 0.35 and acc <= 0.35:
        return "Perimeter Clampdown"
    if freq >= 0.65 and acc <= 0.45:
        return "High-Volume Contest"
    if freq <= 0.40 and acc >= 0.65:
        return "Low-Volume Leak"
    if leak >= 0.75:
        return "Perimeter Sieve"
    return "Mixed Shell"


def _team_offense_archetype(row: pd.Series) -> str:
    freq = float(row.get("off_all_three_freq_pctile") or 0.0)
    acc = float(row.get("off_all_three_acc_pctile") or 0.0)
    if freq >= 0.68 and acc >= 0.62:
        return "Five-Out Pressure"
    if freq >= 0.68 and acc < 0.45:
        return "Spray Volume"
    if freq < 0.40 and acc >= 0.62:
        return "Selective Precision"
    if freq < 0.40 and acc < 0.45:
        return "Inside Leaning"
    return "Balanced Perimeter"


def _lineup_offense_archetype(row: pd.Series) -> str:
    freq = float(row.get("off_all_three_freq_pctile") or 0.0)
    acc = float(row.get("off_all_three_acc_pctile") or 0.0)
    if freq >= 0.70 and acc >= 0.65:
        return "Flamethrower Five"
    if freq >= 0.70 and acc < 0.45:
        return "Spray-and-Pray"
    if freq <= 0.35 and acc >= 0.62:
        return "Selective Snipers"
    if freq <= 0.35 and acc < 0.45:
        return "Paint-Heavy Group"
    return "Balanced Group"


def _lineup_defense_archetype(row: pd.Series) -> str:
    freq = float(row.get("def_all_three_freq_pctile") or 0.0)
    acc = float(row.get("def_all_three_acc_pctile") or 0.0)
    leak = float(row.get("defense_perimeter_leak_index") or 0.0)
    if leak <= 0.25 and freq <= 0.35 and acc <= 0.35:
        return "Perimeter Lock"
    if freq >= 0.65 and acc <= 0.45:
        return "Chase-Off Contest"
    if freq <= 0.40 and acc >= 0.65:
        return "Overhelp Leak"
    if leak >= 0.75:
        return "Lineup Target"
    return "Mixed Coverage"


@dataclass(frozen=True)
class FG3ProfileInputs:
    players_onoff_opponent_shooting_accuracy: Path
    players_onoff_opponent_shooting_frequency: Path
    players_shooting_accuracy: Path
    players_shooting_overall: Path
    league_defense_shooting_accuracy: Path
    league_defense_shooting_frequency: Path
    league_offense_shooting_accuracy: Path
    league_offense_shooting_frequency: Path
    lineups_defense_shooting_accuracy: Path
    lineups_defense_shooting_frequency: Path
    lineups_offense_shooting_accuracy: Path
    lineups_offense_shooting_frequency: Path
    league_summary: Path


@dataclass(frozen=True)
class FG3ProfileOutputs:
    players: pd.DataFrame
    teams: pd.DataFrame
    lineups: pd.DataFrame
    player_vs_team: pd.DataFrame
    player_vs_lineup: pd.DataFrame
    team_vs_shooter_type: pd.DataFrame
    summary: Dict[str, object]


def load_fg3_inputs(paths: FG3ProfileInputs) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for field, path in paths.__dict__.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required FG3 profile input: {p}")
        out[field] = pd.read_csv(p)
    return out


def _build_players(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    acc = frames["players_shooting_accuracy"].copy()
    overall = frames["players_shooting_overall"].copy()
    on_acc = frames["players_onoff_opponent_shooting_accuracy"].copy()
    on_freq = frames["players_onoff_opponent_shooting_frequency"].copy()
    team_aliases = _build_team_alias_map()

    player_col = _find_col(acc, ["Player"])
    team_col = _find_col(acc, ["Team"])
    pos_col = _find_col(acc, ["Pos"])
    min_col = _find_col(acc, ["MIN"])
    mpg_col = _find_col(acc, ["MPG"])
    if not player_col or not team_col:
        raise ValueError("players_shooting_accuracy must include Player and Team columns.")

    base = pd.DataFrame(
        {
            "player": _ensure_text(acc[player_col]),
            "team": _ensure_text(acc[team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases)),
            "position_bucket": _ensure_text(acc[pos_col]) if pos_col else "",
            "minutes_total": pd.to_numeric(acc[min_col], errors="coerce") if min_col else np.nan,
            "minutes_per_game": pd.to_numeric(acc[mpg_col], errors="coerce") if mpg_col else np.nan,
            "acc_corner_three_pct": _to_numeric_series(acc, ["Corner Three"]),
            "acc_non_corner_three_pct": _to_numeric_series(acc, ["Non Corner"]),
            "acc_all_three_pct": _to_numeric_series(acc, ["All Three"]),
        }
    )
    base["corner_minus_non_corner_acc"] = base["acc_corner_three_pct"] - base["acc_non_corner_three_pct"]

    ov_player_col = _find_col(overall, ["Player"])
    ov_team_col = _find_col(overall, ["Team"])
    overall_view = pd.DataFrame(
        {
            "player": _ensure_text(overall[ov_player_col]) if ov_player_col else "",
            "team": _ensure_text(overall[ov_team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases))
            if ov_team_col
            else "",
            "overall_three_pct": _to_numeric_series(overall, ["3P%"]),
            "assisted_three_pct": _to_numeric_series(overall, ["ASTD% Three", "ASTD%Three"]),
        }
    )

    on_player_col = _find_col(on_acc, ["Player"])
    on_team_col = _find_col(on_acc, ["Team"])
    on_acc_view = pd.DataFrame(
        {
            "player": _ensure_text(on_acc[on_player_col]) if on_player_col else "",
            "team": _ensure_text(on_acc[on_team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases))
            if on_team_col
            else "",
            "onoff_opp_acc_corner_three_delta_pct": _to_numeric_series(on_acc, ["TEAM FG%: Corner Three", "Corner Three"]),
            "onoff_opp_acc_non_corner_three_delta_pct": _to_numeric_series(on_acc, ["TEAM FG%: Non Corner", "Non Corner"]),
            "onoff_opp_acc_all_three_delta_pct": _to_numeric_series(on_acc, ["TEAM FG%: All Three", "All Three"]),
        }
    )

    of_player_col = _find_col(on_freq, ["Player"])
    of_team_col = _find_col(on_freq, ["Team"])
    on_freq_view = pd.DataFrame(
        {
            "player": _ensure_text(on_freq[of_player_col]) if of_player_col else "",
            "team": _ensure_text(on_freq[of_team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases))
            if of_team_col
            else "",
            "onoff_opp_freq_corner_three_delta_pct": _to_numeric_series(on_freq, ["SHOT FREQUENCY: Corner Three", "Corner Three"]),
            "onoff_opp_freq_non_corner_three_delta_pct": _to_numeric_series(on_freq, ["SHOT FREQUENCY: Non Corner", "Non Corner"]),
            "onoff_opp_freq_all_three_delta_pct": _to_numeric_series(on_freq, ["SHOT FREQUENCY: All Three", "All Three"]),
        }
    )

    for frame in [base, overall_view, on_acc_view, on_freq_view]:
        frame["player_key"] = frame["player"].map(name_key)
        frame["join_key"] = frame["player_key"] + "|" + frame["team"].astype(str)

    merged = base.merge(
        overall_view.drop(columns=["player", "team", "player_key"]),
        on="join_key",
        how="left",
    ).merge(
        on_acc_view.drop(columns=["player", "team", "player_key"]),
        on="join_key",
        how="left",
    ).merge(
        on_freq_view.drop(columns=["player", "team", "player_key"]),
        on="join_key",
        how="left",
    )

    merged["self_creation_pct"] = 100.0 - merged["assisted_three_pct"]
    merged["defensive_deterrence_raw"] = (
        -0.50 * merged["onoff_opp_freq_all_three_delta_pct"]
        - 0.30 * merged["onoff_opp_acc_all_three_delta_pct"]
        - 0.10 * merged["onoff_opp_freq_non_corner_three_delta_pct"]
        - 0.10 * merged["onoff_opp_acc_non_corner_three_delta_pct"]
    )
    merged["shotmaking_raw"] = _weighted_score(
        merged,
        {
            "acc_all_three_pct": 0.42,
            "overall_three_pct": 0.38,
            "acc_non_corner_three_pct": 0.15,
            "acc_corner_three_pct": 0.05,
        },
    )

    merged["shotmaking_pctile"] = _pct_rank(merged["shotmaking_raw"])
    merged["assisted_three_pctile"] = _pct_rank(merged["assisted_three_pct"])
    merged["self_creation_pctile"] = _pct_rank(merged["self_creation_pct"])
    merged["defensive_deterrence_pctile"] = _pct_rank(merged["defensive_deterrence_raw"])
    merged["minutes_role_pctile"] = _pct_rank(merged["minutes_per_game"])
    merged["corner_accuracy_pctile"] = _pct_rank(merged["acc_corner_three_pct"])
    merged["non_corner_accuracy_pctile"] = _pct_rank(merged["acc_non_corner_three_pct"])
    merged["corner_bias_pctile"] = _pct_rank(merged["corner_minus_non_corner_acc"])

    merged["fg3_attack_index"] = _weighted_score(
        merged,
        {
            "shotmaking_pctile": 0.52,
            "self_creation_pctile": 0.20,
            "minutes_role_pctile": 0.18,
            "corner_bias_pctile": 0.10,
        },
    )
    merged["fg3_profile_eligible"] = (
        merged["minutes_per_game"].fillna(0.0) >= 8.0
    ) & merged["shotmaking_pctile"].notna() & merged["assisted_three_pctile"].notna() & merged["corner_bias_pctile"].notna()
    merged.loc[~merged["fg3_profile_eligible"], "fg3_attack_index"] = np.nan
    merged["fg3_two_way_index"] = _weighted_score(
        merged,
        {
            "fg3_attack_index": 0.72,
            "defensive_deterrence_pctile": 0.28,
        },
    )
    merged["shooter_location_type"] = merged.apply(_shooter_location_type, axis=1)
    merged["shooter_play_type"] = merged.apply(_shooter_play_type, axis=1)
    merged["fg3_player_archetype"] = merged.apply(_player_archetype, axis=1)
    merged.loc[~merged["fg3_profile_eligible"], "shooter_location_type"] = "Insufficient 3PT Sample"
    merged.loc[~merged["fg3_profile_eligible"], "shooter_play_type"] = "Insufficient 3PT Sample"
    merged.loc[~merged["fg3_profile_eligible"], "fg3_player_archetype"] = "Insufficient 3PT Sample"
    return merged.drop(columns=["player_key"]).sort_values(
        ["fg3_attack_index", "minutes_per_game"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_teams(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dacc = frames["league_defense_shooting_accuracy"].copy()
    dfreq = frames["league_defense_shooting_frequency"].copy()
    oacc = frames["league_offense_shooting_accuracy"].copy()
    ofreq = frames["league_offense_shooting_frequency"].copy()
    summary = frames["league_summary"].copy()
    team_aliases = _build_team_alias_map()

    def _team_view(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        team_col = _find_col(df, ["Team"])
        if not team_col:
            raise ValueError("League shooting table must include Team column.")
        out = pd.DataFrame({"team": _ensure_text(df[team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases))})
        out = out[~out["team"].str.contains("AVERAGE", na=False)].copy()
        out[f"{prefix}_all_three_pct"] = _to_numeric_series(df, ["All Three"])
        out[f"{prefix}_corner_three_pct"] = _to_numeric_series(df, ["Corner Three"])
        out[f"{prefix}_non_corner_three_pct"] = _to_numeric_series(df, ["Non Corner"])
        out[f"{prefix}_rim_pct"] = _to_numeric_series(df, ["Rim"])
        out[f"{prefix}_all_mid_pct"] = _to_numeric_series(df, ["All Mid"])
        return out

    team_def_acc = _team_view(dacc, "def_acc")
    team_def_freq = _team_view(dfreq, "def_freq")
    team_off_acc = _team_view(oacc, "off_acc")
    team_off_freq = _team_view(ofreq, "off_freq")

    s_team_col = _find_col(summary, ["Team"])
    team_summary = pd.DataFrame(
        {"team": _ensure_text(summary[s_team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases)) if s_team_col else ""}
    )
    team_summary = team_summary[~team_summary["team"].str.contains("AVERAGE", na=False)].copy()
    team_summary["def_rating"] = _to_numeric_series(summary, ["Defense"])
    team_summary["off_rating"] = _to_numeric_series(summary, ["Offense"])
    team_summary["last2_def_rating"] = _to_numeric_series(summary, ["LAST 2 WEEKS: Defense"])
    team_summary["last2_off_rating"] = _to_numeric_series(summary, ["LAST 2 WEEKS: Offense"])

    out = (
        team_def_acc.merge(team_def_freq, on="team", how="outer")
        .merge(team_off_acc, on="team", how="outer")
        .merge(team_off_freq, on="team", how="outer")
        .merge(team_summary, on="team", how="left")
    )

    out["def_all_three_acc_pctile"] = _pct_rank(out["def_acc_all_three_pct"])
    out["def_all_three_freq_pctile"] = _pct_rank(out["def_freq_all_three_pct"])
    out["def_corner_acc_pctile"] = _pct_rank(out["def_acc_corner_three_pct"])
    out["def_corner_freq_pctile"] = _pct_rank(out["def_freq_corner_three_pct"])
    out["def_non_corner_acc_pctile"] = _pct_rank(out["def_acc_non_corner_three_pct"])
    out["def_non_corner_freq_pctile"] = _pct_rank(out["def_freq_non_corner_three_pct"])

    out["off_all_three_acc_pctile"] = _pct_rank(out["off_acc_all_three_pct"])
    out["off_all_three_freq_pctile"] = _pct_rank(out["off_freq_all_three_pct"])
    out["off_corner_acc_pctile"] = _pct_rank(out["off_acc_corner_three_pct"])
    out["off_corner_freq_pctile"] = _pct_rank(out["off_freq_corner_three_pct"])
    out["off_non_corner_acc_pctile"] = _pct_rank(out["off_acc_non_corner_three_pct"])
    out["off_non_corner_freq_pctile"] = _pct_rank(out["off_freq_non_corner_three_pct"])

    out["defense_perimeter_leak_index"] = _weighted_score(
        out,
        {
            "def_all_three_acc_pctile": 0.40,
            "def_all_three_freq_pctile": 0.30,
            "def_non_corner_acc_pctile": 0.20,
            "def_corner_acc_pctile": 0.10,
        },
    )
    out["vuln_vs_corner_spotup"] = _weighted_score(
        out,
        {
            "def_corner_freq_pctile": 0.55,
            "def_corner_acc_pctile": 0.45,
        },
    )
    out["vuln_vs_above_break_movement"] = _weighted_score(
        out,
        {
            "def_non_corner_freq_pctile": 0.50,
            "def_non_corner_acc_pctile": 0.50,
        },
    )
    out["vuln_vs_pullup_creator"] = _weighted_score(
        out,
        {
            "def_non_corner_acc_pctile": 0.55,
            "def_non_corner_freq_pctile": 0.25,
            "def_all_three_acc_pctile": 0.20,
        },
    )
    out["vuln_vs_hybrid_volume"] = _weighted_score(
        out,
        {
            "def_all_three_freq_pctile": 0.60,
            "def_all_three_acc_pctile": 0.40,
        },
    )
    out["offense_perimeter_pressure_index"] = _weighted_score(
        out,
        {
            "off_all_three_acc_pctile": 0.40,
            "off_all_three_freq_pctile": 0.35,
            "off_non_corner_freq_pctile": 0.15,
            "off_corner_freq_pctile": 0.10,
        },
    )
    out["team_defense_archetype"] = out.apply(_team_defense_archetype, axis=1)
    out["team_offense_archetype"] = out.apply(_team_offense_archetype, axis=1)
    out["team_vulnerability_archetype"] = out.apply(_team_vulnerability_archetype, axis=1)
    return out.sort_values("defense_perimeter_leak_index", ascending=False, kind="mergesort").reset_index(drop=True)


def _build_lineups(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    oacc = frames["lineups_offense_shooting_accuracy"].copy()
    ofreq = frames["lineups_offense_shooting_frequency"].copy()
    dacc = frames["lineups_defense_shooting_accuracy"].copy()
    dfreq = frames["lineups_defense_shooting_frequency"].copy()
    team_aliases = _build_team_alias_map()

    def _lineup_key_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        team_col = _find_col(df, ["Team"])
        if not team_col:
            raise ValueError("Lineup shooting table must include Team column.")
        pg_col = _find_col(df, ["PG"])
        sg_col = _find_col(df, ["SG"])
        sf_col = _find_col(df, ["SF"])
        pf_col = _find_col(df, ["PF"])
        c_col = _find_col(df, ["C"])
        poss_col = _find_col(df, ["Poss"])

        out = pd.DataFrame(
            {
                "team": _ensure_text(df[team_col]).map(lambda x: _canonical_team_abbr(x, team_aliases)),
                "pg": _ensure_text(df[pg_col]) if pg_col else "",
                "sg": _ensure_text(df[sg_col]) if sg_col else "",
                "sf": _ensure_text(df[sf_col]) if sf_col else "",
                "pf": _ensure_text(df[pf_col]) if pf_col else "",
                "c": _ensure_text(df[c_col]) if c_col else "",
                "possessions": pd.to_numeric(df[poss_col], errors="coerce") if poss_col else np.nan,
                f"{prefix}_all_three_pct": _to_numeric_series(df, ["All Three"]),
                f"{prefix}_corner_three_pct": _to_numeric_series(df, ["Corner Three"]),
                f"{prefix}_non_corner_three_pct": _to_numeric_series(df, ["Non Corner"]),
            }
        )
        out = out[~out["team"].str.contains("AVERAGE", na=False)].copy()
        out["lineup_key"] = (
            out["team"]
            + "|"
            + out["pg"]
            + "|"
            + out["sg"]
            + "|"
            + out["sf"]
            + "|"
            + out["pf"]
            + "|"
            + out["c"]
        )
        return out

    off_acc = _lineup_key_frame(oacc, "off_acc")
    off_freq = _lineup_key_frame(ofreq, "off_freq")
    def_acc = _lineup_key_frame(dacc, "def_acc")
    def_freq = _lineup_key_frame(dfreq, "def_freq")

    out = (
        off_acc.merge(
            off_freq[
                [
                    "lineup_key",
                    "off_freq_all_three_pct",
                    "off_freq_corner_three_pct",
                    "off_freq_non_corner_three_pct",
                ]
            ],
            on="lineup_key",
            how="outer",
        )
        .merge(
            def_acc[
                [
                    "lineup_key",
                    "def_acc_all_three_pct",
                    "def_acc_corner_three_pct",
                    "def_acc_non_corner_three_pct",
                ]
            ],
            on="lineup_key",
            how="outer",
        )
        .merge(
            def_freq[
                [
                    "lineup_key",
                    "def_freq_all_three_pct",
                    "def_freq_corner_three_pct",
                    "def_freq_non_corner_three_pct",
                ]
            ],
            on="lineup_key",
            how="outer",
        )
    )

    for col in ["team", "pg", "sg", "sf", "pf", "c", "possessions"]:
        if col not in out.columns:
            out[col] = np.nan
    out["team"] = _ensure_text(out["team"]).map(lambda x: _canonical_team_abbr(x, team_aliases))
    out["lineup_signature"] = out["pg"] + " | " + out["sg"] + " | " + out["sf"] + " | " + out["pf"] + " | " + out["c"]

    out["off_all_three_acc_pctile"] = _pct_rank(out["off_acc_all_three_pct"])
    out["off_all_three_freq_pctile"] = _pct_rank(out["off_freq_all_three_pct"])
    out["def_all_three_acc_pctile"] = _pct_rank(out["def_acc_all_three_pct"])
    out["def_all_three_freq_pctile"] = _pct_rank(out["def_freq_all_three_pct"])
    out["def_corner_acc_pctile"] = _pct_rank(out["def_acc_corner_three_pct"])
    out["def_corner_freq_pctile"] = _pct_rank(out["def_freq_corner_three_pct"])
    out["def_non_corner_acc_pctile"] = _pct_rank(out["def_acc_non_corner_three_pct"])
    out["def_non_corner_freq_pctile"] = _pct_rank(out["def_freq_non_corner_three_pct"])
    out["possessions_pctile"] = _pct_rank(out["possessions"])

    out["offense_perimeter_pressure_index"] = _weighted_score(
        out,
        {
            "off_all_three_acc_pctile": 0.45,
            "off_all_three_freq_pctile": 0.40,
            "possessions_pctile": 0.15,
        },
    )
    out["defense_perimeter_leak_index"] = _weighted_score(
        out,
        {
            "def_all_three_acc_pctile": 0.40,
            "def_all_three_freq_pctile": 0.35,
            "def_non_corner_acc_pctile": 0.15,
            "def_corner_acc_pctile": 0.10,
        },
    )
    out["lineup_offense_archetype"] = out.apply(_lineup_offense_archetype, axis=1)
    out["lineup_defense_archetype"] = out.apply(_lineup_defense_archetype, axis=1)
    return out.sort_values("defense_perimeter_leak_index", ascending=False, kind="mergesort").reset_index(drop=True)


def _tier_from_score(score: float) -> str:
    if not np.isfinite(score):
        return "Unknown"
    if score >= 0.78:
        return "Elite FG3 Target"
    if score >= 0.66:
        return "Strong FG3 Target"
    if score >= 0.54:
        return "Playable"
    if score >= 0.42:
        return "Neutral"
    return "Avoid"


def _build_player_vs_team(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    p = players[
        [
            "player",
            "team",
            "position_bucket",
            "fg3_attack_index",
            "corner_bias_pctile",
            "shooter_location_type",
            "shooter_play_type",
            "fg3_profile_eligible",
        ]
    ].copy()
    p = p[p["fg3_profile_eligible"] == True].copy()  # noqa: E712
    t = teams[
        [
            "team",
            "team_defense_archetype",
            "team_vulnerability_archetype",
            "defense_perimeter_leak_index",
            "def_corner_freq_pctile",
            "def_non_corner_freq_pctile",
            "def_corner_acc_pctile",
            "def_non_corner_acc_pctile",
            "vuln_vs_corner_spotup",
            "vuln_vs_above_break_movement",
            "vuln_vs_pullup_creator",
            "vuln_vs_hybrid_volume",
        ]
    ].copy()
    p["__k"] = 1
    t["__k"] = 1
    cross = p.merge(t, on="__k", suffixes=("_player", "_opp")).drop(columns="__k")
    cross = cross[cross["team_player"] != cross["team_opp"]].copy()
    cross["player_corner_pref"] = pd.to_numeric(cross["corner_bias_pctile"], errors="coerce").fillna(0.5)
    cross["opp_corner_leak"] = _weighted_score(
        cross,
        {"def_corner_freq_pctile": 0.5, "def_corner_acc_pctile": 0.5},
    )
    cross["opp_non_corner_leak"] = _weighted_score(
        cross,
        {"def_non_corner_freq_pctile": 0.5, "def_non_corner_acc_pctile": 0.5},
    )
    cross["zone_fit"] = (
        cross["player_corner_pref"] * cross["opp_corner_leak"]
        + (1.0 - cross["player_corner_pref"]) * cross["opp_non_corner_leak"]
    )
    cross["fg3_matchup_attack_score"] = _weighted_score(
        cross,
        {
            "fg3_attack_index": 0.60,
            "defense_perimeter_leak_index": 0.30,
            "zone_fit": 0.10,
        },
    ).map(lambda x: float(safe_clip(x, 0.0, 1.0)) if np.isfinite(x) else np.nan)
    cross["fg3_matchup_tier"] = cross["fg3_matchup_attack_score"].map(_tier_from_score)
    cross["team_vulnerability_for_shooter"] = cross.apply(_team_vulnerability_for_shooter, axis=1)
    cross["fg3_style_matchup_score"] = _weighted_score(
        cross,
        {
            "fg3_attack_index": 0.46,
            "team_vulnerability_for_shooter": 0.36,
            "zone_fit": 0.18,
        },
    ).map(lambda x: float(safe_clip(x, 0.0, 1.0)) if np.isfinite(x) else np.nan)
    cross["fg3_style_matchup_tier"] = cross["fg3_style_matchup_score"].map(_tier_from_score)
    cross = cross.rename(columns={"team_player": "player_team", "team_opp": "opponent_team"})
    return cross.sort_values(
        ["fg3_style_matchup_score", "fg3_matchup_attack_score", "fg3_attack_index"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_player_vs_lineup(
    players: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    top_n_per_player: int = 7,
) -> pd.DataFrame:
    p = players[["player", "team", "fg3_attack_index", "corner_bias_pctile", "fg3_profile_eligible"]].copy()
    p = p[p["fg3_profile_eligible"] == True].copy()  # noqa: E712
    l = lineups[
        [
            "lineup_key",
            "team",
            "lineup_signature",
            "possessions",
            "possessions_pctile",
            "defense_perimeter_leak_index",
            "lineup_defense_archetype",
            "def_corner_freq_pctile",
            "def_non_corner_freq_pctile",
            "def_corner_acc_pctile",
            "def_non_corner_acc_pctile",
        ]
    ].copy()
    p["__k"] = 1
    l["__k"] = 1
    cross = p.merge(l, on="__k", suffixes=("_player", "_lineup")).drop(columns="__k")
    cross = cross[cross["team_player"] != cross["team_lineup"]].copy()
    cross["player_corner_pref"] = pd.to_numeric(cross["corner_bias_pctile"], errors="coerce").fillna(0.5)
    cross["lineup_corner_leak"] = _weighted_score(
        cross,
        {"def_corner_freq_pctile": 0.5, "def_corner_acc_pctile": 0.5},
    )
    cross["lineup_non_corner_leak"] = _weighted_score(
        cross,
        {"def_non_corner_freq_pctile": 0.5, "def_non_corner_acc_pctile": 0.5},
    )
    cross["zone_fit"] = (
        cross["player_corner_pref"] * cross["lineup_corner_leak"]
        + (1.0 - cross["player_corner_pref"]) * cross["lineup_non_corner_leak"]
    )
    cross["fg3_lineup_attack_score"] = _weighted_score(
        cross,
        {
            "fg3_attack_index": 0.58,
            "defense_perimeter_leak_index": 0.27,
            "zone_fit": 0.10,
            "possessions_pctile": 0.05,
        },
    ).map(lambda x: float(safe_clip(x, 0.0, 1.0)) if np.isfinite(x) else np.nan)
    cross["fg3_lineup_tier"] = cross["fg3_lineup_attack_score"].map(_tier_from_score)
    cross = cross.rename(
        columns={
            "team_player": "player_team",
            "team_lineup": "lineup_team",
        }
    )
    cross = cross.sort_values(
        ["player", "fg3_lineup_attack_score", "possessions"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    if int(top_n_per_player) > 0:
        cross = cross.groupby("player", as_index=False, sort=False).head(int(top_n_per_player)).reset_index(drop=True)
    return cross


def _build_team_vs_shooter_type(player_vs_team: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "opponent_team",
        "shooter_play_type",
        "shooter_location_type",
        "fg3_style_matchup_score",
        "fg3_matchup_attack_score",
        "team_vulnerability_for_shooter",
        "team_defense_archetype",
        "team_vulnerability_archetype",
    }
    if not needed.issubset(set(player_vs_team.columns)):
        return pd.DataFrame()

    grouped = (
        player_vs_team.groupby(["opponent_team", "shooter_play_type"], as_index=False, sort=False)
        .agg(
            sample_matchups=("player", "count"),
            avg_style_matchup_score=("fg3_style_matchup_score", "mean"),
            median_style_matchup_score=("fg3_style_matchup_score", "median"),
            avg_base_matchup_score=("fg3_matchup_attack_score", "mean"),
            avg_team_vulnerability_for_shooter=("team_vulnerability_for_shooter", "mean"),
            dominant_location_type=("shooter_location_type", lambda s: s.value_counts(dropna=False).index[0]),
            team_defense_archetype=("team_defense_archetype", lambda s: s.iloc[0]),
            team_vulnerability_archetype=("team_vulnerability_archetype", lambda s: s.iloc[0]),
        )
        .reset_index(drop=True)
    )

    grouped["vulnerability_rank_within_team"] = grouped.groupby("opponent_team")["avg_style_matchup_score"].rank(
        method="dense", ascending=False
    )
    grouped["resilience_rank_within_team"] = grouped.groupby("opponent_team")["avg_style_matchup_score"].rank(
        method="dense", ascending=True
    )

    def _label(row: pd.Series) -> str:
        score = float(row.get("avg_style_matchup_score") or np.nan)
        vuln_rank = float(row.get("vulnerability_rank_within_team") or np.nan)
        resist_rank = float(row.get("resilience_rank_within_team") or np.nan)
        if np.isfinite(vuln_rank) and vuln_rank <= 2 and np.isfinite(score) and score >= 0.54:
            return "Team Struggles vs Type"
        if np.isfinite(resist_rank) and resist_rank <= 2 and np.isfinite(score) and score <= 0.48:
            return "Team Handles Type Well"
        return "Neutral vs Type"

    grouped["team_vs_type_assessment"] = grouped.apply(_label, axis=1)
    return grouped.sort_values(
        ["opponent_team", "vulnerability_rank_within_team", "avg_style_matchup_score"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def build_fg3_profiles(
    frames: Dict[str, pd.DataFrame],
    *,
    top_lineups_per_player: int = 7,
) -> FG3ProfileOutputs:
    players = _build_players(frames)
    teams = _build_teams(frames)
    lineups = _build_lineups(frames)
    player_vs_team = _build_player_vs_team(players, teams)
    player_vs_lineup = _build_player_vs_lineup(players, lineups, top_n_per_player=top_lineups_per_player)
    team_vs_shooter_type = _build_team_vs_shooter_type(player_vs_team)
    team_vuln_sample: List[Dict[str, object]] = []
    if len(team_vs_shooter_type) > 0:
        team_vuln_sample = (
            team_vs_shooter_type[
                [
                    "opponent_team",
                    "shooter_play_type",
                    "avg_style_matchup_score",
                    "sample_matchups",
                    "team_vs_type_assessment",
                    "vulnerability_rank_within_team",
                ]
            ]
            .sort_values(["opponent_team", "vulnerability_rank_within_team"], ascending=[True, True], kind="mergesort")
            .groupby("opponent_team", as_index=False, sort=False)
            .head(2)
            .head(60)
            .drop(columns=["vulnerability_rank_within_team"])
            .to_dict(orient="records")
        )

    summary = {
        "rows": {
            "players": int(len(players)),
            "teams": int(len(teams)),
            "lineups": int(len(lineups)),
            "player_vs_team": int(len(player_vs_team)),
            "player_vs_lineup": int(len(player_vs_lineup)),
            "team_vs_shooter_type": int(len(team_vs_shooter_type)),
        },
        "top_player_archetypes": players["fg3_player_archetype"].value_counts(dropna=False).head(12).to_dict(),
        "top_shooter_play_types": players["shooter_play_type"].value_counts(dropna=False).to_dict(),
        "top_team_defense_archetypes": teams["team_defense_archetype"].value_counts(dropna=False).to_dict(),
        "top_team_vulnerability_archetypes": teams["team_vulnerability_archetype"].value_counts(dropna=False).to_dict(),
        "top_lineup_defense_archetypes": lineups["lineup_defense_archetype"].value_counts(dropna=False).to_dict(),
        "top_fg3_targets_vs_teams": player_vs_team[
            [
                "player",
                "player_team",
                "opponent_team",
                "fg3_matchup_attack_score",
                "fg3_style_matchup_score",
                "fg3_matchup_tier",
                "fg3_style_matchup_tier",
                "team_defense_archetype",
                "team_vulnerability_archetype",
                "shooter_play_type",
            ]
        ]
        .head(25)
        .to_dict(orient="records"),
        "top_team_vulnerabilities_by_shooter_type": team_vuln_sample,
    }
    return FG3ProfileOutputs(
        players=players,
        teams=teams,
        lineups=lineups,
        player_vs_team=player_vs_team,
        player_vs_lineup=player_vs_lineup,
        team_vs_shooter_type=team_vs_shooter_type,
        summary=summary,
    )


def write_fg3_profile_outputs(
    outputs: FG3ProfileOutputs,
    *,
    out_dir: Path,
    prefix: str = "fg3",
) -> Dict[str, Path]:
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    files = {
        "players_csv": out_path / f"{prefix}_player_profiles.csv",
        "teams_csv": out_path / f"{prefix}_team_profiles.csv",
        "lineups_csv": out_path / f"{prefix}_lineup_profiles.csv",
        "player_vs_team_csv": out_path / f"{prefix}_player_vs_team_matchups.csv",
        "player_vs_lineup_csv": out_path / f"{prefix}_player_vs_lineup_matchups.csv",
        "team_vs_shooter_type_csv": out_path / f"{prefix}_team_vs_shooter_type.csv",
        "team_style_overview_csv": out_path / f"{prefix}_team_style_overview.csv",
        "summary_json": out_path / f"{prefix}_archetype_summary.json",
        "summary_md": out_path / f"{prefix}_archetype_summary.md",
    }

    outputs.players.to_csv(files["players_csv"], index=False)
    outputs.teams.to_csv(files["teams_csv"], index=False)
    outputs.lineups.to_csv(files["lineups_csv"], index=False)
    outputs.player_vs_team.to_csv(files["player_vs_team_csv"], index=False)
    outputs.player_vs_lineup.to_csv(files["player_vs_lineup_csv"], index=False)
    outputs.team_vs_shooter_type.to_csv(files["team_vs_shooter_type_csv"], index=False)
    team_overview = pd.DataFrame()
    if len(outputs.team_vs_shooter_type) > 0:
        tv = outputs.team_vs_shooter_type.copy()
        weak = (
            tv.sort_values(["opponent_team", "avg_style_matchup_score"], ascending=[True, False], kind="mergesort")
            .groupby("opponent_team", as_index=False, sort=False)
            .head(2)
            .copy()
        )
        weak["slot"] = weak.groupby("opponent_team", sort=False).cumcount() + 1
        strong = (
            tv.sort_values(["opponent_team", "avg_style_matchup_score"], ascending=[True, True], kind="mergesort")
            .groupby("opponent_team", as_index=False, sort=False)
            .head(2)
            .copy()
        )
        strong["slot"] = strong.groupby("opponent_team", sort=False).cumcount() + 1

        team_overview = pd.DataFrame({"opponent_team": sorted(tv["opponent_team"].dropna().astype(str).unique())})
        for idx in [1, 2]:
            wk = weak[weak["slot"] == idx][["opponent_team", "shooter_play_type", "avg_style_matchup_score"]].rename(
                columns={
                    "shooter_play_type": f"weak_type_{idx}",
                    "avg_style_matchup_score": f"weak_score_{idx}",
                }
            )
            st = strong[
                strong["slot"] == idx
            ][["opponent_team", "shooter_play_type", "avg_style_matchup_score"]].rename(
                columns={
                    "shooter_play_type": f"strong_type_{idx}",
                    "avg_style_matchup_score": f"strong_score_{idx}",
                }
            )
            team_overview = team_overview.merge(wk, on="opponent_team", how="left").merge(st, on="opponent_team", how="left")
    team_overview.to_csv(files["team_style_overview_csv"], index=False)
    files["summary_json"].write_text(json.dumps(outputs.summary, indent=2), encoding="utf-8")

    md_lines = [
        "# FG3 Archetype Summary",
        "",
        "## Row Counts",
    ]
    for key, value in outputs.summary.get("rows", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Player Archetypes"]
    for key, value in outputs.summary.get("top_player_archetypes", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Shooter Play Types"]
    for key, value in outputs.summary.get("top_shooter_play_types", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Team Defense Archetypes"]
    for key, value in outputs.summary.get("top_team_defense_archetypes", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Team Vulnerability Archetypes"]
    for key, value in outputs.summary.get("top_team_vulnerability_archetypes", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Lineup Defense Archetypes"]
    for key, value in outputs.summary.get("top_lineup_defense_archetypes", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines += ["", "## Top FG3 Targets vs Teams"]
    for row in outputs.summary.get("top_fg3_targets_vs_teams", []):
        player = row.get("player", "")
        pteam = row.get("player_team", "")
        opp = row.get("opponent_team", "")
        score = row.get("fg3_matchup_attack_score", "")
        tier = row.get("fg3_matchup_tier", "")
        archetype = row.get("team_defense_archetype", "")
        style = row.get("shooter_play_type", "")
        md_lines.append(
            f"- {player} ({pteam}) vs {opp}: score={score:.3f} | {tier} | style={style} | opp_def={archetype}"
            if isinstance(score, (float, int))
            else f"- {player} ({pteam}) vs {opp}: {tier} | style={style} | opp_def={archetype}"
        )
    md_lines += ["", "## Team Weaknesses by Shooter Type"]
    for row in outputs.summary.get("top_team_vulnerabilities_by_shooter_type", []):
        team = row.get("opponent_team", "")
        style = row.get("shooter_play_type", "")
        score = row.get("avg_style_matchup_score", "")
        sample = row.get("sample_matchups", "")
        assessment = row.get("team_vs_type_assessment", "")
        if isinstance(score, (float, int)):
            md_lines.append(f"- {team}: {style} | avg_style_score={score:.3f} | n={sample} | {assessment}")
        else:
            md_lines.append(f"- {team}: {style} | n={sample} | {assessment}")
    files["summary_md"].write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return files
