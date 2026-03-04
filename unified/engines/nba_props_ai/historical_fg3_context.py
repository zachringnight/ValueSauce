from __future__ import annotations

import json
import os
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import safe_clip


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET_DIR = _REPO_ROOT / "tmp_repo_reviews" / "mar3_links" / "nba_data" / "datasets"
_DEFAULT_ARTIFACT_DIR = _CORE_DIR / "examples" / "fg3_historical_context"


@dataclass
class HistoricalFG3Context:
    player_vs_team: pd.DataFrame
    player_totals: pd.DataFrame
    defender_totals: pd.DataFrame
    team_totals: pd.DataFrame
    player_zone: pd.DataFrame
    team_zone: pd.DataFrame
    meta: Dict[str, object]


_RUNTIME_MEMO: Dict[Tuple[str, str], HistoricalFG3Context] = {}


def _empty_context(meta: Optional[Dict[str, object]] = None) -> HistoricalFG3Context:
    return HistoricalFG3Context(
        player_vs_team=pd.DataFrame(),
        player_totals=pd.DataFrame(),
        defender_totals=pd.DataFrame(),
        team_totals=pd.DataFrame(),
        player_zone=pd.DataFrame(),
        team_zone=pd.DataFrame(),
        meta=dict(meta or {}),
    )


def default_dataset_dir() -> Path:
    return _DEFAULT_DATASET_DIR


def default_artifact_dir() -> Path:
    return _DEFAULT_ARTIFACT_DIR


def _clean_ratio(x: pd.Series) -> pd.Series:
    out = pd.to_numeric(x, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    denom = pd.to_numeric(d, errors="coerce").replace(0, np.nan)
    out = pd.to_numeric(n, errors="coerce") / denom
    return out.replace([np.inf, -np.inf], np.nan)


def _iter_tar_csv_chunks(path: Path, member_name: str, usecols: List[str], chunksize: int) -> Iterable[pd.DataFrame]:
    with tarfile.open(path, mode="r:xz") as tf:
        try:
            member = tf.getmember(member_name)
        except KeyError:
            return
        handle = tf.extractfile(member)
        if handle is None:
            return
        for chunk in pd.read_csv(handle, usecols=usecols, chunksize=chunksize, low_memory=False):
            yield chunk


def _team_maps() -> Tuple[Dict[int, str], Dict[str, int]]:
    team_id_to_abbr: Dict[int, str] = {}
    abbr_to_team_id: Dict[str, int] = {}
    try:
        from nba_api.stats.static import teams as nba_teams

        for t in nba_teams.get_teams():
            tid = int(t.get("id") or 0)
            abbr = str(t.get("abbreviation") or "").upper().strip()
            if tid <= 0 or not abbr:
                continue
            team_id_to_abbr[tid] = abbr
            abbr_to_team_id[abbr] = tid
    except Exception:
        pass
    return team_id_to_abbr, abbr_to_team_id


def build_historical_fg3_context(
    *,
    dataset_dir: Path,
    seasons: List[int],
    include_playoffs: bool = False,
    chunksize: int = 250_000,
) -> HistoricalFG3Context:
    dataset_path = Path(dataset_dir).resolve()

    player_team_agg: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    player_total_agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    defender_total_agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    team_total_agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    player_zone_agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    team_zone_agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    processed_files: List[str] = []

    matchups_usecols = [
        "person_id",
        "matchups_person_id",
        "team_id",
        "home_team_id",
        "away_team_id",
        "partial_possessions",
        "matchup_three_pointers_made",
        "matchup_three_pointers_attempted",
    ]

    for season in sorted({int(s) for s in seasons}):
        candidates = [f"matchups_{season}.tar.xz"]
        if include_playoffs:
            candidates.append(f"matchups_po_{season}.tar.xz")

        for archive_name in candidates:
            archive = dataset_path / archive_name
            if not archive.exists():
                continue
            member = archive_name.replace(".tar.xz", ".csv")
            file_rows = 0
            for chunk in _iter_tar_csv_chunks(archive, member, matchups_usecols, chunksize):
                if len(chunk) == 0:
                    continue
                df = pd.DataFrame(
                    {
                        "off_player_id": pd.to_numeric(chunk.get("person_id"), errors="coerce"),
                        "def_player_id": pd.to_numeric(chunk.get("matchups_person_id"), errors="coerce"),
                        "team_id": pd.to_numeric(chunk.get("team_id"), errors="coerce"),
                        "home_team_id": pd.to_numeric(chunk.get("home_team_id"), errors="coerce"),
                        "away_team_id": pd.to_numeric(chunk.get("away_team_id"), errors="coerce"),
                        "poss": pd.to_numeric(chunk.get("partial_possessions"), errors="coerce").fillna(0.0),
                        "pa3": pd.to_numeric(chunk.get("matchup_three_pointers_attempted"), errors="coerce").fillna(0.0),
                        "pm3": pd.to_numeric(chunk.get("matchup_three_pointers_made"), errors="coerce").fillna(0.0),
                    }
                )
                if len(df) == 0:
                    continue
                df["def_team_id"] = np.where(
                    df["team_id"] == df["home_team_id"],
                    df["away_team_id"],
                    np.where(df["team_id"] == df["away_team_id"], df["home_team_id"], np.nan),
                )
                df = df.dropna(subset=["off_player_id", "def_team_id"])
                if len(df) == 0:
                    continue
                df["off_player_id"] = df["off_player_id"].astype(int)
                df["def_player_id"] = df["def_player_id"].fillna(0).astype(int)
                df["def_team_id"] = df["def_team_id"].astype(int)
                df["pa3"] = df["pa3"].clip(lower=0.0)
                df["pm3"] = df["pm3"].clip(lower=0.0)
                df["poss"] = df["poss"].clip(lower=0.0)
                file_rows += len(df)

                grouped_player_team = (
                    df.groupby(["off_player_id", "def_team_id"], as_index=False)[["pa3", "pm3", "poss"]].sum()
                )
                for row in grouped_player_team.itertuples(index=False):
                    bucket = player_team_agg[(int(row.off_player_id), int(row.def_team_id))]
                    bucket[0] += float(row.pa3)
                    bucket[1] += float(row.pm3)
                    bucket[2] += float(row.poss)

                grouped_player = df.groupby("off_player_id", as_index=False)[["pa3", "pm3", "poss"]].sum()
                for row in grouped_player.itertuples(index=False):
                    bucket = player_total_agg[int(row.off_player_id)]
                    bucket[0] += float(row.pa3)
                    bucket[1] += float(row.pm3)
                    bucket[2] += float(row.poss)

                grouped_def = df.groupby("def_player_id", as_index=False)[["pa3", "pm3", "poss"]].sum()
                for row in grouped_def.itertuples(index=False):
                    did = int(row.def_player_id)
                    if did <= 0:
                        continue
                    bucket = defender_total_agg[did]
                    bucket[0] += float(row.pa3)
                    bucket[1] += float(row.pm3)
                    bucket[2] += float(row.poss)

                grouped_team = df.groupby("def_team_id", as_index=False)[["pa3", "pm3", "poss"]].sum()
                for row in grouped_team.itertuples(index=False):
                    bucket = team_total_agg[int(row.def_team_id)]
                    bucket[0] += float(row.pa3)
                    bucket[1] += float(row.pm3)
                    bucket[2] += float(row.poss)

            processed_files.append(f"{archive.name}:{file_rows}")

    team_id_to_abbr, abbr_to_team_id = _team_maps()
    shotdetail_usecols = [
        "PLAYER_ID",
        "TEAM_ID",
        "SHOT_TYPE",
        "SHOT_ZONE_BASIC",
        "SHOT_ATTEMPTED_FLAG",
        "SHOT_MADE_FLAG",
        "HTM",
        "VTM",
    ]

    for season in sorted({int(s) for s in seasons}):
        candidates = [f"shotdetail_{season}.tar.xz"]
        if include_playoffs:
            candidates.append(f"shotdetail_po_{season}.tar.xz")

        for archive_name in candidates:
            archive = dataset_path / archive_name
            if not archive.exists():
                continue
            member = archive_name.replace(".tar.xz", ".csv")
            file_rows = 0
            for chunk in _iter_tar_csv_chunks(archive, member, shotdetail_usecols, chunksize):
                if len(chunk) == 0:
                    continue
                df = pd.DataFrame(
                    {
                        "player_id": pd.to_numeric(chunk.get("PLAYER_ID"), errors="coerce"),
                        "team_id": pd.to_numeric(chunk.get("TEAM_ID"), errors="coerce"),
                        "shot_type": chunk.get("SHOT_TYPE", "").astype(str),
                        "zone": chunk.get("SHOT_ZONE_BASIC", "").astype(str),
                        "attempted": pd.to_numeric(chunk.get("SHOT_ATTEMPTED_FLAG"), errors="coerce").fillna(1.0),
                        "made": pd.to_numeric(chunk.get("SHOT_MADE_FLAG"), errors="coerce").fillna(0.0),
                        "htm": chunk.get("HTM", "").astype(str).str.upper().str.strip(),
                        "vtm": chunk.get("VTM", "").astype(str).str.upper().str.strip(),
                    }
                )
                if len(df) == 0:
                    continue
                df = df[df["attempted"] > 0].copy()
                if len(df) == 0:
                    continue
                is_three = df["shot_type"].str.contains("3PT", case=False, na=False)
                df = df[is_three].copy()
                if len(df) == 0:
                    continue
                df = df.dropna(subset=["player_id", "team_id"])
                if len(df) == 0:
                    continue
                df["player_id"] = df["player_id"].astype(int)
                df["team_id"] = df["team_id"].astype(int)

                if team_id_to_abbr and abbr_to_team_id:
                    df["team_abbr"] = df["team_id"].map(team_id_to_abbr).fillna("")
                    df["opp_abbr"] = np.where(
                        df["team_abbr"] == df["htm"],
                        df["vtm"],
                        np.where(df["team_abbr"] == df["vtm"], df["htm"], ""),
                    )
                    df["opp_team_id"] = pd.to_numeric(df["opp_abbr"].map(abbr_to_team_id), errors="coerce")
                else:
                    df["opp_team_id"] = np.nan

                zone = df["zone"].str.lower().str.strip()
                is_corner = zone.isin({"left corner 3", "right corner 3"})
                is_above = zone.eq("above the break 3")
                is_other = ~(is_corner | is_above)

                p_group = (
                    df.assign(
                        corner_pa=is_corner.astype(float),
                        ab_pa=is_above.astype(float),
                        other_pa=is_other.astype(float),
                    )
                    .groupby("player_id", as_index=False)[["corner_pa", "ab_pa", "other_pa"]]
                    .sum()
                )
                for row in p_group.itertuples(index=False):
                    bucket = player_zone_agg[int(row.player_id)]
                    bucket[0] += float(row.corner_pa)
                    bucket[1] += float(row.ab_pa)
                    bucket[2] += float(row.other_pa)

                valid_team = df.dropna(subset=["opp_team_id"]).copy()
                if len(valid_team):
                    valid_team["opp_team_id"] = valid_team["opp_team_id"].astype(int)
                    t_group = (
                        valid_team.assign(
                            pa=1.0,
                            pm=valid_team["made"].clip(lower=0.0),
                            corner_pa=is_corner.loc[valid_team.index].astype(float),
                            ab_pa=is_above.loc[valid_team.index].astype(float),
                            other_pa=is_other.loc[valid_team.index].astype(float),
                        )
                        .groupby("opp_team_id", as_index=False)[["pa", "pm", "corner_pa", "ab_pa", "other_pa"]]
                        .sum()
                    )
                    for row in t_group.itertuples(index=False):
                        bucket = team_zone_agg[int(row.opp_team_id)]
                        bucket[0] += float(row.pa)
                        bucket[1] += float(row.pm)
                        bucket[2] += float(row.corner_pa)
                        bucket[3] += float(row.ab_pa)
                        bucket[4] += float(row.other_pa)

                file_rows += len(df)

            processed_files.append(f"{archive.name}:{file_rows}")

    player_totals = pd.DataFrame(
        [
            {
                "player_id": pid,
                "player_3pa": vals[0],
                "player_3pm": vals[1],
                "player_poss": vals[2],
            }
            for pid, vals in player_total_agg.items()
        ]
    )
    if len(player_totals):
        player_totals["player_pct"] = _safe_div(player_totals["player_3pm"], player_totals["player_3pa"])
        player_totals["player_3pa_per100"] = _safe_div(player_totals["player_3pa"], player_totals["player_poss"]) * 100.0
        player_totals["player_sample_weight"] = safe_clip(0.0, 0.0, 0.0)  # placeholder for dtype
        player_totals["player_sample_weight"] = (
            pd.to_numeric(player_totals["player_3pa"], errors="coerce").fillna(0.0)
            / (
                pd.to_numeric(player_totals["player_3pa"], errors="coerce").fillna(0.0)
                + 140.0
            )
        ).clip(lower=0.0, upper=1.0)

    player_vs_team = pd.DataFrame(
        [
            {
                "player_id": pid,
                "def_team_id": tid,
                "player_vs_team_3pa": vals[0],
                "player_vs_team_3pm": vals[1],
                "player_vs_team_poss": vals[2],
            }
            for (pid, tid), vals in player_team_agg.items()
        ]
    )
    if len(player_vs_team) and len(player_totals):
        player_vs_team = player_vs_team.merge(
            player_totals[
                [
                    "player_id",
                    "player_3pa",
                    "player_3pm",
                    "player_poss",
                    "player_pct",
                    "player_3pa_per100",
                    "player_sample_weight",
                ]
            ],
            on="player_id",
            how="left",
        )
        player_vs_team["player_vs_team_pct"] = _safe_div(
            player_vs_team["player_vs_team_3pm"], player_vs_team["player_vs_team_3pa"]
        )
        player_vs_team["player_vs_team_3pa_per100"] = _safe_div(
            player_vs_team["player_vs_team_3pa"], player_vs_team["player_vs_team_poss"]
        ) * 100.0
        player_vs_team["matchup_pct_ratio"] = _safe_div(
            player_vs_team["player_vs_team_pct"], player_vs_team["player_pct"]
        )
        player_vs_team["matchup_volume_ratio"] = _safe_div(
            player_vs_team["player_vs_team_3pa_per100"], player_vs_team["player_3pa_per100"]
        )
        pa = pd.to_numeric(player_vs_team["player_vs_team_3pa"], errors="coerce").fillna(0.0)
        player_vs_team["matchup_sample_weight"] = (pa / (pa + 90.0)).clip(lower=0.0, upper=1.0)

    defender_totals = pd.DataFrame(
        [
            {
                "defender_id": did,
                "defender_allowed_3pa": vals[0],
                "defender_allowed_3pm": vals[1],
                "defender_poss": vals[2],
            }
            for did, vals in defender_total_agg.items()
            if did > 0
        ]
    )
    team_totals = pd.DataFrame(
        [
            {
                "team_id": tid,
                "team_allowed_3pa": vals[0],
                "team_allowed_3pm": vals[1],
                "team_poss": vals[2],
            }
            for tid, vals in team_total_agg.items()
            if tid > 0
        ]
    )

    league_pct = float("nan")
    if len(team_totals):
        tpa = float(pd.to_numeric(team_totals["team_allowed_3pa"], errors="coerce").sum())
        tpm = float(pd.to_numeric(team_totals["team_allowed_3pm"], errors="coerce").sum())
        if tpa > 0:
            league_pct = float(tpm / tpa)

    if len(defender_totals):
        defender_totals["defender_allowed_pct"] = _safe_div(
            defender_totals["defender_allowed_3pm"], defender_totals["defender_allowed_3pa"]
        )
        defender_totals["defender_pct_ratio"] = _safe_div(defender_totals["defender_allowed_pct"], pd.Series(league_pct, index=defender_totals.index))
        pa = pd.to_numeric(defender_totals["defender_allowed_3pa"], errors="coerce").fillna(0.0)
        defender_totals["defender_sample_weight"] = (pa / (pa + 220.0)).clip(lower=0.0, upper=1.0)

    if len(team_totals):
        team_totals["team_allowed_pct"] = _safe_div(team_totals["team_allowed_3pm"], team_totals["team_allowed_3pa"])
        team_totals["team_allowed_pct_ratio"] = _safe_div(team_totals["team_allowed_pct"], pd.Series(league_pct, index=team_totals.index))
        pa = pd.to_numeric(team_totals["team_allowed_3pa"], errors="coerce").fillna(0.0)
        team_totals["team_sample_weight"] = (pa / (pa + 420.0)).clip(lower=0.0, upper=1.0)

    player_zone = pd.DataFrame(
        [
            {
                "player_id": pid,
                "player_corner_3pa": vals[0],
                "player_above_break_3pa": vals[1],
                "player_other_3pa": vals[2],
            }
            for pid, vals in player_zone_agg.items()
            if pid > 0
        ]
    )
    if len(player_zone):
        player_zone["player_zone_3pa"] = (
            pd.to_numeric(player_zone["player_corner_3pa"], errors="coerce").fillna(0.0)
            + pd.to_numeric(player_zone["player_above_break_3pa"], errors="coerce").fillna(0.0)
            + pd.to_numeric(player_zone["player_other_3pa"], errors="coerce").fillna(0.0)
        )
        player_zone["player_corner_share"] = _safe_div(player_zone["player_corner_3pa"], player_zone["player_zone_3pa"])
        player_zone["player_above_break_share"] = _safe_div(player_zone["player_above_break_3pa"], player_zone["player_zone_3pa"])
        player_zone["player_other_share"] = _safe_div(player_zone["player_other_3pa"], player_zone["player_zone_3pa"])
        player_zone["player_zone_sample_weight"] = (
            pd.to_numeric(player_zone["player_zone_3pa"], errors="coerce").fillna(0.0)
            / (
                pd.to_numeric(player_zone["player_zone_3pa"], errors="coerce").fillna(0.0)
                + 180.0
            )
        ).clip(lower=0.0, upper=1.0)

    team_zone = pd.DataFrame(
        [
            {
                "team_id": tid,
                "team_zone_3pa_allowed": vals[0],
                "team_zone_3pm_allowed": vals[1],
                "team_corner_3pa_allowed": vals[2],
                "team_above_break_3pa_allowed": vals[3],
                "team_other_3pa_allowed": vals[4],
            }
            for tid, vals in team_zone_agg.items()
            if tid > 0
        ]
    )
    if len(team_zone):
        team_zone["team_zone_3pt_pct_allowed"] = _safe_div(
            team_zone["team_zone_3pm_allowed"], team_zone["team_zone_3pa_allowed"]
        )
        team_zone["team_corner_share_allowed"] = _safe_div(
            team_zone["team_corner_3pa_allowed"], team_zone["team_zone_3pa_allowed"]
        )
        team_zone["team_above_break_share_allowed"] = _safe_div(
            team_zone["team_above_break_3pa_allowed"], team_zone["team_zone_3pa_allowed"]
        )
        team_zone["team_other_share_allowed"] = _safe_div(
            team_zone["team_other_3pa_allowed"], team_zone["team_zone_3pa_allowed"]
        )

        league_zone_3pa = float(pd.to_numeric(team_zone["team_zone_3pa_allowed"], errors="coerce").sum())
        league_zone_3pm = float(pd.to_numeric(team_zone["team_zone_3pm_allowed"], errors="coerce").sum())
        lg_corner = float(pd.to_numeric(team_zone["team_corner_3pa_allowed"], errors="coerce").sum())
        lg_ab = float(pd.to_numeric(team_zone["team_above_break_3pa_allowed"], errors="coerce").sum())
        lg_other = float(pd.to_numeric(team_zone["team_other_3pa_allowed"], errors="coerce").sum())

        league_zone_pct = league_zone_3pm / league_zone_3pa if league_zone_3pa > 0 else float("nan")
        league_corner_share = lg_corner / league_zone_3pa if league_zone_3pa > 0 else float("nan")
        league_ab_share = lg_ab / league_zone_3pa if league_zone_3pa > 0 else float("nan")
        league_other_share = lg_other / league_zone_3pa if league_zone_3pa > 0 else float("nan")

        team_zone["team_zone_pct_ratio"] = _safe_div(
            team_zone["team_zone_3pt_pct_allowed"], pd.Series(league_zone_pct, index=team_zone.index)
        )
        team_zone["team_corner_share_ratio"] = _safe_div(
            team_zone["team_corner_share_allowed"], pd.Series(league_corner_share, index=team_zone.index)
        )
        team_zone["team_above_break_share_ratio"] = _safe_div(
            team_zone["team_above_break_share_allowed"], pd.Series(league_ab_share, index=team_zone.index)
        )
        team_zone["team_other_share_ratio"] = _safe_div(
            team_zone["team_other_share_allowed"], pd.Series(league_other_share, index=team_zone.index)
        )
        team_zone["team_zone_sample_weight"] = (
            pd.to_numeric(team_zone["team_zone_3pa_allowed"], errors="coerce").fillna(0.0)
            / (
                pd.to_numeric(team_zone["team_zone_3pa_allowed"], errors="coerce").fillna(0.0)
                + 900.0
            )
        ).clip(lower=0.0, upper=1.0)

    meta: Dict[str, object] = {
        "dataset_dir": str(dataset_path),
        "seasons": [int(s) for s in sorted({int(s) for s in seasons})],
        "include_playoffs": bool(include_playoffs),
        "processed_files": processed_files,
        "rows": {
            "player_vs_team": int(len(player_vs_team)),
            "player_totals": int(len(player_totals)),
            "defender_totals": int(len(defender_totals)),
            "team_totals": int(len(team_totals)),
            "player_zone": int(len(player_zone)),
            "team_zone": int(len(team_zone)),
        },
    }
    return HistoricalFG3Context(
        player_vs_team=player_vs_team,
        player_totals=player_totals,
        defender_totals=defender_totals,
        team_totals=team_totals,
        player_zone=player_zone,
        team_zone=team_zone,
        meta=meta,
    )


def write_historical_fg3_context(
    context: HistoricalFG3Context,
    *,
    out_dir: Path,
    prefix: str = "fg3_hist",
) -> Dict[str, Path]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pfx = str(prefix).strip() or "fg3_hist"

    files = {
        "player_vs_team_csv": out / f"{pfx}_player_vs_team.csv",
        "player_totals_csv": out / f"{pfx}_player_totals.csv",
        "defender_totals_csv": out / f"{pfx}_defender_totals.csv",
        "team_totals_csv": out / f"{pfx}_team_totals.csv",
        "player_zone_csv": out / f"{pfx}_player_zone.csv",
        "team_zone_csv": out / f"{pfx}_team_zone.csv",
        "meta_json": out / f"{pfx}_meta.json",
    }

    context.player_vs_team.to_csv(files["player_vs_team_csv"], index=False)
    context.player_totals.to_csv(files["player_totals_csv"], index=False)
    context.defender_totals.to_csv(files["defender_totals_csv"], index=False)
    context.team_totals.to_csv(files["team_totals_csv"], index=False)
    context.player_zone.to_csv(files["player_zone_csv"], index=False)
    context.team_zone.to_csv(files["team_zone_csv"], index=False)

    meta = dict(context.meta)
    meta["output_files"] = {k: str(v) for k, v in files.items() if k != "meta_json"}
    with files["meta_json"].open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return files


def load_historical_fg3_context(
    *,
    artifact_dir: Optional[Path] = None,
    prefix: str = "fg3_hist",
    force_reload: bool = False,
) -> HistoricalFG3Context:
    root = Path(artifact_dir or default_artifact_dir()).resolve()
    pfx = str(prefix).strip() or "fg3_hist"
    memo_key = (str(root), pfx)
    if not force_reload and memo_key in _RUNTIME_MEMO:
        return _RUNTIME_MEMO[memo_key]

    files = {
        "player_vs_team": root / f"{pfx}_player_vs_team.csv",
        "player_totals": root / f"{pfx}_player_totals.csv",
        "defender_totals": root / f"{pfx}_defender_totals.csv",
        "team_totals": root / f"{pfx}_team_totals.csv",
        "player_zone": root / f"{pfx}_player_zone.csv",
        "team_zone": root / f"{pfx}_team_zone.csv",
        "meta": root / f"{pfx}_meta.json",
    }

    if not files["player_vs_team"].exists() and not files["team_totals"].exists():
        ctx = _empty_context(meta={"artifact_dir": str(root), "prefix": pfx, "available": False})
        _RUNTIME_MEMO[memo_key] = ctx
        return ctx

    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    meta: Dict[str, object] = {"artifact_dir": str(root), "prefix": pfx, "available": True}
    if files["meta"].exists():
        try:
            with files["meta"].open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                meta.update(payload)
        except Exception:
            pass

    ctx = HistoricalFG3Context(
        player_vs_team=_read_csv(files["player_vs_team"]),
        player_totals=_read_csv(files["player_totals"]),
        defender_totals=_read_csv(files["defender_totals"]),
        team_totals=_read_csv(files["team_totals"]),
        player_zone=_read_csv(files["player_zone"]),
        team_zone=_read_csv(files["team_zone"]),
        meta=meta,
    )
    _RUNTIME_MEMO[memo_key] = ctx
    return ctx


def _ratio_factor(ratio: Optional[float], shrink: float, sample_weight: float, lo: float, hi: float) -> float:
    try:
        r = float(ratio)
    except Exception:
        return 1.0
    if not np.isfinite(r):
        return 1.0
    sw = float(safe_clip(float(sample_weight), 0.0, 1.0))
    return float(safe_clip(1.0 + (r - 1.0) * float(shrink) * sw, lo, hi))


def get_runtime_historical_fg3_factors(
    *,
    player_id: int,
    opp_team_id: int,
    primary_defender_id: str = "",
    primary_defender_share: float = 0.0,
    artifact_dir: Optional[Path] = None,
    prefix: str = "fg3_hist",
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "hist_fg3a_volume_factor": 1.0,
        "hist_fg3m_matchup_factor": 1.0,
        "hist_fg3m_defender_factor": 1.0,
        "hist_fg3m_team_allowed_factor": 1.0,
        "hist_fg3m_zone_mix_factor": 1.0,
        "hist_fg3m_factor": 1.0,
        "hist_fg3_sample_weight": 0.0,
        "hist_fg3_matchup_3pa": 0.0,
        "hist_fg3_team_3pa": 0.0,
    }
    ctx_dir = artifact_dir
    if ctx_dir is None:
        env_dir = str(os.environ.get("NBA_PROPS_HIST_FG3_DIR", "")).strip()
        if not env_dir:
            return out
        ctx_dir = Path(env_dir)
    ctx = load_historical_fg3_context(artifact_dir=ctx_dir, prefix=prefix)

    sw_components: List[float] = []

    if len(ctx.player_vs_team):
        rows = ctx.player_vs_team[
            (pd.to_numeric(ctx.player_vs_team.get("player_id"), errors="coerce") == int(player_id))
            & (pd.to_numeric(ctx.player_vs_team.get("def_team_id"), errors="coerce") == int(opp_team_id))
        ]
        if len(rows):
            r = rows.iloc[0]
            matchup_sw = float(pd.to_numeric(pd.Series([r.get("matchup_sample_weight")]), errors="coerce").fillna(0.0).iloc[0])
            sw_components.append(matchup_sw)
            out["hist_fg3_matchup_3pa"] = float(
                pd.to_numeric(pd.Series([r.get("player_vs_team_3pa")]), errors="coerce").fillna(0.0).iloc[0]
            )
            out["hist_fg3m_matchup_factor"] = _ratio_factor(
                ratio=pd.to_numeric(pd.Series([r.get("matchup_pct_ratio")]), errors="coerce").iloc[0],
                shrink=0.22,
                sample_weight=matchup_sw,
                lo=0.90,
                hi=1.10,
            )
            out["hist_fg3a_volume_factor"] = _ratio_factor(
                ratio=pd.to_numeric(pd.Series([r.get("matchup_volume_ratio")]), errors="coerce").iloc[0],
                shrink=0.24,
                sample_weight=matchup_sw,
                lo=0.88,
                hi=1.12,
            )

    if len(ctx.defender_totals):
        try:
            did = int(str(primary_defender_id).strip())
        except Exception:
            did = 0
        share = float(safe_clip(primary_defender_share, 0.0, 1.0))
        if did > 0 and share > 0:
            rows = ctx.defender_totals[pd.to_numeric(ctx.defender_totals.get("defender_id"), errors="coerce") == did]
            if len(rows):
                r = rows.iloc[0]
                defender_sw = float(
                    pd.to_numeric(pd.Series([r.get("defender_sample_weight")]), errors="coerce").fillna(0.0).iloc[0]
                )
                eff_sw = float(safe_clip(defender_sw * share, 0.0, 1.0))
                sw_components.append(eff_sw)
                out["hist_fg3m_defender_factor"] = _ratio_factor(
                    ratio=pd.to_numeric(pd.Series([r.get("defender_pct_ratio")]), errors="coerce").iloc[0],
                    shrink=0.20,
                    sample_weight=eff_sw,
                    lo=0.92,
                    hi=1.08,
                )

    if len(ctx.team_totals):
        rows = ctx.team_totals[pd.to_numeric(ctx.team_totals.get("team_id"), errors="coerce") == int(opp_team_id)]
        if len(rows):
            r = rows.iloc[0]
            team_sw = float(pd.to_numeric(pd.Series([r.get("team_sample_weight")]), errors="coerce").fillna(0.0).iloc[0])
            sw_components.append(team_sw)
            out["hist_fg3_team_3pa"] = float(
                pd.to_numeric(pd.Series([r.get("team_allowed_3pa")]), errors="coerce").fillna(0.0).iloc[0]
            )
            out["hist_fg3m_team_allowed_factor"] = _ratio_factor(
                ratio=pd.to_numeric(pd.Series([r.get("team_allowed_pct_ratio")]), errors="coerce").iloc[0],
                shrink=0.18,
                sample_weight=team_sw,
                lo=0.90,
                hi=1.10,
            )

    if len(ctx.player_zone) and len(ctx.team_zone):
        prow = ctx.player_zone[pd.to_numeric(ctx.player_zone.get("player_id"), errors="coerce") == int(player_id)]
        trow = ctx.team_zone[pd.to_numeric(ctx.team_zone.get("team_id"), errors="coerce") == int(opp_team_id)]
        if len(prow) and len(trow):
            p = prow.iloc[0]
            t = trow.iloc[0]
            p_corner = float(pd.to_numeric(pd.Series([p.get("player_corner_share")]), errors="coerce").fillna(0.0).iloc[0])
            p_ab = float(pd.to_numeric(pd.Series([p.get("player_above_break_share")]), errors="coerce").fillna(0.0).iloc[0])
            p_other = float(pd.to_numeric(pd.Series([p.get("player_other_share")]), errors="coerce").fillna(0.0).iloc[0])
            mix_denom = max(p_corner + p_ab + p_other, 1e-6)
            p_corner /= mix_denom
            p_ab /= mix_denom
            p_other /= mix_denom

            ratio_corner = float(
                pd.to_numeric(pd.Series([t.get("team_corner_share_ratio")]), errors="coerce").fillna(1.0).iloc[0]
            )
            ratio_ab = float(
                pd.to_numeric(pd.Series([t.get("team_above_break_share_ratio")]), errors="coerce").fillna(1.0).iloc[0]
            )
            ratio_other = float(
                pd.to_numeric(pd.Series([t.get("team_other_share_ratio")]), errors="coerce").fillna(1.0).iloc[0]
            )
            zone_ratio = float(p_corner * ratio_corner + p_ab * ratio_ab + p_other * ratio_other)

            p_sw = float(
                pd.to_numeric(pd.Series([p.get("player_zone_sample_weight")]), errors="coerce").fillna(0.0).iloc[0]
            )
            t_sw = float(
                pd.to_numeric(pd.Series([t.get("team_zone_sample_weight")]), errors="coerce").fillna(0.0).iloc[0]
            )
            zone_sw = float(safe_clip(min(p_sw, t_sw), 0.0, 1.0))
            sw_components.append(zone_sw)
            out["hist_fg3m_zone_mix_factor"] = _ratio_factor(
                ratio=zone_ratio,
                shrink=0.18,
                sample_weight=zone_sw,
                lo=0.90,
                hi=1.10,
            )

    out["hist_fg3m_factor"] = float(
        safe_clip(
            out["hist_fg3m_matchup_factor"]
            * out["hist_fg3m_defender_factor"]
            * out["hist_fg3m_team_allowed_factor"]
            * out["hist_fg3m_zone_mix_factor"],
            0.86,
            1.14,
        )
    )
    if sw_components:
        out["hist_fg3_sample_weight"] = float(safe_clip(max(sw_components), 0.0, 1.0))
    return out
