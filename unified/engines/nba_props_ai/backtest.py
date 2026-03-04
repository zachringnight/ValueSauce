from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import OVER_SIDES, UNDER_SIDES
from .utils import american_to_implied_prob

GAME_CARD_COLUMNS = [
    "game_key",
    "game_date_local",
    "player_team",
    "opponent",
    "home",
    "override_team_out",
    "override_team_in",
    "override_opp_out",
    "override_opp_in",
    "minutes_caps",
    "minutes_targets",
    "spread",
    "vegas_team_total",
    "vegas_game_total",
    "notes",
]

PROPS_CARD_COLUMNS = ["game_key", "player", "market", "line", "odds"]


@dataclass(frozen=True)
class ParsedGameKey:
    away: str
    home: str


def parse_game_key(game_key: str) -> ParsedGameKey:
    token = str(game_key or "").strip().upper()
    if "|" in token:
        token = token.split("|", 1)[0].strip()
    if "@" not in token:
        raise ValueError(f"game_key '{game_key}' must look like AWY@HOME.")
    away, home = [x.strip() for x in token.split("@", 1)]
    if not away or not home:
        raise ValueError(f"game_key '{game_key}' must include away and home abbreviations.")
    return ParsedGameKey(away=away, home=home)


def normalize_home_value(value: Any, *, player_team: str, parsed: ParsedGameKey) -> str:
    if value is None or str(value).strip() == "":
        return "1" if player_team == parsed.home else "0"
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "home", "h"}:
        return "1"
    if raw in {"0", "false", "f", "no", "n", "away", "a"}:
        return "0"
    raise ValueError(f"home value '{value}' must be 0/1 or home/away style boolean.")


def _normalize_side_token(value: Any) -> str:
    token = str(value or "").strip().upper()
    if token in OVER_SIDES:
        return "OVER"
    if token in UNDER_SIDES:
        return "UNDER"
    raise ValueError(f"Unsupported side '{value}'. Use OVER/UNDER.")


def build_run_cards_from_bet_slice(slice_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build game/props cards for a single asof slice from bet-log rows.

    Returns:
      game_card_df, props_card_df, mapping_df
    """
    rows: List[Dict[str, Any]] = []
    game_rows: List[Dict[str, Any]] = []
    seen_games: set[str] = set()

    for source_idx, row in slice_df.reset_index().iterrows():
        src_id = int(row["index"])
        orig_key_raw = str(row["game_key"]).strip().upper()
        parsed = parse_game_key(orig_key_raw)
        orig_key = f"{parsed.away}@{parsed.home}"

        player_team = str(row.get("player_team") or "").strip().upper()
        if not player_team:
            player_team = parsed.away

        opponent = str(row.get("opponent") or "").strip().upper()
        if not opponent:
            opponent = parsed.home if player_team == parsed.away else parsed.away

        home_value = normalize_home_value(row.get("home"), player_team=player_team, parsed=parsed)
        side_norm = _normalize_side_token(row.get("side"))
        market = str(row.get("market") or "").strip().upper()
        player_name = str(row.get("player") or "").strip()
        line = float(pd.to_numeric(row.get("line"), errors="raise"))
        odds = int(pd.to_numeric(row.get("odds"), errors="raise"))

        run_game_key = f"{orig_key}|{player_team}"
        game_date_local = pd.to_datetime(row["asof_utc"], utc=True).date().isoformat()

        if run_game_key not in seen_games:
            seen_games.add(run_game_key)
            game_rows.append(
                {
                    "game_key": run_game_key,
                    "game_date_local": game_date_local,
                    "player_team": player_team,
                    "opponent": opponent,
                    "home": home_value,
                    "override_team_out": "",
                    "override_team_in": "",
                    "override_opp_out": "",
                    "override_opp_in": "",
                    "minutes_caps": "",
                    "minutes_targets": "",
                    "spread": "",
                    "vegas_team_total": "",
                    "vegas_game_total": "",
                    "notes": f"backtest_source_game={orig_key_raw}",
                }
            )

        rows.append(
            {
                "_row_id": src_id,
                "run_game_key": run_game_key,
                "orig_game_key": orig_key_raw,
                "player": player_name,
                "player_norm": player_name.strip(),
                "market": market,
                "line": line,
                "odds": odds,
                "side_norm": side_norm,
            }
        )

    map_df = pd.DataFrame(rows)
    props_df = pd.DataFrame(
        [
            {
                "game_key": r["run_game_key"],
                "player": r["player"],
                "market": r["market"],
                "line": r["line"],
                "odds": r["odds"],
            }
            for r in rows
        ]
    )
    games_df = pd.DataFrame(game_rows)

    for col in GAME_CARD_COLUMNS:
        if col not in games_df.columns:
            games_df[col] = ""
    games_df = games_df[GAME_CARD_COLUMNS]

    for col in PROPS_CARD_COLUMNS:
        if col not in props_df.columns:
            props_df[col] = ""
    props_df = props_df[PROPS_CARD_COLUMNS]

    return games_df, props_df, map_df


def attach_model_side_scores(joined_df: pd.DataFrame) -> pd.DataFrame:
    out = joined_df.copy()
    out["p_over"] = pd.to_numeric(out.get("p_over"), errors="coerce")
    out["p_under"] = pd.to_numeric(out.get("p_under"), errors="coerce")
    out["odds"] = pd.to_numeric(out.get("odds"), errors="coerce")
    out["implied_prob"] = out["odds"].apply(
        lambda x: american_to_implied_prob(int(x)) if pd.notna(x) else np.nan
    )
    out["p_model_side"] = np.where(out["side_norm"] == "UNDER", out["p_under"], out["p_over"])
    out["edge_prob_side"] = out["p_model_side"] - out["implied_prob"]
    out["edge_cents_side"] = out["edge_prob_side"] * 100.0
    return out
