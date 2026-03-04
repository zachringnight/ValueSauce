from __future__ import annotations

from collections import defaultdict
from typing import Dict

import pandas as pd


BALANCED_MAX_BETS = 8
BALANCED_MIN_EDGE_CENTS = 2.0
BALANCED_MAX_PER_GAME = 2
BALANCED_MAX_PER_PLAYER = 1

FG3_PLUS_MONEY_ODDS = 110
FG3_PLUS_MONEY_MIN_EDGE_CENTS = 4.0
FG3_HIGH_LINE = 3.5
FG3_HIGH_LINE_MIN_EDGE_CENTS = 5.0
FG3_UNDER_SHORT_LINE = 0.5
FG3_UNDER_HEAVY_JUICE_ODDS = -200
FG3_UNDER_HEAVY_JUICE_MIN_EDGE_CENTS = 7.0
FG3_ENABLE_ARCHETYPE_GATE = True
FG3_OVER_STYLE_SCORE_BAD_THRESHOLD = 0.48
FG3_OVER_STYLE_SCORE_BAD_MIN_EDGE_CENTS = 6.0
FG3_OVER_HANDLES_TYPE_MIN_EDGE_CENTS = 6.0
FG3_UNDER_STYLE_SCORE_HOT_THRESHOLD = 0.60
FG3_UNDER_STYLE_SCORE_HOT_MIN_EDGE_CENTS = 8.0
FG3_UNDER_STRUGGLES_TYPE_MIN_EDGE_CENTS = 8.0
FG3_UNDER_VULNERABILITY_THRESHOLD = 0.62
FG3_UNDER_VULNERABILITY_MIN_EDGE_CENTS = 8.0
FG3_UNDER_CREATOR_LEAK_MIN_EDGE_CENTS = 9.0

_FG3_MARKETS = {
    "FG3M",
    "3PM",
    "THREESMADE",
    "THREEPOINTSMADE",
}


def _is_fg3_market(market: object) -> bool:
    normalized = str(market or "").upper().replace(" ", "").replace("_", "")
    return normalized in _FG3_MARKETS


def _passes_fg3_risk_gate(
    row: pd.Series,
    *,
    fg3_over_plus_money_odds: int,
    fg3_over_plus_money_min_edge_cents: float,
    fg3_over_high_line: float,
    fg3_over_high_line_min_edge_cents: float,
    fg3_under_short_line: float,
    fg3_under_heavy_juice_odds: int,
    fg3_under_heavy_juice_min_edge_cents: float,
) -> bool:
    if not _is_fg3_market(row.get("market")):
        return True

    side = str(row.get("recommended_side") or "").strip().lower()
    edge = pd.to_numeric(row.get("edge_cents_side"), errors="coerce")
    odds = pd.to_numeric(row.get("recommended_odds_american"), errors="coerce")
    line = pd.to_numeric(row.get("line"), errors="coerce")
    if pd.isna(edge) or pd.isna(odds) or pd.isna(line):
        return True

    if side == "over":
        required_edge = 0.0
        if int(odds) >= int(fg3_over_plus_money_odds):
            required_edge = max(required_edge, float(fg3_over_plus_money_min_edge_cents))
        if float(line) >= float(fg3_over_high_line):
            required_edge = max(required_edge, float(fg3_over_high_line_min_edge_cents))
        return float(edge) >= required_edge

    if side == "under":
        if float(line) <= float(fg3_under_short_line) and int(odds) <= int(fg3_under_heavy_juice_odds):
            return float(edge) >= float(fg3_under_heavy_juice_min_edge_cents)
        return True

    return True


def _passes_fg3_archetype_gate(
    row: pd.Series,
    *,
    fg3_over_style_score_bad_threshold: float,
    fg3_over_style_score_bad_min_edge_cents: float,
    fg3_over_handles_type_min_edge_cents: float,
    fg3_under_style_score_hot_threshold: float,
    fg3_under_style_score_hot_min_edge_cents: float,
    fg3_under_struggles_type_min_edge_cents: float,
    fg3_under_vulnerability_threshold: float,
    fg3_under_vulnerability_min_edge_cents: float,
    fg3_under_creator_leak_min_edge_cents: float,
) -> bool:
    if not _is_fg3_market(row.get("market")):
        return True

    side = str(row.get("recommended_side") or "").strip().lower()
    edge = pd.to_numeric(row.get("edge_cents_side"), errors="coerce")
    if pd.isna(edge):
        return True

    style_score = pd.to_numeric(row.get("fg3_style_matchup_score"), errors="coerce")
    vulnerability_for_shooter = pd.to_numeric(row.get("team_vulnerability_for_shooter"), errors="coerce")
    assessment = str(row.get("team_vs_type_assessment") or "").strip().lower()
    shooter_play_type = str(row.get("shooter_play_type") or "").strip().lower()
    team_vuln_archetype = str(row.get("team_vulnerability_archetype") or "").strip().lower()

    required_edge = 0.0
    if side == "over":
        if pd.notna(style_score) and float(style_score) < float(fg3_over_style_score_bad_threshold):
            required_edge = max(required_edge, float(fg3_over_style_score_bad_min_edge_cents))
        if "handles" in assessment:
            required_edge = max(required_edge, float(fg3_over_handles_type_min_edge_cents))
    elif side == "under":
        if pd.notna(style_score) and float(style_score) > float(fg3_under_style_score_hot_threshold):
            required_edge = max(required_edge, float(fg3_under_style_score_hot_min_edge_cents))
        if "struggles" in assessment:
            required_edge = max(required_edge, float(fg3_under_struggles_type_min_edge_cents))
        if pd.notna(vulnerability_for_shooter) and float(vulnerability_for_shooter) >= float(fg3_under_vulnerability_threshold):
            required_edge = max(required_edge, float(fg3_under_vulnerability_min_edge_cents))
        if "creator" in team_vuln_archetype and shooter_play_type in {"pull-up creator", "on-ball shotmaker"}:
            required_edge = max(required_edge, float(fg3_under_creator_leak_min_edge_cents))

    return float(edge) >= required_edge


def build_recommended_card(
    ranked_df: pd.DataFrame,
    *,
    max_bets: int = BALANCED_MAX_BETS,
    min_edge_cents: float = BALANCED_MIN_EDGE_CENTS,
    max_per_game: int = BALANCED_MAX_PER_GAME,
    max_per_player: int = BALANCED_MAX_PER_PLAYER,
    enable_fg3_risk_gate: bool = True,
    fg3_over_plus_money_odds: int = FG3_PLUS_MONEY_ODDS,
    fg3_over_plus_money_min_edge_cents: float = FG3_PLUS_MONEY_MIN_EDGE_CENTS,
    fg3_over_high_line: float = FG3_HIGH_LINE,
    fg3_over_high_line_min_edge_cents: float = FG3_HIGH_LINE_MIN_EDGE_CENTS,
    fg3_under_short_line: float = FG3_UNDER_SHORT_LINE,
    fg3_under_heavy_juice_odds: int = FG3_UNDER_HEAVY_JUICE_ODDS,
    fg3_under_heavy_juice_min_edge_cents: float = FG3_UNDER_HEAVY_JUICE_MIN_EDGE_CENTS,
    enable_fg3_archetype_gate: bool = FG3_ENABLE_ARCHETYPE_GATE,
    fg3_over_style_score_bad_threshold: float = FG3_OVER_STYLE_SCORE_BAD_THRESHOLD,
    fg3_over_style_score_bad_min_edge_cents: float = FG3_OVER_STYLE_SCORE_BAD_MIN_EDGE_CENTS,
    fg3_over_handles_type_min_edge_cents: float = FG3_OVER_HANDLES_TYPE_MIN_EDGE_CENTS,
    fg3_under_style_score_hot_threshold: float = FG3_UNDER_STYLE_SCORE_HOT_THRESHOLD,
    fg3_under_style_score_hot_min_edge_cents: float = FG3_UNDER_STYLE_SCORE_HOT_MIN_EDGE_CENTS,
    fg3_under_struggles_type_min_edge_cents: float = FG3_UNDER_STRUGGLES_TYPE_MIN_EDGE_CENTS,
    fg3_under_vulnerability_threshold: float = FG3_UNDER_VULNERABILITY_THRESHOLD,
    fg3_under_vulnerability_min_edge_cents: float = FG3_UNDER_VULNERABILITY_MIN_EDGE_CENTS,
    fg3_under_creator_leak_min_edge_cents: float = FG3_UNDER_CREATOR_LEAK_MIN_EDGE_CENTS,
) -> pd.DataFrame:
    """
    Build a deterministic recommendation card with simple concentration limits.
    """
    if ranked_df is None or len(ranked_df) == 0:
        return pd.DataFrame(columns=[] if ranked_df is None else ranked_df.columns)

    view = ranked_df.copy()
    if "edge_cents_side" not in view.columns:
        view["edge_cents_side"] = pd.to_numeric(view.get("edge_cents_over"), errors="coerce")
    if "model_prob_side" not in view.columns:
        view["model_prob_side"] = pd.to_numeric(view.get("p_over"), errors="coerce")
    if "eligible_for_recommendation" not in view.columns:
        has_side = view.get("recommended_side").astype(str).str.lower().isin({"over", "under"}) if "recommended_side" in view.columns else False
        has_odds = pd.to_numeric(view.get("recommended_odds_american"), errors="coerce").notna() if "recommended_odds_american" in view.columns else False
        if isinstance(has_side, pd.Series) and isinstance(has_odds, pd.Series):
            view["eligible_for_recommendation"] = has_side & has_odds
        else:
            view["eligible_for_recommendation"] = pd.to_numeric(view["edge_cents_side"], errors="coerce").notna()

    view["edge_cents_side"] = pd.to_numeric(view["edge_cents_side"], errors="coerce")
    view["model_prob_side"] = pd.to_numeric(view["model_prob_side"], errors="coerce")
    eligible = view["eligible_for_recommendation"].astype(bool) if "eligible_for_recommendation" in view.columns else True
    view = view[eligible].copy()
    view = view[view["edge_cents_side"].fillna(-1e9) >= float(min_edge_cents)].copy()
    if enable_fg3_risk_gate and len(view):
        view = view[
            view.apply(
                _passes_fg3_risk_gate,
                axis=1,
                fg3_over_plus_money_odds=fg3_over_plus_money_odds,
                fg3_over_plus_money_min_edge_cents=fg3_over_plus_money_min_edge_cents,
                fg3_over_high_line=fg3_over_high_line,
                fg3_over_high_line_min_edge_cents=fg3_over_high_line_min_edge_cents,
                fg3_under_short_line=fg3_under_short_line,
                fg3_under_heavy_juice_odds=fg3_under_heavy_juice_odds,
                fg3_under_heavy_juice_min_edge_cents=fg3_under_heavy_juice_min_edge_cents,
            )
        ].copy()
    if enable_fg3_archetype_gate and len(view):
        view = view[
            view.apply(
                _passes_fg3_archetype_gate,
                axis=1,
                fg3_over_style_score_bad_threshold=fg3_over_style_score_bad_threshold,
                fg3_over_style_score_bad_min_edge_cents=fg3_over_style_score_bad_min_edge_cents,
                fg3_over_handles_type_min_edge_cents=fg3_over_handles_type_min_edge_cents,
                fg3_under_style_score_hot_threshold=fg3_under_style_score_hot_threshold,
                fg3_under_style_score_hot_min_edge_cents=fg3_under_style_score_hot_min_edge_cents,
                fg3_under_struggles_type_min_edge_cents=fg3_under_struggles_type_min_edge_cents,
                fg3_under_vulnerability_threshold=fg3_under_vulnerability_threshold,
                fg3_under_vulnerability_min_edge_cents=fg3_under_vulnerability_min_edge_cents,
                fg3_under_creator_leak_min_edge_cents=fg3_under_creator_leak_min_edge_cents,
            )
        ].copy()
    if len(view) == 0:
        return view.reset_index(drop=True)

    view = view.sort_values(
        ["edge_cents_side", "model_prob_side", "game_key", "player_name", "market", "line"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    game_counts: Dict[str, int] = defaultdict(int)
    player_counts: Dict[str, int] = defaultdict(int)
    picks = []
    cap = max(0, int(max_bets))
    per_game_cap = max(1, int(max_per_game))
    per_player_cap = max(1, int(max_per_player))

    for _, row in view.iterrows():
        if cap and len(picks) >= cap:
            break
        game_key = str(row.get("game_key") or "")
        player_name = str(row.get("player_name") or "")
        if game_counts[game_key] >= per_game_cap:
            continue
        if player_counts[player_name] >= per_player_cap:
            continue
        picks.append(row)
        game_counts[game_key] += 1
        player_counts[player_name] += 1

    if not picks:
        return view.iloc[0:0].copy()
    return pd.DataFrame(picks).reset_index(drop=True)
