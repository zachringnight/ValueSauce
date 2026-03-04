"""100-paper-bet ledger with full logging and close capture.

Selects and logs the best 100 paper bets from model output,
capturing all required fields for post-hoc grading.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils import now_iso

LEDGER_COLUMNS = [
    "bet_id",
    "asof_utc",
    "game_key",
    "player",
    "market",
    "line",
    "side",
    "odds",
    "bookmaker",
    "event_id",
    "event_start_utc",
    # Model fields
    "mu",
    "sigma",
    "p_over",
    "p_under",
    "model_prob_side",
    "edge_cents_side",
    "fair_odds_side",
    # Close capture
    "close_odds",
    "close_line",
    "close_captured_utc",
    # Result (filled post-hoc)
    "result",
    "actual_stat",
    "units",
    "notes",
]


def build_paper_ledger(
    results_df: pd.DataFrame,
    *,
    n_bets: int = 100,
    min_edge_cents: float = 2.0,
) -> pd.DataFrame:
    """Build a paper-bet ledger from pipeline projection results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Pipeline output with columns: player_name, market, line,
        recommended_side, edge_cents_side, model_prob_side, mu, sigma,
        p_over, p_under, odds_over_american, odds_under_american,
        bookmaker, event_id, event_start_utc, game_key.
    n_bets : int
        Number of paper bets to select (default 100).
    min_edge_cents : float
        Minimum edge to qualify.

    Returns
    -------
    pd.DataFrame with LEDGER_COLUMNS.
    """
    df = results_df.copy()

    # Ensure required columns exist
    for col in ["edge_cents_side", "recommended_side", "eligible_for_recommendation"]:
        if col not in df.columns:
            return pd.DataFrame(columns=LEDGER_COLUMNS)

    # Filter eligible bets with sufficient edge
    edge = pd.to_numeric(df.get("edge_cents_side"), errors="coerce").fillna(0)
    eligible = df.get("eligible_for_recommendation", False).astype(bool)
    mask = eligible & (edge >= min_edge_cents)
    candidates = df[mask].copy()

    if len(candidates) == 0:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    # Sort by edge descending, take top n_bets
    candidates["_sort_edge"] = pd.to_numeric(candidates["edge_cents_side"], errors="coerce").fillna(0)
    candidates = candidates.sort_values("_sort_edge", ascending=False).head(n_bets)

    now = now_iso()
    rows: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(candidates.iterrows(), 1):
        side = str(row.get("recommended_side", "over")).lower()
        if side == "over":
            odds = row.get("odds_over_american", row.get("odds_american"))
            fair = row.get("fair_over_odds")
        else:
            odds = row.get("odds_under_american")
            fair = row.get("fair_under_odds")

        rows.append({
            "bet_id": f"PB-{idx:04d}",
            "asof_utc": str(row.get("asof_utc", now)),
            "game_key": str(row.get("game_key", "")),
            "player": str(row.get("player_name", "")),
            "market": str(row.get("market", "")),
            "line": float(row.get("line", 0)),
            "side": side.upper(),
            "odds": int(odds) if pd.notna(odds) else None,
            "bookmaker": str(row.get("bookmaker", "")),
            "event_id": str(row.get("event_id", "")),
            "event_start_utc": str(row.get("event_start_utc", "")),
            "mu": float(row.get("mu", 0)),
            "sigma": float(row.get("sigma", 0)),
            "p_over": float(row.get("p_over", 0)),
            "p_under": float(row.get("p_under", 0)),
            "model_prob_side": float(row.get("model_prob_side", 0)),
            "edge_cents_side": float(row.get("edge_cents_side", 0)),
            "fair_odds_side": int(fair) if pd.notna(fair) else None,
            "close_odds": None,
            "close_line": None,
            "close_captured_utc": None,
            "result": "PENDING",
            "actual_stat": None,
            "units": None,
            "notes": "",
        })

    ledger = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = None
    return ledger[LEDGER_COLUMNS]
