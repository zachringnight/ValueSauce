"""Failure-case review for known difficult scenarios.

Evaluates model performance on:
  - 0.5 / 1.5 lines (FG3M)
  - Bench shooters
  - Creator-out games
  - Blowout favorites
  - B2B road spots
  - Late injury flips
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FailureCaseReport:
    """Performance summary for one failure-case bucket."""
    case_name: str
    n: int
    graded: int
    wins: int
    hit_rate: float
    roi: float
    units: float
    brier: float
    avg_edge: float
    clv: float


def _case_metrics(df: pd.DataFrame, case_name: str) -> FailureCaseReport:
    n = len(df)
    graded = int(df["is_graded"].sum()) if "is_graded" in df.columns else 0
    wins = int(df["is_win"].sum()) if "is_win" in df.columns else 0
    hit_rate = float(wins / graded) if graded > 0 else 0.0
    units_total = float(pd.to_numeric(df.get("units"), errors="coerce").fillna(0).sum())
    roi = float(units_total / graded * 100) if graded > 0 else 0.0

    brier = 0.0
    if graded > 0 and "model_prob_side" in df.columns:
        gdf = df[df["is_graded"]]
        p = pd.to_numeric(gdf["model_prob_side"], errors="coerce").fillna(0.5).values
        y = gdf["is_win"].astype(float).values
        brier = float(np.mean((p - y) ** 2))

    edge_col = pd.to_numeric(df.get("edge_vs_market"), errors="coerce").fillna(0)
    avg_edge = float(edge_col.mean()) if n > 0 else 0.0

    # CLV
    clv = 0.0
    if "model_prob_side" in df.columns and "implied_prob" in df.columns:
        mp = pd.to_numeric(df["model_prob_side"], errors="coerce")
        ip = pd.to_numeric(df["implied_prob"], errors="coerce")
        valid = mp.notna() & ip.notna()
        if valid.sum() > 0:
            clv = float((mp[valid] - ip[valid]).mean() * 100)

    return FailureCaseReport(
        case_name=case_name, n=n, graded=graded, wins=wins,
        hit_rate=hit_rate, roi=roi, units=units_total,
        brier=brier, avg_edge=avg_edge, clv=clv,
    )


def run_failure_cases(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all failure-case buckets.

    The DataFrame should have columns: market, line, odds, spread,
    rest_days, is_graded, is_win, units, model_prob_side, implied_prob,
    edge_vs_market, and optionally player_minutes, is_bench, creator_out,
    late_injury_flip.

    Returns a DataFrame with one row per case.
    """
    df = prepared_df.copy()
    market = df["market"].astype(str).str.upper() if "market" in df.columns else pd.Series("", index=df.index)
    line = pd.to_numeric(df["line"], errors="coerce").fillna(0) if "line" in df.columns else pd.Series(0, index=df.index)
    spread = pd.to_numeric(df["spread"], errors="coerce").fillna(0) if "spread" in df.columns else pd.Series(0, index=df.index)
    rest = pd.to_numeric(df["rest_days"], errors="coerce").fillna(-1).astype(int) if "rest_days" in df.columns else pd.Series(-1, index=df.index, dtype=int)
    is_home = df["is_home"] if "is_home" in df.columns else pd.Series(True, index=df.index)
    if is_home.dtype == object:
        is_home = is_home.astype(str).str.lower().isin(["1", "true", "yes", "home"])

    results: List[FailureCaseReport] = []

    # 1. 0.5 / 1.5 lines (FG3M)
    fg3m_mask = market == "FG3M"
    low_lines = fg3m_mask & (line <= 1.5)
    if low_lines.sum() > 0:
        results.append(_case_metrics(df[low_lines], "fg3m_0.5_1.5_lines"))

    # 2. Bench shooters (minutes < 22 or flagged)
    bench_mask = pd.Series(False, index=df.index)
    if "player_minutes" in df.columns:
        pm = pd.to_numeric(df["player_minutes"], errors="coerce").fillna(30)
        bench_mask = pm < 22
    elif "is_bench" in df.columns:
        bench_mask = df["is_bench"].astype(bool)
    if bench_mask.sum() > 0:
        results.append(_case_metrics(df[bench_mask], "bench_shooters"))

    # 3. Creator-out games
    creator_mask = pd.Series(False, index=df.index)
    if "creator_out" in df.columns:
        creator_mask = df["creator_out"].astype(bool)
    elif "flags" in df.columns:
        creator_mask = df["flags"].astype(str).str.contains("creator_out|star.*out", case=False, na=False)
    if creator_mask.sum() > 0:
        results.append(_case_metrics(df[creator_mask], "creator_out_games"))

    # 4. Blowout favorites (spread <= -10)
    blowout_mask = spread <= -10
    if blowout_mask.sum() > 0:
        results.append(_case_metrics(df[blowout_mask], "blowout_favorites"))

    # 5. B2B road spots (rest=0, is_home=False)
    b2b_road = (rest == 0) & (~is_home)
    if b2b_road.sum() > 0:
        results.append(_case_metrics(df[b2b_road], "b2b_road"))

    # 6. Late injury flips
    late_flip_mask = pd.Series(False, index=df.index)
    if "late_injury_flip" in df.columns:
        late_flip_mask = df["late_injury_flip"].astype(bool)
    elif "flags" in df.columns:
        late_flip_mask = df["flags"].astype(str).str.contains("late.*injury|injury.*flip", case=False, na=False)
    if late_flip_mask.sum() > 0:
        results.append(_case_metrics(df[late_flip_mask], "late_injury_flips"))

    # 7. All FG3M (general)
    if fg3m_mask.sum() > 0:
        results.append(_case_metrics(df[fg3m_mask], "all_fg3m"))

    # 8. High-line PTS (>30.5)
    high_pts = (market == "PTS") & (line > 30.5)
    if high_pts.sum() > 0:
        results.append(_case_metrics(df[high_pts], "high_pts_lines"))

    # 9. Overall baseline
    results.append(_case_metrics(df, "overall"))

    return pd.DataFrame([r.__dict__ for r in results])
