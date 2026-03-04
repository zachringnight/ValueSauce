"""Execution realism audit.

Detects stale-line, suspended-line, and post-freeze information leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class LeakageFlag:
    """One detected leakage concern."""
    row_idx: int
    category: str  # stale_line | suspended_line | post_freeze | future_data | timestamp_order
    detail: str
    severity: str  # error | warning


def audit_leakage(prepared_df: pd.DataFrame) -> List[LeakageFlag]:
    """Run the full leakage audit on a prepared bet log.

    Checks:
    1. Stale lines — bet placed more than 4h before tip with no refresh evidence.
    2. Suspended lines — odds at exactly -10000 or 0 (market pulled).
    3. Post-freeze — bet placed after game start.
    4. Future data — model probability suspiciously close to outcome
       (perfect Brier on a large slice signals lookahead).
    5. Timestamp ordering — bets not in monotonic asof_utc order.
    """
    flags: List[LeakageFlag] = []
    df = prepared_df.copy()

    # ---------- timestamps ----------
    asof = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce") if "asof_utc" in df.columns else pd.Series(dtype="datetime64[ns, UTC]", index=df.index)
    event_start = pd.to_datetime(df["event_start_utc"], utc=True, errors="coerce") if "event_start_utc" in df.columns else pd.Series(dtype="datetime64[ns, UTC]", index=df.index)

    # 1. Stale lines (>4h before tip)
    if len(asof) > 0 and asof.notna().any() and len(event_start) > 0 and event_start.notna().any():
        delta_min = (event_start - asof).dt.total_seconds() / 60.0
        stale_mask = delta_min > 240
        for idx in df.index[stale_mask]:
            flags.append(LeakageFlag(
                row_idx=int(idx),
                category="stale_line",
                detail=f"Bet placed {delta_min.loc[idx]:.0f}m before tip "
                       f"(player={df.at[idx, 'player']}, market={df.at[idx, 'market']})",
                severity="warning",
            ))

    # 2. Suspended lines
    odds_col = pd.to_numeric(df.get("odds"), errors="coerce")
    if odds_col.notna().any():
        suspended = odds_col.isin([0, -10000, 10000])
        for idx in df.index[suspended]:
            flags.append(LeakageFlag(
                row_idx=int(idx),
                category="suspended_line",
                detail=f"Odds={odds_col.loc[idx]} suggests suspended market "
                       f"(player={df.at[idx, 'player']})",
                severity="error",
            ))

    # 3. Post-freeze / post-start
    if len(asof) > 0 and asof.notna().any() and len(event_start) > 0 and event_start.notna().any():
        post = asof > event_start
        for idx in df.index[post]:
            flags.append(LeakageFlag(
                row_idx=int(idx),
                category="post_freeze",
                detail=f"Bet asof ({asof.loc[idx]}) is AFTER event start "
                       f"({event_start.loc[idx]}) — possible post-freeze leak "
                       f"(player={df.at[idx, 'player']})",
                severity="error",
            ))

    # 4. Future-data (too-perfect model probabilities)
    if "model_prob_side" in df.columns and "is_win" in df.columns:
        graded = df[df.get("is_graded", False) == True].copy()
        if len(graded) >= 30:
            p = pd.to_numeric(graded["model_prob_side"], errors="coerce").fillna(0.5)
            y = graded["is_win"].astype(float)
            brier = float(np.mean((p - y) ** 2))
            if brier < 0.05:
                flags.append(LeakageFlag(
                    row_idx=-1,
                    category="future_data",
                    detail=f"Brier score = {brier:.4f} on {len(graded)} graded bets "
                           f"is suspiciously low — possible lookahead bias",
                    severity="error",
                ))

    # 5. Timestamp ordering
    if asof.notna().sum() >= 2:
        sorted_check = asof.dropna()
        if not sorted_check.is_monotonic_increasing:
            first_violation = None
            for i in range(1, len(sorted_check)):
                if sorted_check.iloc[i] < sorted_check.iloc[i - 1]:
                    first_violation = sorted_check.index[i]
                    break
            if first_violation is not None:
                flags.append(LeakageFlag(
                    row_idx=int(first_violation),
                    category="timestamp_order",
                    detail="asof_utc is not monotonically increasing — "
                           "bets may have been re-ordered post-hoc",
                    severity="warning",
                ))

    return flags


def leakage_report(flags: List[LeakageFlag]) -> Dict[str, Any]:
    """Summarize leakage audit flags into a report dict."""
    errors = [f for f in flags if f.severity == "error"]
    warnings = [f for f in flags if f.severity == "warning"]

    by_category: Dict[str, int] = {}
    for f in flags:
        by_category[f.category] = by_category.get(f.category, 0) + 1

    return {
        "total_flags": len(flags),
        "errors": len(errors),
        "warnings": len(warnings),
        "by_category": by_category,
        "pass": len(errors) == 0,
        "details": [
            {
                "row": f.row_idx,
                "category": f.category,
                "severity": f.severity,
                "detail": f.detail,
            }
            for f in flags
        ],
    }


def audit_log_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing critical log fields.

    Promotion gate: zero missing critical fields.
    """
    critical_fields = [
        "asof_utc", "game_key", "player", "market", "line",
        "odds", "side", "result",
    ]
    model_fields = [
        "p_over", "p_under", "mu", "sigma",
        "recommended_side", "edge_cents_side",
    ]

    missing_critical: Dict[str, int] = {}
    missing_model: Dict[str, int] = {}

    for col in critical_fields:
        if col not in df.columns:
            missing_critical[col] = len(df)
        else:
            n_miss = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
            if n_miss > 0:
                missing_critical[col] = n_miss

    for col in model_fields:
        if col not in df.columns:
            missing_model[col] = len(df)
        else:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                missing_model[col] = n_miss

    return {
        "total_rows": len(df),
        "missing_critical": missing_critical,
        "missing_model": missing_model,
        "critical_pass": len(missing_critical) == 0,
        "model_pass": len(missing_model) == 0,
    }
