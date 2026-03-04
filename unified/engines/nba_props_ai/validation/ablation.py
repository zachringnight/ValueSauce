"""Ablation study runner.

Removes each major feature block independently and measures degradation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Feature blocks that can be disabled via the feature_gate mechanism
# ---------------------------------------------------------------------------

ABLATION_BLOCKS: Dict[str, Dict[str, float]] = {
    "matchup_context": {
        "disable_matchup_context": 0.0,
    },
    "lineup_context": {
        "lineup_pace_factor": 0.0,
        "lineup_minutes_factor": 0.0,
        "onoff_usage_factor": 0.0,
        "onoff_ast_factor": 0.0,
        "onoff_reb_factor": 0.0,
        "onoff_fg3a_factor": 0.0,
    },
    "splits_context": {
        "disable_player_splits_context": 0.0,
    },
    "shot_profile": {
        "disable_shot_context": 0.0,
    },
    "pace_factor": {
        "pace_factor": 0.0,
    },
    "team_total_factor": {
        "team_total_factor": 0.0,
    },
    "market_trend": {
        "market_trend_factor": 0.0,
    },
    "opponent_absence": {
        "opponent_absence_factor": 0.0,
    },
    "role_context": {
        "role_pts_factor": 0.0,
        "role_reb_factor": 0.0,
        "role_ast_factor": 0.0,
        "role_fg3m_factor": 0.0,
    },
    "historical_fg3": {
        "hist_fg3a_volume_factor": 0.0,
        "fg3m_market_factor": 0.0,
    },
    "blowout_risk": {
        "blowout_minutes_factor": 0.0,
        "blowout_var_factor": 0.0,
    },
    "foul_risk": {
        "foul_minutes_factor": 0.0,
        "foul_var_factor": 0.0,
    },
}


@dataclass
class AblationResult:
    """Result of one ablation run."""
    block_name: str
    disabled_keys: List[str]
    n_props: int
    brier_baseline: float
    brier_ablated: float
    brier_delta: float
    log_loss_baseline: float
    log_loss_ablated: float
    log_loss_delta: float
    clv_baseline: float
    clv_ablated: float
    clv_delta: float
    roi_baseline: float
    roi_ablated: float
    roi_delta: float


def build_ablation_gate(
    baseline_gate: Optional[Dict[str, float]],
    block_name: str,
) -> Dict[str, float]:
    """Merge block-specific disable weights into the baseline gate."""
    gate = dict(baseline_gate or {})
    overrides = ABLATION_BLOCKS.get(block_name, {})
    gate.update(overrides)
    return gate


def run_ablation_study(
    bet_log_df: pd.DataFrame,
    run_pipeline_fn: Callable,
    baseline_gate: Optional[Dict[str, float]] = None,
    blocks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run ablation study by disabling each feature block.

    Parameters
    ----------
    bet_log_df : pd.DataFrame
        Prepared bet-log with graded results.
    run_pipeline_fn : callable
        Function(gate: dict) -> pd.DataFrame that re-runs the pipeline
        with the given feature gate and returns a joined DataFrame with
        columns: model_prob_side, is_win, is_graded, units, implied_prob.
    baseline_gate : dict or None
        The baseline feature gate (all weights at production values).
    blocks : list[str] or None
        Which blocks to ablate.  Defaults to all ABLATION_BLOCKS.

    Returns
    -------
    pd.DataFrame  with one row per ablated block.
    """
    import numpy as np
    from .metrics import _safe_brier, _safe_log_loss, _clv

    if blocks is None:
        blocks = list(ABLATION_BLOCKS.keys())

    # Baseline run
    baseline_df = run_pipeline_fn(baseline_gate)
    gdf_b = baseline_df[baseline_df["is_graded"]] if "is_graded" in baseline_df.columns else baseline_df
    p_b = pd.to_numeric(gdf_b.get("model_prob_side"), errors="coerce").fillna(0.5).values
    y_b = gdf_b["is_win"].astype(float).values if "is_win" in gdf_b.columns else np.zeros(len(gdf_b))
    brier_b = _safe_brier(p_b, y_b)
    ll_b = _safe_log_loss(p_b, y_b)
    clv_b = _clv(gdf_b)
    units_b = pd.to_numeric(gdf_b.get("units"), errors="coerce").fillna(0).sum()
    roi_b = float(units_b / len(gdf_b) * 100) if len(gdf_b) > 0 else 0.0

    results: List[AblationResult] = []
    for block in blocks:
        if block not in ABLATION_BLOCKS:
            continue
        gate = build_ablation_gate(baseline_gate, block)
        ablated_df = run_pipeline_fn(gate)
        gdf_a = ablated_df[ablated_df["is_graded"]] if "is_graded" in ablated_df.columns else ablated_df
        p_a = pd.to_numeric(gdf_a.get("model_prob_side"), errors="coerce").fillna(0.5).values
        y_a = gdf_a["is_win"].astype(float).values if "is_win" in gdf_a.columns else np.zeros(len(gdf_a))
        brier_a = _safe_brier(p_a, y_a)
        ll_a = _safe_log_loss(p_a, y_a)
        clv_a = _clv(gdf_a)
        units_a = pd.to_numeric(gdf_a.get("units"), errors="coerce").fillna(0).sum()
        roi_a = float(units_a / len(gdf_a) * 100) if len(gdf_a) > 0 else 0.0

        results.append(AblationResult(
            block_name=block,
            disabled_keys=list(ABLATION_BLOCKS[block].keys()),
            n_props=len(gdf_a),
            brier_baseline=brier_b,
            brier_ablated=brier_a,
            brier_delta=brier_a - brier_b,
            log_loss_baseline=ll_b,
            log_loss_ablated=ll_a,
            log_loss_delta=ll_a - ll_b,
            clv_baseline=clv_b,
            clv_ablated=clv_a,
            clv_delta=clv_a - clv_b,
            roi_baseline=roi_b,
            roi_ablated=roi_a,
            roi_delta=roi_a - roi_b,
        ))

    return pd.DataFrame([r.__dict__ for r in results])
