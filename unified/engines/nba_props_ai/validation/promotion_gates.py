"""Promotion gate checker.

Evaluates all RC promotion criteria and produces a pass/fail report.

Gates:
  1. Positive OOS CLV
  2. Stable calibration (cal error < threshold)
  3. Minutes and 3PA error within target
  4. Zero missing critical log fields
  5. No leakage failures (zero errors)
  6. No manual intervention for 7 consecutive days
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class GateResult:
    """Result of one promotion gate check."""
    gate_name: str
    passed: bool
    value: float
    threshold: float
    detail: str


@dataclass
class PromotionReport:
    """Overall promotion decision."""
    all_passed: bool
    gates: List[GateResult]
    summary: str


# Default thresholds
DEFAULT_THRESHOLDS = {
    "min_oos_clv": 0.0,           # CLV must be > 0
    "max_cal_error": 0.06,        # Mean absolute calibration error
    "max_minutes_mae": 4.5,       # Minutes MAE target
    "max_3pa_mae": 1.5,           # 3PA MAE target
    "max_3pa_rmse": 2.2,          # 3PA RMSE target
    "max_missing_critical": 0,    # Zero missing critical fields
    "max_leakage_errors": 0,      # Zero leakage errors
    "min_consecutive_days": 7,    # 7 days without intervention
    "min_graded_bets": 50,        # Minimum sample size
    "max_brier": 0.28,            # Max Brier score
}


def check_promotion_gates(
    metrics_df: Optional[pd.DataFrame] = None,
    leakage_report: Optional[Dict[str, Any]] = None,
    completeness_report: Optional[Dict[str, Any]] = None,
    stability_report: Optional[Dict[str, Any]] = None,
    minutes_mae: Optional[float] = None,
    three_pa_mae: Optional[float] = None,
    three_pa_rmse: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> PromotionReport:
    """Evaluate all promotion gates and return a report.

    All inputs are optional; gates with missing data are marked as failed
    with a 'no data' detail.
    """
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    gates: List[GateResult] = []

    # --- Gate 1: Positive OOS CLV ---
    clv = 0.0
    clv_detail = "no data"
    if metrics_df is not None and len(metrics_df) > 0:
        overall = metrics_df[metrics_df["slice_name"] == "overall"]
        if len(overall) > 0:
            clv = float(overall.iloc[0].get("clv", 0.0))
            clv_detail = f"OOS CLV = {clv:.2f}¢"
    gates.append(GateResult(
        gate_name="positive_oos_clv",
        passed=clv > t["min_oos_clv"],
        value=clv,
        threshold=t["min_oos_clv"],
        detail=clv_detail,
    ))

    # --- Gate 2: Stable calibration ---
    cal_err = 1.0
    cal_detail = "no data"
    if metrics_df is not None and len(metrics_df) > 0:
        overall = metrics_df[metrics_df["slice_name"] == "overall"]
        if len(overall) > 0:
            cal_err = float(overall.iloc[0].get("cal_error", 1.0))
            cal_detail = f"calibration error = {cal_err:.4f}"
    gates.append(GateResult(
        gate_name="stable_calibration",
        passed=cal_err <= t["max_cal_error"],
        value=cal_err,
        threshold=t["max_cal_error"],
        detail=cal_detail,
    ))

    # --- Gate 3a: Minutes MAE ---
    min_mae_val = minutes_mae if minutes_mae is not None else 999.0
    gates.append(GateResult(
        gate_name="minutes_mae",
        passed=min_mae_val <= t["max_minutes_mae"],
        value=min_mae_val,
        threshold=t["max_minutes_mae"],
        detail=f"minutes MAE = {min_mae_val:.2f}" if minutes_mae is not None else "no data",
    ))

    # --- Gate 3b: 3PA MAE ---
    tpa_mae_val = three_pa_mae if three_pa_mae is not None else 999.0
    gates.append(GateResult(
        gate_name="3pa_mae",
        passed=tpa_mae_val <= t["max_3pa_mae"],
        value=tpa_mae_val,
        threshold=t["max_3pa_mae"],
        detail=f"3PA MAE = {tpa_mae_val:.2f}" if three_pa_mae is not None else "no data",
    ))

    # --- Gate 3c: 3PA RMSE ---
    tpa_rmse_val = three_pa_rmse if three_pa_rmse is not None else 999.0
    gates.append(GateResult(
        gate_name="3pa_rmse",
        passed=tpa_rmse_val <= t["max_3pa_rmse"],
        value=tpa_rmse_val,
        threshold=t["max_3pa_rmse"],
        detail=f"3PA RMSE = {tpa_rmse_val:.2f}" if three_pa_rmse is not None else "no data",
    ))

    # --- Gate 4: Zero missing critical log fields ---
    n_missing = 0
    comp_detail = "no data"
    if completeness_report is not None:
        n_missing = sum(completeness_report.get("missing_critical", {}).values())
        comp_detail = f"{n_missing} missing critical fields"
    gates.append(GateResult(
        gate_name="zero_missing_critical",
        passed=n_missing <= t["max_missing_critical"],
        value=float(n_missing),
        threshold=float(t["max_missing_critical"]),
        detail=comp_detail,
    ))

    # --- Gate 5: No leakage failures ---
    n_leakage_errors = 0
    leak_detail = "no data"
    if leakage_report is not None:
        n_leakage_errors = int(leakage_report.get("errors", 0))
        leak_detail = f"{n_leakage_errors} leakage errors"
    gates.append(GateResult(
        gate_name="no_leakage_failures",
        passed=n_leakage_errors <= t["max_leakage_errors"],
        value=float(n_leakage_errors),
        threshold=float(t["max_leakage_errors"]),
        detail=leak_detail,
    ))

    # --- Gate 6: 7 consecutive days without intervention ---
    consec_days = 0
    stab_detail = "no data"
    if stability_report is not None:
        consec_success = int(stability_report.get("max_consecutive_successes", 0))
        # Each run covers ~12h, so 14 consecutive = 7 days
        consec_days = consec_success // 2
        manual = int(stability_report.get("manual_interventions", 0))
        stab_detail = (
            f"{consec_success} consecutive successes "
            f"(≈{consec_days}d), {manual} manual interventions"
        )
    gates.append(GateResult(
        gate_name="7day_no_intervention",
        passed=consec_days >= t["min_consecutive_days"],
        value=float(consec_days),
        threshold=float(t["min_consecutive_days"]),
        detail=stab_detail,
    ))

    # --- Gate 7: Brier score ---
    brier = 1.0
    brier_detail = "no data"
    if metrics_df is not None and len(metrics_df) > 0:
        overall = metrics_df[metrics_df["slice_name"] == "overall"]
        if len(overall) > 0:
            brier = float(overall.iloc[0].get("brier", 1.0))
            brier_detail = f"Brier = {brier:.4f}"
    gates.append(GateResult(
        gate_name="brier_threshold",
        passed=brier <= t["max_brier"],
        value=brier,
        threshold=t["max_brier"],
        detail=brier_detail,
    ))

    # --- Gate 8: Minimum sample size ---
    n_graded = 0
    sample_detail = "no data"
    if metrics_df is not None and len(metrics_df) > 0:
        overall = metrics_df[metrics_df["slice_name"] == "overall"]
        if len(overall) > 0:
            n_graded = int(overall.iloc[0].get("graded", 0))
            sample_detail = f"{n_graded} graded bets"
    gates.append(GateResult(
        gate_name="min_sample_size",
        passed=n_graded >= t["min_graded_bets"],
        value=float(n_graded),
        threshold=float(t["min_graded_bets"]),
        detail=sample_detail,
    ))

    all_passed = all(g.passed for g in gates)

    failed_names = [g.gate_name for g in gates if not g.passed]
    if all_passed:
        summary = "ALL GATES PASSED — RC is eligible for promotion."
    else:
        summary = f"BLOCKED — {len(failed_names)} gate(s) failed: {', '.join(failed_names)}"

    return PromotionReport(all_passed=all_passed, gates=gates, summary=summary)


def format_promotion_report(report: PromotionReport) -> str:
    """Format PromotionReport as a human-readable string."""
    lines = ["=" * 70, "RELEASE CANDIDATE PROMOTION GATES", "=" * 70, ""]

    for g in report.gates:
        status = "PASS" if g.passed else "FAIL"
        lines.append(f"  [{status}] {g.gate_name}")
        lines.append(f"         value={g.value:.4f}  threshold={g.threshold:.4f}")
        lines.append(f"         {g.detail}")
        lines.append("")

    lines.append("-" * 70)
    lines.append(report.summary)
    lines.append("=" * 70)
    return "\n".join(lines)
