"""Promotion-gate checker for NBA 3PM Props Engine v1.1 release candidate.

Evaluates all mandatory gates that must pass before the engine can be promoted
from paper-trade to live. Every gate is binary: PASS or FAIL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Acceptance thresholds (Section O) ─────────────────────────────────────────

MINUTES_MAE_STARTERS = 2.8       # starters (>= 28 mpg)
MINUTES_MAE_ROTATION = 3.5       # rotation (18-28 mpg)
THREE_PA_MAE_HIGH_VOL = 1.25     # >= 8 3PA per 36
THREE_PA_MAE_STANDARD = 1.00     # 5-8 3PA per 36
MIN_PAPER_BETS = 100
MIN_CONSECUTIVE_CLEAN_DAYS = 7


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class PromotionGateResults:
    gates: list[GateResult] = field(default_factory=list)
    all_passed: bool = False
    summary_table: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.all_passed = all(g.passed for g in self.gates)
        self.summary_table = [
            {
                "gate": g.name,
                "status": "PASS" if g.passed else "FAIL",
                "value": f"{g.value:.4f}" if g.value is not None else "—",
                "threshold": (
                    f"{g.threshold}" if g.threshold is not None else "—"
                ),
                "detail": g.detail,
            }
            for g in self.gates
        ]


class PromotionGates:
    """Evaluate all mandatory promotion gates."""

    def evaluate(
        self,
        walk_forward_results,      # WalkForwardResults
        paper_ledger_results,      # PaperLedgerResults
        stability_results,         # StabilityResults
        execution_audit_results,   # ExecutionAuditResults
        tracking_comparison=None,  # TrackingComparisonResults (optional)
    ) -> PromotionGateResults:
        gates: list[GateResult] = []

        # Gate 1: Positive OOS CLV
        gates.append(self._check_positive_clv(walk_forward_results))

        # Gate 2: Stable calibration
        gates.append(self._check_calibration(walk_forward_results))

        # Gate 3a: Minutes MAE within target (starters)
        gates.append(self._check_minutes_mae_starters(walk_forward_results))

        # Gate 3b: Minutes MAE within target (rotation)
        gates.append(self._check_minutes_mae_rotation(walk_forward_results))

        # Gate 4a: 3PA MAE within target (high-volume)
        gates.append(self._check_3pa_mae_high_vol(walk_forward_results))

        # Gate 4b: 3PA MAE within target (standard)
        gates.append(self._check_3pa_mae_standard(walk_forward_results))

        # Gate 5: Production beats all baselines
        gates.append(self._check_beats_baselines(walk_forward_results))

        # Gate 6: Minimum 100 paper bets
        gates.append(self._check_min_paper_bets(paper_ledger_results))

        # Gate 7: Zero missing critical log fields
        gates.append(self._check_no_missing_fields(paper_ledger_results))

        # Gate 8: No leakage failures
        gates.append(self._check_no_leakage(execution_audit_results))

        # Gate 9: 7 consecutive clean days
        gates.append(self._check_stability(stability_results))

        # Gate 10: No manual intervention required
        gates.append(self._check_no_intervention(stability_results))

        result = PromotionGateResults(gates=gates)

        for g in gates:
            level = "INFO" if g.passed else "ERROR"
            logger.log(
                logging.getLevelName(level),
                "Gate [%s]: %s — %s",
                "PASS" if g.passed else "FAIL",
                g.name,
                g.detail,
            )

        logger.info(
            "Promotion verdict: %s (%d/%d gates passed)",
            "PROMOTED" if result.all_passed else "BLOCKED",
            sum(1 for g in gates if g.passed),
            len(gates),
        )

        return result

    # ── Individual gate checks ────────────────────────────────────────────

    @staticmethod
    def _check_positive_clv(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        clv = metrics.get("clv_mean", None)
        if clv is None:
            return GateResult(
                "Positive OOS CLV", False,
                "CLV not available in results", None, 0.0,
            )
        return GateResult(
            "Positive OOS CLV",
            clv > 0,
            f"CLV = {clv:+.4f} probability points",
            clv,
            0.0,
        )

    @staticmethod
    def _check_calibration(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        brier = metrics.get("brier_score", None)
        if brier is None:
            return GateResult(
                "Stable calibration", False,
                "Brier score not available", None, None,
            )
        # Compare vs bookmaker baseline brier as calibration anchor
        baseline_brier = (
            wf.baseline_metrics.get("bookmaker", {}).get("brier_score", 0.25)
            if wf and wf.baseline_metrics
            else 0.25
        )
        passed = brier <= baseline_brier + 0.01  # within 1% of book calibration
        return GateResult(
            "Stable calibration",
            passed,
            f"Brier = {brier:.4f} (book = {baseline_brier:.4f})",
            brier,
            baseline_brier + 0.01,
        )

    @staticmethod
    def _check_minutes_mae_starters(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        mae = metrics.get("minutes_mae_starters", None)
        if mae is None:
            return GateResult(
                "Minutes MAE starters", False,
                "Not available", None, MINUTES_MAE_STARTERS,
            )
        return GateResult(
            "Minutes MAE starters",
            mae <= MINUTES_MAE_STARTERS,
            f"MAE = {mae:.2f}",
            mae,
            MINUTES_MAE_STARTERS,
        )

    @staticmethod
    def _check_minutes_mae_rotation(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        mae = metrics.get("minutes_mae_rotation", None)
        if mae is None:
            return GateResult(
                "Minutes MAE rotation", False,
                "Not available", None, MINUTES_MAE_ROTATION,
            )
        return GateResult(
            "Minutes MAE rotation",
            mae <= MINUTES_MAE_ROTATION,
            f"MAE = {mae:.2f}",
            mae,
            MINUTES_MAE_ROTATION,
        )

    @staticmethod
    def _check_3pa_mae_high_vol(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        mae = metrics.get("three_pa_mae_high_vol", None)
        if mae is None:
            return GateResult(
                "3PA MAE high-volume", False,
                "Not available", None, THREE_PA_MAE_HIGH_VOL,
            )
        return GateResult(
            "3PA MAE high-volume",
            mae <= THREE_PA_MAE_HIGH_VOL,
            f"MAE = {mae:.2f}",
            mae,
            THREE_PA_MAE_HIGH_VOL,
        )

    @staticmethod
    def _check_3pa_mae_standard(wf) -> GateResult:
        metrics = wf.production_metrics if wf else {}
        mae = metrics.get("three_pa_mae_standard", None)
        if mae is None:
            return GateResult(
                "3PA MAE standard", False,
                "Not available", None, THREE_PA_MAE_STANDARD,
            )
        return GateResult(
            "3PA MAE standard",
            mae <= THREE_PA_MAE_STANDARD,
            f"MAE = {mae:.2f}",
            mae,
            THREE_PA_MAE_STANDARD,
        )

    @staticmethod
    def _check_beats_baselines(wf) -> GateResult:
        if not wf or not wf.baseline_metrics:
            return GateResult(
                "Beats all baselines", False,
                "Baseline metrics not available", None, None,
            )
        prod = wf.production_metrics or {}
        prod_ll = prod.get("log_loss", 999)
        prod_clv = prod.get("clv_mean", -999)

        beaten = []
        not_beaten = []
        for name, bm in wf.baseline_metrics.items():
            bl_ll = bm.get("log_loss", 999)
            bl_clv = bm.get("clv_mean", -999)
            # Must beat on BOTH log_loss and CLV
            if prod_ll <= bl_ll and prod_clv >= bl_clv:
                beaten.append(name)
            else:
                not_beaten.append(name)

        passed = len(not_beaten) == 0
        detail = (
            f"Beats: {beaten}. Does not beat: {not_beaten}."
            if not_beaten
            else f"Beats all {len(beaten)} baselines"
        )
        return GateResult("Beats all baselines", passed, detail)

    @staticmethod
    def _check_min_paper_bets(pl) -> GateResult:
        n = pl.n_bets if pl else 0
        return GateResult(
            "Minimum paper bets",
            n >= MIN_PAPER_BETS,
            f"{n} bets (min {MIN_PAPER_BETS})",
            float(n),
            float(MIN_PAPER_BETS),
        )

    @staticmethod
    def _check_no_missing_fields(pl) -> GateResult:
        n_missing = pl.missing_fields_count if pl else -1
        return GateResult(
            "Zero missing critical fields",
            n_missing == 0,
            f"{n_missing} records with missing fields",
            float(n_missing),
            0.0,
        )

    @staticmethod
    def _check_no_leakage(ea) -> GateResult:
        violations = ea.total_leakage_violations if ea else -1
        return GateResult(
            "No leakage failures",
            violations == 0,
            f"{violations} leakage violations",
            float(violations),
            0.0,
        )

    @staticmethod
    def _check_stability(sr) -> GateResult:
        days = sr.consecutive_clean_days if sr else 0
        return GateResult(
            "7 consecutive clean days",
            days >= MIN_CONSECUTIVE_CLEAN_DAYS,
            f"{days} consecutive clean days",
            float(days),
            float(MIN_CONSECUTIVE_CLEAN_DAYS),
        )

    @staticmethod
    def _check_no_intervention(sr) -> GateResult:
        no_fail = sr.no_manual_intervention if sr else False
        return GateResult(
            "No manual intervention",
            no_fail,
            "No FAIL days" if no_fail else "FAIL days detected",
        )

    # ── Report formatting ─────────────────────────────────────────────────

    @staticmethod
    def format_report(results: PromotionGateResults) -> str:
        lines = [
            "=" * 78,
            "  PROMOTION GATE EVALUATION",
            "=" * 78,
            "",
        ]

        max_name = max(len(g.name) for g in results.gates)
        for g in results.gates:
            status = "PASS" if g.passed else "FAIL"
            marker = "  " if g.passed else "**"
            val_str = f"{g.value:.4f}" if g.value is not None else "—"
            thr_str = f"{g.threshold}" if g.threshold is not None else "—"
            lines.append(
                f"  {marker}[{status}] {g.name:<{max_name}}  "
                f"val={val_str}  thr={thr_str}"
            )
            lines.append(f"         {g.detail}")
            lines.append("")

        lines.append("-" * 78)
        n_pass = sum(1 for g in results.gates if g.passed)
        n_total = len(results.gates)
        verdict = "PROMOTED" if results.all_passed else "BLOCKED"
        lines.append(
            f"  VERDICT: {verdict}  ({n_pass}/{n_total} gates passed)"
        )
        lines.append("=" * 78)

        return "\n".join(lines)
