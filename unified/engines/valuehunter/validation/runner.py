"""Master orchestrator for the release-candidate validation pack.

Runs every validation module, aggregates results, evaluates promotion gates,
and produces a consolidated text report.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Consolidated output of the full validation pack."""

    generated_at: str = ""
    engine_version: str = "1.1.0"

    # Individual section results
    walk_forward: Optional[object] = None
    metrics_tables: Optional[dict] = None
    ablation: Optional[object] = None
    tracking_comparison: Optional[object] = None
    execution_audit: Optional[object] = None
    paper_ledger: Optional[object] = None
    stability: Optional[object] = None
    failure_cases: Optional[object] = None
    promotion_gates: Optional[object] = None

    # Top-level verdict
    promoted: bool = False
    gates_passed: int = 0
    gates_total: int = 0


class ValidationRunner:
    """Orchestrates the full RC validation pack.

    Usage
    -----
    >>> runner = ValidationRunner(repository, models, ...)
    >>> report = runner.run(start_date, end_date)
    >>> print(runner.format_full_report(report))
    """

    def __init__(
        self,
        repository,
        feature_builder,
        minutes_model,
        three_pa_model,
        make_rate_model,
        simulator,
        decision_engine,
        baselines: Optional[dict] = None,
        leakage_detector=None,
        drift_monitor=None,
        settings=None,
    ):
        self.repository = repository
        self.feature_builder = feature_builder
        self.minutes_model = minutes_model
        self.three_pa_model = three_pa_model
        self.make_rate_model = make_rate_model
        self.simulator = simulator
        self.decision_engine = decision_engine
        self.baselines = baselines or {}
        self.leakage_detector = leakage_detector
        self.drift_monitor = drift_monitor
        self.settings = settings

    def run(
        self,
        start_date: date,
        end_date: date,
        train_window_days: int = 180,
        retrain_every_days: int = 30,
        stability_days: int = 7,
        min_paper_bets: int = 100,
    ) -> ValidationReport:
        """Execute the full validation pack."""
        report = ValidationReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # ── 1. Walk-Forward OOS ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("SECTION 1: Walk-Forward OOS Evaluation")
        logger.info("=" * 60)
        wf_results = self._run_walk_forward(
            start_date, end_date, train_window_days, retrain_every_days,
        )
        report.walk_forward = wf_results

        # ── 2. Metrics Tables ─────────────────────────────────────────────
        logger.info("SECTION 2: Metrics Tables")
        metrics_tables = self._run_metrics_tables(wf_results)
        report.metrics_tables = metrics_tables

        # ── 3. Ablation Report ────────────────────────────────────────────
        logger.info("SECTION 3: Ablation Report")
        ablation = self._run_ablation(start_date, end_date, train_window_days)
        report.ablation = ablation

        # ── 4. Tracking vs Fallback ───────────────────────────────────────
        logger.info("SECTION 4: Tracking-Rich vs Fallback Comparison")
        tracking_cmp = self._run_tracking_comparison(wf_results)
        report.tracking_comparison = tracking_cmp

        # ── 5. Execution Realism Audit ────────────────────────────────────
        logger.info("SECTION 5: Execution Realism Audit")
        exec_audit = self._run_execution_audit(start_date, end_date)
        report.execution_audit = exec_audit

        # ── 6. Paper-Bet Ledger ───────────────────────────────────────────
        logger.info("SECTION 6: 100-Paper-Bet Ledger")
        paper = self._run_paper_ledger(start_date, end_date, min_paper_bets)
        report.paper_ledger = paper

        # ── 7. Pipeline Stability ─────────────────────────────────────────
        logger.info("SECTION 7: 7-Day Pipeline Stability")
        stability = self._run_stability(end_date, stability_days)
        report.stability = stability

        # ── 8. Failure-Case Review ────────────────────────────────────────
        logger.info("SECTION 8: Failure-Case Review")
        failures = self._run_failure_cases(wf_results)
        report.failure_cases = failures

        # ── Promotion Gates ───────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PROMOTION GATE EVALUATION")
        logger.info("=" * 60)
        gates = self._evaluate_promotion_gates(
            wf_results, paper, stability, exec_audit, tracking_cmp,
        )
        report.promotion_gates = gates
        report.promoted = gates.all_passed if gates else False
        report.gates_passed = (
            sum(1 for g in gates.gates if g.passed) if gates else 0
        )
        report.gates_total = len(gates.gates) if gates else 0

        return report

    # ── Section runners ───────────────────────────────────────────────────

    def _run_walk_forward(self, start_date, end_date, train_window, retrain):
        try:
            from .walk_forward import WalkForwardEvaluator

            evaluator = WalkForwardEvaluator(
                repository=self.repository,
                feature_builder=self.feature_builder,
                minutes_model=self.minutes_model,
                three_pa_model=self.three_pa_model,
                make_rate_model=self.make_rate_model,
                simulator=self.simulator,
                decision_engine=self.decision_engine,
                baselines=self.baselines,
                leakage_detector=self.leakage_detector,
            )
            return evaluator.run(
                start_date=start_date,
                end_date=end_date,
                train_window_days=train_window,
                retrain_every_days=retrain,
            )
        except Exception as e:
            logger.error("Walk-forward evaluation failed: %s", e)
            return None

    def _run_metrics_tables(self, wf_results):
        if wf_results is None:
            return None
        try:
            from .metrics_tables import MetricsTableGenerator

            gen = MetricsTableGenerator(wf_results)
            return gen.generate_all()
        except Exception as e:
            logger.error("Metrics table generation failed: %s", e)
            return None

    def _run_ablation(self, start_date, end_date, train_window):
        try:
            from .walk_forward import WalkForwardEvaluator
            from .ablation import AblationReport

            evaluator = WalkForwardEvaluator(
                repository=self.repository,
                feature_builder=self.feature_builder,
                minutes_model=self.minutes_model,
                three_pa_model=self.three_pa_model,
                make_rate_model=self.make_rate_model,
                simulator=self.simulator,
                decision_engine=self.decision_engine,
                baselines=self.baselines,
                leakage_detector=self.leakage_detector,
            )
            ablation = AblationReport(
                repository=self.repository,
                feature_builder=self.feature_builder,
                simulator=self.simulator,
                decision_engine=self.decision_engine,
                walk_forward_evaluator=evaluator,
            )
            return ablation.run(start_date, end_date, train_window)
        except Exception as e:
            logger.error("Ablation report failed: %s", e)
            return None

    def _run_tracking_comparison(self, wf_results):
        if wf_results is None:
            return None
        try:
            from .tracking_comparison import TrackingComparison

            cmp = TrackingComparison(wf_results)
            return cmp.run()
        except Exception as e:
            logger.error("Tracking comparison failed: %s", e)
            return None

    def _run_execution_audit(self, start_date, end_date):
        try:
            from .execution_audit import ExecutionRealismAudit
            from ..backtest.leakage import LeakageDetector

            detector = self.leakage_detector or LeakageDetector()
            audit = ExecutionRealismAudit(
                repository=self.repository,
                leakage_detector=detector,
            )
            return audit.run(start_date, end_date)
        except Exception as e:
            logger.error("Execution audit failed: %s", e)
            return None

    def _run_paper_ledger(self, start_date, end_date, min_bets):
        try:
            from .paper_ledger import PaperBetLedger

            ledger = PaperBetLedger(repository=self.repository)
            return ledger.build_ledger(start_date, end_date, min_bets)
        except Exception as e:
            logger.error("Paper ledger failed: %s", e)
            return None

    def _run_stability(self, end_date, n_days):
        try:
            from .stability_report import StabilityReport

            sr = StabilityReport(
                repository=self.repository,
                drift_monitor=self.drift_monitor,
            )
            return sr.run(end_date, n_days)
        except Exception as e:
            logger.error("Stability report failed: %s", e)
            return None

    def _run_failure_cases(self, wf_results):
        if wf_results is None:
            return None
        try:
            from .failure_cases import FailureCaseReview

            review = FailureCaseReview(wf_results)
            return review.run()
        except Exception as e:
            logger.error("Failure-case review failed: %s", e)
            return None

    def _evaluate_promotion_gates(
        self, wf, paper, stability, exec_audit, tracking_cmp,
    ):
        try:
            from .promotion_gates import PromotionGates

            gates = PromotionGates()
            return gates.evaluate(
                walk_forward_results=wf,
                paper_ledger_results=paper,
                stability_results=stability,
                execution_audit_results=exec_audit,
                tracking_comparison=tracking_cmp,
            )
        except Exception as e:
            logger.error("Promotion gate evaluation failed: %s", e)
            return None

    # ── Full report formatting ────────────────────────────────────────────

    @staticmethod
    def format_full_report(report: ValidationReport) -> str:
        """Produce the consolidated human-readable validation report."""
        sep = "=" * 78
        thin = "-" * 78
        lines: list[str] = []

        lines.append(sep)
        lines.append("  NBA 3PM PROPS ENGINE v1.1 — RELEASE-CANDIDATE VALIDATION PACK")
        lines.append(f"  Generated: {report.generated_at}")
        lines.append(sep)
        lines.append("")

        # ── Section 1: Walk-Forward ───────────────────────────────────────
        lines.append(f"{'1. WALK-FORWARD OOS RESULTS':^78}")
        lines.append(thin)
        if report.walk_forward:
            wf = report.walk_forward
            pm = wf.production_metrics or {}
            lines.append(f"  Predictions: {len(wf.predictions)}")
            lines.append(f"  Folds:       {len(wf.folds)}")
            lines.append("")
            lines.append("  Production Model Metrics:")
            for k, v in sorted(pm.items()):
                if isinstance(v, float):
                    lines.append(f"    {k:<35} {v:>10.4f}")
                else:
                    lines.append(f"    {k:<35} {v!s:>10}")
            lines.append("")
            if wf.baseline_metrics:
                lines.append("  vs Baselines:")
                for name, bm in wf.baseline_metrics.items():
                    bl_ll = bm.get("log_loss", "—")
                    bl_clv = bm.get("clv_mean", "—")
                    lines.append(
                        f"    {name:<25} log_loss={bl_ll!s:>8}  "
                        f"CLV={bl_clv!s:>8}"
                    )
            lines.append("")
            if wf.sliced_metrics:
                lines.append("  Sliced Metrics:")
                for dim, slices in wf.sliced_metrics.items():
                    lines.append(f"    {dim}:")
                    for sv, sm in slices.items():
                        if isinstance(sm, dict):
                            summary_vals = {
                                k: f"{v:.4f}" if isinstance(v, float) else v
                                for k, v in list(sm.items())[:4]
                            }
                            lines.append(f"      {sv:<20} {summary_vals}")
                lines.append("")
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 2: Metrics Tables ─────────────────────────────────────
        lines.append(f"{'2. METRICS TABLES':^78}")
        lines.append(thin)
        if report.metrics_tables:
            for table_name, table_data in report.metrics_tables.items():
                lines.append(f"  {table_name}:")
                if isinstance(table_data, list) and table_data:
                    # Print column headers
                    cols = list(table_data[0].keys())
                    header = "  ".join(f"{c:>12}" for c in cols)
                    lines.append(f"    {header}")
                    for row in table_data[:20]:  # Cap at 20 rows
                        vals = "  ".join(
                            f"{row.get(c, ''):>12}" if not isinstance(row.get(c), float)
                            else f"{row[c]:>12.4f}"
                            for c in cols
                        )
                        lines.append(f"    {vals}")
                lines.append("")
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 3: Ablation ───────────────────────────────────────────
        lines.append(f"{'3. ABLATION REPORT':^78}")
        lines.append(thin)
        if report.ablation and hasattr(report.ablation, "summary_table"):
            for row in report.ablation.summary_table:
                block = row.get("feature_block", "?")
                d_ll = row.get("delta_log_loss", 0)
                d_clv = row.get("delta_clv", 0)
                rank = row.get("importance_rank", "?")
                lines.append(
                    f"  #{rank:<3} {block:<30} "
                    f"ΔLogLoss={d_ll:>+8.4f}  ΔCLV={d_clv:>+8.4f}"
                )
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 4: Tracking Comparison ────────────────────────────────
        lines.append(f"{'4. TRACKING-RICH VS FALLBACK':^78}")
        lines.append(thin)
        if report.tracking_comparison and hasattr(
            report.tracking_comparison, "summary_table"
        ):
            for row in report.tracking_comparison.summary_table:
                metric = row.get("metric", "?")
                tr = row.get("tracking_rich", "—")
                fb = row.get("fallback", "—")
                delta = row.get("delta", "—")
                lines.append(
                    f"  {metric:<25} tracking={tr!s:>8}  "
                    f"fallback={fb!s:>8}  Δ={delta!s:>8}"
                )
            if hasattr(report.tracking_comparison, "warnings"):
                for w in report.tracking_comparison.warnings:
                    lines.append(f"  ⚠ {w}")
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 5: Execution Audit ────────────────────────────────────
        lines.append(f"{'5. EXECUTION REALISM AUDIT':^78}")
        lines.append(thin)
        if report.execution_audit:
            ea = report.execution_audit
            lines.append(
                f"  Decisions audited:     {ea.total_decisions_audited}"
            )
            lines.append(
                f"  Leakage violations:    {ea.total_leakage_violations}"
            )
            lines.append(f"  Stale odds:            {ea.total_stale_odds}")
            lines.append(
                f"  Stale injuries:        {ea.total_stale_injuries}"
            )
            lines.append(
                f"  Suspended lines:       {ea.total_suspended_lines}"
            )
            lines.append(
                f"  Post-freeze refs:      {ea.total_post_freeze_refs}"
            )
            lines.append(
                f"  VERDICT:               {'PASS' if ea.passed else 'FAIL'}"
            )
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 6: Paper Ledger ───────────────────────────────────────
        lines.append(f"{'6. PAPER-BET LEDGER':^78}")
        lines.append(thin)
        if report.paper_ledger:
            pl = report.paper_ledger
            lines.append(f"  Total bets:            {pl.n_bets}")
            lines.append(f"  Settled:               {pl.n_settled}")
            lines.append(f"  Wins / Losses / Push:  {pl.n_wins} / {pl.n_losses} / {pl.n_pushes}")
            lines.append(f"  Hit rate:              {pl.hit_rate:.3f}")
            lines.append(f"  ROI:                   {pl.roi:+.4f}")
            lines.append(f"  Avg CLV:               {pl.avg_clv:+.4f}")
            lines.append(f"  Avg edge:              {pl.avg_edge:+.4f}")
            lines.append(f"  Max drawdown:          {pl.max_drawdown:.4f}")
            lines.append(f"  Missing fields:        {pl.missing_fields_count}")
            lines.append(
                f"  Close capture rate:    {pl.close_capture_rate:.1%}"
            )
            lines.append(
                f"  Meets minimum:         {'YES' if pl.meets_minimum else 'NO'}"
            )
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 7: Stability ──────────────────────────────────────────
        lines.append(f"{'7. PIPELINE STABILITY (7-DAY)':^78}")
        lines.append(thin)
        if report.stability:
            sr = report.stability
            for dh in sr.daily_reports:
                status_marker = (
                    "  " if dh.status == "PASS"
                    else "! " if dh.status == "WARN"
                    else "**"
                )
                lines.append(
                    f"  {status_marker}[{dh.status}] {dh.date}  "
                    f"games={dh.n_games}  scored={dh.n_players_scored}  "
                    f"bets={dh.n_bets}"
                )
                for issue in dh.issues:
                    lines.append(f"         {issue}")
            lines.append("")
            lines.append(
                f"  Consecutive clean days: {sr.consecutive_clean_days}"
            )
            lines.append(
                f"  Meets 7-day requirement: "
                f"{'YES' if sr.meets_7day_requirement else 'NO'}"
            )
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Section 8: Failure Cases ──────────────────────────────────────
        lines.append(f"{'8. FAILURE-CASE REVIEW':^78}")
        lines.append(thin)
        if report.failure_cases and hasattr(
            report.failure_cases, "summary_table"
        ):
            for row in report.failure_cases.summary_table:
                case = row.get("failure_case", "?")
                n = row.get("n_predictions", 0)
                hr = row.get("hit_rate", 0)
                roi = row.get("roi", 0)
                flag = row.get("flag", "")
                marker = " !" if flag else "  "
                lines.append(
                    f"  {marker}{case:<25} n={n:<5} hit={hr:.3f}  "
                    f"roi={roi:+.4f}  {flag}"
                )
            if hasattr(report.failure_cases, "critical_warnings"):
                for w in report.failure_cases.critical_warnings:
                    lines.append(f"  ** CRITICAL: {w}")
        else:
            lines.append("  [NOT AVAILABLE]")
        lines.append("")

        # ── Promotion Gates ───────────────────────────────────────────────
        if report.promotion_gates:
            from .promotion_gates import PromotionGates

            lines.append(PromotionGates.format_report(report.promotion_gates))
        else:
            lines.append(sep)
            lines.append("  PROMOTION GATES: [NOT EVALUATED]")
            lines.append(sep)

        return "\n".join(lines)

    @staticmethod
    def save_report(report: ValidationReport, output_dir: str) -> str:
        """Save the validation report to files."""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save text report
        text_path = os.path.join(output_dir, f"validation_report_{ts}.txt")
        text = ValidationRunner.format_full_report(report)
        with open(text_path, "w") as f:
            f.write(text)

        # Save structured JSON
        json_path = os.path.join(output_dir, f"validation_report_{ts}.json")
        json_data = {
            "generated_at": report.generated_at,
            "engine_version": report.engine_version,
            "promoted": report.promoted,
            "gates_passed": report.gates_passed,
            "gates_total": report.gates_total,
        }
        if report.promotion_gates:
            json_data["gates"] = report.promotion_gates.summary_table
        if report.paper_ledger:
            json_data["paper_ledger_summary"] = {
                "n_bets": report.paper_ledger.n_bets,
                "hit_rate": report.paper_ledger.hit_rate,
                "roi": report.paper_ledger.roi,
                "avg_clv": report.paper_ledger.avg_clv,
            }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        logger.info("Reports saved to %s", output_dir)
        return text_path
