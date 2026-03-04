"""Failure-case review for NBA 3PM Props Engine validation pack.

Performs a detailed review of model performance on known-difficult
scenarios: half-lines, bench shooters, creator-out games, blowout
favorites, back-to-back road games, and late injury flips.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FailureCaseResults:
    """Aggregated results across all failure-case categories."""

    half_lines: dict = field(default_factory=dict)
    bench_shooters: dict = field(default_factory=dict)
    creator_out: dict = field(default_factory=dict)
    blowout_favorites: dict = field(default_factory=dict)
    b2b_road: dict = field(default_factory=dict)
    late_injury_flips: dict = field(default_factory=dict)
    all_cases: dict = field(default_factory=dict)
    critical_warnings: list[str] = field(default_factory=list)
    summary_table: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure-case review
# ---------------------------------------------------------------------------

class FailureCaseReview:
    """Detailed review of model performance on known-difficult scenarios.

    Parameters
    ----------
    walk_forward_results : dict
        The results dictionary returned by a walk-forward evaluator.
        Expected to contain a ``"predictions"`` key holding a list of
        per-prediction dicts, each with keys such as:

        - ``line``, ``mc_result`` (with ``p_over``),
          ``actual`` (with ``three_pm``, ``minutes``, ``three_pa``),
          ``odds_snapshot`` (with ``odds_over_decimal``, ``odds_under_decimal``),
          ``feature_snapshot`` or inline feature keys (``archetype``,
          ``primary_creator_out``, ``secondary_creator_out``,
          ``is_back_to_back``, ``is_home``, ``spread``, etc.),
        - Optional edge/bet fields (``edge``, ``stake``, ``result``,
          ``fill_odds_decimal``).
    """

    def __init__(self, walk_forward_results: dict):
        self.results = walk_forward_results
        self.predictions: list[dict] = walk_forward_results.get(
            "predictions", []
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> FailureCaseResults:
        """Analyse every failure-case category and return aggregated results."""
        logger.info(
            "Running failure-case review on %d predictions",
            len(self.predictions),
        )

        half_lines = self._review_half_lines()
        bench_shooters = self._review_bench_shooters()
        creator_out = self._review_creator_out_games()
        blowout_favorites = self._review_blowout_favorites()
        b2b_road = self._review_b2b_road()
        late_injury_flips = self._review_late_injury_flips()

        all_cases = {
            "half_lines": half_lines,
            "bench_shooters": bench_shooters,
            "creator_out": creator_out,
            "blowout_favorites": blowout_favorites,
            "b2b_road": b2b_road,
            "late_injury_flips": late_injury_flips,
        }

        # Collect critical warnings across all cases
        critical: list[str] = []
        for case_name, case_result in all_cases.items():
            flags = case_result.get("flags", [])
            for flag in flags:
                critical.append(f"[{case_name}] {flag}")

        summary_table = self._build_summary_table(all_cases)

        fc_results = FailureCaseResults(
            half_lines=half_lines,
            bench_shooters=bench_shooters,
            creator_out=creator_out,
            blowout_favorites=blowout_favorites,
            b2b_road=b2b_road,
            late_injury_flips=late_injury_flips,
            all_cases=all_cases,
            critical_warnings=critical,
            summary_table=summary_table,
        )

        logger.info(
            "Failure-case review complete. %d critical warning(s)",
            len(critical),
        )

        return fc_results

    # ------------------------------------------------------------------
    # Failure-case analyses
    # ------------------------------------------------------------------

    def _review_half_lines(self) -> dict:
        """Review predictions on 0.5 and 1.5 lines.

        These lines have extreme binary outcomes.  The 0.5 line is
        essentially "will they make any 3?" -- very different from 2.5+.
        """
        preds = [
            p for p in self.predictions
            if self._get_line(p) in (0.5, 1.5)
        ]
        n = len(preds)
        if n == 0:
            return self._empty_case("half_lines")

        # Separate by specific line for finer detail
        preds_05 = [p for p in preds if self._get_line(p) == 0.5]
        preds_15 = [p for p in preds if self._get_line(p) == 1.5]

        p_over_arr = self._extract_p_over(preds)
        actual_over_arr = self._extract_actual_over(preds)

        # 3PM metrics (log loss, brier)
        tpm = BacktestMetrics.compute_3pm_metrics(actual_over_arr, p_over_arr)

        # Betting metrics
        bet_preds = [p for p in preds if self._has_bet(p)]
        n_bets = len(bet_preds)
        hit_rate, roi, avg_edge = self._compute_bet_stats(bet_preds)

        # Calibration check
        mean_predicted = float(np.mean(p_over_arr))
        mean_actual = float(np.mean(actual_over_arr))
        calibration_gap = abs(mean_predicted - mean_actual)

        flags: list[str] = []
        if calibration_gap > 0.05:
            flags.append(
                f"Calibration off by {calibration_gap:.3f} on half-lines "
                f"(predicted={mean_predicted:.3f}, actual={mean_actual:.3f})"
            )

        return {
            "case": "half_lines",
            "n_predictions": n,
            "n_predictions_05": len(preds_05),
            "n_predictions_15": len(preds_15),
            "n_bets": n_bets,
            "log_loss": tpm.log_loss,
            "brier_score": tpm.brier_score,
            "hit_rate": hit_rate,
            "roi": roi,
            "avg_edge": avg_edge,
            "mean_predicted_p_over": round(mean_predicted, 4),
            "mean_actual_over_rate": round(mean_actual, 4),
            "calibration_gap": round(calibration_gap, 4),
            "flags": flags,
        }

    def _review_bench_shooters(self) -> dict:
        """Review bench_microwave archetype predictions.

        Key concern: volatile minutes make volume prediction unreliable.
        """
        preds = [
            p for p in self.predictions
            if self._get_archetype(p) == "bench_microwave"
        ]
        n = len(preds)
        if n == 0:
            return self._empty_case("bench_shooters")

        # Minutes MAE
        minutes_errors = self._compute_minutes_errors(preds)
        minutes_mae = float(np.mean(np.abs(minutes_errors))) if len(minutes_errors) > 0 else 0.0

        # 3PA MAE
        tpa_errors = self._compute_tpa_errors(preds)
        tpa_mae = float(np.mean(np.abs(tpa_errors))) if len(tpa_errors) > 0 else 0.0

        # Betting
        bet_preds = [p for p in preds if self._has_bet(p)]
        n_bets = len(bet_preds)
        hit_rate, roi, avg_edge = self._compute_bet_stats(bet_preds)

        # Count low-edge bets
        low_edge_bets = sum(
            1 for p in bet_preds
            if abs(self._get_edge(p)) < 0.03
        )

        flags: list[str] = []
        if minutes_mae > 4.0:
            flags.append(
                f"Bench shooter minutes MAE = {minutes_mae:.2f} "
                f"(exceeds 4.0 threshold)"
            )
        if n_bets > 0 and low_edge_bets / n_bets > 0.5:
            flags.append(
                f"{low_edge_bets}/{n_bets} bench shooter bets have "
                f"edge < 3%"
            )

        return {
            "case": "bench_shooters",
            "n_predictions": n,
            "n_bets": n_bets,
            "minutes_mae": round(minutes_mae, 3),
            "tpa_mae": round(tpa_mae, 3),
            "hit_rate": hit_rate,
            "roi": roi,
            "avg_edge": avg_edge,
            "low_edge_bets": low_edge_bets,
            "flags": flags,
        }

    def _review_creator_out_games(self) -> dict:
        """Review predictions where primary or secondary creator is out.

        Key concern: shot distribution shifts when creators are absent.
        """
        preds_out = [
            p for p in self.predictions
            if (
                self._get_feature(p, "primary_creator_out") is True
                or self._get_feature(p, "secondary_creator_out") is True
            )
        ]
        preds_in = [
            p for p in self.predictions
            if (
                self._get_feature(p, "primary_creator_out") is not True
                and self._get_feature(p, "secondary_creator_out") is not True
            )
        ]

        n_out = len(preds_out)
        n_in = len(preds_in)
        if n_out == 0:
            return self._empty_case("creator_out")

        # 3PA MAE when creator is out vs available
        tpa_errors_out = self._compute_tpa_errors(preds_out)
        tpa_mae_out = (
            float(np.mean(np.abs(tpa_errors_out)))
            if len(tpa_errors_out) > 0
            else 0.0
        )

        tpa_errors_in = self._compute_tpa_errors(preds_in)
        tpa_mae_in = (
            float(np.mean(np.abs(tpa_errors_in)))
            if len(tpa_errors_in) > 0
            else 0.0
        )

        # Betting (creator out)
        bet_preds_out = [p for p in preds_out if self._has_bet(p)]
        n_bets_out = len(bet_preds_out)
        hit_rate_out, roi_out, avg_edge_out = self._compute_bet_stats(
            bet_preds_out
        )

        # Betting (creator in)
        bet_preds_in = [p for p in preds_in if self._has_bet(p)]
        n_bets_in = len(bet_preds_in)
        hit_rate_in, roi_in, _ = self._compute_bet_stats(bet_preds_in)

        tpa_mae_delta = tpa_mae_out - tpa_mae_in

        flags: list[str] = []
        if tpa_mae_delta > 0.5:
            flags.append(
                f"3PA MAE increases by {tpa_mae_delta:.2f} when creator "
                f"is out (out={tpa_mae_out:.2f}, in={tpa_mae_in:.2f})"
            )

        return {
            "case": "creator_out",
            "n_predictions_out": n_out,
            "n_predictions_in": n_in,
            "n_bets": n_bets_out,
            "tpa_mae_creator_out": round(tpa_mae_out, 3),
            "tpa_mae_creator_in": round(tpa_mae_in, 3),
            "tpa_mae_delta": round(tpa_mae_delta, 3),
            "hit_rate_out": hit_rate_out,
            "hit_rate_in": hit_rate_in,
            "roi_out": roi_out,
            "roi_in": roi_in,
            "avg_edge": avg_edge_out,
            "flags": flags,
        }

    def _review_blowout_favorites(self) -> dict:
        """Review heavy favourites (spread <= -10).

        Key concern: bench players get extended time, starters sit early.
        """
        preds = [
            p for p in self.predictions
            if self._get_spread(p) <= -10
        ]
        n = len(preds)
        if n == 0:
            return self._empty_case("blowout_favorites")

        # Split by starter / bench
        starters = [
            p for p in preds
            if self._get_feature(p, "started") is True
            or self._get_feature(p, "is_starter") is True
        ]
        bench = [
            p for p in preds
            if p not in starters
        ]

        # Minutes MAE overall
        min_errors_all = self._compute_minutes_errors(preds)
        min_mae_all = (
            float(np.mean(np.abs(min_errors_all)))
            if len(min_errors_all) > 0
            else 0.0
        )

        # Minutes MAE starters
        min_errors_start = self._compute_minutes_errors(starters)
        min_mae_starters = (
            float(np.mean(np.abs(min_errors_start)))
            if len(min_errors_start) > 0
            else 0.0
        )

        # Minutes MAE bench
        min_errors_bench = self._compute_minutes_errors(bench)
        min_mae_bench = (
            float(np.mean(np.abs(min_errors_bench)))
            if len(min_errors_bench) > 0
            else 0.0
        )

        # Betting
        bet_preds = [p for p in preds if self._has_bet(p)]
        n_bets = len(bet_preds)
        hit_rate, roi, avg_edge = self._compute_bet_stats(bet_preds)

        flags: list[str] = []
        if min_mae_starters > 4.0:
            flags.append(
                f"Starter minutes MAE = {min_mae_starters:.2f} in blowout "
                f"favourites (exceeds 4.0 threshold)"
            )

        return {
            "case": "blowout_favorites",
            "n_predictions": n,
            "n_starters": len(starters),
            "n_bench": len(bench),
            "n_bets": n_bets,
            "minutes_mae_overall": round(min_mae_all, 3),
            "minutes_mae_starters": round(min_mae_starters, 3),
            "minutes_mae_bench": round(min_mae_bench, 3),
            "hit_rate": hit_rate,
            "roi": roi,
            "avg_edge": avg_edge,
            "flags": flags,
        }

    def _review_b2b_road(self) -> dict:
        """Review back-to-back road games.

        Key concern: fatigue + travel impact both minutes and efficiency.
        """
        preds_b2b_road = [
            p for p in self.predictions
            if (
                self._get_feature(p, "is_back_to_back") is True
                and self._get_feature(p, "is_home") is False
            )
        ]
        preds_non_b2b_home = [
            p for p in self.predictions
            if (
                self._get_feature(p, "is_back_to_back") is not True
                and self._get_feature(p, "is_home") is True
            )
        ]

        n = len(preds_b2b_road)
        if n == 0:
            return self._empty_case("b2b_road")

        # Minutes MAE
        min_errors = self._compute_minutes_errors(preds_b2b_road)
        minutes_mae = (
            float(np.mean(np.abs(min_errors)))
            if len(min_errors) > 0
            else 0.0
        )

        min_errors_home = self._compute_minutes_errors(preds_non_b2b_home)
        minutes_mae_home = (
            float(np.mean(np.abs(min_errors_home)))
            if len(min_errors_home) > 0
            else 0.0
        )

        # Make-rate error
        mr_error_b2b = self._compute_make_rate_errors(preds_b2b_road)
        mr_error_b2b_mean = (
            float(np.mean(np.abs(mr_error_b2b)))
            if len(mr_error_b2b) > 0
            else 0.0
        )
        mr_error_home = self._compute_make_rate_errors(preds_non_b2b_home)
        mr_error_home_mean = (
            float(np.mean(np.abs(mr_error_home)))
            if len(mr_error_home) > 0
            else 0.0
        )

        # Betting
        bet_preds = [p for p in preds_b2b_road if self._has_bet(p)]
        n_bets = len(bet_preds)
        hit_rate, roi, avg_edge = self._compute_bet_stats(bet_preds)

        # Comparison stats
        bet_preds_home = [p for p in preds_non_b2b_home if self._has_bet(p)]
        hit_rate_home, roi_home, _ = self._compute_bet_stats(bet_preds_home)

        make_rate_delta = mr_error_b2b_mean - mr_error_home_mean

        flags: list[str] = []
        if make_rate_delta > 0.03:
            flags.append(
                f"Make-rate error increases by {make_rate_delta:.3f} on "
                f"B2B road (b2b_road={mr_error_b2b_mean:.3f}, "
                f"non_b2b_home={mr_error_home_mean:.3f})"
            )

        return {
            "case": "b2b_road",
            "n_predictions": n,
            "n_non_b2b_home": len(preds_non_b2b_home),
            "n_bets": n_bets,
            "minutes_mae_b2b_road": round(minutes_mae, 3),
            "minutes_mae_non_b2b_home": round(minutes_mae_home, 3),
            "make_rate_error_b2b_road": round(mr_error_b2b_mean, 4),
            "make_rate_error_non_b2b_home": round(mr_error_home_mean, 4),
            "make_rate_delta": round(make_rate_delta, 4),
            "hit_rate_b2b_road": hit_rate,
            "hit_rate_non_b2b_home": hit_rate_home,
            "roi_b2b_road": roi,
            "roi_non_b2b_home": roi_home,
            "avg_edge": avg_edge,
            "flags": flags,
        }

    def _review_late_injury_flips(self) -> dict:
        """Review bets affected by late injury changes.

        Filter: injury snapshot changed between T-90 and T-5 (or near tip).
        Key concern: late scratches force rapid re-pricing.
        """
        overall_hit_rate = self._compute_overall_hit_rate()

        preds_affected = [
            p for p in self.predictions
            if self._get_feature(p, "injury_flip_late") is True
            or self._get_feature(p, "late_scratch") is True
        ]
        n = len(preds_affected)
        if n == 0:
            return self._empty_case("late_injury_flips")

        # Betting on affected predictions
        bet_preds = [p for p in preds_affected if self._has_bet(p)]
        n_bets = len(bet_preds)
        hit_rate, roi, avg_edge = self._compute_bet_stats(bet_preds)

        # Count bets where the injury change plausibly altered outcome
        bets_outcome_changed = sum(
            1 for p in bet_preds
            if self._get_feature(p, "injury_changed_outcome") is True
        )

        flags: list[str] = []
        if (
            overall_hit_rate is not None
            and hit_rate is not None
            and (overall_hit_rate - hit_rate) > 0.10
        ):
            flags.append(
                f"Hit rate on late-flip bets ({hit_rate:.3f}) is "
                f"{overall_hit_rate - hit_rate:.3f} below overall "
                f"({overall_hit_rate:.3f})"
            )

        return {
            "case": "late_injury_flips",
            "n_predictions": n,
            "n_bets": n_bets,
            "n_bets_outcome_changed": bets_outcome_changed,
            "hit_rate": hit_rate,
            "hit_rate_overall": overall_hit_rate,
            "roi": roi,
            "avg_edge": avg_edge,
            "flags": flags,
        }

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_line(pred: dict) -> float:
        """Extract the prop line from a prediction record."""
        line = pred.get("line")
        if line is not None:
            return float(line)
        odds = pred.get("odds_snapshot", {})
        if odds and odds.get("line") is not None:
            return float(odds["line"])
        return -1.0

    @staticmethod
    def _get_archetype(pred: dict) -> str:
        """Extract the player archetype from a prediction record."""
        # Try direct key first
        arch = pred.get("archetype")
        if arch:
            return str(arch)
        # Try in feature_snapshot or feature_json
        fs = pred.get("feature_snapshot", {})
        if isinstance(fs, dict):
            arch = fs.get("archetype") or fs.get("feature_json", {}).get(
                "archetype"
            )
            if arch:
                return str(arch)
        # Try inline
        fj = pred.get("feature_json", {})
        if isinstance(fj, dict) and fj.get("archetype"):
            return str(fj["archetype"])
        return "unknown"

    @staticmethod
    def _get_spread(pred: dict) -> float:
        """Extract the game spread from a prediction record."""
        spread = pred.get("spread")
        if spread is not None:
            return float(spread)
        odds = pred.get("odds_snapshot", {})
        if isinstance(odds, dict) and odds.get("spread") is not None:
            return float(odds["spread"])
        fs = pred.get("feature_snapshot", {})
        if isinstance(fs, dict):
            s = fs.get("spread") or fs.get("feature_json", {}).get("spread")
            if s is not None:
                return float(s)
        return 0.0

    @staticmethod
    def _get_feature(pred: dict, key: str) -> Any:
        """Extract an arbitrary feature value from a prediction record.

        Searches across multiple possible nesting locations.
        """
        # Direct top-level key
        val = pred.get(key)
        if val is not None:
            return val
        # In feature_snapshot dict
        fs = pred.get("feature_snapshot", {})
        if isinstance(fs, dict):
            val = fs.get(key)
            if val is not None:
                return val
            fj = fs.get("feature_json", {})
            if isinstance(fj, dict):
                val = fj.get(key)
                if val is not None:
                    return val
        # In actual dict
        actual = pred.get("actual", {})
        if isinstance(actual, dict):
            val = actual.get(key)
            if val is not None:
                return val
        return None

    @staticmethod
    def _get_edge(pred: dict) -> float:
        """Extract the bet edge from a prediction."""
        edge = pred.get("edge")
        if edge is not None:
            return float(edge)
        return 0.0

    @staticmethod
    def _has_bet(pred: dict) -> bool:
        """Return True if this prediction includes a filled bet."""
        if pred.get("result") is not None:
            return True
        side = pred.get("recommended_side") or pred.get("side")
        if side and side != "no_bet":
            return True
        return False

    # ------------------------------------------------------------------
    # Array extraction helpers
    # ------------------------------------------------------------------

    def _extract_p_over(self, preds: list[dict]) -> np.ndarray:
        """Extract model P(over) as a numpy array."""
        values = []
        for p in preds:
            mc = p.get("mc_result", {})
            val = mc.get("p_over") if isinstance(mc, dict) else None
            if val is None:
                val = p.get("mc_p_over", 0.5)
            values.append(float(val) if val is not None else 0.5)
        return np.array(values)

    def _extract_actual_over(self, preds: list[dict]) -> np.ndarray:
        """Extract actual over flag (1 if actual 3PM > line, else 0)."""
        values = []
        for p in preds:
            actual = p.get("actual", {})
            actual_3pm = actual.get("three_pm", 0) if isinstance(actual, dict) else 0
            if actual_3pm is None:
                actual_3pm = p.get("actual_3pm", 0)
            line = self._get_line(p)
            values.append(1.0 if float(actual_3pm) > line else 0.0)
        return np.array(values)

    def _compute_minutes_errors(self, preds: list[dict]) -> np.ndarray:
        """Compute (actual_minutes - predicted_minutes) per prediction."""
        errors = []
        for p in preds:
            actual = p.get("actual", {})
            actual_min = (
                actual.get("minutes", None) if isinstance(actual, dict) else None
            )
            mp = p.get("minutes_prediction", {})
            pred_min = mp.get("p50") if isinstance(mp, dict) else None
            if actual_min is not None and pred_min is not None:
                errors.append(float(actual_min) - float(pred_min))
        return np.array(errors) if errors else np.array([])

    def _compute_tpa_errors(self, preds: list[dict]) -> np.ndarray:
        """Compute (actual_3pa - predicted_3pa) per prediction."""
        errors = []
        for p in preds:
            actual = p.get("actual", {})
            actual_tpa = (
                actual.get("three_pa", None) if isinstance(actual, dict) else None
            )
            tp = p.get("tpa_prediction", {})
            pred_tpa = tp.get("expected_3pa") if isinstance(tp, dict) else None
            if actual_tpa is not None and pred_tpa is not None:
                errors.append(float(actual_tpa) - float(pred_tpa))
        return np.array(errors) if errors else np.array([])

    def _compute_make_rate_errors(self, preds: list[dict]) -> np.ndarray:
        """Compute make-rate prediction error per prediction.

        make_rate_error = actual_fg3_pct - predicted_make_rate
        """
        errors = []
        for p in preds:
            actual = p.get("actual", {})
            actual_3pm = (
                float(actual.get("three_pm", 0))
                if isinstance(actual, dict) and actual.get("three_pm") is not None
                else None
            )
            actual_3pa = (
                float(actual.get("three_pa", 0))
                if isinstance(actual, dict) and actual.get("three_pa") is not None
                else None
            )
            if actual_3pm is None or actual_3pa is None or actual_3pa == 0:
                continue

            actual_rate = actual_3pm / actual_3pa

            mr = p.get("make_rate_prediction", {})
            pred_rate = (
                mr.get("make_rate") or mr.get("expected_make_rate")
                if isinstance(mr, dict)
                else None
            )
            if pred_rate is not None:
                errors.append(actual_rate - float(pred_rate))
        return np.array(errors) if errors else np.array([])

    # ------------------------------------------------------------------
    # Betting stats helper
    # ------------------------------------------------------------------

    def _compute_bet_stats(
        self, bet_preds: list[dict]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (hit_rate, roi, avg_edge) for a list of bet predictions.

        If BacktestMetrics data is available via result/edge/stake/odds
        columns, we use it.  Otherwise fall back to simple counts.
        """
        if not bet_preds:
            return None, None, None

        # Try full BacktestMetrics path
        edges_list = []
        results_list = []
        stakes_list = []
        odds_list = []
        for p in bet_preds:
            edge = p.get("edge", 0.0)
            result = p.get("result")
            stake = p.get("stake", 1.0)
            odds_dec = p.get("fill_odds_decimal")
            if odds_dec is None:
                odds_snap = p.get("odds_snapshot", {})
                side = p.get("side") or p.get("recommended_side", "over")
                if isinstance(odds_snap, dict):
                    odds_dec = odds_snap.get(
                        f"odds_{side}_decimal",
                        odds_snap.get("odds_over_decimal", 1.91),
                    )
                else:
                    odds_dec = 1.91
            if result is not None:
                edges_list.append(float(edge) if edge else 0.0)
                results_list.append(float(result))
                stakes_list.append(float(stake) if stake else 1.0)
                odds_list.append(float(odds_dec) if odds_dec else 1.91)

        if results_list:
            bm = BacktestMetrics.compute_betting_metrics(
                edges=np.array(edges_list),
                results=np.array(results_list),
                stakes=np.array(stakes_list),
                odds_decimal=np.array(odds_list),
            )
            return (
                round(bm.hit_rate, 4),
                round(bm.roi, 4),
                round(bm.avg_edge, 4),
            )

        # Fallback: simple count-based hit rate
        wins = sum(1 for p in bet_preds if p.get("result", 0) > 0)
        hit_rate = wins / len(bet_preds) if bet_preds else 0.0
        return round(hit_rate, 4), None, None

    def _compute_overall_hit_rate(self) -> Optional[float]:
        """Return the overall hit rate across all bets in the dataset."""
        bet_preds = [p for p in self.predictions if self._has_bet(p)]
        if not bet_preds:
            return None
        wins = sum(1 for p in bet_preds if p.get("result", 0) > 0)
        return round(wins / len(bet_preds), 4)

    # ------------------------------------------------------------------
    # Empty-case helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_case(case_name: str) -> dict:
        """Return a placeholder dict when no predictions match a filter."""
        return {
            "case": case_name,
            "n_predictions": 0,
            "n_bets": 0,
            "hit_rate": None,
            "roi": None,
            "avg_edge": None,
            "flags": [],
        }

    # ------------------------------------------------------------------
    # Table / report generation
    # ------------------------------------------------------------------

    def generate_table(self, results: FailureCaseResults) -> list[dict]:
        """Generate a summary table with one row per failure case.

        Columns: failure_case, n_predictions, n_bets, hit_rate, roi,
                 avg_edge, key_metric, key_metric_value, flag.
        """
        return self._build_summary_table(results.all_cases)

    @staticmethod
    def _build_summary_table(all_cases: dict) -> list[dict]:
        """Build a list-of-dicts summary table from all case results."""

        # Map each case to its "key metric" (the one most worth watching)
        key_metrics_map = {
            "half_lines": ("calibration_gap", "calibration_gap"),
            "bench_shooters": ("minutes_mae", "minutes_mae"),
            "creator_out": ("tpa_mae_delta", "tpa_mae_delta"),
            "blowout_favorites": ("minutes_mae_starters", "minutes_mae_starters"),
            "b2b_road": ("make_rate_delta", "make_rate_delta"),
            "late_injury_flips": ("hit_rate_vs_overall", "hit_rate"),
        }

        rows: list[dict] = []
        for case_name, case_result in all_cases.items():
            key_metric_label, key_metric_key = key_metrics_map.get(
                case_name, ("n/a", "n/a")
            )
            key_metric_value = case_result.get(key_metric_key)

            flags = case_result.get("flags", [])
            flag_str = "; ".join(flags) if flags else ""

            rows.append({
                "failure_case": case_name,
                "n_predictions": case_result.get("n_predictions", 0),
                "n_bets": case_result.get("n_bets", 0),
                "hit_rate": case_result.get("hit_rate"),
                "roi": case_result.get("roi"),
                "avg_edge": case_result.get("avg_edge"),
                "key_metric": key_metric_label,
                "key_metric_value": key_metric_value,
                "flag": flag_str,
            })

        return rows

    def format_report(self, results: FailureCaseResults) -> str:
        """Produce a human-readable text report.

        Parameters
        ----------
        results : FailureCaseResults
            The results object returned by :meth:`run`.

        Returns
        -------
        str
            Multi-line formatted report string.
        """
        lines: list[str] = []

        lines.append("=" * 72)
        lines.append("  FAILURE-CASE REVIEW REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(
            f"  Total predictions analysed : {len(self.predictions)}"
        )
        lines.append(
            f"  Critical warnings          : {len(results.critical_warnings)}"
        )
        lines.append("")

        # Per-case detail
        for case_name in [
            "half_lines",
            "bench_shooters",
            "creator_out",
            "blowout_favorites",
            "b2b_road",
            "late_injury_flips",
        ]:
            case = results.all_cases.get(case_name, {})
            n_pred = case.get("n_predictions", 0)
            n_bets = case.get("n_bets", 0)
            flags = case.get("flags", [])

            lines.append(f"  --- {case_name.upper().replace('_', ' ')} ---")
            lines.append(f"    Predictions : {n_pred}")
            lines.append(f"    Bets        : {n_bets}")

            # Show all numeric metrics
            skip_keys = {"case", "flags", "n_predictions", "n_bets"}
            for k, v in case.items():
                if k in skip_keys:
                    continue
                if isinstance(v, (int, float)):
                    lines.append(f"    {k:<35}: {v}")
                elif v is not None:
                    lines.append(f"    {k:<35}: {v}")

            if flags:
                lines.append("    FLAGS:")
                for f in flags:
                    lines.append(f"      ** {f}")
            lines.append("")

        # Summary table
        lines.append("  SUMMARY TABLE")
        header = (
            "  {:<22} {:>7} {:>6} {:>9} {:>8} {:>9} {:<20} {:>12} {}"
        ).format(
            "Case", "N_Pred", "N_Bet", "HitRate", "ROI",
            "AvgEdge", "KeyMetric", "Value", "Flag",
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for row in results.summary_table:
            hr = (
                f"{row['hit_rate']:.4f}"
                if row.get("hit_rate") is not None
                else "n/a"
            )
            roi_str = (
                f"{row['roi']:.4f}"
                if row.get("roi") is not None
                else "n/a"
            )
            edge_str = (
                f"{row['avg_edge']:.4f}"
                if row.get("avg_edge") is not None
                else "n/a"
            )
            kv = (
                f"{row['key_metric_value']:.4f}"
                if isinstance(row.get("key_metric_value"), (int, float))
                else "n/a"
            )
            lines.append(
                "  {:<22} {:>7} {:>6} {:>9} {:>8} {:>9} {:<20} {:>12} {}".format(
                    row["failure_case"],
                    row["n_predictions"],
                    row["n_bets"],
                    hr,
                    roi_str,
                    edge_str,
                    row.get("key_metric", ""),
                    kv,
                    row.get("flag", ""),
                )
            )
        lines.append("")

        # Critical warnings
        if results.critical_warnings:
            lines.append("  CRITICAL WARNINGS")
            for w in results.critical_warnings:
                lines.append(f"    ** {w}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)
