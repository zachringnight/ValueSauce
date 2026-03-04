"""Ablation report for NBA 3PM Props Engine validation pack.

Removes each major feature block independently and measures the impact
on model performance via walk-forward evaluation.  The resulting
importance ranking shows which feature groups contribute most to
predictive accuracy and betting edge.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature block definitions
# ---------------------------------------------------------------------------

FEATURE_BLOCKS: dict[str, list[str]] = {
    "tracking_shot_mix": [
        "catch_shoot_3pa_per_min",
        "pull_up_3pa_per_min",
        "catch_shoot_share",
        "pull_up_share",
        "assisted_3pm_share",
        "unassisted_3pm_share",
    ],
    "tracking_touches": [
        "touches_per_min",
        "time_of_poss_per_min",
        "avg_seconds_per_touch",
        "avg_dribbles_per_touch",
    ],
    "opponent_shooting_env": [
        "opp_3pa_allowed",
        "opp_3pm_allowed",
        "opp_fg3_pct_allowed",
        "opp_dribble_env",
        "opp_touch_env",
        "opp_closest_def_env",
    ],
    "teammate_injury_context": [
        "primary_creator_out",
        "secondary_creator_out",
        "starting_big_out",
        "high_volume_wing_out",
    ],
    "rest_travel": [
        "rest_days",
        "is_back_to_back",
        "is_3in4",
        "travel_miles",
        "time_zone_shift",
        "altitude_flag",
    ],
    "game_environment": [
        "spread",
        "team_total",
        "blowout_probability",
        "is_home",
    ],
    "archetype_interactions": [
        "archetype",  # and any archetype_x_* interaction terms
    ],
    "rolling_volume": [
        "3pa_per_min_l5",
        "3pa_per_min_l10",
        "3pa_per_min_l20",
        "3pa_per_36_l10",
        "3pa_per_36_l20",
    ],
    "empirical_bayes_shrinkage": [
        "eb_fg3_pct",
        "empirical_bayes_fg3_pct",
    ],
}

# Metric names tracked in ablation deltas
METRIC_NAMES = [
    "log_loss",
    "brier",
    "clv",
    "roi",
    "minutes_mae",
    "3pa_mae",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationResults:
    """Container for ablation study outputs."""

    full_model_metrics: dict = field(default_factory=dict)
    ablation_deltas: dict = field(default_factory=dict)
    # block_name -> {metric: delta_value}

    feature_importance_rank: list[tuple[str, float]] = field(default_factory=list)
    # (block_name, importance_score) sorted descending

    summary_table: list[dict] = field(default_factory=list)
    # Formatted rows for display


# ---------------------------------------------------------------------------
# Core ablation report
# ---------------------------------------------------------------------------

class AblationReport:
    """Remove each major feature block independently and measure impact.

    The ablation study proceeds as follows:
    1. Run the full model (no ablation) as baseline and collect metrics.
    2. For each feature block in ``FEATURE_BLOCKS``:
       a. Zero-out / mask all features in that block.
       b. Retrain models with ablated features.
       c. Re-run walk-forward evaluation.
       d. Compute delta in key metrics relative to the baseline.
    3. Rank feature blocks by importance (largest negative impact when
       removed = most important).

    Parameters
    ----------
    repository :
        Database / data access layer.
    feature_builder :
        Feature snapshot builder.
    simulator :
        Monte Carlo simulator for 3PM distribution.
    decision_engine :
        Betting decision engine.
    walk_forward_evaluator :
        Walk-forward evaluation runner that accepts feature overrides.
    """

    def __init__(
        self,
        repository,
        feature_builder,
        simulator,
        decision_engine,
        walk_forward_evaluator,
    ) -> None:
        self.repository = repository
        self.feature_builder = feature_builder
        self.simulator = simulator
        self.decision_engine = decision_engine
        self.walk_forward_evaluator = walk_forward_evaluator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date: str,
        train_window_days: int = 180,
    ) -> AblationResults:
        """Execute the full ablation study over a date range.

        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD) for the walk-forward evaluation window.
        end_date : str
            End date (YYYY-MM-DD) for the walk-forward evaluation window.
        train_window_days : int
            Number of days in each walk-forward training window.

        Returns
        -------
        AblationResults
            Comprehensive ablation results including deltas, rankings,
            and a summary table.
        """
        logger.info(
            "Starting ablation study from %s to %s (train_window=%d days)",
            start_date,
            end_date,
            train_window_days,
        )

        # Step 1: Run the full (unablated) model as baseline
        logger.info("Running baseline (full model) walk-forward evaluation")
        baseline_wf_results = self.walk_forward_evaluator.run(
            start_date=start_date,
            end_date=end_date,
            train_window_days=train_window_days,
            ablated_features=None,
        )
        full_model_metrics = self._extract_metrics(baseline_wf_results)
        logger.info("Baseline metrics: %s", full_model_metrics)

        # Step 2: Ablate each block and re-evaluate
        ablation_deltas: dict[str, dict[str, float]] = {}

        for block_name, block_features in FEATURE_BLOCKS.items():
            logger.info(
                "Ablating block '%s' (%d features): %s",
                block_name,
                len(block_features),
                block_features,
            )

            # Determine the complete set of features to mask, including
            # any archetype interaction terms (archetype_x_*).
            features_to_mask = list(block_features)
            if block_name == "archetype_interactions":
                features_to_mask = self._expand_archetype_interactions(
                    features_to_mask, baseline_wf_results
                )

            # Run walk-forward with this block ablated
            ablated_wf_results = self.walk_forward_evaluator.run(
                start_date=start_date,
                end_date=end_date,
                train_window_days=train_window_days,
                ablated_features=features_to_mask,
            )
            ablated_metrics = self._extract_metrics(ablated_wf_results)

            # Compute deltas: ablated - baseline
            # Positive delta for log_loss/brier/mae means the model got
            # *worse* when the block was removed (block is important).
            # Negative delta for clv/roi means the model got *worse*.
            deltas: dict[str, float] = {}
            for metric in METRIC_NAMES:
                baseline_val = full_model_metrics.get(metric, 0.0)
                ablated_val = ablated_metrics.get(metric, 0.0)
                deltas[metric] = round(ablated_val - baseline_val, 6)

            ablation_deltas[block_name] = deltas
            logger.info("Block '%s' deltas: %s", block_name, deltas)

        # Step 3: Rank blocks by importance
        feature_importance_rank = self._compute_importance_ranking(
            ablation_deltas
        )

        # Step 4: Build summary table
        summary_table = self._build_summary_table(
            full_model_metrics, ablation_deltas, feature_importance_rank
        )

        results = AblationResults(
            full_model_metrics=full_model_metrics,
            ablation_deltas=ablation_deltas,
            feature_importance_rank=feature_importance_rank,
            summary_table=summary_table,
        )

        logger.info(
            "Ablation study complete. Most important block: %s (score=%.4f)",
            feature_importance_rank[0][0] if feature_importance_rank else "N/A",
            feature_importance_rank[0][1] if feature_importance_rank else 0.0,
        )

        return results

    def generate_table(self, results: AblationResults) -> list[dict]:
        """Generate a formatted table from ablation results.

        Each row contains:
        - feature_block: name of the ablated feature group
        - n_features_removed: number of features zeroed out
        - delta_log_loss: change in log loss when block removed
        - delta_brier: change in Brier score when block removed
        - delta_clv: change in CLV when block removed
        - delta_roi: change in ROI when block removed
        - delta_minutes_mae: change in minutes MAE when block removed
        - delta_3pa_mae: change in 3PA MAE when block removed
        - importance_rank: 1-based rank (1 = most important)

        Returns
        -------
        list[dict]
            Sorted by importance (most important first).
        """
        return results.summary_table

    def format_report(self, results: AblationResults) -> str:
        """Produce a human-readable text report of the ablation study.

        Parameters
        ----------
        results : AblationResults
            Output of :meth:`run`.

        Returns
        -------
        str
            Multi-line text report suitable for logging or file output.
        """
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Full model baseline
        lines.append("BASELINE (FULL MODEL) METRICS:")
        lines.append("-" * 40)
        for metric, value in sorted(results.full_model_metrics.items()):
            lines.append(f"  {metric:<20s}: {value:>10.4f}")
        lines.append("")

        # Importance ranking
        lines.append("FEATURE BLOCK IMPORTANCE RANKING:")
        lines.append("-" * 40)
        for rank, (block_name, score) in enumerate(
            results.feature_importance_rank, start=1
        ):
            n_features = len(FEATURE_BLOCKS.get(block_name, []))
            lines.append(
                f"  #{rank:<3d} {block_name:<30s} "
                f"(score={score:>8.4f}, n_features={n_features})"
            )
        lines.append("")

        # Detailed ablation table
        lines.append("DETAILED ABLATION DELTAS:")
        lines.append("-" * 100)

        # Header
        header = (
            f"  {'Block':<28s} {'N':<4s} "
            f"{'dLogLoss':>9s} {'dBrier':>9s} {'dCLV':>9s} "
            f"{'dROI':>9s} {'dMinMAE':>9s} {'d3paMAE':>9s} "
            f"{'Rank':>5s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for row in results.summary_table:
            lines.append(
                f"  {row['feature_block']:<28s} {row['n_features_removed']:<4d} "
                f"{row['delta_log_loss']:>9.4f} {row['delta_brier']:>9.4f} "
                f"{row['delta_clv']:>9.4f} {row['delta_roi']:>9.4f} "
                f"{row['delta_minutes_mae']:>9.4f} {row['delta_3pa_mae']:>9.4f} "
                f"{row['importance_rank']:>5d}"
            )

        lines.append("")

        # Interpretation notes
        lines.append("INTERPRETATION:")
        lines.append("-" * 40)
        lines.append(
            "  Positive delta_log_loss / delta_brier / delta_*_mae means "
            "performance DEGRADED when block was removed (block is valuable)."
        )
        lines.append(
            "  Negative delta_clv / delta_roi means profitability DECREASED "
            "when block was removed (block is valuable)."
        )
        lines.append(
            "  Importance score combines all metric deltas into a single "
            "score (higher = more important)."
        )
        lines.append("")

        # Flag blocks with negligible impact
        negligible = [
            block_name
            for block_name, score in results.feature_importance_rank
            if abs(score) < 0.001
        ]
        if negligible:
            lines.append("NEGLIGIBLE BLOCKS (candidates for removal):")
            lines.append("-" * 40)
            for block_name in negligible:
                lines.append(f"  - {block_name}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metrics(self, wf_results: Any) -> dict[str, float]:
        """Extract the standard set of metrics from walk-forward results.

        Handles both dict-style and object-style walk-forward results.

        Returns
        -------
        dict
            Keys are the items in ``METRIC_NAMES``, values are floats.
        """
        metrics: dict[str, float] = {}

        # Try attribute access first, fall back to dict access
        def _get(obj: Any, key: str, default: float = 0.0) -> float:
            if isinstance(obj, dict):
                return float(obj.get(key, default))
            return float(getattr(obj, key, default))

        # Navigate different possible result structures
        if isinstance(wf_results, dict):
            result_metrics = wf_results.get("metrics", wf_results)
        else:
            result_metrics = getattr(wf_results, "metrics", wf_results)

        # Extract 3PM metrics (log_loss, brier)
        three_pm = (
            result_metrics.get("three_pm", {})
            if isinstance(result_metrics, dict)
            else getattr(result_metrics, "three_pm", {})
        )
        metrics["log_loss"] = _get(three_pm, "log_loss")
        metrics["brier"] = _get(three_pm, "brier_score", _get(three_pm, "brier"))

        # Extract betting metrics (clv, roi)
        betting = (
            result_metrics.get("betting", {})
            if isinstance(result_metrics, dict)
            else getattr(result_metrics, "betting", {})
        )
        metrics["clv"] = _get(betting, "clv_mean", _get(betting, "clv"))
        metrics["roi"] = _get(betting, "roi")

        # Extract minutes metrics
        minutes = (
            result_metrics.get("minutes", {})
            if isinstance(result_metrics, dict)
            else getattr(result_metrics, "minutes", {})
        )
        metrics["minutes_mae"] = _get(
            minutes, "mae_overall", _get(minutes, "minutes_mae")
        )

        # Extract 3PA metrics
        three_pa = (
            result_metrics.get("three_pa", {})
            if isinstance(result_metrics, dict)
            else getattr(result_metrics, "three_pa", {})
        )
        metrics["3pa_mae"] = _get(three_pa, "mae", _get(three_pa, "3pa_mae"))

        return metrics

    def _expand_archetype_interactions(
        self,
        base_features: list[str],
        wf_results: Any,
    ) -> list[str]:
        """Expand archetype block to include any archetype_x_* interaction terms.

        Scans the walk-forward results for feature names matching the
        ``archetype_x_*`` pattern and adds them to the ablation list.

        Parameters
        ----------
        base_features : list[str]
            The initial list (typically just ``["archetype"]``).
        wf_results :
            Walk-forward results that may contain feature name info.

        Returns
        -------
        list[str]
            Expanded list including any discovered interaction terms.
        """
        expanded = list(base_features)

        # Try to extract feature names from wf results
        feature_names: list[str] = []
        if isinstance(wf_results, dict):
            feature_names = wf_results.get("feature_names", [])
            # Also check inside predictions for feature columns
            predictions = wf_results.get("predictions", [])
            if predictions and isinstance(predictions[0], dict):
                feature_names = list(predictions[0].get("features", {}).keys())
        else:
            feature_names = getattr(wf_results, "feature_names", [])

        # Add any archetype_x_* interaction columns
        for fname in feature_names:
            if fname.startswith("archetype_x_") or fname.startswith("archetype_"):
                if fname not in expanded:
                    expanded.append(fname)

        logger.info(
            "Expanded archetype block from %d to %d features",
            len(base_features),
            len(expanded),
        )
        return expanded

    def _compute_importance_ranking(
        self,
        ablation_deltas: dict[str, dict[str, float]],
    ) -> list[tuple[str, float]]:
        """Compute a single importance score per block and rank them.

        The importance score aggregates the impact of removing each block:
        - For error metrics (log_loss, brier, minutes_mae, 3pa_mae): a
          positive delta means the model got worse -- that block is
          important.
        - For value metrics (clv, roi): a negative delta means the model
          got worse -- that block is important.

        The score is the sum of normalised absolute deltas, with sign
        flipped for value metrics so that higher score always means
        more important.

        Returns
        -------
        list[tuple[str, float]]
            (block_name, importance_score) sorted descending by score.
        """
        # Metrics where a positive delta = model degraded (block important)
        error_metrics = {"log_loss", "brier", "minutes_mae", "3pa_mae"}
        # Metrics where a negative delta = model degraded (block important)
        value_metrics = {"clv", "roi"}

        # Collect all deltas across blocks for normalisation
        all_deltas: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
        for block_deltas in ablation_deltas.values():
            for metric in METRIC_NAMES:
                all_deltas[metric].append(block_deltas.get(metric, 0.0))

        # Compute standard deviations for normalisation (avoid division by zero)
        metric_std: dict[str, float] = {}
        for metric, values in all_deltas.items():
            std = float(np.std(values)) if len(values) > 1 else 1.0
            metric_std[metric] = std if std > 1e-10 else 1.0

        # Compute importance score per block
        scores: list[tuple[str, float]] = []
        for block_name, deltas in ablation_deltas.items():
            score = 0.0
            for metric in METRIC_NAMES:
                delta = deltas.get(metric, 0.0)
                normalised = delta / metric_std[metric]

                if metric in error_metrics:
                    # Positive delta = worse performance = important
                    score += normalised
                elif metric in value_metrics:
                    # Negative delta = worse performance = important
                    score -= normalised

            scores.append((block_name, round(score, 4)))

        # Sort descending by importance
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _build_summary_table(
        self,
        full_model_metrics: dict[str, float],
        ablation_deltas: dict[str, dict[str, float]],
        importance_rank: list[tuple[str, float]],
    ) -> list[dict]:
        """Build the formatted summary table.

        Returns
        -------
        list[dict]
            One row per feature block, sorted by importance rank.
        """
        # Build a lookup from block_name to rank
        rank_lookup = {
            block_name: rank
            for rank, (block_name, _) in enumerate(importance_rank, start=1)
        }

        table: list[dict] = []
        for block_name, _ in importance_rank:
            deltas = ablation_deltas.get(block_name, {})
            block_features = FEATURE_BLOCKS.get(block_name, [])

            # For archetype_interactions, the actual count may be larger
            # due to interaction term expansion, but we report the base count
            n_features = len(block_features)

            row = {
                "feature_block": block_name,
                "n_features_removed": n_features,
                "delta_log_loss": deltas.get("log_loss", 0.0),
                "delta_brier": deltas.get("brier", 0.0),
                "delta_clv": deltas.get("clv", 0.0),
                "delta_roi": deltas.get("roi", 0.0),
                "delta_minutes_mae": deltas.get("minutes_mae", 0.0),
                "delta_3pa_mae": deltas.get("3pa_mae", 0.0),
                "importance_rank": rank_lookup.get(block_name, 0),
            }
            table.append(row)

        return table
