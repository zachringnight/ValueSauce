"""Walk-forward out-of-sample evaluation engine for NBA 3PM Props.

Performs expanding-window walk-forward evaluation to validate the full
decomposition pipeline (minutes -> 3PA -> make-rate -> Monte Carlo -> decision)
against three baselines:

1. Rolling average of raw 3PM
2. Direct single-GBM 3PM predictor
3. Bookmaker no-vig implied probabilities

Each fold trains on an expanding historical window and scores on a held-out
test window that the models have never seen, ensuring strict temporal ordering
and no look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Slicing constants
# --------------------------------------------------------------------------- #

LINE_BUCKETS: Dict[str, Callable[[float], bool]] = {
    "0.5": lambda l: l == 0.5,
    "1.5": lambda l: l == 1.5,
    "2.5": lambda l: l == 2.5,
    "3.5+": lambda l: l >= 3.5,
}

SPREAD_BUCKETS: Dict[str, Callable[[float], bool]] = {
    "blowout_fav": lambda s: s <= -10,
    "moderate_fav": lambda s: -10 < s <= -3,
    "tossup": lambda s: -3 < s < 3,
    "moderate_dog": lambda s: 3 <= s < 10,
    "blowout_dog": lambda s: s >= 10,
}

REST_BUCKETS: Dict[str, Callable[[int], bool]] = {
    "b2b": lambda r: r == 0,
    "1_day": lambda r: r == 1,
    "2_days": lambda r: r == 2,
    "3plus": lambda r: r >= 3,
}

TIME_BUCKETS: List[str] = [
    "D-1_1705_LOCAL",
    "B2B_1305_LOCAL",
    "GAMEDAY_1130_LOCAL",
    "T-90",
    "T-30",
    "T-5",
]

ARCHETYPES: List[str] = [
    "movement_wing_shooter",
    "pull_up_guard",
    "stretch_big",
    "stationary_spacer",
    "bench_microwave",
]


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardResults:
    """Container for walk-forward evaluation outputs.

    Attributes
    ----------
    predictions : list[dict]
        All per-prediction rows with actuals, model outputs, baselines,
        and betting outcomes.
    folds : list[dict]
        Per-fold metadata including training/test date ranges and sample
        counts.
    production_metrics : dict
        Aggregated metrics for the production (decomposition) model.
    baseline_metrics : dict
        Dict of ``baseline_name -> aggregated metrics``.
    sliced_metrics : dict
        Dict of ``dimension -> {slice_value -> metrics}``.
    """

    predictions: List[Dict[str, Any]] = field(default_factory=list)
    folds: List[Dict[str, Any]] = field(default_factory=list)
    production_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    sliced_metrics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Walk-forward evaluator
# --------------------------------------------------------------------------- #


class WalkForwardEvaluator:
    """Expanding-window walk-forward OOS evaluation engine.

    Orchestrates the full validation workflow: splitting into temporal
    folds, retraining models on expanding training windows, scoring
    held-out test windows with frozen features, running baselines on the
    same test data, and collecting per-prediction rows for downstream
    metrics computation.

    Parameters
    ----------
    repository :
        Data repository providing access to historical game data, player
        stats, odds, and features.
    feature_builder :
        ``FeatureSnapshotBuilder`` for constructing point-in-time feature
        snapshots.
    minutes_model :
        ``MinutesModel`` instance for minutes prediction.
    three_pa_model :
        ``ThreePAModel`` instance for 3PA count prediction.
    make_rate_model :
        ``MakeRateModel`` instance for make-rate prediction.
    simulator :
        ``MonteCarloSimulator`` instance for distributional pricing.
    decision_engine :
        ``DecisionEngine`` instance for bet evaluation.
    baselines : dict
        Dictionary mapping baseline names to baseline instances.
        Expected keys: ``"rolling_avg"``, ``"direct_3pm"``,
        ``"bookmaker"``.
    leakage_detector : optional
        ``LeakageDetector`` instance for temporal leakage checks.  When
        provided, every feature snapshot is validated before scoring.
    """

    def __init__(
        self,
        repository: Any,
        feature_builder: Any,
        minutes_model: Any,
        three_pa_model: Any,
        make_rate_model: Any,
        simulator: Any,
        decision_engine: Any,
        baselines: Dict[str, Any],
        leakage_detector: Optional[Any] = None,
    ) -> None:
        self.repository = repository
        self.feature_builder = feature_builder
        self.minutes_model = minutes_model
        self.three_pa_model = three_pa_model
        self.make_rate_model = make_rate_model
        self.simulator = simulator
        self.decision_engine = decision_engine
        self.baselines = baselines
        self.leakage_detector = leakage_detector

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        start_date: str,
        end_date: str,
        train_window_days: int = 180,
        retrain_every_days: int = 30,
        freeze_offset_minutes: int = -60,
    ) -> WalkForwardResults:
        """Execute the full walk-forward evaluation.

        Parameters
        ----------
        start_date : str
            ISO-8601 date string for the start of the evaluation period.
            The first training window ends here; the first test window
            starts from this date.
        end_date : str
            ISO-8601 date string for the end of the evaluation period.
        train_window_days : int
            Minimum number of days of training data in the initial fold.
            Subsequent folds use an expanding window from the earliest
            available date.
        retrain_every_days : int
            Number of days in each test window before retraining.
        freeze_offset_minutes : int
            Minutes before tipoff at which to freeze features.  Negative
            values mean *before* tipoff (e.g. ``-60`` = 1 hour before).

        Returns
        -------
        WalkForwardResults
            Complete results container with predictions, fold metadata,
            production metrics, baseline metrics, and sliced metrics.
        """
        logger.info(
            "Starting walk-forward evaluation: %s to %s "
            "(train_window=%dd, retrain_every=%dd, freeze_offset=%dm)",
            start_date,
            end_date,
            train_window_days,
            retrain_every_days,
            freeze_offset_minutes,
        )

        folds = self._generate_folds(
            start_date, end_date, train_window_days, retrain_every_days
        )
        logger.info("Generated %d walk-forward folds", len(folds))

        all_predictions: List[Dict[str, Any]] = []
        fold_metadata: List[Dict[str, Any]] = []

        for fold_idx, fold in enumerate(folds):
            logger.info(
                "Fold %d/%d: train=[%s, %s], test=[%s, %s]",
                fold_idx + 1,
                len(folds),
                fold["train_start"],
                fold["train_end"],
                fold["test_start"],
                fold["test_end"],
            )

            # --- (a) Train all models on the training window ---------------
            train_data = self._get_training_data(
                fold["train_start"], fold["train_end"]
            )
            n_train = self._train_models(train_data)

            # --- (b) Score test-window games --------------------------------
            test_games = self._get_test_games(
                fold["test_start"], fold["test_end"]
            )
            n_test = len(test_games)
            logger.info(
                "Fold %d: %d training samples, %d test games",
                fold_idx + 1,
                n_train,
                n_test,
            )

            fold_predictions = self._score_test_window(
                test_games, freeze_offset_minutes, fold_idx
            )

            # --- (c) Score baselines on the same test window ----------------
            fold_predictions = self._score_baselines(
                fold_predictions, test_games
            )

            all_predictions.extend(fold_predictions)
            fold_metadata.append(
                {
                    "fold_index": fold_idx,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_predictions": len(fold_predictions),
                }
            )

        logger.info(
            "Walk-forward complete: %d total predictions across %d folds",
            len(all_predictions),
            len(folds),
        )

        # --- Aggregate metrics -------------------------------------------
        production_metrics = self._compute_production_metrics(all_predictions)
        baseline_metrics = self._compute_baseline_metrics(all_predictions)
        sliced_metrics = self._compute_sliced_metrics(all_predictions)

        return WalkForwardResults(
            predictions=all_predictions,
            folds=fold_metadata,
            production_metrics=production_metrics,
            baseline_metrics=baseline_metrics,
            sliced_metrics=sliced_metrics,
        )

    def slice_results(
        self, results: WalkForwardResults, dimension: str
    ) -> Dict[str, WalkForwardResults]:
        """Slice results by any supported dimension.

        Parameters
        ----------
        results : WalkForwardResults
            Complete results from ``run()``.
        dimension : str
            The dimension to slice by.  Supported values:
            ``"line_bucket"``, ``"spread_bucket"``, ``"rest_bucket"``,
            ``"archetype"``, ``"time_bucket"``, ``"tracking_available"``,
            ``"is_home"``, ``"is_b2b"``.

        Returns
        -------
        dict[str, WalkForwardResults]
            A mapping of slice value to a ``WalkForwardResults`` containing
            only the predictions belonging to that slice.
        """
        sliced: Dict[str, WalkForwardResults] = {}

        if dimension == "line_bucket":
            buckets = LINE_BUCKETS
        elif dimension == "spread_bucket":
            buckets = SPREAD_BUCKETS
        elif dimension == "rest_bucket":
            buckets = REST_BUCKETS
        elif dimension == "archetype":
            # Build a dict of archetype -> filter
            buckets = {a: (lambda arch, a=a: arch == a) for a in ARCHETYPES}
        elif dimension == "time_bucket":
            buckets = {t: (lambda tb, t=t: tb == t) for t in TIME_BUCKETS}
        elif dimension == "tracking_available":
            buckets = {
                "True": lambda v: v is True,
                "False": lambda v: v is False,
            }
        elif dimension == "is_home":
            buckets = {
                "home": lambda v: v is True,
                "away": lambda v: v is False,
            }
        elif dimension == "is_b2b":
            buckets = {
                "b2b": lambda v: v is True,
                "not_b2b": lambda v: v is False,
            }
        else:
            logger.warning("Unknown dimension '%s', returning empty", dimension)
            return sliced

        for bucket_name, filter_fn in buckets.items():
            bucket_preds = [
                p for p in results.predictions
                if filter_fn(p.get(dimension, p.get(dimension.replace("_bucket", ""), None)))
            ]
            if bucket_preds:
                sliced[bucket_name] = WalkForwardResults(
                    predictions=bucket_preds,
                    folds=results.folds,
                    production_metrics=self._compute_production_metrics(bucket_preds),
                    baseline_metrics=self._compute_baseline_metrics(bucket_preds),
                    sliced_metrics={},
                )

        return sliced

    # ------------------------------------------------------------------ #
    # Bucket assignment
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assign_buckets(row: Dict[str, Any]) -> Dict[str, Any]:
        """Assign line_bucket, spread_bucket, and rest_bucket to a prediction row.

        Parameters
        ----------
        row : dict
            A single prediction row containing at least ``line``,
            ``spread``, and ``rest_days`` fields.

        Returns
        -------
        dict
            The input row augmented with ``line_bucket``,
            ``spread_bucket``, and ``rest_bucket`` keys.
        """
        # Line bucket
        line = row.get("line", 0.0)
        row["line_bucket"] = "unknown"
        for bucket_name, check_fn in LINE_BUCKETS.items():
            if check_fn(line):
                row["line_bucket"] = bucket_name
                break

        # Spread bucket
        spread = row.get("spread", 0.0)
        if spread is None:
            spread = 0.0
        row["spread_bucket"] = "unknown"
        for bucket_name, check_fn in SPREAD_BUCKETS.items():
            if check_fn(spread):
                row["spread_bucket"] = bucket_name
                break

        # Rest bucket
        rest_days = row.get("rest_days", 1)
        if rest_days is None:
            rest_days = 1
        row["rest_bucket"] = "unknown"
        for bucket_name, check_fn in REST_BUCKETS.items():
            if check_fn(rest_days):
                row["rest_bucket"] = bucket_name
                break

        return row

    # ------------------------------------------------------------------ #
    # Fold generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_folds(
        start_date: str,
        end_date: str,
        train_window_days: int,
        retrain_every_days: int,
    ) -> List[Dict[str, str]]:
        """Generate expanding-window walk-forward folds.

        The first fold trains on ``[start_date - train_window_days, start_date)``
        and tests on ``[start_date, start_date + retrain_every_days)``.
        Each subsequent fold expands the training window to include all
        prior data up to the new test start.

        Parameters
        ----------
        start_date : str
            ISO date of the first test window start.
        end_date : str
            ISO date of the evaluation end.
        train_window_days : int
            Minimum training window size in days.
        retrain_every_days : int
            Test window size in days.

        Returns
        -------
        list[dict]
            Each dict has ``train_start``, ``train_end``, ``test_start``,
            ``test_end`` as ISO date strings.
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        train_origin = start - timedelta(days=train_window_days)

        folds: List[Dict[str, str]] = []
        current_test_start = start

        while current_test_start < end:
            current_test_end = min(
                current_test_start + timedelta(days=retrain_every_days),
                end,
            )

            folds.append(
                {
                    "train_start": train_origin.strftime("%Y-%m-%d"),
                    "train_end": (current_test_start - timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    ),
                    "test_start": current_test_start.strftime("%Y-%m-%d"),
                    "test_end": current_test_end.strftime("%Y-%m-%d"),
                }
            )

            current_test_start = current_test_end

        return folds

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def _get_training_data(
        self, train_start: str, train_end: str
    ) -> Dict[str, Any]:
        """Retrieve training data from the repository.

        Parameters
        ----------
        train_start : str
            Training window start date.
        train_end : str
            Training window end date.

        Returns
        -------
        dict
            Training data payload with features and targets for all
            three sub-models.
        """
        try:
            data = self.repository.get_training_data(train_start, train_end)
            return data
        except Exception as exc:
            logger.error(
                "Failed to retrieve training data for [%s, %s]: %s",
                train_start,
                train_end,
                exc,
            )
            return {}

    def _train_models(self, train_data: Dict[str, Any]) -> int:
        """Train all three sub-models on the given training data.

        Parameters
        ----------
        train_data : dict
            Training data payload from the repository.

        Returns
        -------
        int
            Number of training samples used.
        """
        if not train_data:
            logger.warning("Empty training data; skipping model training")
            return 0

        n_train = 0

        # --- Minutes model ------------------------------------------------
        try:
            X_min = np.asarray(train_data.get("minutes_X", []), dtype=np.float64)
            y_min = np.asarray(train_data.get("minutes_y", []), dtype=np.float64)
            if X_min.size > 0 and y_min.size > 0:
                self.minutes_model.fit(X_min, y_min)
                n_train = max(n_train, len(y_min))
                logger.info("Minutes model trained on %d samples", len(y_min))
        except Exception as exc:
            logger.error("Minutes model training failed: %s", exc)

        # --- 3PA model ----------------------------------------------------
        try:
            X_3pa = np.asarray(train_data.get("three_pa_X", []), dtype=np.float64)
            y_3pa = np.asarray(train_data.get("three_pa_y", []), dtype=np.float64)
            minutes_exp = train_data.get("three_pa_minutes_exposure")
            if minutes_exp is not None:
                minutes_exp = np.asarray(minutes_exp, dtype=np.float64)
            if X_3pa.size > 0 and y_3pa.size > 0:
                self.three_pa_model.fit(X_3pa, y_3pa, minutes_exposure=minutes_exp)
                n_train = max(n_train, len(y_3pa))
                logger.info("3PA model trained on %d samples", len(y_3pa))
        except Exception as exc:
            logger.error("3PA model training failed: %s", exc)

        # --- Make-rate model ----------------------------------------------
        try:
            X_mr = np.asarray(train_data.get("make_rate_X", []), dtype=np.float64)
            y_makes = np.asarray(train_data.get("make_rate_y_makes", []), dtype=np.float64)
            y_attempts = train_data.get("make_rate_y_attempts")
            if y_attempts is not None:
                y_attempts = np.asarray(y_attempts, dtype=np.float64)
            if X_mr.size > 0 and y_makes.size > 0:
                self.make_rate_model.fit(X_mr, y_makes, y_attempts=y_attempts)
                n_train = max(n_train, len(y_makes))
                logger.info("Make-rate model trained on %d samples", len(y_makes))
        except Exception as exc:
            logger.error("Make-rate model training failed: %s", exc)

        # --- Direct 3PM baseline (if present) -----------------------------
        direct_baseline = self.baselines.get("direct_3pm")
        if direct_baseline is not None:
            try:
                X_direct = np.asarray(
                    train_data.get("direct_3pm_X", train_data.get("three_pa_X", [])),
                    dtype=np.float64,
                )
                y_direct = np.asarray(
                    train_data.get("direct_3pm_y", []), dtype=np.float64
                )
                if X_direct.size > 0 and y_direct.size > 0:
                    direct_baseline.fit(X_direct, y_direct)
                    logger.info(
                        "Direct 3PM baseline trained on %d samples", len(y_direct)
                    )
            except Exception as exc:
                logger.error("Direct 3PM baseline training failed: %s", exc)

        return n_train

    # ------------------------------------------------------------------ #
    # Test-window scoring
    # ------------------------------------------------------------------ #

    def _get_test_games(
        self, test_start: str, test_end: str
    ) -> List[Dict[str, Any]]:
        """Retrieve test-window games from the repository.

        Parameters
        ----------
        test_start : str
            Test window start date.
        test_end : str
            Test window end date.

        Returns
        -------
        list[dict]
            List of game dicts, each containing player-game-level data
            with actuals, odds, context, and historical game logs.
        """
        try:
            games = self.repository.get_test_games(test_start, test_end)
            return games if games else []
        except Exception as exc:
            logger.error(
                "Failed to retrieve test games for [%s, %s]: %s",
                test_start,
                test_end,
                exc,
            )
            return []

    def _score_test_window(
        self,
        test_games: List[Dict[str, Any]],
        freeze_offset_minutes: int,
        fold_idx: int,
    ) -> List[Dict[str, Any]]:
        """Score all test-window games using the production pipeline.

        For each player-game in the test window:
        1. Build a feature snapshot frozen at ``freeze_offset_minutes``
           before tipoff.
        2. Optionally run leakage checks.
        3. Predict minutes (p10/p50/p90), 3PA (mean/dispersion),
           make-rate (mean/uncertainty).
        4. Run Monte Carlo simulation.
        5. Run the decision engine to get edge, EV, recommended side.
        6. Record the full prediction row including actuals.

        Parameters
        ----------
        test_games : list[dict]
            Games to score.
        freeze_offset_minutes : int
            Minutes before tipoff for feature freeze.
        fold_idx : int
            Current fold index (for metadata).

        Returns
        -------
        list[dict]
            Per-prediction rows.
        """
        predictions: List[Dict[str, Any]] = []

        for game in test_games:
            try:
                row = self._score_single_game(
                    game, freeze_offset_minutes, fold_idx
                )
                if row is not None:
                    row = self._assign_buckets(row)
                    predictions.append(row)
            except Exception as exc:
                logger.warning(
                    "Failed to score game_id=%s player_id=%s: %s",
                    game.get("game_id", "?"),
                    game.get("player_id", "?"),
                    exc,
                )

        return predictions

    def _score_single_game(
        self,
        game: Dict[str, Any],
        freeze_offset_minutes: int,
        fold_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Score a single player-game opportunity.

        Parameters
        ----------
        game : dict
            Player-game data from the repository.
        freeze_offset_minutes : int
            Feature freeze offset.
        fold_idx : int
            Fold index.

        Returns
        -------
        dict or None
            The prediction row, or None if scoring fails.
        """
        game_id = game.get("game_id")
        player_id = game.get("player_id")
        game_date = game.get("game_date", "")
        tipoff_time = game.get("tipoff_time_utc")

        # --- Build feature snapshot at freeze offset ----------------------
        freeze_timestamp = game.get("freeze_timestamp")
        if freeze_timestamp is None and tipoff_time is not None:
            try:
                tipoff_dt = datetime.fromisoformat(str(tipoff_time))
                freeze_dt = tipoff_dt + timedelta(minutes=freeze_offset_minutes)
                freeze_timestamp = freeze_dt.isoformat()
            except (ValueError, TypeError):
                freeze_timestamp = game_date

        try:
            snapshot = self.feature_builder.build_snapshot(
                player_id=player_id,
                game_id=game_id,
                freeze_timestamp=freeze_timestamp or game_date,
                player_games=game.get("player_games", []),
                tracking_games=game.get("tracking_games"),
                opponent_shooting=game.get("opponent_shooting"),
                game_context=game.get("game_context", {}),
                injury_snapshot=game.get("injury_snapshot"),
                teammate_statuses=game.get("teammate_statuses"),
                odds_snapshot_timestamp=game.get("odds_snapshot_timestamp"),
            )
        except Exception as exc:
            logger.warning(
                "Feature build failed for %s/%s: %s", game_id, player_id, exc
            )
            return None

        # --- Leakage check ------------------------------------------------
        if self.leakage_detector is not None and freeze_timestamp:
            try:
                freeze_dt = datetime.fromisoformat(str(freeze_timestamp))
                passed, violations = self.leakage_detector.run_all_checks(
                    snapshot, freeze_dt
                )
                if not passed:
                    logger.warning(
                        "Leakage detected for %s/%s: %s",
                        game_id,
                        player_id,
                        violations,
                    )
                    return None
            except Exception as exc:
                logger.warning("Leakage check error: %s", exc)

        # --- Extract feature arrays for model inference -------------------
        minutes_features = np.asarray(
            game.get("minutes_features", snapshot.get("_minutes_X", [])),
            dtype=np.float64,
        ).reshape(1, -1) if game.get("minutes_features") or snapshot.get("_minutes_X") else None

        three_pa_features = np.asarray(
            game.get("three_pa_features", snapshot.get("_three_pa_X", [])),
            dtype=np.float64,
        ).reshape(1, -1) if game.get("three_pa_features") or snapshot.get("_three_pa_X") else None

        make_rate_features = np.asarray(
            game.get("make_rate_features", snapshot.get("_make_rate_X", [])),
            dtype=np.float64,
        ).reshape(1, -1) if game.get("make_rate_features") or snapshot.get("_make_rate_X") else None

        # --- Predict minutes (p10/p50/p90) --------------------------------
        minutes_p10, minutes_p50, minutes_p90 = 0.0, 20.0, 36.0
        if minutes_features is not None and minutes_features.size > 0:
            try:
                quantiles = self.minutes_model.predict_quantiles(minutes_features)
                minutes_p10 = float(quantiles["p10"][0])
                minutes_p50 = float(quantiles["p50"][0])
                minutes_p90 = float(quantiles["p90"][0])
            except Exception as exc:
                logger.warning("Minutes prediction failed: %s", exc)

        # --- Predict 3PA (mean/dispersion) --------------------------------
        three_pa_mean, three_pa_dispersion = 5.0, 3.0
        if three_pa_features is not None and three_pa_features.size > 0:
            try:
                dist_params = self.three_pa_model.predict_distribution_params(
                    three_pa_features
                )
                three_pa_mean = float(dist_params["mean"][0])
                three_pa_dispersion = float(dist_params["dispersion"][0])
            except Exception as exc:
                logger.warning("3PA prediction failed: %s", exc)

        # --- Predict make probability (mean/uncertainty) ------------------
        make_prob_mean, make_prob_uncertainty = 0.36, 0.04
        if make_rate_features is not None and make_rate_features.size > 0:
            try:
                mr_preds = self.make_rate_model.predict_with_uncertainty(
                    make_rate_features
                )
                make_prob_mean = float(mr_preds["mean"][0])
                make_prob_uncertainty = float(mr_preds["uncertainty"][0])
            except Exception as exc:
                logger.warning("Make-rate prediction failed: %s", exc)

        # --- Monte Carlo simulation ---------------------------------------
        line = float(game.get("line", 2.5))
        try:
            sim_result = self.simulator.simulate(
                minutes_p10=minutes_p10,
                minutes_p50=minutes_p50,
                minutes_p90=minutes_p90,
                three_pa_mean=three_pa_mean,
                three_pa_dispersion=three_pa_dispersion,
                make_prob_mean=make_prob_mean,
                make_prob_uncertainty=make_prob_uncertainty,
                line=line,
            )
            sim_p_over = sim_result.p_over
            sim_p_under = sim_result.p_under
        except Exception as exc:
            logger.warning("Simulation failed: %s", exc)
            sim_p_over = 0.5
            sim_p_under = 0.5
            sim_result = None

        # --- Decision engine evaluation -----------------------------------
        odds_over = game.get("odds_over", -110)
        odds_under = game.get("odds_under", -110)
        novig_over = game.get("novig_over", 0.5)
        novig_under = game.get("novig_under", 0.5)
        spread = game.get("spread", 0.0)

        edge = 0.0
        ev = 0.0
        recommended_side = "no_bet"

        if sim_result is not None:
            try:
                odds_prop = {
                    "over_price": odds_over,
                    "under_price": odds_under,
                    "sportsbook": game.get("sportsbook", "unknown"),
                    "snapshot_timestamp_utc": game.get("odds_snapshot_timestamp"),
                    "line": line,
                    "nba_game_id": game_id,
                    "nba_player_id": player_id,
                }
                game_context = game.get("game_context", {})
                if "spread" not in game_context:
                    game_context["spread"] = spread

                feature_snapshot_for_decision = {
                    "feature_snapshot_id": snapshot.get("meta_feature_json_hash", ""),
                    "model_run_id": f"wf_fold_{fold_idx}",
                    "tracking_available": snapshot.get("meta_tracking_available", False),
                    "fallback_confidence": game.get("fallback_confidence", 1.0),
                    "player_archetype": snapshot.get("archetype", ""),
                    "team_change_days": game.get("team_change_days"),
                    "minutes_p10": minutes_p10,
                    "minutes_p90": minutes_p90,
                }

                decision = self.decision_engine.evaluate_opportunity(
                    simulation_result=sim_result,
                    odds_prop=odds_prop,
                    game_context=game_context,
                    feature_snapshot=feature_snapshot_for_decision,
                )
                edge = decision.get("edge_over", 0.0) if decision.get("recommended_side") == "over" else decision.get("edge_under", 0.0)
                ev = decision.get("_ev_over", 0.0) if decision.get("recommended_side") == "over" else decision.get("_ev_under", 0.0)
                recommended_side = decision.get("recommended_side", "no_bet")
            except Exception as exc:
                logger.warning("Decision engine failed: %s", exc)

        # --- Actuals and outcome ------------------------------------------
        actual_minutes = game.get("actual_minutes")
        actual_3pa = game.get("actual_3pa")
        actual_3pm = game.get("actual_3pm")
        actual_over = bool(actual_3pm > line) if actual_3pm is not None else None

        # Bet result and PnL
        bet_result = None
        pnl_units = 0.0
        if recommended_side != "no_bet" and actual_over is not None:
            if recommended_side == "over":
                bet_result = "win" if actual_over else "loss"
            elif recommended_side == "under":
                bet_result = "win" if not actual_over else "loss"

            if bet_result == "win":
                bet_odds = odds_over if recommended_side == "over" else odds_under
                decimal_odds = self._american_to_decimal(float(bet_odds))
                pnl_units = decimal_odds - 1.0
            elif bet_result == "loss":
                pnl_units = -1.0

        # CLV (closing line value) in probability points
        clv_prob_pts = 0.0
        if novig_over is not None and sim_p_over is not None:
            if recommended_side == "over":
                clv_prob_pts = sim_p_over - float(novig_over)
            elif recommended_side == "under":
                clv_prob_pts = sim_p_under - float(novig_under)

        # --- Construct the prediction row ---------------------------------
        archetype = snapshot.get("archetype", game.get("archetype", "unknown"))
        tracking_available = snapshot.get(
            "meta_tracking_available", game.get("tracking_available", False)
        )
        is_home = game.get("is_home", False)
        rest_days = game.get("rest_days", 1)
        is_b2b = game.get("is_b2b", rest_days == 0 if rest_days is not None else False)
        time_bucket = game.get("time_bucket", "T-60")

        prediction_row: Dict[str, Any] = {
            # Identifiers
            "game_id": game_id,
            "player_id": player_id,
            "game_date": game_date,
            "fold_index": fold_idx,
            # Context
            "archetype": archetype,
            "line": line,
            "line_bucket": "",  # assigned by _assign_buckets
            "tracking_available": tracking_available,
            "is_home": is_home,
            "rest_days": rest_days,
            "is_b2b": is_b2b,
            "spread": spread if spread is not None else 0.0,
            "spread_bucket": "",  # assigned by _assign_buckets
            "time_bucket": time_bucket,
            # Minutes predictions
            "predicted_minutes_p10": minutes_p10,
            "predicted_minutes_p50": minutes_p50,
            "predicted_minutes_p90": minutes_p90,
            # 3PA predictions
            "predicted_3pa_mean": three_pa_mean,
            "predicted_3pa_dispersion": three_pa_dispersion,
            # Make-rate predictions
            "predicted_make_prob_mean": make_prob_mean,
            "predicted_make_prob_uncertainty": make_prob_uncertainty,
            # Simulation outputs
            "sim_p_over": sim_p_over,
            "sim_p_under": sim_p_under,
            # Baseline placeholders (filled by _score_baselines)
            "baseline_rolling_avg_p_over": None,
            "baseline_direct_p_over": None,
            "baseline_book_p_over": None,
            # Actuals
            "actual_minutes": actual_minutes,
            "actual_3pa": actual_3pa,
            "actual_3pm": actual_3pm,
            "actual_over": actual_over,
            # Odds
            "odds_over": odds_over,
            "odds_under": odds_under,
            "novig_over": novig_over,
            "novig_under": novig_under,
            # Decision
            "edge": edge,
            "ev": ev,
            "recommended_side": recommended_side,
            "bet_result": bet_result,
            "pnl_units": pnl_units,
            "clv_prob_pts": clv_prob_pts,
        }

        return prediction_row

    # ------------------------------------------------------------------ #
    # Baseline scoring
    # ------------------------------------------------------------------ #

    def _score_baselines(
        self,
        predictions: List[Dict[str, Any]],
        test_games: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score all baselines on the test predictions.

        Parameters
        ----------
        predictions : list[dict]
            Prediction rows from the production model.
        test_games : list[dict]
            Raw test game data (needed for baseline inputs).

        Returns
        -------
        list[dict]
            Updated prediction rows with baseline scores filled in.
        """
        # Build a lookup for test games by (game_id, player_id)
        game_lookup: Dict[tuple, Dict[str, Any]] = {}
        for g in test_games:
            key = (g.get("game_id"), g.get("player_id"))
            game_lookup[key] = g

        rolling_avg = self.baselines.get("rolling_avg")
        direct_3pm = self.baselines.get("direct_3pm")
        bookmaker = self.baselines.get("bookmaker")

        for pred in predictions:
            key = (pred["game_id"], pred["player_id"])
            game = game_lookup.get(key, {})
            line = pred["line"]

            # --- Rolling average baseline ---------------------------------
            if rolling_avg is not None:
                try:
                    player_3pm_history = game.get("player_3pm_history", [])
                    pred["baseline_rolling_avg_p_over"] = rolling_avg.predict_p_over(
                        player_games=player_3pm_history, line=line
                    )
                except Exception:
                    pred["baseline_rolling_avg_p_over"] = 0.5

            # --- Direct 3PM baseline --------------------------------------
            if direct_3pm is not None and hasattr(direct_3pm, "is_fitted") and direct_3pm.is_fitted:
                try:
                    features = game.get(
                        "direct_3pm_features",
                        game.get("three_pa_features"),
                    )
                    if features is not None:
                        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
                        predicted_3pm = float(direct_3pm.predict(X)[0])
                        # Use a Poisson CDF approximation for p_over
                        from scipy.stats import poisson
                        pred["baseline_direct_p_over"] = float(
                            1.0 - poisson.cdf(int(line), mu=max(predicted_3pm, 0.01))
                        )
                    else:
                        pred["baseline_direct_p_over"] = 0.5
                except Exception:
                    pred["baseline_direct_p_over"] = 0.5

            # --- Bookmaker baseline ---------------------------------------
            if bookmaker is not None:
                try:
                    novig_over = pred.get("novig_over", 0.5)
                    pred["baseline_book_p_over"] = bookmaker.predict_p_over(
                        closing_novig_over_prob=float(novig_over) if novig_over is not None else 0.5
                    )
                except Exception:
                    pred["baseline_book_p_over"] = 0.5

        return predictions

    # ------------------------------------------------------------------ #
    # Metrics computation
    # ------------------------------------------------------------------ #

    def _compute_production_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute aggregated metrics for the production model.

        Parameters
        ----------
        predictions : list[dict]
            All prediction rows.

        Returns
        -------
        dict
            Aggregated production metrics including minutes, 3PA, 3PM,
            and betting metrics.
        """
        from ..backtest.metrics import BacktestMetrics

        if not predictions:
            return {}

        metrics: Dict[str, Any] = {}

        # Filter to rows with valid actuals
        valid = [p for p in predictions if p.get("actual_3pm") is not None]
        if not valid:
            return metrics

        # --- Minutes metrics ----------------------------------------------
        min_valid = [p for p in valid if p.get("actual_minutes") is not None]
        if min_valid:
            actual_min = np.array([p["actual_minutes"] for p in min_valid], dtype=np.float64)
            pred_p50 = np.array([p["predicted_minutes_p50"] for p in min_valid], dtype=np.float64)
            pred_p10 = np.array([p["predicted_minutes_p10"] for p in min_valid], dtype=np.float64)
            pred_p90 = np.array([p["predicted_minutes_p90"] for p in min_valid], dtype=np.float64)

            min_metrics = BacktestMetrics.compute_minutes_metrics(
                actual=actual_min,
                predicted_p50=pred_p50,
                predicted_p10=pred_p10,
                predicted_p90=pred_p90,
            )
            metrics["minutes"] = {
                "mae_overall": min_metrics.mae_overall,
                "mae_starters": min_metrics.mae_starters,
                "mae_rotation": min_metrics.mae_rotation,
                "interval_coverage_80": min_metrics.interval_coverage_80,
                "n": len(min_valid),
            }

        # --- 3PA metrics --------------------------------------------------
        tpa_valid = [p for p in valid if p.get("actual_3pa") is not None]
        if tpa_valid:
            actual_3pa = np.array([p["actual_3pa"] for p in tpa_valid], dtype=np.float64)
            pred_3pa = np.array([p["predicted_3pa_mean"] for p in tpa_valid], dtype=np.float64)

            tpa_metrics = BacktestMetrics.compute_3pa_metrics(
                actual=actual_3pa, predicted=pred_3pa
            )
            metrics["three_pa"] = {
                "mae": tpa_metrics.mae,
                "rmse": tpa_metrics.rmse,
                "count_calibration": tpa_metrics.count_calibration,
                "n": len(tpa_valid),
            }

        # --- 3PM (over/under) metrics ------------------------------------
        pm_valid = [p for p in valid if p.get("actual_over") is not None and p.get("sim_p_over") is not None]
        if pm_valid:
            actual_over = np.array([float(p["actual_over"]) for p in pm_valid], dtype=np.float64)
            p_over = np.array([p["sim_p_over"] for p in pm_valid], dtype=np.float64)

            pm_metrics = BacktestMetrics.compute_3pm_metrics(
                actual_over=actual_over, p_over=p_over
            )
            metrics["three_pm"] = {
                "log_loss": pm_metrics.log_loss,
                "brier_score": pm_metrics.brier_score,
                "sharpness": pm_metrics.sharpness,
                "n": len(pm_valid),
            }

        # --- Betting metrics ----------------------------------------------
        bet_preds = [p for p in valid if p.get("recommended_side") not in (None, "no_bet")]
        if bet_preds:
            edges = np.array([p["edge"] for p in bet_preds], dtype=np.float64)
            results = np.array(
                [1.0 if p["bet_result"] == "win" else (-1.0 if p["bet_result"] == "loss" else 0.0) for p in bet_preds],
                dtype=np.float64,
            )
            stakes = np.ones(len(bet_preds), dtype=np.float64)

            bet_odds = []
            for p in bet_preds:
                if p["recommended_side"] == "over":
                    bet_odds.append(self._american_to_decimal(float(p["odds_over"])))
                else:
                    bet_odds.append(self._american_to_decimal(float(p["odds_under"])))
            odds_decimal = np.array(bet_odds, dtype=np.float64)

            clv_pts = np.array([p["clv_prob_pts"] for p in bet_preds], dtype=np.float64)

            bet_metrics = BacktestMetrics.compute_betting_metrics(
                edges=edges,
                results=results,
                stakes=stakes,
                odds_decimal=odds_decimal,
                clv_pts=clv_pts,
            )
            metrics["betting"] = {
                "n_bets": bet_metrics.n_bets,
                "hit_rate": bet_metrics.hit_rate,
                "avg_edge": bet_metrics.avg_edge,
                "clv_mean": bet_metrics.clv_mean,
                "roi": bet_metrics.roi,
                "max_drawdown": bet_metrics.max_drawdown,
                "turnover": bet_metrics.turnover,
            }

        return metrics

    def _compute_baseline_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for each baseline.

        Parameters
        ----------
        predictions : list[dict]
            All prediction rows with baseline scores filled in.

        Returns
        -------
        dict
            ``{baseline_name: {metric_name: value}}``.
        """
        from ..backtest.metrics import BacktestMetrics

        baseline_metrics: Dict[str, Dict[str, Any]] = {}

        valid = [
            p for p in predictions
            if p.get("actual_over") is not None
        ]
        if not valid:
            return baseline_metrics

        actual_over = np.array(
            [float(p["actual_over"]) for p in valid], dtype=np.float64
        )

        baseline_keys = {
            "rolling_avg": "baseline_rolling_avg_p_over",
            "direct_3pm": "baseline_direct_p_over",
            "bookmaker": "baseline_book_p_over",
        }

        for baseline_name, col_name in baseline_keys.items():
            bl_valid = [
                (p, i)
                for i, p in enumerate(valid)
                if p.get(col_name) is not None
            ]
            if not bl_valid:
                continue

            bl_preds = np.array(
                [p[col_name] for p, _ in bl_valid], dtype=np.float64
            )
            bl_actuals = np.array(
                [actual_over[i] for _, i in bl_valid], dtype=np.float64
            )

            pm_metrics = BacktestMetrics.compute_3pm_metrics(
                actual_over=bl_actuals, p_over=bl_preds
            )
            baseline_metrics[baseline_name] = {
                "log_loss": pm_metrics.log_loss,
                "brier_score": pm_metrics.brier_score,
                "sharpness": pm_metrics.sharpness,
                "n": len(bl_valid),
            }

        return baseline_metrics

    def _compute_sliced_metrics(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics sliced by all standard dimensions.

        Parameters
        ----------
        predictions : list[dict]
            All prediction rows.

        Returns
        -------
        dict
            ``{dimension: {slice_value: metrics_dict}}``.
        """
        dimensions = [
            "line_bucket",
            "spread_bucket",
            "rest_bucket",
            "archetype",
            "time_bucket",
            "tracking_available",
            "is_home",
            "is_b2b",
        ]

        sliced: Dict[str, Dict[str, Any]] = {}

        for dim in dimensions:
            dim_slices: Dict[str, Any] = {}

            # Gather unique values for this dimension
            unique_values = set()
            for p in predictions:
                val = p.get(dim)
                if val is not None:
                    unique_values.add(val)

            for val in sorted(unique_values, key=str):
                slice_preds = [p for p in predictions if p.get(dim) == val]
                if slice_preds:
                    dim_slices[str(val)] = self._compute_production_metrics(
                        slice_preds
                    )

            if dim_slices:
                sliced[dim] = dim_slices

        return sliced

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _american_to_decimal(american: float) -> float:
        """Convert American odds to decimal odds.

        Parameters
        ----------
        american : float
            American odds (e.g. -110, +150).

        Returns
        -------
        float
            Decimal odds.
        """
        if american > 0:
            return 1.0 + american / 100.0
        elif american < 0:
            return 1.0 + 100.0 / abs(american)
        else:
            return 1.0
