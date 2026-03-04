"""Benchmark / baseline models for NBA 3PM props (Section I5).

These models serve as comparison benchmarks to ensure the full
decomposition pipeline (minutes -> 3PA -> make rate) actually
adds value over simpler approaches.

Three baselines are provided:

1. **RollingAverageBaseline** - Simple rolling average of raw 3PM.
   The most naive approach; any production model must beat this.

2. **DirectThreePMBaseline** - Single GBM that directly predicts 3PM
   from game-level features.  The spec explicitly warns against using
   this in production because it conflates minutes, shot selection, and
   shooting skill.  However it is a useful upper-bound baseline.

3. **BookmakerBaseline** - Uses the sportsbook's no-vig implied
   probability as the prediction.  This is the "market efficiency"
   benchmark: if the model cannot beat the book, there is no edge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Optional heavy dependencies ------------------------------------------------
try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------------------------------- #
# 1. Rolling Average Baseline
# --------------------------------------------------------------------------- #


class RollingAverageBaseline:
    """Predict 3PM using a simple rolling average of recent games.

    This is the simplest possible baseline.  It uses no contextual
    features and is purely backward-looking.

    Usage
    -----
    >>> baseline = RollingAverageBaseline()
    >>> pred = baseline.predict_3pm(player_games=[2, 3, 1, 4, 0], window=5)
    >>> p_over = baseline.predict_p_over(player_games=[2, 3, 1, 4, 0], line=2.5)
    """

    def __init__(self) -> None:
        self.name = "rolling_average_baseline"
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def predict_3pm(
        self,
        player_games: List[float] | np.ndarray,
        window: int = 10,
    ) -> float:
        """Predict 3PM as the rolling average of recent games.

        Parameters
        ----------
        player_games : list or ndarray
            Chronologically ordered list of 3PM values per game, where
            the last element is the most recent game.
        window : int
            Number of recent games to average over.

        Returns
        -------
        float
            Predicted 3PM (rolling average).
        """
        games = np.asarray(player_games, dtype=np.float64)
        if len(games) == 0:
            self.logger.warning("No game data provided; returning 0.0")
            return 0.0

        recent = games[-window:]
        return float(np.mean(recent))

    def predict_p_over(
        self,
        player_games: List[float] | np.ndarray,
        line: float,
        window: int = 10,
    ) -> float:
        """Predict the probability of going OVER a given 3PM line.

        Uses the empirical fraction of recent games where the player
        exceeded the line.

        Parameters
        ----------
        player_games : list or ndarray
            Chronologically ordered 3PM values per game.
        line : float
            The sportsbook line (e.g., 2.5).
        window : int
            Number of recent games to consider.

        Returns
        -------
        float
            Empirical over-probability in [0, 1].
        """
        games = np.asarray(player_games, dtype=np.float64)
        if len(games) == 0:
            return 0.5  # uninformative prior

        recent = games[-window:]
        over_count = np.sum(recent > line)
        return float(over_count / len(recent))

    def evaluate(
        self,
        all_player_games: Dict[str, List[float]],
        all_actuals: Dict[str, float],
        window: int = 10,
    ) -> Dict[str, float]:
        """Evaluate MAE across multiple players.

        Parameters
        ----------
        all_player_games : dict
            Mapping of ``player_id -> list_of_3pm_values`` (historical).
        all_actuals : dict
            Mapping of ``player_id -> actual_3pm`` for the target game.
        window : int
            Rolling window size.

        Returns
        -------
        dict
            ``{"mae": float, "n_players": int}``
        """
        errors = []
        for pid, actual in all_actuals.items():
            if pid not in all_player_games:
                continue
            pred = self.predict_3pm(all_player_games[pid], window=window)
            errors.append(abs(actual - pred))

        if not errors:
            return {"mae": float("nan"), "n_players": 0}

        return {
            "mae": float(np.mean(errors)),
            "n_players": len(errors),
        }


# --------------------------------------------------------------------------- #
# 2. Direct 3PM Baseline (GBM)
# --------------------------------------------------------------------------- #


class DirectThreePMBaseline:
    """Single GBM that directly predicts 3PM from game-level features.

    **WARNING**: The spec (Section I5) explicitly states this should NOT
    be used as the production model because it conflates minutes
    uncertainty, shot selection, and shooting accuracy.  However it is a
    valuable benchmark -- if the decomposition pipeline cannot beat a
    single model, the decomposition is not adding value.
    """

    def __init__(self) -> None:
        self.name = "direct_3pm_baseline"
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._model: Any = None
        self.is_fitted: bool = False
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit a single GBM to predict 3PM directly.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(n_samples, n_features)``.
        y : np.ndarray
            Observed 3PM counts ``(n_samples,)``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.logger.info(
            "Fitting DirectThreePMBaseline (%d samples, %d features)", *X.shape
        )

        if _HAS_LIGHTGBM:
            self._model = lgb.LGBMRegressor(
                objective="poisson",
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            self._model = GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )

        self._model.fit(X, y)
        self.is_fitted = True

        self.feature_importances_ = (
            np.asarray(self._model.feature_importances_, dtype=np.float64)
            if hasattr(self._model, "feature_importances_")
            else None
        )

        train_preds = self._model.predict(X)
        train_mae = float(mean_absolute_error(y, train_preds))
        train_rmse = float(np.sqrt(mean_squared_error(y, train_preds)))
        self.logger.info(
            "DirectThreePMBaseline fit complete. MAE=%.3f, RMSE=%.3f",
            train_mae,
            train_rmse,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 3PM.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted 3PM counts.
        """
        if not self.is_fitted:
            raise RuntimeError("DirectThreePMBaseline has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        preds = self._model.predict(X)
        return np.maximum(preds, 0.0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on held-out data.

        Returns
        -------
        dict
            ``{"mae": float, "rmse": float}``
        """
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        return {
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        }


# --------------------------------------------------------------------------- #
# 3. Bookmaker Baseline
# --------------------------------------------------------------------------- #


class BookmakerBaseline:
    """Use the sportsbook's no-vig implied probability as the prediction.

    This is the market-efficiency benchmark.  If the model's predicted
    probabilities cannot beat the book's no-vig probabilities (measured
    by log-loss or Brier score), there is no exploitable edge.

    Usage
    -----
    >>> bm = BookmakerBaseline()
    >>> p_over = bm.predict_p_over(closing_novig_over_prob=0.55)
    """

    def __init__(self) -> None:
        self.name = "bookmaker_baseline"
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def predict_p_over(self, closing_novig_over_prob: float) -> float:
        """Return the book's no-vig over probability directly.

        Parameters
        ----------
        closing_novig_over_prob : float
            The sportsbook's closing no-vig implied probability for the
            OVER side of the prop.

        Returns
        -------
        float
            The same probability (identity function).
        """
        return float(np.clip(closing_novig_over_prob, 0.0, 1.0))

    def predict_p_over_batch(
        self,
        closing_novig_over_probs: np.ndarray,
    ) -> np.ndarray:
        """Batch prediction - return the book's no-vig probabilities.

        Parameters
        ----------
        closing_novig_over_probs : np.ndarray
            Array of no-vig over probabilities.

        Returns
        -------
        np.ndarray
            Clipped probabilities.
        """
        probs = np.asarray(closing_novig_over_probs, dtype=np.float64)
        return np.clip(probs, 0.0, 1.0)

    def evaluate(
        self,
        closing_novig_over_probs: np.ndarray,
        actuals_over: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the bookmaker baseline via Brier score and log-loss.

        Parameters
        ----------
        closing_novig_over_probs : np.ndarray
            No-vig over probabilities from the book.
        actuals_over : np.ndarray
            Binary outcomes: 1 if the player went over, 0 otherwise.

        Returns
        -------
        dict
            ``{"brier_score": float, "log_loss": float, "accuracy": float}``
        """
        probs = np.asarray(closing_novig_over_probs, dtype=np.float64)
        actuals = np.asarray(actuals_over, dtype=np.float64)

        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        # Brier score: mean squared error of probability predictions
        brier = float(np.mean((probs - actuals) ** 2))

        # Log loss: negative mean log-likelihood
        log_loss = -float(
            np.mean(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))
        )

        # Classification accuracy at 0.5 threshold
        predicted_class = (probs >= 0.5).astype(float)
        accuracy = float(np.mean(predicted_class == actuals))

        results = {
            "brier_score": brier,
            "log_loss": log_loss,
            "accuracy": accuracy,
        }

        self.logger.info(
            "BookmakerBaseline eval: Brier=%.4f, LogLoss=%.4f, Acc=%.3f",
            brier,
            log_loss,
            accuracy,
        )
        return results
