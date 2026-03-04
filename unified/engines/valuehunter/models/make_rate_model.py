"""3-Point Make Rate model (Section I3).

Models the probability that each three-point attempt goes in, i.e.
3P% = 3PM / 3PA.  This is a *proportion* bounded in (0, 1), which
makes it fundamentally different from the count-based 3PA model.

Three estimation methods are supported:

  - **beta_regression** (default): Fits a GBM on the logit-transformed
    make rate.  This is a practical approximation to true Beta regression
    that works well with tree-based learners.
  - **logit_gbm**: Fits a GBM directly on logit(3PM/3PA) with sample
    weights proportional to attempt volume (more attempts = more signal).
  - **empirical_bayes_only**: Skips model fitting entirely and returns
    shrinkage estimates that blend the player's observed rate with
    archetype and league priors.

All methods apply Bayesian shrinkage so that small-sample players are
pulled toward reasonable priors rather than exhibiting extreme
predicted rates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)

# Optional heavy dependencies ------------------------------------------------
try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor

# --------------------------------------------------------------------------- #
# Shrinkage priors (Section I3)
# --------------------------------------------------------------------------- #

LEAGUE_3PT_PRIOR: float = 0.363
CATCH_SHOOT_PRIOR: float = 0.374
PULL_UP_PRIOR: float = 0.329
PRIOR_WEIGHT: int = 100  # equivalent sample-size for the prior


_VALID_METHODS = {"beta_regression", "logit_gbm", "empirical_bayes_only"}


class MakeRateModel(BaseModel):
    """Predicts 3-point make probability (3PM / 3PA).

    The model's output feeds into the downstream Monte-Carlo simulation
    as the success probability in a Binomial draw conditioned on the
    predicted 3PA count.
    """

    def __init__(
        self,
        version: str = "1.0",
        method: str = "beta_regression",
    ):
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got '{method}'"
            )
        super().__init__(name="make_rate_model", version=version)
        self.method = method

        # Fitted objects
        self._model: Any = None
        self._shrinkage_prior_rate: float = LEAGUE_3PT_PRIOR
        self._shrinkage_prior_weight: int = PRIOR_WEIGHT

        # Book-keeping
        self.feature_names: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        y_makes: np.ndarray,
        y_attempts: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Fit the make-rate model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(n_samples, n_features)``.
        y_makes : np.ndarray
            Observed 3PM counts ``(n_samples,)``.  When using
            ``empirical_bayes_only`` this may also be the raw proportion
            (3PM / 3PA) directly.
        y_attempts : np.ndarray, optional
            Observed 3PA counts ``(n_samples,)``.  Required for
            ``beta_regression`` and ``logit_gbm`` methods.
        feature_names : list[str], optional
            Human-readable feature names.
        """
        X = np.asarray(X, dtype=np.float64)
        y_makes = np.asarray(y_makes, dtype=np.float64)

        self.feature_names = feature_names

        if self.method == "empirical_bayes_only":
            self._fit_empirical_bayes(y_makes, y_attempts)
        else:
            if y_attempts is None:
                raise ValueError(
                    f"y_attempts is required for method='{self.method}'"
                )
            y_attempts = np.asarray(y_attempts, dtype=np.float64)

            if self.method == "beta_regression":
                self._fit_beta_regression(X, y_makes, y_attempts)
            elif self.method == "logit_gbm":
                self._fit_logit_gbm(X, y_makes, y_attempts)

        # Metadata
        self.metadata = ModelMetadata(
            model_name=self.name,
            model_version=self.version,
            hyperparams={
                "method": self.method,
                "league_prior": LEAGUE_3PT_PRIOR,
                "prior_weight": PRIOR_WEIGHT,
            },
        )

        self.is_fitted = True
        self.logger.info("MakeRateModel fit complete (method=%s)", self.method)

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted make probability (mean).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted 3P% for each sample, clipped to (0.05, 0.60).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        if self.method == "empirical_bayes_only":
            # No feature-based model; return the global shrinkage rate
            return np.full(X.shape[0], self._shrinkage_prior_rate)

        logit_preds = self._model.predict(X)
        probs = self._sigmoid(logit_preds)

        # Clip to a sensible range
        return np.clip(probs, 0.05, 0.60)

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return predicted make probability with uncertainty band.

        The uncertainty is derived from the Beta posterior:
        ``Beta(alpha + makes, beta + attempts - makes)``.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        dict
            ``{"mean": ndarray, "uncertainty": ndarray}``
            where *uncertainty* is the posterior standard deviation.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        mean_preds = self.predict(X)

        # Uncertainty from Beta posterior with the prior as a pseudo-count
        alpha_param = mean_preds * self._shrinkage_prior_weight
        beta_param = (1 - mean_preds) * self._shrinkage_prior_weight
        # Var(Beta) = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        total = alpha_param + beta_param
        variance = (alpha_param * beta_param) / (total ** 2 * (total + 1))
        uncertainty = np.sqrt(variance)

        return {"mean": mean_preds, "uncertainty": uncertainty}

    # ------------------------------------------------------------------ #
    # Shrinkage
    # ------------------------------------------------------------------ #

    @staticmethod
    def _shrink_to_prior(
        player_makes: float,
        player_attempts: float,
        prior_rate: float,
        prior_weight: float,
    ) -> float:
        """Compute shrinkage (empirical-Bayes) estimate of make rate.

        Blends the player's observed rate with a prior:

            shrunk_rate = (player_makes + prior_weight * prior_rate)
                        / (player_attempts + prior_weight)

        This is equivalent to a Beta-Binomial conjugate update where the
        prior has ``alpha = prior_weight * prior_rate`` and
        ``beta = prior_weight * (1 - prior_rate)`` pseudo-counts.

        Parameters
        ----------
        player_makes : float
            Total 3PM for the player.
        player_attempts : float
            Total 3PA for the player.
        prior_rate : float
            The prior three-point percentage (e.g., league average 0.363).
        prior_weight : float
            Equivalent sample size for the prior (higher = more
            regularisation toward the prior).

        Returns
        -------
        float
            Shrinkage-adjusted make rate.
        """
        return (player_makes + prior_weight * prior_rate) / (
            player_attempts + prior_weight
        )

    @staticmethod
    def shrink_with_archetype(
        player_makes: float,
        player_attempts: float,
        is_catch_and_shoot_dominant: bool = True,
    ) -> float:
        """Apply hierarchical shrinkage using archetype-specific priors.

        Two-level shrinkage:
          1. Shrink toward the archetype prior (catch-and-shoot vs pull-up).
          2. Shrink the archetype prior toward the league prior.

        Parameters
        ----------
        player_makes : float
            Total 3PM.
        player_attempts : float
            Total 3PA.
        is_catch_and_shoot_dominant : bool
            If True, use the catch-and-shoot prior; otherwise pull-up.

        Returns
        -------
        float
            Hierarchically shrunk make rate.
        """
        archetype_prior = (
            CATCH_SHOOT_PRIOR if is_catch_and_shoot_dominant else PULL_UP_PRIOR
        )
        # First level: shrink archetype prior toward league
        blended_prior = 0.7 * archetype_prior + 0.3 * LEAGUE_3PT_PRIOR

        return MakeRateModel._shrink_to_prior(
            player_makes, player_attempts, blended_prior, PRIOR_WEIGHT
        )

    # ------------------------------------------------------------------ #
    # Private: fitting helpers
    # ------------------------------------------------------------------ #

    def _fit_beta_regression(
        self,
        X: np.ndarray,
        y_makes: np.ndarray,
        y_attempts: np.ndarray,
    ) -> None:
        """Fit GBM on logit-transformed make rate (practical Beta regression).

        Observations with zero attempts are dropped.  The logit transform
        maps (0, 1) to (-inf, +inf) so that standard regression losses work.
        """
        self.logger.info("Fitting beta-regression (logit-GBM) model")

        # Filter zero-attempt observations
        mask = y_attempts > 0
        X_filt = X[mask]
        makes_filt = y_makes[mask]
        attempts_filt = y_attempts[mask]

        # Compute proportions with shrinkage to avoid logit(0) or logit(1)
        proportions = np.array([
            self._shrink_to_prior(m, a, LEAGUE_3PT_PRIOR, PRIOR_WEIGHT)
            for m, a in zip(makes_filt, attempts_filt)
        ])

        # Logit transform
        logit_y = self._logit(proportions)

        # Sample weights proportional to attempts (more attempts = more info)
        sample_weights = attempts_filt / np.mean(attempts_filt)

        self._model = self._build_gbm()
        self._model.fit(X_filt, logit_y, sample_weight=sample_weights)

        self.feature_importances_ = self._extract_feature_importance(self._model)

    def _fit_logit_gbm(
        self,
        X: np.ndarray,
        y_makes: np.ndarray,
        y_attempts: np.ndarray,
    ) -> None:
        """Fit GBM directly on logit(3PM/3PA) with attempt-weighted samples."""
        self.logger.info("Fitting logit-GBM model")

        mask = y_attempts > 0
        X_filt = X[mask]
        makes_filt = y_makes[mask]
        attempts_filt = y_attempts[mask]

        # Raw proportions clipped away from boundaries
        raw_rate = np.clip(makes_filt / attempts_filt, 0.01, 0.99)
        logit_y = self._logit(raw_rate)

        sample_weights = attempts_filt / np.mean(attempts_filt)

        self._model = self._build_gbm()
        self._model.fit(X_filt, logit_y, sample_weight=sample_weights)

        self.feature_importances_ = self._extract_feature_importance(self._model)

    def _fit_empirical_bayes(
        self,
        y_makes: np.ndarray,
        y_attempts: Optional[np.ndarray],
    ) -> None:
        """Compute global shrinkage estimate only (no feature-based model)."""
        self.logger.info("Fitting empirical-Bayes-only model (no features)")

        if y_attempts is not None:
            y_attempts = np.asarray(y_attempts, dtype=np.float64)
            total_makes = float(np.sum(y_makes))
            total_attempts = float(np.sum(y_attempts))
            self._shrinkage_prior_rate = self._shrink_to_prior(
                total_makes, total_attempts, LEAGUE_3PT_PRIOR, PRIOR_WEIGHT
            )
        else:
            # y_makes is already the proportion
            self._shrinkage_prior_rate = float(np.mean(y_makes))

        self.logger.info(
            "Empirical Bayes global rate: %.4f", self._shrinkage_prior_rate
        )

    # ------------------------------------------------------------------ #
    # Private: utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_gbm() -> Any:
        """Build a GBM regressor for logit-scale prediction."""
        if _HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                objective="regression",
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=30,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            return GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=30,
                random_state=42,
            )

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        """Logit transform: log(p / (1-p)), with clipping for safety."""
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Inverse logit (sigmoid): 1 / (1 + exp(-x))."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    @staticmethod
    def _extract_feature_importance(model: Any) -> Optional[np.ndarray]:
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=np.float64)
        return None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit() first."
            )
