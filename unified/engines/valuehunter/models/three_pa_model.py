"""3-Point Attempt (3PA) opportunity model (Section I2).

Models the count of three-point attempts a player takes in a game,
conditional on predicted minutes played and contextual features.

Three distribution families are supported:
  - **negative_binomial** (default): NegBin GLM with log link and minutes
    exposure.  The dispersion parameter captures player-level over-
    dispersion that plain Poisson misses.
  - **poisson**: Poisson GLM with log link (a restricted special case).
  - **gbm_count**: Gradient-boosted trees with Poisson / Tweedie loss.

When ``statsmodels`` is not installed the code falls back to
``sklearn.linear_model.PoissonRegressor``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_MIN_EXPOSURE = 1e-3

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)

# Optional heavy dependencies ------------------------------------------------
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial as NegBinFamily
    from statsmodels.genmod.families import Poisson as PoissonFamily
    from statsmodels.genmod.families.links import Log as LogLink

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor


# Supported distribution families
_VALID_DISTRIBUTIONS = {"negative_binomial", "poisson", "gbm_count"}


class ThreePAModel(BaseModel):
    """Count model for three-point attempts.

    The key insight is that 3PA is a *count* variable with significant
    over-dispersion across players and games.  The Negative Binomial
    distribution explicitly models this via a dispersion parameter
    (``alpha``), which is critical for realistic Monte-Carlo simulation
    downstream.
    """

    def __init__(
        self,
        version: str = "1.0",
        distribution: str = "negative_binomial",
    ):
        if distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {_VALID_DISTRIBUTIONS}, "
                f"got '{distribution}'"
            )
        super().__init__(name="three_pa_model", version=version)
        self.distribution = distribution

        # Fitted model object (type varies by distribution)
        self._model: Any = None
        self._dispersion: Optional[float] = None  # NegBin alpha
        self._fallback_sklearn: bool = False

        # Book-keeping
        self.feature_names: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate core model inputs for numeric stability and shape consistency."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        if y is not None:
            if y.ndim != 1:
                raise ValueError("y must be a 1D array")
            if y.shape[0] != X.shape[0]:
                raise ValueError("y length must match number of rows in X")
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain only finite values")
            if np.any(y < 0):
                raise ValueError("y must contain non-negative counts")

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        minutes_exposure: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Fit the 3PA count model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(n_samples, n_features)``.
        y : np.ndarray
            Observed 3PA counts ``(n_samples,)``.
        minutes_exposure : np.ndarray, optional
            Predicted (or actual) minutes for each observation.  Used as an
            offset (``log(minutes)``) in GLM variants so that the model
            learns a *rate* that scales with playing time.
        feature_names : list[str], optional
            Human-readable feature labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._validate_inputs(X, y)

        self.feature_names = feature_names
        offset = self._prepare_offset(minutes_exposure, n_samples=X.shape[0])

        if self.distribution == "negative_binomial":
            self._fit_negative_binomial(X, y, offset)
        elif self.distribution == "poisson":
            self._fit_poisson(X, y, offset)
        elif self.distribution == "gbm_count":
            self._fit_gbm_count(X, y)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Mark as fitted before computing training metrics (predict checks this)
        self.is_fitted = True

        # Metrics on training set
        preds = self.predict(X, minutes_exposure=minutes_exposure)
        train_mae = float(mean_absolute_error(y, preds))
        train_rmse = float(np.sqrt(mean_squared_error(y, preds)))

        self.metadata = ModelMetadata(
            model_name=self.name,
            model_version=self.version,
            hyperparams={
                "distribution": self.distribution,
                "fallback_sklearn": self._fallback_sklearn,
                "dispersion": self._dispersion,
            },
            metrics={
                "train_mae": train_mae,
                "train_rmse": train_rmse,
            },
        )

        self.is_fitted = True
        self.logger.info(
            "ThreePAModel fit complete (dist=%s). MAE=%.3f, RMSE=%.3f",
            self.distribution,
            train_mae,
            train_rmse,
        )

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(
        self,
        X: np.ndarray,
        minutes_exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return expected 3PA count (mean prediction).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        minutes_exposure : np.ndarray, optional
            Minutes exposure for GLM offset.

        Returns
        -------
        np.ndarray
            Predicted mean 3PA for each sample.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        self._validate_inputs(X)
        offset = self._prepare_offset(minutes_exposure, n_samples=X.shape[0])

        if self.distribution in ("negative_binomial", "poisson"):
            return self._predict_glm(X, offset)
        else:  # gbm_count
            return self._predict_gbm(X)

    def predict_distribution_params(
        self,
        X: np.ndarray,
        minutes_exposure: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Return distribution parameters for Monte-Carlo simulation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        minutes_exposure : np.ndarray, optional
            Minutes exposure for GLM offset.

        Returns
        -------
        dict
            ``{"mean": ndarray, "dispersion": ndarray}``
            For Poisson, dispersion is broadcast to ones (variance == mean).
            For NegBin, dispersion is the estimated alpha.
            For GBM, dispersion is estimated from residuals.
        """
        self._check_fitted()
        mu = self.predict(X, minutes_exposure=minutes_exposure)

        if self.distribution == "negative_binomial" and self._dispersion is not None:
            dispersion = np.full_like(mu, self._dispersion)
        elif self.distribution == "poisson":
            dispersion = np.ones_like(mu)
        else:
            # GBM fallback: estimate dispersion from training residuals
            dispersion = np.full_like(mu, self._dispersion if self._dispersion else 1.0)

        return {"mean": mu, "dispersion": dispersion}

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        minutes_exposure: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model.

        Returns
        -------
        dict
            ``mae``, ``rmse``, ``count_calibration`` (mean predicted / mean
            actual).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._validate_inputs(X, y)

        preds = self.predict(X, minutes_exposure=minutes_exposure)

        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))

        mean_pred = float(np.mean(preds))
        mean_actual = float(np.mean(y))
        count_calibration = mean_pred / mean_actual if mean_actual > 0 else float("nan")

        results = {
            "mae": mae,
            "rmse": rmse,
            "count_calibration": count_calibration,
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
        }

        self.logger.info(
            "Evaluation: MAE=%.3f, RMSE=%.3f, calibration=%.3f",
            mae,
            rmse,
            count_calibration,
        )
        return results

    # ------------------------------------------------------------------ #
    # Private: fitting helpers
    # ------------------------------------------------------------------ #

    def _fit_negative_binomial(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: Optional[np.ndarray],
    ) -> None:
        """Fit a Negative Binomial GLM with log link."""
        if _HAS_STATSMODELS:
            self.logger.info("Fitting NegativeBinomial GLM via statsmodels")
            X_const = sm.add_constant(X)
            try:
                glm = sm.GLM(
                    y,
                    X_const,
                    family=NegBinFamily(link=LogLink(), alpha=1.0),
                    offset=offset,
                )
                result = glm.fit(maxiter=100, disp=False)
                self._model = result
                # Estimate dispersion (alpha) via auxiliary OLS
                self._dispersion = self._estimate_negbin_dispersion(y, result.mu)
                self._fallback_sklearn = False
                self.logger.info(
                    "NegBin GLM converged. Dispersion alpha=%.4f",
                    self._dispersion,
                )
                return
            except Exception as exc:
                self.logger.warning(
                    "statsmodels NegBin failed (%s), falling back to sklearn Poisson",
                    exc,
                )

        # Fallback: sklearn PoissonRegressor (no dispersion parameter)
        self.logger.info("Falling back to sklearn PoissonRegressor")
        self._fit_sklearn_poisson(X, y, offset)
        self._fallback_sklearn = True

    def _fit_poisson(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: Optional[np.ndarray],
    ) -> None:
        """Fit a Poisson GLM with log link."""
        if _HAS_STATSMODELS:
            self.logger.info("Fitting Poisson GLM via statsmodels")
            X_const = sm.add_constant(X)
            try:
                glm = sm.GLM(
                    y,
                    X_const,
                    family=PoissonFamily(link=LogLink()),
                    offset=offset,
                )
                result = glm.fit(maxiter=100, disp=False)
                self._model = result
                self._dispersion = 1.0  # Poisson: variance == mean
                self._fallback_sklearn = False
                return
            except Exception as exc:
                self.logger.warning(
                    "statsmodels Poisson failed (%s), falling back to sklearn",
                    exc,
                )

        self._fit_sklearn_poisson(X, y, offset)
        self._fallback_sklearn = True

    def _fit_sklearn_poisson(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: Optional[np.ndarray],
    ) -> None:
        """Fallback Poisson fit via sklearn."""
        # sklearn PoissonRegressor doesn't support offsets directly.
        # We approximate an offset-aware fit by modeling a rate target:
        #   y_rate = y / exposure, exposure = exp(offset)
        # and weighting by exposure so higher-minute samples contribute more.
        # Prediction then rescales rate back to counts.
        model = PoissonRegressor(alpha=0.01, max_iter=300)
        if offset is not None:
            exposure = np.exp(offset)
            y_rate = y / exposure
            model.fit(X, y_rate, sample_weight=exposure)
        else:
            model.fit(X, y)
        self._model = model
        train_mu = self._predict_sklearn_poisson(X, offset)
        self._dispersion = self._estimate_negbin_dispersion(y, train_mu)

    def _fit_gbm_count(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a GBM with Poisson / Tweedie loss for count prediction."""
        self.logger.info("Fitting GBM count model")
        if _HAS_LIGHTGBM:
            model = lgb.LGBMRegressor(
                objective="poisson",
                n_estimators=400,
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
            model = GradientBoostingRegressor(
                loss="squared_error",  # sklearn lacks native Poisson loss in older versions
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )

        model.fit(X, y)
        self._model = model
        self.feature_importances_ = (
            np.asarray(model.feature_importances_, dtype=np.float64)
            if hasattr(model, "feature_importances_")
            else None
        )

        preds = model.predict(X)
        self._dispersion = self._estimate_negbin_dispersion(y, preds)

    # ------------------------------------------------------------------ #
    # Private: prediction helpers
    # ------------------------------------------------------------------ #

    def _predict_glm(
        self,
        X: np.ndarray,
        offset: Optional[np.ndarray],
    ) -> np.ndarray:
        """Predict from statsmodels GLM or sklearn PoissonRegressor."""
        if self._fallback_sklearn:
            # sklearn PoissonRegressor
            return self._predict_sklearn_poisson(X, offset)
        else:
            # statsmodels GLM result
            X_const = sm.add_constant(X)
            if offset is not None:
                return self._model.predict(X_const, offset=offset)
            return self._model.predict(X_const)

    def _predict_sklearn_poisson(
        self,
        X: np.ndarray,
        offset: Optional[np.ndarray],
    ) -> np.ndarray:
        """Predict counts from sklearn Poisson model with optional offset rescaling."""
        rate_or_count = self._model.predict(X)
        if offset is None:
            return np.maximum(rate_or_count, 0.0)
        return np.maximum(rate_or_count * np.exp(offset), 0.0)

    def _predict_gbm(self, X: np.ndarray) -> np.ndarray:
        """Predict from GBM count model."""
        preds = self._model.predict(X)
        return np.maximum(preds, 0.0)

    # ------------------------------------------------------------------ #
    # Private: utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_offset(
        minutes_exposure: Optional[np.ndarray],
        n_samples: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Convert minutes exposure to a validated log-offset for GLM models."""
        if minutes_exposure is None:
            return None

        minutes_exposure = np.asarray(minutes_exposure, dtype=np.float64)
        if minutes_exposure.ndim != 1:
            raise ValueError("minutes_exposure must be a 1D array")
        if n_samples is not None and minutes_exposure.shape[0] != n_samples:
            raise ValueError(
                "minutes_exposure length must match number of rows in X"
            )
        if not np.all(np.isfinite(minutes_exposure)):
            raise ValueError("minutes_exposure must contain only finite values")
        if np.any(minutes_exposure < 0):
            raise ValueError("minutes_exposure cannot contain negative values")

        # Use a tiny floor for zero-minute rows to preserve monotonic scaling
        # while avoiding log(0) instability in offset-based GLMs.
        minutes_safe = np.clip(minutes_exposure, _MIN_EXPOSURE, None)
        return np.log(minutes_safe)

    @staticmethod
    def _estimate_negbin_dispersion(
        y: np.ndarray,
        mu: np.ndarray,
    ) -> float:
        """Estimate NegBin dispersion alpha using Pearson residual moments.

        For NegBin: Var(Y_i) = mu_i + alpha * mu_i^2.
        Solving per-row and averaging yields:
            alpha_i ~= ((y_i - mu_i)^2 - mu_i) / mu_i^2
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if y.shape != mu.shape:
            raise ValueError("y and mu must have the same shape")

        mu_safe = np.clip(mu, 1e-8, None)
        alpha_terms = ((y - mu_safe) ** 2 - mu_safe) / (mu_safe ** 2)
        alpha = float(np.mean(alpha_terms))

        if not np.isfinite(alpha):
            return 1.0
        return max(alpha, 0.01)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit() first."
            )
