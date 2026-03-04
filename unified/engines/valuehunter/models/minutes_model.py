"""Minutes prediction model (Section I1).

Predicts expected minutes played using a two-stage approach:
  1. Ridge regression baseline for stability
  2. GBM quantile model for distributional predictions (p10, p50, p90)

Features from Section H1 include rolling averages, team pace, rest days,
opponent defensive rating, home/away indicator, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)

# Try to import LightGBM; fall back to sklearn GradientBoosting
try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor


class MinutesModel(BaseModel):
    """Predicts player minutes with quantile uncertainty estimates.

    Two sub-models:
      - Ridge regression baseline (regularised linear model)
      - GBM quantile model producing p10 / p50 / p90

    The point prediction returned by ``predict()`` is the p50 from the
    quantile model.  ``predict_quantiles()`` returns the full set of
    quantile predictions for downstream Monte-Carlo sampling.
    """

    # Default feature set labels (Section H1).  Callers may override via
    # ``fit(..., feature_names=...)``.
    DEFAULT_FEATURE_NAMES: List[str] = [
        "minutes_rolling_10",
        "minutes_rolling_5",
        "minutes_season_avg",
        "team_pace",
        "rest_days",
        "opp_def_rating",
        "is_home",
        "is_starter",
        "game_spread",
        "game_total",
        "back_to_back",
        "minutes_std_10",
        "team_minutes_share",
        "blowout_risk",
    ]

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, version: str = "1.0"):
        super().__init__(name="minutes_model", version=version)

        # Sub-models
        self._ridge: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._quantile_models: Dict[str, Any] = {}  # keyed by "p10", "p50", "p90"

        # Book-keeping
        self.feature_names: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._role_mae: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Train Ridge baseline and GBM quantile models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target minutes array of shape ``(n_samples,)``.
        feature_names : list[str], optional
            Human-readable feature names.  Falls back to
            ``DEFAULT_FEATURE_NAMES`` when *None*.
        **kwargs
            Extra keyword arguments.  Recognised keys:
            - ``role_buckets`` : np.ndarray of shape ``(n_samples,)``
              with integer labels (0 = starter, 1 = rotation, 2 = bench).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.feature_names = feature_names or self.DEFAULT_FEATURE_NAMES

        # ---- 1. Ridge baseline ---- #
        self.logger.info("Fitting Ridge baseline (%d samples, %d features)", *X.shape)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_scaled, y)

        ridge_preds = self._ridge.predict(X_scaled)
        ridge_mae = float(np.mean(np.abs(y - ridge_preds)))
        self.logger.info("Ridge baseline MAE: %.3f", ridge_mae)

        # ---- 2. GBM quantile models ---- #
        quantiles = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
        self._quantile_models = {}

        for label, alpha in quantiles.items():
            self.logger.info("Fitting quantile model q=%.2f (%s)", alpha, label)
            model = self._build_quantile_model(alpha)
            model.fit(X, y)
            self._quantile_models[label] = model

        # ---- 3. Feature importance (from p50 model) ---- #
        self.feature_importances_ = self._extract_feature_importance(
            self._quantile_models["p50"]
        )

        # ---- 4. Role-bucket MAE ---- #
        role_buckets = kwargs.get("role_buckets")
        self._role_mae = self._compute_role_mae(X, y, role_buckets)

        # ---- 5. Metadata ---- #
        p50_preds = self._quantile_models["p50"].predict(X)
        overall_mae = float(np.mean(np.abs(y - p50_preds)))

        self.metadata = ModelMetadata(
            model_name=self.name,
            model_version=self.version,
            hyperparams={
                "ridge_alpha": 1.0,
                "quantile_method": "lightgbm" if _HAS_LIGHTGBM else "sklearn_gbr",
            },
            metrics={
                "ridge_mae": ridge_mae,
                "gbm_p50_mae": overall_mae,
                "role_mae": self._role_mae,
            },
        )

        self.is_fitted = True
        self.logger.info("MinutesModel fit complete. GBM-p50 MAE=%.3f", overall_mae)

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return p50 (median) minutes predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted minutes (p50) for each sample.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return self._quantile_models["p50"].predict(X)

    def predict_quantiles(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return quantile predictions for uncertainty estimation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        dict
            ``{"p10": ndarray, "p50": ndarray, "p90": ndarray}``
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return {
            label: model.predict(X)
            for label, model in self._quantile_models.items()
        }

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        role_buckets: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on held-out data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True minutes.
        role_buckets : np.ndarray, optional
            Integer role labels per sample (0=starter, 1=rotation, 2=bench).

        Returns
        -------
        dict
            Contains ``mae_overall``, ``mae_by_role``, ``interval_coverage``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        quantiles = self.predict_quantiles(X)
        p50 = quantiles["p50"]
        p10 = quantiles["p10"]
        p90 = quantiles["p90"]

        # Overall MAE
        mae_overall = float(np.mean(np.abs(y - p50)))

        # MAE by role bucket
        mae_by_role = self._compute_role_mae(X, y, role_buckets)

        # 80 % prediction interval coverage (p10 to p90)
        in_interval = (y >= p10) & (y <= p90)
        interval_coverage = float(np.mean(in_interval))

        results = {
            "mae_overall": mae_overall,
            "mae_by_role": mae_by_role,
            "interval_coverage_80": interval_coverage,
        }

        self.logger.info(
            "Evaluation: MAE=%.3f, 80%% coverage=%.3f",
            mae_overall,
            interval_coverage,
        )
        return results

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit() first."
            )

    @staticmethod
    def _build_quantile_model(alpha: float) -> Any:
        """Build a quantile regression model for the given quantile *alpha*.

        Uses LightGBM if available, otherwise sklearn
        ``GradientBoostingRegressor`` with ``loss='quantile'``.
        """
        if _HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=300,
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
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )

    @staticmethod
    def _extract_feature_importance(model: Any) -> np.ndarray:
        """Extract feature-importance array from the fitted GBM."""
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=np.float64)
        return np.array([])

    def _compute_role_mae(
        self,
        X: np.ndarray,
        y: np.ndarray,
        role_buckets: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute MAE split by role bucket."""
        role_labels = {0: "starter", 1: "rotation", 2: "bench"}
        result: Dict[str, float] = {}

        if role_buckets is None:
            return result

        role_buckets = np.asarray(role_buckets)
        preds = self._quantile_models["p50"].predict(X) if self.is_fitted or "p50" in self._quantile_models else np.zeros(len(y))

        for code, label in role_labels.items():
            mask = role_buckets == code
            if mask.sum() == 0:
                continue
            result[label] = float(np.mean(np.abs(y[mask] - preds[mask])))

        return result
