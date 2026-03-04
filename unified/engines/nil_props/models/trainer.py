"""Model training — linear baselines and tree models."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from engines.nil_props.models.columns import (
    CONVERSION_FEATURES,
    CONVERSION_TARGET,
    DIRECT_ASSIST_FEATURES,
    DIRECT_ASSIST_TARGET,
    MINUTES_FEATURES,
    MINUTES_TARGET,
    OPPORTUNITY_FEATURES,
    OPPORTUNITY_TARGET,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for a trained model and its metrics."""

    model_type: str  # 'minutes', 'opportunity', 'conversion'
    model_name: str  # 'linear', 'ridge', 'lgb'
    model: object
    features: list[str]
    target: str
    metrics: dict
    run_id: str
    train_rows: int
    created_at: datetime
    feature_medians: dict | None = None  # median values from training for NaN imputation
    residual_std: float | None = None  # std of residuals from training predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def impute_nans(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values using training medians (consistent with training)."""
        X = X.copy()
        if self.feature_medians:
            for col in self.features:
                if col in X.columns and X[col].isna().any():
                    X[col] = X[col].fillna(self.feature_medians.get(col, 0.0))
        return X


class ModelTrainer:
    """Trains and evaluates models for each layer."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_minutes_model(
        self,
        df: pd.DataFrame,
        train_end_date: str | None = None,
        use_tree: bool = True,
    ) -> ModelResult:
        """Train minutes prediction model."""
        return self._train_layer(
            df=df,
            model_type="minutes",
            features=MINUTES_FEATURES,
            target=MINUTES_TARGET,
            train_end_date=train_end_date,
            use_tree=use_tree,
            clip_range=(0, 48),
        )

    def train_opportunity_model(
        self,
        df: pd.DataFrame,
        train_end_date: str | None = None,
        use_tree: bool = True,
    ) -> ModelResult:
        """Train potential assists prediction model."""
        return self._train_layer(
            df=df,
            model_type="opportunity",
            features=OPPORTUNITY_FEATURES,
            target=OPPORTUNITY_TARGET,
            train_end_date=train_end_date,
            use_tree=use_tree,
            clip_range=(0, 30),
        )

    def train_conversion_model(
        self,
        df: pd.DataFrame,
        train_end_date: str | None = None,
        use_tree: bool = False,
    ) -> ModelResult:
        """Train assist conversion rate model."""
        return self._train_layer(
            df=df,
            model_type="conversion",
            features=CONVERSION_FEATURES,
            target=CONVERSION_TARGET,
            train_end_date=train_end_date,
            use_tree=use_tree,
            clip_range=(0, 1),
            use_ridge=True,
        )

    def _train_layer(
        self,
        df: pd.DataFrame,
        model_type: str,
        features: list[str],
        target: str,
        train_end_date: str | None,
        use_tree: bool,
        clip_range: tuple[float, float],
        use_ridge: bool = False,
    ) -> ModelResult:
        """Generic training logic for any layer."""
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"{model_type}: missing features {missing}, using available")

        work = df.copy()

        # Filter to rows with valid target
        work = work.dropna(subset=[target])
        work = work.dropna(subset=available_features, how="all")

        # Fill remaining NaNs in features with median, then 0 if all-NaN
        # Store medians for consistent inference-time imputation
        feature_medians = {}
        for col in available_features:
            med = work[col].median()
            feature_medians[col] = float(med) if pd.notna(med) else 0.0
            if work[col].isna().any():
                work[col] = work[col].fillna(feature_medians[col])

        if work.empty:
            raise ValueError(f"No valid training data for {model_type}")

        # Time-based split
        if train_end_date:
            train_mask = work["game_date"] <= pd.Timestamp(train_end_date)
        else:
            # Use 80% chronological split
            split_idx = int(len(work) * 0.8)
            dates_sorted = work["game_date"].sort_values()
            split_date = dates_sorted.iloc[split_idx]
            train_mask = work["game_date"] <= split_date

        train = work[train_mask]
        test = work[~train_mask]

        if len(train) < 10:
            raise ValueError(f"Too few training rows for {model_type}: {len(train)}")

        X_train = train[available_features]
        y_train = train[target]
        X_test = test[available_features] if len(test) > 0 else pd.DataFrame()
        y_test = test[target] if len(test) > 0 else pd.Series(dtype=float)

        # Train linear baseline
        if use_ridge:
            linear_model = Ridge(alpha=1.0)
        else:
            linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        best_model = linear_model
        best_name = "ridge" if use_ridge else "linear"

        # Try tree model if available and requested
        if use_tree and HAS_LGB and len(train) >= 30:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_child_samples=max(5, len(train) // 20),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(X_train, y_train)

            if len(test) > 0:
                linear_mae = mean_absolute_error(
                    y_test, np.clip(linear_model.predict(X_test), *clip_range)
                )
                lgb_mae = mean_absolute_error(
                    y_test, np.clip(lgb_model.predict(X_test), *clip_range)
                )
                if lgb_mae < linear_mae:
                    best_model = lgb_model
                    best_name = "lgb"
                    logger.info(
                        f"{model_type}: LGB ({lgb_mae:.3f}) beats linear ({linear_mae:.3f})"
                    )
                else:
                    logger.info(
                        f"{model_type}: Linear ({linear_mae:.3f}) beats LGB ({lgb_mae:.3f})"
                    )
            else:
                best_model = lgb_model
                best_name = "lgb"

        # Compute metrics
        metrics = {"train_rows": len(train), "test_rows": len(test)}
        if len(test) > 0:
            preds = np.clip(best_model.predict(X_test), *clip_range)
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
            metrics["r2"] = float(r2_score(y_test, preds))
            metrics["mean_actual"] = float(y_test.mean())
            metrics["mean_pred"] = float(preds.mean())

        run_id = f"{model_type}_{best_name}_{uuid.uuid4().hex[:8]}"

        # Compute residual std from training predictions
        train_preds = np.clip(best_model.predict(X_train), *clip_range)
        residual_std = float(np.std(y_train.values - train_preds))

        # Save model
        model_path = self.model_dir / f"{run_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        logger.info(f"Trained {model_type} ({best_name}): {metrics}")

        return ModelResult(
            model_type=model_type,
            model_name=best_name,
            model=best_model,
            features=available_features,
            target=target,
            metrics=metrics,
            run_id=run_id,
            train_rows=len(train),
            created_at=datetime.utcnow(),
            feature_medians=feature_medians,
            residual_std=residual_std,
        )

    def train_direct_assist_model(
        self,
        df: pd.DataFrame,
        train_end_date: str | None = None,
        use_tree: bool = True,
    ) -> ModelResult:
        """Train direct assists prediction model (bypasses three-layer decomposition)."""
        return self._train_layer(
            df=df,
            model_type="direct_assists",
            features=DIRECT_ASSIST_FEATURES,
            target=DIRECT_ASSIST_TARGET,
            train_end_date=train_end_date,
            use_tree=use_tree,
            clip_range=(0, 25),
        )

    def train_baseline_rolling(self, df: pd.DataFrame) -> dict:
        """Compute baseline: rolling 10-game average assists."""
        work = df.dropna(subset=["assists", "ast_last_10"]).copy()
        if work.empty:
            return {"mae": None, "rmse": None}
        preds = work["ast_last_10"]
        actual = work["assists"]
        return {
            "mae": float(mean_absolute_error(actual, preds)),
            "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
            "mean_actual": float(actual.mean()),
            "mean_pred": float(preds.mean()),
        }
