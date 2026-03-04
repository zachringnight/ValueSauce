"""Basketball team-value regression model.

The model estimates a single target (for example: expected point differential
or team strength rating) from basketball box-score style features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Sequence


EPSILON = 1e-12


def _to_float_list(values: Iterable[float]) -> list[float]:
    converted = [float(v) for v in values]
    if not converted:
        raise ValueError("at least one value is required")
    if any(not isfinite(v) for v in converted):
        raise ValueError("all values must be finite")
    return converted


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve Ax=b with Gaussian elimination and partial pivoting."""

    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot_value = aug[pivot_row][col]
        if abs(pivot_value) < EPSILON:
            raise ValueError("unable to solve system; matrix is singular")

        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < EPSILON:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _kfold_indices(sample_count: int, folds: int) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if folds > sample_count:
        raise ValueError("folds cannot exceed sample count")

    buckets = [[] for _ in range(folds)]
    for i in range(sample_count):
        buckets[i % folds].append(i)
    return buckets


@dataclass
class BasketballValueModel:
    """Ridge-regularized linear model for basketball value prediction.

    Features are standardized during training for better numerical stability
    across basketball metrics with different scales.
    """

    ridge_alpha: float = 1e-3
    feature_names: tuple[str, ...] | None = None

    weights: list[float] | None = None
    bias: float = 0.0
    _means: list[float] = field(default_factory=list, init=False)
    _scales: list[float] = field(default_factory=list, init=False)

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        sample_weights: Sequence[float] | None = None,
    ) -> "BasketballValueModel":
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if not features:
            raise ValueError("features cannot be empty")

        row_count = len(features)
        if row_count != len(targets):
            raise ValueError("feature and target counts must match")

        column_count = len(features[0])
        if column_count == 0:
            raise ValueError("features must include at least one column")

        matrix = [_to_float_list(row) for row in features]
        if any(len(row) != column_count for row in matrix):
            raise ValueError("all feature rows must have the same number of columns")

        y = _to_float_list(targets)
        if self.feature_names is not None and len(self.feature_names) != column_count:
            raise ValueError("feature_names length must match feature column count")

        if sample_weights is None:
            w = [1.0] * row_count
        else:
            w = _to_float_list(sample_weights)
            if len(w) != row_count:
                raise ValueError("sample_weights length must match feature count")
            if any(v <= 0 for v in w):
                raise ValueError("sample_weights must be positive")

        total_w = sum(w)
        self._means = [sum(weight * row[c] for weight, row in zip(w, matrix)) / total_w for c in range(column_count)]

        self._scales = []
        for c in range(column_count):
            var = sum(weight * (row[c] - self._means[c]) ** 2 for weight, row in zip(w, matrix)) / total_w
            scale = var**0.5
            if scale <= EPSILON:
                scale = 1.0
            self._scales.append(scale)

        x_std = [[(row[c] - self._means[c]) / self._scales[c] for c in range(column_count)] for row in matrix]

        y_mean = sum(weight * value for weight, value in zip(w, y)) / total_w
        y_centered = [value - y_mean for value in y]

        xtx = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
        xty = [0.0 for _ in range(column_count)]

        for row, target, weight in zip(x_std, y_centered, w):
            for i in range(column_count):
                xty[i] += weight * row[i] * target
                for j in range(column_count):
                    xtx[i][j] += weight * row[i] * row[j]

        for i in range(column_count):
            xtx[i][i] += self.ridge_alpha

        standardized_weights = _solve_linear_system(xtx, xty)
        self.weights = [standardized_weights[c] / self._scales[c] for c in range(column_count)]
        self.bias = y_mean - sum(self.weights[c] * self._means[c] for c in range(column_count))
        return self

    def tune_ridge_alpha(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        candidates: Sequence[float],
        folds: int = 5,
        sample_weights: Sequence[float] | None = None,
    ) -> float:
        values = _to_float_list(candidates)
        if any(v < 0 for v in values):
            raise ValueError("all ridge candidates must be non-negative")

        fold_groups = _kfold_indices(len(features), folds)

        best_alpha = values[0]
        best_score = float("-inf")

        for alpha in values:
            fold_scores: list[float] = []
            for val_idx in fold_groups:
                val_set = set(val_idx)
                train_idx = [i for i in range(len(features)) if i not in val_set]

                train_x = [features[i] for i in train_idx]
                train_y = [targets[i] for i in train_idx]
                val_x = [features[i] for i in val_idx]
                val_y = [targets[i] for i in val_idx]

                train_w = None
                if sample_weights is not None:
                    train_w = [sample_weights[i] for i in train_idx]

                model = BasketballValueModel(ridge_alpha=alpha, feature_names=self.feature_names)
                try:
                    model.fit(train_x, train_y, sample_weights=train_w)
                except ValueError:
                    fold_scores.append(float("-inf"))
                    continue
                fold_scores.append(model.score_r2(val_x, val_y))

            mean_score = sum(fold_scores) / len(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        self.ridge_alpha = best_alpha
        self.fit(features, targets, sample_weights=sample_weights)
        return best_alpha

    def predict_one(self, feature_row: Sequence[float]) -> float:
        if self.weights is None:
            raise RuntimeError("model must be fitted before prediction")

        row = _to_float_list(feature_row)
        if len(row) != len(self.weights):
            raise ValueError("feature row width does not match fitted model")

        return self.bias + sum(w * x for w, x in zip(self.weights, row))

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if not features:
            return []
        return [self.predict_one(row) for row in features]

    def score_r2(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> float:
        y_true = _to_float_list(targets)
        preds = self.predict(features)
        if len(preds) != len(y_true):
            raise ValueError("feature and target counts must match")

        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum((v - y_mean) ** 2 for v in y_true)
        ss_res = sum((a - b) ** 2 for a, b in zip(y_true, preds))
        return 1.0 if ss_tot <= EPSILON else 1 - (ss_res / ss_tot)

    def score_rmse(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> float:
        y_true = _to_float_list(targets)
        preds = self.predict(features)
        if len(preds) != len(y_true):
            raise ValueError("feature and target counts must match")
        mse = sum((a - b) ** 2 for a, b in zip(y_true, preds)) / len(y_true)
        return mse**0.5

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.weights is None:
            raise RuntimeError("model must be fitted before requesting feature importance")

        names = self.feature_names or tuple(f"feature_{i}" for i in range(len(self.weights)))
        pairs = list(zip(names, self.weights))
        return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)


# Backwards compatibility with previous API.
ValueModel = BasketballValueModel
