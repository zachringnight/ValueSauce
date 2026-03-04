import math

import pytest

from model import BasketballValueModel, ValueModel


def test_fit_and_predict_learns_basketball_signal():
    # Features: [off_rating, def_rating_allowed, rebound_rate]
    features = [
        [112, 105, 52],
        [108, 110, 49],
        [120, 102, 55],
        [100, 115, 47],
        [115, 107, 54],
    ]

    # Synthetic strength model:
    # strength = 0.7*off_rating - 0.6*def_rating_allowed + 1.5*rebound_rate + 12
    targets = [
        0.7 * 112 - 0.6 * 105 + 1.5 * 52 + 12,
        0.7 * 108 - 0.6 * 110 + 1.5 * 49 + 12,
        0.7 * 120 - 0.6 * 102 + 1.5 * 55 + 12,
        0.7 * 100 - 0.6 * 115 + 1.5 * 47 + 12,
        0.7 * 115 - 0.6 * 107 + 1.5 * 54 + 12,
    ]

    model = BasketballValueModel(ridge_alpha=1e-6).fit(features, targets)
    pred = model.predict_one([118, 104, 53])
    expected = 0.7 * 118 - 0.6 * 104 + 1.5 * 53 + 12
    assert math.isclose(pred, expected, rel_tol=1e-6)


def test_regularization_handles_collinearity():
    # Second feature is 2x first feature -> exact multicollinearity.
    features = [[1, 2], [2, 4], [3, 6], [4, 8]]
    targets = [3, 6, 9, 12]

    model = BasketballValueModel(ridge_alpha=1.0).fit(features, targets)
    assert model.score_r2(features, targets) > 0.95


def test_weighted_fit_respects_high_leverage_games():
    # Last sample has heavier weight and should pull prediction near its local pattern.
    features = [[100], [101], [102], [120]]
    targets = [10, 11, 12, 30]
    weights = [1, 1, 1, 10]

    model = BasketballValueModel(ridge_alpha=1e-6).fit(features, targets, sample_weights=weights)
    assert model.predict_one([120]) > 25


def test_tune_ridge_alpha_runs_cross_validation_and_fits():
    features = [[i, 2 * i + 1] for i in range(1, 16)]
    targets = [3.5 * i + 2.0 for i in range(1, 16)]

    model = BasketballValueModel()
    best = model.tune_ridge_alpha(features, targets, candidates=[0.0, 1e-4, 1e-2, 1.0], folds=5)

    assert best in {0.0, 1e-4, 1e-2, 1.0}
    assert model.score_r2(features, targets) > 0.99


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="fitted"):
        BasketballValueModel().predict_one([1, 2, 3])


@pytest.mark.parametrize(
    "features,targets,error",
    [
        ([], [], "features cannot be empty"),
        ([[1], [2]], [1], "counts must match"),
        ([[]], [1], "at least one column"),
        ([[1], [2, 3]], [1, 2], "same number of columns"),
    ],
)
def test_fit_validation(features, targets, error):
    with pytest.raises(ValueError, match=error):
        BasketballValueModel().fit(features, targets)


def test_validation_for_non_finite_values_and_weights():
    with pytest.raises(ValueError, match="finite"):
        BasketballValueModel().fit([[1, float("inf")]], [1])

    with pytest.raises(ValueError, match="positive"):
        BasketballValueModel().fit([[1], [2]], [1, 2], sample_weights=[1, 0])


def test_feature_name_validation_and_importance():
    model = BasketballValueModel(feature_names=("off_rating", "def_rating"))
    with pytest.raises(ValueError, match="feature_names"):
        model.fit([[1, 2, 3], [2, 3, 4]], [1, 2])

    fitted = BasketballValueModel(feature_names=("off_rating", "def_rating")).fit(
        [[110, 105], [112, 103], [108, 109], [115, 101]],
        [5.0, 7.1, 2.9, 9.4],
    )
    importance = fitted.feature_importance()
    assert importance[0][0] in {"off_rating", "def_rating"}


def test_rmse_metric_and_backward_compatible_alias():
    model = ValueModel(ridge_alpha=1e-6).fit([[1, 2], [2, 3], [3, 4]], [3, 5, 7])
    assert isinstance(model, BasketballValueModel)
    assert model.score_rmse([[1, 2], [2, 3], [3, 4]], [3, 5, 7]) < 1e-2
