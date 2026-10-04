"""XGBoost fingerprint baseline shared by every condition."""

from typing import TYPE_CHECKING

import numpy as np

from influence_pruning.config import ModelSettings

if TYPE_CHECKING:
    from xgboost import XGBRegressor


def fit_regressor(
    train_fingerprints: np.ndarray,
    train_targets: np.ndarray,
    settings: ModelSettings,
    seed: int,
    *,
    require_explicit_base_score: bool = False,
) -> "XGBRegressor":
    """Fit one XGBoost regressor on the endpoint's common scale.

    The default constructor path is the historical one: ``XGBRegressor(**params)`` with no
    ``base_score``. ``require_explicit_base_score`` is opt-in and used only where the model is
    handed to an influence estimator: XGBoost 3.x leaves ``base_score`` as ``None`` (estimated
    internally), while the tree-influence parser requires a float bias. For squared error the
    explicit training mean is the value XGBoost estimates anyway, so the two paths agree
    numerically, but only the opt-in path changes the constructor arguments.

    :param train_fingerprints: ``(n_train, n_bits)`` binary fingerprint matrix.
    :param train_targets: training targets on the endpoint's common scale.
    :param settings: model parameters shared by every condition.
    :param seed: matched seed for this repetition.
    :param require_explicit_base_score: set ``base_score`` to the training mean so an influence
        estimator can parse the fitted model.
    :returns: the fitted regressor.
    """
    from xgboost import XGBRegressor

    targets = np.asarray(train_targets, dtype=np.float32)
    params = settings.xgboost_params(seed)
    if require_explicit_base_score:
        params["base_score"] = float(targets.mean())
    regressor = XGBRegressor(**params)
    regressor.fit(np.asarray(train_fingerprints, dtype=np.float32), targets)
    return regressor


def fit_predict(
    train_fingerprints: np.ndarray,
    train_targets: np.ndarray,
    eval_fingerprints: np.ndarray,
    settings: ModelSettings,
    seed: int,
) -> np.ndarray:
    """Fit one XGBoost model and predict the evaluation molecules.

    :param train_fingerprints: ``(n_train, n_bits)`` binary fingerprint matrix.
    :param train_targets: training targets on the endpoint's common scale.
    :param eval_fingerprints: ``(n_eval, n_bits)`` binary fingerprint matrix.
    :param settings: model parameters shared by every condition.
    :param seed: matched seed for this repetition.
    :returns: predicted values for the evaluation molecules.
    """
    regressor = fit_regressor(train_fingerprints, train_targets, settings, seed)
    return np.asarray(
        regressor.predict(np.asarray(eval_fingerprints, dtype=np.float32)), dtype=float
    )
