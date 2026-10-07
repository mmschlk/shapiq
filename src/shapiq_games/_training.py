"""Shared helpers of the games that train or combine models per coalition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Metric",
    "MetricName",
    "empty_model_score",
    "fit_and_score",
    "resolve_metric",
    "resolve_task",
]

type Metric = Callable[[np.ndarray, np.ndarray], float]
type MetricName = Literal["accuracy", "r2", "neg_mse", "neg_mae"]

_METRICS: dict[str, Metric] = {
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    "r2": lambda y_true, y_pred: float(r2_score(y_true, y_pred)),
    "neg_mse": lambda y_true, y_pred: -float(mean_squared_error(y_true, y_pred)),
    "neg_mae": lambda y_true, y_pred: -float(mean_absolute_error(y_true, y_pred)),
}


def resolve_task(task: str | None, model: object) -> str:
    """Return ``task``, or infer it from the model when ``None``.

    Raises:
        ValueError: If ``task`` is neither ``"classification"`` nor ``"regression"``.
    """
    if task is None:
        from ._base import is_classifier

        return "classification" if is_classifier(model) else "regression"
    if task not in ("classification", "regression"):
        msg = f"task must be 'classification' or 'regression', got {task!r}."
        raise ValueError(msg)
    return task


def resolve_metric(metric: MetricName | Metric | None, task: str) -> Metric:
    """Return the metric ``metric(y_true, y_pred) -> float`` (higher is better).

    Args:
        metric: A metric name, a callable, or ``None`` for the default of the task (accuracy for
            classification, R² for regression).
        task: ``"classification"`` or ``"regression"``.

    Returns:
        The metric function.
    """
    if metric is None:
        metric = "accuracy" if task == "classification" else "r2"
    if callable(metric):
        return metric
    try:
        return _METRICS[metric]
    except KeyError:
        msg = f"Unknown metric {metric!r}. Choose one of {sorted(_METRICS)} or pass a callable."
        raise ValueError(msg) from None


def _seeded_clone(model: Any, random_state: int | None) -> Any:  # noqa: ANN401
    """Return an unfitted clone of ``model``, with every unset ``random_state`` set."""
    estimator = clone(model)
    if random_state is not None and hasattr(estimator, "get_params"):
        unset = {
            name: random_state
            for name, value in estimator.get_params(deep=True).items()
            if (name == "random_state" or name.endswith("__random_state")) and value is None
        }
        if unset:
            estimator.set_params(**unset)
    return estimator


def _too_few_rows(model: Any, n_rows: int) -> bool:  # noqa: ANN401
    """Whether ``model`` cannot be fitted on ``n_rows`` rows (e.g. k-NN with ``k > n_rows``)."""
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    return any(
        isinstance(value, int) and value > n_rows
        for name, value in params.items()
        if name == "n_neighbors" or name.endswith("__n_neighbors")  # also inside a pipeline
    )


def fit_and_score(
    model: Any,  # noqa: ANN401
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: str,
    metric: Metric,
    random_state: int | None = None,
) -> float:
    """Fit a fresh clone of ``model`` and return its test metric.

    A clone is fitted for every call, so evaluations never share state, and an unset
    ``random_state`` of the clone is set to ``random_state``, so the value does not change between
    calls. Classification labels are encoded as ``0, ..., m - 1`` for each fit (some libraries
    require that, and a coalition may lack classes) and decoded for scoring. If a coalition cannot
    train the model (a single class, or fewer rows than a nearest-neighbor model's ``k``), the
    model without features is used instead: the majority class or the training mean.

    Args:
        model: An unfitted scikit-learn compatible estimator (it is cloned, never fitted itself).
        x_train: The training features.
        y_train: The training labels.
        x_test: The test features.
        y_test: The test labels.
        task: ``"classification"`` or ``"regression"``.
        metric: The metric ``metric(y_true, y_pred)``.
        random_state: The seed of clones without one. Defaults to ``None`` (left unset).

    Returns:
        The metric of the fitted model on the test data.
    """
    classes, encoded = np.unique(y_train, return_inverse=True)
    single_class = task == "classification" and classes.shape[0] < 2
    if single_class or _too_few_rows(model, x_train.shape[0]):
        return empty_model_score(y_train, x_test, y_test, task=task, metric=metric)
    estimator = _seeded_clone(model, random_state)
    if task == "classification":
        estimator.fit(x_train, encoded)
        predictions = classes[np.asarray(estimator.predict(x_test)).astype(int).reshape(-1)]
    else:
        predictions = estimator.fit(x_train, y_train).predict(x_test)
    return float(metric(y_test, predictions))


def empty_model_score(
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: str,
    metric: Metric,
) -> float:
    """Return the test metric of a model without features (majority class or training mean)."""
    dummy = (
        DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()
    )
    dummy.fit(np.zeros((y_train.shape[0], 1)), y_train)
    return float(metric(y_test, dummy.predict(np.zeros((x_test.shape[0], 1)))))
