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


def fit_and_score(
    model: Any,  # noqa: ANN401
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: str,
    metric: Metric,
) -> float:
    """Fit a fresh clone of ``model`` and return its test metric.

    A clone is fitted for every call, so evaluations never share state. If the training labels of
    a classification task contain a single class, a constant predictor of that class is used
    instead (many libraries refuse to fit a single class).

    Args:
        model: An unfitted scikit-learn compatible estimator (it is cloned, never fitted itself).
        x_train: The training features.
        y_train: The training labels.
        x_test: The test features.
        y_test: The test labels.
        task: ``"classification"`` or ``"regression"``.
        metric: The metric ``metric(y_true, y_pred)``.

    Returns:
        The metric of the fitted model on the test data.
    """
    if task == "classification" and np.unique(y_train).shape[0] < 2:
        fitted = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    else:
        fitted = clone(model).fit(x_train, y_train)
    return float(metric(y_test, fitted.predict(x_test)))


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
