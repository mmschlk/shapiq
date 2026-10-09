"""Shared helpers of the games that train or combine models per coalition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from ._base import is_classifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq_games.typing import Metric, MetricName, Task

    # a metric of many predictions at once: (y_true, predictions of shape (n, n_test)) -> (n,)
    type RowMetric = Callable[[np.ndarray, np.ndarray], np.ndarray]

__all__ = [
    "empty_model_score",
    "fit_and_score",
    "resolve_metric",
    "resolve_row_metric",
    "resolve_task",
]


# The named metrics of scikit-learn for many predictions at once, with scikit-learn's arithmetic
# (the tests check that the values are the same bit for bit) but without its input validation,
# which costs about 0.4 ms per call.
def _accuracy(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.mean(predictions == np.asarray(y_true), axis=1)


def _neg_mse(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    errors = np.asarray(y_true, dtype=float) - np.asarray(predictions, dtype=float)
    return -np.mean(errors**2, axis=1)


def _neg_mae(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    errors = np.asarray(predictions, dtype=float) - np.asarray(y_true, dtype=float)
    return -np.mean(np.abs(errors), axis=1)


def _r2(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    if y_true.shape[0] < 2:  # undefined, as in scikit-learn
        return np.full(predictions.shape[0], np.nan)
    numerator = np.sum((y_true - np.asarray(predictions, dtype=float)) ** 2, axis=1)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    # scikit-learn's force_finite: 1 for perfect predictions, 0 for a constant target
    if denominator == 0:
        return np.where(numerator == 0, 1.0, 0.0)
    return np.where(numerator == 0, 1.0, 1 - numerator / denominator)


_ROW_METRICS: dict[str, RowMetric] = {
    "accuracy": _accuracy,
    "r2": _r2,
    "neg_mse": _neg_mse,
    "neg_mae": _neg_mae,
}


def resolve_task(task: Task | None, model: object) -> Task:
    """Return ``task``, or infer it from the model when ``None``.

    Raises:
        ValueError: If ``task`` is neither ``"classification"`` nor ``"regression"``.
    """
    if task is None:
        return "classification" if is_classifier(model) else "regression"
    if task not in ("classification", "regression"):
        msg = f"task must be 'classification' or 'regression', got {task!r}."
        raise ValueError(msg)
    return task


def resolve_metric(metric: MetricName | Metric | None, task: Task) -> Metric:
    """Return the metric ``metric(y_true, y_pred) -> float`` (higher is better).

    Args:
        metric: A metric name, a callable, or ``None`` for the default of the task (accuracy for
            classification, R² for regression).
        task: ``"classification"`` or ``"regression"``.

    Returns:
        The metric function.
    """
    if callable(metric):
        return metric
    row_metric = resolve_row_metric(metric, task)
    return lambda y_true, y_pred: float(row_metric(y_true, np.asarray(y_pred)[None])[0])


def resolve_row_metric(metric: MetricName | Metric | None, task: Task) -> RowMetric:
    """Return the metric of many predictions at once (see :func:`resolve_metric`).

    Returns:
        A function mapping the labels and predictions of shape ``(n, n_test)`` to ``n`` values.
    """
    if callable(metric):
        scalar = metric
        return lambda y_true, predictions: np.array([scalar(y_true, p) for p in predictions])
    if metric is None:
        metric = "accuracy" if task == "classification" else "r2"
    try:
        return _ROW_METRICS[metric]
    except KeyError:
        msg = f"Unknown metric {metric!r}. Choose one of {sorted(_ROW_METRICS)} or pass a callable."
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
    task: Task,
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
    task: Task,
    metric: Metric,
) -> float:
    """Return the test metric of a model without features (majority class or training mean)."""
    dummy = (
        DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()
    )
    dummy.fit(np.zeros((y_train.shape[0], 1)), y_train)
    return float(metric(y_test, dummy.predict(np.zeros((x_test.shape[0], 1)))))
