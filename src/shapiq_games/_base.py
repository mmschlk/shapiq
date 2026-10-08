"""Shared building blocks implementing the game contract of :mod:`shapiq_games`.

Every game in this package follows the same contract:

1. Its values are a pure function of its constructor arguments, including ``random_state``.
   Randomness is drawn once at construction, never during evaluation, so the value of a
   coalition does not depend on call order, batching, or repetition.
2. Local games take the explained point ``x`` as an index into the explanation data or as an
   array. The default is index ``0``, never a random point.
3. Classification games resolve ``class_index`` once at construction. Following the convention of
   the shapiq explainers, ``None`` means class ``1`` for classifiers.

Building games from names (datasets, models, seeds) for benchmarks is the job of
:mod:`shapiq_benchmark.setups`, not of the games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from shapiq.typing import CoalitionMatrix, FloatVector
    from shapiq_games.typing import PredictFunction

__all__ = [
    "as_bool_coalitions",
    "is_classifier",
    "make_predict_function",
    "resolve_class_index",
    "resolve_x",
]


def as_bool_coalitions(coalitions: ArrayLike) -> CoalitionMatrix:
    """Return the coalitions as a two-dimensional boolean matrix.

    ``shapiq.Game.__call__`` already hands boolean coalitions to the value function, but a value
    function can also be called directly, with a vector or with zeros and ones. Games that use
    coalitions as masks therefore convert them first.

    Args:
        coalitions: A coalition vector or matrix with entries in ``{0, 1}``.

    Returns:
        The coalitions as a boolean matrix of shape ``(n_coalitions, n_players)``.
    """
    coalitions = np.asarray(coalitions)
    if coalitions.ndim == 1:
        coalitions = coalitions.reshape(1, -1)
    return coalitions.astype(bool, copy=False)


def resolve_x(x: int | np.integer | ArrayLike, data: np.ndarray) -> FloatVector:
    """Resolve the explained point given as an index or as an array.

    Args:
        x: An index into ``data`` or the point itself, either of shape ``(n_features,)`` or
            ``(1, n_features)``.
        data: The data the index refers to, of shape ``(n_samples, n_features)``.

    Returns:
        The explained point as a one-dimensional array of shape ``(n_features,)``.

    Raises:
        IndexError: If the index is out of range.
        ValueError: If the point does not match the number of features of ``data``.
    """
    if isinstance(x, int | np.integer) and not isinstance(x, bool):
        index = int(x)
        if not -data.shape[0] <= index < data.shape[0]:
            msg = f"x={index} is out of range for data with {data.shape[0]} rows."
            raise IndexError(msg)
        return np.array(data[index], copy=True)
    point = np.asarray(x)
    if point.ndim == 2 and point.shape[0] == 1:
        point = point[0]
    if point.ndim != 1 or point.shape[0] != data.shape[1]:
        msg = (
            f"x must be an index or a point with {data.shape[1]} features, "
            f"but got an array of shape {np.shape(x)}."
        )
        raise ValueError(msg)
    return np.array(point, copy=True)


def is_classifier(model: object) -> bool:
    """Return whether ``model`` is a classifier (scikit-learn compatible)."""
    estimator_type = getattr(model, "_estimator_type", None)
    if estimator_type is None:
        tags = getattr(model, "__sklearn_tags__", None)
        if callable(tags):
            try:
                estimator_type = tags().estimator_type
            except Exception:  # noqa: BLE001 - foreign estimators may implement tags badly
                estimator_type = None
    if estimator_type is not None:
        return estimator_type == "classifier"
    return hasattr(model, "predict_proba")


def resolve_class_index(model: object, class_index: int | None) -> int | None:
    """Resolve the explained class following the convention of the shapiq explainers.

    Args:
        model: The model.
        class_index: The requested class index, or ``None``.

    Returns:
        ``None`` for regressors, otherwise the class index (``1`` if ``class_index`` is ``None``).

    Raises:
        ValueError: If the class index is out of range for the classifier.
    """
    if not is_classifier(model):
        return None
    resolved = 1 if class_index is None else int(class_index)
    classes = getattr(model, "classes_", None)
    if classes is not None and not 0 <= resolved < len(classes):
        msg = f"class_index={resolved} is out of range for a model with {len(classes)} classes."
        raise ValueError(msg)
    return resolved


def make_predict_function(
    model: object,
    class_index: int | None,
) -> PredictFunction:
    """Build a function returning one output per row: a class probability or a regression value.

    Args:
        model: A fitted scikit-learn compatible model.
        class_index: The class whose probability is returned for classifiers, ``None`` for
            regressors.

    Returns:
        A function mapping a ``(n_samples, n_features)`` matrix to a ``(n_samples,)`` vector.
    """
    if class_index is None:
        predict = model.predict  # type: ignore[attr-defined]

        def _predict(x: np.ndarray) -> FloatVector:
            return np.asarray(predict(x), dtype=float).reshape(-1)

        return _predict

    predict_proba = model.predict_proba  # type: ignore[attr-defined]

    def _predict_proba(x: np.ndarray) -> FloatVector:
        return np.asarray(predict_proba(x), dtype=float)[:, class_index]

    return _predict_proba
