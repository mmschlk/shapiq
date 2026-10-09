"""Shared building blocks implementing the game contract of :mod:`shapiq_games`.

Every game in this package follows the same contract:

1. Its values are a pure function of its constructor arguments, including ``random_state``.
   Randomness is drawn once at construction, never during evaluation, so the value of a
   coalition does not depend on call order, batching, or repetition.
2. Local games take the explained point ``x`` explicitly, never a random one. Games that hold
   explanation data also accept an index into it (default ``0``).
3. Classification games resolve ``class_index`` once at construction. Following the convention of
   the shapiq explainers, ``None`` means class ``1`` for classifiers; the image games, whose
   models have many classes, explain the class predicted for the full image instead.
4. Games that explain one model output store it before centering: ``empty_value`` is the value
   of the empty coalition and ``original_model_output`` the value of the grand coalition (the
   model's output on the full input).

Building games from names (datasets, models, seeds) for benchmarks is the job of
:mod:`shapiq_benchmark.setups`, not of the games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.utils.modules import safe_isinstance

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from shapiq.typing import CoalitionMatrix, FloatVector
    from shapiq_games.typing import PredictFunction

__all__ = [
    "as_bool_coalitions",
    "is_classifier",
    "make_predict_function",
    "predicts_row_by_row",
    "resolve_class_index",
    "resolve_predict_function",
    "resolve_x",
]


# Models whose output for a row is the same, bit for bit, in a call with any other rows: tree
# models sum their trees row by row. Other models need not be: BLAS and torch pick kernels and
# summation orders by the number of rows, which changes the last digits.
_ROW_BY_ROW_MODELS = [
    f"{module}.{kind}{task}"
    for module, kind in (
        ("sklearn.tree", "DecisionTree"),
        ("sklearn.tree", "ExtraTree"),
        ("sklearn.ensemble", "RandomForest"),
        ("sklearn.ensemble", "ExtraTrees"),
        ("sklearn.ensemble", "GradientBoosting"),
        ("sklearn.ensemble", "HistGradientBoosting"),
        ("xgboost", "XGB"),
        ("lightgbm", "LGBM"),
        ("catboost", "CatBoost"),
    )
    for task in ("Classifier", "Regressor")
]
_GRADIENT_BOOSTING = [f"sklearn.ensemble.GradientBoosting{t}" for t in ("Classifier", "Regressor")]


def predicts_row_by_row(model: object) -> bool:
    """Return whether the model's output for a row does not depend on the other rows of a call.

    Games may then pass the rows of many coalitions to the model at once without their values
    depending on the batch. This holds bit for bit for tree models (scikit-learn, XGBoost,
    LightGBM, CatBoost), also given as a bound method such as ``model.predict``; other models
    are evaluated one coalition at a time. (A scikit-learn forest with several ``n_jobs`` adds its
    trees in the order its threads finish, so its output varies in the last digits anyway.)
    """
    model = getattr(model, "__self__", model)
    if safe_isinstance(model, _GRADIENT_BOOSTING) and getattr(model, "init", None) not in (
        None,
        "zero",
    ):  # its initial model, e.g. a linear one, is evaluated on all rows at once
        return False
    return safe_isinstance(model, _ROW_BY_ROW_MODELS)


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


def resolve_predict_function(
    model: object,
    class_index: int | None,
) -> tuple[PredictFunction, int | None]:
    """Turn a model into a function with one output per row, and resolve the explained class.

    Args:
        model: A fitted scikit-learn compatible model, or a callable mapping a
            ``(n_samples, n_features)`` matrix to ``(n_samples,)`` outputs, used as is.
        class_index: The requested class for classifiers, or ``None`` (see
            :func:`resolve_class_index`).

    Returns:
        The prediction function and the resolved class index (``None`` for regressors and
        callables).
    """
    if callable(model) and not hasattr(model, "predict"):
        return model, None  # type: ignore[return-value]
    resolved = resolve_class_index(model, class_index)
    return make_predict_function(model, resolved), resolved
