"""The output space of tree models, as explained by the tree algorithms of shapiq.

shapiq's tree conversion explains probabilities for scikit-learn decision trees and forests, and
raw margins (log-odds) for gradient boosting classifiers. The tree games evaluate the model's own
predictions in that same space, so that a bug in the conversion is caught when the games are
compared with the tree algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.utils.modules import safe_isinstance
from shapiq_games._base import is_classifier, make_predict_function, resolve_class_index

if TYPE_CHECKING:
    from shapiq.typing import FloatVector

__all__ = ["is_margin_classifier", "model_output", "tree_class_index"]

_MARGIN_CLASSIFIERS = [
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "xgboost.XGBClassifier",
    "lightgbm.LGBMClassifier",
    "catboost.CatBoostClassifier",
]


def is_margin_classifier(model: Any) -> bool:  # noqa: ANN401
    """Return whether shapiq explains the raw margins (log-odds) of this classifier."""
    return safe_isinstance(model, _MARGIN_CLASSIFIERS)


def tree_class_index(model: Any, class_index: int | None) -> int | None:  # noqa: ANN401
    """Resolve the explained class of a tree model (see :func:`resolve_class_index`).

    An explicit ``class_index`` is kept for models that are not recognized as classifiers, e.g. a
    raw LightGBM ``Booster``; the tree conversion checks it.
    """
    if class_index is not None and not is_classifier(model):
        return int(class_index)
    return resolve_class_index(model, class_index)


def _raw_margins(model: Any, data: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return the raw margins of a margin classifier, shape ``(n,)`` or ``(n, n_classes)``."""
    if safe_isinstance(model, "xgboost.XGBClassifier"):
        return model.predict(data, output_margin=True)
    if safe_isinstance(model, "lightgbm.LGBMClassifier"):
        return model.predict(data, raw_score=True)
    if safe_isinstance(model, "catboost.CatBoostClassifier"):
        return model.predict(data, prediction_type="RawFormulaVal")
    return model.decision_function(data)  # scikit-learn gradient boosting


def _booster_output(output: Any, class_index: int | None) -> FloatVector:  # noqa: ANN401
    """Return the output of the class from a raw booster's outputs (one, or one per class)."""
    output = np.asarray(output, dtype=float)
    if output.ndim == 2:
        if class_index is None:
            msg = "The model has one output per class: pass the class_index to explain."
            raise ValueError(msg)
        return output[:, class_index]
    if class_index not in (None, 1):
        msg = f"class_index={class_index} cannot be selected from a model with one output."
        raise ValueError(msg)
    return output


def model_output(model: Any, data: np.ndarray, class_index: int | None) -> FloatVector:  # noqa: ANN401
    """Return the model output for every row in the space the tree algorithms explain.

    Args:
        model: The fitted model.
        data: The rows of shape ``(n, n_features)``.
        class_index: The explained class, or ``None`` for regressors.

    Returns:
        The probability of the class (scikit-learn trees and forests, other classifiers), the raw
        margin of the class (gradient boosting classifiers), the raw score (a LightGBM
        ``Booster``, the sum of its trees), or the prediction (regressors), of shape ``(n,)``.
    """
    if safe_isinstance(model, "lightgbm.basic.Booster"):  # its predict gives probabilities
        return _booster_output(model.predict(data, raw_score=True), class_index)
    if class_index is not None and not is_classifier(model):  # e.g. another raw booster
        return _booster_output(model.predict(data), class_index)
    if class_index is None or not is_margin_classifier(model):
        return make_predict_function(model, class_index)(data)
    margins = np.asarray(_raw_margins(model, data), dtype=float)
    if margins.ndim == 1:  # binary: one margin, the log-odds of class 1; class 0 has its negative
        return margins if class_index == 1 else -margins
    return margins[:, class_index]
