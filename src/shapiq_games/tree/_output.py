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

if TYPE_CHECKING:
    from shapiq.typing import FloatVector

__all__ = ["is_margin_classifier", "model_output"]

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


def _raw_margins(model: Any, data: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return the raw margins of a margin classifier, shape ``(n,)`` or ``(n, n_classes)``."""
    if safe_isinstance(model, "xgboost.XGBClassifier"):
        return model.predict(data, output_margin=True)
    if safe_isinstance(model, "lightgbm.LGBMClassifier"):
        return model.predict(data, raw_score=True)
    if safe_isinstance(model, "catboost.CatBoostClassifier"):
        return model.predict(data, prediction_type="RawFormulaVal")
    return model.decision_function(data)  # scikit-learn gradient boosting


def model_output(model: Any, data: np.ndarray, class_index: int | None) -> FloatVector:  # noqa: ANN401
    """Return the model output for every row in the space the tree algorithms explain.

    Args:
        model: The fitted model.
        data: The rows of shape ``(n, n_features)``.
        class_index: The explained class, or ``None`` for regressors.

    Returns:
        The probability of the class (scikit-learn trees and forests, other classifiers), the raw
        margin of the class (gradient boosting classifiers), or the prediction (regressors), of
        shape ``(n,)``.
    """
    if class_index is None:
        return np.asarray(model.predict(data), dtype=float).reshape(-1)
    if not is_margin_classifier(model):
        return np.asarray(model.predict_proba(data), dtype=float)[:, class_index]
    margins = np.asarray(_raw_margins(model, data), dtype=float)
    if margins.ndim == 1:  # binary: one margin, the log-odds of class 1; class 0 has its negative
        return margins if class_index == 1 else -margins
    return margins[:, class_index]
