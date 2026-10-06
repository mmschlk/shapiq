"""The output space of tree models, as explained by the tree algorithms of shapiq.

shapiq's tree conversion explains probabilities for scikit-learn decision trees and forests, and
raw margins (log-odds) for gradient boosting classifiers. The tree games evaluate the model's own
predictions in that same space, so that a bug in the conversion is caught when the games are
compared with the tree algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shapiq.utils.modules import safe_isinstance

__all__ = ["check_class_index", "is_margin_classifier", "model_output"]

_MARGIN_CLASSIFIERS = (
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "xgboost.XGBClassifier",
    "lightgbm.LGBMClassifier",
    "catboost.CatBoostClassifier",
)


def is_margin_classifier(model: Any) -> bool:  # noqa: ANN401
    """Return whether shapiq explains the raw margins (log-odds) of this classifier."""
    return safe_isinstance(model, _MARGIN_CLASSIFIERS)


def check_class_index(model: Any, class_index: int | None) -> None:  # noqa: ANN401
    """Reject class 0 of binary margin classifiers.

    For binary gradient boosting models, shapiq's tree conversion has a single margin, the one of
    the positive class, and explains it whatever class is requested. Class 0 is rejected instead of
    silently explaining class 1 (its margin is the negative of the class-1 margin).

    Raises:
        ValueError: If ``class_index`` is ``0`` for a binary margin classifier.
    """
    classes = getattr(model, "classes_", None)
    binary = classes is not None and len(classes) == 2
    if class_index == 0 and binary and is_margin_classifier(model):
        msg = (
            f"class_index=0 is not supported for binary {type(model).__name__} models: shapiq's "
            "tree algorithms explain the margin of the positive class. Use class_index=1 (the "
            "class-0 margin is its negative)."
        )
        raise ValueError(msg)


def _raw_margins(model: Any, data: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return the raw margins of a margin classifier, shape ``(n,)`` or ``(n, n_classes)``."""
    if safe_isinstance(model, "xgboost.XGBClassifier"):
        import xgboost as xgb

        return model.get_booster().predict(xgb.DMatrix(data), output_margin=True)
    if safe_isinstance(model, "lightgbm.LGBMClassifier"):
        return model.predict(data, raw_score=True)
    if safe_isinstance(model, "catboost.CatBoostClassifier"):
        return model.predict(data, prediction_type="RawFormulaVal")
    return model.decision_function(data)  # scikit-learn gradient boosting


def model_output(model: Any, data: np.ndarray, class_index: int | None) -> np.ndarray:  # noqa: ANN401
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
    if margins.ndim == 1:  # binary: the margin of the positive class
        return margins
    return margins[:, class_index]
