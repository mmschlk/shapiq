"""Dispatched conversion of tree-based models to the unified tree layout."""

from __future__ import annotations

from typing import cast

from flextype import flexdispatch

from shapiq._lazy_types import (
    CATBOOST_MODEL,
    LIGHTGBM_BOOSTER,
    LIGHTGBM_MODEL,
    SKLEARN_DECISION_TREE,
    SKLEARN_FOREST,
    XGBOOST_BOOSTER,
    XGBOOST_MODEL,
)
from shapiq.trees._model import TreeModel


@flexdispatch
def to_tree_model(model: object) -> tuple[TreeModel, ...]:
    """Convert a tree-based model to the unified ``TreeModel`` layout.

    Conversions dispatch on the model's type and register lazily: passing a
    scikit-learn tree or forest, an XGBoost, LightGBM, or CatBoost model
    materializes that library's converters without shapiq ever importing it
    on its own. Boosters convert to their margin (raw-score) output, with
    multiclass rounds becoming vector-valued leaves. ``TreeModel`` instances
    and sequences of them pass through, so hand-built trees need no
    registration.

    Args:
        model: A supported tree-based model, a ``TreeModel``, or a sequence
            of ``TreeModel``s.

    Returns:
        The ensemble as a tuple of ``TreeModel``s whose values add up to the
        model's prediction.

    Raises:
        TypeError: If no conversion is registered for the model's type.
    """
    if isinstance(model, (tuple, list)) and all(isinstance(t, TreeModel) for t in model):
        return tuple(cast("TreeModel", tree) for tree in model)
    msg = (
        f"no tree conversion is registered for {type(model).__name__}; pass a "
        "supported model (scikit-learn trees and forests, XGBoost, LightGBM, "
        "CatBoost), or build TreeModel node arrays directly"
    )
    raise TypeError(msg)


@to_tree_model.register
def _tree_model_passthrough(model: TreeModel) -> tuple[TreeModel, ...]:
    return (model,)


@to_tree_model.delayed_register((SKLEARN_DECISION_TREE, SKLEARN_FOREST))
def _register_sklearn_conversions(_: type) -> None:
    """Materialize the scikit-learn converters on first contact."""
    import shapiq.trees._sklearn  # noqa: F401, PLC0415 - registers the handlers


@to_tree_model.delayed_register((XGBOOST_BOOSTER, XGBOOST_MODEL))
def _register_xgboost_conversions(_: type) -> None:
    """Materialize the XGBoost converters on first contact."""
    import shapiq.trees._xgboost  # noqa: F401, PLC0415 - registers the handlers


@to_tree_model.delayed_register((LIGHTGBM_BOOSTER, LIGHTGBM_MODEL))
def _register_lightgbm_conversions(_: type) -> None:
    """Materialize the LightGBM converters on first contact."""
    import shapiq.trees._lightgbm  # noqa: F401, PLC0415 - registers the handlers


@to_tree_model.delayed_register(CATBOOST_MODEL)
def _register_catboost_conversions(_: type) -> None:
    """Materialize the CatBoost converters on first contact."""
    import shapiq.trees._catboost  # noqa: F401, PLC0415 - registers the handlers
