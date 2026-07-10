"""Dispatched conversion of tree-based models to the unified tree layout."""

from __future__ import annotations

from typing import cast

from flextype import flexdispatch

from shapiq._lazy_types import SKLEARN_DECISION_TREE, SKLEARN_FOREST
from shapiq.trees._model import TreeModel


@flexdispatch
def to_tree_model(model: object) -> tuple[TreeModel, ...]:
    """Convert a tree-based model to the unified ``TreeModel`` layout.

    Conversions dispatch on the model's type and register lazily: passing a
    scikit-learn tree or forest materializes the scikit-learn converters
    without shapiq ever importing scikit-learn on its own. ``TreeModel``
    instances and sequences of them pass through, so hand-built trees need
    no registration.

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
        "supported model (scikit-learn decision trees and forests), or build "
        "TreeModel node arrays directly"
    )
    raise TypeError(msg)


@to_tree_model.register
def _tree_model_passthrough(model: TreeModel) -> tuple[TreeModel, ...]:
    return (model,)


@to_tree_model.delayed_register((SKLEARN_DECISION_TREE, SKLEARN_FOREST))
def _register_sklearn_conversions(_: type) -> None:
    """Materialize the scikit-learn converters on first contact."""
    import shapiq.trees._sklearn  # noqa: F401, PLC0415 - registers the handlers
