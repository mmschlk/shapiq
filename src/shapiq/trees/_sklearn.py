"""Scikit-learn converters to the unified tree layout."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble._forest import BaseForest  # noqa: TC002 - registration needs the class
from sklearn.tree._classes import BaseDecisionTree  # noqa: TC002 - registration needs the class

from shapiq.trees._conversion import to_tree_model
from shapiq.trees._model import TreeModel, trusted_tree_model


def _from_sklearn_tree(tree: BaseDecisionTree, *, scale: float = 1.0) -> TreeModel:
    """Convert one fitted scikit-learn tree, scaling its leaf values."""
    inner = tree.tree_
    values = np.asarray(inner.value)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    return trusted_tree_model(
        children_left=np.asarray(inner.children_left, dtype=np.int64),
        children_right=np.asarray(inner.children_right, dtype=np.int64),
        features=np.asarray(inner.feature, dtype=np.int64),
        thresholds=np.asarray(inner.threshold, dtype=np.float64),
        values=np.asarray(values * scale, dtype=np.float64),
    )


@to_tree_model.register
def _sklearn_tree_to_model(model: BaseDecisionTree) -> tuple[TreeModel, ...]:
    return (_from_sklearn_tree(model),)


@to_tree_model.register
def _sklearn_forest_to_model(model: BaseForest) -> tuple[TreeModel, ...]:
    # forests predict the mean over trees, while tree games sum them
    scale = 1.0 / len(model.estimators_)
    return tuple(_from_sklearn_tree(tree, scale=scale) for tree in model.estimators_)
