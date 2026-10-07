"""The path-dependent game of a tree ensemble (the game explained by path-dependent TreeSHAP)."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq.tree.validation import validate_tree_model
from shapiq_games._base import as_bool_coalitions, resolve_class_index

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel

__all__ = ["PathDependentTreeGame"]

_CHUNK_SIZE = 4096


def _tree_expectation(tree: TreeModel, x: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
    """Evaluate the path-dependent expectation of one tree for many coalitions at once.

    At a split on a feature in the coalition, the instance follows its path. At a split on an
    absent feature, both children are visited, weighted by their share of the training samples
    (the node sample weights). The tree output for a coalition is the weighted sum of the reached
    leaf values.
    """
    x = tree.cast_input(np.asarray(x, dtype=np.float64))
    n_coalitions = coalitions.shape[0]
    output = np.zeros(n_coalitions)
    weights: dict[int, np.ndarray] = {int(tree.root_node_id): np.ones(n_coalitions)}
    queue = deque([int(tree.root_node_id)])
    while queue:
        node = queue.popleft()
        node_weight = weights.pop(node)
        if tree.leaf_mask[node]:
            output += node_weight * tree.values[node]
            continue
        feature = tree.feature_map_internal_original[int(tree.features[node])]
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        left_weight = tree.node_sample_weight[left]
        right_weight = tree.node_sample_weight[right]
        total = left_weight + right_weight
        share_left = left_weight / total if total > 0 else 0.5
        follows_left = 1.0 if tree.goes_left(node, x[feature]) else 0.0
        fraction_left = np.where(coalitions[:, feature], follows_left, share_left)
        for child, fraction in ((left, fraction_left), (right, 1.0 - fraction_left)):
            if child in weights:
                weights[child] += node_weight * fraction
            else:
                weights[child] = node_weight * fraction
                queue.append(child)
    return output


class PathDependentTreeGame(Game):
    r"""The path-dependent game of a tree model.

    The value of a coalition :math:`S` is the expected output of the tree model where features in
    :math:`S` take the value of the explained point :math:`x` and absent features are integrated
    out along the tree paths, weighted by the training samples that reached each node:

    .. math::
        v(S) = \sum_{t} \mathbb{E}_{\text{path}}[f_t(x_S, X_{\bar S})]

    This is the game that path-dependent TreeSHAP and TreeSHAP-IQ explain, so its exact values
    for every index are available through :class:`~shapiq.tree.TreeExplainer`. Any tree model
    supported by :class:`~shapiq.tree.TreeExplainer` works (scikit-learn, XGBoost, LightGBM,
    CatBoost). The output space is the one of the tree explainer (probabilities for scikit-learn
    trees and forests, margins for gradient boosting classifiers; class 0 of a binary booster is
    the negated margin of class 1).

    Attributes:
        model: The tree model.
        x: The explained point.
        class_index: The explained class for classifiers, ``None`` for regressors.
        trees: The trees in the unified :class:`~shapiq.tree.TreeModel` format.

    Examples:
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> X, y = make_regression(n_samples=200, n_features=4, random_state=0)
        >>> model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
        >>> game = PathDependentTreeGame(model, x=X[0], normalize=False)
        >>> bool(np.isclose(game(game.grand_coalition)[0], model.predict(X[:1])[0]))
        True
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        x: np.ndarray,
        *,
        class_index: int | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the path-dependent tree game.

        Args:
            model: A fitted tree model or ensemble, or trees in the
                :class:`~shapiq.tree.TreeModel` format.
            x: The explained point of shape ``(n_features,)``.
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        self.model = model
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.class_index = resolve_class_index(model, class_index)
        self.trees: list[TreeModel] = validate_tree_model(model, class_label=self.class_index)
        n_players = self.x.shape[0]
        empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=empty_value,
            verbose=verbose,
        )

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(coalitions.shape[0])
        for start in range(0, coalitions.shape[0], _CHUNK_SIZE):
            chunk = coalitions[start : start + _CHUNK_SIZE]
            values[start : start + _CHUNK_SIZE] = sum(
                _tree_expectation(tree, self.x, chunk) for tree in self.trees
            )
        return values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the path-dependent expectation of the tree model for the coalitions."""
        return self._evaluate(as_bool_coalitions(coalitions))
