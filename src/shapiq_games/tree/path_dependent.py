"""The path-dependent game of a tree ensemble (the game explained by path-dependent TreeSHAP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq.tree.validation import validate_tree_model
from shapiq_games._base import as_bool_coalitions, resolve_class_index

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel
    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues, IntVector

__all__ = ["PathDependentTreeGame"]

# coalitions per pass: at least 64, and more while a level's weights stay small (cache-sized)
_MIN_CHUNK, _MAX_ELEMENTS = 64, 2**16


@dataclass(frozen=True)
class _Level:
    """The nodes at one depth of a tree, in the order of a breadth-first traversal."""

    leaf_positions: IntVector  # positions of the leaves among the nodes of the level
    leaf_values: FloatVector
    internal_positions: IntVector  # positions of the decision nodes among the nodes of the level
    features: IntVector  # the (original) feature of every decision node
    follows_left: FloatVector  # 1 where the explained point goes left, 0 where it goes right
    share_left: FloatVector  # the share of the training samples that went left


def _levels(tree: TreeModel, x: FloatVector) -> list[_Level]:
    """Split a tree into levels and decide the explained point's path at every decision node.

    The nodes of a level are the children of the decision nodes of the level above, left child
    first: the order in which a breadth-first traversal visits them.
    """
    x = tree.cast_input(np.asarray(x, dtype=np.float64))
    to_original = np.zeros(max(tree.feature_map_internal_original, default=0) + 1, dtype=int)
    for internal, original in tree.feature_map_internal_original.items():
        to_original[internal] = original
    levels = []
    nodes = np.array([int(tree.root_node_id)])
    while nodes.size:
        is_leaf = np.asarray(tree.leaf_mask[nodes], dtype=bool)
        internal = nodes[~is_leaf]
        left, right = tree.children_left[internal], tree.children_right[internal]
        # in float64: the weights of some libraries (XGBoost) are float32
        left_weight = tree.node_sample_weight[left].astype(np.float64)
        total = left_weight + tree.node_sample_weight[right].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            share_left = np.where(total > 0, left_weight / total, 0.5)
        follows_left = [
            tree.goes_left(int(node), x[to_original[tree.features[node]]]) for node in internal
        ]
        levels.append(
            _Level(
                leaf_positions=np.flatnonzero(is_leaf),
                leaf_values=np.asarray(tree.values[nodes[is_leaf]], dtype=np.float64),
                internal_positions=np.flatnonzero(~is_leaf),
                features=to_original[tree.features[internal]],
                follows_left=np.asarray(follows_left, dtype=np.float64),
                share_left=share_left,
            )
        )
        nodes = np.column_stack([left, right]).reshape(-1)  # left child first
    return levels


def _tree_expectation(levels: list[_Level], coalitions_by_player: np.ndarray) -> GameValues:
    """Evaluate the path-dependent expectation of one tree for many coalitions at once.

    At a split on a feature in the coalition, the instance follows its path. At a split on an
    absent feature, both children are visited, weighted by their share of the training samples
    (the node sample weights). The tree output for a coalition is the weighted sum of the reached
    leaf values. All nodes of a level are processed at once; the weights are multiplied along
    every path, and the leaves are summed one after the other in breadth-first order, so a value
    does not depend on the other coalitions.

    Args:
        levels: The levels of the tree (see :func:`_levels`).
        coalitions_by_player: The coalitions as a boolean matrix of shape
            ``(n_players, n_coalitions)``.

    Returns:
        The expectation for every coalition.
    """
    n_coalitions = coalitions_by_player.shape[1]
    output = np.zeros(n_coalitions)
    weights = np.ones((1, n_coalitions))  # one row per node of the level
    for level in levels:
        if level.leaf_positions.size:
            leaves = weights[level.leaf_positions] * level.leaf_values[:, None]
            leaves[0] += output
            output = np.cumsum(leaves, axis=0)[-1]  # in order (a sum may pair the leaves up)
        if not level.internal_positions.size:
            break
        parents = weights[level.internal_positions]
        present = coalitions_by_player[level.features]
        fraction = np.where(present, level.follows_left[:, None], level.share_left[:, None])
        weights = np.stack([parents * fraction, parents * (1.0 - fraction)], axis=1)
        weights = weights.reshape(-1, n_coalitions)  # left child first
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
        empty_value: The value of the empty coalition before centering.

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
        self._levels = [_levels(tree, self.x) for tree in self.trees]
        self._widest_level = max(
            level.leaf_positions.size + 2 * level.internal_positions.size
            for levels in self._levels
            for level in levels
        )
        n_players = self.x.shape[0]
        self.empty_value = float(self.value_function(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the path-dependent expectation of the tree model for the coalitions."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        step = max(_MIN_CHUNK, _MAX_ELEMENTS // self._widest_level)
        for start in range(0, coalitions.shape[0], step):
            chunk = np.ascontiguousarray(coalitions[start : start + step].T)
            values[start : start + step] = sum(
                _tree_expectation(levels, chunk) for levels in self._levels
            )
        return values
