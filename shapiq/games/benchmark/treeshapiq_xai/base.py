"""This module contains the base TreeSHAP-IQ xai game."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.tree import TreeExplainer, TreeModel
from shapiq.games.base import Game

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from shapiq.utils.custom_types import Model


class TreeSHAPIQXAI(Game):
    """A TreeSHAP-IQ explanation game for tree models.

    The game is based on the TreeSHAP-IQ algorithm and is used to explain the predictions of tree
    models. TreeSHAP-IQ is used to compute Shapley interaction values for tree models.
    """

    def __init__(
        self,
        x: np.ndarray,
        tree_model: Model,
        *,
        class_label: int | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initializes the TreeSHAP-IQ explanation game.

        Args:
            x: The feature vector to be explained. If ``None``, then the first data point is used.
                If an integer, then the data point at the given index is used. If a numpy array,
                then the data point is used as is. Defaults to ``None``.

            tree_model: The tree model to explain as a callable function. The model must be a
                decision tree or random forest model.

            class_label: The class label to use for the model. If ``None``, then the model is
                assumed to be a regression model and / or the default behaviour of
                :class:`~shapiq.explainer.tree.TreeExplainer` is used.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.
        """
        n_players = x.shape[-1]

        self.model = copy.deepcopy(tree_model)
        self.class_label = class_label

        # set up explainer for model transformation (we don't need the explainer here for the gt)
        self._tree_explainer = TreeExplainer(
            model=tree_model,
            min_order=1,
            max_order=1,
            index="SII",
            class_index=class_label,
        )
        # compute ground truth values
        self.empty_value = float(self._tree_explainer.baseline_value)

        # get attributes for manual tree traversal and evaluation
        self._trees: list[TreeModel] = self._tree_explainer._trees  # noqa: SLF001
        self.x_explain = x
        # transform x_explain into a 1-dimensional array if it is a 2-dimensional array
        if x.ndim == 2 and x.shape[0] == 1:
            self.x_explain = x.flatten()
        elif x.ndim > 2:
            msg = "x_explain must be a 1-dimensional or 2-dimensional array."
            raise ValueError(msg)

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the output of the tree model given coalitions of features.

        Args:
            coalitions: A binary matrix of feature subsets present (`True`) or absent (`False`).
                The matrix is expected to be in shape `(n_coalitions, n_features)`.

        Returns:
            The worth of the coalitions as a vector.

        """
        worth = np.zeros(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:  # empty subset
                worth[i] = self.empty_value
                continue
            worth[i] = self.compute_tree_output_from_coalition(coalition)
        return worth

    def compute_tree_output_from_coalition(self, coalition: np.ndarray) -> float:
        """Computes the output of the tree models contained in the game.

        Args:
            coalition: A binary vector of feature indices present (`True`) and absent (`False`) from
                the coalition.

        Returns:
            The prediction given partial feature information as an average of individual tree
                predictions.

        """
        output = 0.0
        for tree in self._trees:
            tree_prediction = _get_tree_prediction(
                node_id=tree.nodes[0],
                tree=tree,
                coalition=coalition,
                x_explain=self.x_explain,
            )
            output += tree_prediction
        return output

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Computes the exact interaction values for the game.

        Args:
            index: The type of interaction index to be computed.
            order: The order of interactions to be computed.

        Returns:
            The exact interaction values for the game.

        """
        tree_explainer = TreeExplainer(
            model=self.model,
            min_order=0,
            max_order=order,
            index=index,
            class_index=self.class_label,
        )
        return tree_explainer.explain(x=self.x_explain)


def _get_tree_prediction(
    node_id: int,
    tree: TreeModel,
    coalition: np.ndarray,
    x_explain: np.ndarray,
) -> float:
    """Traverses the tree and retrieves the prediction of the tree given subsets of features.

    Args:
        node_id: The current node in the tree model as an integer.
        tree: The tree model to traverse and get the predictions for.
        coalition: The binary coalition vector denoting what features are present (`True`) and
            absent (`False`).
        x_explain: The feature vector which is to be explained with numerical feature values.

    Returns:
         The tree prediction given partial feature information.

    """
    if tree.leaf_mask[node_id]:  # end of recursion (base case, return the leaf prediction)
        return tree.values[node_id]
    # not a leaf we have to go deeper
    feature_id, threshold = tree.features[node_id], tree.thresholds[node_id]
    is_present = bool(coalition[feature_id])
    left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]
    if is_present:
        next_node = left_child if x_explain[feature_id] <= threshold else right_child
        tree_prediction = _get_tree_prediction(next_node, tree, coalition, x_explain)
    else:  # feature is out of coalition we have to go both ways and average the predictions
        prediction_left = _get_tree_prediction(left_child, tree, coalition, x_explain)
        prediction_right = _get_tree_prediction(right_child, tree, coalition, x_explain)
        # get weights (tree probabilities of going left or right)
        left_weight = tree.node_sample_weight[left_child]
        right_weight = tree.node_sample_weight[right_child]
        sum_of_weights = left_weight + right_weight
        # scale predictions
        prediction_left *= left_weight / sum_of_weights
        prediction_right *= right_weight / sum_of_weights
        tree_prediction = prediction_left + prediction_right
    return tree_prediction
