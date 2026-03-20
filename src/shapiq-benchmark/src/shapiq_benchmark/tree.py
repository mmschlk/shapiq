"""Benchmark based on tree models using TreeSHAPIQ as the ground truth computer."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from intiq.interventional import InterventionalGame, InterventionalTreeExplainer
from shapiq import Game, TreeExplainer
from shapiq.explainer.tree.treeshapiq import TreeSHAPIQIndices
from shapiq.explainer.tree.validation import validate_tree_model

from .base import Benchmark, GroundTruthComputer

if TYPE_CHECKING:
    from shapiq.explainer.tree import TreeModel
    from shapiq.interaction_values import InteractionValues


class InterventionalTreeBenchmark(Benchmark):
    """A benchmark for interventional explanations using tree models."""

    def __init__(
        self,
        tree_model: object,
        x_explain: np.ndarray,
        reference_data: np.ndarray,
        *,
        class_label: int | None = None,
    ) -> None:
        """Initializes the Interventional Tree benchmark."""
        computer = InterventionalTreeComputer(
            tree_model=tree_model,
            x_explain=x_explain,
            reference_data=reference_data,
            class_label=class_label,
        )
        game = InterventionalGame(
            model=tree_model,
            reference_data=reference_data,
            target_instance=x_explain,
            class_index=class_label,
        )
        super().__init__(game=game, computer=computer)


class TreeSHAPIQBenchmark(Benchmark):
    """A benchmark for TreeSHAP-IQ explanations using tree models."""

    def __init__(
        self,
        tree_model: object,
        x_explain: np.ndarray,
        *,
        normalize: bool = True,
        class_label: int | None = None,
    ) -> None:
        """Initializes the TreeSHAP-IQ benchmark."""
        computer = TreeSHAPIQComputer(tree_model=tree_model, x_explain=x_explain)
        game = TreeLocalXAI(
            tree_model=tree_model,
            x_explain=x_explain,
            class_label=class_label,
            normalize=normalize,
        )
        super().__init__(game=game, computer=computer)


class TreeSHAPIQComputer(GroundTruthComputer[TreeSHAPIQIndices]):
    """Ground truth computer for tree-based models using shapiq's TreeSHAP-IQ algorithm."""

    tree_model: object
    """The tree model to compute the exact values for. The tree model must be a tree-based model,
    which can be explained by shapiq's TreeExplainer."""

    x_explain: np.ndarray
    """The feature vector to be explained."""

    def __init__(self, tree_model: object, x_explain: np.ndarray) -> None:
        """Initialize the TreeSHAP-IQ ground truth computer."""
        self.tree_model = tree_model
        self.x_explain = _make_x_vector(deepcopy(x_explain))

    def exact_values(self, index: TreeSHAPIQIndices, order: int) -> InteractionValues:
        """Compute the exact values using TreeSHAP-IQ."""
        explainer = TreeExplainer(model=self.tree_model, index=index, max_order=order, min_order=0)
        return explainer.explain(x=self.x_explain)


class InterventionalTreeComputer(GroundTruthComputer[TreeSHAPIQIndices]):
    """Ground truth computer for tree-based models using interventional TreeSHAP."""

    tree_model: object
    """The tree model to compute the exact values for. The tree model must be a tree-based model,
    which can be explained by intiq.InterventionalTreeExplainer."""

    x_explain: np.ndarray
    """The feature vector to be explained."""

    reference_data: np.ndarray
    """The reference data used for interventional explanations."""

    def __init__(
        self,
        tree_model: object,
        x_explain: np.ndarray,
        reference_data: np.ndarray,
        class_label: int | None = None,
    ) -> None:
        """Initialize the interventional TreeSHAP ground truth computer."""
        self.tree_model = tree_model
        self.x_explain = deepcopy(x_explain)
        self.reference_data = deepcopy(reference_data)
        self.class_label = class_label

    def exact_values(self, index: TreeSHAPIQIndices, order: int) -> InteractionValues:
        """Compute the exact values using interventional TreeSHAP."""
        # print(self.tree_model)
        explainer = InterventionalTreeExplainer(
            model=self.tree_model,
            index=index,
            max_order=order,
            data=self.reference_data,
            debug=False,
            class_index=self.class_label,
        )
        return explainer.explain_function_cii(x=self.x_explain, use_cython=False)


class TreeLocalXAI(Game):
    """A TreeSHAP-IQ explanation game for tree models.

    The game is based on the TreeSHAP-IQ algorithm and is used to explain the predictions of tree
    models. TreeSHAP-IQ is used to compute Shapley interaction values for tree models.
    """

    def __init__(
        self,
        x_explain: np.ndarray,
        tree_model: object,
        *,
        class_label: int | None = None,
        normalize: bool = True,
    ) -> None:
        """Initializes the TreeSHAP-IQ explanation game."""
        # get the TreeModel(s) from the tree model
        _trees = validate_tree_model(model=tree_model, class_label=class_label)
        if not isinstance(_trees, list):
            _trees = [_trees]
        self._trees: list[TreeModel] = deepcopy(_trees)

        # compute the output of the tree without any features present (empty coalition)
        empty_coalition = np.zeros(x_explain.shape[-1], dtype=bool)
        self._empty_value = compute_tree_output_from_coalition(
            coalition=empty_coalition, trees=self._trees, x_explain=x_explain
        )

        # set up x_explain
        self._x_explain = _make_x_vector(deepcopy(x_explain))

        super().__init__(
            n_players=self._x_explain.shape[-1],
            normalize=normalize,
            normalization_value=self._empty_value,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the output of the tree model for the given coalitions.

        The value function computes the output of a tree model provided different coalitions of
        features. The output is computed by traversing the tree with present and absent features.
        If a feature is present in the coalition, all splits in the tree using that feature are
        considered normally. If a feature is absent, the tree is traversed in both directions
        using both children of the split with the likelihood of going left or right derived from
        the training data (this is related to tree-dependent TreeSHAP / TreeSHAP-IQ).

        Args:
            coalitions: A binary matrix of feature subsets present (`True`) or absent (`False`).
                The matrix is expected to be in shape `(n_coalitions, n_features)`.

        Returns:
            The worth of the coalitions as a vector.
        """
        worth = np.zeros(len(coalitions), dtype=np.float32)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                worth[i] = self._empty_value
                continue
            worth[i] = compute_tree_output_from_coalition(coalition, self._trees, self._x_explain)
        return worth


def decision_function_smaller_equal(value: float, threshold: float, is_left_default: bool) -> bool:
    return (value <= threshold) or (np.isnan(value) and is_left_default)


def decision_function_smaller(value: float, threshold: float, is_left_default: bool) -> bool:
    return (value < threshold) or (np.isnan(value) and is_left_default)


def compute_tree_output_from_coalition(
    coalition: np.ndarray, trees: list[TreeModel], x_explain: np.ndarray
) -> float:
    """Computes the output of the tree models contained in the game.

    Args:
        coalition: A binary vector denoting the features present (`True`) or absent (`False`).
            The vector is expected to be in shape `(n_features,)`.
        trees: A list of tree models to compute the output for.
        x_explain: The feature vector which is to be explained with numerical feature values.

    Returns:
        The output of the tree models given the coalition of features.
    """
    output = 0.0
    for tree in trees:
        tree_prediction = _get_tree_prediction(
            node_id=int(tree.nodes[0]),
            tree=tree,
            coalition=coalition,
            x_explain=x_explain,
        )
        output += tree_prediction
    return output


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
    if tree.decision_type == "<=":
        decision_function = decision_function_smaller_equal
    elif tree.decision_type == "<":
        decision_function = decision_function_smaller
    else:
        msg = f"Unsupported decision type: {tree.decision_type}"
        raise ValueError(msg)
    if tree.leaf_mask[node_id]:  # end of recursion (base case, return the leaf prediction)
        return tree.values[node_id]
    # not a leaf we have to go deeper
    feature_id, threshold = tree.features[node_id], tree.thresholds[node_id]
    is_present = bool(coalition[feature_id])
    left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]
    if is_present:
        next_node = (
            left_child
            if decision_function(
                x_explain[feature_id], threshold, tree.children_left_default[node_id]
            )
            else right_child
        )
        tree_prediction = _get_tree_prediction(int(next_node), tree, coalition, x_explain)
    else:  # feature is out of coalition we have to go both ways and average the predictions
        prediction_left = _get_tree_prediction(int(left_child), tree, coalition, x_explain)
        prediction_right = _get_tree_prediction(int(right_child), tree, coalition, x_explain)
        # get weights (tree probabilities of going left or right)
        left_weight = tree.node_sample_weight[left_child]
        right_weight = tree.node_sample_weight[right_child]
        sum_of_weights = left_weight + right_weight

        tree_prediction = (
            left_weight * prediction_left + right_weight * prediction_right
        ) / sum_of_weights
        # # scale predictions
        # prediction_left *= left_weight / sum_of_weights
        # prediction_right *= right_weight / sum_of_weights
        # tree_prediction = prediction_left + prediction_right
    return tree_prediction


def _make_x_vector(x: np.ndarray) -> np.ndarray:
    """Flattens the input vector if it is a 2D array with a single row."""
    if x.ndim == 2 and x.shape[0] == 1:  # noqa: PLR2004
        x = x.flatten()
    elif x.ndim > 2 or (x.shape[0] > 1 and x.ndim > 1):  # noqa: PLR2004
        msg = (
            f"x_explain must be a 1-dimensional or 2-dimensional array of shape (1, n_features), "
            f"not {x.shape}."
        )
        raise ValueError(msg)
    return x
