"""Linear TreeShap Explainer Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.special as sp

from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion.edges import create_edge_tree
from shapiq.tree.validation import validate_tree_model
from shapiq.utils.sets import generate_interaction_lookup, powerset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import IntVector, Model


def get_norm_weight(M: int) -> np.ndarray:
    """Get normalization weights for Linear Tree Shap."""
    return np.array([sp.binom(M, i) for i in range(M + 1)])


def get_N_prime(max_size: int = 10) -> np.ndarray:
    """Get N' matrix for Linear Tree Shap."""
    N = np.zeros((max_size + 2, max_size + 2))
    for i in range(max_size + 2):
        N[i, : i + 1] = get_norm_weight(i)
    N_prime = np.zeros((max_size + 2, max_size + 2))
    for i in range(max_size + 2):
        N_prime[i, : i + 1] = N[: i + 1, : i + 1].dot(1 / N[i, : i + 1])
    return N_prime


def get_N_v2(D: np.ndarray) -> np.ndarray:
    """Get N_v2 matrix for Linear Tree Shap."""
    depth = D.shape[0]
    Ns = np.zeros((depth + 1, depth))
    for i in range(1, depth + 1):
        Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(1.0 / get_norm_weight(i - 1))
    return Ns


class LinearTreeSHAP:
    """Linear TreeShap Explainer for DecisionTree models.

    This class computes Shapley values for tree-based models using a linear approximation.
    It supports both iterative and C++ implementations for efficient computation.

    Attributes:
        clf: The tree-based model to explain.
        edge_tree: Edge-based representation of the tree for efficient traversal.
    """

    def __init__(
        self,
        model: Model,
        *,
        base_func: Callable[[int], np.ndarray] = np.polynomial.chebyshev.chebpts2,
    ) -> None:
        """Initialize the LinearTreeExplainer."""
        self.clf = model
        self._tree = validate_tree_model(model, class_label=None)[0]
        self._relevant_features: np.ndarray = np.array(list(self._tree.feature_ids), dtype=int)
        self._tree.reduce_feature_complexity()
        self._n_nodes: int = self._tree.n_nodes
        self._n_features_in_tree: int = self._tree.n_features_in_tree
        self._max_feature_id: int = self._tree.max_feature_id
        self._feature_ids: set = self._tree.feature_ids
        self._max_order = 1
        # precompute interaction lookup tables
        self._interactions_lookup_relevant: dict[tuple, int] = generate_interaction_lookup(
            self._relevant_features,
            0,
            1,
        )
        self._interactions_lookup: dict[int, dict[tuple, int]] = {}  # lookup for interactions
        self._interaction_update_positions: dict[int, dict[int, IntVector]] = {}  # lookup
        self._init_interaction_lookup_tables()

        self.edge_tree = create_edge_tree(
            children_left=self._tree.children_left,
            children_right=self._tree.children_right,
            features=self._tree.features,
            node_sample_weight=self._tree.node_sample_weight,
            values=self._tree.values,
            max_interaction=1,
            n_features=self._max_feature_id + 1,
            n_nodes=self._n_nodes,
            subset_updates_pos_store=self._interaction_update_positions,
        )
        self.N = get_N_prime(self.edge_tree.max_depth)
        self.Base = base_func(self.edge_tree.max_depth)
        self.Offset = np.vander(self.Base + 1).T[::-1]
        self.N_v2 = get_N_v2(self.Base)

    def _init_interaction_lookup_tables(self) -> None:
        """Initializes the lookup tables for the interaction subsets."""
        for order in range(1, self._max_order + 1):
            order_interactions_lookup = generate_interaction_lookup(
                self._n_features_in_tree,
                order,
                order,
            )
            self._interactions_lookup[order] = order_interactions_lookup
            _, interaction_update_positions = self._precompute_subsets_with_feature(
                interaction_order=order,
                n_features=self._n_features_in_tree,
                order_interactions_lookup=order_interactions_lookup,
            )
            self._interaction_update_positions[order] = interaction_update_positions

    @staticmethod
    def _precompute_subsets_with_feature(
        n_features: int,
        interaction_order: int,
        order_interactions_lookup: dict[tuple, int],
    ) -> tuple[dict[int, list[tuple]], dict[int, IntVector]]:
        """Precomputes the subsets of interactions that include a given feature.

        Args:
            n_features: The number of features in the model.
            interaction_order: The interaction order to be computed.
            order_interactions_lookup: The lookup table of interaction subsets to their positions
                in the interaction values array for a given interaction order (e.g. all 2-way
                interactions for order ``2``).

        Returns:
            interaction_updates: A dictionary (lookup table) containing the interaction subsets
                for each feature given an interaction order.
            interaction_update_positions: A dictionary (lookup table) containing the positions of
                the interaction subsets to update for each feature given an interaction order.

        """
        # stores interactions that include feature i (needs to be updated when feature i appears)
        interaction_updates: dict[int, list[tuple]] = {}
        # stores position of interactions that include feature i
        interaction_update_positions: dict[int, np.ndarray] = {}

        # prepare the interaction updates and positions
        for feature_i in range(n_features):
            positions = np.zeros(
                int(sp.binom(n_features - 1, interaction_order - 1)),
                dtype=int,
            )
            interaction_update_positions[feature_i] = positions.copy()
            interaction_updates[feature_i] = []

        # fill the interaction updates and positions
        position_counter = np.zeros(n_features, dtype=int)  # used to keep track of the position
        for interaction in powerset(
            range(n_features),
            min_size=interaction_order,
            max_size=interaction_order,
        ):
            for i in interaction:
                interaction_updates[i].append(interaction)
                position = position_counter[i]
                interaction_update_positions[i][position] = order_interactions_lookup[interaction]
                position_counter[i] += 1

        return interaction_updates, interaction_update_positions

    def shap_values_cpp_iterative(self, X: np.ndarray) -> np.ndarray:
        """Shapley Value computation using an Iterative C++ Implementation of LinearTreeShap.

        Args:
            X (np.ndarray): Datapoints

        Returns:
            np.ndarray: The computed shapley values
        """
        from .cext import (
            linear_tree_shap_iterative,  # ty: ignore[unresolved-import]
        )

        V = np.zeros_like(X, dtype=np.float64)
        V = np.ascontiguousarray(V)

        orig_feature_indices = np.array(
            [
                self._tree.feature_map_internal_original[i] if i != -2 else i
                for i in self._tree.features
            ],
            dtype=np.int32,
        )
        weights = 1 / self.edge_tree.p_e_values
        linear_tree_shap_iterative(
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(self.edge_tree.empty_predictions, dtype=np.float64),
            np.ascontiguousarray(self._tree.thresholds, dtype=np.float64),
            np.ascontiguousarray(self.edge_tree.ancestors, dtype=np.int32),
            np.ascontiguousarray(self.edge_tree.edge_heights, dtype=np.int32),
            np.ascontiguousarray(orig_feature_indices, dtype=np.int32),
            np.ascontiguousarray(self._tree.children_left, dtype=np.int32),
            np.ascontiguousarray(self._tree.children_right, dtype=np.int32),
            self.edge_tree.max_depth,
            self._tree.n_nodes,
            self.Base,
            np.ascontiguousarray(self.Offset, dtype=np.float64),
            np.ascontiguousarray(self.N_v2, dtype=np.float64),
            X.astype(np.float32),
            V,
        )
        return V

    def explain_function(self, x: np.ndarray) -> InteractionValues:
        """Computes the Shapley values for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.

        Returns:
            The interaction values for the instance.
        """
        if len(x.shape) != 1:
            x = x.flatten()
            msg = "explain expects a single instance, not a batch."
            raise TypeError(msg)
        shap_values = self.shap_values_cpp_iterative(x.reshape(1, -1)).flatten()
        shap_interactions: dict[tuple[int, ...], float] = {
            (feature,): float(shap_values[feature])
            for feature in range(x.shape[0])  # One entry per feature present in the tree
        }
        return InteractionValues(
            values=shap_interactions,
            baseline_value=(
                self._tree.empty_prediction
                if self._tree.empty_prediction is not None
                else np.sum(self.edge_tree.empty_predictions)
            ),
            min_order=0,
            max_order=1,
            index="SV",
            n_players=self._n_features_in_tree,
        )
