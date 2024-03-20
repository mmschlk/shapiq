"""This module contains the tree explainer implementation."""
import copy
from math import factorial
from typing import Any, Optional, Union

import numpy as np
from approximator import transforms_sii_to_ksii
from interaction_values import InteractionValues
from scipy.special import binom

from shapiq.utils import generate_interaction_lookup, powerset

from .base import EdgeTree, TreeModel
from .conversion.edges import create_edge_tree
from .validation import _validate_model


class TreeSHAPIQ:
    """
    The explainer for tree-based models using the TreeSHAP-IQ algorithm.

    Args:
        model: The tree-based model to explain as an explainer.tree.conversion.TreeModel.
        max_order: The maximum interaction order to be computed. An interaction order of 1
            corresponds to the Shapley value. Any value higher than 1 computes the Shapley
            interactions values up to that order. Defaults to 2.
        min_order: The minimum interaction order to be computed. Defaults to 1.
        interaction_type: The type of interaction to be computed. The interaction type can be
            "k-SII" (default) or "SII", "STI", "FSI", "BZF".
    """

    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        min_order: int = 1,
        interaction_type: str = "k-SII",
        verbose: bool = False,
    ) -> None:
        # set parameters
        self._root_node_id = 0
        self.verbose = verbose
        self._max_order: int = max_order
        self._min_order: int = min_order
        self._interaction_type: str = interaction_type

        # validate and parse model
        validated_model = _validate_model(model)  # the parsed and validated model
        # TODO: add support for other sample weights
        self._tree: TreeModel = copy.deepcopy(validated_model)
        self._relevant_features: np.ndarray = np.array(list(self._tree.feature_ids), dtype=int)
        self._tree.reduce_feature_complexity()
        self._n_nodes: int = self._tree.n_nodes
        self._n_features_in_tree: int = self._tree.n_features_in_tree
        self._max_feature_id: int = self._tree.max_feature_id
        self._feature_ids: set = self._tree.feature_ids

        # precompute interaction lookup tables
        self._interactions_lookup_relevant: dict[tuple, int] = generate_interaction_lookup(
            self._relevant_features, self._min_order, self._max_order
        )
        self._interactions_lookup: dict[int, dict[tuple, int]] = {}  # lookup for interactions
        self._interaction_update_positions: dict[int, dict[int, np.ndarray[int]]] = {}  # lookup
        self._init_interaction_lookup_tables()

        # get the edge representation of the tree
        edge_tree = create_edge_tree(
            children_left=self._tree.children_left,
            children_right=self._tree.children_right,
            features=self._tree.features,
            node_sample_weight=self._tree.node_sample_weight,
            values=self._tree.values,
            max_interaction=self._max_order,
            n_features=self._max_feature_id + 1,
            n_nodes=self._n_nodes,
            subset_updates_pos_store=self._interaction_update_positions,
        )
        self._edge_tree: EdgeTree = copy.deepcopy(edge_tree)

        # compute the empty prediction
        self.empty_prediction: float = float(
            np.sum(self._edge_tree.empty_predictions[self._tree.leaf_mask])
        )

        # stores the interaction scores up to a given order
        self.subset_ancestors_store: dict = {}
        self.D_store: dict = {}
        self.D_powers_store: dict = {}
        self.Ns_id_store: dict = {}
        self.Ns_store: dict = {}
        self.n_interpolation_size = self._n_features_in_tree
        if self._interaction_type in ("SII", "k-SII"):  # SP is of order at most d_max
            self.n_interpolation_size = min(self._edge_tree.max_depth, self._n_features_in_tree)
        self._init_summary_polynomials()

        # stores the nodes that are active in the tree for a given instance (new for each instance)
        self._activations: np.ndarray[bool] = np.zeros(self._n_nodes, dtype=bool)

        # print tree information
        if self.verbose:
            self._print_tree_info()

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        """Computes the Shapley Interaction values for a given instance x and interaction order.
            This function is the main explanation function of this class.

        Args:
            x_explain (np.ndarray): Instance to be explained.

        Returns:
            InteractionValues: The computed Shapley Interaction values.
        """
        x_explain_relevant = x_explain[self._relevant_features]
        n_players = max(x_explain.shape[0], self._n_features_in_tree)

        # compute the Shapley Interaction values
        interactions = np.asarray([], dtype=float)
        for order in range(self._min_order, self._max_order + 1):
            self.shapley_interactions = np.zeros(
                int(binom(self._n_features_in_tree, order)), dtype=float
            )
            self._prepare_variables_for_order(interaction_order=order)
            self._compute_shapley_interaction_values(x_explain_relevant, order=order, node_id=0)
            # append the computed Shapley Interaction values to the result
            interactions = np.append(interactions, self.shapley_interactions.copy())

        shapley_interaction_values = InteractionValues(
            values=interactions,
            index=self._interaction_type,
            min_order=self._min_order,
            max_order=self._max_order,
            n_players=n_players,
            estimated=False,
            interaction_lookup=self._interactions_lookup_relevant,
        )

        if self._interaction_type == "k-SII":
            shapley_interaction_values = transforms_sii_to_ksii(shapley_interaction_values)

        return shapley_interaction_values

    def _compute_shapley_interaction_values(
        self,
        x_explain: np.ndarray,
        order: int = 1,
        node_id: int = 0,
        *,
        summary_poly_down: np.ndarray[float] = None,
        summary_poly_up: np.ndarray[float] = None,
        interaction_poly_down: np.ndarray[float] = None,
        quotient_poly_down: np.ndarray[float] = None,
        depth: int = 0,
    ) -> None:
        # reset activations for new calculations
        if node_id == 0:
            self._activations.fill(False)

        # get polynomials if None
        polynomials = self._get_polynomials(
            order=order,
            summary_poly_down=summary_poly_down,
            summary_poly_up=summary_poly_up,
            interaction_poly_down=interaction_poly_down,
            quotient_poly_down=quotient_poly_down,
        )
        summary_poly_down, summary_poly_up, interaction_poly_down, quotient_poly_down = polynomials

        # get related nodes (surrounding) nodes
        left_child = int(self._tree.children_left[node_id])
        right_child = int(self._tree.children_right[node_id])
        parent_id = int(self._edge_tree.parents[node_id])
        ancestor_id = int(self._edge_tree.ancestors[node_id])

        # get feature information
        feature_id = int(self._tree.features[parent_id])
        feature_threshold = self._tree.thresholds[node_id]
        child_edge_feature = self._tree.features[node_id]

        # get height of related nodes
        current_height = int(self._edge_tree.edge_heights[node_id])
        left_height = int(self._edge_tree.edge_heights[left_child])
        right_height = int(self._edge_tree.edge_heights[right_child])

        # get path information
        is_leaf = bool(self._tree.leaf_mask[node_id])
        has_ancestor = bool(self._edge_tree.has_ancestors[node_id])
        activations = self._activations

        # if feature_id > -1:
        try:
            interaction_sets = self.subset_updates_pos[feature_id]
        except KeyError:
            interaction_sets = np.array([], dtype=int)

        # if node is not a leaf -> set activations for children nodes accordingly
        if not is_leaf:
            if x_explain[child_edge_feature] <= feature_threshold:
                activations[left_child], activations[right_child] = True, False
            else:
                activations[left_child], activations[right_child] = False, True

        # if node is not the root node -> calculate the summary polynomials
        if node_id != self._root_node_id:
            # set activations of current node in relation to the ancestor (for setting p_e to zero)
            if has_ancestor:
                activations[node_id] &= activations[ancestor_id]
            # if node is active get the correct p_e value
            p_e_current = self._edge_tree.p_e_values[node_id] if activations[node_id] else 0.0
            # update summary polynomial
            summary_poly_down[depth] = summary_poly_down[depth - 1] * (self.D + p_e_current)
            # update quotient polynomials
            quotient_poly_down[depth, :] = quotient_poly_down[depth - 1, :].copy()
            quotient_poly_down[depth, interaction_sets] = quotient_poly_down[
                depth, interaction_sets
            ] * (self.D + p_e_current)
            # update interaction polynomial
            interaction_poly_down[depth, :] = interaction_poly_down[depth - 1, :].copy()
            interaction_poly_down[depth, interaction_sets] = interaction_poly_down[
                depth, interaction_sets
            ] * (-self.D + p_e_current)
            # remove previous polynomial factor if node has ancestors
            if has_ancestor:
                p_e_ancestor = 0.0
                if activations[ancestor_id]:
                    p_e_ancestor = self._edge_tree.p_e_values[ancestor_id]
                summary_poly_down[depth] = summary_poly_down[depth] / (self.D + p_e_ancestor)
                quotient_poly_down[depth, interaction_sets] = quotient_poly_down[
                    depth, interaction_sets
                ] / (self.D + p_e_ancestor)
                interaction_poly_down[depth, interaction_sets] = interaction_poly_down[
                    depth, interaction_sets
                ] / (-self.D + p_e_ancestor)

        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        if is_leaf:  # recursion base case
            summary_poly_up[depth] = (
                summary_poly_down[depth] * self._edge_tree.empty_predictions[node_id]
            )
        else:  # not a leaf -> continue recursion
            # left child
            self._compute_shapley_interaction_values(
                x_explain,
                order=order,
                node_id=left_child,
                summary_poly_down=summary_poly_down,
                summary_poly_up=summary_poly_up,
                interaction_poly_down=interaction_poly_down,
                quotient_poly_down=quotient_poly_down,
                depth=depth + 1,
            )
            summary_poly_up[depth] = (
                summary_poly_up[depth + 1] * self.D_powers[current_height - left_height]
            )
            # right child
            self._compute_shapley_interaction_values(
                x_explain,
                order=order,
                node_id=right_child,
                summary_poly_down=summary_poly_down,
                summary_poly_up=summary_poly_up,
                interaction_poly_down=interaction_poly_down,
                quotient_poly_down=quotient_poly_down,
                depth=depth + 1,
            )
            summary_poly_up[depth] += (
                summary_poly_up[depth + 1] * self.D_powers[current_height - right_height]
            )

        # if node is not the root node -> calculate the Shapley Interaction values for the node
        if node_id is not self._root_node_id:
            interactions_seen = interaction_sets[
                self.interaction_height[node_id][interaction_sets] == order
            ]
            if len(interactions_seen) > 0:
                D_power = self.D_powers[0]
                index_quotient = current_height - order
                if self._interaction_type not in ("SII", "k-SII"):
                    D_power = self.D_powers[self._n_features_in_tree - current_height]
                    index_quotient = self._n_features_in_tree - order
                interaction_update = np.dot(
                    interaction_poly_down[depth, interactions_seen],
                    self.Ns_id[self.n_interpolation_size, : self.n_interpolation_size],
                )
                interaction_update *= self._psi(
                    summary_poly_up[depth, :],  # TODO fix: wrong at this point (should not be zero)
                    D_power,
                    quotient_poly_down[depth, interactions_seen],
                    self.Ns,
                    index_quotient,
                )
                self.shapley_interactions[interactions_seen] += interaction_update
            # if node has ancestors -> adjust the Shapley Interaction values for the node
            ancestor_node_id = self.subset_ancestors[node_id][interaction_sets]  # inter. ancestors
            if np.mean(ancestor_node_id) > -1:  # if there exist ancestors # TODO why mean?
                ancestor_node_id_exists = ancestor_node_id > -1
                interactions_with_ancestor = interaction_sets[ancestor_node_id_exists]
                cond_interaction_seen = (
                    self.interaction_height[parent_id][interactions_with_ancestor] == order
                )
                interactions_ancestors = ancestor_node_id[
                    ancestor_node_id_exists
                ]  # ancestors of interactions with ancestor
                interactions_with_ancestor_to_update = interactions_with_ancestor[
                    cond_interaction_seen
                ]
                if len(interactions_with_ancestor_to_update) > 0:
                    ancestor_heights = self._edge_tree.edge_heights[
                        interactions_ancestors[cond_interaction_seen]
                    ]
                    D_power = self.D_powers[ancestor_heights - current_height].astype(int)
                    index_quotient = ancestor_heights - order
                    if self._interaction_type not in ("SII", "k-SII"):
                        D_power = self.D_powers[self._n_features_in_tree - current_height]
                        index_quotient = self._n_features_in_tree - order
                    update = np.dot(
                        interaction_poly_down[depth - 1, interactions_with_ancestor_to_update],
                        self.Ns_id[self.n_interpolation_size, : self.n_interpolation_size],
                    )
                    update *= self._psi(
                        summary_poly_up[depth],
                        D_power,
                        quotient_poly_down[depth - 1, interactions_with_ancestor_to_update],
                        self.Ns,
                        int(index_quotient),
                    )
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= update

    @staticmethod
    def _psi(E, D_power, quotient_poly, Ns, degree):
        # TODO: add docstring
        d = degree + 1
        n = Ns[d, :d]
        return ((E * D_power / quotient_poly)[:, :d]).dot(n) / d

    def _init_summary_polynomials(self):
        """Initializes summary polynomial variables. This function is called once during the
        initialization of the explainer."""
        for order in range(1, self._max_order + 1):
            subset_ancestors: dict[int, np.ndarray] = self._precalculate_interaction_ancestors(
                interaction_order=order, n_features=self._n_features_in_tree
            )
            self.subset_ancestors_store[order] = subset_ancestors
            self.D_store[order] = np.polynomial.chebyshev.chebpts2(self.n_interpolation_size)
            self.D_powers_store[order] = self._cache(self.D_store[order])
            if self._interaction_type in ("SII", "k-SII"):
                self.Ns_store[order] = self._get_N(self.D_store[order])
            else:
                self.Ns_store[order] = self._get_N_cii(self.D_store[order], order)
            self.Ns_id_store[order] = self._get_N_id(self.D_store[order])

    def _get_polynomials(
        self,
        order: int,
        summary_poly_down: np.ndarray[float] = None,
        summary_poly_up: np.ndarray[float] = None,
        interaction_poly_down: np.ndarray[float] = None,
        quotient_poly_down: np.ndarray[float] = None,
    ):
        if summary_poly_down is None:
            summary_poly_down = np.zeros((self._edge_tree.max_depth + 1, self.n_interpolation_size))
            summary_poly_down[0, :] = 1
        if summary_poly_up is None:
            summary_poly_up = np.zeros((self._edge_tree.max_depth + 1, self.n_interpolation_size))
        if interaction_poly_down is None:
            interaction_poly_down = np.zeros(
                (
                    self._edge_tree.max_depth + 1,
                    int(binom(self._n_features_in_tree, order)),
                    self.n_interpolation_size,
                )
            )
            interaction_poly_down[0, :] = 1
        if quotient_poly_down is None:
            quotient_poly_down = np.zeros(
                (
                    self._edge_tree.max_depth + 1,
                    int(binom(self._n_features_in_tree, order)),
                    self.n_interpolation_size,
                )
            )
            quotient_poly_down[0, :] = 1
        return summary_poly_down, summary_poly_up, interaction_poly_down, quotient_poly_down

    def _prepare_variables_for_order(self, interaction_order: int):
        """Retrieves the precomputed variables for a given interaction order. This function is
            called before the recursive explanation function is called.

        Args:
            interaction_order (int): The interaction order for which the storage variables should be
                loaded.
        """
        self.subset_updates_pos = self._interaction_update_positions[interaction_order]
        self.subset_ancestors = self.subset_ancestors_store[interaction_order]
        self.D = self.D_store[interaction_order]
        self.D_powers = self.D_powers_store[interaction_order]
        self.interaction_height = self._edge_tree.interaction_height_store[interaction_order]
        self.Ns_id = self.Ns_id_store[interaction_order]
        self.Ns = self.Ns_store[interaction_order]

    def _init_interaction_lookup_tables(self) -> None:
        """Initializes the lookup tables for the interaction subsets."""
        for order in range(1, self._max_order + 1):
            order_interactions_lookup = generate_interaction_lookup(
                self._n_features_in_tree, order, order
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
        n_features: int, interaction_order: int, order_interactions_lookup: dict[tuple, int]
    ) -> tuple[dict[int, list[tuple]], dict[int, np.ndarray[int]]]:
        """Precomputes the subsets of interactions that include a given feature.

        Args:
            n_features: The number of features in the model.
            interaction_order: The interaction order to be computed.
            order_interactions_lookup: The lookup table of interaction subsets to their positions
                in the interaction values array for a given interaction order (e.g. all 2-way
                interactions for order 2).

        Returns:
            interaction_updates: A dictionary (lookup table) containing the interaction subsets
                for each feature given an interaction order.
            interaction_update_positions: A dictionary (lookup table) containing the positions of
                the interaction subsets to update for each feature given an interaction order.
        """
        # stores interactions that include feature i (needs to be updated when feature i appears)
        interaction_updates: dict[int, list[tuple]] = {}
        # stores position of interactions that include feature i
        interaction_update_positions: dict[int, np.ndarray[int]] = {}

        # prepare the interaction updates and positions
        for feature_i in range(n_features):
            positions = np.zeros(int(binom(n_features - 1, interaction_order - 1)), dtype=int)
            interaction_update_positions[feature_i] = positions.copy()
            interaction_updates[feature_i] = []

        # fill the interaction updates and positions
        position_counter = np.zeros(n_features, dtype=int)  # used to keep track of the position
        for interaction in powerset(
            range(n_features), min_size=interaction_order, max_size=interaction_order
        ):
            for i in interaction:
                interaction_updates[i].append(interaction)
                position = position_counter[i]
                interaction_update_positions[i][position] = order_interactions_lookup[interaction]
                position_counter[i] += 1

        return interaction_updates, interaction_update_positions

    def _precalculate_interaction_ancestors(
        self, interaction_order, n_features
    ) -> dict[int, np.ndarray]:
        """Calculates the position of the ancestors of the interactions for the tree for a given
        order of interactions."""

        # stores position of interactions
        counter_interaction = 0
        subset_ancestors: dict[int, np.ndarray[int]] = {}

        for node_id in self._tree.nodes[1:]:  # for all nodes except the root node
            subset_ancestors[node_id] = np.full(
                int(binom(n_features, interaction_order)), -1, dtype=int
            )
        for S in powerset(range(n_features), interaction_order, interaction_order):
            # self.shapley_interactions_lookup[S] = counter_interaction
            for node_id in self._tree.nodes[1:]:  # for all nodes except the root node
                subset_ancestor = -1
                for i in S:
                    subset_ancestor = max(
                        subset_ancestor, self._edge_tree.ancestor_nodes[node_id][i]
                    )
                subset_ancestors[node_id][counter_interaction] = subset_ancestor
            counter_interaction += 1
        return subset_ancestors

    def _get_N(self, interpolated_poly: np.ndarray[float]) -> np.ndarray[float]:
        # TODO: test/validate docstring
        """
        Computes the N matrix for the Shapley interaction values.

        Args:
            interpolated_poly: The interpolated polynomial. (D in old implementation)

        Returns:
            The N matrix.
        """
        depth = interpolated_poly.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(interpolated_poly[:i]).T).dot(
                1.0 / self._get_norm_weight(i - 1)
            )
        return Ns

    def _get_N_cii(self, interpolated_poly, order) -> np.ndarray[float]:
        depth = interpolated_poly.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(interpolated_poly[:i]).T).dot(
                i * np.array([self._get_subset_weight_cii(j, order) for j in range(i)])
            )
        return Ns

    def _get_subset_weight_cii(self, t, order) -> Optional[float]:
        # TODO: add docstring
        if self._interaction_type == "STI":
            return self._max_order / (
                self._n_features_in_tree * binom(self._n_features_in_tree - 1, t)
            )
        if self._interaction_type == "FSI":
            return (
                factorial(2 * self._max_order - 1)
                / factorial(self._max_order - 1) ** 2
                * factorial(self._max_order + t - 1)
                * factorial(self._n_features_in_tree - t - 1)
                / factorial(self._n_features_in_tree + self._max_order - 1)
            )
        if self._interaction_type == "BZF":
            return 1 / (2 ** (self._n_features_in_tree - order))

    @staticmethod
    def _get_N_id(D) -> np.ndarray[float]:
        # TODO: add docstring and rename variables
        depth = D.shape[0]
        Ns_id = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns_id[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(np.ones(i))  # TODO remove ()
        return Ns_id

    @staticmethod
    def _get_norm_weight(M) -> np.ndarray[float]:
        # TODO: add docstring and rename variables
        return np.array([binom(M, i) for i in range(M + 1)])

    @staticmethod
    def _cache(interpolated_poly: np.ndarray[float]) -> np.ndarray[float]:
        """Caches the powers of the interpolated polynomial.

        Args:
            interpolated_poly: The interpolated polynomial. (D in old implementation)

        Returns:
            The cached powers of the interpolated polynomial.
        """
        return np.vander(interpolated_poly + 1).T[::-1]

    def _print_tree_info(self) -> None:
        """Prints information about the tree to be explained."""
        information = "Tree information:"
        information += f"\nNumber of nodes: {self._n_nodes}"
        information += f"\nNumber of features: {self._n_features_in_tree}"
        information += f"\nMaximum interaction order: {self._max_order}"
        information += f"\nInteraction type: {self._interaction_type}"
        # add empty prediction from _tree and self to information TODO: remove one in final
        information += f"\nEmpty prediction (from _tree): {self._tree.empty_prediction}"
        information += f"\nEmpty prediction (from self): {self.empty_prediction}"
        print(information)
