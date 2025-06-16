"""Implementation of the tree explainer."""

from __future__ import annotations

import copy
from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy as sp

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from shapiq.utils.sets import generate_interaction_lookup, powerset

from .conversion.edges import create_edge_tree
from .validation import validate_tree_model

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model

    from .base import EdgeTree, TreeModel


TreeSHAPIQIndices = Literal["SV", "SII", "k-SII"]


class TreeSHAPIQ:
    """The TreeSHAP-IQ computation class.

    This class implements the TreeSHAP-IQ algorithm for computing Shapley Interaction values for
    tree-based models. It is used internally by the
    :class:`~shapiq.explainer.tree.explainer.TreeExplainer`. The TreeSHAP-IQ algorithm is presented
    in `Muschalik et al. (2024)` [Mus24]_.

    TreeSHAP-IQ is an algorithm for computing Shapley Interaction values for tree-based models.
    It is based on the Linear TreeSHAP algorithm by `Yu et al. (2022)` [Yu22]_, but extended to
    compute Shapley Interaction values up to a given order. TreeSHAP-IQ needs to visit each node
    only once and makes use of polynomial arithmetic to compute the Shapley Interaction values
    efficiently.

    Note:
        This class is not intended to be used directly. Instead, use the ``TreeExplainer`` class to
        explain tree-based models which internally uses then the TreeSHAP-IQ algorithm.

    References:
        .. [Yu22] Peng Yu, Chao Xu, Albert Bifet, Jesse Read Linear Tree Shap (2022). In: Proceedings of 36th Conference on Neural Information Processing Systems. https://openreview.net/forum?id=OzbkiUo24g
        .. [Mus24] Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, & Eyke Hüllermeier (2024). Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles. In: Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14388-14396. https://doi.org/10.1609/aaai.v38i13.29352

    """

    def __init__(
        self,
        model: dict | TreeModel | Model,
        *,
        max_order: int = 2,
        min_order: int = 1,
        index: TreeSHAPIQIndices = "k-SII",
        verbose: bool = False,
    ) -> None:
        """Initializes the TreeSHAP-IQ explainer.

        Args:
            model: A single tree model to explain. Note that unlike the
                :class:`~shapiq.explainer.tree.explainer.TreeExplainer` class, TreeSHAP-IQ only
                supports a single tree. It can be a dictionary representation of the tree, a
                :class:`~shapiq.explainer.tree.base.TreeModel` object, or any other single tree
                model supported by the :meth:`~shapiq.explainer.tree.validation.validate_tree_model`
                function.

            max_order: The maximum interaction order to be computed. An interaction order of ``1``
                corresponds to the Shapley value. Any value higher than ``1`` computes the Shapley
                interaction values up to that order. Defaults to ``2``.

            min_order: The minimum interaction order to be computed. Defaults to ``1``. Note that
                setting min_order currently does not have any effect on the computation.

            index: The type of interaction to be computed.

            verbose: Whether to print information about the tree during initialization. Defaults to
                ``False``.

        """
        # set parameters
        self._root_node_id = 0
        self.verbose = verbose
        if max_order < min_order or max_order < 1 or min_order < 1:
            msg = (
                "The maximum order must be greater than the minimum order and both must be greater "
                "than 0."
            )
            raise ValueError(msg)
        self._max_order: int = max_order
        self._min_order: int = min_order
        self._index: str = index
        self._base_index: str = get_computation_index(self._index)

        # validate and parse model
        validated_model = validate_tree_model(model)  # the parsed and validated model
        # TODO(mmshlk): add support for other sample weights https://github.com/mmschlk/shapiq/issues/99
        self._tree: TreeModel = copy.deepcopy(validated_model)
        self._relevant_features: np.ndarray = np.array(list(self._tree.feature_ids), dtype=int)
        self._tree.reduce_feature_complexity()
        self._n_nodes: int = self._tree.n_nodes
        self._n_features_in_tree: int = self._tree.n_features_in_tree
        self._max_feature_id: int = self._tree.max_feature_id
        self._feature_ids: set = self._tree.feature_ids

        # precompute interaction lookup tables
        self._interactions_lookup_relevant: dict[tuple, int] = generate_interaction_lookup(
            self._relevant_features,
            self._min_order,
            self._max_order,
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
        computed_empty_prediction = float(
            np.sum(self._edge_tree.empty_predictions[self._tree.leaf_mask]),
        )
        tree_empty_prediction = self._tree.empty_prediction
        if tree_empty_prediction is None:
            tree_empty_prediction = computed_empty_prediction
        self.empty_prediction: float = tree_empty_prediction

        # stores the interaction scores up to a given order
        self.subset_ancestors_store: dict = {}
        self.D_store: dict = {}
        self.D_powers_store: dict = {}
        self.Ns_id_store: dict = {}
        self.Ns_store: dict = {}
        self.n_interpolation_size = self._n_features_in_tree
        if self._index in ("SV", "SII", "k-SII"):  # SP is of order at most d_max
            self.n_interpolation_size = min(self._edge_tree.max_depth, self._n_features_in_tree)
        try:
            self._init_summary_polynomials()
            self._trivial_computation = False
        except ValueError:
            if self._n_features_in_tree == 1:
                self._trivial_computation = True  # for one feature the computation is trivial
            else:
                raise

        # stores the nodes that are active in the tree for a given instance (new for each instance)
        self._activations: np.ndarray = np.zeros(self._n_nodes, dtype=bool)

        # print tree information
        if self.verbose:
            self._print_tree_info()

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Computes the Shapley Interaction values for a given instance ``x`` and interaction order.

        Note:
            This function is the main explanation function of this class.

        Args:
            x (np.ndarray): Instance to be explained.

        Returns:
            InteractionValues: The computed Shapley Interaction values.

        """
        x_relevant = x[self._relevant_features]
        n_players = max(x.shape[0], self._n_features_in_tree)

        if self._trivial_computation:
            interactions = self._compute_trivial_shapley_interaction_values(x)
        else:
            # compute the Shapley Interaction values
            interactions = np.asarray([], dtype=float)
            for order in range(self._min_order, self._max_order + 1):
                shapley_interactions = np.zeros(
                    int(sp.special.binom(self._n_features_in_tree, order)),
                    dtype=float,
                )
                self.shapley_interactions = shapley_interactions
                self._prepare_variables_for_order(interaction_order=order)
                self._compute_shapley_interaction_values(x_relevant, order=order, node_id=0)
                # append the computed Shapley Interaction values to the result
                interactions = np.append(interactions, self.shapley_interactions.copy())

        shapley_interaction_values = InteractionValues(
            values=interactions,
            index=self._base_index,
            min_order=self._min_order,
            max_order=self._max_order,
            n_players=n_players,
            estimated=False,
            interaction_lookup=self._interactions_lookup_relevant,
            baseline_value=self.empty_prediction,
        )

        return finalize_computed_interactions(
            shapley_interaction_values,
            target_index=self._index,
        )

    def _compute_trivial_shapley_interaction_values(self, x: np.ndarray) -> np.ndarray:
        """Computes the Shapley interactions for the case of only one feature in the tree.

        Computing the Shapley interactions for the case of only one feature in the tree is trivial
        since only the main effect of this feature is considered, i.e., the first order value of the
        single feature gets the full effect and all higher order values are zero.

        Args:
            x: The original instance to be explained.

        Returns:
            np.ndarray: The computed Shapley Interaction values.

        """
        full_prediction = self._tree.predict_one(x)
        main_effect = full_prediction - self.empty_prediction
        shapley_interactions = np.zeros(1, dtype=float)
        shapley_interactions[0] = main_effect
        return shapley_interactions

    def _compute_shapley_interaction_values(
        self,
        x: np.ndarray,
        order: int = 1,
        node_id: int = 0,
        *,
        summary_poly_down: np.ndarray[float] = None,
        summary_poly_up: np.ndarray[float] = None,
        interaction_poly_down: np.ndarray[float] = None,
        quotient_poly_down: np.ndarray[float] = None,
        depth: int = 0,
    ) -> None:
        """Computes the Shapley Interaction values for a given instance x and interaction order.

        Note:
            This function is called recursively for each node in the tree.

        Args:
            x: The instance to be explained.

            order: The interaction order for which the Shapley Interaction values should be
                computed. Defaults to ``1``.

            node_id: The node ID of the current node in the tree. Defaults to ``0``.

            summary_poly_down: The summary polynomial for the current node. Defaults to ``None``
                (at init time).

            summary_poly_up: The summary polynomial propagated up the tree. Defaults to ``None``
                (at init time).

            interaction_poly_down: The interaction polynomial for the current node. Defaults to
                ``None`` (at init time).

            quotient_poly_down: The quotient polynomial for the current node. Defaults to ``None``
                (at init time).

            depth: The depth of the current node in the tree. Defaults to ``0``.

        """
        # fmt: off
        # manually formatted for better readability in formulas and equations
        # reset activations for new calculations
        if node_id == 0:
            self._activations.fill(False)  # noqa: FBT003

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
            if x[child_edge_feature] <= feature_threshold:
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
            quotient_poly_down[depth, interaction_sets] = quotient_poly_down[depth, interaction_sets] * (self.D + p_e_current)
            # update interaction polynomial
            interaction_poly_down[depth, :] = interaction_poly_down[depth - 1, :].copy()
            interaction_poly_down[depth, interaction_sets] = interaction_poly_down[depth, interaction_sets] * (-self.D + p_e_current)
            # remove previous polynomial factor if node has ancestors
            if has_ancestor:
                p_e_ancestor = 0.0
                if activations[ancestor_id]:
                    p_e_ancestor = self._edge_tree.p_e_values[ancestor_id]
                # rescale the polynomials
                summary_poly_down[depth] = summary_poly_down[depth] / (self.D + p_e_ancestor)
                quotient_poly_down[depth, interaction_sets] = quotient_poly_down[depth, interaction_sets] / (self.D + p_e_ancestor)
                interaction_poly_down[depth, interaction_sets] = interaction_poly_down[depth, interaction_sets] / (-self.D + p_e_ancestor)

        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        if is_leaf:  # recursion base case
            summary_poly_up[depth] = (
                summary_poly_down[depth] * self._edge_tree.empty_predictions[node_id]
            )
        else:  # not a leaf -> continue recursion
            # left child
            self._compute_shapley_interaction_values(
                x,
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
                x,
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
                self._int_height[node_id][interaction_sets] == order
            ]
            if len(interactions_seen) > 0:
                if self._index not in ("SV", "SII", "k-SII"):  # for CII
                    D_power = self.D_powers[self._n_features_in_tree - current_height]
                    index_quotient = self._n_features_in_tree - order
                else:  # for SII and k-SII
                    D_power = self.D_powers[0]
                    index_quotient = current_height - order
                interaction_update = np.dot(
                    interaction_poly_down[depth, interactions_seen],
                    self.Ns_id[self.n_interpolation_size, : self.n_interpolation_size],
                )
                interaction_update *= self._psi(
                    summary_poly_up[depth, :],
                    D_power,
                    quotient_poly_down[depth, interactions_seen],
                    self.Ns,
                    index_quotient,
                )
                self.shapley_interactions[interactions_seen] += interaction_update

            # if node has ancestors -> adjust the Shapley Interaction values for the node
            ancestors_of_interactions = self.subset_ancestors[node_id][interaction_sets]
            if np.any(ancestors_of_interactions > -1):  # at least one ancestor exists (not -1)
                ancestor_node_id_exists = ancestors_of_interactions > -1  # get mask of ancestors
                interactions_with_ancestor = interaction_sets[ancestor_node_id_exists]
                cond_interaction_seen = (
                    self._int_height[parent_id][interactions_with_ancestor] == order
                )
                interactions_ancestors = ancestors_of_interactions[ancestor_node_id_exists]
                interactions_with_ancestor_to_update = interactions_with_ancestor[
                    cond_interaction_seen
                ]
                if len(interactions_with_ancestor_to_update) > 0:
                    ancestor_heights = self._edge_tree.edge_heights[
                        interactions_ancestors[cond_interaction_seen]
                    ]
                    if self._index not in ("SV", "SII", "k-SII"):  # for CII
                        D_power = self.D_powers[self._n_features_in_tree - current_height]
                        index_quotient = self._n_features_in_tree - order
                    else:  # for SII and k-SII
                        D_power = self.D_powers[ancestor_heights - current_height]
                        index_quotient = ancestor_heights - order
                    update = np.dot(
                        interaction_poly_down[depth - 1, interactions_with_ancestor_to_update],
                        self.Ns_id[self.n_interpolation_size, : self.n_interpolation_size],
                    )
                    to_update = self._psi_ancestor(
                        summary_poly_up[depth],
                        D_power,
                        quotient_poly_down[depth - 1, interactions_with_ancestor_to_update],
                        self.Ns,
                        index_quotient,
                    )
                    if to_update.shape == (1, 1):
                        update *= to_update[0]  # cast out shape of (1, 1) to float
                    else:
                        update *= to_update  # something errors here for CII
                    # fmt: on
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= update

    @staticmethod
    def _psi_ancestor(
        E: np.ndarray,
        D_power: np.ndarray,
        quotient_poly: np.ndarray,
        Ns: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """Similar to _psi but with ancestors."""
        d = degree + 1
        n = Ns[d].T  # Variant of _psi that can deal with multiple inputs in degree
        return np.diag((E * D_power / quotient_poly).dot(n)) / (d)

    @staticmethod
    def _psi(
        E: np.ndarray,
        D_power: np.ndarray,
        quotient_poly: np.ndarray,
        Ns: np.ndarray,
        degree: int,
    ) -> np.ndarray[float]:
        """Computes the psi function for the TreeSHAP-IQ algorithm.

        It scales the interaction polynomials with the summary polynomial and the quotient
        polynomial. For details, refer to `Muschalik et al. (2024) <https://doi.org/10.48550/arXiv.2401.12069>`_.

        Args:
            E: The summary polynomial.
            D_power: The power of the D polynomial.
            quotient_poly: The quotient polynomial.
            Ns: The Ns polynomial.
            degree: The degree of the interaction polynomial.

        Returns:
            np.ndarray: The computed psi function.

        """
        d = degree + 1
        n = Ns[d, :d]
        return ((E * D_power / quotient_poly)[:, :d]).dot(n) / d

    def _init_summary_polynomials(self) -> None:
        """Initializes the summary polynomial variables.

        Note:
            This function is called once during the initialization of the explainer.
        """
        for order in range(1, self._max_order + 1):
            subset_ancestors: dict[int, np.ndarray] = self._precalculate_interaction_ancestors(
                interaction_order=order,
                n_features=self._n_features_in_tree,
            )
            self.subset_ancestors_store[order] = subset_ancestors

            # If the tree has only one feature, we assign a default value of 0
            self.D_store[order] = np.polynomial.chebyshev.chebpts2(self.n_interpolation_size)

            self.D_powers_store[order] = self._cache(self.D_store[order])
            if self._index in ("SV", "SII", "k-SII"):
                self.Ns_store[order] = self._get_n_matrix(self.D_store[order])
            else:
                self.Ns_store[order] = self._get_n_cii_matrix(self.D_store[order], order)
            self.Ns_id_store[order] = self._get_n_id_matrix(self.D_store[order])

    def _get_polynomials(
        self,
        order: int,
        summary_poly_down: np.ndarray[float] | None = None,
        summary_poly_up: np.ndarray[float] | None = None,
        interaction_poly_down: np.ndarray[float] | None = None,
        quotient_poly_down: np.ndarray[float] | None = None,
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """Retrieves the polynomials for a given interaction order.

        This function initializes the polynomials for the first call of the recursive explanation
        function.

        Args:
            order: The interaction order for which the polynomials should be loaded.

            summary_poly_down: The summary polynomial for the current node. Defaults to ``None``.

            summary_poly_up: The summary polynomial propagated up the tree. Defaults to ``None``.

            interaction_poly_down: The interaction polynomial for the current node. Defaults to
                ``None``.

            quotient_poly_down: The quotient polynomial for the current node. Defaults to ``None``.

        Returns:
            The summary polynomial down, the summary polynomial up, the interaction polynomial down,
                and the quotient polynomial down.

        """
        if summary_poly_down is None:
            summary_poly_down = np.zeros((self._edge_tree.max_depth + 1, self.n_interpolation_size))
            summary_poly_down[0, :] = 1
        if summary_poly_up is None:
            summary_poly_up = np.zeros((self._edge_tree.max_depth + 1, self.n_interpolation_size))
        if interaction_poly_down is None:
            interaction_poly_down = np.zeros(
                (
                    self._edge_tree.max_depth + 1,
                    int(sp.special.binom(self._n_features_in_tree, order)),
                    self.n_interpolation_size,
                ),
            )
            interaction_poly_down[0, :] = 1
        if quotient_poly_down is None:
            quotient_poly_down = np.zeros(
                (
                    self._edge_tree.max_depth + 1,
                    int(sp.special.binom(self._n_features_in_tree, order)),
                    self.n_interpolation_size,
                ),
            )
            quotient_poly_down[0, :] = 1
        return summary_poly_down, summary_poly_up, interaction_poly_down, quotient_poly_down

    def _prepare_variables_for_order(self, interaction_order: int) -> None:
        """Retrieves the precomputed variables for a given interaction order.

        This function is called before the recursive explanation function is called.

        Args:
            interaction_order (int): The interaction order for which the storage variables should be
                loaded.

        """
        self.subset_updates_pos = self._interaction_update_positions[interaction_order]
        self.subset_ancestors = self.subset_ancestors_store[interaction_order]
        self.D = self.D_store[interaction_order]
        self.D_powers = self.D_powers_store[interaction_order]
        self._int_height = self._edge_tree.interaction_height_store[interaction_order]
        self.Ns_id = self.Ns_id_store[interaction_order]
        self.Ns = self.Ns_store[interaction_order]

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
    ) -> tuple[dict[int, list[tuple]], dict[int, np.ndarray[int]]]:
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
                int(sp.special.binom(n_features - 1, interaction_order - 1)),
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

    def _precalculate_interaction_ancestors(
        self,
        interaction_order: int,
        n_features: int,
    ) -> dict[int, np.ndarray]:
        """Computes the ancestors of the interactions for a given order of interactions.

        Calculates the position of the ancestors of the interactions for the tree for a given
        order of interactions.

        Args:
            interaction_order: The interaction order for which the ancestors should be computed.
            n_features: The number of features in the model.

        Returns:
            subset_ancestors: A dictionary containing the ancestors of the interactions for each
                node in the tree.

        """
        # stores position of interactions
        subset_ancestors: dict[int, np.ndarray] = {}

        for node_id in self._tree.nodes[1:]:  # for all nodes except the root node
            subset_ancestors[node_id] = np.full(
                int(sp.special.binom(n_features, interaction_order)), -1, dtype=int
            )
        for i, S in enumerate(powerset(range(n_features), interaction_order, interaction_order)):
            for node_id in self._tree.nodes[1:]:  # for all nodes except the root node
                subset_ancestor = -1
                for feature in S:
                    subset_ancestor = max(
                        subset_ancestor,
                        self._edge_tree.ancestor_nodes[node_id][feature],
                    )
                subset_ancestors[node_id][i] = subset_ancestor
        return subset_ancestors

    @staticmethod
    def _get_n_matrix(interpolated_poly: np.ndarray) -> np.ndarray:
        """Computes the N matrix for the Shapley interaction values.

        Args:
            interpolated_poly: The interpolated polynomial.

        Returns:
            The N matrix.

        """
        depth = interpolated_poly.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(interpolated_poly[:i]).T).dot(
                1.0 / np.array([sp.special.binom(i - 1, k) for k in range(i)])
            )
        return Ns

    def _get_n_cii_matrix(self, interpolated_poly: np.ndarray, order: int) -> np.ndarray:
        """Computes the N matrix for the CII index."""
        depth = interpolated_poly.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(interpolated_poly[:i]).T).dot(
                i * np.array([self._get_subset_weight_cii(j, order) for j in range(i)]),
            )
        return Ns

    def _get_subset_weight_cii(self, t: int, order: int) -> float | None:
        """Computes the weight for a given subset size and interaction order.

        Args:
            t: The size of the subset.
            order: The interaction order.

        Returns:
            float | None: The weight for the subset, or None if the index is not supported.
        """
        if self._index == "STII":
            return self._max_order / (
                self._n_features_in_tree * sp.special.binom(self._n_features_in_tree - 1, t)
            )
        if self._index == "FSII":
            return (
                factorial(2 * self._max_order - 1)
                / factorial(self._max_order - 1) ** 2
                * factorial(self._max_order + t - 1)
                * factorial(self._n_features_in_tree - t - 1)
                / factorial(self._n_features_in_tree + self._max_order - 1)
            )
        if self._index == "BII":
            return 1 / (2 ** (self._n_features_in_tree - order))
        return None

    @staticmethod
    def _get_n_id_matrix(D: np.ndarray) -> np.ndarray:
        """Computes N_id matrix."""
        depth = D.shape[0]
        Ns_id = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns_id[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(np.ones(i))
        return Ns_id

    @staticmethod
    def _cache(interpolated_poly: np.ndarray[float]) -> np.ndarray[float]:
        """Caches the powers of the interpolated polynomial.

        Args:
            interpolated_poly: The interpolated polynomial.

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
        information += f"\nInteraction index: {self._index}"
        information += f"\nEmpty prediction (from _tree): {self._tree.empty_prediction}"
        information += f"\nEmpty prediction (from self): {self.empty_prediction}"
