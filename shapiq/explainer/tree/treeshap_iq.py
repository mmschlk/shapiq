"""This module contains the tree explainer implementation."""
import copy
from typing import Any, Union
import warnings

import numpy as np
from scipy.special import binom

from approximator._interaction_values import InteractionValues
from explainer._base import Explainer

from explainer.tree._validate import _validate_model
from explainer.tree.conversion import TreeModel, EdgeTree, create_edge_tree
from utils.sets import generate_interaction_lookup, powerset


class TreeExplainer(Explainer):
    """
    The explainer for tree-based models using the TreeSHAP-IQ algorithm.

    Args:
        model: The tree-based model to explain. This may be a model object or a dictionary
            containing the tree information (i.e. an explainer.tree.conversion.TreeModel).
        max_order: The maximum interaction order to be computed. An interaction order of 1
            corresponds to the Shapley value. Any value higher than 1 computes the Shapley
            interactions values up to that order. Defaults to 2.
        n_features: The number of features of the dataset. If not provided, the number of features
            is inferred from the model. Defaults to None.
        interaction_type: The type of interaction to be computed. The interaction type can be
            "k-SII" (default) or "SII", "STI", "FSI", "BZF".
    """

    def __init__(
        self,
        model: Union[dict, TreeModel, Any],
        max_order: int = 2,
        n_features: int = None,
        interaction_type: str = "k-SII",
        verbose: bool = False,
    ) -> None:
        # TODO init Explainer object

        # set parameters
        self.verbose = verbose
        self._max_order: int = max_order
        self._interaction_type: str = interaction_type

        # validate and parse model
        validated_model = _validate_model(model)  # the parsed and validated model
        # TODO: add support for other sample weights
        self._tree: TreeModel = copy.deepcopy(validated_model)
        self._n_nodes: int = len(self._tree.children_left)
        self._n_features: int = n_features if n_features is not None else self._tree.n_features

        # precompute interaction lookup tables
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
            n_features=self._n_features,
            n_nodes=self._n_nodes,
            subset_updates_pos_store=self._interaction_update_positions,
        )
        self._edge_tree: EdgeTree = copy.deepcopy(edge_tree)

        # compute the empty prediction
        self.empty_prediction: float = float(
            np.sum(self._edge_tree.empty_predictions[self._tree.leaf_mask])
        )

        # TODO: add the interaction value storages

        # print tree information
        if self.verbose:
            self._print_tree_info()

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        pass

    def _init_interaction_lookup_tables(self):
        """Initializes the lookup tables for the interaction subsets."""
        for order in range(1, self._max_order + 1):
            order_interactions_lookup = generate_interaction_lookup(self._n_features, order, order)
            self._interactions_lookup[order] = order_interactions_lookup
            _, interaction_update_positions = self._precompute_subsets_with_feature(
                interaction_order=order,
                n_features=self._n_features,
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

    @staticmethod
    def _precompute_subsets_with_feature_slow(
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
        # throw a deprecation warning
        warnings.warn(
            "The method `precompute_subsets_with_feature_slow` is deprecated and will be "
            "removed in a future version. Use `precompute_subsets_with_feature` instead.",
            DeprecationWarning,
        )
        # stores interactions that include feature i (needs to be updated when feature i appears)
        interaction_updates: dict[int, list[tuple]] = {}
        # stores position of interactions that include feature i
        interaction_update_positions: dict[int, np.ndarray[int]] = {}
        for i in range(n_features):
            subsets = []
            positions = np.zeros(int(binom(n_features - 1, interaction_order - 1)), dtype=int)
            pos_counter = 0
            for interaction in powerset(
                range(n_features), min_size=interaction_order, max_size=interaction_order
            ):
                if i in interaction:
                    positions[pos_counter] = order_interactions_lookup[interaction]
                    subsets.append(interaction)
                    pos_counter += 1
            interaction_update_positions[i] = positions
            interaction_updates[i] = subsets
        return interaction_updates, interaction_update_positions

    def _print_tree_info(self) -> None:
        """Prints information about the tree to be explained."""
        information = "Tree information:"
        information += f"\nNumber of nodes: {self._n_nodes}"
        information += f"\nNumber of features: {self._n_features}"
        information += f"\nMaximum interaction order: {self._max_order}"
        information += f"\nInteraction type: {self._interaction_type}"
        # add empty prediction from _tree and self to information TODO: remove one in final
        information += f"\nEmpty prediction (from _tree): {self._tree.empty_prediction}"
        information += f"\nEmpty prediction (from self): {self.empty_prediction}"
        print(information)
