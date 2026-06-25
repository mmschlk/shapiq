"""This module contains the LShapley class for Shapley value approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

if TYPE_CHECKING:
    from .base import GraphGame


class LShapley:
    """L-Shapley approximation of Shapley values for graph-structured games."""

    def __init__(self, game: GraphGame, max_budget: int) -> None:
        """Initialise LShapley.

        Args:
            game: The graph game to explain.
            max_budget: Maximum number of model calls allowed before the budget is considered
                        exceeded.
        """
        self.last_n_model_calls: int = 0
        self.max_size_neighbors: int = 0
        self.max_budget: int = max_budget
        self.l_hop_distance: int = 0
        self.neighbors: dict = {}

        self.game: GraphGame = game
        self.edge_index: np.ndarray = game.edge_index
        self.n_players: int = game.n_players
        self._grand_coalition_set: set[int] = game.grand_coalition_set

        self._graph: nx.Graph = self._build_graph()

    def _get_neighborhoods(self) -> tuple[dict, int]:
        """Compute the k-hop neighborhoods of every node.

        Returns:
            neighbors: Mapping from node id to a sorted tuple of neighbor ids (including the
                       node itself).
            max_size_neighbors: Size of the largest neighborhood found.
        """
        neighbors: dict = {}
        max_size_neighbors: int = 0
        for neighbor_id in self._grand_coalition_set:
            neighbor_list = self._get_k_neighborhood(neighbor_id)
            max_size_neighbors = max(max_size_neighbors, len(neighbor_list))
            neighbors[neighbor_id] = neighbor_list
        return neighbors, max_size_neighbors

    def _build_graph(self) -> nx.Graph:
        """Convert edge_index to an nx.Graph."""
        G: nx.Graph = nx.Graph()
        G.add_nodes_from(range(self.n_players))
        G.add_edges_from(self.edge_index.T.tolist())
        return G

    def _get_k_neighborhood(self, node: int) -> tuple:
        """Get the k-hop neighborhood of a node."""
        reachable = nx.single_source_shortest_path_length(
            self._graph, node, cutoff=self.l_hop_distance
        )
        return tuple(sorted(reachable.keys()))

    def _check_budget(self, n_coalitions: int, *, break_on_exceeding: bool) -> bool:
        """Check if number of coalitions is within max_budget."""
        if n_coalitions <= self.max_budget:
            return False
        if break_on_exceeding:
            err_msg = f"Budget exceeded: {n_coalitions} coalitions required, but max_budget is {self.max_budget}."
            raise ValueError(err_msg)
        return True

    def _convert_to_coalition_matrix(
        self, coalitions: set | dict, lookup_shift: int = 0
    ) -> tuple[np.ndarray, dict]:
        """Convert a collection of coalitions into a binary matrix and a lookup dict.

        Args:
            coalitions: Either a :class:`set` of coalition tuples or a :class:`dict` whose
                values are coalition tuples.
            lookup_shift: Offset added to every row index stored in the lookup dict (useful when
                the matrix is later concatenated with another one).

        Returns:
            coalition_matrix: Binary (n_coalitions, n_players) array.
            coalition_lookup: Mapping from coalition tuple → row index in the matrix.
        """
        coalition_matrix = np.zeros((len(coalitions), self.n_players))
        coalition_lookup: dict = {}

        if isinstance(coalitions, set):
            for i, S in enumerate(coalitions):
                coalition_matrix[i, list(S)] = 1
                coalition_lookup[S] = lookup_shift + i
        elif isinstance(coalitions, dict):
            for i, (_, S) in enumerate(coalitions.items()):
                coalition_matrix[i, list(S)] = 1
                coalition_lookup[S] = lookup_shift + i

        return coalition_matrix, coalition_lookup

    def _get_all_coalitions(self, max_interaction_size: int) -> set:
        """Collect all coalitions that need to be evaluated.

        The empty coalition ``()`` is always included so that the baseline value can be read
        from the evaluated predictions.

        Args:
            max_interaction_size: Maximum subset size to enumerate per neighborhood.

        Returns:
            Set of coalition tuples (each element is a sorted tuple of player indices).
        """
        moebius_interactions: set = set()
        moebius_interactions.add(())

        for node in self.neighbors:
            for interaction in powerset(self.neighbors[node], max_size=max_interaction_size):
                moebius_interactions.add(interaction)

        return moebius_interactions

    def _shapley_weight(self, neighborhood_size: int, subset_size: int) -> float:
        """Compute the Shapley weight for a subset within a neighbourhood.

        The weight is the reciprocal of the number of orderings in which
        player i could enter a coalition of the given size, normalised by
        the neighbourhood size.
        """
        return float(binom(neighborhood_size - 1, subset_size - 1) ** -1) / neighborhood_size

    def _l_shapley_routine(
        self,
        neighborhood_of_i: tuple,
        node_id: int,
        masked_predictions: np.ndarray,
        coalition_lookup: dict,
        max_interaction_size: int,
    ) -> float:
        """Compute the L-Shapley value for a single node.

        Iterates over all subsets of the node's neighbourhood up to
        ``max_interaction_size``, skipping any that do not include
        ``node_id``. For each qualifying subset S, the marginal
        contribution of the node is calculated
        """
        shapley_value: float = 0.0
        neighborhood_size = len(neighborhood_of_i)
        player_set = {node_id}

        for subset in powerset(neighborhood_of_i, max_size=max_interaction_size):
            if not player_set.issubset(set(subset)):
                continue
            subset_without_player = tuple(sorted(set(subset) - player_set))
            marginal_contribution = (
                masked_predictions[coalition_lookup[subset]]
                - masked_predictions[coalition_lookup[subset_without_player]]
            )
            weight = self._shapley_weight(neighborhood_size, len(subset))
            shapley_value += float(weight * marginal_contribution)

        return shapley_value

    def explain(
        self, max_interaction_size: int, *, break_on_exceeding_budget: bool, index: str = "SV"
    ) -> tuple[InteractionValues, bool]:
        """Compute L-Shapley values for all players.

        Args:
            max_interaction_size: Maximum neighborhood size (controls how many coalition
                subsets are enumerated per player).
            break_on_exceeding_budget: If True, raise a :class:`ValueError` when the number of
                required model evaluations exceeds ``self.max_budget``; if False, set the
                ``exceeded_budget`` flag and continue.
            index: The index of the interaction values.

        Returns:
            A tuple ``(interaction_values, exceeded_budget)`` where interaction_values is the
            resulting :class:`~shapiq.interaction_values.InteractionValues` object and
            exceeded_budget is a boolean flag.

        Raises:
            ValueError: If break_on_exceeding_budget is True and the budget is exceeded.
        """
        self.l_hop_distance = self.game.l_hop_distance
        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        max_interaction_size = min(self.max_size_neighbors, max_interaction_size)

        coalitions = self._get_all_coalitions(max_interaction_size)
        exceeded_budget = self._check_budget(
            len(coalitions), break_on_exceeding=break_on_exceeding_budget
        )

        coalition_matrix, coalition_lookup = self._convert_to_coalition_matrix(coalitions)
        masked_predictions = self.game(coalition_matrix)
        self.last_n_model_calls = int(coalition_matrix.shape[0])

        shapley_values = np.zeros(self.n_players)
        shapley_values_lookup: dict = {}

        for node_id in self._grand_coalition_set:
            player_tuple = (node_id,)
            shapley_values_lookup[player_tuple] = node_id
            shapley_values[node_id] = self._l_shapley_routine(
                neighborhood_of_i=self.neighbors[node_id],
                node_id=node_id,
                masked_predictions=masked_predictions,
                coalition_lookup=coalition_lookup,
                max_interaction_size=max_interaction_size,
            )

        baseline_value = float(masked_predictions[coalition_lookup[()]])

        int_values = InteractionValues(
            values=shapley_values,
            interaction_lookup=shapley_values_lookup,
            min_order=0,
            max_order=1,
            n_players=self.n_players,
            index=index,
            baseline_value=baseline_value,
            estimation_budget=self.last_n_model_calls,
        )

        return int_values, exceeded_budget
