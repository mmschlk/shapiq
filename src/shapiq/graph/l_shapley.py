"""This module contains the LShapley class for Shapley value approximation."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

if TYPE_CHECKING:
    from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame


class LShapley:
    """L-Shapley approximation of Shapley values for graph-structured games.

    Attributes:
        last_n_model_calls: Number of model calls used in the last call to :meth:`explain`.
        max_size_neighbors: The maximum neighborhood size found across all nodes, set after
            calling :meth:`explain`.
    """

    def __init__(self, game: GraphGame, max_budget: int) -> None:
        """Initialise LShapley.

        Args:
            game: The graph game to explain.
            max_budget: Maximum number of model calls allowed before the budget is considered
                exceeded.
        """
        self.last_n_model_calls: int = 0
        self.max_size_neighbors: int = 0

        self.edge_index = game.edge_index
        self.n_players = game.n_players
        self._grand_coalition_set = game.grand_coalition_set
        self.n_jobs = mp.cpu_count() - 1
        self.output_dim = game.output_dim
        self.game = game
        self._grand_coalition_prediction = game(np.ones(self.game.n_players, dtype=bool))
        self.max_budget = max_budget

        self.max_neighborhood_size: int = 0
        self.neighbors: dict = {}

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

    def _get_k_neighborhood(self, node: int) -> tuple:
        """Return the k-hop neighborhood of *node* as a sorted tuple.

        The hop limit k is taken from ``self.max_neighborhood_size``, which is set by
        :meth:`explain` before this method is called.
        """
        neighbors: set[int] = set()
        queue: list[tuple[int, int]] = [(node, 0)]
        visited: set[int] = {node}
        while queue:
            curr_node, dist = queue.pop(0)
            if dist > self.max_neighborhood_size:
                continue
            neighbors.add(curr_node)
            if dist < self.max_neighborhood_size:
                for edge in self.edge_index.T:
                    if edge[0] == curr_node and edge[1] not in visited:
                        queue.append((edge[1], dist + 1))
                        visited.add(edge[1])
        return tuple(sorted(neighbors))

    def _convert_to_coalition_matrix(
        self, coalitions: set | dict, lookup_shift: int = 0
    ) -> tuple[np.ndarray, dict]:
        """Convert a collection of coalitions into a binary matrix and a lookup dict.

        Args:
            coalitions: Either a :class:`set` of coalition tuples or a :class:`dict` whose
                *values* are coalition tuples.
            lookup_shift: Offset added to every row index stored in the lookup dict (useful when
                the matrix is later concatenated with another one).

        Returns:
            coalition_matrix: Binary (n_coalitions × n_players) array.
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
        # Always include the empty coalition for the baseline value
        moebius_interactions.add(())

        for node in self.neighbors:
            for interaction in powerset(self.neighbors[node], max_size=max_interaction_size):
                moebius_interactions.add(interaction)

        return moebius_interactions

    def _l_shapley_routine(
        self,
        neighborhood_of_i: tuple,
        player_i: tuple,
        masked_predictions: np.ndarray,
        coalition_lookup: dict,
        max_interaction_size: int,
    ) -> float:
        """Compute the L-Shapley value for a single player.

        Args:
            neighborhood_of_i: Sorted tuple of node ids in the neighborhood of the player.
            player_i: Single-element tuple containing the player's node id.
            masked_predictions: Array of game evaluations indexed via *coalition_lookup*.
            coalition_lookup: Mapping from coalition tuple → row index in *masked_predictions*.
            max_interaction_size: Maximum subset size considered.

        Returns:
            The L-Shapley value for *player_i*.
        """
        shapley_value: float = 0.0
        size_neighborhood = len(neighborhood_of_i)

        for subset in powerset(neighborhood_of_i, max_size=max_interaction_size):
            if set(player_i).issubset(set(subset)):
                no_player_i = tuple(sorted(set(subset) - set(player_i)))
                marginal_contribution = (
                    masked_predictions[coalition_lookup[subset]]
                    - masked_predictions[coalition_lookup[no_player_i]]
                )
                weight = binom(size_neighborhood - 1, len(subset) - 1) ** (-1)
                shapley_value += weight * marginal_contribution

        shapley_value /= size_neighborhood
        return shapley_value

    def explain(
        self,
        max_interaction_size: int,
        *,
        break_on_exceeding_budget: bool,
    ) -> tuple[InteractionValues, bool]:
        """Compute L-Shapley values for all players.

        Args:
            max_interaction_size: Maximum k-hop neighborhood size (controls how many coalition
                subsets are enumerated per player).
            break_on_exceeding_budget: If *True*, raise a :class:`ValueError` when the number of
                required model evaluations exceeds ``self.max_budget``; if *False*, set the
                ``exceeded_budget`` flag and continue.

        Returns:
            A tuple ``(interaction_values, exceeded_budget)`` where *interaction_values* is the
            resulting :class:`~shapiq.interaction_values.InteractionValues` object and
            *exceeded_budget* is a boolean flag.

        Raises:
            ValueError: If *break_on_exceeding_budget* is *True* and the budget is exceeded.
        """
        self.max_neighborhood_size = max_interaction_size

        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        max_interaction_size = min(self.max_size_neighbors, max_interaction_size)

        coalitions = self._get_all_coalitions(max_interaction_size)

        exceeded_budget = False
        if len(coalitions) > self.max_budget:
            exceeded_budget = True
            if break_on_exceeding_budget:
                msg = "Exceeded budget."
                raise ValueError(msg)

        coalition_matrix, coalition_lookup = self._convert_to_coalition_matrix(coalitions)
        masked_predictions = self.game(coalition_matrix)

        self.last_n_model_calls = int(np.shape(coalition_matrix)[0])

        # Compute L-Shapley values for every player
        shapley_values = np.zeros(self.n_players)
        shapley_values_lookup: dict = {}

        for player_i in self._grand_coalition_set:
            neighborhood_of_i = self.neighbors[player_i]
            player_i_tuple = (player_i,)
            shapley_values_lookup[player_i_tuple] = player_i
            shapley_values[player_i] = self._l_shapley_routine(
                neighborhood_of_i,
                player_i_tuple,
                masked_predictions,
                coalition_lookup,
                max_interaction_size,
            )

        baseline_value = float(masked_predictions[coalition_lookup[()]])

        int_values = InteractionValues(
            values=shapley_values,
            interaction_lookup=shapley_values_lookup,
            min_order=0,
            max_order=1,
            n_players=self.n_players,
            index="SV",
            baseline_value=baseline_value,
            estimation_budget=self.last_n_model_calls,
        )

        return int_values, exceeded_budget
