"""This module contains the GraphSHAP-IQ class."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Dict, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

if TYPE_CHECKING:
    from shapiq.games import Game  # noqa: F401

logger = logging.getLogger(__name__)

class GraphSHAPIQ:
    """A class for computing Shapley interactions on graph-structured data using the Möbius transform.

    Attributes:
        edge_index: The edge index of the graph.
        n_players: Number of players (nodes) in the graph.
        max_neighborhood_size: Maximum size of neighborhoods to consider.
        neighbors: Dictionary mapping each node to its neighborhood.
        max_size_neighbors: Maximum size of any neighborhood in the graph.
        total_budget: Total number of coalitions to evaluate (or upper bound).
        budget_estimated: Whether the budget was estimated or computed exactly.
    """

    def __init__(self, game: Game, *, verbose: bool = False) -> None:
        """Initialize the GraphSHAPIQ class.

        Args:
            game: The game object representing the graph and prediction function.
            verbose: If True, prints debug information.
        """
        self._last_n_model_calls: int | None = None
        self.edge_index = game.edge_index
        self.n_players = game.n_players
        self.sparsify_threshold = 1e-10
        self.max_neighborhood_size = game.max_neighborhood_size
        self._grand_coalition_set = game._grand_coalition_set  # noqa: SLF001
        self.n_jobs = max(1, mp.cpu_count() - 1)  # Ensure at least 1 job
        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        self.output_dim = game.output_dim
        self.game = game
        self._grand_coalition_prediction = game(np.ones(self.n_players, dtype=float))
        self.verbose = verbose

        # Compute or estimate total budget
        if self.max_size_neighbors <= 22:
            moebius_interactions, _ = self._get_all_coalitions(
                max_interaction_size=self.max_size_neighbors,
                efficiency_routine=False,
            )
            self.total_budget = len(moebius_interactions)
            self.budget_estimated = False
        else:
            self.total_budget = 0
            for node in self.neighbors:
                self.total_budget += 2 ** len(self.neighbors[node])
            self.total_budget = min(2 ** self.n_players, self.total_budget)
            self.budget_estimated = True

        if verbose:
            logger.info("Max size neighbors: %d", self.max_size_neighbors)
            logger.info("Total budget: %d", self.total_budget)

    def _get_neighborhoods(self) -> tuple:
        """Computes the neighborhoods of each node and caps the max_interaction_size.

        Neighborhood is capped to max_interaction_size at the size of
        the largest neighborhood.

        Returns:
            neighbors: A dictionary containing all neighbors of each node
            max_interaction_size: max_interaction_size capped at the largest neighborhood size
        """
        neighbors = {}
        max_size_neighbors = 0
        neighbors_list = [
            self._get_k_neighborhood(node_id) for node_id in self._grand_coalition_set
        ]

        for neighbor_id, neighbor_list in enumerate(neighbors_list):
            max_size_neighbors = max(max_size_neighbors, len(neighbor_list))
            neighbors[neighbor_id] = neighbor_list

        return neighbors, max_size_neighbors

    def _get_k_neighborhood(self, node: int) -> tuple[int, ...]:
        neighbors = set()
        queue = [(node, 0)]
        visited = {node}
        while queue:
            curr_node, dist = queue.pop(0)
            if dist > self.max_neighborhood_size:
                break
            if dist <= self.max_neighborhood_size:
                neighbors.add(curr_node)
            if dist < self.max_neighborhood_size:
                # Find neighbors of current node
                for edge in self.edge_index.T:
                    if edge[0] == curr_node and edge[1] not in visited:
                        queue.append((edge[1], dist + 1))
                        visited.add(edge[1])
        return tuple(sorted(neighbors))

    def compute_moebius_transform(
        self,
        coalitions: Union[Set[Tuple[int, ...]], Dict[Tuple[int, ...], Any]],
        coalition_predictions: NDArray[np.floating],
        coalition_lookup: Dict[Tuple[int, ...], int],
    ) -> InteractionValues:
        """Compute the Möbius transform for the given coalitions.

        Args:
            coalitions: Set or dict of coalitions (tuples of player indices).
            coalition_predictions: Predictions for each coalition.
            coalition_lookup: Mapping from coalition tuples to their indices.

        Returns:
            InteractionValues object containing the Möbius values.
        """
        moebius_values = np.zeros(len(coalition_lookup), dtype=float)
        moebius_lookup: Dict[Tuple[int, ...], int] = {}

        for i, coalition in enumerate(coalitions):
            moebius_values[i] = 0.0
            moebius_lookup[coalition] = i
            for subset in powerset(coalition):
                sign = (-1) ** (len(coalition) - len(subset))
                moebius_values[i] += sign * coalition_predictions[coalition_lookup[subset]]

        return InteractionValues(
            values=moebius_values,
            interaction_lookup=moebius_lookup,
            min_order=0,
            max_order=self.n_players,
            n_players=self.n_players,
            index="Moebius",
            baseline_value=float(moebius_values[moebius_lookup[()]]),
        )

    def _convert_to_coalition_matrix(
        self,
        coalitions: Union[Set[Tuple[int, ...]], Dict[Any, Tuple[int, ...]]],
        lookup_shift: int = 0,
    ) -> Tuple[NDArray[np.floating], Dict[Tuple[int, ...], int]]:
        """Convert a set or dict of coalitions to a binary matrix and lookup dict.

        Args:
            coalitions: Set or dict of coalitions.
            lookup_shift: Offset to add to the lookup indices.

        Returns:
            Tuple of (coalition_matrix, coalition_lookup), where:
                - coalition_matrix: Binary matrix of shape (n_coalitions, n_players).
                - coalition_lookup: Mapping from coalition tuples to row indices.
        """
        coalition_matrix = np.zeros((len(coalitions), self.n_players), dtype=float)
        coalition_lookup: Dict[Tuple[int, ...], int] = {}

        if isinstance(coalitions, set):
            for i, coalition in enumerate(coalitions):
                coalition_matrix[i, list(coalition)] = 1
                coalition_lookup[coalition] = lookup_shift + i
        elif isinstance(coalitions, dict):
            for i, (_, coalition) in enumerate(coalitions.items()):
                coalition_matrix[i, list(coalition)] = 1
                coalition_lookup[coalition] = lookup_shift + i

        return coalition_matrix, coalition_lookup

    def _efficiency_routine(
        self,
        masked_predictions: NDArray[np.floating],
        moebius_coefficients: InteractionValues,
        incomplete_neighborhoods: Set[Tuple[int, ...]],
        incomplete_neighborhoods_lookup: Dict[Tuple[int, ...], int],
        max_interaction_size: int,
        grand_coalition_prediction_node: NDArray[np.floating],
    ) -> InteractionValues:
        """Adjust Möbius coefficients to ensure efficiency.

        Args:
            masked_predictions: Predictions for all coalitions.
            moebius_coefficients: Current Möbius coefficients.
            incomplete_neighborhoods: Neighborhoods not fully covered by coalitions.
            incomplete_neighborhoods_lookup: Lookup for incomplete neighborhoods.
            max_interaction_size: Maximum interaction size considered.
            grand_coalition_prediction_node: Prediction for the grand coalition.

        Returns:
            InteractionValues with adjusted Möbius coefficients.
        """
        incomplete_neighborhoods_sorted = sorted(incomplete_neighborhoods, key=len)
        n_incomplete = len(incomplete_neighborhoods_lookup)
        additional_values = np.zeros(n_incomplete, dtype=float)
        additional_lookup: Dict[Tuple[int, ...], int] = {}

        for i, neighborhood in enumerate(incomplete_neighborhoods_sorted):
            # Compute sum of Möbius coefficients for subsets of the neighborhood
            sum_coefficients = 0.0
            for interaction in powerset(neighborhood, max_size=max_interaction_size):
                sum_coefficients += moebius_coefficients[interaction]

            # Add contributions from smaller incomplete neighborhoods
            for other_neighborhood in incomplete_neighborhoods_sorted:
                if (
                    set(other_neighborhood).issubset(neighborhood)
                    and len(other_neighborhood) < len(neighborhood)
                ):
                    sum_coefficients += additional_values[
                        additional_lookup[other_neighborhood]
                    ]

            # Compute and assign the efficiency gap
            gap = (
                masked_predictions[incomplete_neighborhoods_lookup[neighborhood]]
                - sum_coefficients
            )
            additional_values[i] = gap
            additional_lookup[neighborhood] = i

        # Adjust the largest neighborhood to maintain efficiency
        if additional_lookup:
            additional_values[-1] = (
                grand_coalition_prediction_node
                - np.sum(moebius_coefficients.values)
                - np.sum(additional_values[:-1])
            )

        return InteractionValues(
            values=additional_values,
            index="Moebius",
            max_order=self.n_players,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=additional_lookup,
            baseline_value=0.0,
        )

    def explain(
        self,
        max_interaction_size: int | None = None,
        order: int | None = None,
        *,
        efficiency_routine: bool = True,
    ) -> Tuple[
        Union[InteractionValues, Dict[int, InteractionValues]],
        Union[InteractionValues, Dict[int, InteractionValues]],
    ]:
        """Compute Shapley interactions for the graph.

        Args:
            max_interaction_size: Maximum size of interactions to consider.
                If None, uses the maximum neighborhood size.
            order: Maximum order of interactions to return. If None, uses n_players.
            efficiency_routine: If True, ensures efficiency by adjusting neighborhood interactions.

        Returns:
            Tuple of (moebius_coefficients, shapley_interactions).
        """
        if order is None:
            order = self.n_players

        capped_interaction_size = min(
            self.max_size_neighbors,
            max_interaction_size if max_interaction_size is not None else self.max_size_neighbors,
        )

        # Get coalitions and incomplete neighborhoods
        moebius_interactions, incomplete_neighborhoods = self._get_all_coalitions(
            max_interaction_size=capped_interaction_size,
            efficiency_routine=efficiency_routine,
        )

        # Convert to matrices
        (
            incomplete_neighborhoods_matrix,
            incomplete_neighborhoods_lookup,
        ) = self._convert_to_coalition_matrix(
            incomplete_neighborhoods,
            lookup_shift=len(moebius_interactions),
        )
        moebius_coalition_matrix, moebius_coalition_lookup = self._convert_to_coalition_matrix(
            moebius_interactions
        )

        # Evaluate all coalitions
        all_coalitions = np.vstack((moebius_coalition_matrix, incomplete_neighborhoods_matrix))
        masked_predictions = self.game(all_coalitions)
        self._last_n_model_calls = int(all_coalitions.shape[0])

        # Handle single or multi-output cases
        if self.output_dim == 1:
            moebius_coefficients, shapley_interactions = self._graphshapiq_routine(
                moebius_interactions,
                masked_predictions,
                moebius_coalition_lookup,
                efficiency_routine,
                incomplete_neighborhoods,
                incomplete_neighborhoods_lookup,
                capped_interaction_size,
                order,
                self._grand_coalition_prediction,
            )
        else:
            moebius_coefficients: Dict[int, InteractionValues] = {}
            shapley_interactions: Dict[int, InteractionValues] = {}
            for idx in range(self.output_dim):
                grand_coalition_pred_node = self._grand_coalition_prediction[:, idx]
                masked_predictions_node = masked_predictions[:, idx]
                moebius_coefficients[idx], shapley_interactions[idx] = self._graphshapiq_routine(
                    moebius_interactions,
                    masked_predictions_node,
                    moebius_coalition_lookup,
                    efficiency_routine,
                    incomplete_neighborhoods,
                    incomplete_neighborhoods_lookup,
                    capped_interaction_size,
                    order,
                    grand_coalition_pred_node,
                )

        return moebius_coefficients, shapley_interactions

    def _graphshapiq_routine(
        self,
        moebius_interactions: Set[Tuple[int, ...]],
        masked_predictions: NDArray[np.floating],
        moebius_coalition_lookup: Dict[Tuple[int, ...], int],
        efficiency_routine: bool,
        incomplete_neighborhoods: Set[Tuple[int, ...]],
        incomplete_neighborhoods_lookup: Dict[Tuple[int, ...], int],
        max_interaction_size: int,
        order: int,
        grand_coalition_prediction_node: NDArray[np.floating],
    ) -> Tuple[InteractionValues, InteractionValues]:
        """Core routine for computing GraphSHAPIQ values.

        Args:
            moebius_interactions: Set of coalitions to compute Möbius values for.
            masked_predictions: Predictions for all coalitions.
            moebius_coalition_lookup: Lookup for Möbius coalitions.
            efficiency_routine: Whether to enforce efficiency.
            incomplete_neighborhoods: Neighborhoods not fully covered.
            incomplete_neighborhoods_lookup: Lookup for incomplete neighborhoods.
            max_interaction_size: Maximum interaction size.
            order: Maximum order of interactions.
            grand_coalition_prediction_node: Prediction for the grand coalition.

        Returns:
            Tuple of (final Möbius coefficients, Shapley interactions).
        """
        # Compute Möbius transform
        moebius_coefficients = self.compute_moebius_transform(
            coalitions=moebius_interactions,
            coalition_predictions=masked_predictions,
            coalition_lookup=moebius_coalition_lookup,
        )
        moebius_coefficients.sparsify(self.sparsify_threshold)

        # Adjust for efficiency if needed
        if efficiency_routine:
            additional_coefficients = self._efficiency_routine(
                masked_predictions,
                moebius_coefficients,
                incomplete_neighborhoods,
                incomplete_neighborhoods_lookup,
                max_interaction_size,
                grand_coalition_prediction_node,
            )
            final_moebius = moebius_coefficients + additional_coefficients
        else:
            final_moebius = moebius_coefficients

        # Convert to Shapley interactions
        converter = MoebiusConverter(moebius_coefficients=final_moebius)
        interactions = converter.compute(index="k-SII", order=order)
        interactions.sparsify(self.sparsify_threshold)

        return copy.deepcopy(final_moebius), copy.deepcopy(interactions)

    def _get_all_coalitions(
        self,
        max_interaction_size: int,
        *,
        efficiency_routine: bool,
    ) -> Tuple[Set[Tuple[int, ...]], Set[Tuple[int, ...]]]:
        """Collect all coalitions for Möbius transform and efficiency adjustment.

        Args:
            max_interaction_size: Maximum size of interactions to consider.
            efficiency_routine: If True, includes incomplete neighborhoods.

        Returns:
            Tuple of (moebius_interactions, incomplete_neighborhoods).
        """
        moebius_interactions: Set[Tuple[int, ...]] = set()
        incomplete_neighborhoods: Set[Tuple[int, ...]] = set()

        for node in self.neighbors:
            # Add all subsets of the neighborhood up to max_interaction_size
            for interaction in powerset(
                self.neighbors[node],
                max_size=max_interaction_size,
            ):
                moebius_interactions.add(interaction)

            # Add incomplete neighborhoods if efficiency_routine is enabled
            if efficiency_routine and len(self.neighbors[node]) > max_interaction_size:
                incomplete_neighborhoods.add(tuple(self.neighbors[node]))

        return moebius_interactions, incomplete_neighborhoods

    @property
    def last_n_model_calls(self) -> int | None:
        """Return the number of model calls from the last explanation."""
        return self._last_n_model_calls
