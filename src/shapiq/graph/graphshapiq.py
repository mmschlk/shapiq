"""This module contains the GraphSHAP-IQ class."""

from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Literal

import numpy as np

from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

GraphSHAPIQIndices = Literal["SV", "SII", "k-SII"]


class GraphSHAPIQ:
    """Graph shapiq."""

    def __init__(self, game, *, verbose: bool = False) -> None:  # noqa: ANN001
        """Initialize GraphSHAPIQ class."""
        self._last_n_model_calls: int | None = None
        self.edge_index = game.edge_index
        self.n_players = game.n_players
        self.sparsify_threshold = 10e-10
        self.max_neighborhood_size = game.max_neighborhood_size
        self._grand_coalition_set = game._grand_coalition_set  # noqa: SLF001
        self.n_jobs = mp.cpu_count() - 1
        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        if verbose:
            print(f"Max size neighbors: {self.max_size_neighbors}")  # noqa: T201
        self.output_dim = game.output_dim
        self.game = game
        self._grand_coalition_prediction = game(np.ones(self.game.n_players))
        # This is optional and only for experiments - computes total budget for that graph
        if self.max_size_neighbors <= 22:
            moebius_interactions, _ = self._get_all_coalitions(
                max_interaction_size=self.max_size_neighbors, efficiency_routine=False
            )
            self.total_budget = len(moebius_interactions)
            self.budget_estimated = False
        else:
            # or estimates it using an upper bound
            self.total_budget = 0
            for node in self.neighbors:
                self.total_budget += 2 ** (len(self.neighbors[node]))
                self.budget_estimated = True
            self.total_budget = int(min(2**self.n_players, self.total_budget))
        if verbose:
            print(f"Total budget: {self.total_budget}")  # noqa: T201
        self.verbose = verbose

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
        with mp.Pool(self.n_jobs) as pool:
            neighbors_list = list(
                pool.starmap(
                    self._get_k_neighborhood, [(node_id,) for node_id in self._grand_coalition_set]
                )
            )

        for neighbor_id, neighbor_list in enumerate(neighbors_list):
            max_size_neighbors = max(max_size_neighbors, len(neighbor_list))
            neighbors[neighbor_id] = neighbor_list

        return neighbors, max_size_neighbors

    def _get_k_neighborhood(self, node) -> tuple:  # noqa: ANN001
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
        coalitions: dict,
        coalition_predictions: np.ndarray,
        coalition_lookup: np.ndarray,
    ) -> InteractionValues:
        """Compute moebius traffo."""
        moebius_values = np.zeros(len(coalition_lookup))
        moebius_lookup = {}

        for i, coalition in enumerate(coalitions):
            moebius_values[i] = 0
            moebius_lookup[coalition] = i
            for L in powerset(coalition):
                moebius_values[i] += (-1) ** (len(coalition) - len(L)) * coalition_predictions[
                    coalition_lookup[L]
                ]

        return InteractionValues(
            values=moebius_values,
            interaction_lookup=moebius_lookup,
            min_order=0,
            max_order=self.n_players,
            n_players=self.n_players,
            index="Moebius",
            baseline_value=float(moebius_values[moebius_lookup[()]]),
        )

    def _convert_to_coalition_matrix(self, coalitions: set | dict, lookup_shift: int = 0) -> tuple:
        """Convert to coalition matrix."""
        coalition_matrix = np.zeros((len(coalitions), self.n_players))
        coalition_lookup = {}

        if type(coalitions) is set:
            for i, S in enumerate(coalitions):
                coalition_matrix[i, S] = 1
                coalition_lookup[S] = lookup_shift + i
        if type(coalitions) is dict:
            for i, (_, S) in enumerate(coalitions.items()):
                coalition_matrix[i, S] = 1
                coalition_lookup[S] = lookup_shift + i
        return coalition_matrix, coalition_lookup

    def _efficiency_routine(
        self,
        masked_predictions,  # noqa: ANN001
        moebius_coefficients,  # noqa: ANN001
        incomplete_neighborhoods,  # noqa: ANN001
        incomplete_neighborhoods_lookup,  # noqa: ANN001
        max_interaction_size,  # noqa: ANN001
        _grand_coalition_prediction_node,  # noqa: ANN001
    ) -> InteractionValues:
        # Add neighborhood interactions
        incomplete_neighborhoods_sorted = sorted(incomplete_neighborhoods, key=len)
        incomplete_neighborhoods_size = len(incomplete_neighborhoods_lookup)
        additional_moebius_coefficients_values = np.zeros(incomplete_neighborhoods_size)
        # FIX 1: added explicit type annotation to satisfy mypy [var-annotated]
        additional_moebius_coefficients_lookup: dict[tuple, int] = {}

        for i, neighborhood_coalition in enumerate(incomplete_neighborhoods_sorted):
            # Compute efficiency gap due to incomplete neighborhood
            sum_of_moebius_coefficients = 0
            for interaction in powerset(neighborhood_coalition, max_size=max_interaction_size):
                sum_of_moebius_coefficients += moebius_coefficients[interaction]
            for other_incomplete_moebius in incomplete_neighborhoods_sorted:
                if set(other_incomplete_moebius).issubset(set(neighborhood_coalition)) and len(
                    other_incomplete_moebius
                ) < len(neighborhood_coalition):
                    sum_of_moebius_coefficients += additional_moebius_coefficients_values[
                        additional_moebius_coefficients_lookup[other_incomplete_moebius]
                    ]
            # match with prediction of neighborhood
            gap = (
                masked_predictions[incomplete_neighborhoods_lookup[neighborhood_coalition]]
                - sum_of_moebius_coefficients
            )
            # Assign gap to neighborhood moebius coefficient
            additional_moebius_coefficients_values[i] = gap
            additional_moebius_coefficients_lookup[neighborhood_coalition] = i

            if len(additional_moebius_coefficients_lookup) > 0:
                # Maintain efficiency by adjusting the largest neighborhood set
                additional_moebius_coefficients_values[-1] = (
                    _grand_coalition_prediction_node
                    - np.sum(moebius_coefficients.values)
                    - np.sum(additional_moebius_coefficients_values[:-1])
                )

        # Store in InteactionValues Object
        return InteractionValues(
            values=additional_moebius_coefficients_values,
            index="Moebius",
            max_order=self.game.n_players,
            min_order=0,
            n_players=self.game.n_players,
            interaction_lookup=additional_moebius_coefficients_lookup,
            baseline_value=0,
        )

    def explain(
        self,
        max_interaction_size: int | None = None,
        order: int | None = None,
        *,
        efficiency_routine: bool = True,
    ) -> tuple:
        """Explain the prediction."""
        if order is None:
            order = self.n_players

        # Cap max_interaction_size.  Introduce a new local with an explicit int type so
        # mypy can prove None is impossible at the min() call and at _get_all_coalitions.
        capped_interaction_size: int = min(
            self.max_size_neighbors,
            self.max_size_neighbors if max_interaction_size is None else max_interaction_size,
        )

        # Get collection of Möbius interactions to be computed, and complete neighborhoods that are
        # not considered (if efficiency_routine is True).
        moebius_interactions, incomplete_neighborhoods = self._get_all_coalitions(
            max_interaction_size=capped_interaction_size,
            efficiency_routine=efficiency_routine,
        )

        # Convert neighborhoods into coalition matrix
        (
            incomplete_neighborhoods_matrix,
            incomplete_neighborhoods_lookup,
        ) = self._convert_to_coalition_matrix(
            incomplete_neighborhoods, lookup_shift=len(moebius_interactions)
        )
        # Convert collected coalitions into coalition matrix
        moebius_coalition_matrix, moebius_coalition_lookup = self._convert_to_coalition_matrix(
            moebius_interactions
        )

        # Evaluate the (stacked) coalition matrix on the GNN
        all_coalitions = np.vstack((moebius_coalition_matrix, incomplete_neighborhoods_matrix))
        masked_predictions = self.game(all_coalitions)
        # FIX 2: was `self.last_n_model_calls` (public attribute not declared anywhere);
        # the __init__ declares `self._last_n_model_calls: int | None`, so assign there.
        # np.shape returns a tuple[int, ...], so extract the first element to keep the int type.
        self._last_n_model_calls = int(np.shape(all_coalitions)[0])

        if self.output_dim == 1:
            moebius_coefficients, shapley_interactions = self._grapshapiq_routine(
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
            shapley_interactions = {}
            moebius_coefficients = {}
            for idx in range(self.output_dim):
                _grand_coalition_prediction_node = self._grand_coalition_prediction[:, idx]
                masked_predictions_current_node = masked_predictions[:, idx]
                moebius_coefficients[idx], shapley_interactions[idx] = self._grapshapiq_routine(
                    moebius_interactions,
                    masked_predictions_current_node,
                    moebius_coalition_lookup,
                    efficiency_routine,
                    incomplete_neighborhoods,
                    incomplete_neighborhoods_lookup,
                    capped_interaction_size,
                    order,
                    _grand_coalition_prediction_node,
                )

        return moebius_coefficients, shapley_interactions

    def _grapshapiq_routine(
        self,
        moebius_interactions,  # noqa: ANN001
        masked_predictions,  # noqa: ANN001
        moebius_coalition_lookup,  # noqa: ANN001
        efficiency_routine,  # noqa: ANN001
        incomplete_neighborhoods,  # noqa: ANN001
        incomplete_neighborhoods_lookup,  # noqa: ANN001
        max_interaction_size,  # noqa: ANN001
        order,  # noqa: ANN001
        _grand_coalition_prediction_node,  # noqa: ANN001
    ) -> tuple:
        """Run graphshapiq routine."""
        # Compute the Möbius transform
        moebius_coefficients = self.compute_moebius_transform(
            coalitions=moebius_interactions,
            coalition_predictions=masked_predictions,
            coalition_lookup=moebius_coalition_lookup,
        )

        moebius_coefficients.sparsify(self.sparsify_threshold)

        if efficiency_routine:
            # If efficiency_routine is True, then distribute the efficiency gap onto the neighborhood interaction.
            additional_moebius_coefficients = self._efficiency_routine(
                masked_predictions,
                moebius_coefficients,
                incomplete_neighborhoods,
                incomplete_neighborhoods_lookup,
                max_interaction_size,
                _grand_coalition_prediction_node,
            )
            final_moebius_coefficients = moebius_coefficients + additional_moebius_coefficients
        else:
            final_moebius_coefficients = moebius_coefficients

        # Convert the Möbius interactions to Shapley interactions
        converter = MoebiusConverter(
            moebius_coefficients=final_moebius_coefficients,
        )

        interactions = converter.compute(index="k-SII", order=order)
        interactions.sparsify(self.sparsify_threshold)

        return copy.copy(final_moebius_coefficients), copy.copy(interactions)

    def _get_all_coalitions(self, max_interaction_size: int, *, efficiency_routine: bool) -> tuple:
        """Collects all coalitions that will be evaluated.

        i.e. coalitions with non-zero Möbius
        transform of maximum size max_interaction_size. If efficiency_routine is True, then all
        neighborhoods are added to the collection.

        Args:
            max_interaction_size: The maximum interaction size considered
            efficiency_routine: If True, then all neighborhoods will be added to the collection.

        Returns:
            moebius_interactions: The set of Möbius interactions considered
            incomplete_neighborhoods: The set of neighborhoods that are added due to efficiency_routine
        """
        # Get non-zero Möbius values based on the neighborhood
        incomplete_neighborhoods = set()
        moebius_interactions = set()

        for node in self.neighbors:
            # Collect all non-zero Möbius interactions up to order max_interaction_size
            # For these, game evaluations are required
            for interaction in powerset(self.neighbors[node], max_size=max_interaction_size):
                moebius_interactions.add(interaction)
            if efficiency_routine and len(self.neighbors[node]) > max_interaction_size:
                # If not all interactions were considered, add the complete the neighborhood interaction
                # This is required for efficiency later on
                incomplete_neighborhoods.add(self.neighbors[node])

        return moebius_interactions, incomplete_neighborhoods
