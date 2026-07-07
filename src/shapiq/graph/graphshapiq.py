"""This module contains the GraphSHAP-IQ class."""

from __future__ import annotations

import copy
import itertools
import logging
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np

from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from shapiq.game_theory.moebius_converter import ValidMoebiusConverterIndices

    from .base import GraphGame

logger = logging.getLogger(__name__)


class GraphSHAPIQ:
    """A class for computing Shapley interactions on graph-structured data using the Möbius transform.

    Attributes:
        edge_index: The edge index of the graph.
        n_players: Number of players (nodes) in the graph.
        l_hop_distance: Maximum size of neighborhoods to consider.
        neighbors: Dictionary mapping each node to its neighborhood.
        max_size_neighbors: Maximum size of any neighborhood in the graph.
        total_budget: Total number of coalitions to evaluate (or upper bound).
        budget_estimated: Whether the budget was estimated or computed exactly.
    """

    def __init__(self, game: GraphGame, *, verbose: bool = False) -> None:
        """Initialize the GraphSHAPIQ class.

        Args:
            game: The game object representing the graph and prediction function.
            verbose: If True, prints debug information.
        """
        self.last_n_model_calls: int | None = None
        self.edge_index = game.edge_index
        self.n_players = game.n_players
        self._graph: nx.Graph = self._build_graph()
        self.sparsify_threshold = 1e-10
        self.l_hop_distance = int(game.l_hop_distance)
        self._grand_coalition_set = game.grand_coalition_set
        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        self.game = game
        self._grand_coalition_prediction = game(np.ones(self.n_players, dtype=float))
        self.verbose = verbose

        # Compute or estimate total budget
        if self.max_size_neighbors <= 22:
            moebius_interactions, _ = self._get_all_coalitions(
                max_subset_size=self.max_size_neighbors,
                efficiency_routine=False,
            )
            self.total_budget = len(moebius_interactions)
            self.budget_estimated = False
        else:
            self.total_budget = 0
            for node in self.neighbors:
                self.total_budget += 2 ** len(self.neighbors[node])
            self.total_budget = min(2**self.n_players, self.total_budget)
            self.budget_estimated = True

        if verbose:
            logger.info("Max size neighbors: %d", self.max_size_neighbors)
            logger.info("Total budget: %d", self.total_budget)

    def _build_graph(self) -> nx.Graph:
        """Convert edge_index and nodes to an nx.Graph."""
        G: nx.Graph = nx.Graph()
        G.add_nodes_from(range(self.n_players))
        G.add_edges_from(self.edge_index.T.tolist())
        return G

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
        """Get the k-hop neighborhood of a node."""
        reachable = nx.single_source_shortest_path_length(
            self._graph, node, cutoff=self.l_hop_distance
        )
        return tuple(sorted(reachable.keys()))

    def compute_moebius_transform(
        self,
        coalitions: set[tuple[int, ...]],
        coalition_predictions: NDArray[np.floating],
        coalition_lookup: dict[tuple[int, ...], int],
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
        moebius_lookup: dict[tuple[int, ...], int] = {}

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

    def compute_moebius_transform_cpp(
        self,
        coalitions: set[tuple[int, ...]],
        coalition_predictions: NDArray[np.floating],
        coalition_lookup: dict[tuple[int, ...], int],
    ) -> InteractionValues:
        """C++-accelerated drop-in replacement for :meth:`compute_moebius_transform`.

        Computes the exact same Möbius coefficients as the pure-Python
        :meth:`compute_moebius_transform`, but delegates the hot double loop over
        coalitions and their subsets to the ``shapiq.graph.cext`` extension.

        Requires the ``shapiq.graph.cext`` extension to be compiled (e.g. via
        `uv pip install -e .`); otherwise an ``ImportError`` is raised.

        Args:
            coalitions: Set of coalitions (tuples of player indices).
            coalition_predictions: Predictions for each coalition.
            coalition_lookup: Mapping from coalition tuples to their indices.

        Returns:
            InteractionValues object containing the Möbius values.
        """
        from .cext import (
            compute_moebius_transform as _moebius_cpp,  # ty: ignore[unresolved-import]
        )

        # Fix a stable iteration order for the coalitions; this defines both the
        # output index i and the moebius_lookup, matching the Python reference.
        coalition_list = list(coalitions)

        # Build CSR arrays for the coalitions to evaluate. Sizes/offsets are
        # computed vectorized via cumsum, and the flat member array is filled in
        # one pass via fromiter — avoiding a Python-level list.extend() per
        # coalition (see shapiq.tree.interventional.explainer for the same
        # flatten-via-cumsum pattern).
        sizes = np.fromiter(
            (len(c) for c in coalition_list), dtype=np.int32, count=len(coalition_list)
        )
        offsets = np.empty(len(coalition_list) + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        members_flat = np.fromiter(
            itertools.chain.from_iterable(coalition_list), dtype=np.int32, count=int(offsets[-1])
        )

        # Build CSR arrays for coalition_lookup (keys + prediction row index).
        lookup_keys = list(coalition_lookup.keys())
        lookup_sizes = np.fromiter(
            (len(k) for k in lookup_keys), dtype=np.int32, count=len(lookup_keys)
        )
        lookup_offsets = np.empty(len(lookup_keys) + 1, dtype=np.int32)
        lookup_offsets[0] = 0
        np.cumsum(lookup_sizes, out=lookup_offsets[1:])
        lookup_members_flat = np.fromiter(
            itertools.chain.from_iterable(lookup_keys),
            dtype=np.int32,
            count=int(lookup_offsets[-1]),
        )
        lookup_indices = np.fromiter(
            coalition_lookup.values(), dtype=np.int32, count=len(coalition_lookup)
        )

        moebius_values = _moebius_cpp(
            members_flat,
            offsets,
            lookup_members_flat,
            lookup_offsets,
            lookup_indices,
            np.ascontiguousarray(coalition_predictions, dtype=np.float64),
        )

        moebius_lookup = {coalition: i for i, coalition in enumerate(coalition_list)}

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
        coalitions: set[tuple[int, ...]],
        lookup_shift: int = 0,
    ) -> tuple[NDArray[np.floating], dict[tuple[int, ...], int]]:
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
        coalition_lookup: dict[tuple[int, ...], int] = {}

        for i, coalition in enumerate(coalitions):
            coalition_matrix[i, list(coalition)] = 1
            coalition_lookup[coalition] = lookup_shift + i

        return coalition_matrix, coalition_lookup

    def _efficiency_routine(
        self,
        masked_predictions: NDArray[np.floating],
        moebius_coefficients: InteractionValues,
        incomplete_neighborhoods: set[tuple[int, ...]],
        incomplete_neighborhoods_lookup: dict[tuple[int, ...], int],
        max_subset_size: int,
        grand_coalition_prediction_node: NDArray[np.floating],
    ) -> InteractionValues:
        """Adjust Möbius coefficients to ensure efficiency.

        The efficiency axiom states that the sum of all Möbius coefficients must equal
        the grand coalition prediction, i.e.,
        :math:`\\sum_{S \\subseteq N} m(S) = v(N)`.

        When neighborhoods are incomplete, that is, when their size exceeds
        ``max_subset_size``, the Möbius transform is truncated and the efficiency axiom
        is violated. This routine corrects for this by computing a gap term for each
        incomplete neighborhood:

        :math:`gap(S) = v(S) - \\sum_{T \\subseteq S, |T| \\leq k} m(T)
        - \\sum_{S' \\subset S, S' \text{ incomplete}} gap(S')`

        where :math:`k` is ``max_subset_size``.

        Here, :math:`v(S)` is the game value of the incomplete neighborhood :math:`S`,
        :math:`m(T)` are the already-computed Möbius coefficients for subsets
        :math:`T \\subseteq S`, and :math:`gap(S')` are correction terms for smaller
        incomplete neighborhoods :math:`S' \\subset S`.

        Note:
            This method should only be called when ``incomplete_neighborhoods`` is non-empty.
            The returned ``InteractionValues`` has ``baseline_value=0.0`` since the
            corrections are gap terms only; the empty coalition contribution is already
            captured in the original ``moebius_coefficients``.

        Args:
            masked_predictions: Predictions for all coalitions, including incomplete
                neighborhoods. Indices are given by ``incomplete_neighborhoods_lookup``.
            moebius_coefficients: Current Möbius coefficients computed from the
                truncated powerset via ``compute_moebius_transform``.
            incomplete_neighborhoods: Set of neighborhood tuples whose size exceeds
                ``max_subset_size`` and therefore were not fully covered by the Möbius
                transform.
            incomplete_neighborhoods_lookup: Mapping from incomplete neighborhood tuples
                to their row indices in ``masked_predictions``.
            max_subset_size: Maximum subset size used in the Möbius transform. Subsets
                larger than this were not evaluated.
            grand_coalition_prediction_node: The model prediction for the grand coalition,
                that is, all nodes present, used to enforce the global efficiency constraint.

        Returns:
            InteractionValues containing the gap correction terms for each incomplete
            neighborhood, with ``baseline_value=0.0`` and ``index="Moebius"``.
        """
        incomplete_neighborhoods_sorted = cast(
            "list[tuple[int, ...]]", sorted(incomplete_neighborhoods, key=len)
        )
        n_incomplete_neighborhoods = len(incomplete_neighborhoods_lookup)
        additional_values = np.zeros(n_incomplete_neighborhoods, dtype=float)
        additional_lookup: dict[tuple[int, ...], int] = {}

        for i, neighborhood in enumerate(incomplete_neighborhoods_sorted):
            # Compute sum of Möbius coefficients for subsets of the neighborhood
            sum_coefficients = 0.0
            for interaction in powerset(neighborhood, max_size=max_subset_size):
                sum_coefficients += moebius_coefficients[interaction]

            # Add contributions from smaller incomplete neighborhoods
            for other_neighborhood in incomplete_neighborhoods_sorted:
                if set(other_neighborhood).issubset(neighborhood) and len(other_neighborhood) < len(
                    neighborhood
                ):
                    sum_coefficients += additional_values[additional_lookup[other_neighborhood]]

            # Compute and assign the efficiency gap
            gap = (
                masked_predictions[incomplete_neighborhoods_lookup[neighborhood]] - sum_coefficients
            )
            additional_values[i] = gap
            additional_lookup[neighborhood] = i

        # Maintain global efficiency only if an incomplete neighborhood is the grand coalition.
        if additional_lookup:
            grand_coalition = tuple(sorted(self._grand_coalition_set))

            if grand_coalition in additional_lookup:
                grand_idx = additional_lookup[grand_coalition]
                additional_values[grand_idx] = (
                    grand_coalition_prediction_node
                    - np.sum(moebius_coefficients.values)
                    - (np.sum(additional_values) - additional_values[grand_idx])
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
        max_subset_size: int | None = None,
        order: int | None = None,
        *,
        efficiency_routine: bool = True,
        index: ValidMoebiusConverterIndices = "k-SII",
        use_cpp: bool = True,
    ) -> tuple[InteractionValues, InteractionValues]:
        """Compute Shapley interactions for the graph.

        Args:
            max_subset_size: Maximum size of interactions to consider.
                If None, uses the maximum neighborhood size.
            order: Maximum order of interactions to return. If None, uses n_players.
            efficiency_routine: If True, ensures efficiency by adjusting neighborhood interactions.
            index: The type of the interaction values.
            use_cpp: If True, use the C++-accelerated Möbius transform
                (:meth:`compute_moebius_transform_cpp`). Requires the
                ``shapiq.graph.cext`` extension to be compiled. Defaults to ``True``.

        Returns:
            Tuple of (moebius_coefficients, shapley_interactions).
        """
        if order is None:
            order = self.n_players

        capped_interaction_size = min(
            self.max_size_neighbors,
            max_subset_size if max_subset_size is not None else self.max_size_neighbors,
        )

        # Get coalitions and incomplete neighborhoods
        moebius_interactions, incomplete_neighborhoods = self._get_all_coalitions(
            max_subset_size=capped_interaction_size,
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
        masked_predictions = self.game(all_coalitions.astype(bool))
        self.last_n_model_calls = int(all_coalitions.shape[0])

        moebius_coefficients, shapley_interactions = self._graphshapiq_routine(
            moebius_interactions,
            masked_predictions,
            moebius_coalition_lookup,
            incomplete_neighborhoods,
            incomplete_neighborhoods_lookup,
            capped_interaction_size,
            order,
            self._grand_coalition_prediction[0],
            index,
            efficiency_routine=efficiency_routine,
            use_cpp=use_cpp,
        )
        return moebius_coefficients, shapley_interactions

    def _graphshapiq_routine(
        self,
        moebius_interactions: set[tuple[int, ...]],
        masked_predictions: NDArray[np.floating],
        moebius_coalition_lookup: dict[tuple[int, ...], int],
        incomplete_neighborhoods: set[tuple[int, ...]],
        incomplete_neighborhoods_lookup: dict[tuple[int, ...], int],
        max_subset_size: int,
        order: int,
        grand_coalition_prediction_node: NDArray[np.floating],
        index: ValidMoebiusConverterIndices = "k-SII",
        *,
        efficiency_routine: bool,
        use_cpp: bool = False,
    ) -> tuple[InteractionValues, InteractionValues]:
        """Core routine for computing GraphSHAPIQ values.

        Args:
            moebius_interactions: Set of coalitions to compute Möbius values for.
            masked_predictions: Predictions for all coalitions.
            moebius_coalition_lookup: Lookup for Möbius coalitions.
            efficiency_routine: Whether to enforce efficiency.
            index: The type of shapley interaction.
            incomplete_neighborhoods: Neighborhoods not fully covered.
            incomplete_neighborhoods_lookup: Lookup for incomplete neighborhoods.
            max_subset_size: Maximum interaction size.
            order: Maximum order of interactions.
            grand_coalition_prediction_node: Prediction for the grand coalition.
            use_cpp: If True, use the C++-accelerated Möbius transform.

        Returns:
            Tuple of (final Möbius coefficients, Shapley interactions).
        """
        if use_cpp:
            moebius_coefficients = self.compute_moebius_transform_cpp(
                coalitions=moebius_interactions,
                coalition_predictions=masked_predictions,
                coalition_lookup=moebius_coalition_lookup,
            )
        else:
            moebius_coefficients = self.compute_moebius_transform(
                coalitions=moebius_interactions,
                coalition_predictions=masked_predictions,
                coalition_lookup=moebius_coalition_lookup,
            )
        moebius_coefficients.sparsify(self.sparsify_threshold)

        if efficiency_routine and incomplete_neighborhoods:
            additional_coefficients = self._efficiency_routine(
                masked_predictions,
                moebius_coefficients,
                incomplete_neighborhoods,
                incomplete_neighborhoods_lookup,
                max_subset_size,
                grand_coalition_prediction_node,
            )
            final_moebius = moebius_coefficients + additional_coefficients
        else:
            final_moebius = moebius_coefficients

        converter = MoebiusConverter(moebius_coefficients=final_moebius)
        interactions = converter.compute(index=index, order=order)
        interactions.sparsify(self.sparsify_threshold)

        return copy.deepcopy(final_moebius), copy.deepcopy(interactions)

    def _get_all_coalitions(
        self,
        max_subset_size: int,
        *,
        efficiency_routine: bool,
    ) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
        """Collect all coalitions for Möbius transform and efficiency adjustment.

        Args:
            max_subset_size: Maximum size of interactions to consider.
            efficiency_routine: If True, includes incomplete neighborhoods.

        Returns:
            Tuple of (moebius_interactions, incomplete_neighborhoods).
        """
        moebius_interactions: set[tuple[int, ...]] = set()
        incomplete_neighborhoods: set[tuple[int, ...]] = set()

        for node in self.neighbors:
            # Add all subsets of the neighborhood up to max_subset_size
            for interaction in powerset(
                self.neighbors[node],
                max_size=max_subset_size,
            ):
                moebius_interactions.add(interaction)

            # Add incomplete neighborhoods if efficiency_routine is enabled
            if efficiency_routine and len(self.neighbors[node]) > max_subset_size:
                incomplete_neighborhoods.add(tuple(self.neighbors[node]))

        return moebius_interactions, incomplete_neighborhoods
