"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues


class TestGraphSHAPIQ:
    """Test class for GraphSHAPIQ.__init__."""

    def test_init_with_gcn_model(self, gcn_graphshapiq, gcn_graph_game):
        """Test that GraphSHAPIQ initializes correctly with a GCN model."""
        assert gcn_graphshapiq.n_players == gcn_graph_game.n_players
        assert gcn_graphshapiq.edge_index is gcn_graph_game.edge_index
        assert gcn_graphshapiq.l_hop_distance == gcn_graph_game.max_neighborhood_size
        assert gcn_graphshapiq.output_dim == gcn_graph_game.output_dim
        assert gcn_graphshapiq.last_n_model_calls is None
        assert gcn_graphshapiq._grand_coalition_prediction is not None
        assert isinstance(gcn_graphshapiq.neighbors, dict)
        assert len(gcn_graphshapiq.neighbors) == gcn_graph_game.n_players
        assert 1 <= gcn_graphshapiq.max_size_neighbors <= gcn_graph_game.n_players
        assert gcn_graphshapiq.total_budget > 0
        assert isinstance(gcn_graphshapiq.budget_estimated, bool)

    def test_init_with_gin_model(self, gin_graphshapiq, gin_graph_game):
        """Test that GraphSHAPIQ initializes correctly with a GIN model."""
        assert gin_graphshapiq.n_players == gin_graph_game.n_players
        assert gin_graphshapiq.last_n_model_calls is None
        assert isinstance(gin_graphshapiq.neighbors, dict)
        assert len(gin_graphshapiq.neighbors) == gin_graph_game.n_players

    def test_init_with_gat_model(self, gat_graphshapiq, gat_graph_game):
        """Test that GraphSHAPIQ initializes correctly with a GAT model."""
        assert gat_graphshapiq.n_players == gat_graph_game.n_players
        assert gat_graphshapiq.last_n_model_calls is None
        assert isinstance(gat_graphshapiq.neighbors, dict)
        assert len(gat_graphshapiq.neighbors) == gat_graph_game.n_players

    def test_init_with_small_graph(self, gcn_graphshapiq_small, gcn_graph_game_small):
        """Test that GraphSHAPIQ initializes correctly with a small graph."""
        assert gcn_graphshapiq_small.n_players == gcn_graph_game_small.n_players
        assert len(gcn_graphshapiq_small.neighbors) == gcn_graph_game_small.n_players
        assert gcn_graphshapiq_small.total_budget > 0

    def test_init_with_disconnected_graph(
        self, gcn_graphshapiq_disconnected, gcn_graph_game_disconnected
    ):
        """Test that GraphSHAPIQ initializes correctly with a disconnected graph."""
        assert gcn_graphshapiq_disconnected.n_players == gcn_graph_game_disconnected.n_players
        assert len(gcn_graphshapiq_disconnected.neighbors) == gcn_graph_game_disconnected.n_players
        assert gcn_graphshapiq_disconnected.total_budget > 0

    def test_init_with_single_node_graph(self, gcn_graphshapiq_single_node):
        """Test that GraphSHAPIQ initializes correctly with a single node graph."""
        assert gcn_graphshapiq_single_node.n_players == 1
        assert gcn_graphshapiq_single_node.max_size_neighbors == 1
        assert len(gcn_graphshapiq_single_node.neighbors) == 1
        assert gcn_graphshapiq_single_node.total_budget > 0

    def test_get_k_neighborhood_simple_graph(self, gcn_graphshapiq):
        """Test _get_k_neighborhood on a cycle graph with l_hop_distance=2."""
        # simple_graph is a cycle 0↔1↔2↔3↔0, all nodes reachable within depth 2
        assert gcn_graphshapiq._get_k_neighborhood(0) == (0, 1, 2, 3)
        assert gcn_graphshapiq._get_k_neighborhood(1) == (0, 1, 2, 3)
        assert gcn_graphshapiq._get_k_neighborhood(2) == (0, 1, 2, 3)
        assert gcn_graphshapiq._get_k_neighborhood(3) == (0, 1, 2, 3)

    def test_get_k_neighborhood_small_graph(self, gcn_graphshapiq_small):
        """Test _get_k_neighborhood on a chain graph with l_hop_distance=2."""
        # small_graph is a chain 0↔1↔2↔3↔4
        assert gcn_graphshapiq_small._get_k_neighborhood(0) == (0, 1, 2)
        assert gcn_graphshapiq_small._get_k_neighborhood(2) == (0, 1, 2, 3, 4)
        assert gcn_graphshapiq_small._get_k_neighborhood(4) == (2, 3, 4)

    def test_get_k_neighborhood_disconnected_graph(self, gcn_graphshapiq_disconnected):
        """Test _get_k_neighborhood on a disconnected graph with l_hop_distance=2."""
        # disconnected_graph: component 1 is 0↔1, component 2 is 2↔3↔4
        assert gcn_graphshapiq_disconnected._get_k_neighborhood(0) == (0, 1)
        assert gcn_graphshapiq_disconnected._get_k_neighborhood(1) == (0, 1)
        assert gcn_graphshapiq_disconnected._get_k_neighborhood(2) == (2, 3, 4)
        assert gcn_graphshapiq_disconnected._get_k_neighborhood(3) == (2, 3, 4)
        assert gcn_graphshapiq_disconnected._get_k_neighborhood(4) == (2, 3, 4)

    def test_get_k_neighborhood_single_node_graph(self, gcn_graphshapiq_single_node):
        """Test _get_k_neighborhood on a single node graph with l_hop_distance=2."""
        assert gcn_graphshapiq_single_node._get_k_neighborhood(0) == (0,)

    def test_get_neighborhoods_simple_graph(self, gcn_graphshapiq):
        """Test that _get_neighborhoods returns correct structure for simple graph."""
        assert len(gcn_graphshapiq.neighbors) == gcn_graphshapiq.n_players
        assert gcn_graphshapiq.neighbors[0] == (0, 1, 2, 3)
        assert gcn_graphshapiq.neighbors[1] == (0, 1, 2, 3)
        assert gcn_graphshapiq.neighbors[2] == (0, 1, 2, 3)
        assert gcn_graphshapiq.neighbors[3] == (0, 1, 2, 3)
        assert gcn_graphshapiq.max_size_neighbors == 4  # all nodes reachable

    def test_get_neighborhoods_small_graph(self, gcn_graphshapiq_small):
        """Test that _get_neighborhoods returns correct structure for chain graph."""
        assert len(gcn_graphshapiq_small.neighbors) == gcn_graphshapiq_small.n_players
        assert gcn_graphshapiq_small.neighbors[0] == (0, 1, 2)
        assert gcn_graphshapiq_small.neighbors[2] == (0, 1, 2, 3, 4)
        assert gcn_graphshapiq_small.neighbors[4] == (2, 3, 4)
        assert gcn_graphshapiq_small.max_size_neighbors == 5  # node 2 reaches all

    def test_get_neighborhoods_disconnected_graph(self, gcn_graphshapiq_disconnected):
        """Test that _get_neighborhoods returns correct structure for disconnected graph."""
        assert len(gcn_graphshapiq_disconnected.neighbors) == gcn_graphshapiq_disconnected.n_players
        assert gcn_graphshapiq_disconnected.neighbors[0] == (0, 1)
        assert gcn_graphshapiq_disconnected.neighbors[2] == (2, 3, 4)
        assert gcn_graphshapiq_disconnected.max_size_neighbors == 3  # component 2 is larger

    def test_get_neighborhoods_single_node_graph(self, gcn_graphshapiq_single_node):
        """Test that _get_neighborhoods returns correct structure for single node graph."""
        assert len(gcn_graphshapiq_single_node.neighbors) == 1
        assert gcn_graphshapiq_single_node.neighbors[0] == (0,)
        assert gcn_graphshapiq_single_node.max_size_neighbors == 1

    def test_get_all_coalitions_complete_simple_graph(self, gcn_graphshapiq):
        """Test _get_all_coalitions with max_subset_size=max_size_neighbors on simple graph."""
        moebius_interactions, incomplete_neighborhoods = gcn_graphshapiq._get_all_coalitions(
            max_subset_size=gcn_graphshapiq.max_size_neighbors,
            efficiency_routine=True,
        )
        # No incomplete neighborhoods since max_subset_size == max_size_neighbors
        assert incomplete_neighborhoods == set()
        # Largest tuple size equals max_size_neighbors
        assert max(len(t) for t in moebius_interactions) == gcn_graphshapiq.max_size_neighbors
        # Empty coalition always present
        assert () in moebius_interactions
        # Full neighborhood present
        assert (0, 1, 2, 3) in moebius_interactions

    def test_get_all_coalitions_incomplete_simple_graph(self, gcn_graphshapiq):
        """Test _get_all_coalitions with max_subset_size=1 on simple graph."""
        moebius_interactions, incomplete_neighborhoods = gcn_graphshapiq._get_all_coalitions(
            max_subset_size=1,
            efficiency_routine=True,
        )
        # All neighborhoods have size 4 > 1, so all are incomplete
        assert (0, 1, 2, 3) in incomplete_neighborhoods
        # Largest tuple size equals max_subset_size
        assert max(len(t) for t in moebius_interactions) == 1
        # Only singletons and empty coalition present
        assert () in moebius_interactions
        assert (0,) in moebius_interactions
        assert (1,) in moebius_interactions
        assert (2,) in moebius_interactions
        assert (3,) in moebius_interactions
        # No pairs present
        assert (0, 1) not in moebius_interactions

    def test_get_all_coalitions_complete_disconnected_graph(self, gcn_graphshapiq_disconnected):
        """Test _get_all_coalitions with max_subset_size=max_size_neighbors on disconnected graph."""
        moebius_interactions, incomplete_neighborhoods = (
            gcn_graphshapiq_disconnected._get_all_coalitions(
                max_subset_size=gcn_graphshapiq_disconnected.max_size_neighbors,
                efficiency_routine=True,
            )
        )
        assert incomplete_neighborhoods == set()
        assert (
            max(len(t) for t in moebius_interactions)
            == gcn_graphshapiq_disconnected.max_size_neighbors
        )
        # Nodes from different components never appear together
        assert (0, 2) not in moebius_interactions
        assert (0, 3) not in moebius_interactions
        assert (1, 2) not in moebius_interactions

    def test_get_all_coalitions_incomplete_disconnected_graph(self, gcn_graphshapiq_disconnected):
        """Test _get_all_coalitions with max_subset_size=1 on disconnected graph."""
        moebius_interactions, incomplete_neighborhoods = (
            gcn_graphshapiq_disconnected._get_all_coalitions(
                max_subset_size=1,
                efficiency_routine=True,
            )
        )
        # Both components have neighborhood size > 1, so both are incomplete
        assert len(incomplete_neighborhoods) == 2
        assert max(len(t) for t in moebius_interactions) == 1
        # Nodes from different components never appear together
        assert (0, 2) not in moebius_interactions

    def test_get_all_coalitions_single_node_graph(self, gcn_graphshapiq_single_node):
        """Test _get_all_coalitions on single node graph — always complete."""
        moebius_interactions, incomplete_neighborhoods = (
            gcn_graphshapiq_single_node._get_all_coalitions(
                max_subset_size=1,
                efficiency_routine=True,
            )
        )
        # Single node neighborhood is never incomplete
        assert incomplete_neighborhoods == set()
        assert () in moebius_interactions
        assert (0,) in moebius_interactions

    def test_get_all_coalitions_efficiency_routine_false(self, gcn_graphshapiq):
        """Test that incomplete_neighborhoods is always empty when efficiency_routine=False."""
        moebius_interactions, incomplete_neighborhoods = gcn_graphshapiq._get_all_coalitions(
            max_subset_size=1,
            efficiency_routine=False,
        )
        assert incomplete_neighborhoods == set()
        # moebius_interactions still populated normally
        assert () in moebius_interactions
        assert (0,) in moebius_interactions

    def test_convert_to_coalition_matrix_shape(self, gcn_graphshapiq):
        """Test that the coalition matrix has the correct shape."""
        coalitions = {(), (0,), (1,), (0, 1)}
        matrix, _ = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        assert matrix.shape == (len(coalitions), gcn_graphshapiq.n_players)

    def test_convert_to_coalition_matrix_empty_coalition(self, gcn_graphshapiq):
        """Test that the empty coalition maps to an all-zero row."""
        coalitions = {()}
        matrix, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        assert np.all(matrix[lookup[()]] == 0)

    def test_convert_to_coalition_matrix_full_coalition(self, gcn_graphshapiq):
        """Test that the full coalition maps to an all-ones row."""
        full = tuple(range(gcn_graphshapiq.n_players))
        coalitions = {full}
        matrix, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        assert np.all(matrix[lookup[full]] == 1)

    def test_convert_to_coalition_matrix_partial_coalition(self, gcn_graphshapiq):
        """Test that a partial coalition correctly sets the right positions to 1."""
        coalition = (0, 2)
        coalitions = {coalition}
        matrix, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        row = matrix[lookup[coalition]]
        assert row[0] == 1
        assert row[1] == 0
        assert row[2] == 1
        assert row[3] == 0

    def test_convert_to_coalition_matrix_lookup_no_shift(self, gcn_graphshapiq):
        """Test that lookup indices start at 0 without a shift."""
        coalitions = {(), (0,), (1,), (0, 1)}
        _, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        assert min(lookup.values()) == 0
        assert max(lookup.values()) == len(coalitions) - 1

    def test_convert_to_coalition_matrix_lookup_with_shift(self, gcn_graphshapiq):
        """Test that lookup indices are correctly offset by lookup_shift."""
        coalitions = {(), (0,), (1,), (0, 1)}
        shift = 10
        _, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions, lookup_shift=shift)
        assert min(lookup.values()) == shift
        assert max(lookup.values()) == shift + len(coalitions) - 1

    def test_convert_to_coalition_matrix_lookup_consistency(self, gcn_graphshapiq):
        """Test that each lookup index points to the correct row in the matrix."""
        coalitions = {(), (0,), (2,), (0, 2)}
        matrix, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        for coalition, idx in lookup.items():
            row = matrix[idx]
            for player in range(gcn_graphshapiq.n_players):
                if player in coalition:
                    assert row[player] == 1
                else:
                    assert row[player] == 0

    def test_convert_to_coalition_matrix_single_coalition(self, gcn_graphshapiq):
        """Test with a single coalition input."""
        coalitions = {(0, 1)}
        matrix, lookup = gcn_graphshapiq._convert_to_coalition_matrix(coalitions)
        assert matrix.shape == (1, gcn_graphshapiq.n_players)
        assert len(lookup) == 1

    def test_compute_moebius_transform_simple(self, gcn_graphshapiq):
        """Test Möbius transform with hand-crafted values for 2 players."""
        # Define coalitions and known game values
        # v(()) = 0, v((0,)) = 1, v((1,)) = 2, v((0,1)) = 4
        coalitions = {(), (0,), (1,), (0, 1)}
        coalition_predictions = np.array([0.0, 1.0, 2.0, 4.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        # m(()) = v(()) = 0
        assert result[()] == pytest.approx(0.0)
        # m((0,)) = v((0,)) - v(()) = 1 - 0 = 1
        assert result[(0,)] == pytest.approx(1.0)
        # m((1,)) = v((1,)) - v(()) = 2 - 0 = 2
        assert result[(1,)] == pytest.approx(2.0)
        # m((0,1)) = v((0,1)) - v((0,)) - v((1,)) + v(()) = 4 - 1 - 2 + 0 = 1
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_compute_moebius_transform_returns_interaction_values(self, gcn_graphshapiq):
        """Test that the return type is InteractionValues with correct attributes."""
        coalitions = {(), (0,), (1,), (0, 1)}
        coalition_predictions = np.array([0.0, 1.0, 2.0, 4.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        assert isinstance(result, InteractionValues)
        assert result.n_players == gcn_graphshapiq.n_players
        assert result.index == "Moebius"
        assert result.min_order == 0
        assert result.max_order == gcn_graphshapiq.n_players

    def test_compute_moebius_transform_baseline_value(self, gcn_graphshapiq):
        """Test that baseline_value equals m(()) = v(())."""
        coalitions = {(), (0,), (1,), (0, 1)}
        coalition_predictions = np.array([0.5, 1.0, 2.0, 4.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        # baseline_value = m(()) = v(()) = 0.5
        assert result.baseline_value == pytest.approx(0.5)

    def test_compute_moebius_transform_empty_coalition_only(self, gcn_graphshapiq):
        """Test Möbius transform with only the empty coalition."""
        coalitions = {()}
        coalition_predictions = np.array([3.0])
        coalition_lookup = {(): 0}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        assert result[()] == pytest.approx(3.0)
        assert result.baseline_value == pytest.approx(3.0)

    def test_compute_moebius_transform_additive_game(self, gcn_graphshapiq):
        """Test Möbius transform on an additive game — no interactions, so m(S)=0 for |S|>1."""
        # Additive game: v(S) = sum of player values, no synergy
        # v(()) = 0, v((0,)) = 1, v((1,)) = 2, v((0,1)) = 3
        coalitions = {(), (0,), (1,), (0, 1)}
        coalition_predictions = np.array([0.0, 1.0, 2.0, 3.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        # m((0,1)) = 3 - 1 - 2 + 0 = 0 — no interaction in additive game
        assert result[(0, 1)] == pytest.approx(0.0)
        assert result[(0,)] == pytest.approx(1.0)
        assert result[(1,)] == pytest.approx(2.0)

    def test_compute_moebius_transform_three_players(self, gcn_graphshapiq):
        """Test Möbius transform with three players and known values."""
        # v(()) = 0, v((0,)) = 1, v((1,)) = 1, v((2,)) = 1
        # v((0,1)) = 2, v((0,2)) = 2, v((1,2)) = 2, v((0,1,2)) = 6
        coalitions = {(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)}
        coalition_predictions = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 6.0])
        coalition_lookup = {
            (): 0,
            (0,): 1,
            (1,): 2,
            (2,): 3,
            (0, 1): 4,
            (0, 2): 5,
            (1, 2): 6,
            (0, 1, 2): 7,
        }

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        # m((0,1)) = 2 - 1 - 1 + 0 = 0
        assert result[(0, 1)] == pytest.approx(0.0)
        # m((0,1,2)) = 6 - 2 - 2 - 2 + 1 + 1 + 1 - 0 = 3
        assert result[(0, 1, 2)] == pytest.approx(3.0)

    @pytest.fixture
    def efficiency_routine_inputs(self, gcn_graphshapiq):
        """Shared setup for _efficiency_routine tests with 3 players and multiple incomplete neighborhoods."""
        coalitions = {(), (0,), (1,), (2,)}
        coalition_predictions = np.array([0.0, 1.0, 2.0, 3.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (2,): 3}

        moebius_coefficients = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        return {
            "moebius_coefficients": moebius_coefficients,
            "incomplete_neighborhoods": {(0, 1), (1, 2), (0, 1, 2)},
            "incomplete_neighborhoods_lookup": {(0, 1): 4, (1, 2): 5, (0, 1, 2): 6},
            "masked_predictions": np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0]),
            "grand_coalition_prediction": np.array(12.0),
        }

    def test_compute_moebius_transform_efficiency_axiom(self, gcn_graphshapiq):
        """Test that sum of all Möbius coefficients equals v(grand coalition)."""
        coalitions = {(), (0,), (1,), (0, 1)}
        coalition_predictions = np.array([0.0, 1.0, 2.0, 4.0])
        coalition_lookup = {(): 0, (0,): 1, (1,): 2, (0, 1): 3}

        result = gcn_graphshapiq.compute_moebius_transform(
            coalitions=coalitions,
            coalition_predictions=coalition_predictions,
            coalition_lookup=coalition_lookup,
        )

        # sum of all Möbius coefficients = v(grand coalition) = 4
        assert np.sum(result.values) == pytest.approx(coalition_predictions[-1])

    def test_efficiency_routine_returns_interaction_values(
        self, gcn_graphshapiq, efficiency_routine_inputs
    ):
        """Test that _efficiency_routine returns an InteractionValues object."""
        result = gcn_graphshapiq._efficiency_routine(
            masked_predictions=efficiency_routine_inputs["masked_predictions"],
            moebius_coefficients=efficiency_routine_inputs["moebius_coefficients"],
            incomplete_neighborhoods=efficiency_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=efficiency_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            grand_coalition_prediction_node=efficiency_routine_inputs["grand_coalition_prediction"],
        )

        assert isinstance(result, InteractionValues)
        assert result.index == "Moebius"
        assert result.n_players == gcn_graphshapiq.n_players
        assert result.baseline_value == pytest.approx(0.0)

    def test_efficiency_routine_gap_multiple_incomplete_neighborhoods(
        self, gcn_graphshapiq, efficiency_routine_inputs
    ):
        """Test that gaps are correctly computed for multiple incomplete neighborhoods."""
        result = gcn_graphshapiq._efficiency_routine(
            masked_predictions=efficiency_routine_inputs["masked_predictions"],
            moebius_coefficients=efficiency_routine_inputs["moebius_coefficients"],
            incomplete_neighborhoods=efficiency_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=efficiency_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            grand_coalition_prediction_node=efficiency_routine_inputs["grand_coalition_prediction"],
        )

        # gap(0,1) = v((0,1)) - (m(()) + m((0,)) + m((1,))) = 5 - 3 = 2
        assert result[(0, 1)] == pytest.approx(2.0)
        # gap(1,2) = v((1,2)) - (m(()) + m((1,)) + m((2,))) = 7 - 5 = 2
        # (0,1) is not a subset of (1,2) so no nested contribution
        assert result[(1, 2)] == pytest.approx(2.0)
        # final adjustment: 12 - (0+1+2+3) - (2+2) = 2
        assert result[(0, 1, 2)] == pytest.approx(2.0)

    def test_efficiency_routine_enforces_efficiency(
        self, gcn_graphshapiq, efficiency_routine_inputs
    ):
        """Test that sum of all Möbius coefficients equals grand coalition prediction."""
        result = gcn_graphshapiq._efficiency_routine(
            masked_predictions=efficiency_routine_inputs["masked_predictions"],
            moebius_coefficients=efficiency_routine_inputs["moebius_coefficients"],
            incomplete_neighborhoods=efficiency_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=efficiency_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            grand_coalition_prediction_node=efficiency_routine_inputs["grand_coalition_prediction"],
        )

        total = np.sum(efficiency_routine_inputs["moebius_coefficients"].values) + np.sum(
            result.values
        )
        assert total == pytest.approx(
            float(efficiency_routine_inputs["grand_coalition_prediction"])
        )

    def test_efficiency_routine_lookup_contains_all_incomplete_neighborhoods(
        self, gcn_graphshapiq, efficiency_routine_inputs
    ):
        """Test that the returned lookup contains exactly all incomplete neighborhoods."""
        result = gcn_graphshapiq._efficiency_routine(
            masked_predictions=efficiency_routine_inputs["masked_predictions"],
            moebius_coefficients=efficiency_routine_inputs["moebius_coefficients"],
            incomplete_neighborhoods=efficiency_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=efficiency_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            grand_coalition_prediction_node=efficiency_routine_inputs["grand_coalition_prediction"],
        )

        assert (0, 1) in result.interaction_lookup
        assert (1, 2) in result.interaction_lookup
        assert (0, 1, 2) in result.interaction_lookup
        assert (
            len(result.interaction_lookup)
            == len(efficiency_routine_inputs["incomplete_neighborhoods"]) + 1
        )

    @pytest.fixture
    def graphshapiq_routine_inputs(self, gcn_graphshapiq):
        """Shared setup for _graphshapiq_routine tests with 3 players."""
        # Möbius coalitions: all subsets up to size 1
        moebius_interactions = {(), (0,), (1,), (2,)}
        moebius_coalition_lookup = {(): 0, (0,): 1, (1,): 2, (2,): 3}

        # Incomplete neighborhoods
        incomplete_neighborhoods = {(0, 1), (0, 1, 2)}
        incomplete_neighborhoods_lookup = {(0, 1): 4, (0, 1, 2): 5}

        # All predictions: moebius coalitions + incomplete neighborhoods
        masked_predictions = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 15.0])
        grand_coalition_prediction = np.array(15.0)

        return {
            "moebius_interactions": moebius_interactions,
            "moebius_coalition_lookup": moebius_coalition_lookup,
            "incomplete_neighborhoods": incomplete_neighborhoods,
            "incomplete_neighborhoods_lookup": incomplete_neighborhoods_lookup,
            "masked_predictions": masked_predictions,
            "grand_coalition_prediction": grand_coalition_prediction,
        }

    def test_graphshapiq_routine_return_types(self, gcn_graphshapiq, graphshapiq_routine_inputs):
        """Test that _graphshapiq_routine returns a tuple of two InteractionValues."""
        final_moebius, interactions = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
        )

        assert isinstance(final_moebius, InteractionValues)
        assert isinstance(interactions, InteractionValues)

    def test_graphshapiq_routine_moebius_attributes(
        self, gcn_graphshapiq, graphshapiq_routine_inputs
    ):
        """Test that final Möbius coefficients have correct attributes."""
        final_moebius, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
        )

        assert final_moebius.index == "Moebius"
        assert final_moebius.n_players == gcn_graphshapiq.n_players

    def test_graphshapiq_routine_interactions_attributes(
        self, gcn_graphshapiq, graphshapiq_routine_inputs
    ):
        """Test that Shapley interactions have correct attributes."""
        _, interactions = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
            index="k-SII",
        )

        assert interactions.index == "k-SII"
        assert interactions.n_players == gcn_graphshapiq.n_players

    def test_graphshapiq_routine_efficiency_axiom_with_routine(
        self, gcn_graphshapiq, graphshapiq_routine_inputs
    ):
        """Test that efficiency axiom holds when efficiency_routine=True."""
        final_moebius, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
        )

        # sum of all final Möbius coefficients = v(grand coalition) = 15
        assert np.sum(final_moebius.values) == pytest.approx(
            float(graphshapiq_routine_inputs["grand_coalition_prediction"])
        )

    def test_graphshapiq_routine_no_efficiency_routine(
        self, gcn_graphshapiq, graphshapiq_routine_inputs
    ):
        """Test that without efficiency_routine, final_moebius equals raw Möbius coefficients."""
        final_moebius_with, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
        )

        final_moebius_without, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=graphshapiq_routine_inputs["incomplete_neighborhoods"],
            incomplete_neighborhoods_lookup=graphshapiq_routine_inputs[
                "incomplete_neighborhoods_lookup"
            ],
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=False,
        )

        # Without efficiency routine, sum of Möbius coefficients does not equal grand coalition
        assert np.sum(final_moebius_without.values) != pytest.approx(
            float(graphshapiq_routine_inputs["grand_coalition_prediction"])
        )
        # And the two results differ
        assert np.sum(final_moebius_with.values) != pytest.approx(
            np.sum(final_moebius_without.values)
        )

    def test_graphshapiq_routine_no_incomplete_neighborhoods(
        self, gcn_graphshapiq, graphshapiq_routine_inputs
    ):
        """Test that efficiency_routine=True with no incomplete neighborhoods behaves like False."""
        final_moebius_no_incomplete, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=set(),  # no incomplete neighborhoods
            incomplete_neighborhoods_lookup={},
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=True,
        )

        final_moebius_false, _ = gcn_graphshapiq._graphshapiq_routine(
            moebius_interactions=graphshapiq_routine_inputs["moebius_interactions"],
            masked_predictions=graphshapiq_routine_inputs["masked_predictions"],
            moebius_coalition_lookup=graphshapiq_routine_inputs["moebius_coalition_lookup"],
            incomplete_neighborhoods=set(),
            incomplete_neighborhoods_lookup={},
            max_subset_size=1,
            order=3,
            grand_coalition_prediction_node=graphshapiq_routine_inputs[
                "grand_coalition_prediction"
            ],
            efficiency_routine=False,
        )

        # Both should give the same result since there are no incomplete neighborhoods
        assert np.sum(final_moebius_no_incomplete.values) == pytest.approx(
            np.sum(final_moebius_false.values)
        )

    def test_explain_single_output(self, gcn_graphshapiq):
        """Test that explain returns correct types and attributes for single output."""
        assert gcn_graphshapiq.last_n_model_calls is None
        moebius, interactions = gcn_graphshapiq.explain(index="SII")
        assert isinstance(moebius, InteractionValues)
        assert isinstance(interactions, InteractionValues)
        assert interactions.index == "SII"
        assert moebius.n_players == gcn_graphshapiq.n_players
        assert interactions.n_players == gcn_graphshapiq.n_players
        assert interactions.max_order == gcn_graphshapiq.n_players
        assert gcn_graphshapiq.last_n_model_calls is not None
        assert gcn_graphshapiq.last_n_model_calls > 0

    def test_explain_efficiency_axiom(self, gcn_graphshapiq):
        """Test that efficiency axiom holds on Möbius coefficients when efficiency_routine=True."""
        moebius, _ = gcn_graphshapiq.explain(efficiency_routine=True)
        grand_coalition_prediction = float(gcn_graphshapiq._grand_coalition_prediction[0])
        assert np.sum(moebius.values) == pytest.approx(grand_coalition_prediction, abs=1e-6)

    def test_explain_max_subset_size_none_defaults_to_max_size_neighbors(self, gcn_graphshapiq):
        """Test that max_subset_size=None defaults to max_size_neighbors."""
        moebius_default, _ = gcn_graphshapiq.explain(max_subset_size=None)
        moebius_explicit, _ = gcn_graphshapiq.explain(
            max_subset_size=gcn_graphshapiq.max_size_neighbors
        )
        assert np.sum(moebius_default.values) == pytest.approx(np.sum(moebius_explicit.values))

    def test_explain_max_subset_size_capped_at_max_size_neighbors(self, gcn_graphshapiq):
        """Test that max_subset_size larger than max_size_neighbors gets capped."""
        moebius_capped, _ = gcn_graphshapiq.explain(
            max_subset_size=gcn_graphshapiq.max_size_neighbors + 10
        )
        moebius_default, _ = gcn_graphshapiq.explain(max_subset_size=None)
        assert np.sum(moebius_capped.values) == pytest.approx(np.sum(moebius_default.values))

    def test_explain_without_efficiency_routine(self, gcn_graphshapiq_small):
        """Test that efficiency axiom does not hold when efficiency_routine=False and there are incomplete neighborhoods."""
        moebius, _ = gcn_graphshapiq_small.explain(
            max_subset_size=1,
            efficiency_routine=False,
        )
        grand_coalition_prediction = float(gcn_graphshapiq_small._grand_coalition_prediction[0])
        assert np.sum(moebius.values) != pytest.approx(grand_coalition_prediction, abs=1e-6)

    def test_explain_with_gin_model(self, gin_graphshapiq):
        """Test that explain works correctly with a GIN model."""
        moebius, interactions = gin_graphshapiq.explain()
        assert isinstance(moebius, InteractionValues)
        assert isinstance(interactions, InteractionValues)
        assert interactions.index == "k-SII"

    def test_explain_with_gat_model(self, gat_graphshapiq):
        """Test that explain works correctly with a GAT model."""
        moebius, interactions = gat_graphshapiq.explain()
        assert isinstance(moebius, InteractionValues)
        assert isinstance(interactions, InteractionValues)
        assert interactions.index == "k-SII"
