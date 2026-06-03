"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

from __future__ import annotations


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
