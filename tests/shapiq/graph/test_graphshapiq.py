"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

from __future__ import annotations

import pytest

from shapiq.graph.base import GraphGame
from shapiq.graph.graphshapiq import GraphSHAPIQ


class TestGraphSHAPIQ:
    """Test class for GraphSHAPIQ.__init__."""

    @pytest.fixture
    def gcn_graph_game(self, gcn_model, simple_graph):
        return GraphGame(
            model=gcn_model, x_graph=simple_graph, task="regression", baseline_strategy="average"
        )

    @pytest.fixture
    def gin_graph_game(self, gin_model, simple_graph):
        return GraphGame(
            model=gin_model, x_graph=simple_graph, task="regression", baseline_strategy="average"
        )

    @pytest.fixture
    def gat_graph_game(self, gat_model, simple_graph):
        return GraphGame(
            model=gat_model, x_graph=simple_graph, task="regression", baseline_strategy="average"
        )

    @pytest.fixture
    def gcn_graph_game_small(self, gcn_model, small_graph):
        return GraphGame(
            model=gcn_model, x_graph=small_graph, task="regression", baseline_strategy="average"
        )

    @pytest.fixture
    def gcn_graph_game_disconnected(self, gcn_model, disconnected_graph):
        return GraphGame(
            model=gcn_model,
            x_graph=disconnected_graph,
            task="regression",
            baseline_strategy="average",
        )

    @pytest.fixture
    def gcn_graph_game_single_node(self, gcn_model, single_node_graph):
        return GraphGame(
            model=gcn_model,
            x_graph=single_node_graph,
            task="regression",
            baseline_strategy="average",
        )

    @pytest.fixture
    def gcn_graphshapiq(self, gcn_graph_game):
        return GraphSHAPIQ(game=gcn_graph_game)

    @pytest.fixture
    def gin_graphshapiq(self, gin_graph_game):
        return GraphSHAPIQ(game=gin_graph_game)

    @pytest.fixture
    def gat_graphshapiq(self, gat_graph_game):
        return GraphSHAPIQ(game=gat_graph_game)

    @pytest.fixture
    def gcn_graphshapiq_small(self, gcn_graph_game_small):
        return GraphSHAPIQ(game=gcn_graph_game_small)

    @pytest.fixture
    def gcn_graphshapiq_disconnected(self, gcn_graph_game_disconnected):
        return GraphSHAPIQ(game=gcn_graph_game_disconnected)

    @pytest.fixture
    def gcn_graphshapiq_single_node(self, gcn_graph_game_single_node):
        return GraphSHAPIQ(game=gcn_graph_game_single_node)

    def test_init_with_gcn_model(self, gcn_graphshapiq, gcn_graph_game):
        """Test that GraphSHAPIQ initializes correctly with a GCN model."""
        assert gcn_graphshapiq.n_players == gcn_graph_game.n_players
        assert gcn_graphshapiq.edge_index is gcn_graph_game.edge_index
        assert gcn_graphshapiq.max_neighborhood_size == gcn_graph_game.max_neighborhood_size
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
