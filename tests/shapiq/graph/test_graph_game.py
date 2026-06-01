"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

import pytest

from shapiq.graph.base import GraphGame
from tests.shapiq.graph.fixtures.gnn_models import (
    gcn_model,
    gin_model,
    gat_model,
    gcn_model_classification,
)
from tests.shapiq.graph.fixtures.graph_data import simple_graph, small_graph

class TestGraphGame:
    """Test class for GraphGame."""

    @pytest.fixture
    def gcn_graph_game(self, gcn_model, simple_graph):
        """Create a GraphGame instance with a GCN model and a simple graph."""
        return GraphGame(
            model=gcn_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="average",
        )

    @pytest.fixture
    def gin_graph_game(self, gin_model, simple_graph):
        """Create a GraphGame instance with a GIN model and a simple graph."""
        return GraphGame(
            model=gin_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="average",
        )

    @pytest.fixture
    def gat_graph_game(self, gat_model, simple_graph):
        """Create a GraphGame instance with a GAT model and a simple graph."""
        return GraphGame(
            model=gat_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="average",
        )

    @pytest.fixture
    def gcn_graph_game_classification(self, gcn_model_classification, simple_graph):
        """Create a GraphGame instance for classification with a GCN model."""
        return GraphGame(
            model=gcn_model_classification,
            x_graph=simple_graph,
            task="classification",
            class_index=0,
            baseline_strategy="average",
        )

    def test_init_with_gcn_model(self, gcn_graph_game):
        """Test that GraphGame initializes correctly with a GCN model."""
        assert gcn_graph_game.model is not None
        assert gcn_graph_game.x_graph is not None
        assert gcn_graph_game.task == "regression"
        assert gcn_graph_game.y_index is None
        assert gcn_graph_game.n_players == gcn_graph_game.x_graph.num_nodes

    def test_init_with_gin_model(self, gin_graph_game):
        """Test that GraphGame initializes correctly with a GIN model."""
        assert gin_graph_game.model is not None
        assert gin_graph_game.x_graph is not None
        assert gin_graph_game.task == "regression"

    def test_init_with_gat_model(self, gat_graph_game):
        """Test that GraphGame initializes correctly with a GAT model."""
        assert gat_graph_game.model is not None
        assert gat_graph_game.x_graph is not None
        assert gat_graph_game.task == "regression"

