"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

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
            baseline_strategy="zeros",
        )

    @pytest.fixture
    def gin_graph_game(self, gin_model, simple_graph):
        """Create a GraphGame instance with a GIN model and a simple graph."""
        return GraphGame(
            model=gin_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="zeros",
        )

    @pytest.fixture
    def gat_graph_game(self, gat_model, simple_graph):
        """Create a GraphGame instance with a GAT model and a simple graph."""
        return GraphGame(
            model=gat_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="zeros",
        )

    @pytest.fixture
    def gcn_graph_game_classification(self, gcn_model_classification, simple_graph):
        """Create a GraphGame instance for classification with a GCN model."""
        return GraphGame(
            model=gcn_model_classification,
            x_graph=simple_graph,
            task="classification",
            class_index=0,
            baseline_strategy="zeros",
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

    def test_init_with_classification_task(self, gcn_graph_game_classification):
        """Test that GraphGame initializes correctly for classification."""
        assert gcn_graph_game_classification.task == "classification"
        assert gcn_graph_game_classification.y_index == 0

    def test_init_with_invalid_task(self, gcn_model, simple_graph):
        """Test that GraphGame raises ValueError for invalid task."""
        with pytest.raises(ValueError, match="task must be 'classification' or 'regression'"):
            GraphGame(
                model=gcn_model,
                x_graph=simple_graph,
                task="invalid_task",
                baseline_strategy="zeros",
            )

    def test_init_with_class_index_for_regression(self, gcn_model, simple_graph):
        """Test that GraphGame raises ValueError if class_index is set for regression."""
        with pytest.raises(ValueError, match="class_index cannot be set for regression tasks"):
            GraphGame(
                model=gcn_model,
                x_graph=simple_graph,
                task="regression",
                class_index=0,
                baseline_strategy="zeros",
            )

    def test_init_with_invalid_model(self, simple_graph):
        """Test that GraphGame raises TypeError for invalid model type."""
        invalid_model = torch.nn.Linear(3, 1)  # Not a GNN model
        with pytest.raises(TypeError, match="Model must be GCN, GIN, or GAT"):
            GraphGame(
                model=invalid_model,
                x_graph=simple_graph,
                task="regression",
                baseline_strategy="zeros",
            )

    def test_init_with_missing_node_features(self, gcn_model):
        """Test that GraphGame raises ValueError if x_graph has no node features."""
        x_graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        with pytest.raises(ValueError, match="x_graph must have node features"):
            GraphGame(
                model=gcn_model,
                x_graph=x_graph,
                task="regression",
                baseline_strategy="zeros",
            )

    @pytest.mark.parametrize("strategy", ["zeros", "average", "min", "max"])
    def test_baseline_strategies(self, gcn_model, simple_graph, strategy):
        """Test that all baseline strategies work correctly."""
        game = GraphGame(
            model=gcn_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy=strategy,
        )
        assert game.baseline is not None
        assert game.baseline.shape == (simple_graph.num_node_features,)

    def test_baseline_as_tensor(self, gcn_model, simple_graph):
        """Test that a custom baseline tensor is used correctly."""
        custom_baseline = torch.randn(simple_graph.num_node_features)
        game = GraphGame(
            model=gcn_model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy=custom_baseline,
        )
        assert torch.allclose(game.baseline, custom_baseline)

    def test_baseline_warning_for_none(self, gcn_model, simple_graph):
        """Test that a warning is raised if baseline_strategy is None."""
        with pytest.warns(UserWarning, match="Baseline is not provided"):
            GraphGame(
                model=gcn_model,
                x_graph=simple_graph,
                task="regression",
                baseline_strategy=None,
            )

    def test_baseline_warning_for_unknown_strategy(self, gcn_model, simple_graph):
        """Test that a warning is raised for an unknown baseline strategy."""
        with pytest.warns(UserWarning, match="Unknown baseline strategy"):
            GraphGame(
                model=gcn_model,
                x_graph=simple_graph,
                task="regression",
                baseline_strategy="unknown_strategy",
            )

    def test_mask_input_all_nodes_active(self, gcn_graph_game):
        """Test that mask_input returns the original graph if all nodes are active."""
        coalition = np.ones(gcn_graph_game.n_players, dtype=np.int64)
        masked_graph = gcn_graph_game.mask_input(coalition)
        assert torch.allclose(masked_graph.x, gcn_graph_game.x_graph.x)

    def test_mask_input_all_nodes_inactive(self, gcn_graph_game):
        """Test that mask_input replaces all node features with the baseline if all nodes are inactive."""
        coalition = np.zeros(gcn_graph_game.n_players, dtype=np.int64)
        masked_graph = gcn_graph_game.mask_input(coalition)
        # All node features should be equal to the baseline
        for i in range(gcn_graph_game.n_players):
            assert torch.allclose(masked_graph.x[i], gcn_graph_game.baseline)

    def test_mask_input_partial_coalition(self, gcn_graph_game):
        """Test that mask_input correctly masks inactive nodes in a partial coalition."""
        coalition = np.array([1, 0, 1, 0])  # nodes 0 and 2 are active
        masked_graph = gcn_graph_game.mask_input(coalition)
        assert torch.allclose(masked_graph.x[0], gcn_graph_game.x_graph.x[0])
        assert torch.allclose(masked_graph.x[2], gcn_graph_game.x_graph.x[2])
        assert torch.allclose(masked_graph.x[1], gcn_graph_game.baseline)
        assert torch.allclose(masked_graph.x[3], gcn_graph_game.baseline)

    def test_value_function_single_coalition(self, gcn_graph_game):
        """Test that value_function works for a single coalition."""
        coalition = np.array([1, 0, 0, 0])  # Only node 0 is active
        result = gcn_graph_game.value_function(coalition)
        assert result.shape == (1,)
        assert isinstance(result[0], (float, np.floating))

    def test_value_function_batch_coalitions(self, gcn_graph_game):
        """Test that value_function works for a batch of coalitions."""
        coalitions = np.array([
            [1, 0, 0, 0],  # Only node 0
            [0, 1, 0, 0],  # only node 1
            [1, 1, 0, 0],  # Nodes 0 and 1
        ])
        result = gcn_graph_game.value_function(coalitions)
        assert result.shape == (3,)
        assert all(isinstance(val, (float, np.floating)) for val in result)

    def test_value_function_all_nodes_active(self, gcn_graph_game):
        """Test that value_function returns the model's prediction for the full graph."""
        coalition = np.ones(gcn_graph_game.n_players, dtype=np.int64)
        result = gcn_graph_game.value_function(coalition)
        with torch.no_grad():
            expected = gcn_graph_game.model(
                x=gcn_graph_game.x_graph.x,
                edge_index=gcn_graph_game.x_graph.edge_index,
                batch=getattr(gcn_graph_game.x_graph, "batch", None),
            ).squeeze().numpy()
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_value_function_all_nodes_inactive(self, gcn_graph_game):
        """Test that value_function returns the baseline prediction for an empty coalition."""
        coalition = np.zeros(gcn_graph_game.n_players, dtype=np.int64)
        result = gcn_graph_game.value_function(coalition)
        with torch.no_grad():
            baseline_graph = gcn_graph_game.mask_input(coalition)
            expected = gcn_graph_game.model(
                x=baseline_graph.x,
                edge_index=baseline_graph.edge_index,
                batch=getattr(baseline_graph, "batch", None),
            ).squeeze().numpy()
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_value_function_classification(self, gcn_graph_game_classification):
        """Test that value_function works for classification tasks."""
        coalition = np.array([1, 0, 0, 0])  # Only node 0 is active
        result = gcn_graph_game_classification.value_function(coalition)
        assert result.shape == (1,)
        assert isinstance(result[0], (float, np.floating))

    @pytest.mark.parametrize("model_fixture", ["gcn_model", "gin_model", "gat_model"])
    def test_value_function_all_gnn_types(self, model_fixture, simple_graph, request):
        """Test that value_function works for all supported GNN types (GCN, GIN, GAT)."""
        model = request.getfixturevalue(model_fixture)
        game = GraphGame(
            model=model,
            x_graph=simple_graph,
            task="regression",
            baseline_strategy="zeros",
        )
        coalition = np.array([1, 0, 0, 0])
        result = game.value_function(coalition)
        assert result.shape == (1,)

    def test_value_function_single_node_graph(self, gcn_model):
        """Test that value_function works for a graph with a single node."""
        x = torch.randn(1, 3)  # 1 node, 3 features
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        x_graph = Data(x=x, edge_index=edge_index)
        game = GraphGame(
            model=gcn_model,
            x_graph=x_graph,
            task="regression",
            baseline_strategy="zeros",
        )
        coalition = np.array([1])
        result = game.value_function(coalition)
        assert result.shape == (1,)

    def test_value_function_disconnected_graph(self, gcn_model, disconnected_graph):
        """Test that value_function works for a disconnected graph."""
        game = GraphGame(
            model=gcn_model,
            x_graph=disconnected_graph,
            task="regression",
            baseline_strategy="zeros",
        )
        coalition = np.array([1, 0, 0, 0, 0])  # Only node 0 is active
        result = game.value_function(coalition)
        assert result.shape == (1,)