"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

import torch
import numpy as np

class TestGraphGame:
    """Test class for GraphGame."""

    # Init Tests
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

    # Masking Tests
    def test_mask_input_partial(self, gcn_graph_game):
        """Test, masking the input of a GraphGame instance with a partial coalition."""
        coalition = np.array([True, False, True, False] + [False] * (gcn_graph_game.n_players - 4))
        masked_graph = gcn_graph_game.mask_input(coalition)
        expected_x = gcn_graph_game.x_graph.x.clone()
        baseline_reshaped = gcn_graph_game.baseline.reshape(1, -1)
        expected_x[~torch.tensor(coalition, dtype=torch.bool)] = baseline_reshaped
        assert torch.allclose(masked_graph.x, expected_x)

