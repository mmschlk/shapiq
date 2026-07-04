"""
Test module for the GraphGame class in shapiq.graph.

This module contains tests for the GraphGame class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shapiq.graph import GraphGame


class TestGraphGame:
    """Test class for GraphGame."""

    # Init Tests
    def test_init_with_gcn_model(self, gcn_graph_game):
        """Test that GraphGame initializes correctly with a GCN model."""
        assert gcn_graph_game.model is not None
        assert gcn_graph_game.x_graph is not None
        assert gcn_graph_game.class_index is None
        assert gcn_graph_game.n_players == gcn_graph_game.x_graph.num_nodes

    def test_init_with_gin_model(self, gin_graph_game):
        """Test that GraphGame initializes correctly with a GIN model."""
        assert gin_graph_game.model is not None
        assert gin_graph_game.x_graph is not None
        assert gin_graph_game.class_index is None

    def test_init_with_gat_model(self, gat_graph_game):
        """Test that GraphGame initializes correctly with a GAT model."""
        assert gat_graph_game.model is not None
        assert gat_graph_game.x_graph is not None
        assert gat_graph_game.class_index is None

    def test_init_strategies(self, gcn_model, simple_graph):
        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy="min")
        assert np.allclose(game.baseline, torch.amin(game.x_graph.x, dim=0))

        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy="max")
        assert np.allclose(game.baseline, torch.amax(game.x_graph.x, dim=0))

        with pytest.raises(NotImplementedError, match="is not supported."):
            GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy="invalid")

        with pytest.raises(ValueError, match="Baseline tensor must have shape"):
            GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy=torch.zeros(2, 3))

    def test_init_sanity_checks(self, gcn_model, simple_graph, empty_graph):
        with pytest.raises(ValueError, match="x_graph must have node features"):
            GraphGame(model=gcn_model, x_graph=empty_graph)

        with pytest.raises(AttributeError, match="The GNN needs a num_layers attribute"):
            delattr(gcn_model, "num_layers")
            GraphGame(model=gcn_model, x_graph=simple_graph)

    def test_init_normalize_true(self, gcn_model, simple_graph):
        """Test that normalization_value is set correctly when normalize=True."""
        game = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=True)
        assert game.normalize is True
        assert game.normalization_value is not None

    # Masking Tests
    def test_mask_input_partial(self, gcn_graph_game):
        """Test masking the input of a GraphGame instance with a partial coalition."""
        coalition = np.array([True, False, True, False] + [False] * (gcn_graph_game.n_players - 4))
        masked_graph = gcn_graph_game.mask_input(coalition)

        expected_x = gcn_graph_game.x_graph.x.clone()
        baseline_reshaped = gcn_graph_game.baseline.reshape(1, -1)
        expected_x[~torch.tensor(coalition, dtype=torch.bool)] = baseline_reshaped

        assert torch.allclose(masked_graph.x, expected_x)

    def test_mask_input_all_active(self, gcn_graph_game):
        """Full coalition should return identical node features."""
        coalition = np.ones(gcn_graph_game.n_players, dtype=bool)
        masked_graph = gcn_graph_game.mask_input(coalition)

        assert torch.allclose(masked_graph.x, gcn_graph_game.x_graph.x)

    def test_mask_input_all_inactive(self, gcn_graph_game):
        """Empty coalition should replace all features with baseline."""
        coalition = np.zeros(gcn_graph_game.n_players, dtype=bool)
        masked_graph = gcn_graph_game.mask_input(coalition)

        baseline = gcn_graph_game.baseline.reshape(1, -1)
        expected = baseline.repeat(gcn_graph_game.n_players, 1)

        assert torch.allclose(masked_graph.x, expected)

    def test_value_function_deterministic(self, gcn_graph_game):
        """Same input should always produce same output."""
        coalition = np.ones(gcn_graph_game.n_players, dtype=bool)

        v1 = gcn_graph_game.value_function(coalition)
        v2 = gcn_graph_game.value_function(coalition)

        assert np.allclose(v1, v2)

    def test_baseline_shape(self, gcn_graph_game):
        """Baseline must match feature dimension."""
        assert gcn_graph_game.baseline.shape[0] == gcn_graph_game.x_graph.x.shape[1]

    # Value function tests
    def test_value_function_scalar_output(self, gcn_graph_game):
        """Test value function for scalar model outputs."""
        coalition = np.array([True, False, True, False] + [False] * (gcn_graph_game.n_players - 4))
        v = gcn_graph_game.value_function(coalition)

        assert isinstance(v[0], float)

    def test_value_function_indexed_output(self, gcn_graph_game_classification):
        """Test value function for indexed model outputs."""
        coalition = np.array(
            [True, False, True, False] + [False] * (gcn_graph_game_classification.n_players - 4)
        )
        v = gcn_graph_game_classification.value_function(coalition)

        assert gcn_graph_game_classification.class_index == 0
        assert isinstance(v[0], float)

    # Normalization value tests
    def test_normalization_value(self, gcn_graph_game):
        """Test that normalization_value is the value of the empty coalition."""
        empty_coalition_value = gcn_graph_game.value_function(np.zeros(gcn_graph_game.n_players))

        assert np.isclose(gcn_graph_game.normalization_value, empty_coalition_value[0])