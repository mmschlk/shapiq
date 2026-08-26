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
            GraphGame(model=gcn_model, x_graph=simple_graph, baseline_value=torch.zeros(2, 3))

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


_pyg_data = pytest.importorskip("torch_geometric.data")
Data = _pyg_data.Data


class TestGameModelEquivalence:
    """The two correctness anchors tying the game to the model (Def. 3.2)."""

    def test_grand_coalition_equals_model_output(self, gcn_model, simple_graph):
        """nu(N) must equal f(X): the unmasked model prediction."""
        game = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=False)
        full = np.ones(game.n_players, dtype=bool)

        v_full = game.value_function(full)[0]
        with torch.no_grad():
            direct = (
                gcn_model(x=simple_graph.x, edge_index=simple_graph.edge_index, batch=None)
                .squeeze()
                .item()
            )

        assert np.isclose(v_full, direct, atol=1e-6)

    def test_empty_coalition_equals_all_baseline_prediction(self, gcn_model, simple_graph):
        """nu(empty) must equal the model evaluated on an all-baseline feature matrix."""
        game = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=False)
        empty = np.zeros(game.n_players, dtype=bool)

        v_empty = game.value_function(empty)[0]
        x_baseline = game.baseline.reshape(1, -1).repeat(game.n_players, 1)
        with torch.no_grad():
            direct = (
                gcn_model(x=x_baseline, edge_index=simple_graph.edge_index, batch=None)
                .squeeze()
                .item()
            )

        assert np.isclose(v_empty, direct, atol=1e-6)

    def test_normalized_game_empty_coalition_is_zero(self, gcn_graph_game):
        """The actual normalization contract: the *game* evaluates empty to ~0.

        Note: value_function is the raw (unnormalized) function; the Game base
        class applies normalization on __call__. If the shapiq Game API differs,
        replace the call with the appropriate normalized accessor.
        """
        empty = np.zeros((1, gcn_graph_game.n_players), dtype=bool)
        normalized = gcn_graph_game(empty)
        assert np.isclose(normalized[0], 0.0, atol=1e-6)


class TestValueFunctionBatching:
    def test_coalition_matrix_returns_one_value_per_row(self, gcn_graph_game):
        n = gcn_graph_game.n_players
        rng = np.random.default_rng(0)
        coalitions = rng.integers(0, 2, size=(5, n)).astype(bool)

        values = gcn_graph_game.value_function(coalitions)

        assert values.shape == (5,)

    def test_batched_rows_match_single_evaluations(self, gcn_graph_game):
        n = gcn_graph_game.n_players
        coalitions = np.stack(
            [np.zeros(n, dtype=bool), np.ones(n, dtype=bool), np.arange(n) % 2 == 0]
        )
        batched = gcn_graph_game.value_function(coalitions)
        singles = np.array([gcn_graph_game.value_function(c)[0] for c in coalitions])
        assert np.allclose(batched, singles)

    def test_non_scalar_output_without_class_index_raises(
        self, gcn_model_classification, simple_graph
    ):
        """Multi-output model + class_index=None must raise, not silently pick."""
        game = GraphGame(model=gcn_model_classification, x_graph=simple_graph, normalize=False)
        coalition = np.ones(game.n_players, dtype=bool)
        with pytest.raises(ValueError, match="not scalar"):
            game.value_function(coalition)


class TestNormalizeFalse:
    def test_normalize_false_property_and_raw_values(self, gcn_model, simple_graph):
        game = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=False)

        assert game.normalize is False

        # Raw empty-coalition value should generally be nonzero and, when the
        # game is called, remain unshifted.
        empty = np.zeros((1, game.n_players), dtype=bool)
        raw = game.value_function(np.zeros(game.n_players, dtype=bool))[0]
        called = game(empty)[0]
        assert np.isclose(called, raw, atol=1e-6)

    def test_normalized_and_unnormalized_raw_values_agree(self, gcn_model, simple_graph):
        """normalize only shifts outputs; the underlying value function is identical."""
        g_norm = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=True)
        g_raw = GraphGame(model=gcn_model, x_graph=simple_graph, normalize=False)
        coalition = np.arange(g_raw.n_players) % 2 == 0
        assert np.allclose(g_norm.value_function(coalition), g_raw.value_function(coalition))


class TestBaselineValue:
    def test_float_baseline_creates_constant_vector(self, gcn_model, simple_graph):
        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_value=0.5)
        expected = torch.full((simple_graph.x.shape[1],), 0.5)
        assert torch.allclose(game.baseline, expected)

    def test_tensor_baseline_with_correct_shape(self, gcn_model, simple_graph):
        n_features = simple_graph.x.shape[1]
        custom = torch.arange(n_features, dtype=torch.float32)
        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_value=custom)
        assert torch.allclose(game.baseline, custom)

    def test_baseline_value_overrides_strategy(self, gcn_model, simple_graph):
        game = GraphGame(
            model=gcn_model,
            x_graph=simple_graph,
            baseline_strategy="max",
            baseline_value=0.0,
        )
        assert torch.allclose(game.baseline, torch.zeros(simple_graph.x.shape[1]))

    def test_invalid_baseline_value_type_raises(self, gcn_model, simple_graph):
        with pytest.raises(TypeError, match="baseline_value must be"):
            GraphGame(model=gcn_model, x_graph=simple_graph, baseline_value="0.0")

    def test_int_baseline_value_currently_raises(self, gcn_model, simple_graph):
        """Pins a surprising quirk: isinstance(value, float) rejects Python ints,
        so baseline_value=0 raises while baseline_value=0.0 works. If the code
        is changed to accept ints (recommended), update this test accordingly.
        """
        with pytest.raises(TypeError):
            GraphGame(model=gcn_model, x_graph=simple_graph, baseline_value=0)

    def test_zeros_strategy_explicit(self, gcn_model, simple_graph):
        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy="zeros")
        assert torch.allclose(game.baseline, torch.zeros(simple_graph.x.shape[1]))

    def test_average_strategy_explicit(self, gcn_model, simple_graph):
        game = GraphGame(model=gcn_model, x_graph=simple_graph, baseline_strategy="average")
        assert torch.allclose(game.baseline, simple_graph.x.mean(dim=0))


class TestInitValidation:
    def test_non_int_num_layers_raises(self, gcn_model, simple_graph):
        gcn_model.num_layers = 2.5
        with pytest.raises(TypeError, match="must be an int"):
            GraphGame(model=gcn_model, x_graph=simple_graph)

    def test_model_set_to_eval_mode(self, gcn_model, simple_graph):
        gcn_model.train()
        game = GraphGame(model=gcn_model, x_graph=simple_graph)
        assert game.model.training is False

    def test_missing_edge_index_currently_raises_attribute_error(self, gcn_model):
        """Documents current behavior: x present but edge_index=None crashes
        with an unhelpful AttributeError on .detach(). Recommendation: add an
        explicit ValueError check in __init__ and update this test's match.
        """
        graph = Data(x=torch.randn(4, 3), edge_index=None)
        with pytest.raises(AttributeError):
            GraphGame(model=gcn_model, x_graph=graph)

    def test_class_index_out_of_range_currently_raises_index_error(
        self, gcn_model_classification, simple_graph
    ):
        """Pins current behavior (raw torch IndexError at evaluation time).
        A friendlier option would be validating class_index in __init__.
        """
        game = GraphGame(
            model=gcn_model_classification,
            x_graph=simple_graph,
            class_index=99,
            normalize=False,
        )
        with pytest.raises(IndexError):
            game.value_function(np.ones(game.n_players, dtype=bool))


class TestMaskingSemantics:
    def test_mask_input_rowwise_manual_check(self, gcn_graph_game):
        """Non-tautological masking test: verify each row against the source
        data directly instead of re-running the masking logic."""
        n = gcn_graph_game.n_players
        coalition = np.zeros(n, dtype=bool)
        coalition[0] = True
        coalition[n - 1] = True

        masked = gcn_graph_game.mask_input(coalition)

        for i in range(n):
            if coalition[i]:
                assert torch.allclose(masked.x[i], gcn_graph_game.x_graph.x[i])
            else:
                assert torch.allclose(masked.x[i], gcn_graph_game.baseline)

    def test_int_coalition_matches_bool_coalition(self, gcn_graph_game):
        n = gcn_graph_game.n_players
        as_bool = np.arange(n) % 2 == 0
        as_int = as_bool.astype(int)
        assert np.allclose(
            gcn_graph_game.value_function(as_bool),
            gcn_graph_game.value_function(as_int),
        )

    def test_original_graph_never_mutated(self, gcn_model):
        """Neither the user's input graph nor the game's stored copy may change
        across repeated masking/evaluation."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        original = Data(x=x, edge_index=edge_index)
        x_snapshot = original.x.clone()

        game = GraphGame(model=gcn_model, x_graph=original)
        stored_snapshot = game.x_graph.x.clone()

        rng = np.random.default_rng(1)
        for _ in range(10):
            coalition = rng.integers(0, 2, size=game.n_players).astype(bool)
            game.mask_input(coalition)
            game.value_function(coalition)

        assert torch.equal(original.x, x_snapshot)
        assert torch.equal(game.x_graph.x, stored_snapshot)


class TestEdgeCaseGraphs:
    def test_single_node_graph_full_and_empty(self, gcn_model, single_node_graph):
        game = GraphGame(model=gcn_model, x_graph=single_node_graph, normalize=False)
        assert game.n_players == 1

        v_full = game.value_function(np.array([True]))
        v_empty = game.value_function(np.array([False]))
        assert v_full.shape == (1,)
        assert np.isfinite(v_full[0])
        assert np.isfinite(v_empty[0])

    def test_single_node_masking(self, gcn_model, single_node_graph):
        game = GraphGame(model=gcn_model, x_graph=single_node_graph)
        masked = game.mask_input(np.array([False]))
        assert torch.allclose(masked.x[0], game.baseline)

    def test_disconnected_components_are_additive(self, gcn_model):
        """For a linear readout + mean pooling, MIs spanning disconnected
        components are zero (Prop. 3.6). Equivalent check on raw values:
        nu(A u B) + nu(empty) == nu(A) + nu(B) for components A, B.
        """
        torch.manual_seed(0)
        x = torch.randn(6, 3)
        # Two path components: {0,1,2} and {3,4,5}
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long
        )
        graph = Data(x=x, edge_index=edge_index)
        game = GraphGame(model=gcn_model, x_graph=graph, normalize=False)

        comp_a = np.array([True, True, True, False, False, False])
        comp_b = ~comp_a
        full = np.ones(6, dtype=bool)
        empty = np.zeros(6, dtype=bool)

        v = lambda c: game.value_function(c)[0]  # noqa: E731
        assert np.isclose(v(full) + v(empty), v(comp_a) + v(comp_b), atol=1e-5)

    def test_graph_with_batch_attribute(self, gcn_model):
        """Exercise the getattr(masked_graph, 'batch', None) code path."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(4, dtype=torch.long))
        game = GraphGame(model=gcn_model, x_graph=graph, normalize=False)

        v = game.value_function(np.ones(4, dtype=bool))
        assert v.shape == (1,)
        assert np.isfinite(v[0])


class TestCoalitionShapeHandling:
    def test_wrong_length_coalition_fails(self, gcn_graph_game):
        """Pins current behavior: a wrong-sized coalition raises somewhere in
        torch broadcasting rather than a clean ValueError. Recommendation:
        validate coalition.shape[-1] == n_players and raise ValueError; then
        tighten this test to pytest.raises(ValueError, match=...).
        """
        bad = np.ones(gcn_graph_game.n_players + 2, dtype=bool)
        with pytest.raises(Exception):  # noqa: B017
            gcn_graph_game.value_function(bad)
