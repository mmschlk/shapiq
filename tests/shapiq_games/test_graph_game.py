"""The GraphGame: a model's output for one graph with absent nodes imputed."""

from __future__ import annotations

import numpy as np
import pytest

import shapiq_games as sg
from shapiq.graph import GraphModel
from tests.shapiq_games.helpers import is_installed


def _graph(n_nodes: int = 5) -> GraphModel:
    edge_index = np.array([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
    x = np.random.default_rng(0).normal(size=(n_nodes, 2))
    return GraphModel(edge_index=edge_index, n_hops=1, x=x)


def _model(node_features: np.ndarray) -> np.ndarray:
    """Two outputs per graph: the sums of the two node features."""
    return node_features.sum(axis=1)


def test_values_are_the_outputs_of_the_imputed_graph():
    graph = _graph()
    background = np.random.default_rng(1).normal(size=(8, 2))
    game = sg.GraphGame(_model, graph, data=background, normalize=False)
    assert game.class_index == 1  # the convention for models with several outputs
    coalition = np.array([True, False, True, True, False])
    imputed = np.where(coalition[:, None], graph.x, background.mean(axis=0))
    assert game(coalition)[0] == pytest.approx(imputed[:, 1].sum())
    assert game.empty_value == pytest.approx(5 * background.mean(axis=0)[1])
    assert game.original_model_output == pytest.approx(graph.x[:, 1].sum())


def test_centered_game_with_a_zero_baseline():
    game = sg.GraphGame(_model, _graph(), class_index=0)
    assert game.empty_value == 0.0
    assert game(np.zeros(5, dtype=bool))[0] == 0.0
    assert game.graph.n_hops == 1


@pytest.mark.skipif(not is_installed("torch_geometric"), reason="torch_geometric is not installed")
def test_torch_geometric_graphs_take_the_layers_of_the_model():
    from tests.shapiq_games.helpers import graph_game

    game = graph_game()
    assert game.graph.n_hops == 2
    assert game.n_players == game.graph.n_nodes == 7
