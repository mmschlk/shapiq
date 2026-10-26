"""The GraphGame: a graph neural network's output for one graph with absent nodes imputed."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402
from torch_geometric.nn import global_add_pool  # noqa: E402

import shapiq_games as sg  # noqa: E402
from shapiq.graph import convert_graph  # noqa: E402
from tests.shapiq_games.helpers import graph_game  # noqa: E402


class SumOverNodes(torch.nn.Module):
    """No message passing: one output per node feature, its sum over the nodes."""

    num_layers = 0

    def forward(self, x, edge_index, batch):
        return global_add_pool(x, batch)


def _graph(n_nodes: int = 5) -> Data:
    edges = [(i, i + 1) for i in range(n_nodes - 1)]
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges]).T
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(n_nodes, 2)), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)


def test_values_are_the_outputs_of_the_imputed_graph():
    graph = _graph()
    background = np.random.default_rng(1).normal(size=(8, 2))
    game = sg.GraphGame(SumOverNodes(), graph, data=background, normalize=False)
    assert game.class_index == 1  # the convention for models with several outputs
    coalition = np.array([True, False, True, True, False])
    x = graph.x.numpy()
    imputed = np.where(coalition[:, None], x, background.mean(axis=0))
    assert game(coalition)[0] == pytest.approx(imputed[:, 1].sum(), rel=1e-5)
    assert game.empty_value == pytest.approx(5 * background.mean(axis=0)[1], rel=1e-5)
    assert game.original_model_output == pytest.approx(x[:, 1].sum(), rel=1e-5)


def test_centered_game_with_a_zero_baseline():
    game = sg.GraphGame(SumOverNodes(), _graph(), class_index=0)
    assert game.empty_value == 0.0
    assert game(np.zeros(5, dtype=bool))[0] == 0.0


def test_n_hops_are_the_layers_of_the_model():
    game = graph_game()
    assert game.graph.n_hops == 2
    assert game.n_players == game.graph.n_nodes == 7


def test_takes_only_graph_neural_networks_on_torch_geometric_graphs():
    with pytest.raises(TypeError, match="torch model"):
        sg.GraphGame(lambda node_features: node_features.sum(axis=(1, 2)), _graph())
    with pytest.raises(TypeError, match="torch_geometric"):
        sg.GraphGame(SumOverNodes(), convert_graph(_graph(), n_hops=0))
    model = SumOverNodes()
    model.num_layers = None
    with pytest.raises(ValueError, match="num_layers"):
        sg.GraphGame(model, _graph())
