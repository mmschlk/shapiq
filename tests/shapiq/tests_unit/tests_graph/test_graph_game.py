"""The GraphGame: a graph neural network's output with absent nodes masked."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from shapiq.graph import GraphGame  # noqa: E402

from .models import GNN, path_graph  # noqa: E402


def forward(model, x, edge_index):
    with torch.no_grad():
        return model(x, edge_index, torch.zeros(len(x), dtype=torch.long))


def test_values_are_the_model_outputs_of_the_masked_graph():
    model, graph = GNN(), path_graph(5)
    game = GraphGame(model, graph, normalize=False)
    coalition = np.array([True, False, True, True, False])
    masked = graph.x.clone()
    masked[~torch.as_tensor(coalition)] = 0.0
    assert game(coalition)[0] == pytest.approx(forward(model, masked, graph.edge_index).item())
    assert game.original_model_output == pytest.approx(
        forward(model, graph.x, graph.edge_index).item()
    )
    assert game.empty_value == pytest.approx(
        forward(model, torch.zeros_like(graph.x), graph.edge_index).item()
    )


def test_value_does_not_depend_on_the_batch():
    game = GraphGame(GNN("gat"), path_graph(6), batch_size=4)
    coalitions = np.random.default_rng(0).random((11, 6)) < 0.5
    together = game.value_function(coalitions)
    alone = np.array([game.value_function(coalition[None])[0] for coalition in coalitions])
    np.testing.assert_array_equal(together, alone)


def test_normalized_game_is_centered():
    game = GraphGame(GNN(), path_graph(4))
    assert game(np.zeros(4, dtype=bool))[0] == 0.0
    assert game.normalization_value == game.empty_value


@pytest.mark.parametrize(
    ("baseline", "expected"),
    [
        ("zeros", lambda x: torch.zeros(3)),
        ("mean", lambda x: x.mean(0)),
        ("min", lambda x: x.amin(0)),
        ("max", lambda x: x.amax(0)),
        (0.5, lambda x: torch.full((3,), 0.5)),
        ([0.1, 0.2, 0.3], lambda x: torch.tensor([0.1, 0.2, 0.3])),
    ],
)
def test_baselines(baseline, expected):
    graph = path_graph(4)
    game = GraphGame(GNN(), graph, baseline=baseline)
    torch.testing.assert_close(game.baseline, expected(graph.x))


def test_rejects_a_baseline_of_the_wrong_size():
    with pytest.raises(ValueError, match="one value per node feature"):
        GraphGame(GNN(), path_graph(4), baseline=[0.0, 1.0])
    with pytest.raises(ValueError, match="baseline"):
        GraphGame(GNN(), path_graph(4), baseline="median")  # ty: ignore[invalid-argument-type]


def test_class_index():
    graph = path_graph(4)
    assert GraphGame(GNN(), graph).class_index is None
    assert GraphGame(GNN(n_outputs=3), graph).class_index == 1
    assert GraphGame(GNN(n_outputs=3), graph, class_index=2).class_index == 2
    with pytest.raises(ValueError, match="class_index"):
        GraphGame(GNN(n_outputs=3), graph, class_index=3)
    with pytest.raises(ValueError, match="one output"):
        GraphGame(GNN(), graph, class_index=1)


def test_n_hops():
    graph = path_graph(4)
    assert GraphGame(GNN(num_layers=3), graph).n_hops == 3
    assert GraphGame(GNN(num_layers=3), graph, n_hops=2).n_hops == 2
    model = GNN()
    del model.num_layers
    with pytest.raises(ValueError, match="n_hops"):
        GraphGame(model, graph)


def test_rejects_a_model_without_one_output_per_graph():
    class NodeModel(torch.nn.Module):
        num_layers = 0

        def forward(self, x, edge_index, batch):
            return x.sum(dim=1)

    with pytest.raises(ValueError, match="one output per graph"):
        GraphGame(NodeModel(), path_graph(4))


def test_rejects_a_graph_without_node_features():
    graph = path_graph(4)
    graph.x = None
    with pytest.raises(ValueError, match="node features"):
        GraphGame(GNN(), graph)
