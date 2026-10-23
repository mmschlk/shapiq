"""Graphs and graph neural networks of torch_geometric, explained and checked against brute force."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import shapiq  # noqa: E402
from shapiq import ExactComputer  # noqa: E402
from shapiq.explainer import GraphExplainer  # noqa: E402
from shapiq.graph import (  # noqa: E402
    GraphBaselineImputer,
    convert_graph,
    get_graph_predict_function,
)
from shapiq.graph.conversion import resolve_n_hops  # noqa: E402

from .models import GNN, path_graph, undirected_graph  # noqa: E402

# receptive fields of 2 hops are smaller than these graphs, so GraphSHAP-IQ truncates
GRAPHS = {
    "path": path_graph(7),
    "tree": undirected_graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (6, 7)], 8),
    "disconnected": undirected_graph([(0, 1), (1, 2), (3, 4), (4, 5)], 7),
}


def forward(model, x, edge_index):
    with torch.no_grad():
        return model(x, edge_index, torch.zeros(len(x), dtype=torch.long))


def test_convert_graph():
    data = path_graph(5)
    graph = convert_graph(data, n_hops=2)
    np.testing.assert_array_equal(graph.edge_index, data.edge_index.numpy())
    np.testing.assert_allclose(graph.x, data.x.numpy())
    assert (graph.n_nodes, graph.n_hops) == (5, 2)
    with pytest.raises(ValueError, match="n_hops"):
        convert_graph(data)
    data.x = None
    with pytest.raises(ValueError, match="node features"):
        convert_graph(data, n_hops=2)


def test_n_hops_are_the_layers_of_the_model():
    assert resolve_n_hops(GNN(num_layers=3), None) == 3
    assert resolve_n_hops(GNN(num_layers=3), 2) == 2
    model = GNN()
    del model.num_layers
    with pytest.raises(ValueError, match="n_hops"):
        resolve_n_hops(model, None)


def test_predict_function_calls_the_model_on_the_masked_graphs():
    model, data = GNN(n_outputs=2), path_graph(5)
    graph = convert_graph(data, n_hops=2)
    predict = get_graph_predict_function(model, graph, class_index=0, batch_size=4)
    masked = data.x.clone()
    masked[[1, 3]] = 0.0
    node_features = np.stack([graph.x, masked.numpy()])
    expected = [
        forward(model, data.x, data.edge_index)[0, 0],
        forward(model, masked, data.edge_index)[0, 0],
    ]
    np.testing.assert_allclose(predict(node_features), expected, rtol=1e-6)


def test_predict_function_does_not_depend_on_the_batch():
    graph = convert_graph(path_graph(6), n_hops=2)
    predict = get_graph_predict_function(GNN("gat"), graph, batch_size=4)
    node_features = np.random.default_rng(0).normal(size=(11, 6, 3))
    together = predict(node_features)
    alone = np.array([predict(features[None])[0] for features in node_features])
    np.testing.assert_array_equal(together, alone)


def test_predict_function_rejects_a_model_without_one_output_per_graph():
    class NodeModel(torch.nn.Module):
        num_layers = 0

        def forward(self, x, edge_index, batch):
            return x.sum(dim=1)

    graph = convert_graph(path_graph(4), n_hops=0)
    with pytest.raises(ValueError, match="one output per graph"):
        get_graph_predict_function(NodeModel(), graph)
    with pytest.raises(ValueError, match="batch_size"):
        get_graph_predict_function(GNN(), graph, batch_size=0)


@pytest.mark.parametrize("layer", ["gcn", "gin", "gat"])
@pytest.mark.parametrize("graph_name", sorted(GRAPHS))
@pytest.mark.parametrize(("index", "order"), [("SV", 1), ("k-SII", 2), ("STII", 3)])
def test_exact_values_match_brute_force(layer, graph_name, index, order):
    model, data = GNN(layer, pooling="add"), GRAPHS[graph_name]
    values = GraphExplainer(model, index=index, max_order=order).explain(data)
    assert not values.estimated
    assert values.estimation_budget < 2**data.num_nodes

    graph = convert_graph(data, n_hops=model.num_layers)
    game = GraphBaselineImputer(
        get_graph_predict_function(model, graph), data=np.zeros(3), x=graph, normalize=False
    )
    expected = ExactComputer(game=game, n_players=data.num_nodes)(index, order)
    for interaction in set(values.interaction_lookup) | set(expected.interaction_lookup):
        if interaction:
            assert values[interaction] == pytest.approx(expected[interaction], abs=1e-5)


def test_shapley_values_add_up_to_the_prediction():
    model, data = GNN(n_outputs=2), path_graph(6)
    background = np.random.default_rng(0).normal(size=(10, 3))
    values = GraphExplainer(model, data=background).explain(data)
    baseline = np.broadcast_to(background.mean(axis=0), (6, 3))
    empty = forward(model, torch.as_tensor(baseline, dtype=torch.float32), data.edge_index)
    assert values.baseline_value == pytest.approx(empty[0, 1].item(), rel=1e-5)
    total = sum(values[(node,)] for node in range(6)) + values.baseline_value
    assert total == pytest.approx(forward(model, data.x, data.edge_index)[0, 1].item(), abs=1e-5)


def test_approximation_is_flagged_and_efficient():
    model, data = GNN(num_layers=3), path_graph(9)
    exact = GraphExplainer(model).explain(data)
    approximated = GraphExplainer(model, max_interaction_size=2).explain(data)
    assert approximated.estimated
    assert approximated.estimation_budget < exact.estimation_budget
    assert sum(approximated[(node,)] for node in range(9)) == pytest.approx(
        sum(exact[(node,)] for node in range(9)), abs=1e-5
    )


def test_explainer_dispatches_graph_neural_networks():
    explainer = shapiq.Explainer(GNN(), index="k-SII", max_order=2)
    assert isinstance(explainer, GraphExplainer)
    assert explainer.explain(path_graph(4)).index == "k-SII"


def test_explain_x():
    explainer = GraphExplainer(GNN(), index="k-SII", max_order=2)
    graphs = [path_graph(4, seed=0), path_graph(5, seed=1)]
    explanations = explainer.explain_X(graphs, verbose=True)
    assert [values.n_players for values in explanations] == [4, 5]
    np.testing.assert_allclose(explanations[1].values, explainer.explain(graphs[1]).values)


def test_rejects_unsupported_indices():
    with pytest.raises(ValueError, match="supports the indices"):
        GraphExplainer(GNN(), index="Fourier", max_order=2)  # ty: ignore[invalid-argument-type]
