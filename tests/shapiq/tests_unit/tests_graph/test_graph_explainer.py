"""The GraphExplainer, checked against brute force on graph neural networks."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import shapiq
from shapiq import ExactComputer
from shapiq.graph import GraphExplainer, GraphGame

from .models import GNN, path_graph, undirected_graph

# receptive fields of 2 hops are smaller than these graphs, so GraphSHAP-IQ truncates
GRAPHS = {
    "path": path_graph(7),
    "tree": undirected_graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (6, 7)], 8),
    "disconnected": undirected_graph([(0, 1), (1, 2), (3, 4), (4, 5)], 7),
}


@pytest.mark.parametrize("layer", ["gcn", "gin", "gat"])
@pytest.mark.parametrize("graph_name", sorted(GRAPHS))
@pytest.mark.parametrize(("index", "order"), [("SV", 1), ("k-SII", 2), ("STII", 3)])
def test_exact_values_match_brute_force(layer, graph_name, index, order):
    model, graph = GNN(layer, pooling="add"), GRAPHS[graph_name]
    explainer = GraphExplainer(model, index=index, max_order=order)
    values = explainer.explain(graph)
    assert not values.estimated
    assert values.estimation_budget < 2**graph.num_nodes

    game = GraphGame(model, graph, normalize=False)
    expected = ExactComputer(game=game, n_players=graph.num_nodes)(index, order)
    for interaction in set(values.interaction_lookup) | set(expected.interaction_lookup):
        if interaction:
            assert values[interaction] == pytest.approx(expected[interaction], abs=1e-5)


def test_shapley_values_add_up_to_the_prediction():
    model, graph = GNN(n_outputs=2), path_graph(6)
    values = GraphExplainer(model).explain(graph)
    game = GraphGame(model, graph)
    assert values.baseline_value == pytest.approx(game.empty_value)
    total = sum(values[(node,)] for node in range(6)) + values.baseline_value
    assert total == pytest.approx(game.original_model_output, abs=1e-5)


def test_approximation_is_flagged_and_efficient():
    model, graph = GNN(num_layers=3), path_graph(9)
    exact = GraphExplainer(model).explain(graph)
    approximated = GraphExplainer(model, max_interaction_size=2).explain(graph)
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
