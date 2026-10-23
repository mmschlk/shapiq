"""The graph package without torch: graphs, the graph imputer, and explanations of functions."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import ExactComputer
from shapiq.explainer import GraphExplainer
from shapiq.graph import (
    GraphBaselineImputer,
    GraphModel,
    GraphSHAPIQ,
    convert_graph,
    get_graph_predict_function,
)


def path(n_nodes: int) -> np.ndarray:
    return np.array([np.arange(n_nodes - 1), np.arange(1, n_nodes)])


def path_graph(n_nodes: int, n_features: int = 2, seed: int = 0) -> GraphModel:
    x = np.random.default_rng(seed).normal(size=(n_nodes, n_features))
    return GraphModel(edge_index=path(n_nodes), n_hops=1, x=x)


class OneHopModel:
    """One message passing layer on a path, a sum over the nodes, and a linear readout."""

    def __init__(self, n_nodes: int, n_features: int = 2, n_outputs: int = 1) -> None:
        rng = np.random.default_rng(1)
        self.adjacency = np.eye(n_nodes, k=1) + np.eye(n_nodes, k=-1)
        self.weights = rng.normal(size=(2, n_features, 4))
        self.readout = rng.normal(size=(4, n_outputs))

    def __call__(self, node_features: np.ndarray) -> np.ndarray:
        neighbors = np.einsum("ij,bjf->bif", self.adjacency, node_features)
        hidden = np.tanh(node_features @ self.weights[0] + neighbors @ self.weights[1])
        return hidden.sum(axis=1) @ self.readout


def test_graph_model_infers_and_checks_its_nodes():
    assert GraphModel(edge_index=path(4), n_hops=1).n_nodes == 4
    assert GraphModel(edge_index=path(4), n_hops=1, n_nodes=6).n_nodes == 6
    assert GraphModel(edge_index=np.zeros((2, 0)), n_hops=1, x=np.ones((3, 2))).n_nodes == 3
    assert path_graph(5, n_features=3).n_features == 3
    with pytest.raises(ValueError, match="shape"):
        GraphModel(edge_index=np.zeros((3, 2)), n_hops=1)
    with pytest.raises(ValueError, match="outside"):
        GraphModel(edge_index=path(4), n_hops=1, n_nodes=3)
    with pytest.raises(ValueError, match="rows"):
        GraphModel(edge_index=path(4), n_hops=1, x=np.ones((5, 2)), n_nodes=4)
    with pytest.raises(ValueError, match="n_hops"):
        GraphModel(edge_index=path(4), n_hops=-1)
    with pytest.raises(ValueError, match="node features"):
        _ = GraphModel(edge_index=path(4), n_hops=1).n_features


def test_convert_graph_keeps_graph_models():
    graph = path_graph(4)
    assert convert_graph(graph) is graph
    assert convert_graph(graph, n_hops=2).n_hops == 2
    with pytest.raises(TypeError, match="not supported"):
        convert_graph([(0, 1)])


def test_predict_function_of_a_callable_resolves_the_explained_output():
    graph = path_graph(4)
    assert get_graph_predict_function(OneHopModel(4), graph).class_index is None
    assert get_graph_predict_function(OneHopModel(4, n_outputs=3), graph).class_index == 1
    function = get_graph_predict_function(OneHopModel(4, n_outputs=3), graph, class_index=2)
    np.testing.assert_allclose(
        function(graph.x[None]), OneHopModel(4, n_outputs=3)(graph.x[None])[:, 2]
    )
    with pytest.raises(ValueError, match="class_index"):
        get_graph_predict_function(OneHopModel(4, n_outputs=3), graph, class_index=3)
    with pytest.raises(ValueError, match="one output"):
        get_graph_predict_function(OneHopModel(4), graph, class_index=1)


def test_imputer_replaces_absent_nodes_by_the_baseline():
    graph = path_graph(3)
    calls = []

    def model(node_features: np.ndarray) -> np.ndarray:
        calls.append(node_features)
        return node_features.sum(axis=(1, 2))

    imputer = GraphBaselineImputer(model, data=np.array([10.0, 20.0]), x=graph, normalize=False)
    assert imputer.n_players == 3
    calls.clear()
    imputer(np.array([[True, False, True]]))
    expected = graph.x.copy()
    expected[1] = [10.0, 20.0]
    np.testing.assert_allclose(calls[0][0], expected)


def test_imputer_baseline_is_the_mean_of_the_background_nodes():
    graph = path_graph(4)
    background = np.random.default_rng(3).normal(size=(20, 2))
    model = get_graph_predict_function(OneHopModel(4), graph)
    imputer = GraphBaselineImputer(model, data=background, x=graph)
    np.testing.assert_allclose(imputer.baseline_values.astype(float), background.mean(0)[None])
    empty = np.broadcast_to(background.mean(0), (1, 4, 2))
    assert imputer.empty_prediction == pytest.approx(OneHopModel(4)(empty)[0, 0])
    assert imputer(np.zeros(4, dtype=bool))[0] == pytest.approx(0.0)  # centered


def test_imputer_needs_node_features_of_the_background():
    with pytest.raises(ValueError, match="node features"):
        GraphBaselineImputer(OneHopModel(4), data=np.zeros(3), x=path_graph(4))
    structure = GraphModel(edge_index=path(4), n_hops=1)
    with pytest.raises(ValueError, match="node features"):
        GraphBaselineImputer(OneHopModel(4), data=np.zeros(2), x=structure)


@pytest.mark.parametrize(("index", "order"), [("SV", 1), ("k-SII", 2), ("STII", 3), ("Moebius", 6)])
def test_explainer_of_a_function_matches_brute_force(index, order):
    graph, model = path_graph(6), OneHopModel(6)
    background = np.random.default_rng(4).normal(size=(10, 2))
    values = GraphExplainer(model, data=background, index=index, max_order=order).explain(graph)
    assert not values.estimated

    game = GraphBaselineImputer(
        get_graph_predict_function(model, graph), data=background, x=graph, normalize=False
    )
    expected = ExactComputer(game=game, n_players=6)(index, order)
    for interaction in set(values.interaction_lookup) | set(expected.interaction_lookup):
        if interaction:
            assert values[interaction] == pytest.approx(expected[interaction], abs=1e-10)
    assert values.baseline_value == pytest.approx(game.empty_prediction)


def test_explainer_explains_several_graphs():
    model = OneHopModel(5)
    graphs = [path_graph(5, seed=0), path_graph(5, seed=1)]
    explanations = GraphExplainer(model, index="k-SII", max_order=2).explain_X(graphs)
    assert [values.n_players for values in explanations] == [5, 5]
    game = GraphBaselineImputer(
        get_graph_predict_function(model, graphs[1]), data=np.zeros(2), x=graphs[1]
    )
    expected = GraphSHAPIQ(game, graphs[1])("k-SII", 2)
    np.testing.assert_allclose(
        [explanations[1][key] for key in expected.interaction_lookup if key],
        [expected[key] for key in expected.interaction_lookup if key],
    )


def test_graph_model_copies_and_compares_by_identity():
    edges, x = path(3), np.ones((3, 2))
    graph = GraphModel(edge_index=edges, n_hops=1, x=x)
    x[0, 0], edges[0, 0] = 5.0, 2
    assert graph.x[0, 0] == 1.0
    assert graph.edge_index[0, 0] == 0
    same = graph
    assert graph == same
    assert graph != GraphModel(edge_index=path(3), n_hops=1, x=np.ones((3, 2)))
    assert hash(graph) == hash(graph)
    with pytest.raises(ValueError, match="integer"):
        GraphModel(edge_index=np.array([[0.7, 1.2], [1.9, 2.0]]), n_hops=1)
    with pytest.raises(ValueError, match="n_nodes"):
        GraphModel(edge_index=np.zeros((2, 0), dtype=int), n_hops=1, n_nodes=-1)


def test_imputer_refits_only_to_graphs_of_the_same_structure():
    graph = path_graph(4)
    model = get_graph_predict_function(OneHopModel(4), graph)
    imputer = GraphBaselineImputer(model, data=np.zeros(2), x=graph)
    other_features = path_graph(4, seed=5)
    imputer.fit(other_features)
    np.testing.assert_allclose(imputer.x, other_features.x)
    rewired = GraphModel(edge_index=np.array([[0, 0, 0], [1, 2, 3]]), n_hops=1, x=graph.x)
    with pytest.raises(ValueError, match="other nodes or edges"):
        imputer.fit(rewired)
    with pytest.raises(ValueError, match="other nodes or edges"):
        GraphBaselineImputer(model, data=np.zeros(2), x=rewired)


def test_large_evaluations_are_chunked(monkeypatch):
    import shapiq.graph.graphshapiq as graphshapiq_module
    import shapiq.graph.imputer as imputer_module

    graph, model = path_graph(8), OneHopModel(8)
    expected = GraphExplainer(model, index="k-SII", max_order=2).explain(graph)
    monkeypatch.setattr(graphshapiq_module, "_CHUNK_SIZE", 7)
    monkeypatch.setattr(imputer_module, "_MAX_VALUES", 3 * graph.x.size)
    chunked = GraphExplainer(model, index="k-SII", max_order=2).explain(graph)
    for interaction in expected.interaction_lookup:
        assert chunked[interaction] == pytest.approx(expected[interaction], abs=1e-12)


def test_integer_node_features_take_the_mode_as_baseline():
    x = np.array([[0, 1], [2, 1], [2, 0]])
    graph = GraphModel(edge_index=path(3), n_hops=1, x=x)
    imputer = GraphBaselineImputer(lambda f: f.sum(axis=(1, 2)), data=x, x=graph)
    np.testing.assert_array_equal(imputer.baseline_values.astype(int), [[2, 1]])


def test_unsupported_graphs_are_reported_as_such():
    with pytest.raises(TypeError, match="not supported"):
        GraphExplainer(OneHopModel(3)).explain([(0, 1), (1, 2)])
