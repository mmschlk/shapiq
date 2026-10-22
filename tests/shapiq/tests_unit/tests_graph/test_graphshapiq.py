"""GraphSHAP-IQ on games that are local on a graph, checked against brute force."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import ExactComputer, Game
from shapiq.graph import GraphSHAPIQ, receptive_fields
from shapiq.graph.graphshapiq import MAX_RECEPTIVE_FIELD_SIZE


class LocalGame(Game):
    """A sum over the nodes of random tables on their receptive fields: local by construction."""

    def __init__(self, edge_index: np.ndarray, n_nodes: int, n_hops: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.fields = receptive_fields(edge_index, n_nodes, n_hops)
        self.tables = [rng.normal(size=2 ** len(field)) for field in self.fields]
        self.n_evaluations = 0
        super().__init__(n_nodes, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        self.n_evaluations += len(coalitions)
        values = np.zeros(len(coalitions))
        for field, table in zip(self.fields, self.tables, strict=True):
            values += table[(coalitions[:, field].astype(np.int64) << np.arange(len(field))).sum(1)]
        return values


def random_graph(seed: int) -> tuple[np.ndarray, int, int]:
    """Return the edges, number of nodes, and number of hops of a small random graph."""
    rng = np.random.default_rng(seed)
    n_nodes = int(rng.integers(3, 10))
    edge_index = np.array(np.nonzero(np.triu(rng.random((n_nodes, n_nodes)) < 0.3, 1)))
    return edge_index, n_nodes, int(rng.integers(0, 3))


def path(n_nodes: int) -> np.ndarray:
    return np.array([np.arange(n_nodes - 1), np.arange(1, n_nodes)])


def assert_values_equal(computed, expected, atol=1e-10) -> None:
    for interaction in set(computed.interaction_lookup) | set(expected.interaction_lookup):
        if interaction:
            assert computed[interaction] == pytest.approx(expected[interaction], abs=atol)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    ("index", "order"),
    [("Moebius", 9), ("SV", 1), ("BV", 1), ("SII", 2), ("k-SII", 3), ("STII", 2), ("BII", 2)],
)
def test_exact_values_match_brute_force(seed, index, order):
    edge_index, n_nodes, n_hops = random_graph(seed)
    game = LocalGame(edge_index, n_nodes, n_hops, seed)
    order = min(order, n_nodes)
    computed = GraphSHAPIQ(game, edge_index, n_hops)(index, order)
    assert not computed.estimated
    assert_values_equal(computed, ExactComputer(game=game, n_players=n_nodes)(index, order))


def test_fsii_matches_brute_force():
    game = LocalGame(path(7), 7, 1, seed=0)
    computed = GraphSHAPIQ(game, path(7), 1)("FSII", 2)
    assert_values_equal(computed, ExactComputer(game=game, n_players=7)("FSII", 2), atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_one_below_the_largest_receptive_field_is_exact_with_the_efficiency_routine(seed):
    edge_index, n_nodes, n_hops = random_graph(seed)
    game = LocalGame(edge_index, n_nodes, n_hops, seed)
    largest = max(len(field) for field in game.fields)
    if largest < 2:
        pytest.skip("only singleton receptive fields")
    graphshapiq = GraphSHAPIQ(game, edge_index, n_hops, max_interaction_size=largest - 1)
    assert graphshapiq.exact
    exact = ExactComputer(game=game, n_players=n_nodes)
    assert_values_equal(graphshapiq("k-SII", 2), exact("k-SII", 2))

    without = GraphSHAPIQ(
        game, edge_index, n_hops, max_interaction_size=largest - 1, efficiency_routine=False
    )
    assert not without.exact
    assert without.budget < graphshapiq.budget


@pytest.mark.parametrize("max_interaction_size", [1, 2, 3, 5])
def test_approximated_values_are_efficient(max_interaction_size):
    """Overlapping receptive fields: the efficiency routine counts every coalition once."""
    side = 4
    edges = [(r * side + c, r * side + c + 1) for r in range(side) for c in range(side - 1)]
    edges += [(r * side + c, (r + 1) * side + c) for r in range(side - 1) for c in range(side)]
    edge_index = np.array(edges).T
    game = LocalGame(edge_index, side * side, 2, seed=0)
    graphshapiq = GraphSHAPIQ(game, edge_index, 2, max_interaction_size=max_interaction_size)
    shapley_values = graphshapiq("SV", 1)
    assert shapley_values.estimated
    assert shapley_values.estimation_budget == graphshapiq.budget
    total = sum(shapley_values[(node,)] for node in range(side * side))
    assert total + shapley_values.baseline_value == pytest.approx(game.grand_coalition_value)


def test_evaluates_exactly_the_budget():
    game = LocalGame(path(6), 6, 1, seed=0)
    graphshapiq = GraphSHAPIQ(game, path(6), 1)
    # the maximal receptive fields {i - 1, i, i + 1} share the subsets of {i, i + 1}
    assert graphshapiq.budget == 4 * 8 - 3 * 4
    graphshapiq("k-SII", 2)
    graphshapiq("SV", 1)
    assert game.n_evaluations == graphshapiq.budget


def test_disconnected_graph_and_isolated_nodes_are_additive():
    edge_index = np.array([[0, 2], [1, 3]])  # 0 - 1, 2 - 3, and the isolated node 4
    game = LocalGame(edge_index, 5, 2, seed=1)
    computed = GraphSHAPIQ(game, edge_index, 2)("Moebius", 5)
    assert set(computed.interaction_lookup) == {
        (),
        *((node,) for node in range(5)),
        (0, 1),
        (2, 3),
    }
    assert_values_equal(computed, ExactComputer(game=game, n_players=5)("Moebius", 5))


def test_no_hops_is_additive():
    game = LocalGame(path(4), 4, 0, seed=2)
    shapley_values = GraphSHAPIQ(game, path(4), 0)("SV", 1)
    empty = game(np.zeros(4, dtype=bool))[0]
    for node in range(4):
        alone = game(np.eye(4, dtype=bool)[node])[0]
        assert shapley_values[(node,)] == pytest.approx(alone - empty)


def test_moebius_up_to_order():
    game = LocalGame(path(5), 5, 1, seed=3)
    moebius = GraphSHAPIQ(game, path(5), 1)("Moebius", 2)
    assert max(len(interaction) for interaction in moebius.interaction_lookup) == 2
    assert moebius.baseline_value == pytest.approx(game(np.zeros(5, dtype=bool))[0])


def test_receptive_fields_ignore_edge_direction():
    fields = receptive_fields(np.array([[1, 2], [0, 1]]), 4, 1)  # 1 -> 0, 2 -> 1, node 3 alone
    assert [field.tolist() for field in fields] == [[0, 1], [0, 1, 2], [1, 2], [3]]
    assert receptive_fields(np.array([[1, 2], [0, 1]]), 4, 2)[0].tolist() == [0, 1, 2]


def test_rejects_invalid_arguments():
    game = LocalGame(path(3), 3, 1, seed=0)
    with pytest.raises(ValueError, match="n_hops"):
        GraphSHAPIQ(game, path(3), -1)
    with pytest.raises(ValueError, match="max_interaction_size"):
        GraphSHAPIQ(game, path(3), 1, max_interaction_size=0)
    with pytest.raises(ValueError, match="shape"):
        GraphSHAPIQ(game, np.zeros((3, 2), dtype=int), 1)
    with pytest.raises(ValueError, match="outside"):
        GraphSHAPIQ(game, np.array([[0], [3]]), 1)
    with pytest.raises(ValueError, match="indices"):
        GraphSHAPIQ(game, path(3), 1)("CHII", 2)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="order"):
        GraphSHAPIQ(game, path(3), 1)("SII", 0)


def test_rejects_too_large_receptive_fields():
    n_nodes = MAX_RECEPTIVE_FIELD_SIZE + 1  # a star: the center sees every node
    edge_index = np.array([np.zeros(n_nodes - 1, dtype=int), np.arange(1, n_nodes)])
    graphshapiq = GraphSHAPIQ(Game(n_nodes, normalize=False), edge_index, 1)
    with pytest.raises(ValueError, match="receptive fields"):
        _ = graphshapiq.budget
