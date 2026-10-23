"""The graph games in the benchmark: their computer, setup, and datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from shapiq_benchmark import setups  # noqa: E402
from shapiq_benchmark.computers import (  # noqa: E402
    BruteForceComputer,
    GraphComputer,
    default_computer,
)
from shapiq_benchmark.datasets import load_graph_dataset  # noqa: E402
from shapiq_benchmark.setups import GraphSetup, setup_from_dict  # noqa: E402
from shapiq_benchmark.setups._gnn import GraphNeuralNetwork  # noqa: E402
from shapiq_games import GraphGame  # noqa: E402
from tests.shapiq_benchmark.test_chain_of_trust import _assert_agrees_with_brute_force  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _graph(edges: list[tuple[int, int]], n_nodes: int, seed: int = 0, label: int = 0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    atoms = torch.randint(0, 3, (n_nodes,), generator=generator)
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).T
    return Data(
        x=torch.nn.functional.one_hot(atoms, 3).float(),
        edge_index=edge_index,
        y=torch.tensor([label]),
    )


def _network(layer: str, *, seed: int = 0) -> GraphNeuralNetwork:
    torch.manual_seed(seed)
    return GraphNeuralNetwork(layer, n_features=3, n_outputs=2, num_layers=2, hidden=8)  # ty: ignore[invalid-argument-type]


# receptive fields of two hops are smaller than these graphs
_GRAPHS = {
    "path": _graph([(i, i + 1) for i in range(7)], 8),
    "tree": _graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (6, 7), (7, 8)], 9),
    "disconnected": _graph([(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)], 8),
}


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("graph", sorted(_GRAPHS))
@pytest.mark.parametrize("layer", ["gcn", "gin", "gat"])
def test_graph_computer_agrees_with_brute_force(layer: str, graph: str, normalize: bool) -> None:  # noqa: FBT001
    game = GraphGame(_network(layer), _GRAPHS[graph], normalize=normalize)
    computer = default_computer(game)
    assert isinstance(computer, GraphComputer)
    # the network computes in float32: brute force sees Möbius coefficients of about 1e-8 outside
    # the receptive fields, which GraphSHAP-IQ does not evaluate
    _assert_agrees_with_brute_force(computer, atol=1e-6)


def test_graph_computer_rejects_games_that_are_not_local() -> None:
    """A nonlinear readout makes nodes interact that share no receptive field."""

    class SquaredReadout(GraphNeuralNetwork):
        def forward(self, x, edge_index, batch):
            return super().forward(x, edge_index, batch) ** 2

    torch.manual_seed(0)
    model = SquaredReadout("gcn", n_features=3, n_outputs=2, num_layers=1, hidden=8)
    game = GraphGame(model, _GRAPHS["path"])
    assert not GraphComputer.supports_game(game)
    assert type(default_computer(game)) is BruteForceComputer
    # in a graph whose nodes all share a receptive field, every game is local
    small = GraphGame(model, _graph([(0, 1), (1, 2)], 3))
    assert GraphComputer.supports_game(small)


def test_graph_computer_warns_for_large_receptive_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    computer = GraphComputer(GraphGame(_network("gcn"), _GRAPHS["path"]))
    monkeypatch.setattr(GraphComputer, "max_budget", 10)
    with pytest.warns(UserWarning, match="max_budget"):
        computer.exact_values("SV", 1)


def _fake_dataset() -> list[Data]:
    """Paths and stars of 5 to 8 nodes, the stars being class 1."""
    graphs = []
    for i in range(24):
        n_nodes, star = 5 + i % 4, i % 2 == 1
        edges = (
            [(0, j) for j in range(1, n_nodes)]
            if star
            else [(j, j + 1) for j in range(n_nodes - 1)]
        )
        graphs.append(_graph(edges, n_nodes, seed=i, label=int(star)))
    return graphs


def test_graph_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setups._graph, "load_graph_dataset", lambda name: _fake_dataset())
    setup = GraphSetup(dataset="MUTAG", model="gin", n_epochs=3, hidden_size=8, x=2)
    game, again = setup.build(), setup.build()
    assert isinstance(game, GraphGame)
    assert game.n_hops == setup.n_layers
    assert game.class_index == 1
    coalitions = np.random.default_rng(0).random((6, game.n_players)) < 0.5
    np.testing.assert_array_equal(game(coalitions), again(coalitions))  # trained reproducibly
    assert isinstance(default_computer(game), GraphComputer)

    assert setup_from_dict(setup.to_dict()) == setup
    assert GraphSetup(dataset="MUTAG", batch_size=16).key == GraphSetup(dataset="MUTAG").key
    assert GraphSetup(dataset="MUTAG", model="gat").key != GraphSetup(dataset="MUTAG").key
    with pytest.raises(ValueError, match="one of"):
        GraphSetup(dataset="CORA")  # ty: ignore[invalid-argument-type]


def _write_tu_dataset(raw: Path, name: str, graphs: list[Data]) -> None:
    """Write graphs in the file format of the TUDataset collection."""
    raw.mkdir(parents=True)
    edges, indicator, node_labels, offset = [], [], [], 0
    for number, graph in enumerate(graphs, start=1):
        edges += [f"{i + offset + 1}, {j + offset + 1}" for i, j in graph.edge_index.T.tolist()]
        indicator += [str(number)] * graph.num_nodes
        node_labels += [str(label) for label in graph.x.argmax(dim=1).tolist()]
        offset += graph.num_nodes
    labels = [str(int(graph.y.item())) for graph in graphs]
    for kind, lines in (
        ("A", edges),
        ("graph_indicator", indicator),
        ("node_labels", node_labels),
        ("graph_labels", labels),
    ):
        (raw / f"{name}_{kind}.txt").write_text("".join(f"{line}\n" for line in lines))


def test_load_graph_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    graphs = _fake_dataset()[:4]
    _write_tu_dataset(tmp_path / "datasets" / "tudataset" / "MUTAG" / "raw", "MUTAG", graphs)
    loaded = load_graph_dataset("MUTAG")  # read from the cache, not downloaded
    assert len(loaded) == 4
    for graph, original in zip(loaded, graphs, strict=True):
        assert graph.num_nodes == original.num_nodes
        assert int(graph.y.item()) == int(original.y.item())
        assert graph.edge_index.shape == original.edge_index.shape
    with pytest.raises(ValueError, match="Unknown graph dataset"):
        load_graph_dataset("CORA")  # ty: ignore[invalid-argument-type]
