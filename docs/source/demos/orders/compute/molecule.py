"""Explain the molecule of GraphSHAP-IQ's Figure 1 at every order: ``results/molecule.json``.

The molecule is graph 71 of Mutagenicity (Kazius et al., 2005, of the TUDataset collection), a
trinitrophenanthrene of 30 atoms that GraphSHAP-IQ's Figure 1 explains (Muschalik et al., ICLR
2025). The model is a graph convolutional network (GCN) as in the paper: two GCN layers, each
followed by batch normalization, LeakyReLU and dropout, the embeddings of both layers concatenated
and summed over the atoms, and a linear readout of the two classes, trained on a stratified 80 %
of the dataset with Adam. The pooling and the readout are linear, so the game is local on the
molecule, and GraphSHAP-IQ computes its Möbius transform exactly from the subsets of the receptive
fields: 7,693 evaluations instead of 2^30.

The players are the atoms; an absent atom keeps its bonds, and its features are zeros. The value of
a coalition is the logit of the class "mutagen", centered: the logit of the empty coalition is
subtracted. From the Möbius transform, ``shapiq`` converts the explanation into FSII of every
order: the Shapley values (order 1), the Faithful Shapley Interactions of order 2, 3, ..., and the
full decomposition, the Möbius transform itself (FSII of the highest order). Every order is scored
by its R^2 beyond size under the Shapley distribution (``faithfulness.py``), the distribution FSII
is faithful to, on 10,000 coalitions drawn from it: how much of the model's output on those
coalitions the explanation reproduces beyond what their sizes alone predict. The Möbius transform is exact, so the model's output on a coalition is the
sum of the Möbius values of its subsets: the coalitions need no evaluation of the model, but for
512 of them, on which the script checks the sums against the model.
The explanation predicts a coalition's value as the sum of the values of its subsets, the empty
one's included: FSII's value of the empty group is the game's value of the empty coalition, 0, so
this prediction needs no constant of its own (for the coalition of all atoms, it is the model's
output).

Usage::

    python molecule.py

The dataset is downloaded to ``results/tudataset/``, the trained model saved to
``results/gcn.pt`` (and not trained again), and the explanation written to
``results/molecule.json``. Training takes about four minutes on four CPU cores, the explanation and
the R^2 of every order about 20 seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from faithfulness import fsii, predictions, r2, sample  # in this folder
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.graph import GraphSHAPIQ
from shapiq_games import GraphGame

HERE = Path(__file__).resolve().parent
GRAPH = 71  # the molecule of GraphSHAP-IQ's Figure 1
MUTAGEN = 0  # the class of mutagenic molecules in Mutagenicity
ATOMS = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]  # node labels
N_TEST = 10_000  # the coalitions every order is scored on, from the Shapley distribution


class GCN(torch.nn.Module):
    """GCN layers with batch normalization, a sum over the atoms, and a linear readout."""

    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2) -> None:
        """Initialize the network with the current seed of torch."""
        super().__init__()
        self.num_layers = num_layers  # GraphSHAP-IQ reads the number of hops from it
        sizes = [n_features] + [hidden] * num_layers
        self.convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in pairwise(sizes))
        self.norms = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden) for _ in sizes[1:])
        self.readout = torch.nn.Linear(hidden * num_layers, 2)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return the two class logits of every graph of the batch."""
        layers = []
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = F.leaky_relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=0.2, training=self.training)
            layers.append(x)
        # the embeddings of all layers, summed over the atoms: linear in every atom's embedding
        return self.readout(global_add_pool(torch.cat(layers, dim=1), batch))


def split(labels: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Return a stratified 80/10/10 split of the graphs into training, validation and test."""
    rng = np.random.default_rng(0)
    train, validation, test = [], [], []
    for label in np.unique(labels):
        graphs = rng.permutation(np.flatnonzero(labels == label)).tolist()
        a, b = int(0.8 * len(graphs)), int(0.9 * len(graphs))
        train += graphs[:a]
        validation += graphs[a:b]
        test += graphs[b:]
    return train, validation, test


def accuracy(model: GCN, graphs: list) -> float:
    """Return the accuracy of the model on the graphs."""
    model.eval()
    with torch.no_grad():
        correct = [
            model(batch.x, batch.edge_index, batch.batch).argmax(1) == batch.y
            for batch in DataLoader(graphs, batch_size=512)
        ]
    return float(torch.cat(correct).float().mean())


def train(dataset: TUDataset, path: Path) -> tuple[GCN, float]:
    """Train the GCN as in the paper (or load it from ``path``), and return its test accuracy."""
    graphs = [dataset[i] for i in range(len(dataset))]
    train_set, validation, test = split(np.array([int(graph.y) for graph in graphs]))
    torch.manual_seed(0)
    model = GCN(dataset.num_node_features)
    if path.exists():
        model.load_state_dict(torch.load(path))
    else:
        loader = DataLoader(
            [graphs[i] for i in train_set],
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=50, min_lr=1e-5
        )
        best, best_state, waiting = 0.0, None, 0
        for _ in range(500):  # with early stopping after 100 epochs without improvement
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                F.cross_entropy(logits, batch.y).backward()
                optimizer.step()
            score = accuracy(model, [graphs[i] for i in validation])
            scheduler.step(score)
            if score > best:
                best, waiting = score, 0
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                waiting += 1
            if waiting >= 100:
                break
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), path)
    return model.eval(), accuracy(model, [graphs[i] for i in test])


def layout(atoms: list[str], bonds: list[tuple[int, int]]) -> np.ndarray:
    """Return 2D coordinates of the molecule, as RDKit draws it, scaled into [0, 1]."""
    molecule = Chem.RWMol()
    for atom in atoms:
        molecule.AddAtom(Chem.Atom(atom))
    for i, j in bonds:
        molecule.AddBond(i, j, Chem.BondType.SINGLE)
    Chem.FastFindRings(molecule)  # the rings, without sanitizing the molecule (no bond orders)
    AllChem.Compute2DCoords(molecule)
    coords = molecule.GetConformer().GetPositions()[:, :2]
    coords[:, 1] *= -1  # y down, as on a page
    coords -= coords.min(axis=0)
    return coords / coords.max()


def main() -> None:
    """Train the GCN, explain the molecule at every order, and write ``results/molecule.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.results.mkdir(parents=True, exist_ok=True)
    dataset = TUDataset(root=str(args.results / "tudataset"), name="Mutagenicity")

    start = time.time()
    model, test_accuracy = train(dataset, args.results / "gcn.pt")
    print(f"GCN: test accuracy {test_accuracy:.3f} ({time.time() - start:.0f} s)", flush=True)

    graph = dataset[GRAPH]
    atoms = [ATOMS[label] for label in graph.x.argmax(dim=1).tolist()]
    bonds = sorted({(min(i, j), max(i, j)) for i, j in graph.edge_index.T.tolist()})
    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        logits = model(graph.x, graph.edge_index, batch)[0]

    game = GraphGame(model, graph, class_index=MUTAGEN)
    graphshapiq = GraphSHAPIQ(game, game.graph)
    moebius = graphshapiq.moebius
    largest = max(len(interaction) for interaction in moebius.interaction_lookup)
    converter = MoebiusConverter(moebius)

    # the game's values from the exact Möbius transform, checked against the model on 512 of them
    coalitions = sample(game.n_players, N_TEST, seed=1)
    y = predictions({key: moebius[key] for key in moebius.interaction_lookup}, coalitions)
    difference = np.max(np.abs(y[:512] - game(coalitions[:512])))
    print(f"largest difference of the Möbius transform to the model: {difference:.1e}", flush=True)
    assert difference < 1e-4  # noqa: S101  (GraphSHAP-IQ is exact, up to float32)
    orders = {}
    for order in range(1, largest + 1):
        values = fsii(converter, order)
        strongest = sorted(
            ((key, value) for key, value in values.items() if key),
            key=lambda item: (len(item[0]), -abs(item[1])),
        )
        orders[order] = {
            "r2": r2(y, predictions(values, coalitions), coalitions.sum(axis=1)),
            "values": [[list(interaction), value] for interaction, value in strongest],
        }
        print(f"order {order}: {len(strongest)} values, R^2 {orders[order]['r2']:.4f}", flush=True)

    result = {
        "index": "FSII",
        "graph": GRAPH,
        "atoms": atoms,
        "bonds": bonds,
        "coords": layout(atoms, bonds).round(4).tolist(),
        "test_accuracy": test_accuracy,
        "logit": float(logits[MUTAGEN]),
        "probability": float(logits.softmax(dim=0)[MUTAGEN]),
        "empty": float(game.empty_value),
        "budget": graphshapiq.budget,
        "largest_order": largest,
        "orders": orders,
    }
    (args.results / "molecule.json").write_text(json.dumps(result))
    print(
        f"molecule {GRAPH}: {len(atoms)} atoms, P(mutagen) {result['probability']:.3f}, "
        f"{graphshapiq.budget} evaluations, interactions up to order {largest}"
    )


if __name__ == "__main__":
    main()
