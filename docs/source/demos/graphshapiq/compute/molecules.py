"""What the GraphSHAP-IQ demo's scripts share: molecules as graphs, the datasets, and the GNN.

A molecule is a graph of its heavy atoms (no hydrogens), linked by its bonds. As in B-XAIC
(Proszewska et al., 2025), a GNN sees an atom only as its element, one-hot over ``SYMBOLS``, and a
bond only as a link: not its order, nor whether it is aromatic. Every task's GNN is B-XAIC's GIN:
three GIN layers with batch normalization, a sum over the atoms, and a linear readout of two
logits, so the explained log-odds are local on the molecule's graph and GraphSHAP-IQ computes their
interactions exactly.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GIN, global_add_pool

if TYPE_CHECKING:
    from pathlib import Path

RDLogger.DisableLog("rdApp.*")  # B-XAIC's few SMILES that RDKit cannot read are skipped

SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]  # B-XAIC's elements
BXAIC = "https://huggingface.co/datasets/mproszewska/B-XAIC/resolve/main/data.csv"
MUTAGENICITY = "https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip"
# the atoms of Mutagenicity's node labels, in their order (its README)
MUTAGENICITY_ATOMS = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]


def one_hot(symbols: list[str]) -> torch.Tensor:
    """Return the elements as B-XAIC's node features: one-hot, other elements as ``"Unk"``."""
    labels = [SYMBOLS.index(s) if s in SYMBOLS else len(SYMBOLS) - 1 for s in symbols]
    return torch.nn.functional.one_hot(torch.tensor(labels), len(SYMBOLS)).float()


def to_graph(mol: Chem.Mol, label: int | None = None) -> Data:
    """Return a molecule's graph: its atoms' elements and its bonds, in both directions."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edges = torch.tensor(bonds + [(j, i) for i, j in bonds], dtype=torch.long).reshape(-1, 2)
    graph = Data(x=one_hot([atom.GetSymbol() for atom in mol.GetAtoms()]), edge_index=edges.T)
    if label is not None:
        graph.y = torch.tensor([label])
    return graph


def download(url: str, path: Path) -> Path:
    """Download a file once, to ``path``."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def load_bxaic(task: str, data: Path) -> tuple[list[Data], list[str]]:
    """Return B-XAIC's molecules as graphs labeled for a task, and their split (``split_0``)."""
    path = download(BXAIC, data / "bxaic" / "data.csv")
    graphs, splits = [], []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                graphs.append(to_graph(mol, int(row[task])))
                splits.append(row["split_0"])
    return graphs, splits


def load_mutagenicity(data: Path, seed: int = 0) -> tuple[list[Data], list[str]]:
    """Return Kazius et al.'s 4,337 molecules of TUDataset's Mutagenicity, mutagens labeled 1.

    The molecules lose their hydrogens, to be read like B-XAIC's, and are split at random into
    80% for training, 10% for validation and 10% for testing.
    """
    path = download(MUTAGENICITY, data / "mutagenicity" / "Mutagenicity.zip")
    with zipfile.ZipFile(path) as archive:

        def read(name: str) -> np.ndarray:
            text = archive.read(f"Mutagenicity/Mutagenicity_{name}.txt").decode()
            return np.loadtxt(io.StringIO(text), delimiter=",", dtype=int, ndmin=1)

        edges, of_graph = read("A") - 1, read("graph_indicator") - 1
        atoms, classes = read("node_labels"), read("graph_labels")  # class 0: mutagen
    edges = edges[np.argsort(of_graph[edges[:, 0]], kind="stable")]
    edge_starts = np.searchsorted(of_graph[edges[:, 0]], np.arange(len(classes) + 1))
    graphs = []
    for g, label in enumerate(classes):
        nodes = np.flatnonzero(of_graph == g)
        heavy = [node for node in nodes if MUTAGENICITY_ATOMS[atoms[node]] != "H"]
        index = {node: k for k, node in enumerate(heavy)}
        own = edges[edge_starts[g] : edge_starts[g + 1]]
        inside = [(index[a], index[b]) for a, b in own if a in index and b in index]
        graphs.append(
            Data(
                x=one_hot([MUTAGENICITY_ATOMS[atoms[node]] for node in heavy]),
                edge_index=torch.tensor(inside, dtype=torch.long).reshape(-1, 2).T,
                y=torch.tensor([int(label == 0)]),
            )
        )
    order = np.random.default_rng(seed).permutation(len(graphs))
    splits = np.empty(len(graphs), dtype=object)
    n = len(graphs)
    splits[order[: int(0.8 * n)]] = "train"
    splits[order[int(0.8 * n) : int(0.9 * n)]] = "valid"
    splits[order[int(0.9 * n) :]] = "test"
    return graphs, splits.tolist()


class GNN(torch.nn.Module):
    """B-XAIC's GIN: GIN layers, a sum over the atoms, and a linear readout of two logits."""

    def __init__(self, num_layers: int = 3, hidden: int = 32) -> None:
        """Initialize the GIN with the current seed of torch."""
        super().__init__()
        self.num_layers = num_layers  # GraphSHAP-IQ reads the radius of the receptive fields
        self.gin = GIN(len(SYMBOLS), hidden, num_layers, out_channels=hidden, norm="batch_norm")
        self.readout = torch.nn.Linear(hidden, 2)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return the two logits of every molecule of the batch."""
        return self.readout(global_add_pool(self.gin(x, edge_index), batch))


class LogOdds(torch.nn.Module):
    """A GNN's log-odds of the task, its logit of yes minus its logit of no.

    The difference of the logits is a linear readout too, so its game is local on the molecule.
    """

    def __init__(self, gnn: GNN) -> None:
        """Wrap the GNN."""
        super().__init__()
        self.gnn = gnn
        self.num_layers = gnn.num_layers

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return the log-odds of every molecule of the batch, of shape ``(n_molecules, 1)``."""
        logits = self.gnn(x, edge_index, batch)
        return logits[:, 1:] - logits[:, :1]
