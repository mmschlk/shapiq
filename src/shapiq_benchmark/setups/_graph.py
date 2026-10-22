"""The setup of the graph games: a graph neural network trained on a graph dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split

from shapiq.graph import GraphBaseline, GraphGame
from shapiq_benchmark.datasets import GraphDatasetName, load_graph_dataset

from ._base import Setup, runtime_field

__all__ = ["GraphSetup"]

GraphModelName = Literal["gcn", "gin", "gat"]
"""The message passing layers of the graph neural networks of :class:`GraphSetup`."""


@dataclass(frozen=True, kw_only=True)
class GraphSetup(Setup, name="graph"):
    """A :class:`~shapiq_games.GraphGame` of a graph neural network trained on a graph dataset.

    The network has ``n_layers`` message passing layers of the ``model`` with ReLU, a mean over
    the nodes, and a linear readout of the class logits, so its game is local on the graph and
    :class:`~shapiq_benchmark.computers.GraphComputer` computes its exact values with
    GraphSHAP-IQ. It is trained with Adam on the training split of the dataset (from
    :func:`~shapiq_benchmark.datasets.load_graph_dataset`, stratified); the explained graph is
    taken from the test split.

    Attributes:
        dataset: The graph dataset. Defaults to ``"MUTAG"``.
        model: The message passing layers, ``"gcn"`` (default), ``"gin"``, or ``"gat"``.
        n_layers: The number of message passing layers, the radius of the receptive fields.
            Defaults to ``2``.
        hidden_size: The size of the node embeddings. Defaults to ``32``.
        n_epochs: The number of training epochs. Defaults to ``100``.
        learning_rate: The learning rate of Adam. Defaults to ``0.01``.
        x: The index of the explained graph in the test split. Defaults to ``0``.
        class_index: The explained class (``None`` means class ``1``).
        baseline: The node features of absent nodes (see :class:`~shapiq_games.GraphGame`).
            Defaults to ``"zeros"``.
        test_size: The fraction of the graphs in the test split. Defaults to ``0.2``.
        random_state: The seed of the split, the initialization, and the training.
        normalize: Whether to center the game. Defaults to ``True``.
        batch_size: The number of masked graphs per forward pass (a runtime field).

    Examples:
        >>> setup = GraphSetup(dataset="MUTAG", model="gin", x=3)
        >>> game = setup.build()  # doctest: +SKIP
    """

    dataset: GraphDatasetName = "MUTAG"
    model: GraphModelName = "gcn"
    n_layers: int = 2
    hidden_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 0.01
    x: int = 0
    class_index: int | None = None
    baseline: GraphBaseline | float = "zeros"
    test_size: float = 0.2
    random_state: int = 42
    normalize: bool = True
    batch_size: int = runtime_field(128)

    def build(self) -> GraphGame:
        """Load the graphs, train the network, and build the game."""
        from ._gnn import train  # imports torch

        graphs = load_graph_dataset(self.dataset)
        labels = np.array([np.asarray(graph.y).item() for graph in graphs], dtype=int)
        train_rows, test_rows = train_test_split(
            np.arange(len(graphs)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        model = train(
            [graphs[i] for i in train_rows],
            self.model,
            n_classes=int(labels.max()) + 1,
            num_layers=self.n_layers,
            hidden=self.hidden_size,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        return GraphGame(
            model,
            graphs[int(np.sort(test_rows)[self.x])],
            class_index=self.class_index,
            baseline=self.baseline,
            batch_size=self.batch_size,
            normalize=self.normalize,
        )
