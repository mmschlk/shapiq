"""The graph neural networks of the graph setups and their training (imports torch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from ._graph import GraphModelName

__all__ = ["GraphNeuralNetwork", "train"]

_TRAINING_BATCH_SIZE = 32


class GraphNeuralNetwork(torch.nn.Module):
    """Message passing layers with ReLU, a mean over the nodes, and a linear readout.

    The pooling and the readout are linear, so the network's game is local on the graph and
    GraphSHAP-IQ computes its values exactly.
    """

    def __init__(
        self, layer: GraphModelName, n_features: int, n_outputs: int, num_layers: int, hidden: int
    ) -> None:
        """Initialize the network with the current seed of torch."""
        super().__init__()
        self.num_layers = num_layers
        convs: list[torch.nn.Module] = []
        for i in range(num_layers):
            size = n_features if i == 0 else hidden
            if layer == "gcn":
                convs.append(GCNConv(size, hidden))
            elif layer == "gin":
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(size, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
                )
                convs.append(GINConv(mlp))
            else:
                convs.append(GATConv(size, hidden))
        self.convs = torch.nn.ModuleList(convs)
        self.readout = torch.nn.Linear(hidden, n_outputs)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return the logits of every graph of the batch."""
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.readout(global_mean_pool(x, batch))


def train(
    graphs: list[Data],
    layer: GraphModelName,
    *,
    n_classes: int,
    num_layers: int,
    hidden: int,
    n_epochs: int,
    learning_rate: float,
    random_state: int,
) -> GraphNeuralNetwork:
    """Train a classifier on the graphs with Adam and the cross-entropy loss, seeded.

    Returns:
        The trained network in evaluation mode.
    """
    torch.manual_seed(random_state)
    model = GraphNeuralNetwork(layer, graphs[0].num_node_features, n_classes, num_layers, hidden)
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(graphs, batch_size=_TRAINING_BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(batch.y).reshape(-1)
            ).backward()
            optimizer.step()
    return model.eval()
