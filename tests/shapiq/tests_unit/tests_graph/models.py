"""Small graph neural networks with a linear global pooling and readout, and graphs to explain."""

from __future__ import annotations

from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool, global_mean_pool


class GNN(torch.nn.Module):
    """``num_layers`` message passing layers, a global pooling, and a linear readout."""

    def __init__(
        self,
        layer: Literal["gcn", "gin", "gat"] = "gcn",
        *,
        num_layers: int = 2,
        n_features: int = 3,
        n_outputs: int = 1,
        pooling: Literal["mean", "add"] = "mean",
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers
        hidden = 8
        convs = []
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
        self.pool = global_mean_pool if pooling == "mean" else global_add_pool
        self.readout = torch.nn.Linear(hidden, n_outputs)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.readout(self.pool(x, batch))


def undirected_graph(edges: list[tuple[int, int]], n_nodes: int, seed: int = 0) -> Data:
    """A graph with random node features and the edges in both directions."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).T
    return Data(x=torch.rand(n_nodes, 3, generator=generator), edge_index=edge_index)


def path_graph(n_nodes: int, seed: int = 0) -> Data:
    return undirected_graph([(i, i + 1) for i in range(n_nodes - 1)], n_nodes, seed)
