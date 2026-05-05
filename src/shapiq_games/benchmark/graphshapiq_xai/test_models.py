"""GNN models for testing."""

from __future__ import annotations

import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN


class GCN2Layer(GCN):
    """GCN with Graph level predicitons for testing purposes."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        *,
        graph_bias: bool = False,
        node_bias: bool = False,
    ) -> None:
        """Initializing parent GCN and linear layer."""
        # Initialize the parent PyG GCN.
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            bias=node_bias,
        )
        # Define the custom final linear layer for our graph-level prediction
        self.fc = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, batch: torch.tensor
    ) -> torch.tensor:
        """Forward pass."""
        # 1. Run the built-in GCN layers from the parent class
        x = super().forward(x, edge_index)

        # 2. Pool the node embeddings into a single graph embedding
        x = global_mean_pool(x, batch)

        # 3. Pass through our final linear classifier
        return self.fc(x)
