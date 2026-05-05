"""Testing script."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame
from shapiq_games.benchmark.graphshapiq_xai.test_models import GCN2Layer

x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=torch.float)

edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

batch = torch.tensor([0, 0], dtype=torch.long)
x_graph = Data(x=x, edge_index=edge_index, batch=batch)

model = GCN2Layer(in_channels=3, hidden_channels=4, num_layers=2, out_channels=2)

game = GraphGame(
    model=model,
    x_graph=x_graph,
    baseline_strategy="max",
    normalize=False,
    class_id=None,
    verbose=True,
)

coalition = np.array([1, 0])
masked_graph = game.mask_input(coalition)

# Uncomment. Linting doesn't let me push this.
"""
print("Original Graph Features")
print(x_graph.x, "\n")
print("Coalition (Mask Array)")
print(coalition, "\n")
print("Masked Graph Features")
print(masked_graph.x)
"""
