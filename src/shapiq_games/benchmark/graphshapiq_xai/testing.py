"""Testing script."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame
from shapiq_games.benchmark.graphshapiq_xai.test_models import GCN2Layer

logger = logging.getLogger()

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

logger.log("Original Graph Features")
logger.log(x_graph.x, "\n")
logger.log("Coalition (Mask Array)")
logger.log(coalition, "\n")
logger.log("Masked Graph Features")
logger.log(masked_graph.x)
