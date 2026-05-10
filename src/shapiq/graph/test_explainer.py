"""Testing script."""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import Data

from shapiq.graph.explainer import GraphExplainer
from shapiq_games.benchmark.graphshapiq_xai.test_models import GCN2Layer

logger = logging.getLogger("explainer")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=torch.float)

edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

batch = torch.tensor([0, 0], dtype=torch.long)
x_graph = Data(x=x, edge_index=edge_index, batch=batch)

model = GCN2Layer(in_channels=3, hidden_channels=4, num_layers=2, out_channels=2)

explainer = GraphExplainer(model, x_graph)

mi = explainer.explain()
logger.info(f"Möbius interactions: {mi}")
