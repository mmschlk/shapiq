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

# Create toy graph
x = torch.tensor(
    [
        [1.0, 5.0, 3.0],
        [4.0, 2.0, 6.0],
    ],
    dtype=torch.float,
)

edge_index = torch.tensor(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=torch.long,
)

batch = torch.tensor([0, 0], dtype=torch.long)

x_graph = Data(
    x=x,
    edge_index=edge_index,
    batch=batch,
)

# Create model
model = GCN2Layer(
    in_channels=3,
    hidden_channels=4,
    num_layers=2,
    out_channels=2,
)

# Create explainer
explainer = GraphExplainer(
    model,
    l_shapley_max_budget=20_000,
)

# ------------------------------------------------------------------
# GraphSHAP-IQ explanation
# ------------------------------------------------------------------

mi = explainer.explain(x_graph)

logger.info("GraphSHAP-IQ Möbius interactions:")
logger.info(mi)

# ------------------------------------------------------------------
# L-Shapley explanation
# ------------------------------------------------------------------

l_shapley_values = explainer.explain(
    x_graph,
    l_shapley=True,
    max_interaction_size=2,
)

logger.info("L-Shapley values:")
logger.info(l_shapley_values)
