"""
GraphSHAP-IQ for GNN Explanation
=================================

This example demonstrates how to use the GraphExplainer class to explain
GNN predictions using the GraphSHAP-IQ algorithm.

We use a small hand-crafted graph and a simple pre-trained GCN to show the API.
"""

from __future__ import annotations

import torch
import torch.nn.functional as f
from torch.nn import Linear

from shapiq.graph import GraphExplainer


def _check_import_torch_geometric():  # noqa: ANN202
    """Import torch_geometric Data or raise a helpful optional-dependency error."""
    try:
        import torch_geometric
    except ImportError as error:
        msg = (
            "GraphExplainer requires the optional graph dependencies. "
            "Install them with `pip install shapiq[graph]`."
        )
        raise ImportError(msg) from error

    return torch_geometric


torch_geometric = _check_import_torch_geometric()

# %%
# A Small Dummy Graph
# -------------------
# We define a simple 6-node graph by hand. Node features are random
# 8-dimensional vectors and the edges form a small connected structure.
#
# .. code-block:: text
#
#     0 — 1 — 2
#     |       |
#     3 — 4 — 5

torch.manual_seed(0)

NUM_NODES = 6
FEATURE_DIM = 8

x = torch.rand(NUM_NODES, FEATURE_DIM)

edge_index = torch.tensor(
    [
        [0, 1, 1, 2, 0, 3, 2, 5, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 0, 5, 2, 4, 3, 5, 4],
    ],
    dtype=torch.long,
)

# batch vector: all nodes belong to the same single graph
batch = torch.zeros(NUM_NODES, dtype=torch.long)

graph = torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch)
print(graph)

# %%
# Define a Simple 2-Layer GCN
# ----------------------------
# A lightweight GCN with two convolutional layers, global-sum pooling, and a
# linear classification head. In practice this model would be trained; here we
# use random weights to keep the example self-contained and focused on the
# GraphSHAP-IQ API.

NUM_CLASSES = 2


class SimpleGCN(torch.nn.Module):
    """Two-layer GCN for graph classification."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        """Initialize the GNN."""
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden)
        self.conv2 = torch_geometric.nn.GCNConv(hidden, hidden)
        self.head = Linear(hidden, out_channels)
        self.num_layers = 2

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass."""
        x = f.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        return self.head(x)


model = SimpleGCN(FEATURE_DIM, 16, NUM_CLASSES)
model.eval()

with torch.no_grad():
    logits = model(graph.x, graph.edge_index, graph.batch)
    pred = int(logits.argmax(dim=1).item())
print(f"Model prediction: {pred}")

# %%
# Explain with GraphSHAP-IQ
# --------------------------
# :class:`~shapiq.graph.GraphExplainer` computes *k-SII interaction indices* for
# every subset of nodes up to ``max_subset_size``.
#
# - ``max_order=1`` -> plain Shapley values (one value per node).
# - ``max_order=2`` -> additionally captures pairwise node interactions.

max_order = 2

# Instantiate the GraphExplainer
explainer = GraphExplainer(
    model,
    baseline_strategy="average",
    max_order=max_order,
    class_index=1,
)

# Explain the prediction using the graph we classified
iv = explainer.explain(graph)
print(iv)

node_names = [f"Patch {i}" for i in range(graph.x.shape[0])]
iv.plot_network(feature_names=node_names)
