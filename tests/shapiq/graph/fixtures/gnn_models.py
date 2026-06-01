"""
Fixtures for GNN models (GCN, GIN, GAT) for testing in shapiq.graph.

These fixtures provide pre-defined PyTorch Geometric models for use in tests
for GraphGame, GraphSHAPIQ, and GraphExplainer.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv

class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) model for testing.

    Args:
        num_layers (int): Number of GCN layers. Default: 2.
        hidden_dim (int): Dimension of hidden layers. Default: 16.
        out_dim (int): Dimension of the output. Default: 1 (regression).
                      For classification: number of classes.
        in_dim (int): Dimension of input features. Default: 3.
    """
    def __init__(self, num_layers=2, hidden_dim=16, out_dim=1, in_dim=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN) model for testing.

    Args:
        num_layers (int): Number of GIN layers. Default: 2.
        hidden_dim (int): Dimension of hidden layers. Default: 16.
        out_dim (int): Dimension of the output. Default: 1 (regression).
        in_dim (int): Dimension of input features. Default: 3.
    """
    def __init__(self, num_layers=2, hidden_dim=16, out_dim=1, in_dim=3):
        super().__init__()
        self.convs = nn.ModuleList()
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return self.lin(x)

class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) model for testing.

    Args:
        num_layers (int): Number of GAT layers. Default: 2.
        hidden_dim (int): Dimension of hidden layers. Default: 16.
        out_dim (int): Dimension of the output. Default: 1 (regression).
        in_dim (int): Dimension of input features. Default: 3.
        heads (int): Number of attention heads. Default: 1.
    """
    def __init__(self, num_layers=2, hidden_dim=16, out_dim=1, in_dim=3, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.lin = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

@pytest.fixture
def gcn_model():
    """
    Fixture for a GCN model with default parameters.

    Returns:
        GCNModel: A GCN model with 2 layers, hidden_dim=16, out_dim=1.
    """
    return GCNModel()

@pytest.fixture
def gin_model():
    """
    Fixture for a GIN model with default parameters.

    Returns:
        GINModel: A GIN model with 2 layers, hidden_dim=16, out_dim=1.
    """
    return GINModel()

@pytest.fixture
def gat_model():
    """
    Fixture for a GAT model with default parameters.

    Returns:
        GATModel: A GAT model with 2 layers, hidden_dim=16, out_dim=1.
    """
    return GATModel()

@pytest.fixture(params=[1, 2, 3])
def gcn_model_varied_layers(request):
    """
    Fixture for GCN models with a variable number of layers (1, 2, 3).

    Args:
        request: Pytest fixture request object.

    Returns:
        GCNModel: A GCN model with `request.param` layers.
    """
    return GCNModel(num_layers=request.param)

@pytest.fixture(params=[8, 16, 32])
def gcn_model_varied_hidden_dim(request):
    """
    Fixture for GCN models with a variable hidden dimension (8, 16, 32).

    Args:
        request: Pytest fixture request object.

    Returns:
        GCNModel: A GCN model with `request.param` hidden dimensions.
    """
    return GCNModel(hidden_dim=request.param)

@pytest.fixture
def gcn_model_classification():
    """
    Fixture for a GCN model for classification (2 classes).

    Returns:
        GCNModel: A GCN model with out_dim=2.
    """
    return GCNModel(out_dim=2)

@pytest.fixture
def gin_model_classification():
    """
    Fixture for a GIN model for classification (2 classes).

    Returns:
        GINModel: A GIN model with out_dim=2.
    """
    return GINModel(out_dim=2)

@pytest.fixture
def gat_model_classification():
    """
    Fixture for a GAT model for classification (2 classes).

    Returns:
        GATModel: A GAT model with out_dim=2.
    """
    return GATModel(out_dim=2)

# --- Fixtures for Models with Different Input Dimensions ---
@pytest.fixture(params=[1, 3, 5, 10])
def gcn_model_varied_in_dim(request):
    """
    Fixture for GCN models with a variable input dimension (1, 3, 5, 10).

    Args:
        request: Pytest fixture request object.

    Returns:
        GCNModel: A GCN model with `request.param` input features.
    """
    return GCNModel(in_dim=request.param)

@pytest.fixture(params=["gcn", "gin", "gat"])
def any_gnn_model(request):
    """
    Fixture for any GNN model (GCN, GIN, or GAT).

    Args:
        request: Pytest fixture request object.

    Returns:
        nn.Module: A GNN model of type `request.param`.
    """
    if request.param == "gcn":
        return GCNModel()
    elif request.param == "gin":
        return GINModel()
    elif request.param == "gat":
        return GATModel()