import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data

class SimpleGCN(nn.Module):
    def __init__(self,
                 num_node_features: int = 3,
                 output_dim: int = 1,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, 16))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(16, 16))
        self.lin = nn.Linear(16, output_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.lin(x)

class SimpleGIN(nn.Module):
    def __init__(self,
                 num_node_features: int = 3,
                 output_dim: int = 1,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        mlp = nn.Sequential(nn.Linear(num_node_features, 16), nn.ReLU(), nn.Linear(16, 16))
        self.convs.append(GINConv(mlp))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
            self.convs.append(GINConv(mlp))
        self.lin = nn.Linear(16, output_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.lin(x)

class SimpleGAT(nn.Module):
    def __init__(self,
                 num_node_features: int = 3,
                 output_dim: int = 1,
                 num_layers: int = 2,
                 heads: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, 16, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(16 * heads, 16, heads=heads))
        self.lin = nn.Linear(16 * heads, output_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.lin(x)

@pytest.fixture
def gcn_model():
    return SimpleGCN(num_node_features=3, output_dim=1, num_layers=2)

@pytest.fixture
def gin_model():
    return SimpleGIN(num_node_features=3, output_dim=1, num_layers=2)

@pytest.fixture
def gat_model():
    return SimpleGAT(num_node_features=3, output_dim=1, num_layers=2, heads=1)

@pytest.fixture
def gcn_model_classification():
    return SimpleGCN(num_node_features=3, output_dim=2, num_layers=2)

@pytest.fixture
def gin_model_classification():
    return SimpleGIN(num_node_features=3, output_dim=2, num_layers=2)

@pytest.fixture
def gat_model_classification():
    return SimpleGAT(num_node_features=3, output_dim=2, num_layers=2, heads=1)

@pytest.fixture(params=[1, 2, 3])
def gcn_model_varied_layers(request):
    return SimpleGCN(num_node_features=3, output_dim=1, num_layers=request.param)

@pytest.fixture(params=[8, 16, 32])
def gcn_model_varied_hidden_dim(request):
    return SimpleGCN(num_node_features=3, output_dim=1, num_layers=2, hidden_dim=request.param)

@pytest.fixture(params=[1, 3, 5, 10])
def gcn_model_varied_in_dim(request):
    return SimpleGCN(num_node_features=request.param, output_dim=1, num_layers=2)

@pytest.fixture(params=["gcn", "gin", "gat"])
def any_gnn_model(request):
    if request.param == "gcn":
        return SimpleGCN(num_node_features=3, output_dim=1, num_layers=2)
    elif request.param == "gin":
        return SimpleGIN(num_node_features=3, output_dim=1, num_layers=2)
    elif request.param == "gat":
        return SimpleGAT(num_node_features=3, output_dim=1, num_layers=2, heads=1)

@pytest.fixture
def simple_graph():
    x = torch.randn(4, 3)  # 4 Nodes, 3 Features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3], [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def small_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def disconnected_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def single_node_graph():
    x = torch.randn(1, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)