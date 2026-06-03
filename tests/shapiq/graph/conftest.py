"""
Central pytest fixtures for GraphGame and GraphSHAPIQ tests.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data

from shapiq.graph.base import GraphGame
from shapiq.graph.graphshapiq import GraphSHAPIQ

class SimpleGCN(nn.Module):
    def __init__(self, num_node_features=3, output_dim=1, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(num_node_features, 16)])
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
    def __init__(self, num_node_features=3, output_dim=1, num_layers=2):
        super().__init__()
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
    def __init__(self, num_node_features=3, output_dim=1, num_layers=2, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(num_node_features, 16, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(16 * heads, 16, heads=heads))

        self.lin = nn.Linear(16 * heads, output_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x).relu()

        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return self.lin(x)

@pytest.fixture
def simple_graph():
    x = torch.randn(4, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]],
        dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def small_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4],
         [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def disconnected_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 4],
         [1, 0, 3, 2, 4, 3]],
        dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def single_node_graph():
    x = torch.randn(1, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

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


# =========================================================
# GraphGame fixtures
# =========================================================

@pytest.fixture
def gcn_graph_game(gcn_model, simple_graph):
    return GraphGame(
        model=gcn_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )


@pytest.fixture
def gin_graph_game(gin_model, simple_graph):
    return GraphGame(
        model=gin_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )


@pytest.fixture
def gat_graph_game(gat_model, simple_graph):
    return GraphGame(
        model=gat_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )


@pytest.fixture
def gcn_graph_game_small(gcn_model, small_graph):
    return GraphGame(
        model=gcn_model,
        x_graph=small_graph,
        task="regression",
        baseline_strategy="average",
    )


@pytest.fixture
def gcn_graph_game_disconnected(gcn_model, disconnected_graph):
    return GraphGame(
        model=gcn_model,
        x_graph=disconnected_graph,
        task="regression",
        baseline_strategy="average",
    )


@pytest.fixture
def gcn_graph_game_single_node(gcn_model, single_node_graph):
    return GraphGame(
        model=gcn_model,
        x_graph=single_node_graph,
        task="regression",
        baseline_strategy="average",
    )


# =========================================================
# GraphSHAPIQ fixtures
# =========================================================

@pytest.fixture
def gcn_graphshapiq(gcn_graph_game):
    return GraphSHAPIQ(game=gcn_graph_game)


@pytest.fixture
def gin_graphshapiq(gin_graph_game):
    return GraphSHAPIQ(game=gin_graph_game)


@pytest.fixture
def gat_graphshapiq(gat_graph_game):
    return GraphSHAPIQ(game=gat_graph_game)


@pytest.fixture
def gcn_graphshapiq_small(gcn_graph_game_small):
    return GraphSHAPIQ(game=gcn_graph_game_small)


@pytest.fixture
def gcn_graphshapiq_disconnected(gcn_graph_game_disconnected):
    return GraphSHAPIQ(game=gcn_graph_game_disconnected)


@pytest.fixture
def gcn_graphshapiq_single_node(gcn_graph_game_single_node):
    return GraphSHAPIQ(game=gcn_graph_game_single_node)