import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from shapiq_games.benchmark.graphshapiq_xai.base import GraphGame
import pytest

# Define a simple GNN model
class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features: int, output_dim: int = 1):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, output_dim)
        self.num_layers = 2

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define a test graph
def create_test_graph(num_nodes=5, num_node_features=3):
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# Test for correct initialization
def test_init_default():
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph)
    assert game.n_players == 5
    assert game.output_dim == 1
    assert game.y_index == 0
    assert torch.allclose(game.baseline, torch.zeros(3))

def test_init_with_class_index():
    """Test for initialization with custom class index"""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph, class_index=1)
    assert game.y_index == 1

def test_init_with_baseline_strategy():
    """Test for initialization with baseline strategy average"""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph, baseline_strategy="average")
    assert torch.allclose(game.baseline, x_graph.x.mean(dim=0))

def test_init_normalize():
    """Test for initialization with normalization"""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph, normalize=True)
    assert game.normalize is True
    assert game.normalization_value is not None

@pytest.mark.filterwarnings("error:Baseline is not provided")
def test_init_warning_no_baseline():
    """Test for initialization with no baseline strategy"""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    with pytest.warns(UserWarning, match="Baseline is not provided"):
        GraphGame(model, x_graph, baseline_strategy=None)

def test_mask_input_all_active():
    """Test mask_input, if all nodes are active."""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph, baseline_strategy="zeros")
    coalition = np.array([1, 1, 1, 1, 1])
    masked_graph = game.mask_input(coalition)
    assert torch.allclose(masked_graph.x, x_graph.x)

def test_mask_input_all_inactive():
    """Test mask_input, if all nodes are inactive."""
    model = SimpleGNN(num_node_features=3)
    x_graph = create_test_graph()
    game = GraphGame(model, x_graph, baseline_strategy="zeros")
    coalition = np.array([0, 0, 0, 0, 0])
    masked_graph = game.mask_input(coalition)
    for i in range(5):
        assert torch.allclose(masked_graph.x[i], game.baseline)