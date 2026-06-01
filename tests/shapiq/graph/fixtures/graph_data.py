import pytest
import torch
from torch_geometric.data import Data

@pytest.fixture
def simple_graph():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 3],
        [1, 0, 2, 1, 3, 2, 3, 0]
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def small_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
