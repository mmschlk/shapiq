"""
Test module for the GraphGame class in shapiq.graph.
"""
import pytest

from shapiq.graph.base import GraphGame
from tests.shapiq.graph.fixtures.gnn_models import (
    gcn_model,
    gin_model,
    gat_model,
    gcn_model_classification,
)
from tests.shapiq.graph.fixtures.graph_data import simple_graph, small_graph

@pytest.fixture
def gcn_graph_game(gcn_model, simple_graph):
    """Create a GraphGame instance with a GCN model and a simple graph."""
    return GraphGame(
        model=gcn_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )

@pytest.fixture
def gin_graph_game(gin_model, simple_graph):
    """Create a GraphGame instance with a GIN model and a simple graph."""
    return GraphGame(
        model=gin_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )

@pytest.fixture
def gat_graph_game(gat_model, simple_graph):
    """Create a GraphGame instance with a GAT model and a simple graph."""
    return GraphGame(
        model=gat_model,
        x_graph=simple_graph,
        task="regression",
        baseline_strategy="average",
    )

@pytest.fixture
def gcn_graph_game_classification(gcn_model_classification, simple_graph):
    """Create a GraphGame instance for classification with a GCN model."""
    return GraphGame(
        model=gcn_model_classification,
        x_graph=simple_graph,
        task="classification",
        class_index=0,
        baseline_strategy="average",
    )