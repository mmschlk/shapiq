"""
Test module for the GraphExplainer class in shapiq.graph.

This module contains tests for the GraphExplainer class, which is used to compute
Shapley interaction values for graph neural networks (GNNs) using the GraphSHAP-IQ algorithm.
"""


import pytest
import numpy as np
from torch_geometric.data import Data
from shapiq.graph import GraphExplainer
from shapiq.interaction_values import InteractionValues

class TestGraphExplainer:
    """Tests for the GraphExplainer class."""

    def test_init_sets_attributes(self, gcn_model):
        expl = GraphExplainer(model=gcn_model, class_index=None, baseline_strategy="average", normalize=True)
        assert expl._model is not None
        assert expl._class_index is None
        assert expl._baseline_strategy == "average"
        assert expl._normalize is True

    def test_explain_requires_data_object(self, gcn_model):
        expl = GraphExplainer(model=gcn_model)
        with pytest.raises(TypeError, match="requires a torch_geometric.data.Data object"):
            expl.explain(x=np.zeros((3,)))  # not a Data object

    def test_explain_returns_interaction_values(self, gcn_model, simple_graph):
        expl = GraphExplainer(model=gcn_model)
        iv = expl.explain(simple_graph)
        assert isinstance(iv, InteractionValues)

    def test_explain_function_l_shapley_and_graphshapiq_branches(self, gcn_model, simple_graph):
        expl = GraphExplainer(model=gcn_model, l_shapley_max_budget=1000000)  # large budget so no runtime error
        # GraphSHAPIQ branch
        iv_exact = expl.explain_function(simple_graph, l_shapley=False)
        assert isinstance(iv_exact, InteractionValues)
        # L-Shapley branch
        iv_l = expl.explain_function(simple_graph, l_shapley=True, max_interaction_size=1)
        assert isinstance(iv_l, InteractionValues)
        assert iv_l.estimated is True

    def test_explain_X_list_and_numpy_error(self, gcn_model, simple_graph):
        expl = GraphExplainer(model=gcn_model)
        # explain_X should raise when passed a numpy array
        with pytest.raises(TypeError):
            expl.explain_X(np.zeros((2,2)))
        # explain_X with a list of Data should return list of InteractionValues
        result = expl.explain_X([simple_graph, simple_graph], n_jobs=None)
        assert isinstance(result, list)
        assert all(isinstance(r, InteractionValues) for r in result)
        result = expl.explain_X([simple_graph, simple_graph], n_jobs=4)
        assert isinstance(result, list)
        assert all(isinstance(r, InteractionValues) for r in result)


    def test__check_total_budget_raises_when_exceeds(self, gcn_model, simple_graph):
        # Set zero budget so the check in explain_function will raise
        expl = GraphExplainer(model=gcn_model, l_shapley_max_budget=0)
        with pytest.raises(RuntimeError, match="exceeds the limit"):
            expl.explain_function(simple_graph, l_shapley=False)