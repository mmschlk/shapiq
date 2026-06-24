"""End-to-end tests for the GraphExplainer class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.graph import GraphExplainer
from shapiq.interaction_values import InteractionValues


class TestGraphExplainerE2E:
    """Tests for the GraphExplainer class."""

    def test_init_sets_attributes(self, gcn_model):
        expl = GraphExplainer(
            model=gcn_model, class_index=None, baseline_strategy="average", normalize=True
        )
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
        expl = GraphExplainer(
            model=gcn_model, l_shapley_max_budget=1000000
        )  # large budget so no runtime error
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
            expl.explain_X(np.zeros((2, 2)))
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
            expl.explain_function(simple_graph, l_shapley=True)

    def test_graphshapiq_estimated_flag_is_false(self, gcn_model, simple_graph):
        """GraphSHAP-IQ is exact so estimated must be False."""
        expl = GraphExplainer(model=gcn_model)
        iv = expl.explain(simple_graph)
        assert iv.estimated is False

    def test_graphshapiq_estimation_budget_set(self, gcn_model, simple_graph):
        """estimation_budget should be populated and positive after explanation."""
        expl = GraphExplainer(model=gcn_model)
        iv = expl.explain(simple_graph)
        assert iv.estimation_budget is not None
        assert iv.estimation_budget > 0

    def test_l_shapley_estimated_flag_is_true(self, gcn_model, simple_graph):
        """L-Shapley is an approximation so estimated must be True."""
        expl = GraphExplainer(model=gcn_model, l_shapley_max_budget=10_000_000)
        iv = expl.explain(simple_graph, l_shapley=True)
        assert iv.estimated is True

    def test_graphshapiq_index_matches_explainer_setting(self, gcn_model, simple_graph):
        """Returned index should match the index set on the explainer."""
        for index in ("k-SII", "SV", "SII"):
            expl = GraphExplainer(model=gcn_model, index=index, max_order=2)
            iv = expl.explain(simple_graph)
            assert iv.index == index

    def test_kwargs_override_index_at_call_time(self, gcn_model, simple_graph):
        """index kwarg passed to explain() should override the explainer-level index."""
        expl = GraphExplainer(model=gcn_model, index="k-SII", max_order=2)
        iv = expl.explain(simple_graph, index="SV")
        assert iv.index == "SV"

    def test_kwargs_override_efficiency_routine_at_call_time(self, gcn_model, simple_graph):
        """efficiency_routine kwarg passed to explain() should override the stored setting."""
        expl = GraphExplainer(model=gcn_model, efficiency_routine=True)
        iv = expl.explain(simple_graph, efficiency_routine=False)
        assert isinstance(iv, InteractionValues)

    def test_kwargs_max_subset_size_reduces_model_calls(self, gcn_model, simple_graph):
        """max_subset_size kwarg should reduce the number of model calls vs the full run."""
        expl = GraphExplainer(model=gcn_model, l_shapley_max_budget=10_000_000)
        iv_full = expl.explain(simple_graph)
        iv_restricted = expl.explain(simple_graph, max_subset_size=1)
        assert iv_restricted.estimation_budget < iv_full.estimation_budget

    def test_max_order_1_produces_only_singletons(self, gcn_model, simple_graph):
        """max_order=1 should produce no interactions of order > 1."""
        expl = GraphExplainer(model=gcn_model, max_order=1, index="SV")
        iv = expl.explain(simple_graph)
        for interaction in iv.interaction_lookup:
            assert len(interaction) <= 1

    def test_max_order_2_produces_pairwise_interactions(self, gcn_model, simple_graph):
        """max_order=2 should produce at least some order-2 interactions."""
        expl = GraphExplainer(model=gcn_model, max_order=2, index="k-SII")
        iv = expl.explain(simple_graph)
        assert any(len(interaction) == 2 for interaction in iv.interaction_lookup)

    def test_l_shapley_and_graphshapiq_sv_node_count_matches(self, gcn_model, simple_graph):
        """Both methods should return one value per node."""
        expl = GraphExplainer(
            model=gcn_model, index="SV", max_order=1, l_shapley_max_budget=10_000_000
        )
        iv_exact = expl.explain(simple_graph, l_shapley=False)
        iv_approx = expl.explain(simple_graph, l_shapley=True)
        assert iv_exact.n_players == iv_approx.n_players
        assert len([k for k in iv_exact.interaction_lookup if len(k) == 1]) == iv_exact.n_players
        assert len([k for k in iv_approx.interaction_lookup if len(k) == 1]) == iv_approx.n_players

    def test_l_shapley_and_graphshapiq_sv_values_close(self, gcn_model, simple_graph):
        """L-Shapley and GraphSHAP-IQ with index='SV' should produce close Shapley values."""
        expl = GraphExplainer(
            model=gcn_model, index="SV", max_order=1, l_shapley_max_budget=10_000_000
        )
        iv_exact = expl.explain(simple_graph, l_shapley=False)
        iv_approx = expl.explain(simple_graph, l_shapley=True)
        exact_vals = np.array([iv_exact[(i,)] for i in range(iv_exact.n_players)])
        approx_vals = np.array([iv_approx[(i,)] for i in range(iv_approx.n_players)])
        assert np.allclose(exact_vals, approx_vals, atol=1e-4)

    def test_explain_X_results_match_individual_explain(self, gcn_model, simple_graph):
        """explain_X should produce identical results to individual explain calls."""
        expl = GraphExplainer(model=gcn_model)
        iv_batch = expl.explain_X([simple_graph, simple_graph])
        iv_single = expl.explain(simple_graph)
        assert np.allclose(iv_batch[0].values, iv_single.values)
        assert np.allclose(iv_batch[1].values, iv_single.values)

    def test_explain_X_parallel_matches_sequential(self, gcn_model, simple_graph):
        """explain_X with n_jobs=-1 should produce same results as sequential."""
        expl = GraphExplainer(model=gcn_model)
        iv_seq = expl.explain_X([simple_graph, simple_graph])
        iv_par = expl.explain_X([simple_graph, simple_graph], n_jobs=-1)
        assert np.allclose(iv_seq[0].values, iv_par[0].values)
        assert np.allclose(iv_seq[1].values, iv_par[1].values)

    def test_normalize_true_baseline_is_zero(self, gcn_model, simple_graph):
        """With normalize=True the returned baseline_value should be zero."""
        expl = GraphExplainer(model=gcn_model, normalize=True)
        iv = expl.explain(simple_graph)
        assert abs(iv.baseline_value) < 1e-6
