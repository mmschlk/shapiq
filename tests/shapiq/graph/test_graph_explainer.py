"""End-to-end tests for the GraphExplainer class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shapiq.graph import GraphExplainer
from shapiq.graph.base import GraphGame
from shapiq.graph.graphshapiq import GraphSHAPIQ
from shapiq.interaction_values import InteractionValues

_pyg_data = pytest.importorskip("torch_geometric.data")
Data = _pyg_data.Data


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
        expl = GraphExplainer(model=gcn_model)
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

    def test_last_computation_exact_tracks_truncation(self, gcn_model):
        """last_computation_exact must reflect the four exactness regimes:
        uncapped -> True; cap == n_max-1 with efficiency routine -> True
        (Corollary D.1); same cap without the routine -> False; cap below
        the D.1 boundary -> False regardless of the routine.
        """

        # 6-node path: 2-hop neighborhoods are strictly smaller than the graph,
        # so max_subset_size caps cause real truncation.
        torch.manual_seed(0)
        n_nodes = 6
        edges = []
        for i in range(n_nodes - 1):
            edges.extend([(i, i + 1), (i + 1, i)])
        graph = Data(
            x=torch.randn(n_nodes, 3),
            edge_index=torch.tensor(edges, dtype=torch.long).T,
        )

        game = GraphGame(model=gcn_model, x_graph=graph, baseline_strategy="average")
        explainer = GraphSHAPIQ(game=game)
        n_max = explainer.max_size_neighbors

        # Guard the setup itself: truncation below n_max - 1 must be possible.
        assert n_max >= 3, "fixture too small to distinguish the truncation regimes"

        # Tri-state: unset before any explain call.
        assert explainer.last_computation_exact is None

        # 1) Uncapped run -> exact.
        explainer.explain(max_subset_size=None, use_cpp=False)
        assert explainer.last_computation_exact is True

        # 2) Cap at n_max - 1 WITH the efficiency routine -> exact (Corollary D.1).
        explainer.explain(max_subset_size=n_max - 1, efficiency_routine=True, use_cpp=False)
        assert explainer.last_computation_exact is True

        # 3) Same cap WITHOUT the routine -> the missing top-order MIs are
        #    simply dropped -> estimated.
        explainer.explain(max_subset_size=n_max - 1, efficiency_routine=False, use_cpp=False)
        assert explainer.last_computation_exact is False

        # 4) Cap below the D.1 boundary -> estimated even with the routine.
        explainer.explain(max_subset_size=n_max - 2, efficiency_routine=True, use_cpp=False)
        assert explainer.last_computation_exact is False


# Some edge cases
# 1 generator input to explain_X is silently consumed -> empty result
# 2 n_jobs=0 is falsy and silently runs sequentially
# 3 the tqdm progress bar is never closed (success or exception path)
# 4 compute_moebius_transform raises an opaque KeyError when the empty
# 5 coalition is missing from coalitions/lookup
# 6 max_subset_size / order / max_order are not validated
# 7 unknown kwargs (typos) are silently swallowed


try:  # pragma: no cover - environment-dependent
    import shapiq.graph.cext  # noqa: F401

    _HAS_CEXT = True
except ImportError:  # pragma: no cover
    _HAS_CEXT = False


requires_cext = pytest.mark.skipif(
    not _HAS_CEXT,
    reason="shapiq.graph.cext extension not compiled (required by GraphExplainer.explain)",
)


@requires_cext
class TestGeneratorInput:
    """Contract: a generator of Data objects must either be fully explained or
    rejected with a clear TypeError -- never silently produce an empty list.

    Current bug: ``all(isinstance(x, Data) for x in X)`` consumes the
    generator, so the explain loop iterates nothing and ``[]`` is returned.
    Recommended fix: ``X = list(X)`` before validation (then this test's
    'explained' branch is the live one). If you instead decide to reject
    generators, the 'raises' branch passes and nothing needs changing here.
    """

    def test_generator_is_not_silently_swallowed(self, gcn_model, simple_graph):
        explainer = GraphExplainer(model=gcn_model)

        gen = (g for g in [simple_graph, simple_graph])
        try:
            result = explainer.explain_X(gen)
        except TypeError:
            return  # rejecting generators explicitly is acceptable
        # If accepted, all items must actually have been explained.
        assert len(result) == 2
        assert all(isinstance(r, InteractionValues) for r in result)

    def test_generator_with_verbose_is_not_worse(self, gcn_model, simple_graph):
        """The verbose path calls len(X) on the input and currently crashes
        with a bare TypeError from len() on a generator. Whatever contract is
        chosen, verbose=True must behave the same as verbose=False."""
        explainer = GraphExplainer(model=gcn_model)
        gen = (g for g in [simple_graph, simple_graph])
        try:
            result = explainer.explain_X(gen, verbose=True)
        except TypeError as err:
            # Acceptable only if it is the same deliberate rejection as the
            # non-verbose path -- i.e. mentions Data objects, not len().
            assert "Data" in str(err)
            return
        assert len(result) == 2


# ---------------------------------------------------------------------------
# #13 -- n_jobs handling
# ---------------------------------------------------------------------------


@requires_cext
class TestNJobsHandling:
    def test_n_jobs_zero_is_rejected(self, gcn_model, simple_graph):
        """Contract: n_jobs=0 is meaningless for joblib and must raise a
        ValueError rather than silently falling back to sequential execution
        (current behavior via ``if n_jobs:``). If you instead decide that 0
        means 'sequential', replace this with an equality-of-results assert
        and document the convention in the docstring.
        """
        explainer = GraphExplainer(model=gcn_model)
        with pytest.raises(ValueError, match="n_jobs"):
            explainer.explain_X([simple_graph], n_jobs=0)

    def test_n_jobs_one_matches_sequential(self, gcn_model, simple_graph):
        """n_jobs=1 (truthy, goes through joblib) must give identical values
        to the default sequential path."""
        explainer = GraphExplainer(model=gcn_model)
        seq = explainer.explain_X([simple_graph, simple_graph])
        par = explainer.explain_X([simple_graph, simple_graph], n_jobs=1)
        assert np.allclose(seq[0].values, par[0].values)
        assert np.allclose(seq[1].values, par[1].values)


# ---------------------------------------------------------------------------
# #14 -- progress bar lifecycle
# ---------------------------------------------------------------------------


class _RecordingPbar:
    """Minimal tqdm stand-in that records update/close calls."""

    instances: list[_RecordingPbar] = []  # noqa: RUF012

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.n_updates = 0
        self.closed = False
        _RecordingPbar.instances.append(self)

    def update(self, n: int = 1) -> None:
        self.n_updates += n

    def close(self) -> None:
        self.closed = True


class TestMoebiusTransformInputValidation:
    """compute_moebius_transform is public; malformed inputs must fail with a
    clear ValueError, not an opaque KeyError from deep inside the loop."""

    def test_missing_empty_coalition_raises_value_error(self, gcn_graphshapiq):
        """The transform needs v(empty) for every subset expansion and for
        baseline_value; a coalition set without () must be rejected upfront.
        Currently: KeyError on coalition_lookup[()].
        """
        coalitions = {(0,), (0, 1)}
        lookup = {(0,): 0, (0, 1): 1}
        predictions = np.array([0.5, 1.0])

        with pytest.raises(ValueError, match="empty coalition"):
            gcn_graphshapiq.compute_moebius_transform(
                coalitions=coalitions,
                coalition_predictions=predictions,
                coalition_lookup=lookup,
            )

    def test_lookup_missing_a_required_subset_raises_value_error(self, gcn_graphshapiq):
        """Every subset of every coalition must be resolvable via the lookup;
        a hole must produce a clear error naming the missing subset, not a
        raw KeyError.
        """
        coalitions = {(), (0, 1)}  # (0,) and (1,) missing from the lookup
        lookup = {(): 0, (0, 1): 1}
        predictions = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match=r"\(0,\)|subset"):
            gcn_graphshapiq.compute_moebius_transform(
                coalitions=coalitions,
                coalition_predictions=predictions,
                coalition_lookup=lookup,
            )


# ---------------------------------------------------------------------------
# #16 -- parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    @requires_cext
    @pytest.mark.parametrize("bad_size", [0, -1, -100])
    def test_non_positive_max_subset_size_rejected_by_explainer(
        self, gcn_model, simple_graph, bad_size
    ):
        """max_subset_size < 1 makes the transform degenerate (empty coalition
        only) or hits undefined powerset behavior; reject with ValueError."""
        explainer = GraphExplainer(model=gcn_model)
        with pytest.raises(ValueError, match="max_subset_size"):
            explainer.explain(simple_graph, max_subset_size=bad_size)

    @pytest.mark.parametrize("bad_order", [0, -1])
    def test_non_positive_order_rejected_by_graphshapiq(self, gcn_graphshapiq, bad_order):
        with pytest.raises(ValueError, match="order"):
            gcn_graphshapiq.explain(order=bad_order)

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_non_positive_max_subset_size_rejected_by_graphshapiq(self, gcn_graphshapiq, bad_size):
        """The same validation must exist on GraphSHAPIQ.explain directly,
        since it is a public entry point independent of the explainer."""
        with pytest.raises(ValueError, match="max_subset_size"):
            gcn_graphshapiq.explain(max_subset_size=bad_size)


class TestUnknownKwargs:
    @requires_cext
    def test_typo_kwarg_at_explain_is_rejected(self, gcn_model, simple_graph):
        """Contract: a misspelled option must raise a warning so the user learns
        their override never took effect. This is the load-bearing safeguard now
        that intentionally unsupported options are silently absent from the"""
        explainer = GraphExplainer(model=gcn_model)
        with pytest.warns():
            explainer.explain(simple_graph, efficency_routine=False)  # deliberate typo
