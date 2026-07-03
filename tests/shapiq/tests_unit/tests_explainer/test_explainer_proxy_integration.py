"""Integration tests for the ProxySHAP and ProxySPEX approximators via the explainers.

These tests verify that the proxy-based approximators (:class:`~shapiq.approximator.ProxySHAP`
and :class:`~shapiq.approximator.ProxySPEX`) are correctly wired into the
:class:`~shapiq.explainer.tabular.TabularExplainer` and
:class:`~shapiq.explainer.tabpfn.TabPFNExplainer` interfaces, both via the string shortcuts
(``"proxyshap"`` / ``"proxyspex"``) and via explicitly constructed approximator instances.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from shapiq import InteractionValues, TabPFNExplainer, TabularExplainer
from shapiq.approximator import ProxySHAP
from shapiq.explainer.configuration import setup_approximator
from shapiq.game_theory.exact import ExactComputer
from tests.shapiq.fixtures.data import BUDGET_NR_FEATURES, BUDGET_NR_FEATURES_SMALL, NR_FEATURES
from tests.shapiq.markers import skip_if_no_lightgbm, skip_if_no_tabpfn

# Indices that are linear in the game (cardinal-probabilistic). For these, the ProxySHAP
# decomposition "exact tree readout + exact residual adjustment" is exact at full budget.
LINEAR_INDEX_ORDER = [("SV", 1), ("k-SII", 2), ("SII", 2)]
# Faithful (least-squares) indices only approximate the value function; checked structurally.
FAITHFUL_INDEX_ORDER = [("FSII", 2), ("FBII", 2)]


# ---------------------------------------------------------------------------------------------
# setup_approximator wiring
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", ["SV", "k-SII", "FSII", "FBII", "SII"])
def test_setup_approximator_proxyshap_string(index):
    """The ``"proxyshap"`` string resolves to a configured ProxySHAP instance."""
    approx = setup_approximator(
        "proxyshap", index=index, max_order=2 if index != "SV" else 1, n_players=NR_FEATURES
    )
    assert approx.__class__.__name__ == "ProxySHAP"
    assert approx.index == index
    assert approx.n == NR_FEATURES


@skip_if_no_lightgbm
@pytest.mark.parametrize("index", ["SV", "k-SII", "FSII", "FBII", "SII"])
def test_setup_approximator_proxyspex_string(index):
    """The ``"proxyspex"`` string resolves to a configured ProxySPEX instance."""
    approx = setup_approximator(
        "proxyspex", index=index, max_order=2 if index != "SV" else 1, n_players=NR_FEATURES
    )
    assert approx.__class__.__name__ == "ProxySPEX"
    assert approx.index == index
    assert approx.n == NR_FEATURES


# ---------------------------------------------------------------------------------------------
# ProxySHAP through the TabularExplainer
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("index", "max_order"), LINEAR_INDEX_ORDER)
def test_proxyshap_tabular_exact_at_full_budget(
    dt_reg_model, background_reg_data, index, max_order
):
    """ProxySHAP matches the exact values at full budget for linear indices.

    For cardinal-probabilistic indices (SV/SII/k-SII) the proxy readout plus the residual
    adjustment is an exact decomposition of the game, so at a full coalition budget the
    explainer must reproduce the :class:`~shapiq.game_theory.exact.ExactComputer` result.
    """
    explainer = TabularExplainer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        approximator="proxyshap",
        index=index,
        max_order=max_order,
        random_state=42,
    )
    x = background_reg_data[0].reshape(1, -1)
    iv = explainer.explain(x, budget=BUDGET_NR_FEATURES)

    assert isinstance(iv, InteractionValues)
    assert iv.index == index
    assert iv.max_order == max_order

    # exact reference on the very same imputer game the explainer used
    exact = ExactComputer(game=explainer.imputer, n_players=NR_FEATURES)
    gt = exact(index=index, order=max_order)
    for interaction in gt.interaction_lookup:
        if interaction == ():
            continue
        assert iv[interaction] == pytest.approx(gt[interaction], abs=1e-4)


@pytest.mark.parametrize(("index", "max_order"), FAITHFUL_INDEX_ORDER)
def test_proxyshap_tabular_faithful_runs(dt_reg_model, background_reg_data, index, max_order):
    """ProxySHAP produces a well-formed explanation for faithful indices via TabularExplainer."""
    explainer = TabularExplainer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        approximator="proxyshap",
        index=index,
        max_order=max_order,
        random_state=42,
    )
    x = background_reg_data[0].reshape(1, -1)
    iv = explainer.explain(x, budget=BUDGET_NR_FEATURES)

    assert isinstance(iv, InteractionValues)
    assert iv.index == index
    assert iv.max_order == max_order
    assert len(iv.values) > 0
    assert np.all(np.isfinite(iv.values))


@pytest.mark.parametrize("adjustment", ["none", "msr", "svarm", "kernel"])
def test_proxyshap_tabular_adjustment_methods(dt_reg_model, background_reg_data, adjustment):
    """All adjustment methods integrate through a custom ProxySHAP instance and TabularExplainer."""
    approximator = ProxySHAP(
        n=NR_FEATURES,
        max_order=2,
        index="k-SII",
        adjustment=adjustment,
        random_state=42,
    )
    explainer = TabularExplainer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        approximator=approximator,
        index="k-SII",
        max_order=2,
        random_state=42,
    )
    x = background_reg_data[0].reshape(1, -1)
    iv = explainer.explain(x, budget=BUDGET_NR_FEATURES)
    assert isinstance(iv, InteractionValues)
    assert iv.index == "k-SII"
    assert np.all(np.isfinite(iv.values))


@pytest.mark.parametrize(("index", "max_order"), [("SV", 1), ("k-SII", 2), ("FSII", 2)])
def test_proxyshap_linear_proxy_tabular(dt_reg_model, background_reg_data, index, max_order):
    """A LinearRegression proxy (linear-in-features path) integrates through TabularExplainer."""
    approximator = ProxySHAP(
        n=NR_FEATURES,
        max_order=max_order,
        index=index,
        proxy_model=LinearRegression(),
        adjustment="none",
        random_state=42,
    )
    explainer = TabularExplainer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        approximator=approximator,
        index=index,
        max_order=max_order,
        random_state=42,
    )
    x = background_reg_data[0].reshape(1, -1)
    iv = explainer.explain(x, budget=BUDGET_NR_FEATURES)

    assert isinstance(iv, InteractionValues)
    # the linear path must honour the requested index (regression for k-SII previously
    # leaked the intermediate SII index)
    assert iv.index == index
    assert iv.max_order == max_order
    assert np.all(np.isfinite(iv.values))


# ---------------------------------------------------------------------------------------------
# ProxySPEX through the TabularExplainer
# ---------------------------------------------------------------------------------------------


@skip_if_no_lightgbm
@pytest.mark.parametrize(("index", "max_order"), [("SV", 2), ("k-SII", 2), ("FSII", 2)])
def test_proxyspex_tabular_runs(dt_reg_model, background_reg_data, index, max_order):
    """ProxySPEX produces a well-formed explanation through the TabularExplainer string API."""
    explainer = TabularExplainer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        approximator="proxyspex",
        index=index,
        max_order=max_order,
        random_state=42,
    )
    x = background_reg_data[0].reshape(1, -1)
    iv = explainer.explain(x, budget=BUDGET_NR_FEATURES)

    assert isinstance(iv, InteractionValues)
    assert iv.index == index
    assert len(iv.values) > 0
    assert np.all(np.isfinite(iv.values))


# ---------------------------------------------------------------------------------------------
# Proxy approximators through the TabPFNExplainer
# ---------------------------------------------------------------------------------------------


@skip_if_no_tabpfn
@pytest.mark.external_libraries
class TestProxyTabPFNExplainer:
    """Integration of the proxy approximators with the dedicated TabPFNExplainer."""

    def test_proxyshap_tabpfn_reg(self, tabpfn_regression_problem):
        """ProxySHAP integrates with the TabPFNExplainer for regression."""
        model, data, labels, x_test = tabpfn_regression_problem
        explainer = TabPFNExplainer(
            model=model,
            data=data,
            labels=labels,
            x_test=x_test,
            approximator="proxyshap",
            index="k-SII",
            max_order=2,
        )
        iv = explainer.explain(x=x_test[0], budget=BUDGET_NR_FEATURES_SMALL)
        assert isinstance(iv, InteractionValues)
        assert iv.index == "k-SII"
        assert np.all(np.isfinite(iv.values))

    def test_proxyshap_tabpfn_clf(self, tabpfn_classification_problem):
        """ProxySHAP integrates with the TabPFNExplainer for classification."""
        model, data, labels, x_test = tabpfn_classification_problem
        explainer = TabPFNExplainer(
            model=model,
            data=data,
            labels=labels,
            x_test=x_test,
            approximator="proxyshap",
            index="SV",
            max_order=1,
        )
        iv = explainer.explain(x=x_test[0], budget=BUDGET_NR_FEATURES_SMALL)
        assert isinstance(iv, InteractionValues)
        assert iv.index == "SV"
        assert np.all(np.isfinite(iv.values))

    @skip_if_no_lightgbm
    def test_proxyspex_tabpfn_reg(self, tabpfn_regression_problem):
        """ProxySPEX integrates with the TabPFNExplainer for regression."""
        model, data, labels, x_test = tabpfn_regression_problem
        explainer = TabPFNExplainer(
            model=model,
            data=data,
            labels=labels,
            x_test=x_test,
            approximator="proxyspex",
            index="k-SII",
            max_order=2,
        )
        iv = explainer.explain(x=x_test[0], budget=BUDGET_NR_FEATURES_SMALL)
        assert isinstance(iv, InteractionValues)
        assert iv.index == "k-SII"
        assert np.all(np.isfinite(iv.values))
