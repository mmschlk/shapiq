"""Smoke tests for plotting functions — verify they run without error."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues
from shapiq.plot import (
    abbreviate_feature_names,
    bar_plot,
    beeswarm_plot,
    force_plot,
    network_plot,
    sentence_plot,
    si_graph_plot,
    stacked_bar_plot,
    upset_plot,
    waterfall_plot,
)
from shapiq.utils import powerset


def _build_iv(n: int = 4, seed: int = 0) -> InteractionValues:
    """Build a small InteractionValues with deterministic non-zero values."""
    rng = np.random.default_rng(seed)
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(powerset(range(n), min_size=1, max_size=2)):
        interaction_lookup[interaction] = i
        values.append(float(rng.normal()))
    return InteractionValues(
        values=np.array(values),
        index="k-SII",
        n_players=n,
        min_order=1,
        max_order=2,
        interaction_lookup=interaction_lookup,
        baseline_value=0.5,
    )


@pytest.fixture
def sample_iv() -> InteractionValues:
    """Small InteractionValues (n_players=4, order 1-2)."""
    return _build_iv()


@pytest.fixture
def sample_iv_list() -> list[InteractionValues]:
    """Small list of InteractionValues for beeswarm/stacked tests."""
    return [_build_iv(seed=i) for i in range(3)]


@pytest.fixture(autouse=True)
def _close_plots():
    """Ensure no figures leak between tests."""
    yield
    plt.close("all")


class TestPlots:
    """Smoke tests: plots run without raising."""

    def test_bar_plot(self, sample_iv):
        bar_plot([sample_iv.get_n_order(order=1)])

    def test_waterfall_plot(self, sample_iv):
        waterfall_plot(sample_iv.get_n_order(order=1))

    def test_force_plot(self, sample_iv):
        force_plot(sample_iv.get_n_order(order=1))

    def test_network_plot(self, sample_iv):
        network_plot(sample_iv)

    def test_stacked_bar_plot(self, sample_iv):
        stacked_bar_plot(sample_iv)

    def test_upset_plot(self, sample_iv):
        upset_plot(sample_iv)

    def test_si_graph_plot(self, sample_iv):
        si_graph_plot(sample_iv)

    def test_sentence_plot(self):
        """sentence_plot requires a 1st-order IV with n_players matching the word list."""
        iv = _build_iv(n=3)
        sentence_plot(iv.get_n_order(order=1), words=["alpha", "beta", "gamma"])

    def test_beeswarm_plot(self, sample_iv_list):
        data = np.random.default_rng(0).normal(size=(len(sample_iv_list), 4))
        beeswarm_plot(sample_iv_list, data=data)


class TestPlotUtils:
    """Tests for plot-level helpers."""

    def test_abbreviate_feature_names_basic(self):
        abbreviated = abbreviate_feature_names(["Alpha Feature", "Beta Value", "Gamma"])
        assert len(abbreviated) == 3
        assert all(isinstance(s, str) for s in abbreviated)

    def test_abbreviate_feature_names_empty(self):
        assert abbreviate_feature_names([]) == []
