"""Smoke tests for plotting functions — verify they run without error."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues
from shapiq.plot import bar_plot, force_plot, waterfall_plot
from shapiq.utils import powerset


@pytest.fixture
def sample_iv():
    """Small InteractionValues for plotting."""
    n = 4
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(powerset(range(n), min_size=1, max_size=2)):
        interaction_lookup[interaction] = i
        values.append(float(i) * 0.1 - 0.3)
    return InteractionValues(
        values=np.array(values),
        index="k-SII",
        n_players=n,
        min_order=1,
        max_order=2,
        interaction_lookup=interaction_lookup,
        baseline_value=0.5,
    )


class TestPlots:
    """Smoke tests: plots run without raising."""

    def test_bar_plot(self, sample_iv):
        bar_plot([sample_iv.get_n_order(order=1)])
        plt.close("all")

    def test_waterfall_plot(self, sample_iv):
        waterfall_plot(sample_iv.get_n_order(order=1))
        plt.close("all")

    def test_force_plot(self, sample_iv):
        force_plot(sample_iv.get_n_order(order=1))
        plt.close("all")
