"""This module contains all tests for the stacked bar plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import stacked_bar_plot


def test_stacked_bar_plot():
    """Tests whether the stacked bar plot can be created."""

    interaction_values = InteractionValues(
        values=np.array([1, -1.5, 1.75, 0.25, -0.5, 0.75, 0.2]),
        index="SII",
        min_order=1,
        max_order=3,
        n_players=3,
        baseline_value=0,
    )
    feature_names = ["a", "b", "c"]
    fig, axes = stacked_bar_plot(
        interaction_values=interaction_values,
        feature_names=feature_names,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()

    fig, axes = stacked_bar_plot(
        interaction_values=interaction_values,
        feature_names=feature_names,
        max_order=2,
        title="Title",
        xlabel="X",
        ylabel="Y",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()

    fig, axes = stacked_bar_plot(
        interaction_values=interaction_values,
        feature_names=None,
        max_order=2,
        title="Title",
        xlabel="X",
        ylabel="Y",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()
