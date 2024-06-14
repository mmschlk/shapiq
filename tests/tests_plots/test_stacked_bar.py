"""This module contains all tests for the stacked bar plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.plot import stacked_bar_plot


def test_stacked_bar_plot():
    """Tests whether the stacked bar plot can be created."""

    n_shapley_values_pos = {
        1: np.asarray([1, 0, 1.75]),
        2: np.asarray([0.25, 0.5, 0.75]),
        3: np.asarray([0.5, 0.25, 0.25]),
    }
    n_shapley_values_neg = {
        1: np.asarray([0, -1.5, 0]),
        2: np.asarray([-0.25, -0.5, -0.75]),
        3: np.asarray([-0.5, -0.25, -0.25]),
    }
    feature_names = ["a", "b", "c"]
    fig, axes = stacked_bar_plot(
        feature_names=feature_names,
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()

    fig, axes = stacked_bar_plot(
        feature_names=feature_names,
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        n_sii_max_order=2,
        title="Title",
        xlabel="X",
        ylabel="Y",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()

    fig, axes = stacked_bar_plot(
        feature_names=None,
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        n_sii_max_order=2,
        title="Title",
        xlabel="X",
        ylabel="Y",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, plt.Axes)
    plt.close()
