"""This module contains all tests for the waterfall plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import waterfall_plot


def test_waterfall_plot(interaction_values_list: list[InteractionValues]):
    """Test the waterfall plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)
    feature_values = np.array([i for i in range(n_players)])

    wp = waterfall_plot(iv, show=False)
    assert wp is not None
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = waterfall_plot(iv, show=False, feature_names=feature_names, feature_values=feature_values)
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = waterfall_plot(iv, show=False, feature_names=None, feature_values=feature_values)
    assert isinstance(wp, plt.Axes)
    plt.close()

    iv = iv.get_n_order(1)
    wp = waterfall_plot(iv, show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = iv.plot_waterfall(show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()
