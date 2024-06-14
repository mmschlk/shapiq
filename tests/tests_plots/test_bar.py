"""This module contains all tests for the bar plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import bar_plot


def test_bar_plot(interaction_values_list: list[InteractionValues]):
    """Test the bar plot function."""
    n_players = interaction_values_list[0].n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    axis = bar_plot(interaction_values_list, show=False, feature_names=feature_names)

    assert axis is not None
    assert isinstance(axis, plt.Axes)
    plt.close()

    axis = bar_plot(interaction_values_list, show=True)
    assert axis is None
    plt.close()
