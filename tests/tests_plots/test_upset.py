"""This module contains all tests for the upset plot."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import upset_plot


def test_upset_plot():
    """Test the force plot function."""
    lookup = {
        (0,): 0,
        (1,): 1,
        (2,): 2,
        (0, 1): 3,
        (0, 2): 4,
        (0, 1, 2): 5,
        (1, 4): 6,
        (2, 3): 7,
        (0, 1, 3): 8,
        (0, 2, 3): 9,
        (0, 1, 2, 3): 10,
        (0, 1, 2, 4): 11,
        (0, 1, 2, 3, 4): 12,
    }
    iv = InteractionValues(
        values=np.array([1, 2, 1.5, -0.9, 0.1, 0.3, -0.2, 0.1, 0.11, -0.1, 0.2, 0.8, 0.05]),
        interaction_lookup=lookup,
        index="k-SII",
        min_order=1,
        max_order=5,
        baseline_value=0.0,
        n_players=5,
    )
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    fig = upset_plot(iv, feature_names=feature_names, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    fig = upset_plot(iv, feature_names=feature_names, color_matrix=True, show=True)
    assert fig is None
    plt.close("all")

    # in the following feature 3 is not shown
    fig = upset_plot(iv, n_interactions=5, all_features=False, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # in the following feature 3 is shown
    fig = upset_plot(iv, n_interactions=5, all_features=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # test once directly from the interaction values
    fig = iv.plot_upset(feature_names=feature_names, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")
