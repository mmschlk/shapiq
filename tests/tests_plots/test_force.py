"""This module contains all tests for the force plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import force_plot


def test_force_plot(interaction_values_list: list[InteractionValues]):
    """Test the force plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)
    feature_values = np.array([i for i in range(n_players)])

    fp = force_plot(iv, show=False)
    assert fp is not None
    assert isinstance(fp, plt.Figure)
    plt.close()

    fp = force_plot(iv, show=False, feature_names=feature_names, feature_values=feature_values)
    assert isinstance(fp, plt.Figure)
    plt.close()

    fp = force_plot(iv, show=False, feature_names=None, feature_values=feature_values)
    assert isinstance(fp, plt.Figure)
    plt.close()

    iv = iv.get_n_order(1)
    fp = force_plot(iv, show=False)
    assert isinstance(fp, plt.Figure)
    plt.close()

    fp = iv.plot_force(show=False)
    assert isinstance(fp, plt.Figure)
    plt.close()

    # test show=True
    output = iv.plot_force(show=True)
    assert output is None
    plt.close("all")
