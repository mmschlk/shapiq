"""This module contains all tests for the waterfall plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shapiq import ExactComputer, InteractionValues, waterfall_plot


def test_waterfall_cooking_game(cooking_game):
    """Test the waterfall plot function with concrete values from the cooking game."""
    exact_computer = ExactComputer(n_players=cooking_game.n_players, game=cooking_game)
    interaction_values = exact_computer(index="k-SII", order=2)
    print(interaction_values.dict_values)
    waterfall_plot(interaction_values, show=False)
    plt.close("all")

    # visual inspection:
    # - E[f(X)] = 10
    # - f(x) = 15
    # - 0, 1, and 2 should individually have negative contributions (go left)
    # - all interactions should have a positive +7 contribution (go right)


def test_waterfall_plot(interaction_values_list: list[InteractionValues]):
    """Test the waterfall plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    wp = waterfall_plot(iv, show=False)
    assert wp is not None
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = waterfall_plot(iv, show=False, feature_names=feature_names)
    assert isinstance(wp, plt.Axes)
    plt.close()

    iv = iv.get_n_order(1)
    wp = waterfall_plot(iv, show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = iv.plot_waterfall(show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()
