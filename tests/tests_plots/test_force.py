"""This module contains all tests for the force plots."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq import ExactComputer, InteractionValues, force_plot


def test_force_cooking_game(cooking_game):
    """Test the force plot function with concrete values from the cooking game."""
    exact_computer = ExactComputer(n_players=cooking_game.n_players, game=cooking_game)
    interaction_values = exact_computer(index="k-SII", order=2)
    print(interaction_values.dict_values)
    feature_names = list(cooking_game.player_name_lookup.keys())
    force_plot(interaction_values, show=True, min_percentage=0.2, feature_names=feature_names)
    plt.close()

    # visual inspection:
    # - E[f(X)] = 10
    # - f(x) = 15
    # - 0, 1, and 2 should individually have negative contributions (go left)
    # - all interactions should have a positive +7 contribution (go right)
    # - feature 0 is too small to be displayed because of min_percentage=0.2


def test_force_plot(interaction_values_list: list[InteractionValues]):
    """Test the force plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    fp = force_plot(iv, show=False)
    assert fp is not None
    assert isinstance(fp, plt.Figure)
    plt.close()

    fp = force_plot(iv, show=False, abbreviate=False)
    assert fp is not None
    assert isinstance(fp, plt.Figure)
    plt.close()

    fp = force_plot(iv, show=False, feature_names=feature_names)
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
