"""This module contains all tests for the bar plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shapiq import ExactComputer, InteractionValues, bar_plot


def test_bar_cooking_game(cooking_game):
    """Test the bar plot function with concrete values from the cooking game."""
    exact_computer = ExactComputer(game=cooking_game, n_players=cooking_game.n_players)
    sv_exact = exact_computer(index="k-SII", order=2)
    bar_plot([sv_exact], show=False)

    # visual inspection:
    # - Order from top to bottom: Base Value, the interactions (all equal), F0, F1, F2


def test_bar_plot(interaction_values_list: list[InteractionValues]):
    """Test the bar plot function."""
    n_players = interaction_values_list[0].n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    axis = bar_plot(interaction_values_list, show=False, feature_names=feature_names)

    assert axis is not None
    assert isinstance(axis, plt.Axes)
    plt.close()

    axis = bar_plot(interaction_values_list, show=False)
    assert isinstance(axis, plt.Axes)
    plt.close()

    # test max_display=None
    output = bar_plot(interaction_values_list, show=False, max_display=None)
    assert output is not None
    assert isinstance(output, plt.Axes)
    plt.close("all")

    # test global = false
    output = bar_plot(interaction_values_list, show=False, global_plot=False)
    assert output is not None
    assert isinstance(output, plt.Axes)
    plt.close("all")

    # test plot_base_value = True
    output = bar_plot(interaction_values_list, show=False, plot_base_value=True)
    assert output is not None
    assert isinstance(output, plt.Axes)
    plt.close("all")


def _labels(axis: plt.Axes) -> set[str]:
    """Returns the y-tick labels of a bar plot."""
    return {label.get_text() for label in axis.get_yticklabels()}


def test_bar_plot_without_baseline_keeps_first_feature():
    """Tests that explanations without the empty interaction keep the first feature's bar."""
    lookup = {(0,): 0, (1,): 1, (2,): 2, (0, 1): 3}
    interaction_values_list = [
        InteractionValues(
            values=np.array([0.5, -0.2, 0.1, 0.3]) * scale,
            interaction_lookup=lookup,
            index="k-SII",
            min_order=1,
            max_order=2,
            n_players=3,
            baseline_value=1.0,
        )
        for scale in (1.0, 2.0)
    ]

    for global_plot in (True, False):
        axis = bar_plot(
            interaction_values_list, show=False, max_display=None, global_plot=global_plot
        )
        assert _labels(axis) == {"F0", "F1", "F2", "F0 x F1"}
        plt.close("all")


def test_bar_plot_base_value(interaction_values_list: list[InteractionValues]):
    """Tests that the empty interaction is shown only with plot_base_value=True."""
    axis = bar_plot(interaction_values_list, show=False, max_display=None)
    assert "Base Value" not in _labels(axis)
    assert "F0" in _labels(axis)
    plt.close("all")

    axis = bar_plot(interaction_values_list, show=False, max_display=None, plot_base_value=True)
    assert "Base Value" in _labels(axis)
    plt.close("all")
