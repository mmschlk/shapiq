"""This module contains all tests for the 2-D scatter plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import scatter_2d_plot

N_SAMPLES = 10
N_PLAYERS = 5

LOOKUP = {
    (): 0,
    (0,): 1,
    (1,): 2,
    (2,): 3,
    (3,): 4,
    (4,): 5,
    (0, 1): 6,
    (0, 2): 7,
    (1, 3): 8,
    (2, 4): 9,
}


@pytest.fixture
def mock_interaction_data() -> tuple[list[InteractionValues], np.ndarray, list[str]]:
    """Creates mock data where (1, 3) is the strongest pairwise interaction."""
    rng = np.random.default_rng(42)
    interaction_values_list = []
    for _ in range(N_SAMPLES):
        values = rng.random(len(LOOKUP)) * 0.2 - 0.1
        values[LOOKUP[(1, 3)]] = rng.choice([-1.0, 1.0]) * (1 + rng.random())
        iv = InteractionValues(
            values=values,
            interaction_lookup=LOOKUP,
            index="k-SII",
            min_order=1,
            max_order=2,
            n_players=N_PLAYERS,
            baseline_value=0.0,
        )
        interaction_values_list.append(iv)
    feature_data = rng.random((N_SAMPLES, N_PLAYERS))
    feature_names = [f"feature_{i}" for i in range(N_PLAYERS)]
    return interaction_values_list, feature_data, feature_names


def test_scatter_2d_plot_basic(mock_interaction_data):
    """Tests that the two features are on the axes and the colors are the interaction values."""
    interaction_values_list, feature_data, _ = mock_interaction_data

    ax = scatter_2d_plot(
        interaction_values_list, feature_data, interaction=(2, 0), abbreviate=False, show=False
    )

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "F0"
    assert ax.get_ylabel() == "F2"
    points = ax.collections[0]
    np.testing.assert_allclose(points.get_offsets(), feature_data[:, [0, 2]])
    expected = [iv[(0, 2)] for iv in interaction_values_list]
    np.testing.assert_allclose(points.get_array(), expected)
    # the colormap is centered at zero
    assert points.norm.vmin == pytest.approx(-points.norm.vmax)
    assert points.norm.vmax == pytest.approx(np.max(np.abs(expected)))
    colorbar_ax = ax.figure.axes[-1]
    assert colorbar_ax.get_ylabel() == "k-SII(F0, F2)"
    plt.close("all")


def test_scatter_2d_plot_default_interaction_and_names(mock_interaction_data):
    """Tests the default pair, feature names and pandas input."""
    interaction_values_list, feature_data, feature_names = mock_interaction_data
    data = pd.DataFrame(feature_data, columns=feature_names)

    ax = scatter_2d_plot(
        interaction_values_list,
        data,
        feature_names=feature_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == "feature_1"
    assert ax.get_ylabel() == "feature_3"
    plt.close("all")

    ax = scatter_2d_plot(
        interaction_values_list,
        data,
        interaction=("feature_2", "feature_4"),
        feature_names=feature_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == "feature_2"
    assert ax.get_ylabel() == "feature_4"
    plt.close("all")


def test_scatter_2d_plot_existing_axes_and_show(mock_interaction_data, monkeypatch):
    """Tests plotting on given axes and the return value with show=True."""
    interaction_values_list, feature_data, _ = mock_interaction_data
    monkeypatch.setattr(plt, "show", lambda: None)

    _fig, ax = plt.subplots()
    result = scatter_2d_plot(interaction_values_list, feature_data, (0, 1), abbreviate=False, ax=ax)
    assert result is None
    assert ax.get_xlabel() == "F0"
    plt.close("all")


def test_scatter_2d_plot_errors(mock_interaction_data):
    """Tests the input validation."""
    interaction_values_list, feature_data, _ = mock_interaction_data

    with pytest.raises(ValueError, match="non-empty list"):
        scatter_2d_plot([], feature_data, show=False)
    with pytest.raises(ValueError, match="must match number of rows"):
        scatter_2d_plot(interaction_values_list, feature_data[:-1], show=False)
    with pytest.raises(TypeError, match="must be a pandas DataFrame or a numpy array"):
        scatter_2d_plot(interaction_values_list, feature_data.tolist(), show=False)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        scatter_2d_plot(interaction_values_list, feature_data, alpha=0, show=False)
    with pytest.raises(ValueError, match="dot_size must be a positive value"):
        scatter_2d_plot(interaction_values_list, feature_data, dot_size=0, show=False)
    with pytest.raises(ValueError, match="exactly two different features"):
        scatter_2d_plot(interaction_values_list, feature_data, interaction=(0,), show=False)
    with pytest.raises(ValueError, match="exactly two different features"):
        scatter_2d_plot(interaction_values_list, feature_data, interaction=(1, 1), show=False)
    with pytest.raises(TypeError, match="tuple of two features or None"):
        scatter_2d_plot(interaction_values_list, feature_data, interaction=1, show=False)
    with pytest.raises(ValueError, match="Unknown feature name"):
        scatter_2d_plot(interaction_values_list, feature_data, interaction=("F0", "x"), show=False)
    with pytest.raises(ValueError, match="not found in InteractionValues"):
        scatter_2d_plot(interaction_values_list, feature_data, interaction=(3, 4), show=False)
    with pytest.raises(ValueError, match="Length of feature_names"):
        scatter_2d_plot(interaction_values_list, feature_data, feature_names=["a"], show=False)

    baseline_only = [
        InteractionValues(
            values=np.ones(1),
            interaction_lookup={(): 0},
            index="SV",
            min_order=0,
            max_order=0,
            n_players=N_PLAYERS,
            baseline_value=0.0,
        )
        for _ in range(N_SAMPLES)
    ]
    with pytest.raises(ValueError, match="No pairwise interactions or first-order values"):
        scatter_2d_plot(baseline_only, feature_data, show=False)
    plt.close("all")


def test_scatter_2d_plot_include_main_effects(mock_interaction_data):
    """Tests that both main effects can be added to the pairwise interaction."""
    interaction_values_list, feature_data, _ = mock_interaction_data

    ax = scatter_2d_plot(
        interaction_values_list,
        feature_data,
        interaction=(0, 2),
        include_main_effects=True,
        abbreviate=False,
        show=False,
    )

    points = ax.collections[0]
    expected = [iv[(0,)] + iv[(2,)] + iv[(0, 2)] for iv in interaction_values_list]
    np.testing.assert_allclose(points.get_array(), expected)
    colorbar_ax = ax.figure.axes[-1]
    assert colorbar_ax.get_ylabel() == "k-SII(F0) + k-SII(F2) + k-SII(F0, F2)"
    plt.close("all")


def test_scatter_2d_plot_first_order():
    """Tests that first-order explanations are colored by the sum of the two values."""
    rng = np.random.default_rng(0)
    lookup = {(): 0, (0,): 1, (1,): 2, (2,): 3}
    interaction_values_list = []
    for _ in range(N_SAMPLES):
        values = rng.random(len(lookup)) * 0.2 - 0.1
        values[lookup[(2,)]] = rng.choice([-1.0, 1.0]) * (1 + rng.random())
        values[lookup[(0,)]] = rng.choice([-1.0, 1.0]) * (0.5 + rng.random())
        interaction_values_list.append(
            InteractionValues(
                values=values,
                interaction_lookup=lookup,
                index="SV",
                min_order=1,
                max_order=1,
                n_players=3,
                baseline_value=0.0,
            )
        )
    feature_data = rng.random((N_SAMPLES, 3))

    # the two features with the highest mean absolute value are used by default
    ax = scatter_2d_plot(interaction_values_list, feature_data, abbreviate=False, show=False)

    points = ax.collections[0]
    np.testing.assert_allclose(points.get_offsets(), feature_data[:, [0, 2]])
    expected = [iv[(0,)] + iv[(2,)] for iv in interaction_values_list]
    np.testing.assert_allclose(points.get_array(), expected)
    colorbar_ax = ax.figure.axes[-1]
    assert colorbar_ax.get_ylabel() == "SV(F0) + SV(F2)"
    plt.close("all")
