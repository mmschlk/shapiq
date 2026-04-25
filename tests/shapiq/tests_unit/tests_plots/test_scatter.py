"""This module contains all tests for the scatter plot."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import collections

from shapiq.interaction_values import InteractionValues
from shapiq.plot import scatter_plot

N_SAMPLES = 10
N_PLAYERS = 5


@pytest.fixture
def mock_interaction_data() -> tuple[list[InteractionValues], np.ndarray, pd.DataFrame, list[str]]:
    """Creates mock data for scatter plot tests."""
    lookup = {
        (): 1,
        (0,): 0,
        (1,): 1,
        (2,): 2,
        (3,): 3,
        (4,): 4,
        (0, 1): 5,
        (0, 2): 6,
        (1, 3): 7,
        (2, 4): 8,
        (0, 1, 2): 9,
    }
    n_interactions = len(lookup)

    interaction_values_list = []
    rng = np.random.default_rng(42)
    for _ in range(N_SAMPLES):
        values = rng.random(n_interactions) * 2 - 1
        iv = InteractionValues(
            values=values,
            interaction_lookup=lookup,
            index="k-SII",
            min_order=1,
            max_order=3,
            n_players=N_PLAYERS,
            baseline_value=rng.random(),
        )
        interaction_values_list.append(iv)

    feature_data_np = rng.random((N_SAMPLES, N_PLAYERS))
    feature_names = [f"feature_{i}" for i in range(N_PLAYERS)]
    feature_data_pd = pd.DataFrame(feature_data_np, columns=feature_names)

    return interaction_values_list, feature_data_np, feature_data_pd, feature_names


def test_scatter_plot_basic(mock_interaction_data):
    """Tests basic scatter plot calls with numpy/pandas inputs and shorthand interaction args."""
    interaction_values_list, feature_data_np, feature_data_pd, feature_names = mock_interaction_data

    ax = scatter_plot(
        interaction_values_list,
        feature_data_np,
        interaction=(0,),
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    assert "feature" in ax.get_xlabel().lower() or "f0" in ax.get_xlabel().lower()
    assert ax.get_ylabel() == "SHAP value"
    plt.close("all")

    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    ax = scatter_plot(
        interaction_values_list,
        feature_data_np,
        interaction=(0,),
        show=False,
    )
    assert ax.get_xlabel().startswith("F")
    plt.close("all")

    # Auto-pick interaction
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() != ""
    plt.close("all")

    # Equivalence of int / str / tuple shorthand for main effects
    labels = []
    for arg in (0, "feature_0", (0,), ("feature_0",)):
        ax = scatter_plot(
            interaction_values_list,
            feature_data_pd,
            interaction=arg,
            feature_names=feature_names,
            show=False,
        )
        labels.append(ax.get_xlabel())
        plt.close("all")
    assert len(set(labels)) == 1

    # Higher-order: default x-axis is first feature in tuple
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0, 1),
        feature_names=feature_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == "feature_0"
    assert "feature_0" in ax.get_ylabel() and "feature_1" in ax.get_ylabel()
    plt.close("all")

    # Higher-order with explicit x_feature by index
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0, 1),
        x_feature=1,
        feature_names=feature_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == "feature_1"
    plt.close("all")

    # Higher-order via names + x_feature by name
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=("feature_0", "feature_1"),
        x_feature="feature_1",
        feature_names=feature_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == "feature_1"
    plt.close("all")


def test_scatter_plot_options(mock_interaction_data):
    """Tests scatter_plot color, jitter, ax, abbreviate options."""
    interaction_values_list, _, feature_data_pd, feature_names = mock_interaction_data

    # color = explicit feature -> colorbar added
    fig, ax = plt.subplots()
    n_axes_before = len(fig.axes)
    scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        color="feature_2",
        feature_names=feature_names,
        abbreviate=False,
        ax=ax,
        show=False,
    )
    assert len(fig.axes) > n_axes_before
    cbar_labels = [a.get_ylabel() for a in fig.axes if a is not ax]
    assert "feature_2" in cbar_labels
    plt.close("all")

    # color = None -> no extra colorbar axes added
    fig, ax = plt.subplots()
    n_axes_before = len(fig.axes)
    scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        color=None,
        feature_names=feature_names,
        ax=ax,
        show=False,
    )
    assert len(fig.axes) == n_axes_before
    plt.close("all")

    # NaN in color feature -> gray points at half alpha
    data_with_nan = feature_data_pd.copy()
    data_with_nan.iloc[0, 2] = np.nan
    test_alpha = 0.8
    ax = scatter_plot(
        interaction_values_list,
        data_with_nan,
        interaction=(0,),
        color="feature_2",
        alpha=test_alpha,
        feature_names=feature_names,
        show=False,
    )
    expected_nan_color = list(mcolors.to_rgba("#777777"))
    expected_nan_alpha = test_alpha * 0.5
    expected_nan_color[3] = expected_nan_alpha
    nan_points_found = False
    for collection in ax.collections:
        colors = collection.get_facecolors()
        if (
            isinstance(collection, collections.PathCollection)
            and collection.get_alpha() == expected_nan_alpha
            and len(colors) > 0
            and np.allclose(colors[0], expected_nan_color)
        ):
            nan_points_found = True
            break
    assert nan_points_found
    plt.close("all")

    # All-same color values -> no error (vmin/vmax epsilon)
    data_const_color = feature_data_pd.copy()
    data_const_color.iloc[:, 2] = 0.5
    ax = scatter_plot(
        interaction_values_list,
        data_const_color,
        interaction=(0,),
        color="feature_2",
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    # ax= passed in returns the same axes
    _, ax_existing = plt.subplots()
    ax_returned = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        feature_names=feature_names,
        ax=ax_existing,
        show=False,
    )
    assert ax_returned is ax_existing
    plt.close("all")

    # abbreviate=False keeps long names
    long_names = [f"a_very_long_feature_name_{i}" for i in range(N_PLAYERS)]
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        feature_names=long_names,
        abbreviate=False,
        show=False,
    )
    assert ax.get_xlabel() == long_names[0]
    plt.close("all")

    # jitter changes plotted x values
    ax = scatter_plot(
        interaction_values_list,
        feature_data_pd,
        interaction=(0,),
        feature_names=feature_names,
        jitter=0.5,
        show=False,
    )
    raw = feature_data_pd.iloc[:, 0].to_numpy()
    plotted = ax.collections[0].get_offsets()[:, 0]
    assert np.max(np.abs(plotted - raw)) > 0
    plt.close("all")


def test_scatter_plot_errors(mock_interaction_data):
    """Tests that scatter_plot raises informative errors for bad inputs."""
    interaction_values_list, feature_data_np, feature_data_pd, feature_names = mock_interaction_data

    with pytest.raises(ValueError, match="non-empty list"):
        scatter_plot([], feature_data_np, interaction=(0,), show=False)

    with pytest.raises(ValueError, match="must match number of rows"):
        scatter_plot(
            interaction_values_list,
            feature_data_np[1:],
            interaction=(0,),
            show=False,
        )

    with pytest.raises(TypeError, match="must be a pandas DataFrame or a numpy array"):
        scatter_plot(interaction_values_list, "not_a_valid_data_type", interaction=(0,), show=False)

    with pytest.raises(ValueError, match="Unknown feature name"):
        scatter_plot(
            interaction_values_list,
            feature_data_pd,
            interaction=("nonexistent",),
            feature_names=feature_names,
            show=False,
        )

    with pytest.raises(ValueError, match="out of range"):
        scatter_plot(
            interaction_values_list,
            feature_data_pd,
            interaction=(99,),
            feature_names=feature_names,
            show=False,
        )

    with pytest.raises(ValueError, match="not found in InteractionValues"):
        scatter_plot(
            interaction_values_list,
            feature_data_pd,
            interaction=(3, 4),
            feature_names=feature_names,
            show=False,
        )

    with pytest.raises(ValueError, match="must be a member of interaction"):
        scatter_plot(
            interaction_values_list,
            feature_data_pd,
            interaction=(0, 1),
            x_feature=2,
            feature_names=feature_names,
            show=False,
        )

    with pytest.raises(ValueError, match="Length of feature_names must match"):
        scatter_plot(
            interaction_values_list,
            feature_data_np,
            interaction=(0,),
            feature_names=feature_names[:-1],
            show=False,
        )

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        scatter_plot(
            interaction_values_list,
            feature_data_np,
            interaction=(0,),
            alpha=-0.1,
            show=False,
        )

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        scatter_plot(
            interaction_values_list,
            feature_data_np,
            interaction=(0,),
            alpha=2.0,
            show=False,
        )

    with pytest.raises(ValueError, match="dot_size must be a positive value"):
        scatter_plot(
            interaction_values_list,
            feature_data_np,
            interaction=(0,),
            dot_size=-1,
            show=False,
        )

    with pytest.raises(ValueError, match="jitter must be non-negative"):
        scatter_plot(
            interaction_values_list,
            feature_data_np,
            interaction=(0,),
            jitter=-0.5,
            show=False,
        )
