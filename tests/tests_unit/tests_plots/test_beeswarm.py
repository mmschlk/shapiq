"""This module contains all tests for the beeswarm plot."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import collections

from shapiq.interaction_values import InteractionValues
from shapiq.plot import beeswarm_plot

N_SAMPLES = 10
N_PLAYERS = 5


@pytest.fixture
def mock_interaction_data() -> tuple[list[InteractionValues], np.ndarray, pd.DataFrame, list[str]]:
    """Creates mock data for beeswarm plot tests.

    Returns:
        A tuple containing a list of InteractionValues, numpy feature data, pandas feature
        data, and a list of feature names.
    """
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


def test_beeswarm_plot_basic(mock_interaction_data):
    """Test the basic beeswarm plot function with numpy and pandas data."""
    interaction_values_list, feature_data_np, feature_data_pd, feature_names = mock_interaction_data

    # numpy data + explicit feature names
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_np,
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    # pandas data + explicit feature names
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        feature_names=feature_names,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    # numpy data + default feature names
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_np,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    assert any("F" in label.get_text() for label in ax.get_yticklabels())
    plt.close("all")

    # data with nan values
    data_with_nan = feature_data_pd.copy()
    data_with_nan.iloc[0, 0] = np.nan
    test_alpha = 0.8
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=data_with_nan,
        alpha=test_alpha,
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
            and np.array_equal(colors[0], expected_nan_color)
        ):
            nan_points_found = True
            break
    assert isinstance(ax, plt.Axes)
    assert nan_points_found, (
        "No scatter plot collection found with the specific style for NaN values."
    )
    plt.close("all")

    # data with all nan values
    data_with_nan = feature_data_pd.copy()
    data_with_nan.iloc[:, :] = np.nan
    test_alpha = 0.8
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=data_with_nan,
        alpha=test_alpha,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    # all same value
    data_with_same_value = feature_data_pd.copy()
    data_with_same_value.iloc[:, 0] = 0.5
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=data_with_same_value,
        alpha=test_alpha,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_beeswarm_plot_options(mock_interaction_data):
    """Test the beeswarm plot function with various options."""
    interaction_values_list, _, feature_data_pd, _ = mock_interaction_data

    # max display
    max_disp = 5
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        max_display=max_disp,
        show=False,
    )
    assert isinstance(ax, plt.Axes)

    # count background rectangles
    n_bg_rectangles = len([p for p in ax.patches if p.get_zorder() == -3])
    total_interactions_available = len(interaction_values_list[0].interaction_lookup)
    expected_interactions = min(max_disp, total_interactions_available)
    expected_rectangles = np.ceil(expected_interactions / 2)
    assert n_bg_rectangles == expected_rectangles
    plt.close("all")

    max_disp = None
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        max_display=max_disp,
        show=False,
    )
    assert isinstance(ax, plt.Axes)
    plt.close("all")

    # existing axes
    _, ax_existing = plt.subplots()
    ax_returned = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        ax=ax_existing,
        show=False,
    )
    assert ax_returned is ax_existing
    plt.close("all")

    # no abbreviation
    long_names = [f"a_very_long_feature_name_{i}" for i in range(N_PLAYERS)]
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        feature_names=long_names,
        abbreviate=False,
        show=False,
    )
    assert any(long_names[0] in label.get_text() for label in ax.get_yticklabels())
    plt.close("all")

    # row_height
    plt.figure()
    ax_large_row = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        row_height=0.8,  # larger height
        show=False,
    )
    fig_large_height = ax_large_row.get_figure().get_size_inches()[1]
    plt.close("all")

    plt.figure()
    ax_small_row = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        row_height=0.2,  # smaller height
        show=False,
    )
    fig_small_height = ax_small_row.get_figure().get_size_inches()[1]
    plt.close("all")
    assert fig_large_height > fig_small_height

    # alpha
    test_alpha = 0.7
    ax = beeswarm_plot(
        interaction_values_list=interaction_values_list,
        data=feature_data_pd,
        alpha=test_alpha,
        show=False,
    )
    main_points_found = any(
        isinstance(c, collections.PathCollection) and c.get_alpha() == test_alpha
        for c in ax.collections
    )
    assert main_points_found, "No scatter plot collection found with the specified alpha."
    plt.close("all")


def test_beeswarm_plot_errors(mock_interaction_data):
    """Test that the beeswarm plot raises errors for invalid input."""
    interaction_values_list, feature_data_np, _, feature_names = mock_interaction_data

    # wrong lengths
    with pytest.raises(ValueError, match="Length of shap_interaction_values must match"):
        beeswarm_plot(interaction_values_list, feature_data_np[1:], show=False)

    # empty interactions
    with pytest.raises(ValueError, match="must be a non-empty list"):
        beeswarm_plot([], feature_data_np, show=False)

    # wrong data type
    with pytest.raises(TypeError, match="must be a pandas DataFrame or a numpy array"):
        beeswarm_plot(interaction_values_list, "not_a_valid_data_type", show=False)

    # wrong feature names length
    with pytest.raises(ValueError, match="Length of feature_names must match n_players"):
        beeswarm_plot(
            interaction_values_list,
            feature_data_np,
            feature_names=feature_names[:-1],
            show=False,
        )

    # invalid row height
    with pytest.raises(ValueError, match="row_height must be a positive value"):
        beeswarm_plot(interaction_values_list, feature_data_np, row_height=-1, show=False)

    # invalid alpha
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        beeswarm_plot(interaction_values_list, feature_data_np, alpha=-0.1, show=False)
