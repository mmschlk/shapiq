"""This test module contains all tests for the explanation plot functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import si_graph_plot


@pytest.mark.parametrize("draw_threshold", [0.0, 0.5])
@pytest.mark.parametrize("compactness", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("n_interactions", [3, None])
def test_si_graph_plot(
    draw_threshold,
    compactness,
    n_interactions,
):
    """Tests the explanation_graph_plot function."""
    example_values = InteractionValues(
        n_players=4,
        values=np.array(
            [
                0.0,  # ()
                -0.2,  # (1)
                0.2,  # (2)
                0.2,  # (3)
                -0.1,  # (4)
                0.2,  # (1, 2)
                -0.2,  # (1, 3)
                0.2,  # (1, 4)
                0.2,  # (2, 3)
                -0.2,  # (2, 4)
                0.2,  # (3, 4)
                0.4,  # (1, 2, 3)
                0.0,  # (1, 2, 4)
                0.0,  # (1, 3, 4)
                0.0,  # (2, 3, 4)
                -0.1,  # (1, 2, 3, 4)
            ],
            dtype=float,
        ),
        index="k-SII",
        interaction_lookup={
            (): 0,
            (1,): 1,
            (2,): 2,
            (3,): 3,
            (4,): 4,
            (1, 2): 5,
            (1, 3): 6,
            (1, 4): 7,
            (2, 3): 8,
            (2, 4): 9,
            (3, 4): 10,
            (1, 2, 3): 11,
            (1, 2, 4): 12,
            (1, 3, 4): 13,
            (2, 3, 4): 14,
            (1, 2, 3, 4): 15,
        },
        baseline_value=0,
        min_order=0,
        max_order=4,
    )
    graph_tuple = [(1, 2), (2, 3), (3, 4), (2, 4), (1, 4)]

    # test without graph and from interaction values
    fig, ax = example_values.plot_si_graph(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    fig, ax = si_graph_plot(
        example_values,
        graph_tuple,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        compactness=compactness,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    fig, ax = si_graph_plot(
        example_values,
        graph_tuple,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=False,
        n_interactions=n_interactions,
        compactness=compactness,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    edges = nx.Graph()
    edges.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (1, 4)])
    pos = nx.spring_layout(edges, seed=1)

    fig, ax = si_graph_plot(
        example_values,
        edges,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        compactness=compactness,
        feature_names=["A", "B", "C", "D"],
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # different parameters
    fig, ax = si_graph_plot(
        example_values,
        edges,
        pos=pos,
        random_seed=1,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        node_area_scaling=True,
        adjust_node_pos=True,
        interaction_direction="positive",
        min_max_interactions=(-0.5, 0.5),
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    fig, ax = si_graph_plot(
        example_values,
        graph=graph_tuple,
        pos=pos,
        random_seed=1,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        node_area_scaling=True,
        adjust_node_pos=True,
        interaction_direction="negative",
        min_max_interactions=(-0.5, 0.5),
        feature_names=["A", "B", "C", "D"],
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
