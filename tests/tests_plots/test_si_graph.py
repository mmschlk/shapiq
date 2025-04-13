"""This test module contains all tests for the explanation plot functions."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from PIL import Image

from shapiq.plot import si_graph_plot


@pytest.mark.parametrize("draw_threshold", [0.0, 0.5])
@pytest.mark.parametrize("compactness", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("n_interactions", [3, None])
def test_si_graph_plot(
    draw_threshold,
    compactness,
    n_interactions,
    example_values,
):
    """Tests the explanation_graph_plot function."""

    graph_tuple = [(1, 2), (2, 3), (3, 4), (2, 4), (1, 4)]

    # test without graph and from interaction values
    fig, ax = example_values.plot_si_graph(show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # test with show=True
    output = example_values.plot_si_graph(show=True)
    assert output is None
    plt.close("all")

    fig, ax = si_graph_plot(
        example_values,
        graph=graph_tuple,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        compactness=compactness,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    fig, ax = si_graph_plot(
        example_values,
        graph=graph_tuple,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=False,
        n_interactions=n_interactions,
        compactness=compactness,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    edges = nx.Graph()
    edges.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (1, 4)])
    pos = nx.spring_layout(edges, seed=1)

    fig, ax = si_graph_plot(
        example_values,
        graph=edges,
        random_seed=1,
        size_factor=0.7,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        compactness=compactness,
        feature_names=["A", "B", "C", "D"],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # different parameters
    fig, ax = si_graph_plot(
        example_values,
        graph=edges,
        pos=pos,
        random_seed=1,
        draw_threshold=draw_threshold,
        plot_explanation=True,
        n_interactions=n_interactions,
        adjust_node_pos=True,
        interaction_direction="positive",
        min_max_interactions=(-0.5, 0.5),
        show=False,
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
        adjust_node_pos=True,
        interaction_direction="negative",
        min_max_interactions=(-0.5, 0.5),
        feature_names=["A", "B", "C", "D"],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_feature_imgs(example_values):
    random_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(random_img)

    n = 7
    img_list = [img for _ in range(n)]
    fig, ax = example_values.plot_si_graph(
        feature_image_patches=img_list,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    img_dict = {index: img_list[index] for index in range(n)}
    graph = [(i, i + 1) for i in range(n - 1)]
    graph.append((n - 1, 0))
    fig, ax = example_values.plot_si_graph(
        feature_image_patches=img_dict,
        show=False,
        # graph=graph,
        # plot_original_nodes = True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_feature_names(example_values):
    feature_names_list = ["A", "B", "C", "D"]
    feature_names = {index + 1: feature_names_list[index] for index in range(4)}
    fig, ax = example_values.plot_si_graph(feature_names=feature_names, show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_legend(example_values):
    from shapiq.plot.si_graph import get_legend
    fig, ax = example_values.plot_si_graph(
        show=False,
    )
    get_legend(ax)
    plt.show()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_param_changes(example_values):
    for random_seed in [1, 42, 103, 98099]:
        example_values.plot_si_graph(
            show=True,
            random_seed=random_seed,
        )
