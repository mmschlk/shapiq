"""This module contains the network plots for the shapiq package."""
import copy
import math
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from utils import powerset
from ._config import RED, BLUE, NEUTRAL


__all__ = [
    "network_plot",
]


def _get_color(value: float) -> str:
    """Returns blue color for negative values and red color for positive values."""
    if value >= 0:
        return RED.hex
    return BLUE.hex


def _min_max_normalization(value: float, min_value: float, max_value: float) -> float:
    """Normalizes the value between min and max"""
    size = (value - min_value) / (max_value - min_value)
    return size


def _add_weight_to_edges_in_graph(
    graph: nx.Graph,
    first_order_values: np.ndarray,
    second_order_values: np.ndarray,
    n_features: int,
    feature_names: list[str],
) -> None:
    """Adds the weights to the edges in the graph."""

    # get min and max value for n_shapley_values
    min_node_value, max_node_value = np.min(first_order_values), np.max(first_order_values)
    min_edge_value, max_edge_value = np.min(second_order_values), np.max(second_order_values)

    all_range = abs(max(max_node_value, max_edge_value) - min(min_node_value, min_edge_value))

    size_scaler = 30

    for node in graph.nodes:
        weight: float = first_order_values[node]
        size = abs(weight) / all_range
        color = _get_color(weight)
        graph.nodes[node]["node_color"] = color
        graph.nodes[node]["node_size"] = size * 250
        graph.nodes[node]["label"] = feature_names[node]
        graph.nodes[node]["linewidths"] = 1
        graph.nodes[node]["edgecolors"] = color

    for edge in powerset(range(n_features), min_size=2, max_size=2):
        weight: float = second_order_values[edge]
        color = _get_color(weight)
        # scale weight between min and max edge value
        size = abs(weight) / all_range
        graph_edge = graph.get_edge_data(*edge)
        graph_edge["width"] = size * (size_scaler + 1)
        graph_edge["color"] = color


def _add_legend_to_axis(axis: plt.Axes) -> None:
    """Adds a legend for order 1 (nodes) and order 2 (edges) interactions to the axis."""
    sizes = [1.0, 0.2, 0.2, 1]
    labels = ["high pos.", "low pos.", "low neg.", "high neg."]
    alphas_line = [0.5, 0.2, 0.2, 0.5]

    # order 1 (circles)
    plot_circles = []
    for i in range(4):
        size = sizes[i]
        if i < 2:
            color = RED.hex
        else:
            color = BLUE.hex
        circle = axis.plot([], [], c=color, marker="o", markersize=size * 8, linestyle="None")
        plot_circles.append(circle[0])

    legend1 = plt.legend(
        plot_circles,
        labels,
        frameon=True,
        framealpha=0.5,
        facecolor="white",
        title=r"$\bf{Order\ 1}$",
        fontsize=7,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        bbox_to_anchor=(1.12, 1.1),
        title_fontsize=7,
        loc="upper right",
    )

    # order 2 (lines)
    plot_lines = []
    for i in range(4):
        size = sizes[i]
        alpha = alphas_line[i]
        if i < 2:
            color = RED.hex
        else:
            color = BLUE.hex
        line = axis.plot([], [], c=color, linewidth=size * 3, alpha=alpha)
        plot_lines.append(line[0])

    legend2 = plt.legend(
        plot_lines,
        labels,
        frameon=True,
        framealpha=0.5,
        facecolor="white",
        title=r"$\bf{Order\ 2}$",
        fontsize=7,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        bbox_to_anchor=(1.12, 0.92),
        title_fontsize=7,
        loc="upper right",
    )

    axis.add_artist(legend1)
    axis.add_artist(legend2)


def network_plot(
    first_order_values: np.ndarray[float],
    second_order_values: np.ndarray[float],
    *,
    feature_names: Optional[list[Any]] = None,
    feature_image_patches: Optional[dict[int, Image.Image]] = None,
    feature_image_patches_size: Optional[Union[float, dict[int, float]]] = 0.2,
    center_image: Optional[Image.Image] = None,
    center_image_size: Optional[float] = 0.6,
) -> tuple[plt.Figure, plt.Axes]:
    """Draws the interaction network.

    An interaction network is a graph where the nodes represent the features and the edges represent
    the interactions. The edge width is proportional to the interaction value. The color of the edge
    is red if the interaction value is positive and blue if the interaction value is negative. The
    interaction values should be derived from the n-Shapley interaction index (n-SII). Below is an
    example of an interaction network with an image in the center.

    .. image:: /_static/network_example.png
        :width: 400
        :align: center

    Args:
        first_order_values: The first order n-SII values of shape (n_features,).
        second_order_values: The second order n-SII values of shape (n_features, n_features). The
            diagonal values are ignored. Only the upper triangular values are used.
        feature_names: The feature names used for plotting. If no feature names are provided, the
            feature indices are used instead. Defaults to None.
        feature_image_patches: A dictionary containing the image patches to be displayed instead of
            the feature labels in the network. The keys are the feature indices and the values are
            the feature images. Defaults to None.
        feature_image_patches_size: The size of the feature image patches. If a dictionary is
            provided, the keys are the feature indices and the values are the feature image patch.
            Defaults to 0.2.
        center_image: The image to be displayed in the center of the network. Defaults to None.
        center_image_size: The size of the center image. Defaults to 0.6.

    Returns:
        The figure and the axis containing the plot.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.axis("off")

    # get the number of features and the feature names
    n_features = first_order_values.shape[0]
    if feature_names is None:
        feature_names = [str(i + 1) for i in range(n_features)]

    # create a fully connected graph up to the n_sii_order
    graph = nx.complete_graph(n_features)

    # add the weights to the edges
    _add_weight_to_edges_in_graph(
        graph=graph,
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        n_features=n_features,
        feature_names=feature_names,
    )

    # get node and edge attributes
    node_colors = nx.get_node_attributes(graph, "node_color").values()
    node_sizes = list(nx.get_node_attributes(graph, "node_size").values())
    node_labels = nx.get_node_attributes(graph, "label")
    node_line_widths = list(nx.get_node_attributes(graph, "linewidths").values())
    node_edge_colors = list(nx.get_node_attributes(graph, "edgecolors").values())
    edge_colors = nx.get_edge_attributes(graph, "color").values()
    edge_widths = list(nx.get_edge_attributes(graph, "width").values())

    # turn edge widths into a list of alpha hues floats from 0.25 to 0.9 depending on the max value
    max_width = max(edge_widths)
    edge_alphas = [max(0, 0 + (width / max_width) * 0.65) for width in edge_widths]

    # arrange the nodes in a circle
    pos = nx.circular_layout(graph)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, alpha=edge_alphas)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=node_line_widths,
        edgecolors=node_edge_colors,
    )

    # add the labels or image patches to the nodes
    for node, (x, y) in pos.items():
        size = graph.nodes[node]["linewidths"]
        label = node_labels[node]
        radius = 1.15 + size / 300
        theta = np.arctan2(x, y)
        if abs(theta) <= 0.001:
            label = "\n" + label
        theta = np.pi / 2 - theta
        if theta < 0:
            theta += 2 * np.pi
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        if feature_image_patches is None:
            axis.text(x, y, label, horizontalalignment="center", verticalalignment="center")
        else:  # draw the image instead of the text
            image = feature_image_patches[node]
            patch_size = feature_image_patches_size
            if isinstance(patch_size, dict):
                patch_size = patch_size[node]
            extend = patch_size / 2
            axis.imshow(image, extent=(x - extend, x + extend, y - extend, y + extend))

    # add the image to the nodes if provided
    if center_image is not None:
        _add_center_image(axis, center_image, center_image_size, n_features)

    # add the legends to the plot
    _add_legend_to_axis(axis)

    return fig, axis


def _add_center_image(
    axis: plt.Axes, center_image: Image.Image, center_image_size: float, n_features: int
):
    """Adds the center image to the axis."""
    # plot the center image
    image_to_plot = Image.fromarray(np.asarray(copy.deepcopy(center_image)))
    extend = center_image_size
    axis.imshow(image_to_plot, extent=(-extend, extend, -extend, extend))

    # add grids with vlines and hlines to the image
    x = np.linspace(-extend, extend, int(math.sqrt(n_features) + 1))
    y = np.linspace(-extend, extend, int(math.sqrt(n_features) + 1))
    axis.vlines(x=x, ymin=-extend, ymax=extend, colors="white", linewidths=2, linestyles="solid")
    axis.hlines(y=y, xmin=-extend, xmax=extend, colors="white", linewidths=2, linestyles="solid")

    # move image to the foreground and edges to the background
    axis.set_zorder(1)
    for edge in axis.collections:
        edge.set_zorder(0)
