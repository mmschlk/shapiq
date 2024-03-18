"""This module contains the network plots for the shapiq package."""

import copy
import math
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from interaction_values import InteractionValues
from matplotlib import pyplot as plt
from PIL import Image
from utils import powerset

from ._config import BLUE, LINES, NEUTRAL, RED

__all__ = [
    "network_plot",
]


def network_plot(
    interaction_values: Optional[InteractionValues] = None,
    *,
    first_order_values: Optional[np.ndarray[float]] = None,
    second_order_values: Optional[np.ndarray[float]] = None,
    feature_names: Optional[list[Any]] = None,
    feature_image_patches: Optional[dict[int, Image.Image]] = None,
    feature_image_patches_size: Optional[Union[float, dict[int, float]]] = 0.2,
    center_image: Optional[Image.Image] = None,
    center_image_size: Optional[float] = 0.6,
    draw_legend: bool = True,
    center_text: Optional[str] = None,
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
        interaction_values: The interaction values as an interaction object.
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
        draw_legend: Whether to draw the legend. Defaults to True.
        center_text: The text to be displayed in the center of the network. Defaults to None.

    Returns:
        The figure and the axis containing the plot.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.axis("off")

    if interaction_values is not None:
        n_players = interaction_values.n_players
        first_order_values = np.zeros(n_players)
        second_order_values = np.zeros((n_players, n_players))
        for interaction in powerset(range(n_players), min_size=1, max_size=2):
            if len(interaction) == 1:
                first_order_values[interaction[0]] = interaction_values[interaction]
            else:
                second_order_values[interaction] = interaction_values[interaction]
    else:
        if first_order_values is None or second_order_values is None:
            raise ValueError(
                "Either interaction_values or first_order_values and second_order_values must be "
                "provided. If interaction_values is provided this will be used."
            )

    # get the number of features and the feature names
    n_features = first_order_values.shape[0]
    if feature_names is None:
        feature_names = [str(i + 1) for i in range(n_features)]

    # create a fully connected graph up to the n_sii_order
    graph = nx.complete_graph(n_features)

    nodes_visit_order = _order_nodes(len(graph.nodes))

    # add the weights to the edges
    _add_weight_to_edges_in_graph(
        graph=graph,
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        n_features=n_features,
        feature_names=feature_names,
        nodes_visit_order=nodes_visit_order,
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
    for i, node in enumerate(nodes_visit_order):
        (x, y) = pos[node]
        size = graph.nodes[node]["linewidths"]
        label = node_labels[i]
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
            image = feature_image_patches[i]
            patch_size = feature_image_patches_size
            if isinstance(patch_size, dict):
                patch_size = patch_size[i]
            extend = patch_size / 2
            axis.imshow(image, extent=(x - extend, x + extend, y - extend, y + extend))

    # add the image to the nodes if provided
    if center_image is not None:
        _add_center_image(axis, center_image, center_image_size, n_features)

    # add the center text if provided
    if center_text is not None:
        background_color = NEUTRAL.hex
        line_color = LINES.hex
        axis.text(
            0,
            0,
            center_text,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor=background_color, alpha=0.5, edgecolor=line_color, pad=7),
            color="black",
            fontsize=plt.rcParams["font.size"] + 3,
        )

    # add the legends to the plot
    if draw_legend:
        _add_legend_to_axis(axis)

    return fig, axis


def _get_color(value: float) -> str:
    """Returns blue color for negative values and red color for positive values.

    Args:
        value (float): The value to determine the color for.

    Returns:
        str: The color as a hex string.
    """
    if value >= 0:
        return RED.hex
    return BLUE.hex


def _add_weight_to_edges_in_graph(
    graph: nx.Graph,
    first_order_values: np.ndarray,
    second_order_values: np.ndarray,
    n_features: int,
    feature_names: list[str],
    nodes_visit_order: list[int],
) -> None:
    """Adds the weights to the edges in the graph.

    Args:
        graph (nx.Graph): The graph to add the weights to.
        first_order_values (np.ndarray): The first order n-SII values.
        second_order_values (np.ndarray): The second order n-SII values.
        n_features (int): The number of features.
        feature_names (list[str]): The names of the features.
        nodes_visit_order (list[int]): The order of the nodes to visit.

    Returns:
        None
    """

    # get min and max value for n_shapley_values
    min_node_value, max_node_value = np.min(first_order_values), np.max(first_order_values)
    min_edge_value, max_edge_value = np.min(second_order_values), np.max(second_order_values)

    all_range = abs(max(max_node_value, max_edge_value) - min(min_node_value, min_edge_value))

    size_scaler = 30

    for i, node_id in enumerate(nodes_visit_order):
        weight: float = first_order_values[i]
        size = abs(weight) / all_range
        color = _get_color(weight)
        graph.nodes[node_id]["node_color"] = color
        graph.nodes[node_id]["node_size"] = size * 250
        graph.nodes[node_id]["label"] = feature_names[node_id]
        graph.nodes[node_id]["linewidths"] = 1
        graph.nodes[node_id]["edgecolors"] = color

    for interaction in powerset(range(n_features), min_size=2, max_size=2):
        weight: float = float(second_order_values[interaction])
        edge = list(sorted(interaction))
        edge[0] = nodes_visit_order.index(interaction[0])
        edge[1] = nodes_visit_order.index(interaction[1])
        edge = tuple(edge)
        color = _get_color(weight)
        # scale weight between min and max edge value
        size = abs(weight) / all_range
        graph_edge = graph.get_edge_data(*edge)
        graph_edge["width"] = size * (size_scaler + 1)
        graph_edge["color"] = color


def _add_legend_to_axis(axis: plt.Axes) -> None:
    """Adds a legend for order 1 (nodes) and order 2 (edges) interactions to the axis.

    Args:
        axis (plt.Axes): The axis to add the legend to.

    Returns:
        None
    """
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

    font_size = plt.rcParams["legend.fontsize"]

    legend1 = plt.legend(
        plot_circles,
        labels,
        frameon=True,
        framealpha=0.5,
        facecolor="white",
        title=r"$\bf{Order\ 1}$",
        fontsize=font_size,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        title_fontsize=font_size,
        loc="best",
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
        fontsize=font_size,
        labelspacing=0.5,
        handletextpad=0.5,
        borderpad=0.5,
        handlelength=1.5,
        title_fontsize=font_size,
        loc="best",
    )

    axis.add_artist(legend1)
    axis.add_artist(legend2)


def _add_center_image(
    axis: plt.Axes, center_image: Image.Image, center_image_size: float, n_features: int
) -> None:
    """Adds the center image to the axis.

    Args:
        axis (plt.Axes): The axis to add the image to.
        center_image (Image.Image): The image to add to the axis.
        center_image_size (float): The size of the center image.
        n_features (int): The number of features.

    Returns:
        None
    """
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


def _get_highest_node_index(n_nodes: int) -> int:
    """Calculates the node with the highest position on the y-axis given the total number of nodes.

    Args:
        n_nodes (int): The total number of nodes.

    Returns:
        int: The index of the highest node.
    """
    n_connections = 0
    # highest node is the last node below 1/4 of all connections in the circle
    while n_connections <= n_nodes / 4:
        n_connections += 1
    n_connections -= 1
    return n_connections


def _order_nodes(n_nodes: int) -> list[int]:
    """Orders the nodes in the network plot.

    Args:
        n_nodes (int): The total number of nodes.

    Returns:
        list[int]: The order of the nodes.
    """
    highest_node = _get_highest_node_index(n_nodes)
    nodes_visit_order = [highest_node]
    desired_order = list(reversed(list(range(n_nodes))))
    highest_node_index = desired_order.index(highest_node)
    nodes_visit_order += desired_order[highest_node_index + 1 :]
    nodes_visit_order += desired_order[:highest_node_index]
    return nodes_visit_order
