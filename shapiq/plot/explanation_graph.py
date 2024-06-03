"""Module for plotting the explanation graph of interaction values."""

import math
from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from ..interaction_values import InteractionValues
from ._config import get_color

NORMAL_NODE_SIZE = 0.125  # 0.125
BASE_ALPHA_VALUE = 1.0  # the transparency level for the highest interaction
BASE_SIZE = 0.05  # the size of the highest interaction edge (with scale factor 1)
ADJUST_NODE_ALPHA = True
SCALE_NODE_EXPLANATIONS_BY_AREA = False


def _normalize_value(
    value: float, max_value: float, base_value: float, cubic_scaling: bool = False
) -> float:
    """Scale a value between 0 and 1 based on the maximum value and a base value.

    Args:
        value: The value to normalize/scale.
        max_value: The maximum value to normalize/scale the value by.
        base_value: The base value to scale the value by. For example, the alpha value for the
            highest interaction (as defined in ``BASE_ALPHA_VALUE``) or the size of the highest
            interaction edge (as defined in ``BASE_SIZE``).
        cubic_scaling: Whether to scale cubically (``True``) or linearly (``False``. default)
            between 0 and 1.

    Returns:
        The normalized/scaled value.
    """
    ratio = abs(value) / abs(max_value)  # ratio is always positive in [0, 1]
    if cubic_scaling:
        ratio = ratio**3
    alpha = ratio * base_value
    return alpha


def _draw_fancy_hyper_edges(
    axis: plt.axis,
    pos: dict,
    graph: nx.Graph,
    hyper_edges: list[tuple],
) -> None:
    """Draws a collection of hyper-edges as a fancy hyper-edge on the graph.

    Note:
        This is also used to draw normal 2-way edges in a fancy way.

    Args:
        axis: The axis to draw the hyper-edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the hyper-edges on.
        hyper_edges: The hyper-edges to draw.
    """
    for hyper_edge in hyper_edges:

        # store all paths for the hyper-edge to combine them later
        all_paths = []

        # make also normal (2-way) edges plottable -> one node becomes the "center" node
        is_hyper_edge = True
        if len(hyper_edge) == 2:
            u, v = hyper_edge
            center_pos = pos[v]
            node_size = graph[u][v]["size"]
            color = graph[u][v]["color"]
            alpha = graph[u][v]["alpha"]
            is_hyper_edge = False
        else:  # a hyper-edge encodes its information in an artificial "center" node
            center_pos = pos[hyper_edge]
            node_size = graph.nodes.get(hyper_edge)["size"]
            color = graph.nodes.get(hyper_edge)["color"]
            alpha = graph.nodes.get(hyper_edge)["alpha"]

        alpha = min(1.0, max(0.0, alpha))

        # draw the connection point of the hyper-edge
        circle = mpath.Path.circle(center_pos, radius=node_size / 2)
        all_paths.append(circle)
        axis.scatter(center_pos[0], center_pos[1], s=0, c="none", lw=0)  # add empty point for limit

        # draw the fancy connections from the other nodes to the center node
        for player in hyper_edge:

            player_pos = pos[player]

            circle_p = mpath.Path.circle(player_pos, radius=node_size / 2)
            all_paths.append(circle_p)
            axis.scatter(player_pos[0], player_pos[1], s=0, c="none", lw=0)  # for axis limits

            # get the direction of the connection
            direction = (center_pos[0] - player_pos[0], center_pos[1] - player_pos[1])
            direction = np.array(direction) / np.linalg.norm(direction)

            # get 90 degree of the direction
            direction_90 = np.array([-direction[1], direction[0]])

            # get the distance between the player and the center node
            distance = np.linalg.norm(center_pos - player_pos)

            # get the position of the start and end of the connection
            start_pos = player_pos - direction_90 * (node_size / 2)
            middle_pos = player_pos + direction * distance / 2
            end_pos_one = center_pos - direction_90 * (node_size / 2)
            end_pos_two = center_pos + direction_90 * (node_size / 2)
            start_pos_two = player_pos + direction_90 * (node_size / 2)

            # create the connection
            connection = mpath.Path(
                [
                    start_pos,
                    middle_pos,
                    end_pos_one,
                    end_pos_two,
                    middle_pos,
                    start_pos_two,
                    start_pos,
                ],
                [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                ],
            )

            # add the connection to the list of all paths
            all_paths.append(connection)

            # break after the first hyper-edge if there are only two players
            if not is_hyper_edge:
                break

        # combine all paths into one patch
        combined_path = mpath.Path.make_compound_path(*all_paths)
        patch = mpatches.PathPatch(combined_path, facecolor=color, lw=0, alpha=alpha)

        axis.add_patch(patch)


def _draw_graph_nodes(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    nodes: Optional[list] = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
) -> None:
    """Draws the nodes of the graph as circles with a fixed size.

    Args:
        ax: The axis to draw the nodes on.
        pos: The positions of the nodes.
        graph: The graph to draw the nodes on.
        nodes: The nodes to draw. If ``None``, all nodes are drawn. Defaults to ``None``.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
    """
    for node in graph.nodes:
        if nodes is not None and node not in nodes:
            continue

        position = pos[node]
        circle = mpath.Path.circle(position, radius=normal_node_size / 2)
        patch = mpatches.PathPatch(circle, facecolor="white", lw=1, alpha=1, edgecolor="black")
        ax.add_patch(patch)

        # add empty scatter for the axis to adjust the limits later
        ax.scatter(position[0], position[1], s=0, c="none", lw=0)


def _draw_explanation_nodes(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    nodes: Optional[list] = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
) -> None:
    """Adds the node level explanations to the graph as circles with varying sizes.

    Args:
        ax: The axis to draw the nodes on.
        pos: The positions of the nodes.
        graph: The graph to draw the nodes on.
        nodes: The nodes to draw. If ``None``, all nodes are drawn. Defaults to ``None``.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
    """
    for node in graph.nodes:
        if isinstance(node, tuple):
            continue
        if nodes is not None and node not in nodes:
            continue
        position = pos[node]
        color = graph.nodes.get(node)["color"]
        explanation_size = graph.nodes.get(node)["size"]
        alpha = 1.0
        if ADJUST_NODE_ALPHA:
            alpha = graph.nodes.get(node)["alpha"]

        alpha = min(1.0, max(0.0, alpha))

        radius = normal_node_size / 2 + explanation_size / 2
        if SCALE_NODE_EXPLANATIONS_BY_AREA:
            # get the radius of a circle with the same area as the combined area
            normal_node_area = math.pi * (normal_node_size / 2) ** 2
            this_node_area = math.pi * (explanation_size / 2) ** 2
            combined_area = normal_node_area + this_node_area
            radius = math.sqrt(combined_area / math.pi)

        circle = mpath.Path.circle(position, radius=radius)
        patch = mpatches.PathPatch(circle, facecolor=color, lw=1, edgecolor="white", alpha=alpha)
        ax.add_patch(patch)

        ax.scatter(position[0], position[1], s=0, c="none", lw=0)  # add empty point for limits


def _draw_graph_edges(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    edges: Optional[list[tuple]] = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
) -> None:
    """Draws black lines between the nodes.

    Args:
        ax: The axis to draw the edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the edges on.
        edges: The edges to draw. If ``None`` (default), all edges are drawn.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
    """
    for u, v in graph.edges:

        if edges is not None and (u, v) not in edges and (v, u) not in edges:
            continue

        u_pos = pos[u]
        v_pos = pos[v]

        direction = v_pos - u_pos
        direction = direction / np.linalg.norm(direction)

        start_point = u_pos + direction * normal_node_size / 2
        end_point = v_pos - direction * normal_node_size / 2

        connection = mpath.Path(
            [start_point, end_point],
            [mpath.Path.MOVETO, mpath.Path.LINETO],
        )

        patch = mpatches.PathPatch(connection, facecolor="none", lw=1, edgecolor="black")
        ax.add_patch(patch)


def _draw_graph_labels(
    ax: plt.axis, pos: dict, graph: nx.Graph, nodes: Optional[list] = None
) -> None:
    """Adds labels to the nodes of the graph.

    Args:
        ax: The axis to draw the labels on.
        pos: The positions of the nodes.
        graph: The graph to draw the labels on.
        nodes: The nodes to draw the labels on. If ``None`` (default), all nodes are drawn.
    """
    for node in graph.nodes:
        if nodes is not None and node not in nodes:
            continue
        label = graph.nodes.get(node)["label"]
        position = pos[node]
        ax.text(
            position[0],
            position[1],
            label,
            fontsize=plt.rcParams["font.size"] + 1,
            ha="center",
            va="center",
            color="black",
        )


def _adjust_position(
    pos: dict, graph: nx.Graph, normal_node_size: float = NORMAL_NODE_SIZE
) -> dict:
    """Moves the nodes in the graph further apart if they are too close together."""
    # get the minimum distance between two nodes
    min_distance = 1e10
    for u, v in graph.edges:
        distance = np.linalg.norm(pos[u] - pos[v])
        min_distance = min(min_distance, distance)

    # adjust the positions if the nodes are too close together
    min_edge_distance = normal_node_size + normal_node_size / 2
    if min_distance < min_edge_distance:
        for node in pos:
            pos[node] = pos[node] * min_edge_distance / min_distance

    return pos


def si_graph_plot(
    interaction_values: InteractionValues,
    graph: Union[list[tuple], nx.Graph],
    n_interactions: Optional[int] = None,
    draw_threshold: float = 0.0,
    random_seed: int = 42,
    size_factor: float = 1.0,
    plot_explanation: bool = True,
    compactness: float = 1.0,
    label_mapping: Optional[dict] = None,
    cubic_scaling: bool = False,
    pos: Optional[dict] = None,
    node_size_scaling: float = 1.0,
    min_max_interactions: Optional[tuple[float, float]] = None,
    adjust_node_pos: bool = False,
    spring_k: Optional[float] = None,
    interaction_direction: Optional[str] = None,
) -> tuple[plt.figure, plt.axis]:
    """Plots the interaction values as an explanation graph.

    An explanation graph is an undirected graph where the nodes represent players and the edges
    represent interactions between the players. The size of the nodes and edges represent the
    strength of the interaction values. The color of the edges represents the sign of the
    interaction values.

    Args:
        interaction_values: The interaction values to plot.
        graph: The underlying graph structure as a list of edge tuples or a networkx graph. If a
            networkx graph is provided, the nodes are used as the players and the edges are used as
            the connections between the players.
        n_interactions: The number of interactions to plot. If ``None``, all interactions are plotted
            according to the draw_threshold.
        draw_threshold: The threshold to draw an edge (i.e. only draw explanations with an
            interaction value higher than this threshold).
        random_seed: The random seed to use for layout of the graph.
        size_factor: The factor to scale the explanations by (a higher value will make the
            interactions and main effects larger). Defaults to ``1.0``.
        plot_explanation: Whether to plot the explanation or only the original graph. Defaults to
            ``True``.
        compactness: A scaling factor for the underlying spring layout. A higher compactness value
            will move the interactions closer to the graph nodes. If your graph looks weird, try
            adjusting this value, e.g. ``[0.1, 1.0, 10.0, 100.0, 1000.0]``. Defaults to ``1.0``.
        label_mapping: A mapping from the player/node indices to the player label. If ``None``, the
            player indices are used as labels. Defaults to ``None``.
        cubic_scaling: Whether to scale the size of explanations cubically (``True``) or linearly
            (``False``, default). Cubic scaling puts more emphasis on larger interactions in the plot.
            Defaults to ``False``.
        pos: The positions of the nodes in the graph. If ``None``, the spring layout is used to
            position the nodes. Defaults to ``None``.
        node_size_scaling: The scaling factor for the node sizes. This can be used to make the nodes
            larger or smaller depending on how the graph looks. Defaults to ``1.0`` (no scaling).
            Negative values will make the nodes smaller, positive values will make the nodes larger.
        min_max_interactions: The minimum and maximum interaction values to use for scaling the
            interactions as a tuple ``(min, max)``. If ``None``, the minimum and maximum interaction
            values are used. Defaults to ``None``.
        adjust_node_pos: Whether to adjust the node positions such that the nodes are at least
            ``NORMAL_NODE_SIZE`` apart. Defaults to ``False``.
        spring_k: The spring constant for the spring layout. If `None`, the spring constant is
            calculated based on the number of nodes in the graph. Defaults to ``None``.
        interaction_direction: The sign of the interaction values to plot. If ``None``, all
            interactions are plotted. Possible values are ``"positive"`` and
            ``"negative"``. Defaults to ``None``.

    Returns:
        The figure and axis of the plot.
    """

    normal_node_size = NORMAL_NODE_SIZE * node_size_scaling
    base_size = BASE_SIZE * node_size_scaling

    # fill the original graph with the edges and nodes
    if isinstance(graph, nx.Graph):
        original_graph = graph
        graph_nodes = list(original_graph.nodes)
        # check if graph has labels
        if "label" not in original_graph.nodes[graph_nodes[0]]:
            for node in graph_nodes:
                node_label = label_mapping.get(node, node) if label_mapping is not None else node
                original_graph.nodes[node]["label"] = node_label
    else:
        original_graph, graph_nodes = nx.Graph(), []
        for edge in graph:
            original_graph.add_edge(*edge)
            nodel_labels = [edge[0], edge[1]]
            if label_mapping is not None:
                nodel_labels = [label_mapping.get(node, node) for node in nodel_labels]
            original_graph.add_node(edge[0], label=nodel_labels[0])
            original_graph.add_node(edge[1], label=nodel_labels[1])
            graph_nodes.extend([edge[0], edge[1]])

    if n_interactions is not None:
        # get the top n interactions
        interaction_values = interaction_values.get_top_k(n_interactions)

    # get the interactions to plot (sufficiently large)
    interactions_to_plot = {}
    min_interaction, max_interaction = 1e10, 0.0
    for interaction, interaction_pos in interaction_values.interaction_lookup.items():
        if len(interaction) == 0:
            continue
        interaction_value = interaction_values.values[interaction_pos]
        min_interaction = min(abs(interaction_value), min_interaction)
        max_interaction = max(abs(interaction_value), max_interaction)
        if abs(interaction_value) > draw_threshold:
            if interaction_direction == "positive" and interaction_value < 0:
                continue
            if interaction_direction == "negative" and interaction_value > 0:
                continue
            interactions_to_plot[interaction] = interaction_value

    if min_max_interactions is not None:
        min_interaction, max_interaction = min_max_interactions

    # create explanation graph
    explanation_graph, explanation_nodes, explanation_edges = nx.Graph(), [], []
    for interaction, interaction_value in interactions_to_plot.items():
        interaction_size = len(interaction)
        interaction_strength = abs(interaction_value)

        attributes = {
            "color": get_color(interaction_value),
            "alpha": _normalize_value(
                interaction_value, max_interaction, BASE_ALPHA_VALUE, cubic_scaling
            ),
            "interaction": interaction,
            "weight": interaction_strength * compactness,
            "size": _normalize_value(
                interaction_value, max_interaction, base_size * size_factor, cubic_scaling
            ),
        }

        # add main effect explanations as nodes
        if interaction_size == 1:
            player = interaction[0]
            explanation_graph.add_node(player, **attributes)
            explanation_nodes.append(player)

        # add 2-way interaction explanations as edges
        if interaction_size >= 2:
            explanation_edges.append(interaction)
            player_last = interaction[-1]
            if interaction_size > 2:
                dummy_node = tuple(interaction)
                explanation_graph.add_node(dummy_node, **attributes)
                player_last = dummy_node
            # add the edges between the players
            for player in interaction[:-1]:
                explanation_graph.add_edge(player, player_last, **attributes)

    # position first the original graph structure
    if pos is None:
        pos = nx.spring_layout(original_graph, seed=random_seed, k=spring_k)
        pos = nx.kamada_kawai_layout(original_graph, scale=1.0, pos=pos)
    else:
        pass
        # pos is given but we need to scale the positions potentially
        min_pos = np.min(list(pos.values()), axis=0)
        max_pos = np.max(list(pos.values()), axis=0)
        pos = {node: (pos[node] - min_pos) / (max_pos - min_pos) for node in pos}

    # adjust pos such that the nodes are at least NORMAL_NODE_SIZE apart
    if adjust_node_pos:
        pos = _adjust_position(pos, original_graph)

    # create the plot
    fig, ax = plt.subplots(figsize=(7, 7))
    if plot_explanation:
        # position now again the hyper-edges onto the normal nodes weight param is weight
        pos_explain = nx.spring_layout(
            explanation_graph, weight="weight", seed=random_seed, pos=pos, fixed=graph_nodes
        )
        pos.update(pos_explain)
        _draw_fancy_hyper_edges(ax, pos, explanation_graph, hyper_edges=explanation_edges)
        _draw_explanation_nodes(
            ax, pos, explanation_graph, nodes=explanation_nodes, normal_node_size=normal_node_size
        )

    # add the original graph structure on top
    _draw_graph_nodes(ax, pos, original_graph, normal_node_size=normal_node_size)
    _draw_graph_edges(ax, pos, original_graph, normal_node_size=normal_node_size)
    _draw_graph_labels(ax, pos, original_graph)

    # tidy up the plot
    ax.set_aspect("equal", adjustable="datalim")  # make y- and x-axis scales equal
    ax.axis("off")  # remove axis

    return fig, ax
