import matplotlib.patches as mpatches
import matplotlib.path as mpath
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from shapiq import InteractionValues

from .network import _get_color


def draw_fancy_hyper_edges(
    axis: plt.axis,
    pos: dict,
    graph: nx.Graph,
    hyper_edges: list[tuple],
) -> None:
    """Draws a collection of hyper-edges as a fancy hyper-edge."""
    for hyper_edge in hyper_edges:

        node_size = graph.nodes.get(hyper_edge)["size"]

        all_paths = []

        # get the position of the center of the hyper-edge (i.e. where the dummy node is placed)
        color = graph.nodes.get(hyper_edge)["color"]

        # draw the connection point of the hyper-edge
        center_pos = pos[hyper_edge]
        circle = mpath.Path.circle(center_pos, radius=node_size / 2)
        all_paths.append(circle)
        axis.scatter(center_pos[0], center_pos[1], s=0, c="none", lw=0)  # add empty for axis limits

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
            all_paths.append(connection)

        # combine all paths into one patch
        combined_path = mpath.Path.make_compound_path(*all_paths)
        patch = mpatches.PathPatch(combined_path, facecolor=color, lw=0, alpha=0.5)

        axis.add_patch(patch)


def draw_nodes(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    nodes: list,
    alpha: float = 1.0,
    line_width: int = 0,
    plot_explanation: bool = True,
) -> None:
    for node in nodes:
        if plot_explanation:
            color = graph.nodes.get(node)["color"]
            size = graph.nodes.get(node)["size"]
            edge_color = "white"
        else:
            color = "none"
            size = 0.125
            edge_color = "black"
        position = pos[node]
        circle = mpath.Path.circle(position, radius=size / 2)
        patch = mpatches.PathPatch(
            circle, facecolor=color, lw=line_width, alpha=alpha, edgecolor=edge_color
        )
        ax.add_patch(patch)

        # add empty scatter for the axis to adjust the limits later
        ax.scatter(position[0], position[1], s=0, c="none", lw=0)


def draw_normal_edges(ax: plt.axis, pos: dict, edges: list[tuple]) -> None:
    """Draws black lines between the nodes."""
    for u, v in edges:
        u_pos = pos[u]
        v_pos = pos[v]

        direction = v_pos - u_pos
        direction = direction / np.linalg.norm(direction)

        start_point = u_pos + direction * 0.125 / 2
        end_point = v_pos - direction * 0.125 / 2

        connection = mpath.Path(
            [start_point, end_point],
            [mpath.Path.MOVETO, mpath.Path.LINETO],
        )

        patch = mpatches.PathPatch(connection, facecolor="none", lw=1, edgecolor="black")
        ax.add_patch(patch)


def draw_edges(ax: plt.axis, pos: dict, graph: nx.Graph, edges: list[tuple]) -> None:
    """Draws the edges of the graph as lines."""
    for u, v in edges:
        color = graph[u][v]["color"]
        size = graph[u][v]["size"]

        u_pos = pos[u]
        v_pos = pos[v]

        direction = v_pos - u_pos
        direction = direction / np.linalg.norm(direction)
        direction_90 = np.array([-direction[1], direction[0]])

        # draw a rectangle between the two nodes
        connection = mpath.Path(
            [
                u_pos + direction_90 * size / 2,
                u_pos - direction_90 * size / 2,
                v_pos - direction_90 * size / 2,
                v_pos + direction_90 * size / 2,
                u_pos + direction_90 * size / 2,
            ],
            [
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.CLOSEPOLY,
            ],
        )

        patch = mpatches.PathPatch(connection, facecolor=color, alpha=0.5, edgecolor="none")
        ax.add_patch(patch)


def draw_labels(ax: plt.axis, pos: dict, graph: nx.Graph, nodes: list) -> None:
    """Adds labels to the nodes of the graph."""
    for node in nodes:
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


def explanation_graph(
    interaction_values: InteractionValues,
    edges: list[tuple],
    draw_threshold: float = 0.0,
    random_seed: int = 42,
    size_factor: float = 0.75,
    plot_explanation: bool = False,
    weight_factor: float = 1,
) -> tuple[plt.figure, plt.axis]:
    """Plots the interaction values as an explanation graph.

    An explanation graph is an undirected (spring_layout) graph where the nodes represent players
    and the edges represent interactions between the players. The value of the interaction is used
    as the weight of the edge. For interactions between more than two players, a hyper-edge is
    created via a dummy node (which is not displayed). The color of the edge indicates the sign of
    the interaction value (the alpha value indicates the strength of the interaction in relation to
    the maximum interaction value).

    Args:
        interaction_values: The interaction values to plot.
        edges: The edges in the graph.
        draw_threshold: The threshold to draw an edge (i.e. only draw edges with a value above the
            threshold).
        random_seed: The random seed to use for the spring_layout algorithm.

    Returns:
        The figure and axis of the plot.
    """
    G = nx.Graph()

    normal_nodes, normal_edges, hyper_edges = [], [], []
    for interaction, interaction_pos in interaction_values.interaction_lookup.items():
        interaction_value = interaction_values.values[interaction_pos]
        interaction_strength = abs(interaction_value)
        interaction_size = len(interaction)
        interaction_color = _get_color(interaction_value)
        normal_weight = 0
        if interaction in edges:
            normal_weight = 1
        if interaction_size == 0:
            continue
        if interaction_size == 1:
            player = interaction[0]
            G.add_node(
                player,
                weight=interaction_strength,
                normal_weight=normal_weight,
                size=interaction_strength * size_factor,
                normal_size=0.125,
                interaction_value=interaction_value,
                color=interaction_color,
                label=player,
            )
            normal_nodes.append(player)
        if interaction_size == 2 and interaction_strength > draw_threshold:
            player1, player2 = interaction
            G.add_edge(
                player1,
                player2,
                weight=interaction_strength,
                normal_weight=normal_weight,
                size=interaction_strength * size_factor,
                normal_size=0.125,
                interaction_value=interaction_value,
                color=interaction_color,
                label=interaction,
            )
            normal_edges.append(interaction)
        if interaction_size > 2 and interaction_strength > draw_threshold:
            dummy_node = tuple(interaction)
            G.add_node(
                dummy_node,
                weight=interaction_strength,
                normal_weight=0,  # not used for first layout
                size=interaction_strength * size_factor,
                normal_size=0.125,
                interaction_value=interaction_value,
                color=interaction_color,
                label=interaction,
            )
            for player in interaction:
                G.add_edge(
                    dummy_node,
                    player,
                    weight=interaction_strength * weight_factor,
                    normal_weight=0,  # not used for first layout
                    size=interaction_strength * size_factor,
                    normal_size=0.125,
                    interaction_value=interaction_value,
                    color=interaction_color,
                    label=interaction,
                )
            hyper_edges.append(interaction)

    # position first the normal nodes
    pos = nx.spring_layout(normal_nodes, seed=random_seed, weight="normal_weight")

    if plot_explanation:
        # position now again the hyper-edges onto the normal nodes weight param is weight
        pos = nx.spring_layout(G, weight="weight", seed=random_seed, pos=pos, fixed=normal_nodes)

    fig, ax = plt.subplots(figsize=(7, 7))
    if plot_explanation:
        draw_fancy_hyper_edges(ax, pos, G, hyper_edges)
        draw_edges(ax, pos, G, normal_edges)
        draw_edges(ax, pos, G, normal_edges)
        draw_nodes(ax, pos, G, normal_nodes, line_width=1, plot_explanation=True)
    draw_nodes(ax, pos, G, normal_nodes, line_width=1, plot_explanation=False)
    draw_normal_edges(ax, pos, edges)
    draw_labels(ax, pos, G, normal_nodes)

    ax.set_aspect("equal", adjustable="datalim")  # make y- and x-axis scales equal
    ax.axis("off")  # remove axis

    return fig, ax
