"""Tests the new explanation_plot function."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues


def _get_edge_color(interaction_value: float) -> str:
    """Returns the color of the edge based on the sign of the interaction value."""
    return "red" if interaction_value > 0 else "blue"


def explanation_plot(interaction_values: InteractionValues) -> tuple[plt.figure, plt.axis]:
    """Plots the interaction values as an explanation graph.

    An explanation graph is an undirected (spring_layout) graph where the nodes represent players
    and the edges represent interactions between the players. The value of the interaction is used
    as the weight of the edge. For interactions between more than two players, a hyper-edge is
    created via a dummy node (which is not displayed). The color of the edge indicates the sign of
    the interaction value (the alpha value indicates the strength of the interaction in relation to
    the maximum interaction value).

    Args:
        interaction_values: The interaction values to plot.

    Returns:
        The figure and axis of the plot.
    """
    G = nx.Graph()

    for interaction, interaction_pos in interaction_values.interaction_lookup.items():
        interaction_value = interaction_values.values[interaction_pos]
        interaction_strength = abs(interaction_value) * 10
        interaction_size = len(interaction)
        if interaction_size == 0:
            continue
        if len(interaction) == 1:
            player = interaction[0]
            G.add_node(player, size=interaction_strength, interaction_value=interaction_value)
        if interaction_size == 2 and interaction_value != 0:
            player1, player2 = interaction
            G.add_edge(
                player1,
                player2,
                weight=interaction_strength,
                interaction_value=interaction_value,
            )
        if interaction_size > 2 and interaction_value != 0:
            dummy_node = str(tuple(interaction))
            G.add_node(dummy_node, size=interaction_strength, interaction_value=interaction_value)
            for player in interaction:
                G.add_edge(
                    dummy_node,
                    player,
                    weight=interaction_strength,
                    interaction_value=interaction_value,
                )

    edge_sizes = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = [_get_edge_color(G[u][v]["interaction_value"]) for u, v in G.edges()]
    node_sizes = list(nx.get_node_attributes(G, "size").values())
    node_colors = list(nx.get_node_attributes(G, "interaction_value").values())
    node_colors = [_get_edge_color(interaction_value) for interaction_value in node_colors]

    pos = nx.spring_layout(G, weight="weight")

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_sizes, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, ax=ax)
    return fig, ax


if __name__ == "__main__":

    example_values = InteractionValues(
        n_players=4,
        values=np.array(
            [
                0.0,  # ()
                0.1,  # (1)
                0.1,  # (2)
                0.1,  # (3)
                0.1,  # (4)
                0.2,  # (1, 2)
                0.2,  # (1, 3)
                0.2,  # (1, 4)
                0.2,  # (2, 3)
                0.2,  # (2, 4)
                0.2,  # (3, 4)
                0.9,  # (1, 2, 3)
                0.0,  # (1, 2, 4)
                0.0,  # (1, 3, 4)
                0.0,  # (2, 3, 4)
                0.0,  # (1, 2, 3, 4)
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

    fig, ax = explanation_plot(example_values)
    plt.show()
