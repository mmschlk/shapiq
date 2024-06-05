"""This script analyzes the Moebius coefficients for the benchmark games and plots the results."""

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

try:
    from shapiq.interaction_values import InteractionValues
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.makedirs("eval", exist_ok=True)
    from shapiq.interaction_values import InteractionValues

HEX_BLACK = "#000000"
MEDIAN_COLOR = "#ef27a6"
LIGHT_GRAY = "#d3d3d3"
DARKER_GRAY = "#a9a9a9"


def plot_box_plot(
    ax: plt.Axes,
    interactions: list[InteractionValues],
    min_size: int = 0,
    max_size: Optional[int] = None,
    showfliers: bool = True,
    add_empty_size: int = 0,
    y_ticks: Optional[list[float]] = None,
) -> None:
    """Plot a box plot of the interaction values."""

    # get a dataframe of the interactions
    interactions_df = []
    for interaction_val in interactions:
        interactions_dict = interaction_val.dict_values
        for interaction, value in interactions_dict.items():
            interactions_df.append({"value": value, "size": len(interaction)})
    interactions_df = pd.DataFrame(interactions_df)

    # restrict the size of the interactions
    if max_size is None:
        max_size = interactions_df["size"].max()
    interactions_df = interactions_df[
        (interactions_df["size"] >= min_size) & (interactions_df["size"] <= max_size)
    ]

    # make the boxplot for each size
    for size in range(min_size, max_size + 1):
        data = interactions_df[interactions_df["size"] == size]["value"]
        ax.boxplot(
            data,
            positions=[size],
            showfliers=showfliers,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=HEX_BLACK + "20", color=HEX_BLACK),
            whiskerprops=dict(color=HEX_BLACK),
            capprops=dict(color=HEX_BLACK),
            flierprops=dict(marker="o", color=HEX_BLACK, markersize=3),
            medianprops=dict(color=MEDIAN_COLOR),
        )
    for i_add in range(add_empty_size):
        ax.boxplot(
            [],
            positions=[max_size + i_add + 1],
            showfliers=False,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=HEX_BLACK + "20", color=HEX_BLACK),
            whiskerprops=dict(color=HEX_BLACK),
            capprops=dict(color=HEX_BLACK),
            flierprops=dict(marker="o", color=HEX_BLACK, markersize=3),
            medianprops=dict(color=MEDIAN_COLOR),
        )

    ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")

    if y_ticks is not None:
        ax.set_yticks(y_ticks)


def conduct_correlation_analysis(max_n_player: int = 15) -> None:
    pass


if __name__ == "__main__":
    from shapiq.exact import ExactComputer
    from shapiq.games.benchmark.benchmark_config import (
        get_game_class_from_name,
        load_games_from_configuration,
    )

    n_games = 10

    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"figure.figsize": (5, 5)})
    plt.rcParams.update({"figure.dpi": 400})
    y_lim = None

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    axis_1 = axes[0]
    axis_2 = axes[1]

    save_path = os.path.join("eval", "moebius_analysis.png")

    # add the AdultCensusDataValuation -------------------------------------------------------------
    game_name = "AdultCensusDataValuation"
    config_id = 2
    n_player_id = 0

    # load a game
    game_class = get_game_class_from_name(game_name)
    games = load_games_from_configuration(
        game_class=game_class,
        config_id=config_id,
        n_player_id=n_player_id,
        only_pre_computed=True,
        n_games=n_games,
    )
    games = list(games)
    moebius_values = []
    n_players = games[0].n_players
    for game in tqdm(games, unit="game"):
        game.verbose = True
        computer = ExactComputer(n_players=game.n_players, game_fun=game)
        moebius = computer(index="Moebius", order=game.n_players)
        moebius_values.append(moebius)

    plot_box_plot(
        ax=axis_1,
        interactions=moebius_values,
        min_size=0,
        max_size=n_players,
        y_ticks=[-100, 0, 100],
    )
    axis_1.set_ylim(-120, 120)

    # add application name to the subplots
    game_title = r"Dat. Val. $n=15$"
    # add as text in bottom center of the subplot as a legend in axis_1
    axis_1.text(
        0.5,
        0.1,
        game_title,
        ha="center",
        va="center",
        transform=axis_1.transAxes,
        fontsize=12,
        # add a frame around the text
        bbox=dict(facecolor="white", edgecolor=DARKER_GRAY, boxstyle="round,pad=0.2"),
    )

    # add the AdultCensusLocalXAI ------------------------------------------------------------------
    game_name = "AdultCensusLocalXAI"
    config_id = 3
    n_player_id = 0

    # load a game
    game_class = get_game_class_from_name(game_name)
    games = load_games_from_configuration(
        game_class=game_class,
        config_id=config_id,
        n_player_id=n_player_id,
        only_pre_computed=True,
        n_games=n_games,
    )
    games = list(games)
    moebius_values = []
    n_players = games[0].n_players
    for game in tqdm(games, unit="game"):
        game.verbose = True
        computer = ExactComputer(n_players=game.n_players, game_fun=game)
        moebius = computer(index="Moebius", order=game.n_players)
        moebius_values.append(moebius)

    plot_box_plot(
        ax=axis_2,
        interactions=moebius_values,
        min_size=0,
        max_size=n_players,
        add_empty_size=1,
        y_ticks=[-0.02, 0.0, 0.02],
    )
    # ylim
    axis_2.set_ylim(-0.024, 0.024)

    # add application name to the subplots
    game_title = r"Loc. Exp. $n=14$"
    # add as text in bottom center of the subplot as a legend in axis_2
    axis_2.text(
        0.5,
        0.1,
        game_title,
        ha="center",
        va="center",
        transform=axis_2.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor=DARKER_GRAY, boxstyle="round,pad=0.2"),
    )

    # finalize the plot ----------------------------------------------------------------------------
    # add x-axis label
    axis_2.set_xlabel("Interaction size", fontsize=14)

    # add y-label for both subplots together
    fig.text(0.0, 0.5, "MIs", va="center", rotation="vertical", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    # save the plot
    plt.savefig(save_path)
    plt.show()
