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


def plot_box_plot(
    interactions: list[InteractionValues],
    min_size: int = 0,
    max_size: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Box plot of the interaction values",
    showfliers: bool = True,
    y_lim: Optional[tuple[float, float]] = None,
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
    fig, ax = plt.subplots()
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
    ax.set_ylabel("MI value")
    ax.set_xlabel("MI size")
    # add grid dotted with LIGHT_GRAY
    ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")
    # remove the title of the boxplot
    plt.suptitle("")
    ax.set_title(title)
    # make the y axis scientific notation
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    from shapiq.exact import ExactComputer
    from shapiq.games.benchmark.benchmark_config import (
        get_game_class_from_name,
        load_games_from_configuration,
    )
    from shapiq.games.benchmark.plot import create_application_name

    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"figure.figsize": (5, 5)})
    plt.rcParams.update({"figure.dpi": 400})
    y_lim = None

    game_name = "AdultCensusDataValuation"  # AdultCensusDataValuation AdultCensusLocalXAI
    config_id = 1
    n_player_id = 0
    if game_name == "AdultCensusDataValuation":
        config_id = 2
        n_player_id = 0
    if game_name == "AdultCensusLocalXAI":
        config_id = 3
        n_player_id = 0

    n_games = 10

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

    # plot the box plot with outliers
    if game_name == "AdultCensusDataValuation":
        y_lim = (-1.2e2, 1.2e2)
    if game_name == "AdultCensusLocalXAI":
        y_lim = (-7e-3, 7e-3)
    save_path = os.path.join("eval", f"{game_name}_{n_games}_moebius_box_plot.png")
    plot_box_plot(
        interactions=moebius_values,
        min_size=0,
        max_size=n_players,
        title=create_application_name(game_name, abbrev=True),
        showfliers=True,
        save_path=save_path,
        y_lim=y_lim,
    )

    # plot the box plot without outliers
    if game_name == "AdultCensusDataValuation":
        y_lim = (-9e1, 9e1)
    if game_name == "AdultCensusLocalXAI":
        y_lim = (-0.6e-3, 0.6e-3)
    save_path = os.path.join("eval", f"{game_name}_{n_games}_moebius_box_plot_no_outliers.png")
    plot_box_plot(
        interactions=moebius_values,
        min_size=0,
        max_size=n_players,
        title=create_application_name(game_name, abbrev=True),
        showfliers=False,
        save_path=save_path,
        y_lim=y_lim,
    )
