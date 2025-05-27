"""This script analyzes the Moebius coefficients for the benchmark games and plots the results."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

try:
    from shapiq.interaction_values import InteractionValues
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.makedirs("eval", exist_ok=True)
    from shapiq.interaction_values import InteractionValues


def plot_box_plot(
    interactions: list[InteractionValues],
    min_size: int = 0,
    max_size: int | None = None,
    save_path: str | None = None,
    title: str = "Box plot of the interaction values",
    *,
    showfliers: bool = True,
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
    # no outlier
    interactions_df.boxplot(by="size", column="value", ax=ax, showfliers=showfliers)
    ax.set_ylabel("MI value")
    ax.set_xlabel("MI size")
    # remove the title of the boxplot
    plt.suptitle("")
    ax.set_title(title)
    # make the y axis scientific notation
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

    game_name = "AdultCensusLocalXAI"  # AdultCensusLocalXAI
    config_id = 3
    n_player_id = 0
    index = "SV"  # "k-SII" or "SV"
    order = 1  # 1 or 2

    n_games = 1

    if index == "SV":
        order = 1

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

    # plot the box plot
    save_path = os.path.join("eval", f"{game_name}_{n_games}_moebius_box_plot.pdf")
    plot_box_plot(
        interactions=moebius_values,
        min_size=0,
        max_size=n_players,
        title=create_application_name(game_name, abbrev=True),
        showfliers=True,
        save_path=save_path,
    )
