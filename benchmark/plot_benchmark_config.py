"""This script plots the benchmark results from a specified configuration."""

import os
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.makedirs("plots", exist_ok=True)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from shapiq.games.benchmark.benchmark_config import GAME_TO_CLASS_MAPPING
    from shapiq.games.benchmark.plot import get_game_title_name, plot_approximation_quality
    from shapiq.games.benchmark.run import load_benchmark_results

    print("Available games:", GAME_TO_CLASS_MAPPING.keys(), "\n")

    # run parameters
    save_fig = True

    # benchmark to plot parameters
    game = "BikeSharingUnsupervisedData"
    config_id = 1
    n_player_id = 0
    index = "SV"
    order = 2
    n_games = 1

    if index == "SV":
        order = 1

    # plot parameters
    log_scale_y = True
    log_scale_min = 1e-8
    log_scale_x = False
    y_lim = None  # 0.0, 0.001

    # create the title
    game_title = get_game_title_name(game)
    index_title = index if index != "k-SII" else f"{order}-SII"
    title = f"{game_title}, config. {config_id}, {index_title}, {n_games} games"

    # load the benchmark results
    results_df, save_path = load_benchmark_results(
        index=index,
        order=order,
        game_class=game,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
    )
    save_name = os.path.basename(save_path).split(".")[0] + ".pdf"
    save_path = os.path.join("plots", save_name)

    # plot the approximation quality
    fig, ax = plot_approximation_quality(
        data=results_df, log_scale_y=log_scale_y, log_scale_min=log_scale_min
    )
    ax.set_title(title)
    if log_scale_x:
        ax.set_xscale("log")
    if y_lim is not None and not log_scale_y:
        ax.set_ylim(y_lim)
    if save_fig:
        fig.savefig(save_path)
    plt.show()
