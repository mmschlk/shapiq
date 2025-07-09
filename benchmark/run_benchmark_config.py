"""This script runs the benchmark from a specified configuration."""

import argparse
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # example python run commands for the CaliforniaHousingLocalXAI game
    # nice -n 19 python run_benchmark_config.py --game CaliforniaHousingLocalXAI --config_id 4 --n_player_id 0 --n_games -1 --index SV --order 2 --n_jobs 60 --rerun_if_exists True

    from shapiq.games.benchmark.run import run_benchmark_from_configuration

    # default values
    game = "CaliforniaHousingLocalXAI"
    config_id = 4
    n_player_id = 0
    index = "k-SII"
    order = 2
    n_games = -1
    n_jobs = 1

    # parse arguments if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default=game)
    parser.add_argument("--config_id", type=int, default=config_id)
    parser.add_argument("--n_player_id", type=int, default=n_player_id)
    parser.add_argument("--n_games", type=int, default=n_games, help="-1 for all games")
    parser.add_argument("--index", type=str, default=index)
    parser.add_argument("--order", type=int, default=order)
    parser.add_argument("--n_jobs", type=int, default=n_jobs)
    parser.add_argument("--rerun_if_exists", type=bool, default=False)
    args = parser.parse_args()

    # parse arguments
    game = args.game
    config_id = args.config_id
    n_player_id = args.n_player_id
    n_games = args.n_games
    if n_games == -1:
        n_games = None
    index = args.index
    order = args.order
    if index == "SV":
        order = 1
    n_jobs = args.n_jobs
    rerun_if_exists = args.rerun_if_exists

    run_benchmark_from_configuration(
        index=index,
        order=order,
        game_class=game,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
        n_jobs=n_jobs,
        max_budget=None,
        rerun_if_exists=rerun_if_exists,
    )
