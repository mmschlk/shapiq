"""This script pre-computes the games provided the benchmark configurations for certain parameters."""

import argparse
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


if __name__ == "__main__":
    from shapiq.games.benchmark.benchmark_config import (
        BENCHMARK_CONFIGURATIONS,
        BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
        GAME_NAME_TO_CLASS_MAPPING,
        get_game_class_from_name,
    )
    from shapiq.games.benchmark.precompute import pre_compute_from_configuration

    # example python run commands for the SentimentAnalysisLocalXAI game
    # nohup nice -n 19 python pre_compute.py --game SentimentAnalysisLocalXAI --config_id 1 --n_player_id 0 --n_jobs 1 > SentimentAnalysisLocalXAI_1_1.log 2>&1 &

    default_game = "BikeSharingClusterExplanation"
    default_config_id = 2
    default_n_player_id = 0
    default_n_jobs = 1
    default_verbose = True

    parser = argparse.ArgumentParser()
    game_choices = list(GAME_NAME_TO_CLASS_MAPPING.keys())
    parser.add_argument(
        "--game",
        type=str,
        required=False,
        choices=game_choices,
        default=default_game,
    )
    parser.add_argument(
        "--config_id",
        type=int,
        required=False,
        default=default_config_id,
        help="The configuration ID to use.",
    )
    parser.add_argument(
        "--n_player_id",
        type=int,
        required=False,
        default=default_n_player_id,
        help="The player ID to use. Defaults to 0. Not all games have multiple player IDs",
    )
    parser.add_argument("--n_jobs", type=int, required=False, default=default_n_jobs)
    parser.add_argument("--verbose", type=bool, required=False, default=default_verbose)
    args = parser.parse_args()
    game = args.game
    config_id = args.config_id
    n_player_id = args.n_player_id
    n_jobs = args.n_jobs
    verbose = args.verbose

    if verbose:
        BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"] = True

    # get the game class
    game_class = get_game_class_from_name(game)

    # get the configuration
    all_game_configs = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["configurations"]
    n_configs = len(all_game_configs)
    if config_id < 1 or config_id > n_configs:
        msg = (
            f"Invalid configuration ID. Must be in [1, {n_configs}] for game {game} which has "
            f"{all_game_configs} configurations."
        )
        raise ValueError(msg)

    game_config = all_game_configs[config_id - 1]

    # run the pre-computation
    pre_compute_from_configuration(
        game_class,
        configuration=game_config,
        n_player_id=n_player_id,
        n_jobs=n_jobs,
    )
