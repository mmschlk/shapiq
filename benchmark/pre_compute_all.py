"""This script pre-computes all benchmark games that are not already pre-computed."""

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

    # example python run command
    # nohup nice -n 19 python pre_compute_all.py --datasets CaliforniaHousing > compute_CaliforniaHousing.log &
    # nohup nice -n 19 python pre_compute_all.py --datasets AdultCensus > compute_AdultCensus.log &
    # nohup nice -n 19 python pre_compute_all.py --datasets BikeSharing > compute_BikeSharing.log &

    datasets_to_precompute = [
        "AdultCensus",
        "BikeSharing",
        "CaliforniaHousing",
        "Sentiment",
        "Image",
        "SynthData",
        "SOUM",
    ]
    max_n_players = 16

    # add arguments to the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        required=False,
        nargs="+",
        help="The datasets to pre-compute. Defaults to all datasets.",
        default=datasets_to_precompute,
    )
    args = parser.parse_args()
    datasets_to_precompute = args.datasets

    # for games to be omitted
    omit_games = []

    n_jobs = 1
    verbose = True
    if verbose:
        BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"] = True

    for game in datasets_to_precompute:
        # get all configurations that include the dataset in the name
        all_game_names = [
            game_name
            for game_name in GAME_NAME_TO_CLASS_MAPPING
            if game in game_name and game_name not in omit_games
        ]
        for game_name in all_game_names:
            game_class = get_game_class_from_name(game_name)
            all_game_class_configs = BENCHMARK_CONFIGURATIONS[game_class]
            for n_player_id, config_per_player_id in enumerate(all_game_class_configs):
                player_id_configs = config_per_player_id["configurations"]
                n_players = config_per_player_id["n_players"]
                precompute = config_per_player_id["precompute"]
                if not precompute:
                    continue
                if n_players > max_n_players:
                    continue
                for _, config in enumerate(player_id_configs):
                    pre_compute_from_configuration(
                        game_class,
                        configuration=config,
                        n_player_id=n_player_id,
                        n_jobs=n_jobs,
                    )
