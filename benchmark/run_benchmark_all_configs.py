"""This script runs all benchmarks for all pre-computed configurations configuration."""

import argparse
import logging
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # example python run command with nohup and nice
    # nohup nice -n 19 python run_benchmark_all_configs.py --n_jobs 100 > configs.log &

    from shapiq.games.benchmark.benchmark_config import (
        BENCHMARK_CONFIGURATIONS,
        GAME_NAME_TO_CLASS_MAPPING,
        get_game_class_from_name,
    )
    from shapiq.games.benchmark.run import run_benchmark_from_configuration

    indices_order = [
        ("k-SII", 2),
        ("SV", 1),
    ]
    rerun_if_exists = False

    # add arguments to the parser ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--omit_regex",
        type=str,
        required=False,
        nargs="+",
        default=[],
    )
    args = parser.parse_args()

    # parse the arguments --------------------------------------------------------------------------
    n_jobs = args.n_jobs
    omit_regex = args.omit_regex

    # print the arguments --------------------------------------------------------------------------

    # get all configurations that are not omitted by the name --------------------------------------
    all_game_names = []
    for game_name in GAME_NAME_TO_CLASS_MAPPING:
        omit = False
        for omit_regex_str in omit_regex:
            if omit_regex_str in game_name:
                omit = True
        if not omit:
            all_game_names.append(game_name)

    # run all configurations -----------------------------------------------------------------------
    n_runs_done, n_configs_tried = 0, 0
    for index, order in indices_order:
        for game_name in all_game_names:
            game_class = get_game_class_from_name(game_name)
            all_game_class_configs = BENCHMARK_CONFIGURATIONS[game_class]
            for n_player_id, config_per_player_id in enumerate(all_game_class_configs):
                player_id_configs = config_per_player_id["configurations"]
                n_players = config_per_player_id["n_players"]
                for i, _config in enumerate(player_id_configs):
                    config_id = i + 1
                    n_configs_tried += 1
                    try:
                        run_benchmark_from_configuration(
                            index=index,
                            order=order,
                            game_class=game_class,
                            game_configuration=config_id,
                            game_n_player_id=n_player_id,
                            game_n_games=None,
                            n_jobs=n_jobs,
                            max_budget=None,
                            rerun_if_exists=rerun_if_exists,
                        )
                        n_runs_done += 1
                    except Exception:
                        logging.exception("Error occurred: Continuing.")
                        continue
