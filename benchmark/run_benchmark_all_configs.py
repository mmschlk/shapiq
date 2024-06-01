"""This script runs all benchmarks for all pre-computed configurations configuration."""

import argparse
import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":

    # example python run command with nohup and nice
    # nohup nice -n 19 python run_benchmark_all_configs.py --n_jobs 70 --rerun_if_exists True > all_configs.log &

    from shapiq.games.benchmark.benchmark_config import (
        BENCHMARK_CONFIGURATIONS,
        GAME_TO_CLASS_MAPPING,
        get_game_class_from_name,
    )
    from shapiq.games.benchmark.run import run_benchmark_from_configuration

    indices_order = [("k-SII", 2), ("SV", 1)]

    # add arguments to the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--rerun_if_exists",
        type=bool,
        required=False,
        default=False,
    )
    args = parser.parse_args()
    n_jobs = args.n_jobs

    n_runs_done, n_configs_tried = 0, 0

    for index, order in indices_order:
        # get all configurations that include the dataset in the name
        all_game_names = [game_name for game_name in GAME_TO_CLASS_MAPPING.keys()]
        for game_name in all_game_names:
            game_class = get_game_class_from_name(game_name)
            all_game_class_configs = BENCHMARK_CONFIGURATIONS[game_class]
            for n_player_id, config_per_player_id in enumerate(all_game_class_configs):
                player_id_configs = config_per_player_id["configurations"]
                n_players = config_per_player_id["n_players"]
                for i, config in enumerate(player_id_configs):
                    config_id = i + 1
                    print()
                    print(f"Pre-computing game: {game_name}, config {config} with ID {config_id}")
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
                            rerun_if_exists=False,
                        )
                        n_runs_done += 1
                        print(f"Ran {n_runs_done} out of {n_configs_tried} configurations.")
                    except Exception as e:
                        print(f"Error occurred: {e}. Continuing.")
                        continue

    print(f"Ran {n_runs_done} out of {n_configs_tried} configurations.")
