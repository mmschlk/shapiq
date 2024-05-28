"""This script pre-computes the language model after the configurations."""

import sys
from pathlib import Path

# add shapiq to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


if __name__ == "__main__":

    from shapiq.games.benchmark import (
        ImageClassifierLocalXAI,  # 1 config for 3 player ids
    )
    from shapiq.games.benchmark.benchmark_config import (
        BENCHMARK_CONFIGURATIONS,
        BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
    )
    from shapiq.games.benchmark.precompute import pre_compute_from_configuration

    BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"] = True

    # get the game class
    game_class = ImageClassifierLocalXAI
    config_id = 1
    n_player_id = 2
    n_jobs = 1

    # get the configuration
    all_game_configs = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["configurations"]
    game_config = all_game_configs[config_id - 1]

    # run the pre-computation
    pre_compute_from_configuration(
        game_class, configuration=game_config, n_player_id=n_player_id, n_jobs=n_jobs
    )
