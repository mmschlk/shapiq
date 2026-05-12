from typing import Any

from shapiq_games.benchmark.local_xai.benchmark_tabular import CaliforniaHousing


def create_game_from_config(run_config: dict[str, Any], base_config: dict[str, Any]):
    game_name = run_config["game"]
    game_seed = run_config["game_seed"]
    max_order = run_config["max_order"]

    if game_name == "SOUM":
        game_params = base_config.get(
            "game_params",
            {
                "n": 10,
                "n_basis_games": 20,
                "min_interaction_size": 1,
                "max_interaction_size": max_order,
                "random_state": game_seed,
            },
        )
    elif game_name == "CaliforniaHousing":
        game_params = base_config.get(
            "game_params",
            {
                "x": 0,
                "model_name": "decision_tree",
                "imputer": "marginal",
                "normalize": True,
                "verbose": False,
                "random_state": game_seed,
            },
        )

        game = CaliforniaHousing(**game_params)
        return game, game_params

    raise ValueError(f"Unknown game: {game_name}")