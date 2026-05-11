from shapiq.approximator import ProxySHAP
from shapiq_games.synthetic import SOUM
from benchmark_runner import run_benchmark


def main() -> None:
    game_seed = 42
    max_order = 2

    game_params = {
        "n": 10,
        "n_basis_games": 20,
        "min_interaction_size": 1,
        "max_interaction_size": max_order,
        "random_state": game_seed,
    }

    game = SOUM(**game_params)

    run_benchmark(
        game=game,
        game_name="SOUM",
        game_params=game_params,
        game_seed=game_seed,
        max_order=max_order,
        number_of_different_approx_seeds=30,
        budget=100,
        index="SII",
        approximator_class=ProxySHAP,
    )


if __name__ == "__main__":
    main()