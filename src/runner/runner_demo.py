from leaderboard.storage.database import MongoDBClient
from shapiq.approximator import ProxySHAP
from shapiq_games.synthetic import SOUM
from benchmark_runner import run_benchmark
from runner_storage_adapter import save_raw_results
import json
from leaderboard.storage.main import load_env



def main() -> None:
    """Run a local SOUM benchmark example."""
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

    benchmark_result = run_benchmark(
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

    # uri, db_name = load_env()

    # db = MongoDBClient(
    #     uri=uri,
    #     db_name=db_name,
    # )
    #
    # save_raw_results(
    #     db=db,
    #     raw_results=benchmark_result["raw_results"],
    # )

    print(json.dumps(benchmark_result["raw_results"][0], indent=2, default=str))


if __name__ == "__main__":
    main()