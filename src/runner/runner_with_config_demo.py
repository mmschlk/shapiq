import json

from leaderboard.storage.database import MongoDBClient
from leaderboard.storage.main import load_env

from benchmark_runner import run_benchmark
from runner_storage_adapter import save_raw_results
from config_loader import load_yaml_config, expand_config
from approximator_registry import get_approximator_class
from game_factory import create_game_from_config
from pathlib import Path


def main() -> None:
    """Run benchmarks from a YAML config and store raw results in MongoDB."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "default_run.yaml"

    config = load_yaml_config(config_path)
    run_configs = expand_config(config)

    uri, db_name = load_env()
    db = MongoDBClient(uri=uri, db_name=db_name)
    db.client.admin.command("ping")
    print("MongoDB connection successful.")

    for run_config in run_configs:
        print("Running benchmark config:")
        print(json.dumps(run_config, indent=2, default=str))

        approximator_class = get_approximator_class(
            run_config["approximator"]
        )

        game, game_params = create_game_from_config(
            run_config=run_config,
            base_config=config,
        )

        benchmark_result = run_benchmark(
            game=game,
            game_name=run_config["game"],
            game_params=game_params,
            game_seed=run_config["game_seed"],
            max_order=run_config["max_order"],
            number_of_different_approx_seeds=run_config["n_seeds"],
            budget=run_config["budget"],
            index=run_config["index"],
            approximator_class=approximator_class,
        )

        save_raw_results(
            db=db,
            raw_results=benchmark_result["raw_results"],
        )

        print("Stored raw results:")
        print(len(benchmark_result["raw_results"]))
        print("First raw result:")
        print(json.dumps(benchmark_result["raw_results"][0], indent=2, default=str))


if __name__ == "__main__":
    main()