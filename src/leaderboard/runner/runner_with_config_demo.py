import json
import yaml
from pathlib import Path
from typing import Any, TypedDict, cast

from leaderboard.storage.connection import MongoDBClient, load_env
from leaderboard.runner.approximator_registry import get_approximator_class
from leaderboard.runner.benchmark_runner import run_benchmark
from config_manager import load_and_validate_config, MVPRunConfig
from leaderboard.runner.custom_types import InteractionIndex
from leaderboard.runner.game_factory import create_game_from_config
from leaderboard.runner.runner_storage_adapter import save_raw_results


class ExpandedRunConfig(TypedDict):
    """Run configuration expanded from validated MVP config."""

    game: str
    index: str
    approximator: str
    max_order: int
    budget: int
    n_seeds: int
    game_seed: int


def expand_validated_config(config_obj: MVPRunConfig) -> list[ExpandedRunConfig]:
    """
    Expand validated MVP config to concrete run configs for the runner.

    Args:
        config_obj: Validated MVPRunConfig object from config_manager.

    Returns:
        List of expanded run configurations (one per approximator-budget combo).
    """
    n_seeds = len(config_obj.seeds)
    game_seed = 42

    run_configs: list[ExpandedRunConfig] = []
    for approximator in config_obj.approximators:
        for budget in config_obj.budgets:
            run_configs.append(
                {
                    "game": config_obj.game,
                    "index": config_obj.index,
                    "approximator": approximator,
                    "max_order": config_obj.max_order,
                    "budget": budget,
                    "n_seeds": n_seeds,
                    "game_seed": game_seed,
                }
            )
    return run_configs


def load_raw_config(path: Path) -> dict[str, Any]:
    """
    Read raw YAML to preserve optional fields like game_params.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed YAML data as dictionary.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}




def main(argsv=None) -> None:
    """Run benchmarks from a YAML config and store raw results in MongoDB."""

    if len(argsv) > 1:
        print(f"Using config file: {argsv[1]}")
        config_path = Path(argsv[1])
    else:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "default_run.yaml"

    # Load and validate config using config_manager interface
    config_obj = load_and_validate_config(config_path)
    if config_obj is None:
        raise ValueError(f"Invalid config file: {config_path}")

    # Expand validated config to concrete run configurations
    run_configs = expand_validated_config(config_obj)

    # Load raw config for optional fields like game_params
    base_config = load_raw_config(config_path)

    # Connect to MongoDB
    db = MongoDBClient.from_env()

    # Test connection
    db._client.admin.command("ping")
    print("MongoDB connection successful.")

    # Run benchmarks for each expanded run configuration
    for run_config in run_configs:
        print("Running benchmark config:")
        print(json.dumps(run_config, indent=2, default=str))

        approximator_class = get_approximator_class(run_config["approximator"])

        game, game_params = create_game_from_config(
            run_config=dict(run_config),
            base_config=base_config,
        )

        benchmark_result = run_benchmark(
            game=game,
            game_name=run_config["game"],
            game_params=game_params,
            game_seed=run_config["game_seed"],
            max_order=run_config["max_order"],
            number_of_different_approx_seeds=run_config["n_seeds"],
            budget=run_config["budget"],
            index=cast(InteractionIndex, run_config["index"]),
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
    # Note: We pass sys.argv to main() to allow config file path specification.
    main(argsv=__import__("sys").argv)