"""Runner with config demo for the leaderboard."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

from config_manager import MVPRunConfig, load_and_validate_config
from leaderboard.runner.approximator_registry import get_approximator_class
from leaderboard.runner.benchmark_runner import run_benchmark
from leaderboard.runner.custom_types import InteractionIndex
from leaderboard.runner.game_factory import create_game_from_config
from leaderboard.storage.connection import DatabaseClientFactory

logging.basicConfig(level=logging.INFO)


class ExpandedRunConfig(TypedDict):
    """Run configuration expanded from validated MVP config."""

    game: str
    index: str
    approximator: str
    max_order: int
    budget: int
    seeds: list[int]
    game_seed: int


def expand_validated_config(config_obj: MVPRunConfig) -> list[ExpandedRunConfig]:
    """Expand validated MVP config to concrete run configs for the runner.

    Args:
        config_obj: Validated MVPRunConfig object from config_manager.

    Returns:
        List of expanded run configurations (one per approximator-budget combo).
    """
    run_configs: list[ExpandedRunConfig] = [
        {
            "game": config_obj.game,
            "index": config_obj.index,
            "approximator": approx,
            "max_order": config_obj.max_order,
            "budget": budget,
            "seeds": list(config_obj.seeds),
            "game_seed": config_obj.game_seed,
        }
        for approx in config_obj.approximators
        for budget in config_obj.budgets
    ]

    logging.info("Expanded %s run configurations from validated config.", len(run_configs))

    return run_configs


def load_raw_config(path: Path) -> dict[str, Any]:
    """Read raw YAML to preserve optional fields like game_params.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed YAML data as dictionary.
    """
    with Path.open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def main() -> None:
    """Run benchmarks from a YAML config and store raw results in MongoDB."""
    # Read system args
    argsv = sys.argv

    project_root = Path(__file__).resolve().parents[3]

    if len(argsv) > 1:
        logging.info("Using config file: %s", argsv[1])
        config_path = Path(argsv[1])
    else:
        config_path = project_root / "configs" / "default_run.yaml"

    # Load and validate config using config_manager interface
    config_obj = load_and_validate_config(config_path)
    if config_obj is None:
        raise FileNotFoundError from None

    # Expand validated config to concrete run configurations
    run_configs = expand_validated_config(config_obj)

    # Load raw config for optional fields like game_params base_config = load_raw_config(config_path)

    # Reuse validated config object so optional fields (e.g., game_params) stay typed.
    base_config = config_obj.model_dump(exclude_none=True)

    # Connect to MongoDB
    mongo_db = DatabaseClientFactory.create_client("mongodb", {})

    # Create a local database client
    output_path = project_root / "data" / "results_raw.jsonl"
    local_db = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(output_path)})

    # Test connection
    if not mongo_db.test_connection():
        raise ConnectionError from None
    logging.info("MongoDB connection successful.")

    # Run benchmarks for each expanded run configuration
    for run_config in run_configs:
        logging.info("Running benchmark config:")
        logging.info(json.dumps(run_config, indent=2, default=str))

        approximator_class = get_approximator_class(run_config["approximator"])

        game, game_params = create_game_from_config(
            run_config=dict(run_config),
            base_config=base_config,
        )

        benchmark_result = run_benchmark(
            game=game,
            game_name=run_config["game"],
            game_params=game_params,
            max_order=run_config["max_order"],
            approx_seeds=run_config["seeds"],
            budget=run_config["budget"],
            index=cast(InteractionIndex, run_config["index"]),
            approximator_class=approximator_class,
        )

        # Insert in local JSONL file
        local_db.insert_many(benchmark_result["raw_results"])

        # Insert in MongoDB
        mongo_db.insert_many(benchmark_result["raw_results"])

        logging.info("Stored raw results:")
        logging.info(len(benchmark_result["raw_results"]))
        logging.info("First raw result:")
        logging.info(json.dumps(benchmark_result["raw_results"][0], indent=2, default=str))


if __name__ == "__main__":
    # Note: We pass sys.argv to main() to allow config file path specification.
    main()
