"""Runner with config demo for the leaderboard."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

from leaderboard.config_manager import MVPRunConfig, constants, load_and_validate_config
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
    ground_truth_method: str


def expand_validated_config(config_obj: MVPRunConfig) -> list[ExpandedRunConfig]:
    """Expand validated MVP config to concrete run configs for the runner.

    Args:
        config_obj: Validated MVPRunConfig object from config_manager.

    Returns:
        List of expanded run configurations (one per approximator-budget combo).
    """
    # run_configs: list[ExpandedRunConfig] = [
    #     {
    #         "game": config_obj.game,
    #         "index": config_obj.index,
    #         "approximator": approx,
    #         "max_order": config_obj.max_order,
    #         "budget": budget,
    #         "seeds": list(config_obj.seeds),
    #         "game_seed": config_obj.game_seed,
    #         "ground_truth_method": config_obj.ground_truth.method,
    #     }
    #     for approx in config_obj.approximators
    #     for budget in config_obj.budgets
    # ]
    run_configs: list[ExpandedRunConfig] = []

    for approx in config_obj.approximators:
        for budget in config_obj.budgets:
            # ---------------------------------------------------------------------
            # 🛡️ Dynamic SPEX Structural Budget Filter
            # ---------------------------------------------------------------------
            if approx == "SPEX":
                from sparse_transform.qsft.signals.input_signal_subsampled import (
                    SubsampledSignal as SubsampledSignalFourier,
                )

                # Default query parameters matching SPEX/__init__ setup
                degree_parameter = 5
                query_args = {
                    "query_method": "complex",
                    "num_subsample": 3,
                    "delays_method_source": "joint-coded",
                    "subsampling_method": "qsft",
                    "delays_method_channel": "identity-siso",
                    "num_repeat": 1,
                    "t": degree_parameter,
                }

                # Dynamically calculate the safe lower-bound bit 'b' for the current n_players
                calculated_b = SubsampledSignalFourier.get_b_for_sample_budget(
                    budget, config_obj.n_players, degree_parameter, 2, query_args
                )

                # If b <= 2, SPEX is mathematically guaranteed to crash with an Insufficient Budget Error
                if calculated_b <= 2:
                    logging.warning(
                        f"⚠️  Skipping {approx} at budget {budget}: "
                        f"Calculated bits (b={calculated_b}) <= 2 is invalid for game '{config_obj.game}' (n={config_obj.n_players})."
                    )
                    continue

            run_configs.append(
                {
                    "game": config_obj.game,
                    "index": config_obj.index,
                    "approximator": approx,
                    "max_order": config_obj.max_order,
                    "budget": budget,
                    "seeds": list(config_obj.seeds),
                    "game_seed": config_obj.game_seed,
                    "ground_truth_method": config_obj.ground_truth.method,
                }
            )
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
    # 1. Capture Raw Unsanitized User Input from YAML File
    try:
        import yaml

        yaml_content = Path(config_path).read_text(encoding="utf-8")
        raw_yaml_config = yaml.safe_load(yaml_content) or {}
    except Exception as e:
        logging.warning("Could not read raw YAML configuration for audit: %s", e)
        raw_yaml_config = {}

    # 2. Load and validate config using config_manager interface
    config_obj = load_and_validate_config(config_path)
    if config_obj is None:
        raise FileNotFoundError from None

    # 3. Reuse Validated and Sanitized Config Object for Running Pipeline
    base_config = config_obj.model_dump(exclude_none=True)

    # ---------------------------------------------------------------------
    # 🔍 AUDIT CHANNEL: Cross-Check Raw Input vs. Sanitized Configuration
    # ---------------------------------------------------------------------
    raw_apps = raw_yaml_config.get("approximators", [])
    sanitized_apps = base_config.get("approximators", [])
    purged_apps = [a for a in raw_apps if a not in sanitized_apps]

    raw_budgets = raw_yaml_config.get("budgets", [])
    sanitized_budgets = base_config.get("budgets", [])
    purged_budgets = [b for b in raw_budgets if b not in sanitized_budgets]

    sanitized_seeds_list = base_config.get("seeds", [])
    seeds_count = len(sanitized_seeds_list)

    print("\n" + "=" * 80)
    print("🛡️  SHAPIQ RUNNER SWEEP CONFIGURATION AUDIT")
    print("-" * 80)
    print(f"  ▶️ Target Game          : '{config_obj.game}' (n_players={config_obj.n_players})")
    print(f"  ▶️ Active Pipeline Type: '{config_obj.game_family}'")
    print(f"  ▶️ Target Interaction  : '{config_obj.index}' (Max Order: {config_obj.max_order})")
    print("-" * 80)

    # Report filtered approximators
    if purged_apps:
        print("⚠️  APPROXIMATOR FILTER NOTICE:")
        print(f"  ❌ Removed incompatible algorithms for {config_obj.index}: {purged_apps}")
        print("-" * 80)

    # Report filtered budgets
    if purged_budgets:
        print("⚠️  BUDGET RANGE FILTER NOTICE:")
        print(f"  ❌ Removed out-of-bounds budgets: {purged_budgets}")
        print("-" * 80)

    print("🎯 GROUND TRUTH CONTROL PROFILE:")
    print(f"  ➡️ Strategy             : '{config_obj.ground_truth.strategy}'")
    print(f"  ➡️ Method               : '{config_obj.ground_truth.method}'")
    print("-" * 80)
    print("📦 FINAL SANITIZED SWEEP EXECUTION LISTS:")
    print(f"  ✅ Run Approximators : {sanitized_apps}")
    print(f"  ✅ Run Budgets       : {sanitized_budgets}")
    print(f"  ✅ Run Seeds Total   : {seeds_count} (values: {sanitized_seeds_list})")
    print("=" * 80 + "\n")
    # ---------------------------------------------------------------------

    # Expand validated config to concrete run configurations
    run_configs = expand_validated_config(config_obj)

    # Load raw config for optional fields like game_params base_config = load_raw_config(config_path)

    # Reuse validated config object so optional fields (e.g., game_params) stay typed.
    # base_config = config_obj.model_dump(exclude_none=True)

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
            ground_truth_method=run_config["ground_truth_method"],
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
