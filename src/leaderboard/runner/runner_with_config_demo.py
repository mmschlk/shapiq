"""Runner with config demo for the leaderboard."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import yaml
from sparse_transform.qsft.signals.input_signal_subsampled import (
    SubsampledSignal as SubsampledSignalFourier,
)

from leaderboard.config_manager import load_and_validate_config
from leaderboard.runner.approximator_registry import get_approximator_class
from leaderboard.runner.benchmark_runner import run_benchmark
from leaderboard.runner.game_factory import create_game_from_config
from leaderboard.storage.connection import DatabaseClientFactory

if TYPE_CHECKING:
    from leaderboard.config_manager import MVPRunConfig
    from leaderboard.runner.custom_types import InteractionIndex

# Configure dedicated module logger to resolve LOG015
logger = logging.getLogger(__name__)
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
    run_configs: list[ExpandedRunConfig] = []

    for approx in config_obj.approximators:
        for budget in config_obj.budgets:
            if approx == "SPEX":
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
                    logger.warning(
                        "⚠️  Skipping %s at budget %s: Calculated bits (b=%s) <= 2 "
                        "is invalid for game '%s' (n=%s).",
                        approx,
                        budget,
                        calculated_b,
                        config_obj.game,
                        config_obj.n_players,
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
    logger.info("Expanded %s run configurations from validated config.", len(run_configs))

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
    argsv = sys.argv
    project_root = Path(__file__).resolve().parents[3]

    if len(argsv) > 1:
        logger.info("Using config file: %s", argsv[1])
        config_path = Path(argsv[1])
    else:
        config_path = project_root / "configs" / "default_run.yaml"

    try:
        yaml_content = Path(config_path).read_text(encoding="utf-8")
        raw_yaml_config = yaml.safe_load(yaml_content) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Could not read raw YAML configuration for audit: %s", e)
        raw_yaml_config = {}

    config_obj = load_and_validate_config(config_path)
    if config_obj is None:
        raise FileNotFoundError from None

    base_config = config_obj.model_dump(exclude_none=True)

    raw_apps = raw_yaml_config.get("approximators", [])
    sanitized_apps = base_config.get("approximators", [])
    purged_apps = [a for a in raw_apps if a not in sanitized_apps]

    raw_budgets = raw_yaml_config.get("budgets", [])
    sanitized_budgets = base_config.get("budgets", [])
    purged_budgets = [b for b in raw_budgets if b not in sanitized_budgets]

    sanitized_seeds_list = base_config.get("seeds", [])
    seeds_count = len(sanitized_seeds_list)

    model_name = config_obj.game_params.get("model_name", "N/A")
    imputer_name = config_obj.game_params.get("imputer", "N/A")

    sys.stdout.write("\n" + "=" * 80 + "\n")
    sys.stdout.write("🛡️  SHAPIQ RUNNER SWEEP CONFIGURATION AUDIT\n")
    sys.stdout.write("-" * 80 + "\n")
    sys.stdout.write(
        f"  ▶️ Target Game          : '{config_obj.game}' (n_players={config_obj.n_players})\n"
    )
    sys.stdout.write(f"  ▶️ Active Pipeline Type: '{config_obj.game_family}'\n")
    sys.stdout.write(f"  ▶️ Game Model Backend   : '{model_name}'\n")
    sys.stdout.write(f"  ▶️ Feature Imputer      : '{imputer_name}'\n")
    sys.stdout.write(
        f"  ▶️ Target Interaction  : '{config_obj.index}' (Max Order: {config_obj.max_order})\n"
    )
    sys.stdout.write("-" * 80 + "\n")

    if purged_apps:
        sys.stdout.write("⚠️  APPROXIMATOR FILTER NOTICE:\n")
        sys.stdout.write(
            f"  ❌ Removed incompatible algorithms for {config_obj.index}: {purged_apps}\n"
        )
        sys.stdout.write("-" * 80 + "\n")

    if purged_budgets:
        sys.stdout.write("⚠️  BUDGET RANGE FILTER NOTICE:\n")
        sys.stdout.write(f"  ❌ Removed out-of-bounds budgets: {purged_budgets}\n")
        sys.stdout.write("-" * 80 + "\n")

    sys.stdout.write("🎯 GROUND TRUTH CONTROL PROFILE:\n")
    sys.stdout.write(f"  ➡️ Strategy             : '{config_obj.ground_truth.strategy}'\n")
    sys.stdout.write(f"  ➡️ Method               : '{config_obj.ground_truth.method}'\n")
    sys.stdout.write("-" * 80 + "\n")
    sys.stdout.write("📦 FINAL SANITIZED SWEEP EXECUTION LISTS:\n")
    sys.stdout.write(f"  ✅ Run Approximators : {sanitized_apps}\n")
    sys.stdout.write(f"  ✅ Run Budgets       : {sanitized_budgets}\n")
    sys.stdout.write(f"  ✅ Run Seeds Total   : {seeds_count} (values: {sanitized_seeds_list})\n")
    sys.stdout.write("=" * 80 + "\n\n")

    run_configs = expand_validated_config(config_obj)
    mongo_db = DatabaseClientFactory.create_client("mongodb", {})

    output_path = project_root / "data" / "results_raw.jsonl"
    local_db = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(output_path)})

    if not mongo_db.test_connection():
        raise ConnectionError from None
    logger.info("MongoDB connection successful.")

    for run_config in run_configs:
        logger.info("Running benchmark config:")
        logger.info(json.dumps(run_config, indent=2, default=str))

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
            index=cast("InteractionIndex", run_config["index"]),
            approximator_class=approximator_class,
            ground_truth_method=run_config["ground_truth_method"],
        )

        local_db.insert_many(benchmark_result["raw_results"])
        mongo_db.insert_many(benchmark_result["raw_results"])

        logger.info("Stored raw results:")
        logger.info(len(benchmark_result["raw_results"]))
        logger.info("First raw result:")
        logger.info(json.dumps(benchmark_result["raw_results"][0], indent=2, default=str))


if __name__ == "__main__":
    main()
