import yaml
from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a dictionary/object at the top level.")

    return config


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expands a YAML config into concrete benchmark configurations.

    Supports both:
        approximator: "ProxySHAP"
        budget: 100

    and:
        approximators: ["ProxySHAP", "KernelSHAPIQ"]
        budgets: [100, 500, 1000]
    """

    game = config["game"]
    index = config["index"]
    max_order = config["max_order"]
    n_seeds = config.get("n_seeds", 1)
    game_seed = config.get("game_seed", 42)

    approximators = as_list(
        config.get("approximators", config.get("approximator"))
    )

    budgets = as_list(
        config.get("budgets", config.get("budget"))
    )

    if approximators == [None]:
        raise ValueError("Config must contain 'approximator' or 'approximators'.")

    if budgets == [None]:
        raise ValueError("Config must contain 'budget' or 'budgets'.")

    run_configs = []

    for approximator in approximators:
        for budget in budgets:
            run_configs.append(
                {
                    "game": game,
                    "index": index,
                    "approximator": approximator,
                    "max_order": max_order,
                    "budget": budget,
                    "n_seeds": n_seeds,
                    "game_seed": game_seed,
                }
            )

    return run_configs