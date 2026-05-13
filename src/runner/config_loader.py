import yaml
from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a benchmark configuration from a YAML file.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        ValueError: If the YAML file does not contain a dictionary/object at the
            top level.
    """
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a dictionary/object at the top level.")

    return config


def as_list(value: Any) -> list[Any]:
    """Wrap a value inside a list if it is not already a list.

        Args:
            value: The value to convert to a list.

        Returns:
            "value" if it is already a list, otherwise "[value]".
        """
    if isinstance(value, list):
        return value
    return [value]


def expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    A YAML config is expanded into a concrete benchmark config.

    The configuration may use either singular fields such as "approximator"
    and "budget" or plural fields such as "approximators" and
    "budgets". Plural fields are expanded into all combinations of
    approximators and budgets.

    Args:
        config: The loaded benchmark configuration.

    Returns:
        A list of concrete run configurations, one for each approximator-budget
        combination.

    Raises:
        KeyError: If a required configuration entry is missing.
        ValueError: If no approximators or budgets are provided.
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