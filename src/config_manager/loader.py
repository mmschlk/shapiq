"""Configuration Loader.

This module handles loading and validating configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .models import MVPRunConfig

logging.basicConfig(level=logging.INFO)


def load_and_validate_config(yaml_path: str | Path) -> MVPRunConfig | None:
    """Load and validate a configuration file.

    Attempts to load a YAML configuration file and validate it against the MVPRunConfig schema.
    If validation fails, an error message is printed and None is returned.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        MVPRunConfig if successful, None if validation fails.

    Example:
        >>> config = load_and_validate_config("configs/default_run.yaml")
        >>> if config:
        ...     print(config.game)
    """
    try:
        with Path(yaml_path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        config = MVPRunConfig(**data)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        logging.exception("Configuration Validation Error")
        raise
    else:
        logging.info(
            "Configuration '%s' loaded successfully.",
            yaml_path,
        )
        return config
