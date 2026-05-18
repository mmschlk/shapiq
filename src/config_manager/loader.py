"""
Configuration Loader.

This module handles loading and validating configuration files.
"""

import yaml
from typing import TYPE_CHECKING, Union
from .models import MVPRunConfig
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    pass

def load_and_validate_config(yaml_path: Union[str, Path]) -> Union[MVPRunConfig, None]:
    """
    Load and validate a configuration file.
    
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
        logging.info(f"✅ Configuration '{yaml_path}' loaded successfully.")
        return config
    except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
        logging.error(f"❌ Configuration Validation Error:\n{e}")
        return None
