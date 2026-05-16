"""
Configuration Loader

This module handles loading and validating configuration files.
"""

import yaml
from pathlib import Path
from typing import Union
from .models import MVPRunConfig


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
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        config = MVPRunConfig(**data)
        print(f"✅ Configuration '{yaml_path}' loaded successfully.")
        return config
    except Exception as e:
        print(f"❌ Configuration Validation Error:\n{e}")
        return None
