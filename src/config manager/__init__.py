"""
Config Manager Package

Configuration management for the shapiq benchmark leaderboard.

Public API:
  - MVPRunConfig: Main configuration model
  - GroundTruthConfig: Ground truth configuration
  - load_and_validate_config(): Load and validate YAML configs
  - VALID_INDICES: Type hint for valid index types
  - ALL_SUPPORTED_APPROXIMATORS: List of supported approximators
"""

from .config import (
    VALID_INDICES,
    ALL_SUPPORTED_APPROXIMATORS,
    GroundTruthConfig,
    MVPRunConfig,
    load_and_validate_config,
)

__all__ = [
    "VALID_INDICES",
    "ALL_SUPPORTED_APPROXIMATORS",
    "GroundTruthConfig",
    "MVPRunConfig",
    "load_and_validate_config",
]
