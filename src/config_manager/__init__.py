"""
Config Manager Package.

Configuration management for the shapiq benchmark leaderboard.

Public API:
  - MVPRunConfig: Main configuration model;
  - GroundTruthConfig: Ground truth configuration;
  - load_and_validate_config(): Load and validate YAML configs;
  - VALID_INDICES: Type hint for valid index types;
  - ALL_SUPPORTED_APPROXIMATORS: List of supported approximators.
  - Custom exceptions in config_exceptions.py for specific validation errors.
"""

from .config import (
    ALL_SUPPORTED_APPROXIMATORS,
    GroundTruthConfig,
    MVPRunConfig,
    VALID_INDICES,
    load_and_validate_config,
)

from .config_exceptions import (
    ApproximatorIndexIncompatibleError,
    ApproximatorNotFoundError,
    InvalidBudgetError,
    InvalidConfigMissingFieldsError,
    InvalidConfigMissingApproximatorsError,
    InvalidConfigMissingBudgetsError,
    InvalidOrderForIndexError,
    InvalidYAMLTypeError,
    UnsupportedApproximatorError,
)

__all__ = [
    "ALL_SUPPORTED_APPROXIMATORS",
    "GroundTruthConfig",
    "load_and_validate_config",
    "MVPRunConfig",
    "VALID_INDICES",
    "InvalidBudgetError",
    "InvalidConfigMissingFieldsError",
    "InvalidConfigMissingApproximatorsError",
    "InvalidConfigMissingBudgetsError",
    "InvalidOrderForIndexError",
    "InvalidYAMLTypeError",
    "ApproximatorIndexIncompatibleError",
    "ApproximatorNotFoundError",
    "UnsupportedApproximatorError",
]