"""
Configuration Module - Public API.

This module provides a clean public interface for configuration management by re-exporting from internal modules:
  - constants: Configuration constants and whitelists
  - models: Data models and validators
  - loader: Configuration loading and validation

For implementation details, see the respective submodules.
"""

# Re-export from constants
from .constants import VALID_INDICES, ALL_SUPPORTED_APPROXIMATORS

# Re-export from models
from .models import GroundTruthConfig, MVPRunConfig

# Re-export from loader
from .loader import load_and_validate_config

__all__ = [
    # Constants
    "VALID_INDICES",
    "ALL_SUPPORTED_APPROXIMATORS",
    # Models
    "GroundTruthConfig",
    "MVPRunConfig",
    # Loader
    "load_and_validate_config",
]
