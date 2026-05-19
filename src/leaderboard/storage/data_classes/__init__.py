"""This module defines data classes and enums used for representing run configurations, statuses, and types in the leaderboard storage system.

Includes:
- RunConfig: A data class representing the configuration of a benchmark run, including game, index
    approximator, max order, budget, seeds, and game seed.
- RunStatus: An enum representing the status of a run (e.g., SUCCESS, FAILURE).
- ApproximatorType: An enum representing different types of approximators used in the benchmarks.
- GroundTruthMethod: An enum representing different methods for computing ground truth interaction values.
"""

from .run_config import RunConfig

__all__ = ["RunConfig"]
