"""This module contains all functions for conducting benchmarks with the SHAPIQ package."""

from .configuration import (
    BENCHMARK_CONFIGURATIONS,
    GAME_CLASS_TO_NAME_MAPPING,
    GAME_NAME_TO_CLASS_MAPPING,
    print_benchmark_configurations,
)
from .load import download_game_data, load_game_data, load_games_from_configuration
from .plot import plot_approximation_quality
from .run import load_benchmark_results, run_benchmark, run_benchmark_from_configuration

__all__ = [
    # # configuration
    "print_benchmark_configurations",
    "BENCHMARK_CONFIGURATIONS",
    "GAME_CLASS_TO_NAME_MAPPING",
    "GAME_NAME_TO_CLASS_MAPPING",
    # # loading
    "load_games_from_configuration",
    "download_game_data",
    "load_game_data",
    # # running benchmarks
    "run_benchmark_from_configuration",
    "run_benchmark",
    "load_benchmark_results",
    # plotting benchmark results
    "plot_approximation_quality",
]
