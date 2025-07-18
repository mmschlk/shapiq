"""This module contains stubs for deprecated functions and throws a descriptive error."""

from __future__ import annotations

import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_deprecated_symbols = {
    # benchmark functionality
    "benchmark": "shapiq_benchmark",
    "BENCHMARK_CONFIGURATIONS": "shapiq_benchmark",
    "GAME_CLASS_TO_NAME_MAPPING": "shapiq_benchmark",
    "GAME_NAME_TO_CLASS_MAPPING": "shapiq_benchmark",
    "download_game_data": "shapiq_benchmark",
    "load_benchmark_results": "shapiq_benchmark",
    "load_game_data": "shapiq_benchmark",
    "load_games_from_configuration": "shapiq_benchmark",
    "plot_approximation_quality": "shapiq_benchmark",
    "print_benchmark_configurations": "shapiq_benchmark",
    "run_benchmark": "shapiq_benchmark",
    "run_benchmark_from_configuration": "shapiq_benchmark",
    # games
    "SOUM": "shapiq_games",
}


def try_import_deprecated_from_new_package(name: str, new_package: str) -> ModuleType:
    """Try to import a deprecated symbol or module from a new package."""
    msg = (
        f"shapiq.{name} is deprecated and has been moved to `{new_package}`.\n"
        f"Use `from {new_package} import {name}` instead of `from shapiq import {name}`. "
        "Note that the API may have changed.\n"
    )
    if find_spec(new_package) is None:
        pypi_name = new_package.replace("_", "-")  # shapiq_benchmark -> shapiq-benchmark
        msg += f"Install it via: `pip install {pypi_name}`.\n"
        raise ImportError(msg)
    warnings.warn(msg, stacklevel=2)
    return import_module(f"{new_package}.{name}")


def try_import_deprecated(name: str) -> ModuleType | None:
    """Try to import a deprecated symbol from the shapiq package."""
    if name in _deprecated_symbols:
        return try_import_deprecated_from_new_package(name, _deprecated_symbols[name])
    return None
