"""This module contains all markers for the tests."""

import importlib.util

import pytest

__all__ = [
    "skip_if_no_tabpfn",
    "skip_if_no_tensorflow",
    "skip_if_no_keras",
    "skip_if_no_xgboost",
    "skip_if_no_lightgbm",
]


def is_installed(pkg_name: str) -> bool:
    """Check if a package is installed without importing it."""
    return importlib.util.find_spec(pkg_name) is not None


skip_if_no_tabpfn = pytest.mark.skipif(
    not is_installed("tabpfn"),
    reason="TabPFN is not available.",
)

skip_if_no_tensorflow = pytest.mark.skipif(
    not is_installed("tensorflow"),
    reason="tensorflow is not installed",
)

skip_if_no_xgboost = pytest.mark.skipif(
    not is_installed("xgboost"),
    reason="xgboost is not installed",
)

skip_if_no_keras = pytest.mark.skipif(not is_installed("keras"), reason="keras is not installed")

skip_if_no_lightgbm = pytest.mark.skipif(
    not is_installed("lightgbm"),
    reason="lightgbm is not installed",
)
