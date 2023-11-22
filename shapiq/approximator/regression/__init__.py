"""This module contains the regression-based approximators to estimate Shapley interaction values.
"""
from ._base import Regression
from .fsi import RegressionFSI

__all__ = ["RegressionFSI", "Regression"]
