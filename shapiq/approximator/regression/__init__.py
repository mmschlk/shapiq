"""This module contains the regression-based approximators to estimate Shapley interaction values.
"""
from .sii import RegressionSII
from .fsi import RegressionFSI

__all__ = ["RegressionSII", "RegressionFSI"]
