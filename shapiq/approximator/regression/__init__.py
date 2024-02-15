"""This module contains the regression-based approximators to estimate Shapley interaction values.
"""

from .fsi import RegressionFSI
from .sii import RegressionSII
from .sv import KernelSHAP

__all__ = ["RegressionSII", "RegressionFSI", "KernelSHAP"]
