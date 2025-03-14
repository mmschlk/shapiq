"""Regression-based approximators to estimate Shapley interaction values."""

from ._base import Regression
from .fsi import RegressionFSII, RegressionFBII
from .kadd_shap import kADDSHAP
from .kernelshap import KernelSHAP
from .kernelshapiq import InconsistentKernelSHAPIQ, KernelSHAPIQ

__all__ = [
    "kADDSHAP",
    "RegressionFSII",
    "KernelSHAP",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "Regression",
    "RegressionFBII",
]
