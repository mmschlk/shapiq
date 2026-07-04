"""Regression-based approximators to estimate Shapley interaction values."""

from .base import Regression
from .faithful import RegressionFBII, RegressionFSII
from .kadd_shap import kADDSHAP
from .kernelshap import KernelSHAP
from .kernelshapiq import InconsistentKernelSHAPIQ, KernelSHAPIQ
from .polyshap import PolySHAPKAdd, PolySHAPPartial, PolySHAPPrior
from .oddshap import OddSHAP

__all__ = [
    "kADDSHAP",
    "RegressionFSII",
    "KernelSHAP",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "Regression",
    "RegressionFBII",
    "PolySHAPKAdd",
    "PolySHAPPartial",
    "PolySHAPPrior",
    "OddSHAP",
]
