"""Regression-based approximators to estimate Shapley interaction values."""

from .base import Regression
from .faithful import RegressionFBII, RegressionFSII
from .kadd_shap import kADDSHAP
from .kernelshap import KernelSHAP
from .kernelshapiq import InconsistentKernelSHAPIQ, KernelSHAPIQ
from .leverageshap import LeverageSHAP
from .oddshap import OddSHAP
from .polyshap import PolySHAP

__all__ = [
    "kADDSHAP",
    "RegressionFSII",
    "KernelSHAP",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "Regression",
    "RegressionFBII",
    "LeverageSHAP",
    "PolySHAP",
    "OddSHAP",
]
