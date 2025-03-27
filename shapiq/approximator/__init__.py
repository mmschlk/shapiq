"""Approximators to estimate the Shapley interaction values."""

from ._base import Approximator
from .marginals import OwenSamplingSV, StratifiedSamplingSV
from .montecarlo import SHAPIQ, SVARM, SVARMIQ, UnbiasedKernelSHAP
from .permutation.sii import PermutationSamplingSII
from .permutation.stii import PermutationSamplingSTII
from .permutation.sv import PermutationSamplingSV
from .regression import InconsistentKernelSHAPIQ, KernelSHAP, KernelSHAPIQ, RegressionFSII, kADDSHAP
from .sparse import SPEX

# TODO JK: Check if addition of SPEX is okay


# contains all SV approximators
SV_APPROXIMATORS: list[Approximator.__class__] = [
    OwenSamplingSV,
    StratifiedSamplingSV,
    SVARM,
    UnbiasedKernelSHAP,
    PermutationSamplingSV,
    KernelSHAP,
    kADDSHAP,
    SPEX,
]

# contains all SI approximators
SI_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSII,
    PermutationSamplingSTII,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAPIQ,
    RegressionFSII,
]

# contains all approximators that can be used for SII
SII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
]

# contains all approximators that can be used for STII
STII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSTII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
]

# contains all approximators that can be used for FSII
FSII_APPROXIMATORS: list[Approximator.__class__] = [
    RegressionFSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
]

__all__ = [
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "PermutationSamplingSV",
    "StratifiedSamplingSV",
    "OwenSamplingSV",
    "KernelSHAP",
    "RegressionFSII",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "SHAPIQ",
    "SVARM",
    "SVARMIQ",
    "kADDSHAP",
    "SPEX",
    "UnbiasedKernelSHAP",
    "SV_APPROXIMATORS",
    "SI_APPROXIMATORS",
    "SII_APPROXIMATORS",
    "STII_APPROXIMATORS",
    "FSII_APPROXIMATORS",
]

# Path: shapiq/approximator/__init__.py
