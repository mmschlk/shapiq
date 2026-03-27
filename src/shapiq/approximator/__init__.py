"""Algorithms for approximating Shapley values and interaction indices.

All approximators inherit from :class:`~shapiq.approximator.Approximator` and follow
a common interface: pass a :class:`~shapiq.Game` and a computational budget, and receive
:class:`~shapiq.interaction_values.InteractionValues` as output.
"""

from .base import Approximator
from .marginals import OwenSamplingSV, StratifiedSamplingSV
from .montecarlo import SHAPIQ, SVARM, SVARMIQ, UnbiasedKernelSHAP
from .permutation.sii import PermutationSamplingSII
from .permutation.stii import PermutationSamplingSTII
from .permutation.sv import PermutationSamplingSV
from .proxy import MSRBiased, ProxySHAP
from .regression import (
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    RegressionFBII,
    RegressionFSII,
    kADDSHAP,
)
from .sparse import SPEX, ProxySPEX

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
    ProxySPEX,
    ProxySHAP,
    MSRBiased,
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
    ProxySHAP,
    MSRBiased,
]

# contains all approximators that can be used for SII
SII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
    ProxySPEX,
    ProxySHAP,
    MSRBiased,
]

# contains all approximators that can be used for STII
STII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSTII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
    ProxySPEX,
    ProxySHAP,
    MSRBiased,
]

# contains all approximators that can be used for FSII
FSII_APPROXIMATORS: list[Approximator.__class__] = [
    RegressionFSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
    SPEX,
    ProxySPEX,
    ProxySHAP,
    MSRBiased,
]

# contains all approximators that can be used for FBII
FBII_APPROXIMATORS: list[Approximator.__class__] = [
    RegressionFBII,
    SPEX,
    ProxySPEX,
    ProxySHAP,
    MSRBiased,
]

__all__ = [
    "Approximator",
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "PermutationSamplingSV",
    "StratifiedSamplingSV",
    "OwenSamplingSV",
    "KernelSHAP",
    "RegressionFSII",
    "RegressionFBII",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "ProxySPEX",
    "ProxySHAP",
    "MSRBiased",
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
    "FBII_APPROXIMATORS",
]
