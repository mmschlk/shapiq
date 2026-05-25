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
from .regression import (
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    RegressionFBII,
    RegressionFSII,
    kADDSHAP,
)

try:
    from .sparse import SPEX

    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False

try:
    from .proxy import ProxySHAP, ProxySPEX, RegressionMSR

    _HAS_PROXY = True
except ImportError:
    _HAS_PROXY = False

# contains all SV approximators
SV_APPROXIMATORS: list[Approximator.__class__] = [
    OwenSamplingSV,
    StratifiedSamplingSV,
    SVARM,
    UnbiasedKernelSHAP,
    PermutationSamplingSV,
    KernelSHAP,
    kADDSHAP,
]
if _HAS_SPARSE:
    SV_APPROXIMATORS.append(SPEX)
if _HAS_PROXY:
    SV_APPROXIMATORS.extend([ProxySPEX, RegressionMSR])

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
if _HAS_PROXY:
    SI_APPROXIMATORS.extend([ProxySHAP, RegressionMSR])

# contains all approximators that can be used for SII
SII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
]
if _HAS_SPARSE:
    SII_APPROXIMATORS.append(SPEX)
if _HAS_PROXY:
    SII_APPROXIMATORS.extend([ProxySPEX, ProxySHAP])

# contains all approximators that can be used for STII
STII_APPROXIMATORS: list[Approximator.__class__] = [
    PermutationSamplingSTII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
]
if _HAS_SPARSE:
    STII_APPROXIMATORS.append(SPEX)
if _HAS_PROXY:
    STII_APPROXIMATORS.append(ProxySPEX)

# contains all approximators that can be used for FSII
FSII_APPROXIMATORS: list[Approximator.__class__] = [
    RegressionFSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
]
if _HAS_SPARSE:
    FSII_APPROXIMATORS.append(SPEX)
if _HAS_PROXY:
    FSII_APPROXIMATORS.extend([ProxySPEX, ProxySHAP])

# contains all approximators that can be used for FBII
FBII_APPROXIMATORS: list[Approximator.__class__] = [
    RegressionFBII,
]
if _HAS_SPARSE:
    FBII_APPROXIMATORS.append(SPEX)
if _HAS_PROXY:
    FBII_APPROXIMATORS.extend([ProxySPEX, ProxySHAP])

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
    "RegressionMSR",
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
