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
from .proxy import ProxySHAP, ProxySPEX, RegressionMSR
from .regression import (
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    OddSHAP,
    PolySHAP,
    RegressionFBII,
    RegressionFSII,
    kADDSHAP,
)

try:
    from .shapleig import ShaplEIG
except ImportError as _e:

    class ShaplEIG(Approximator):
        """Placeholder raised when the optional ``shapleig`` extra is not installed."""

        _import_error = _e

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise an informative ImportError pointing to the missing extra."""
            raise self._import_error


try:
    from .sparse import SPEX
except ImportError as _e:

    class SPEX(Approximator):
        """Placeholder raised when the optional ``sparse`` extra is not installed."""

        _import_error = _e

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise an informative ImportError pointing to the missing extra."""
            raise self._import_error


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
    RegressionMSR,
    ProxySPEX,
    OddSHAP,
    ShaplEIG,
    PolySHAP,
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
    SPEX,
    ProxySPEX,
    ProxySHAP,
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
]

# contains all approximators that can be used for FBII
FBII_APPROXIMATORS: list[Approximator.__class__] = [RegressionFBII, SPEX, ProxySPEX, ProxySHAP]


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
    "OddSHAP",
    "RegressionMSR",
    "ShaplEIG",
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
    "PolySHAP",
]
