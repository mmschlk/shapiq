"""Approximator registry for the leaderboard runner.

This registry maps approximator names from config_manager to their corresponding
classes in shapiq.approximator. All 19 supported approximators are registered here.
"""

from __future__ import annotations

from leaderboard.config_manager.config_exceptions import (
    UnsupportedApproximatorError,
)
from shapiq.approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    Approximator,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    MSRBiased,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    ProxySHAP,
    ProxySPEX,
    RegressionFBII,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)

# Organized by category for clarity.
APPROXIMATOR_REGISTRY = {
    # Sampling-based SV methods
    "OwenSamplingSV": OwenSamplingSV,
    "StratifiedSamplingSV": StratifiedSamplingSV,
    "PermutationSamplingSV": PermutationSamplingSV,
    # Sampling-based Interaction methods
    "PermutationSamplingSII": PermutationSamplingSII,
    "PermutationSamplingSTII": PermutationSamplingSTII,
    # Monte Carlo / Stochastic methods
    "SVARM": SVARM,
    "SVARMIQ": SVARMIQ,
    "SHAPIQ": SHAPIQ,
    "UnbiasedKernelSHAP": UnbiasedKernelSHAP,
    # Regression methods
    "KernelSHAP": KernelSHAP,
    "KernelSHAPIQ": KernelSHAPIQ,
    "InconsistentKernelSHAPIQ": InconsistentKernelSHAPIQ,
    "RegressionFSII": RegressionFSII,
    "RegressionFBII": RegressionFBII,
    "kADDSHAP": kADDSHAP,
    # Proxy / Sparse methods
    "SPEX": SPEX,
    "ProxySPEX": ProxySPEX,
    "ProxySHAP": ProxySHAP,
    "MSRBiased": MSRBiased,
}


def get_approximator_class(name: str) -> type[Approximator]:
    """Return the approximator class registered under the given name.

    Args:
        name: the name of the approximator.

    Returns:
        the approximator class associated with the name.

    Raises:
        UnsupportedApproximatorError: If there is no registered approximator corresponding to the name.
    """
    try:
        return APPROXIMATOR_REGISTRY[name]
    except KeyError:
        available = APPROXIMATOR_REGISTRY.keys()
        raise UnsupportedApproximatorError(name, list(available)) from None
