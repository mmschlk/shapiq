"""Approximator registry for the leaderboard runner."""

from __future__ import annotations

from config_manager.config_exceptions import (
    UnsupportedApproximatorError,
)
from shapiq.approximator import (
    Approximator,
    KernelSHAPIQ,
    PermutationSamplingSV,
    ProxySHAP,
)

APPROXIMATOR_REGISTRY = {
    "ProxySHAP": ProxySHAP,
    "KernelSHAPIQ": KernelSHAPIQ,
    "PermutationSamplingSV": PermutationSamplingSV,
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
