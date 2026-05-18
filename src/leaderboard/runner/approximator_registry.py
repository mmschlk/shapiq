"""Approximator registry for the leaderboard runner.
"""

from shapiq.approximator import (
    Approximator,
    KernelSHAPIQ,
    PermutationSamplingSV,
    ProxySHAP,
)

from config_manager.config_exceptions import (
    ApproximatorIndexIncompatibleError,
    ApproximatorNotFoundError,
    UnsupportedApproximatorError,
)


APPROXIMATOR_REGISTRY = {
    "ProxySHAP": ProxySHAP,
    "KernelSHAPIQ": KernelSHAPIQ,
    "PermutationSamplingSV": PermutationSamplingSV,
}


def get_approximator_class(name: str) -> type[Approximator]:
    """
    Return the approximator class registered under the given name.

    Args:
        name: the name of the approximator.

    Returns:
        the approximator class associated with the name.

    Raises:
        ValueError: If there is no registered approximator corresponding to the name.
    """
    try:
        return APPROXIMATOR_REGISTRY[name]
    except KeyError:
        available = APPROXIMATOR_REGISTRY.keys()
        raise UnsupportedApproximatorError(name, list(available)) from None