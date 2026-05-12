from shapiq.approximator import ProxySHAP, KernelSHAPIQ
from shapiq.approximator import Approximator


APPROXIMATOR_REGISTRY = {
    "ProxySHAP": ProxySHAP,
    "KernelSHAPIQ": KernelSHAPIQ,
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
        available = ", ".join(APPROXIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown approximator '{name}'. Available: {available}"
        )