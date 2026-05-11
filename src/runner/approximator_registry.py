from shapiq.approximator import ProxySHAP, KernelSHAPIQ


APPROXIMATOR_REGISTRY = {
    "ProxySHAP": ProxySHAP,
    "KernelSHAPIQ": KernelSHAPIQ,
}


def get_approximator_class(name: str):
    try:
        return APPROXIMATOR_REGISTRY[name]
    except KeyError:
        available = ", ".join(APPROXIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown approximator '{name}'. Available: {available}"
        )