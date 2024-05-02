"""This module provides functions to initialize Shapley interaction and SV approximators."""

from shapiq.approximator._base import Approximator

from shapiq.approximator import (
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    RegressionFSII,
    KernelSHAP,
    kADDSHAP,
    PermutationSamplingSV,
    SVARM,
    UnbiasedKernelSHAP,
    OwenSamplingSV,
    StratifiedSamplingSV,
)


def get_interaction_approximator(
    approx_name: str, n_players: int, index: str, max_order: int
) -> "Approximator":
    """Returns an initialized Shapley interaction approximator based on the name.

    Args:
        approx_name: The name of the approximator.
        n_players: The number of players.
        index: The interaction index to be used.
        max_order: The interaction order of the approximation.

    Returns:
        Approximator: The initialized approximator.
    """

    if approx_name == KernelSHAPIQ.__name__:
        return KernelSHAPIQ(n=n_players, index=index, max_order=max_order)
    if approx_name == InconsistentKernelSHAPIQ.__name__:
        return InconsistentKernelSHAPIQ(n=n_players, index=index, max_order=max_order)
    if approx_name == SHAPIQ.__name__:
        return SHAPIQ(n=n_players, index=index, max_order=max_order)
    if approx_name == SVARMIQ.__name__:
        return SVARMIQ(n=n_players, index=index, max_order=max_order)
    if approx_name == PermutationSamplingSII.__name__:
        return PermutationSamplingSII(n=n_players, index=index, max_order=max_order)
    if approx_name == PermutationSamplingSTII.__name__:
        return PermutationSamplingSTII(n=n_players, max_order=max_order)
    if approx_name == RegressionFSII.__name__:
        return RegressionFSII(n=n_players, max_order=max_order)
    raise ValueError(f"Approximator {approx_name} not found.")


def get_sv_approximator(approx_name: str, n_players: int) -> "Approximator":
    """Returns an initialized SV approximator based on the name.

    Args:
        approx_name: The name of the approximator.
        n_players: The number of players.

    Returns:
        Approximator: The initialized approximator.
    """
    if approx_name == KernelSHAP.__name__:
        return KernelSHAP(n=n_players)
    if approx_name == kADDSHAP.__name__:
        return kADDSHAP(n=n_players)
    if approx_name == PermutationSamplingSV.__name__:
        return PermutationSamplingSV(n=n_players)
    if approx_name == SVARM.__name__:
        return SVARM(n=n_players)
    if approx_name == UnbiasedKernelSHAP.__name__:
        return UnbiasedKernelSHAP(n=n_players)
    if approx_name == OwenSamplingSV.__name__:
        return OwenSamplingSV(n=n_players)
    if approx_name == StratifiedSamplingSV.__name__:
        return StratifiedSamplingSV(n=n_players)
    return get_interaction_approximator(approx_name, n_players, index="SV", max_order=1)
