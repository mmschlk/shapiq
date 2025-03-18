"""This module contains the selection functionality for setting up the proper approximators for the
selected index, order in the explainer."""

from ..approximator import (
    SHAPIQ,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFSII,
    UnbiasedKernelSHAP,
)
from ..approximator._base import Approximator

APPROXIMATOR_CONFIGURATIONS = {
    "regression": {
        "SII": InconsistentKernelSHAPIQ,
        "FSII": RegressionFSII,
        "k-SII": InconsistentKernelSHAPIQ,
        "SV": KernelSHAP,
    },
    "permutation": {
        "SII": PermutationSamplingSII,
        "STII": PermutationSamplingSTII,
        "k-SII": PermutationSamplingSII,
        "SV": PermutationSamplingSV,
    },
    "montecarlo": {
        "SII": SHAPIQ,
        "STII": SHAPIQ,
        "FSII": SHAPIQ,
        "k-SII": SHAPIQ,
        "SV": UnbiasedKernelSHAP,
    },
    "svarm": {
        "SII": SVARMIQ,
        "STII": SVARMIQ,
        "FSII": SVARMIQ,
        "k-SII": SVARMIQ,
        "SV": SVARM,
    },
}

AVAILABLE_INDICES = {"SII", "k-SII", "STII", "FSII", "SV"}


# TODO: add test wich is parametarized with all possible combinations and also checks for random_state
def setup_approximator(
    approximator: str | Approximator,
    index: str,
    max_order: int,
    n_players: int,
    random_state: int | None = None,
) -> Approximator:
    """Setup the approximator for the explainer based on the selected index and order.

    Args:
        approximator: The approximator to be used. If ``"auto"``, the approximator is selected based
            on the index and max_order.
        index: The index to be used for the approximator.
        max_order: The maximum order of interactions to be computed.
        n_players: The number of players in the game.
        random_state: The random state to initialize the approximator with.

    Returns:
        The initialized approximator.
    """

    if isinstance(approximator, Approximator):  # if the approximator is already given
        return approximator

    if approximator == "auto":
        if max_order == 1:
            return KernelSHAP(n=n_players, random_state=random_state)
        elif index == "SV":
            return KernelSHAP(n=n_players, random_state=random_state)
        elif index == "FSII":
            return RegressionFSII(n=n_players, max_order=max_order, random_state=random_state)
        elif index == "SII" or index == "k-SII":
            return KernelSHAPIQ(
                n=n_players,
                max_order=max_order,
                random_state=random_state,
                index=index,
            )
        else:
            return SVARMIQ(
                n=n_players,
                max_order=max_order,
                top_order=False,
                random_state=random_state,
                index=index,
            )
    # assume that the approximator is a string
    try:
        approximator = APPROXIMATOR_CONFIGURATIONS[approximator][index]
    except KeyError:
        raise ValueError(
            f"Invalid approximator `{approximator}` or index `{index}`. "
            f"Valid configuration are described in {APPROXIMATOR_CONFIGURATIONS}."
        )
    # initialize the approximator class with params
    init_approximator = approximator(n=n_players, max_order=max_order, random_state=random_state)
    return init_approximator
