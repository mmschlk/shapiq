"""Configuration for the ``shapiq`` explainers."""

from __future__ import annotations

from typing import Literal

from shapiq import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    KernelSHAP,
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFBII,
    RegressionFSII,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.approximator.base import Approximator, ValidApproximationIndices
from shapiq.game_theory.indices import index_generalizes_bv, index_generalizes_sv

ValidApproximatorTypes = Literal["spex", "montecarlo", "svarm", "permutation", "regression"]
APPROXIMATOR_CONFIGURATIONS: dict[
    ValidApproximatorTypes, dict[ValidApproximationIndices, type[Approximator]]
] = {
    "regression": {
        "SII": KernelSHAPIQ,
        "FSII": RegressionFSII,
        "FBII": RegressionFBII,
        "k-SII": KernelSHAPIQ,
        "SV": KernelSHAP,
        "BV": RegressionFBII,
        "kADD-SHAP": kADDSHAP,
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
        "FBII": SHAPIQ,
        "k-SII": SHAPIQ,
        "SV": UnbiasedKernelSHAP,
        "BV": SHAPIQ,
        "BII": SHAPIQ,
        "CHII": SHAPIQ,
    },
    "svarm": {
        "SII": SVARMIQ,
        "STII": SVARMIQ,
        "FSII": SVARMIQ,
        "FBII": SVARMIQ,
        "k-SII": SVARMIQ,
        "SV": SVARM,
        "BV": SVARM,
        "BII": SVARMIQ,
        "CHII": SVARMIQ,
    },
    "spex": {
        "SII": SPEX,
        "STII": SPEX,
        "FSII": SPEX,
        "FBII": SPEX,
        "k-SII": SPEX,
        "SV": SPEX,
        "BV": SPEX,
    },
}


def choose_spex(max_order: int, n_players: int) -> bool:
    """Decide whether to use SPEX based on the number of players and max order.

    Args:
        max_order: The maximum order of interactions to be computed.
        n_players: The number of players in the game.

    Returns:
        True if SPEX should be used, False otherwise.
    """
    if max_order == 2 and n_players > 64:
        return True
    if max_order == 3 and n_players > 32:
        return True
    if max_order == 4 and n_players > 16:
        return True
    return bool(max_order > 4 and n_players > 16)


def setup_approximator_automatically(
    index: ValidApproximationIndices,
    max_order: int,
    n_players: int,
    random_state: int | None = None,
) -> Approximator:
    """Select the approximator automatically based on the index and max_order.

    Args:
        index: The index to be used for the approximator.
        max_order: The maximum order of interactions to be computed.
        n_players: The number of players in the game.
        random_state: The random state to initialize the approximator with.

    Returns:
        The selected approximator.
    """
    if choose_spex(max_order=max_order, n_players=n_players):
        return SPEX(n=n_players, max_order=max_order, index=index, random_state=random_state)
    if index == "SV" or (max_order == 1 and (index == "SV" or index_generalizes_sv(index))):
        return KernelSHAP(n=n_players, random_state=random_state)
    if index == "BV" or (max_order == 1 and (index == "BV" or index_generalizes_bv(index))):
        return RegressionFBII(n=n_players, max_order=1, random_state=random_state)
    if index == "FSII":
        return RegressionFSII(n=n_players, max_order=max_order, random_state=random_state)
    if index == "FBII":
        return RegressionFBII(n=n_players, max_order=max_order, random_state=random_state)
    if index in {"SII", "k-SII"}:
        return KernelSHAPIQ(
            n=n_players, max_order=max_order, index=index, random_state=random_state
        )
    return SVARMIQ(
        n=n_players,
        max_order=max_order,
        top_order=False,
        random_state=random_state,
        index=index,
    )


def setup_approximator(
    approximator: ValidApproximatorTypes | Approximator,
    index: ValidApproximationIndices,
    max_order: int,
    n_players: int,
    random_state: int | None = None,
) -> Approximator:
    """Set up the approximator for the explainer based on the selected index and order.

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
    # we simply return the approximator if it is already an instance of Approximator
    if isinstance(approximator, Approximator):
        return approximator

    # if the approximator is "auto", we set it up automatically
    if approximator == "auto":
        return setup_approximator_automatically(
            index=index,
            max_order=max_order,
            n_players=n_players,
            random_state=random_state,
        )

    # if the approx is a string and not "auto", we get it from the configurations and set it up
    if isinstance(approximator, str):
        if approximator in APPROXIMATOR_CONFIGURATIONS:
            approximator = APPROXIMATOR_CONFIGURATIONS[approximator][index]
        else:
            msg = (
                f"Invalid approximator `{approximator}`. "
                f"Valid configurations are described in {APPROXIMATOR_CONFIGURATIONS}."
            )
            raise ValueError(msg)

    if not issubclass(approximator, Approximator):
        msg = (
            f"Invalid approximator class `{approximator}`. "
            f"Expected a subclass of `Approximator`, but got {type(approximator)}."
        )
        raise TypeError(msg)

    # initialize the approximator class with params
    return approximator(n=n_players, max_order=max_order, random_state=random_state, index=index)
