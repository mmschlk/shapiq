"""Configuration for the ``shapiq`` explainers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

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
    ProxySHAP,
    ProxySPEX,
    RegressionFBII,
    RegressionFSII,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.approximator.base import Approximator, ValidApproximationIndices
from shapiq.approximator.regression.base import Regression
from shapiq.game_theory.exact import ExactComputer
from shapiq.game_theory.indices import index_generalizes_bv, index_generalizes_sv

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues


class ExactApproximator(Approximator[ValidApproximationIndices]):
    """Computes the exact values with :class:`~shapiq.game_theory.exact.ExactComputer`.

    The explainers use this wrapper for ``approximator="exact"``, so that exact computation runs
    through the same interface as the approximators. It evaluates the game on all ``2**n``
    coalitions and ignores the budget.
    """

    def __init__(
        self,
        n: int,
        max_order: int = 1,
        index: ValidApproximationIndices = "SV",
        *,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ExactApproximator.

        Args:
            n: The number of players.

            max_order: The maximum interaction order. Defaults to ``1``.

            index: The interaction index to compute. Defaults to ``"SV"``.

            random_state: Not used, since the computation is deterministic. Only stored for
                compatibility with the other approximators. Defaults to ``None``.
        """
        super().__init__(n, max_order=max_order, index=index, random_state=random_state)
        if n > 16:
            warnings.warn(
                f"The number of players ({n}) is greater than 16. Computing the exact values "
                f"evaluates the game on all 2**{n} coalitions and may take a long time.",
                stacklevel=2,
            )

    def approximate(
        self,
        budget: int,  # noqa: ARG002
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Computes the exact interaction values of the game.

        Args:
            budget: Not used, since all ``2**n`` coalitions are evaluated.

            game: The game to compute the interaction values of.

            **kwargs: Additional keyword arguments (not used, only for compatibility).

        Returns:
            The exact interaction values.
        """
        # a fresh computer for every call, since the game (e.g. a refitted imputer) may change
        exact_computer = ExactComputer(game, n_players=self.n)
        interaction_values = exact_computer(self.index, self.max_order)
        interaction_values.estimated = False
        interaction_values.estimation_budget = 2**self.n
        return interaction_values


ValidApproximatorTypes = Literal[
    "spex", "montecarlo", "svarm", "permutation", "regression", "proxyshap", "proxyspex", "exact"
]
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
    "proxyshap": {
        "SII": ProxySHAP,
        "STII": ProxySHAP,
        "FSII": ProxySHAP,
        "FBII": ProxySHAP,
        "k-SII": ProxySHAP,
        "SV": ProxySHAP,
        "BV": ProxySHAP,
        "BII": ProxySHAP,
        "CHII": ProxySHAP,
    },
    "proxyspex": {
        "SII": ProxySPEX,
        "STII": ProxySPEX,
        "FSII": ProxySPEX,
        "FBII": ProxySPEX,
        "k-SII": ProxySPEX,
        "SV": ProxySPEX,
        "BV": ProxySPEX,
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
    "exact": dict.fromkeys(ExactApproximator.valid_indices, ExactApproximator),
}


def _spex_available() -> bool:
    """Return whether the real ``SPEX`` is importable (i.e. the ``sparse`` extra is installed).

    When ``shapiq[sparse]`` is missing, ``SPEX`` is bound to a placeholder class that raises an
    ``ImportError`` on instantiation; the placeholder is tagged with ``_import_error``.
    """
    return not hasattr(SPEX, "_import_error")


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
    if choose_spex(max_order=max_order, n_players=n_players) and index in SPEX.valid_indices:
        if _spex_available():
            return SPEX(
                n=n_players,
                max_order=max_order,
                index=index,
                random_state=random_state,
            )
        # SPEX is the placeholder (optional ``sparse`` extra not installed). Warn and fall
        # through to the dense approximators so ``"auto"`` still returns a working approximator.
        warnings.warn(
            "SPEX would be selected for this large problem, but the optional 'sparse' extra is "
            "not installed (pip install shapiq[sparse]). Falling back to a dense approximator, "
            "which may be slow for this number of players.",
            stacklevel=2,
        )
    if index == "SV" or (max_order == 1 and (index == "SV" or index_generalizes_sv(index))):
        return KernelSHAP(n=n_players, random_state=random_state)
    if index == "BV" or (max_order == 1 and (index == "BV" or index_generalizes_bv(index))):
        return RegressionFBII(n=n_players, max_order=1, random_state=random_state)
    if index == "FSII":
        return RegressionFSII(n=n_players, max_order=max_order, random_state=random_state)
    if index == "FBII":
        return RegressionFBII(n=n_players, max_order=max_order, random_state=random_state)
    if index in KernelSHAPIQ.valid_indices:
        return KernelSHAPIQ(
            n=n_players,
            max_order=max_order,
            index=index,
            random_state=random_state,
        )
    if index in SVARMIQ.valid_indices:
        return SVARMIQ(
            n=n_players,
            max_order=max_order,
            top_order=False,
            random_state=random_state,
            index=index,
        )
    msg = f"Could not set up approximator automatically for index {index}."
    raise ValueError(msg)


def setup_approximator(
    approximator: ValidApproximatorTypes | Approximator | Literal["auto"],
    index: ValidApproximationIndices,
    max_order: int,
    n_players: int,
    random_state: int | None = None,
) -> Approximator:
    """Set up the approximator for the explainer based on the selected index and order.

    Args:
        approximator: The approximator to be used. If ``"auto"``, the approximator is selected based
            on the index and max_order. If ``"exact"``, the values are computed exactly by
            evaluating all ``2**n_players`` coalitions.
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
    if approximator == "auto":  # if the approximator is "auto", we set it up automatically
        return setup_approximator_automatically(
            index=index,
            max_order=max_order,
            n_players=n_players,
            random_state=random_state,
        )
    if isinstance(approximator, str):
        # if the approx is a string and not "auto", we get it from the configurations and set it up
        if approximator in APPROXIMATOR_CONFIGURATIONS:
            approximator_cls: type[Approximator] = APPROXIMATOR_CONFIGURATIONS[approximator][index]
        else:
            msg = (
                f"Invalid approximator `{approximator}`. "
                f"Valid configurations are described in {APPROXIMATOR_CONFIGURATIONS}."
            )
            raise ValueError(msg)
    else:
        msg = f"Invalid approximator `{approximator}`. "
        raise TypeError(msg)

    # initialize the approximator class with params
    if issubclass(approximator_cls, Regression):
        return approximator_cls(
            n=n_players,
            max_order=max_order,
            random_state=random_state,
            index=index,
        )
    return approximator_cls(
        n=n_players,
        max_order=max_order,
        random_state=random_state,
        index=index,
    )
