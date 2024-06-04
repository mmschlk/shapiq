"""This module contains the KernelSHAP regression approximator for estimating the SV.
Regression with Faithful Shapley Interaction (FSII) index approximation."""

from typing import Optional

import numpy as np

from ._base import Regression


class KernelSHAP(Regression):
    """Estimates the FSII values using the weighted least square approach.

    Args:
        n: The number of players.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Attributes:
        n: The number of players.
        N: The set of players (starting from ``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For FSII, min_order is equal to ``1``.
        iteration_cost: The cost of a single iteration of the regression FSII.

    Example:
        >>> from shapiq.games.benchmark import DummyGame
        >>> from approximator import KernelSHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = KernelSHAP(n=5)
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SV, order=1, estimated=False, estimation_budget=32,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
                (3,): 0.2,
                (4,): 0.2,
            }
        )
    """

    def __init__(
        self,
        n: int,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n,
            max_order=1,
            index="SII",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
