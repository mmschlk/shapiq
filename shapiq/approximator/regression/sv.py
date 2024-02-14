"""This module contains the KernelSHAP regression approximator for estimating the SV.
Regression with Faithful Shapley Interaction (FSI) index approximation."""

from typing import Optional

from ._base import Regression


class KernelSHAP(Regression):
    """Estimates the FSI values using the weighted least square approach.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        random_state: The random state of the estimator. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For FSI, min_order is equal to 1.
        iteration_cost: The cost of a single iteration of the regression FSI.

    Example:
        >>> from games import DummyGame
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

    def __init__(self, n: int, random_state: Optional[int] = None):
        super().__init__(n, max_order=1, index="SV", random_state=random_state)
