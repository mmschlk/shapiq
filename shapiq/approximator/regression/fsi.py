"""Regression with Faithful Shapley Interaction (FSI) index approximation."""

from typing import Optional

from ..k_sii import KShapleyMixin
from ._base import Regression


class RegressionFSI(Regression, KShapleyMixin):
    """Estimates the FSI values [1] using the weighted least square approach.

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

    References:
        [1]:  Tsai, C.-P., Yeh, C.-K., & Ravikumar, P. (2023). Faith-Shap: The Faithful Shapley
            Interaction Index. J. Mach. Learn. Res., 24, 94:1-94:42. Retrieved from
            http://jmlr.org/papers/v24/22-0202.html

    Example:
        >>> from games import DummyGame
        >>> from approximator import RegressionFSI
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionFSI(n=5, max_order=2)
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=FSI, order=2, estimated=False, estimation_budget=32,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
                (3,): 0.2,
                (4,): 0.2,
                (0, 1): 0,
                (0, 2): 0,
                (0, 3): 0,
                (0, 4): 0,
                (1, 2): 1.0,
                (1, 3): 0,
                (1, 4): 0,
                (2, 3): 0,
                (2, 4): 0,
                (3, 4): 0
            }
        )
    """

    def __init__(self, n: int, max_order: int, random_state: Optional[int] = None):
        super().__init__(n, max_order, index="FSI", random_state=random_state)
