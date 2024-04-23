"""Regression with Shapley interaction index (SII) approximation."""

from typing import Optional

import numpy as np

from ._base import Regression


class kADDSHAP(Regression):
    """Estimates the kADD-SHAP values using the kADD-SHAP algorithm.
    Algorithm described in https://doi.org/10.1016/j.artint.2023.104014

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to `1`.
        random_state: The random state of the estimator. Defaults to `None`.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition
             of a certain size. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to 1.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self,
        n: int,
        max_order: int = 1,
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            n,
            max_order,
            index="kADD-SHAP",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
