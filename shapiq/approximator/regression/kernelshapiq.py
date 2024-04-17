"""Regression with Shapley interaction index (SII) approximation."""

from typing import Optional

from ._base import Regression

AVAILABLE_INDICES_KERNELSHAPIQ = ["k-SII", "SII"]


class KernelSHAPIQ(Regression):
    """Estimates the SII values using KernelSHAP-IQ.
    Algorithm described in TODO: add citation

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to 1.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self, n: int, max_order: int, index: str = "k-SII", random_state: Optional[int] = None
    ):
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                f"Index {index} not available for KernelSHAP-IQ. Choose from "
                f"{AVAILABLE_INDICES_KERNELSHAPIQ}."
            )
        super().__init__(n, max_order, index=index, sii_consistent=True, random_state=random_state)


class InconsistentKernelSHAPIQ(Regression):
    """Estimates the SII values using Inconsistent KernelSHAP-IQ.
    Algorithm similar to kADD-SHAP, described in TODO: add citation

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        random_state: The random state of the estimator. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to 1.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self, n: int, max_order: int, index: str = "k-SII", random_state: Optional[int] = None
    ):
        super().__init__(n, max_order, index=index, sii_consistent=False, random_state=random_state)
