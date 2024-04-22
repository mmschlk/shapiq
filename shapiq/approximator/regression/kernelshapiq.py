"""Regression with Shapley interaction index (SII) approximation."""

from typing import Optional

import numpy as np

from ._base import Regression

AVAILABLE_INDICES_KERNELSHAPIQ = {"k-SII", "SII"}


class KernelSHAPIQ(Regression):
    """Estimates the SII values using KernelSHAP-IQ.
    Algorithm described in TODO: add citation

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to `None`.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition
             of a certain size. Defaults to `None`.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str = "k-SII",
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
    ):
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                f"Index {index} not available for KernelSHAP-IQ. Choose from "
                f"{AVAILABLE_INDICES_KERNELSHAPIQ}."
            )
        super().__init__(
            n,
            max_order,
            index=index,
            sii_consistent=True,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )


class InconsistentKernelSHAPIQ(Regression):
    """Estimates the SII values using Inconsistent KernelSHAP-IQ.
    Algorithm similar to kADD-SHAP, described in TODO: add citation

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        random_state: The random state of the estimator. Defaults to `None`.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition
             of a certain size. Defaults to `None`.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str = "k-SII",
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            sii_consistent=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
