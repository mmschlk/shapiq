"""Regression with Shapley interaction index (SII) approximation."""

import numpy as np

from ._base import Regression


class RegressionFSII(Regression):
    """Estimates the FSII values using KernelSHAP.

    The Faithful KernelSHAP regression is described in Tsai et al. (2023)[1]_. The method estimates
    the Faithful Shapley interaction index (FSII).

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to ``2``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAPIQ
            approximator for estimating the Shapley interaction index (SII) and the
            k-Shapley interaction index (k-SII).

    References:
        .. [1] Tsai, C.-P., Yeh, C.-K., and Ravikumar, P. (2023). In Journal of Machine Learning Research 24(94), pp. 1--42. url: http://jmlr.org/papers/v24/22-0202.html

    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            n,
            max_order,
            index="FSII",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
