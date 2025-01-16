"""This module contains the KernelSHAP regression approximator for estimating the SV."""

from typing import Optional

import numpy as np

from ._base import Regression


class KernelSHAP(Regression):
    """The KernelSHAP regression approximator for estimating the Shapley values.

    The KernelSHAP approximator is described in Lundberg and Lee (2017)[1]_. The method estimates
    the Shapley values by solving a weighted regression problem, where the Shapley values are the
    coefficients of the regression problem.

    Args:
        n: The number of players.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

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

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAPIQ
            approximator for estimating the Shapley interaction index (SII) and the
            k-Shapley interaction index (k-SII).
        - :class:`~shapiq.approximator.regression.fsi.RegressionFSII`: The Faithful KernelSHAP
            approximator for estimating the Faithful Shapley interaction index (FSII).

    References:
        .. [1] Lundberg, S., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In Proceedings of The 31st Conference on Neural Information Processing Systems. url: https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
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
