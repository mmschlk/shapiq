"""This module contains the KernelSHAP regression approximator for estimating the SV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .base import Regression

if TYPE_CHECKING:
    from typing import Any

    import numpy as np


class KernelSHAP(Regression):
    """The KernelSHAP regression approximator for estimating the Shapley values.

    The KernelSHAP approximator is described in Lundberg and Lee (2017)[1]_. The method estimates
    the Shapley values by solving a weighted regression problem, where the Shapley values are the
    coefficients of the regression problem.

    Example:
        >>> from shapiq.games.benchmark import DummyGame
        >>> from shapiq.approximator import KernelSHAP
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

    valid_indices: tuple[Literal["SV"]] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the KernelSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state of the estimator. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used only for compatibility).
        """
        super().__init__(
            n,
            max_order=1,
            index="SV",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
