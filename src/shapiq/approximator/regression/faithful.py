"""Regression with Shapley interaction index (SII) approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from .base import Regression

if TYPE_CHECKING:
    from typing import Any

    import numpy as np


ValidRegressionFSIIIndices = Literal["FSII", "SV"]
ValidRegressionFBIIIndices = Literal["FBII", "SV"]


class RegressionFSII(Regression):
    """Estimates the FSII values using KernelSHAP.

    The Faithful KernelSHAP regression is described in Tsai et al. (2023) [Tsa23]_. The method
    estimates the Faithful Shapley interaction index (FSII).

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAPIQ
            approximator for estimating the Shapley interaction index (SII) and the
            k-Shapley interaction index (k-SII).

    References:
        .. [Tsa23] Tsai, C.-P., Yeh, C.-K., and Ravikumar, P. (2023). In Journal of Machine Learning Research 24(94), pp. 1--42. url: http://jmlr.org/papers/v24/22-0202.html

    """

    valid_indices: tuple[ValidRegressionFSIIIndices] = tuple(get_args(ValidRegressionFSIIIndices))
    """The valid indices for the RegressionFSII approximator."""

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the RegressionFSII approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state of the estimator. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        super().__init__(
            n,
            max_order,
            index="FSII",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )


class RegressionFBII(Regression):
    """Estimates the FBII values using KernelSHAP.

    The Faithful KernelSHAP regression is described in Tsai et al. (2023) [Tsa23a]_. The method
    estimates the Faithful Banzhaf interaction index (FBII).

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAPIQ
            approximator for estimating the Shapley interaction index (SII) and the
            k-Shapley interaction index (k-SII).

    References:
        .. [Tsa23a] Tsai, C.-P., Yeh, C.-K., and Ravikumar, P. (2023). In Journal of Machine Learning Research 24(94), pp. 1--42. url: http://jmlr.org/papers/v24/22-0202.html

    """

    valid_indices: ValidRegressionFBIIIndices = tuple(get_args(ValidRegressionFBIIIndices))
    """The valid indices for the RegressionFBII approximator."""

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        *,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the RegressionFBII approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state of the estimator. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        super().__init__(
            n,
            max_order,
            index="FBII",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
