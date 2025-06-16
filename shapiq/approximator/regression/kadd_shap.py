"""Regression with Shapley interaction index (SII) approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .base import Regression

if TYPE_CHECKING:
    import numpy as np


class kADDSHAP(Regression):  # noqa: N801
    """The kADD-SHAP regression approximator for estimating the kADD-SHAP values.

    Estimates the kADD-SHAP values using the kADD-SHAP regression algorithm. The Algorithm is
    described in Pelegrina et al. (2023) [1]_ and is related to Inconsistent KernelSHAP-IQ [2]_.

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kernelshapiq.InconsistentKernelSHAPIQ`: The
            Inconsistent KernelSHAP-IQ approximator for estimating the Shapley interaction index
            (SII) and the k-Shapley interaction index (k-SII).
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAP-IQ
            approximator for estimating the Shapley interaction index (SII) and the k-Shapley
            interaction index (k-SII).

    References:
        .. [1] Pelegrina, G. D., Duarte, L. T., Grabisch, M. (2023). A k-additive Choquet integral-based approach to approximate the SHAP values for local interpretability in machine learning. In Artificial Intelligence 325, pp. 104014. doi: https://doi.org/10.1016/j.artint.2023.104014.
        .. [2] Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., and Hammer, B. (2024). KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions. In Proceedings of the 41 st International Conference on Machine Learning. url: https://openreview.net/forum?id=d5jXW2H4gg

    """

    valid_indices: Literal["kADD-SHAP"] = ("kADD-SHAP",)
    """The valid indices for this approximator."""

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
        """Initialize the kADD-SHAP approximator.

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
            index="kADD-SHAP",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
