"""Regression with Shapley interaction index (SII) approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .base import Regression

if TYPE_CHECKING:
    import numpy as np

ValidRegressionkADDSHAPIndices = Literal["kADD-SHAP"]


class kADDSHAP(Regression[ValidRegressionkADDSHAPIndices]):  # noqa: N801
    """The kADD-SHAP regression approximator for estimating the kADD-SHAP values.

    Estimates the kADD-SHAP values using the kADD-SHAP regression algorithm. The Algorithm is
    described in Pelegrina et al. (2023) :cite:t:`Pelegrina.2023` and is related to
    Inconsistent KernelSHAP-IQ :cite:t:`Fumagalli.2024`.

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kernelshapiq.InconsistentKernelSHAPIQ`: The
            Inconsistent KernelSHAP-IQ approximator for estimating the Shapley interaction index
            (SII) and the k-Shapley interaction index (k-SII).
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAP-IQ
            approximator for estimating the Shapley interaction index (SII) and the k-Shapley
            interaction index (k-SII).

    """

    valid_indices = ("kADD-SHAP",)
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
