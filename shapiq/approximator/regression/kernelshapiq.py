"""Regression with Shapley interaction index (SII) approximation."""

import numpy as np

from ._base import Regression

AVAILABLE_INDICES_KERNELSHAPIQ = {"k-SII", "SII"}


class KernelSHAPIQ(Regression):
    """The KernelSHAP-IQ regression approximator for estimating the Shapley interaction index (SII)
    and the k-Shapley interaction index (k-SII).

    The KernelSHAP-IQ approximator is described in Fumagalli et al. (2024)[1]_. The method estimates
    the Shapley interaction index (SII) and the k-Shapley interaction index (k-SII) by solving a
    weighted regression problem, where the Shapley interaction indices are the coefficients of the
    regression problem. The estimation happens in subsequent steps, where first the first-order SII
    values are estimated. Then, the second-order SII values are estimated using the first-order
    estimations and their residuals. This process is repeated up to the desired interaction order.
    Another variant of KernelSHAP-IQ is the Inconsistent KernelSHAP-IQ[1]_, which works in a similar
    way but does not converge to the true SII values, but often provides better estimates for lower
    computational budgets.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to ``2``.
        index: The interaction index to be used. Choose from ``['k-SII', 'SII']``. Defaults to
            ``'k-SII'``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a
            coalition of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshapiq.InconsistentKernelSHAPIQ`: The
            Inconsistent KernelSHAP-IQ approximator for estimating the Shapley interaction index
            (SII) and the k-Shapley interaction index (k-SII).
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.fsi.RegressionFSII`: The Faithful KernelSHAP
            approximator for estimating the Faithful Shapley interaction index (FSII).

    References:
        .. [1] Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., and Hammer, B. (2024). KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions. In Proceedings of the 41 st International Conference on Machine Learning. url: https://openreview.net/forum?id=d5jXW2H4gg

    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
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
    """The Inconsistent KernelSHAP-IQ regression approximator for estimating the Shapley interaction
    index (SII) and the k-Shapley interaction index (k-SII).

    Inconsistent KernelSHAP-IQ[1]_ is a variant of the KernelSHAP-IQ estimator that does not
    converge to the true SII values, but often provides better estimates for lower computational
    budgets. The algorithm is also similar to kADD-SHAP[2]_. For details, we refer to
    Fumagalli et al. (2024)[1]_.

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
        - :class:`~shapiq.approximator.regression.kernelshapiq.KernelSHAPIQ`: The KernelSHAPIQ
            approximator for estimating the Shapley interaction index (SII) and the
            k-Shapley interaction index (k-SII).
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The KernelSHAP
            approximator for estimating the Shapley values.
        - :class:`~shapiq.approximator.regression.kadd_shap.kADDSHAP`: The kADD-SHAP approximator
            for estimating the kADD-SHAP values.
        - :class:`~shapiq.approximator.regression.fsi.RegressionFSII`: The Faithful KernelSHAP
            approximator for estimating the Faithful Shapley interaction index (FSII).

    References:
        .. [1] Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., and Hammer, B. (2024). KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions. In Proceedings of the 41 st International Conference on Machine Learning. url: https://openreview.net/forum?id=d5jXW2H4gg
        .. [2] Pelegrina, G. D., Duarte, L. T., Grabisch, M. (2023). A k-additive Choquet integral-based approach to approximate the SHAP values for local interpretability in machine learning. In Artificial Intelligence 325, pp. 104014. doi: https://doi.org/10.1016/j.artint.2023.104014.
    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        if index not in AVAILABLE_INDICES_KERNELSHAPIQ:
            raise ValueError(
                f"Index {index} not available for KernelSHAP-IQ. Choose from "
                f"{AVAILABLE_INDICES_KERNELSHAPIQ}."
            )
        super().__init__(
            n,
            max_order,
            index=index,
            sii_consistent=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
