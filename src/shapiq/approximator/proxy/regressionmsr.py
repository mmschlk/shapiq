"""RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from shapiq.approximator.proxy._routes import _base_estimator
from shapiq.utils.modules import safe_isinstance
from shapiq.utils.sets import log_binom

from .proxyshap import ProxySHAP

if TYPE_CHECKING:
    from shapiq.approximator.proxy._models import ProxyLiteral, ProxyModel, ProxyModelWithHPO


ValidRegressionMSRIndices = Literal["SV", "BV"]


def _proxy_selects_linear_kernel(
    proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral,
) -> bool:
    """Check if the proxy model selects the linear-proxy kernel for the closed-form sampling weights.

    Args:
        proxy_model: The model used as the proxy. Either an estimator/HPO wrapper or a string tag (``"xgboost"``, ``"lightgbm"``, ``"tree"``, ``"linear"``); see :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` for details.

    Returns:
        ``True`` if the proxy model is a linear model or the string tag ``"linear"``, ``False`` otherwise.
    """
    if isinstance(proxy_model, str):
        return proxy_model == "linear"
    return safe_isinstance(_base_estimator(proxy_model), "sklearn.linear_model._base.LinearModel")


class RegressionMSR(ProxySHAP):
    """RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method.

    The regression model is trained on the sampled coalitions, and its predictions are adjusted
    using the MSR method to better match the true value function. The method was proposed by
    Witter et al. (2025) :cite:t:`Witter.2025` and is designed to provide more accurate
    approximations of the Shapley values, especially in cases where the value function is complex
    and non-linear.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import RegressionMSR
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionMSR(n=5, index="SV")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SV, max_order=1, min_order=0, estimated=False, estimation_budget=32,
            n_players=5, baseline_value=0.0
        )

    """

    def __init__(
        self,
        n: int,
        index: ValidRegressionMSRIndices,
        *,
        proxy_model: ProxyModel | ProxyModelWithHPO | ProxyLiteral = "xgboost",
        sampling_weights: np.ndarray | None = None,
        pairing_trick: bool = True,
        kfolds: int = 1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the RegressionMSR approximator.

        Args:
            n: The number of players in the game.
            index: The index to be approximated. Either ``"SV"`` or ``"BV"``.
            proxy_model: The model used as the proxy. Either an estimator/HPO wrapper or a
                string tag (``"xgboost"`` (default), ``"lightgbm"``, ``"tree"``, ``"linear"``);
                see :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP` for details.
            sampling_weights: The sampling weights for the coalitions, as a per-size mass of
                shape ``(n + 1,)``. If None, the sampling scheme of
                Witter et al. (2025) is used: ``binom(n, s)`` for ``"BV"``, uniform
                (Leverage SHAP) for ``"SV"`` with a linear proxy, and ``1 / sqrt(s * (n - s))``
                for ``"SV"`` otherwise.
            pairing_trick: Whether to use the pairing trick for sampling coalitions. Default is True.
            kfolds: Number of folds for the out-of-fold proxy residuals; see
                :class:`~shapiq.approximator.proxy.proxyshap.ProxySHAP`. Default is 1, which is the recommended value by Witter et al. (2025) :cite:t:`Witter.2025`.
            random_state: The random state for reproducibility. Default is None.

        """
        if index not in get_args(ValidRegressionMSRIndices):
            msg = f"Invalid index '{index}'. Must be one of {get_args(ValidRegressionMSRIndices)}."
            raise ValueError(msg)
        if sampling_weights is None:
            # The sampling kernels of :cite:t:`Witter.2025`, as per-size masses. The sampler
            # normalizes them itself, so these are written unnormalized.
            sizes = np.arange(n + 1)
            if index == "BV":
                # Closed-form binomial sampling weights for the Banzhaf value, as per-size masses.
                log_mass = np.asarray(log_binom(n, sizes))
                sampling_weights = np.exp(log_mass - np.max(log_mass))
            elif _proxy_selects_linear_kernel(proxy_model):
                # Shapley, linear proxy -- Leverage SHAP :cite:t:`Musco.2025`: uniform over sizes,
                # uniform over coalitions of the same size.
                sampling_weights = np.ones(n + 1)
            else:
                # Shapley, non-linear proxy -- MSR :cite:t:`Witter.2025`: 1 / sqrt(s * (n - s)) over sizes,
                sampling_weights = np.empty(n + 1)
                sampling_weights[1:n] = 1 / np.sqrt(sizes[1:n] * (n - sizes[1:n]))
                sampling_weights[0] = sampling_weights[n] = 1 / n
        super().__init__(
            n=n,
            max_order=1,
            index=index,
            proxy_model=proxy_model,
            apply_msr_adjustment=True,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            kfolds=kfolds,
            random_state=random_state,
        )
