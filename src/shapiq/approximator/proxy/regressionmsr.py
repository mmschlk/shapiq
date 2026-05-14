"""RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .proxyshap import ProxySHAP

if TYPE_CHECKING:
    import numpy as np

    from shapiq.typing import Model


class RegressionMSR(ProxySHAP):
    """RegressionMSR is a proxy-based approximator that uses a regression model to approximate the value function and applies the MSR adjustment method.

    The regression model is trained on a subset of the coalitions, and its predictions are adjusted using the MSR method to better match the true value function.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import RegressionMSR
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = RegressionMSR(n=5, index="SV")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SV, max_order=1, estimated=True, estimation_budget=100
        )

    """

    def __init__(
        self,
        n: int,
        index: str,
        *,
        proxy_model: Model | None = None,
        sampling_weights: np.ndarray | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the RegressionMSR approximator.

        Args:
            n: The number of players in the game.
            index: The index to be approximated. Must be a valid index for the chosen adjustment method.
            proxy_model: The regression model to be used as the proxy. If None, a default regression model will be used.
            sampling_weights: The sampling weights for the coalitions. If None, uniform weights will be used.
            pairing_trick: Whether to use the pairing trick for sampling coalitions. Default is True.
            random_state: The random state for reproducibility. Default is None.

        """
        super().__init__(
            n=n,
            max_order=1,
            index=index,
            proxy_model=proxy_model,
            adjustment="msr",
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
        )
