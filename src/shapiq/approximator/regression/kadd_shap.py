"""Regression with Shapley interaction index (SII) approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .base import Regression

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues

ValidRegressionkADDSHAPIndices = Literal["kADD-SHAP", "SV"]


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

    valid_indices: tuple[ValidRegressionkADDSHAPIndices, ...] = ("kADD-SHAP", "SV")
    """The valid indices for this approximator."""

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        *,
        index: ValidRegressionkADDSHAPIndices = "kADD-SHAP",
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the kADD-SHAP approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The index to estimate. With ``"kADD-SHAP"`` (default), the full k-additive
                solution up to ``max_order`` is returned. With ``"SV"``, the order-1 part of the
                k-additive solution is returned as the Shapley value estimate.

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
            index=index,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
        # the SV estimate is read off the k-additive solution, so the computation always runs
        # with the kADD-SHAP weights (index="SV" would otherwise dispatch to the SII routine)
        self.approximation_index = "kADD-SHAP"

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any | None,
        **kwargs: Any,
    ) -> InteractionValues:
        """Approximates the kADD-SHAP values or, for ``index="SV"``, the Shapley values.

        Args:
            budget: The budget of the approximation.

            game: The game to be approximated.

            *args: Additional positional arguments (not used for compatibility).

            **kwargs: Additional arguments (not used for compatibility).

        Returns:
            The estimated interaction values. For ``index="SV"``, only the orders 0 and 1 of the
            k-additive solution are returned, which finalizes to the ``"SV"`` index.
        """
        interaction_values = super().approximate(budget, game, *args, **kwargs)
        if self.index == "SV":
            return interaction_values.get_n_order(min_order=0, max_order=1)
        return interaction_values
