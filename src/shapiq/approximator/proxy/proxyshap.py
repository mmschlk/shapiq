"""ProxySHAP approximator class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.approximator.base import Approximator
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.approximator.montecarlo.svarmiq import SVARMIQ
from shapiq.approximator.regression.kernelshapiq import KernelSHAPIQ
from shapiq.game import Game
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues


class ResidualGame(Game):
    """Residual game class for adjusting the proxy model's predictions."""

    def __init__(self, n_players: int, game_values: np.ndarray) -> None:
        """Initialize the residual game with the given values for each coalition."""
        super().__init__(n_players=n_players, normalize=False)
        self.vals = game_values

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:  # noqa: ARG002
        """Return the values of the given coalitions in the residual game.

        Args:
            coalitions: A binary matrix of shape (n_samples, n_features) where each row represents a coalition and each column represents a feature. A value of 1 indicates that the feature is included in the coalition, while a value of 0 indicates that it is not.
            Note: The coalitions are expected to be ordered in the same way as the values in self.vals, i.e., the i-th row of coalitions corresponds to the i-th entry in self.vals.

        Returns:
            A vector of shape (n_samples,) where each entry is the value of the corresponding coalition in the residual game.
        """
        return self.vals


class ProxySHAP(Approximator):
    """ProxySHAP is a proxy-based approximator that uses a regression model to approximate the value function and applies an adjustment method to better match the true value function.

    It extends RegressionMSR able to compute any-order cardinal-probabilistic indices and supports multiple adjustment methods, including MSR, SVARMIQ, and KernelSHAPIQ.

    The regression model is trained on a subset of the coalitions, and its predictions are adjusted using the selected method to better match the true value function.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import ProxySHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = ProxySHAP(n=5, max_order=2, index="SII", adjustment="svarm")
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SII, max_order=2, estimated=True, estimation_budget=100
        )
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: str = "SII",
        proxy_model: object | None = None,
        adjustment: str = "msr",
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum order of interactions to consider.
            index: Index of the instance to explain.
            proxy_model: Optional proxy model to use for approximating the value function. If None, a default XGBoost regressor will be used.
            adjustment: Method for adjusting the proxy model's predictions to better match the true value function. Options are "none" (no adjustment), "msr","svarm" (statified MSR), "kernel" (KernelSHAPIQ).
            sampling_weights: Optional array of weights for the sampling procedure. The weights must be of shape (n + 1,) and are used to determine the probability of sampling a coalition. Defaults to None.
            pairing_trick: If True, the pairing trick is applied to the sampling procedure. Defaults to True.
            random_state: The random state of the estimator. Defaults to None.
        """
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
            initialize_dict=False,
        )
        self._sampling_weights = sampling_weights
        self._pairing_trick = pairing_trick
        if proxy_model is not None:
            self.proxy_model = proxy_model
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as e:
                msg = "XGBoost is required for the default proxy model. Please install it with 'pip install xgboost' or provide a custom proxy_model."
                raise ImportError(msg) from e
            self.proxy_model = XGBRegressor(random_state=random_state)

        self.set_adjustment_method(adjustment)

    def set_adjustment_method(self, adjustment: str) -> None:
        """Select the method for adjusting the proxy model's predictions."""
        if adjustment not in {"none", "msr", "svarm", "kernel"}:
            msg = f"Invalid adjustment method: {adjustment}"
            raise ValueError(msg)
        self.adjustment = adjustment
        match adjustment:
            case "msr":
                self.adjustment_method = SHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "svarm":
                self.adjustment_method = SVARMIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "kernel":
                self.adjustment_method = KernelSHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )

    def approximate(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **_: dict
    ) -> InteractionValues:
        """Approximate the Shapley values using the proxy model and adjustment method."""
        # 1. Sample coalitions and fit proxy tree
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = game(coalitions_matrix)
        baseline_value = coalition_values[0]  # Value of the empty coalition
        coalition_values -= baseline_value  # Normalize values
        self.proxy_model.fit(  # ty: ignore[unresolved-attribute]
            coalitions_matrix, coalition_values
        )

        # 2. Compute exact index&max_order for the proxy model
        explainer = InterventionalTreeExplainer(
            self.proxy_model,
            data=np.zeros((1, self.n)),  # reference data for boolean tree
            class_index=None,
            index=self.index,
            max_order=self.max_order,
            bool_tree=True,
        )
        proxy_values = explainer.explain_function(np.ones((1, self.n)))
        proxy_interactions = InteractionValues(
            values=proxy_values.interactions,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n,
            min_order=0,
            estimated=budget >= 2**self.n,
            estimation_budget=budget,
            baseline_value=float(baseline_value),
        )
        if self.adjustment != "none":
            residual_values = (
                coalition_values
                - self.proxy_model.predict(  # ty: ignore[unresolved-attribute]
                    coalitions_matrix
                )
            )
            residual_values -= residual_values[0]  # Normalize residuals
            residual_game = ResidualGame(n_players=self.n, game_values=residual_values)
            proxy_interactions += self.adjustment_method.approximate(budget, residual_game)
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions[()] = baseline_value  # Ensure empty coalition value is correct
        return proxy_interactions
