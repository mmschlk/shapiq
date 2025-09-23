from typing import Callable

from shapiq.approximator.base import Approximator
import numpy as np

from shapiq.approximator.sampling import CoalitionSampler
from shapiq.games.base import Game
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

import xgboost as xgb
from xgboost import XGBRegressor

from scipy.special import binom

from shapiq.approximator.montecarlo.base import MonteCarlo
from shapiq.explainer import TreeExplainer


class RegressionMSR(Approximator):
    """ """

    def __init__(
        self,
        n: int,
        *,
        random_state: int | None = None,
        pairing_trick: bool = False,
        replacement: bool = True,
        sampling_weights: np.ndarray = None,
    ) -> None:
        """Initialize the MonteCarlo approximator.

        Args:
            n: The number of players.

            random_state: The random state to use for the approximation. Defaults to ``None``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.

            replacement: If ``True``, sampling is done with replacement. Defaults to ``True``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.
        """
        super().__init__(
            n,
            min_order=0,
            max_order=1,
            top_order=False,
            index="SV",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

        # initialize sampler
        if sampling_weights is None:  # init default sampling weights
            sampling_weights = self._init_sampling_weights()
        self._sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            replacement=replacement,
            random_state=self._random_state,
        )

    def shapley_weight(self, coalition_size: int):
        return 1 / ((self.n) * binom(self.n - 1, coalition_size))

    def approximate(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        # sample with current budget
        self._sampler.sample(budget)
        coalitions_matrix = (
            self._sampler.coalitions_matrix
        )  # binary matrix of coalitionshttps://xgboost.readthedocs.io/en/stable/parameter.html

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)

        # fit XGBoost regression model to the game values using coalition_matrix as input

        # Initialize the regressor
        model = XGBRegressor(
            # n_estimators=100,
            # learning_rate=0.05,
            # max_depth=5,
            # subsample=0.8,
            # colsample_bytree=0.8,
            random_state=self._random_state,
        )
        # Fit the model
        model.fit(coalitions_matrix, game_values)
        # compute Shapley values of XGBoost
        explainer = TreeExplainer(
            model, index="SV", max_order=1, feature_perturbation="tree_path_dependent"
        )
        tree_shapley_values = explainer.explain(np.ones(self.n))

        # Predict on test set
        predicted_values = model.predict(coalitions_matrix)
        # compute the residual game values
        residual_values = game_values - predicted_values

        # estimate the Shapley values of the residual game values using MSR (SHAPIQ order 1)
        shapley_values_residual = np.zeros(self.n + 1)

        shapley_weights = np.zeros((self.n + 1, 2))
        for coalition_size in range(self.n + 1):
            for intersection_size in range(2):
                shapley_weights[
                    coalition_size, intersection_size
                ] = self.shapley_weight(coalition_size - intersection_size)

        for i in range(self.n):
            # get sampling parameters
            coalitions_size = self._sampler.coalitions_size
            set_i_binary = np.zeros(self.n, dtype=int)
            set_i_binary[i] = 1
            intersections_size = np.sum(coalitions_matrix * set_i_binary, axis=1)

            weights = shapley_weights[coalitions_size, intersections_size]

            shapley_values_residual[self.interaction_lookup[(i,)]] = np.sum(
                weights * residual_values * self._sampler.sampling_adjustment_weights
            )

        baseline_value = float(game_values[self._sampler.empty_coalition_index])
        shapley_values_residual[self.interaction_lookup[()]] = baseline_value

        shapley_residuals = InteractionValues(
            shapley_values_residual,
            index=self.approximation_index,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=baseline_value,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
        )

        shapley_value_estimates = tree_shapley_values + shapley_residuals

        return finalize_computed_interactions(
            shapley_value_estimates, target_index=self.index
        )
