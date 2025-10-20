from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import shap
from scipy.special import binom

# import random forest regressor
from xgboost import XGBRegressor

from shapiq import UnbiasedKernelSHAP
from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.games.base import Game
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions


class residualGame(Game):
    def __init__(self, n_players, game_values) -> None:
        super().__init__(n_players=n_players, normalize=False)
        self.game_values = game_values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self.game_values


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
        regression_adjustment: bool = True,
        shapley_weighted_inputs: bool = False,
        residual_estimator: Approximator = None,
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

        if residual_estimator is None:
            self.residual_estimator = UnbiasedKernelSHAP(
                n=n,
                pairing_trick=pairing_trick,
                replacement=replacement,
                random_state=random_state,
                sampling_weights=sampling_weights,
            )
        else:
            self.residual_estimator = residual_estimator

        self.regression_adjustment = regression_adjustment
        self.shapley_weighted_inputs = shapley_weighted_inputs
        if shapley_weighted_inputs:
            self.shapley_weights = np.zeros(n + 1)
            for i in range(n + 1):
                if i == 0 or i == n:
                    self.shapley_weights[i] = 0
                else:
                    self.shapley_weights[i] = 1 / (binom(n - 2, i - 1))

        # init runtime dictionary of type float
        self.runtime_last_approximate_run: dict[str, float] = {}

    def shapley_weight(self, coalition_size: int):
        return 1 / ((self.n) * binom(self.n - 1, coalition_size))

    def approximate(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        approximate_start_time = time.time()
        # sample with current budget
        self._sampler.sample(budget)
        coalitions_matrix = (
            self._sampler.coalitions_matrix
        )  # binary matrix of coalitionshttps://xgboost.readthedocs.io/en/stable/parameter.html
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = sampling_end_time - approximate_start_time

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)
        baseline_value = game_values[0]
        game_values -= baseline_value

        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )
        # fit XGBoost regression model to the game values using coalition_matrix as input

        # Initialize the regressor
        model = XGBRegressor(
            # n_estimators=100,
            # learning_rate=1,
            # max_depth=1,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # n_jobs=1,
            random_state=self._random_state,
        )

        # set weights for regression
        if self.shapley_weighted_inputs:
            coalition_weights = (
                self.shapley_weights[self._sampler.coalitions_size]
                * self._sampler.sampling_adjustment_weights
            )
        else:
            coalition_weights = np.ones_like(game_values)
        # Fit the model
        model.fit(coalitions_matrix, game_values, sample_weight=coalition_weights)
        # compute Shapley values of XGBoost
        explainer = shap.TreeExplainer(
            model, feature_perturbation="interventional", data=np.zeros((1, self.n))
        )
        treeshap_result = explainer.shap_values(np.ones(self.n))

        tree_shapley_values = np.zeros(self.n + 1)

        if self.regression_adjustment:
            tree_shapley_values[0] = baseline_value  # explainer.expected_value
            tree_shapley_values[1:] = treeshap_result
        else:
            tree_shapley_values[0] = baseline_value  # explainer.expected_value
            tree_shapley_values[1:] = treeshap_result
            tree_shapley_values[1:] += (
                1 / self.n * (game_values[1] - np.sum(tree_shapley_values[1:]))
            )

        shapley_tree = InteractionValues(
            tree_shapley_values,
            index=self.approximation_index,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=tree_shapley_values[0],
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
        )

        if self.regression_adjustment:
            # Predict on test set
            predicted_values = model.predict(coalitions_matrix)
            # compute the residual game values
            residual_values = game_values - predicted_values

            residual_game = residualGame(n_players=self.n, game_values=residual_values)
            shapley_residuals = self.residual_estimator.approximate(
                budget=budget, game=residual_game
            )
            # reset empty set and baseline
            shapley_residuals.baseline_value = 0.0
            shapley_residuals[tuple()] = 0.0
            shapley_value_estimates = shapley_tree + shapley_residuals
        else:
            shapley_value_estimates = shapley_tree

        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = (
            regression_end_time - game_evaluation_end_time
        )

        result = finalize_computed_interactions(shapley_value_estimates, target_index=self.index)

        shapiq_post_processing_end_time = time.time()
        self.runtime_last_approximate_run["shapiq_post_processing"] = (
            shapiq_post_processing_end_time - regression_end_time
        )
        self.runtime_last_approximate_run["total"] = (
            shapiq_post_processing_end_time - approximate_start_time
        )
        return result
