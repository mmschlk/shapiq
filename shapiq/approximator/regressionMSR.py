from typing import Callable

from shapiq.approximator.base import Approximator
import numpy as np
import time
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.games.base import Game
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions

import xgboost as xgb
from xgboost import XGBRegressor

from scipy.special import binom

from shapiq.approximator.montecarlo.base import MonteCarlo
import shap

from shapiq import UnbiasedKernelSHAP


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

        self.residual_estimator = UnbiasedKernelSHAP(
            n=n,
            pairing_trick=pairing_trick,
            replacement=replacement,
            random_state=random_state,
        )

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
        )  # binary matrix of coalitions
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = (
            sampling_end_time - approximate_start_time
        )

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)

        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )
        # fit XGBoost regression model to the game values using coalition_matrix as input

        # Initialize the regressor
        model = XGBRegressor(
            # n_estimators=100,
            # learning_rate=0.05,
            # max_depth=5,
            # subsample=0.8,
            # colsample_bytree=0.8,
            random_state=self._random_state,
            n_jobs=1,
        )
        # Fit the model
        model.fit(coalitions_matrix, game_values)
        # compute Shapley values of XGBoost
        explainer = shap.TreeExplainer(
            model, feature_perturbation="interventional", data=np.zeros((1, self.n))
        )
        treeshap_result = explainer.shap_values(np.ones(self.n))

        tree_shapley_values = np.zeros(self.n + 1)
        tree_shapley_values[0] = explainer.expected_value
        tree_shapley_values[1:] = treeshap_result

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

        # Predict on test set
        predicted_values = model.predict(coalitions_matrix)
        # compute the residual game values
        residual_values = game_values - predicted_values

        residual_game = residualGame(n_players=self.n, game_values=residual_values)
        shapley_residuals = self.residual_estimator.approximate(
            budget=budget, game=residual_game
        )

        shapley_value_estimates = shapley_tree + shapley_residuals

        regression_end_time = time.time()
        self.runtime_last_approximate_run["regression"] = (
            regression_end_time - game_evaluation_end_time
        )

        result = finalize_computed_interactions(
            shapley_value_estimates, target_index=self.index
        )

        shapiq_post_processing_end_time = time.time()
        self.runtime_last_approximate_run["shapiq_post_processing"] = (
            shapiq_post_processing_end_time - regression_end_time
        )
        self.runtime_last_approximate_run["total"] = (
            shapiq_post_processing_end_time - approximate_start_time
        )
        return result
