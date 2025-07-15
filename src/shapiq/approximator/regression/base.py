"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable


ValidRegressionIndices = Literal["SV", "SII", "k-SII", "FSII", "kADD-SHAP", "BV", "FBII"]


class Regression(Approximator):
    """This class is the base class for all regression approximators.

    Regression approximators are based on a representation of the interaction index as a solution
    to a weighted least square problem. The objective of this optimization problem is approximated
    and then solved exactly. For the Shapley value this method is known as KernelSHAP.
    """

    valid_indices: tuple[ValidRegressionIndices] = tuple(get_args(ValidRegressionIndices))
    """The valid indices for the regression approximator. Overrides the valid indices of the base
    class Approximator."""

    def __init__(
        self,
        n: int,
        max_order: int,
        index: ValidRegressionIndices,
        *,
        sii_consistent: bool = True,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the Regression approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation.

            index: The interaction index to be estimated. Available indices are ``[FSII, SII, k-SII,
                kADD-SHAP, FBII, SV, BV]``. Defaults to ``k-SII``.

            sii_consistent: If ``True``, the KernelSHAP-IQ method is used for SII, else Inconsistent
                KernelSHAP-IQ. Defaults to ``True``.

            pairing_trick: If `True`, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state to use for the approximation. Defaults to ``None``.
        """
        super().__init__(
            n,
            min_order=0,
            max_order=max_order,
            index=index,
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
        self._bernoulli_numbers = bernoulli(self.n)  # used for SII
        # used for SII, if False, then Inconsistent KernelSHAP-IQ is used
        self._sii_consistent = sii_consistent

    def _init_kernel_weights(self, interaction_size: int) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition. The kernel
        weights depend on the size of the interactions and are set to a large number for the edges.

        Args:
            interaction_size: The size of the interaction.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).

        """
        # vector that determines the kernel weights for the regression
        weight_vector = np.zeros(shape=self.n + 1)
        if self.approximation_index == "FBII":
            for coalition_size in range(self.n + 1):
                weight_vector[coalition_size] = 1 / (2**self.n)
            return weight_vector
        if self.approximation_index in ["k-SII", "SII", "kADD-SHAP", "FSII"]:
            for coalition_size in range(self.n + 1):
                if (coalition_size < interaction_size) or (
                    coalition_size > self.n - interaction_size
                ):
                    weight_vector[coalition_size] = self._big_M
                else:
                    weight_vector[coalition_size] = 1 / (
                        (self.n - 2 * interaction_size + 1)
                        * binom(self.n - 2 * interaction_size, coalition_size - interaction_size)
                    )
            return weight_vector
        msg = f"Index {self.index} not available for Regression Approximator."
        raise ValueError(msg)  # pragma: no cover

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        *args: Any | None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """The main approximation routine for the regression approximators.

        The regression estimators approximate Shapley Interactions based on a representation as a
        weighted least-square (WLSQ) problem. The regression approximator first approximates the
        objective of the WLSQ problem by Monte Carlo sampling and then computes an exact WLSQ
        solution based on the approximated objective. This approximation is an extension of
        KernelSHAP with different variants of kernel weights and regression settings.
        For details on KernelSHAP, refer to
        `Lundberg and Lee (2017) <https://doi.org/10.48550/arXiv.1705.07874>`_.

        Args:
            budget: The budget of the approximation.

            game: The game to be approximated.

            *args: Additional positional arguments (not used for compatibility).

            **kwargs: Additional arguments (not used for compatibility).

        Returns:
            The `InteractionValues` object containing the estimated interaction values.

        """
        # initialize the kernel weights
        kernel_weights_dict = {}
        for interaction_size in range(1, self.max_order + 1):
            kernel_weights_dict[interaction_size] = self._init_kernel_weights(interaction_size)

        # get the coalitions
        self._sampler.sample(budget)

        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)

        if self.approximation_index == "SII" and self._sii_consistent:
            shapley_interactions_values = self.kernel_shap_iq_routine(
                kernel_weights_dict=kernel_weights_dict,
                game_values=game_values,
            )
        else:
            shapley_interactions_values = self.regression_routine(
                kernel_weights=kernel_weights_dict[1],
                game_values=game_values,
                index_approximation=self.approximation_index,
            )

        baseline_value = float(game_values[self._sampler.empty_coalition_index])

        interactions = InteractionValues(
            values=shapley_interactions_values,
            index=self.approximation_index,
            interaction_lookup=self.interaction_lookup,
            baseline_value=baseline_value,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
        )

        return finalize_computed_interactions(interactions, target_index=self.index)

    def kernel_shap_iq_routine(
        self,
        kernel_weights_dict: dict,
        game_values: np.ndarray,
    ) -> np.ndarray:
        """The main regression routine for the KernelSHAP-IQ approximator.

        This method solves for each interaction_size up to self.max_order separate regression
        problems iteratively. The following regression is thereby fit on the residuals of the
        previous regression problem.
        For details, refer to `Fumagalli et al. (2024) <https://doi.org/10.48550/arXiv.2405.10852>`_.

        Args:
            kernel_weights_dict: The weights of the regression problem as a dictionary per
                interaction size containing a numpy array with the regression weights per
                coalition size.
            game_values: The computed game values for the sampled coalitions.

        Returns:
            The approximated SII values of the KernelSHAP-IQ routine

        """
        coalitions_matrix = self._sampler.coalitions_matrix
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights
        coalitions_size = np.sum(coalitions_matrix, axis=1)

        # set up the storage mechanisms
        empty_coalition_value = float(game_values[coalitions_size == 0][0])
        residual_game_values = {1: copy.copy(game_values)}
        residual_game_values[1] -= empty_coalition_value
        sii_values = np.array([empty_coalition_value])

        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order,
            index="SII",
        )

        # iterate over the interaction sizes and compute the sii values via the WLSQ regression
        idx_order = 1  # start of by excluding the empty coalition
        for interaction_size in range(1, self.max_order + 1):
            # this may be further optimized by precomputing the regression matrix
            # for multiple weights currently, it is computed for each interaction size
            # however, the computation is not really a bottleneck anymore
            regression_matrix, regression_weights = _get_regression_matrices(
                kernel_weights=kernel_weights_dict[interaction_size],
                regression_coefficient_weight=regression_coefficient_weight,
                sampling_adjustment_weights=sampling_adjustment_weights,
                coalitions_matrix=coalitions_matrix,
                max_order=self.max_order,
                n=self.n,
            )
            n_interactions = int(binom(self.n, interaction_size))
            regression_matrix = regression_matrix[:, idx_order : idx_order + n_interactions]
            idx_order += n_interactions

            if interaction_size <= 2:
                # get \phi_i via solving the regression problem
                sii_values_current_size = solve_regression(
                    X=regression_matrix,
                    y=residual_game_values[interaction_size],
                    kernel_weights=regression_weights,
                )
            else:
                # for order > 2 use ground truth weights for sizes < interaction_size and > n -
                # interaction_size
                ground_truth_weights_indicator = (coalitions_size < interaction_size) + (
                    coalitions_size > self.n - interaction_size
                )
                weights_from_ground_truth = self._get_ground_truth_sii_weights(
                    coalitions_matrix[ground_truth_weights_indicator],
                    interaction_size,
                )
                sii_values_current_size_minus = np.dot(
                    weights_from_ground_truth.T,
                    residual_game_values[interaction_size][ground_truth_weights_indicator],
                )

                # for interaction_size <= coalition size <= n-interaction_size solve the WLSQ
                game_values_plus = copy.deepcopy(residual_game_values[interaction_size])
                game_values_plus[ground_truth_weights_indicator] = 0

                # get \phi_i via solving the regression problem
                sii_values_current_size_plus = solve_regression(
                    X=regression_matrix,
                    y=game_values_plus,
                    kernel_weights=regression_weights,
                )

                sii_values_current_size = (
                    sii_values_current_size_minus + sii_values_current_size_plus
                )

            approximations = np.dot(regression_matrix, sii_values_current_size)
            sii_values = np.hstack((sii_values, sii_values_current_size))
            residual_game_values[interaction_size + 1] = (
                residual_game_values[interaction_size] - approximations
            )

        return sii_values

    def regression_routine(
        self,
        kernel_weights: np.ndarray,
        game_values: np.ndarray,
        index_approximation: str,
    ) -> np.ndarray[float]:
        """The main regression routine for the regression approximators.

        Solves a weighted least-square problem based on the representation of the target index.
        First, approximates the objective of the regression problem and then solves the regression
        problem using the approximated objective. Extends on KernelSHAP in different forms and
        computes all interactions using a single regression problem.

        Args:
            kernel_weights: An array of the regression weights associated with each coalition size.
            game_values: The computed game values for the sampled coalitions.
            index_approximation: The current index that is approximated.

        Returns:
            A numpy array of the approximated interaction values.

        """
        coalitions_matrix = self._sampler.coalitions_matrix
        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights
        empty_coalition_value = float(game_values[np.sum(coalitions_matrix, axis=1) == 0][0])
        regression_response = game_values - empty_coalition_value
        regression_coefficient_weight = self._get_regression_coefficient_weights(
            max_order=self.max_order,
            index=index_approximation,
        )

        regression_matrix, regression_weights = _get_regression_matrices(
            kernel_weights=kernel_weights,
            regression_coefficient_weight=regression_coefficient_weight,
            sampling_adjustment_weights=sampling_adjustment_weights,
            coalitions_matrix=coalitions_matrix,
            max_order=self.max_order,
            n=self.n,
        )

        shapley_interactions_values = solve_regression(
            X=regression_matrix,
            y=regression_response,
            kernel_weights=regression_weights,
        )

        if index_approximation in ["kADD-SHAP", "FBII"]:
            shapley_interactions_values[0] += empty_coalition_value
        else:
            shapley_interactions_values[0] = empty_coalition_value

        return shapley_interactions_values

    def _get_regression_coefficient_weights(self, max_order: int, index: str) -> np.ndarray:
        """Get the regression coefficient weights.

        Pre-computes the regression coefficient weights based on the index and the max_order.
        Bernoulli weights for SII and kADD-SHAP. Binary weights for FSI.

        Args:
                max_order: The highest interaction size considered
                index: The interaction index

        Returns:
               An array of the regression coefficient weights.

        """
        if index == "SII":
            return self._get_bernoulli_weights(max_order=max_order)
        if index == "kADD-SHAP":
            return self._get_kadd_weights(max_order=max_order)
        if index in ["FSII", "FBII"]:
            # Default weights for FSI
            weights = np.zeros((max_order + 1, max_order + 1))
            # Including the zero interaction size unharmful for FSII, due to \mu(\emptyset)= \infty thus \phi(\emptyset)=0
            for interaction_size in range(max_order + 1):
                # 1 if interaction is fully contained, else 0.
                weights[interaction_size, interaction_size] = 1
            return weights
        msg = f"Index {index} not valid in Regression Approximator."
        raise ValueError(msg)  # pragma: no cover

    def _get_bernoulli_weights(self, max_order: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for a given max_order.

        Args:
            max_order: The highest interaction size considered

        Returns:
            An array of the (regression coefficient) Bernoulli weights for all interaction sizes up
                to the max_order.

        """
        bernoulli_weights = np.zeros((max_order + 1, max_order + 1))
        for interaction_size in range(1, max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._bernoulli_weights(
                    intersection_size,
                    interaction_size,
                )
        return bernoulli_weights

    def _get_kadd_weights(self, max_order: int) -> np.ndarray:
        """Pre-computes and array of Bernoulli weights for a given max_order.

        Args:
            max_order: The highest interaction size considered

        Returns:
            An array of the (regression coefficient) Bernoulli weights for all interaction sizes up
                to the max_order.

        """
        bernoulli_weights = np.zeros((max_order + 1, max_order + 1))
        for interaction_size in range(max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._kadd_weights(
                    intersection_size,
                    interaction_size,
                )
        return bernoulli_weights

    def _bernoulli_weights(self, intersection_size: int, interaction_size: int) -> float:
        """Computes the weights of SII in the k-additive approximation.

        The weights are based on the size of the interaction and
        the size of the intersection of the interaction and the coalition.

        Args:
            intersection_size: The size of the intersection
            interaction_size: The size of the interaction

        Returns:
            The weight of SII in the k-additive approximation.

        """
        weight = 0
        for sum_index in range(1, intersection_size + 1):
            weight += (
                binom(intersection_size, sum_index)
                * self._bernoulli_numbers[interaction_size - sum_index]
            )
        return weight

    def _kadd_weights(self, intersection_size: int, interaction_size: int) -> float:
        """Get the k-additive weights.

        Computes the weights of SII in the k-additive approximation. The weights are based on the
        size of the interaction and the size of the intersection of the interaction and the
        coalition.

        Note:
            Similar to ``_bernoulli_weights`` but sum ranges from zero.

        Args:
            intersection_size: The size of the intersection
            interaction_size: The size of the interaction

        Returns:
            The weight of SII in the k-additive approximation.

        """
        weight = 0
        for sum_index in range(intersection_size + 1):
            weight += (
                binom(intersection_size, sum_index)
                * self._bernoulli_numbers[interaction_size - sum_index]
            )
        return weight

    def _get_ground_truth_sii_weights(
        self, coalitions: np.ndarray, interaction_size: int
    ) -> np.ndarray:
        """Returns the ground truth SII weights for the coalitions per interaction.

        Args:
            coalitions: A binary coalition matrix for which the ground truth weights should be
                computed
            interaction_size: The size of the interaction

        Returns:
            An array of weights with weights per coalition and per interaction

        """
        coalition_sizes = np.unique(np.sum(coalitions, axis=1))

        ground_truth_sii_weights = np.zeros((len(coalition_sizes), interaction_size + 1))

        # Pre-compute weights
        for coalition_size_pos, coalition_size in enumerate(coalition_sizes):
            for intersection_size in range(
                max(0, coalition_size + interaction_size - self.n),
                min(coalition_size, interaction_size) + 1,
            ):
                ground_truth_sii_weights[coalition_size_pos, intersection_size] = (
                    self._ground_truth_sii_weight(
                        coalition_size,
                        interaction_size,
                        intersection_size,
                    )
                )

        # Compute ground truth weights
        coalitions_sii_weights = np.zeros(
            (np.shape(coalitions)[0], int(binom(self.n, interaction_size))),
            dtype=float,
        )

        for coalition_pos, coalition in enumerate(coalitions):
            coalition_size = np.sum(coalition)
            for interaction_pos, interaction in enumerate(
                powerset(
                    self._grand_coalition_set,
                    min_size=interaction_size,
                    max_size=interaction_size,
                ),
            ):
                intersection_size = np.sum(coalition[list(interaction)])
                coalitions_sii_weights[coalition_pos, interaction_pos] = ground_truth_sii_weights[
                    list(coalition_sizes).index(coalition_size),
                    intersection_size,
                ]

        return coalitions_sii_weights

    def _ground_truth_sii_weight(
        self,
        coalition_size: int,
        interaction_size: int,
        intersection_size: int,
    ) -> float:
        """Get the SII weights.

        Returns the ground truth SII weight for a given coalition size, interaction size and its
        intersection size.

        Args:
            coalition_size: The size of the coalition.
            interaction_size: The size of the interaction.
            intersection_size: The size of the intersection of the coalition and the interaction.

        Returns:
            The ground truth SII weight

        """
        return (-1) ** (interaction_size - intersection_size) / (
            (self.n - interaction_size + 1)
            * binom(self.n - interaction_size, coalition_size - intersection_size)
        )


def _get_regression_matrices(
    kernel_weights: np.ndarray,
    regression_coefficient_weight: np.ndarray,
    sampling_adjustment_weights: np.ndarray,
    coalitions_matrix: np.ndarray,
    max_order: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs the regression matrix and regression weights for the regression problem.

    Args:
        kernel_weights: The weights for the regression problem for each coalition.
        regression_coefficient_weight: The weights for the regression coefficients.
        sampling_adjustment_weights: The weights for the sampling procedure.
        coalitions_matrix: The coalitions matrix.
        max_order: The maximum order of the approximation.
        n: The number of players.

    Returns:
        A tuple containing the regression matrix and the regression weights.

    """
    # Step 1: Precompute interactions masks
    interaction_masks = []
    interaction_sizes = []

    for interaction_size in range(max_order + 1):
        for interaction in powerset(range(n), min_size=interaction_size, max_size=interaction_size):
            mask = np.zeros(n, dtype=int)
            mask[list(interaction)] = 1
            interaction_masks.append(mask)
            interaction_sizes.append(interaction_size)

    interaction_masks = np.array(interaction_masks).T  # Shape: (n, n_interactions)
    interaction_sizes = np.array(interaction_sizes)  # Shape: (n_interactions,)

    # compute intersection sizes via matrix multiplication
    intersection_sizes = coalitions_matrix @ interaction_masks

    # use intersection sizes and interaction sizes to index regression_coefficient_weight
    regression_matrix = regression_coefficient_weight[interaction_sizes, intersection_sizes]

    # compute regression weights
    regression_weights = kernel_weights[np.sum(coalitions_matrix, axis=1)]

    # adjust regression weights with the sampling weights
    regression_weights *= sampling_adjustment_weights

    return regression_matrix, regression_weights


def solve_regression(X: np.ndarray, y: np.ndarray, kernel_weights: np.ndarray) -> np.ndarray:
    """Solves the Shapley regression problem.

    Solves the regression problem using the weighted least squares method. Returns all approximated
    interactions.

    Args:
        X: The regression matrix of shape ``[n_coalitions, n_interactions]``.
        y: The response vector for each coalition of shape ``[n_coalitions]``.
        kernel_weights: The weights for the regression problem for each coalition.

    Returns:
        The solution to the regression problem.

    """
    try:
        # try solving via solve function
        WX = kernel_weights[:, np.newaxis] * X
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.linalg.solve(X.T @ WX, WX.T @ y)
    except np.linalg.LinAlgError:
        # solve WLSQ via lstsq function and throw warning
        W_sqrt = np.sqrt(kernel_weights)
        X = W_sqrt[:, np.newaxis] * X
        y = W_sqrt * y
        phi = np.linalg.lstsq(X, y, rcond=None)[0]
    return phi
