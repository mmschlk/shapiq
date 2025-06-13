"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from scipy.special import binom, factorial

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable


ValidMonteCarloIndices = Literal["k-SII", "SII", "STII", "FSII", "FBII", "SV", "CHII", "BII", "BV"]


class MonteCarlo(Approximator):
    """This class is the base class for all MonteCarlo approximators, e.g. SHAP-IQ and SVARM-IQ.

    MonteCarlo approximators are based on a representation of the interaction index as a weighted
    sum over discrete derivatives. The sum is re-written and approximated using Monte Carlo
    sampling. The sum may be stratified by coalition size or by the intersection size of the
    coalition and the interaction. The standard form for approximation is based on Theorem 1 by
    `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_.
    """

    valid_indices: tuple[ValidMonteCarloIndices] = tuple(get_args(ValidMonteCarloIndices))
    """The valid indices for this approximator."""

    def __init__(
        self,
        n: int,
        max_order: int,
        index: Literal["k-SII", "SII", "STII", "FSII", "FBII", "SV", "CHII", "BII", "BV"] = "k-SII",
        *,
        stratify_coalition_size: bool = True,
        stratify_intersection: bool = True,
        top_order: bool = False,
        random_state: int | None = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray = None,
    ) -> None:
        """Initialize the MonteCarlo approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation.

            index: The interaction index to be estimated.

            stratify_coalition_size: If ``True`` (default), then each coalition size is estimated
                separately.

            stratify_intersection: If ``True`` (default), then each coalition is stratified by the
                intersection with the interaction.

            top_order: If ``True``, then only highest order interaction values are computed, e.g.
                required for ``'FSII'``. Defaults to ``False``.

            random_state: The random state to use for the approximation. Defaults to ``None``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.
        """
        if index in ["FSII", "FBII"]:
            top_order = True
        super().__init__(
            n,
            min_order=0,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
        self.stratify_coalition_size = stratify_coalition_size
        self.stratify_intersection = stratify_intersection

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """Approximates the Shapley interaction values using Monte Carlo sampling.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.

        Returns:
            The approximated Shapley interaction values.

        """
        # sample with current budget
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix  # binary matrix of coalitions

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)

        index_approximation = self.index
        if self.index == "k-SII":  # For k-SII approximate SII values and then aggregate
            index_approximation = "SII"

        # approximate the shapley interaction values using Monte Carlo
        shapley_interactions_values = self.monte_carlo_routine(
            game_values=game_values,
            coalitions_matrix=coalitions_matrix,
            index_approximation=index_approximation,
        )

        baseline_value = float(game_values[self._sampler.empty_coalition_index])

        interactions = InteractionValues(
            shapley_interactions_values,
            index=self.approximation_index,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=baseline_value,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
        )

        return finalize_computed_interactions(interactions, target_index=self.index)

    def monte_carlo_routine(
        self,
        game_values: np.ndarray,
        coalitions_matrix: np.ndarray,
        index_approximation: str,
    ) -> np.ndarray:
        """Approximates the Shapley interaction values using Monte Carlo sampling.

        Args:
            game_values: The game values for the coalitions. The values are of shape
                ``(n_coalitions,)``.
            coalitions_matrix: The coalitions matrix used to evaluate the game. The matrix is of
                shape ``(n_coalitions, n)``.
            index_approximation: The index to approximate.

        Returns:
            The approximated Shapley interaction values as a numpy array.

        """
        # get sampling parameters
        coalitions_size = self._sampler.coalitions_size

        # get standard form weights, i.e. (-1) ** (s-|T\cap S|) * w(t,|T \cap S|), where w is the
        # discrete derivative weight and S the interactions, T the coalition, s the size of the
        # interactions
        standard_form_weights = self._get_standard_form_weights(index_approximation)
        shapley_interaction_values = np.zeros(len(self.interaction_lookup))

        # mean center games for better performance
        empty_coalition_value = float(game_values[self._sampler.empty_coalition_index])
        game_values_centered = game_values - empty_coalition_value

        # compute approximations per interaction with monte carlo
        for interaction, interaction_pos in self.interaction_lookup.items():
            interaction_binary = np.zeros(self.n, dtype=int)
            interaction_binary[list(interaction)] = 1
            interaction_size = len(interaction)
            # find intersection sizes with current interaction
            intersections_size = np.sum(coalitions_matrix * interaction_binary, axis=1)
            # pre-compute all coalition weights with interaction, coalition, and intersection size
            interaction_weights = standard_form_weights[
                interaction_size,
                coalitions_size,
                intersections_size,
            ]

            # get the sampling adjustment weights depending on the stratification strategy
            if self.stratify_coalition_size and self.stratify_intersection:  # this is SVARM-IQ
                sampling_adjustment_weights = self._svarmiq_routine(interaction)
            elif not self.stratify_coalition_size and self.stratify_intersection:
                sampling_adjustment_weights = self._intersection_stratification(interaction)
            elif self.stratify_coalition_size and not self.stratify_intersection:
                sampling_adjustment_weights = self._coalition_size_stratification()
            else:  # this is SHAP-IQ
                sampling_adjustment_weights = self._shapiq_routine()

            # compute interaction approximation (using adjustment weights and interaction weights)
            shapley_interaction_values[interaction_pos] = np.sum(
                game_values_centered * interaction_weights * sampling_adjustment_weights,
            )

        # manually set emptyset interaction to baseline
        if self.min_order == 0:
            shapley_interaction_values[self.interaction_lookup[()]] = empty_coalition_value

        return shapley_interaction_values

    def _intersection_stratification(self, interaction: tuple[int, ...]) -> np.ndarray:
        """Computes the adjusted sampling weights for all coalitions and a single interactions.

        The approach uses intersection stratification over all subsets of the interaction.

        Args:
            interaction: The interaction for the intersection stratification.

        Returns:
            The adjusted sampling weights as numpy array for all coalitions.

        """
        sampling_adjustment_weights = np.ones(self._sampler.n_coalitions)
        interaction_size = len(interaction)
        interaction_binary = np.zeros(self.n, dtype=int)
        interaction_binary[list(interaction)] = 1
        # Stratify by intersection, but not by coalition size
        for intersection in powerset(interaction):
            # Stratify by intersection of coalition and interaction
            intersection_size = len(intersection)
            intersection_binary = np.zeros(self.n, dtype=int)
            intersection_binary[list(intersection)] = 1
            # Compute current stratum
            in_stratum = np.prod(
                self._sampler.coalitions_matrix * interaction_binary == intersection_binary,
                axis=1,
            ).astype(bool)
            # Flag all coalitions that belong to the stratum and are sampled
            in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
            # Compute probabilities for a sample to be placed in this stratum
            stratum_probabilities = np.ones(self._sampler.n_coalitions)
            stratum_probability = 0
            # The probability is the sum over all coalition_sizes, due to law of total expectation
            for sampling_size, sampling_size_prob in enumerate(
                self._sampler.sampling_size_probabilities,
            ):
                if sampling_size_prob > 0:
                    stratum_probability += (
                        sampling_size_prob
                        * binom(
                            self.n - interaction_size,
                            sampling_size - intersection_size,
                        )
                        / binom(self.n, sampling_size)
                    )
            stratum_probabilities[in_stratum_and_sampled] = stratum_probability
            # Get sampled coalitions per stratum
            stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
            n_samples_helper = np.array([1, stratum_n_samples])
            coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
            # Set weights for current stratum
            sampling_adjustment_weights[in_stratum] = (
                self._sampler.coalitions_counter[in_stratum]
                * stratum_probabilities[in_stratum]
                / (
                    coalitions_n_samples[in_stratum]
                    * self._sampler.coalitions_size_probability[in_stratum]
                    * self._sampler.coalitions_in_size_probability[in_stratum]
                )
            )
        return sampling_adjustment_weights

    def _coalition_size_stratification(self) -> np.ndarray:
        """Computes the adjusted sampling weights for all coalitions stratified by coalition size.

        Returns:
            The adjusted sampling weights as numpy array for all coalitions.

        """
        sampling_adjustment_weights = np.ones(self._sampler.n_coalitions)
        # Stratify by coalition size but not by intersection
        size_strata = np.unique(self._sampler.coalitions_size)
        for size_stratum in size_strata:
            # Stratify by coalition size
            in_stratum = self._sampler.coalitions_size == size_stratum
            in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
            stratum_probabilities = np.ones(self._sampler.n_coalitions)
            # set probabilities as 1 or the number of coalitions with a coalition size
            stratum_probabilities[in_stratum_and_sampled] = 1 / binom(
                self.n,
                self._sampler.coalitions_size[in_stratum_and_sampled],
            )
            # Get sampled coalitions per stratum
            stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
            n_samples_helper = np.array([1, stratum_n_samples])
            coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
            # Set sampling adjustment weights for stratum
            sampling_adjustment_weights[in_stratum] = self._sampler.coalitions_counter[
                in_stratum
            ] / (coalitions_n_samples[in_stratum] * stratum_probabilities[in_stratum])
        return sampling_adjustment_weights

    def _svarmiq_routine(self, interaction: tuple[int, ...]) -> np.ndarray:
        """Apply the SVARM-IQ routine to compute the sampling adjustment weights.

        Computes the adjusted sampling weights for the SVARM-IQ monte carlo routine.
        The method deploys both, intersection and coalition size stratification.
        For details, refer to `Kolpaczki et al. (2024) <https://doi.org/10.48550/arXiv.2401.13371>`_.

        Args:
            interaction: The interaction for the intersection stratification.

        Returns:
            The sampling adjustment weights for the SVARM-IQ routine.

        """
        sampling_adjustment_weights = np.ones(self._sampler.n_coalitions)
        interaction_size = len(interaction)
        interaction_binary = np.zeros(self.n, dtype=int)
        interaction_binary[list(interaction)] = 1
        size_strata = np.unique(self._sampler.coalitions_size)
        for intersection in powerset(interaction):
            # stratify by intersection for interaction and coalition
            intersection_size = len(intersection)
            intersection_binary = np.zeros(self.n, dtype=int)
            intersection_binary[list(intersection)] = 1
            # Compute current intersection stratum
            in_intersection_stratum = np.prod(
                self._sampler.coalitions_matrix * interaction_binary == intersection_binary,
                axis=1,
            ).astype(bool)
            for size_stratum in size_strata:
                # compute current intersection-coalition-size stratum
                in_stratum = in_intersection_stratum * (
                    self._sampler.coalitions_size == size_stratum
                )
                in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
                stratum_probabilities = np.ones(self._sampler.n_coalitions)  # default prob. are 1
                # set stratum probabilities (without size probabilities, since they cancel with
                # coalitions size probabilities): stratum probabilities are number of coalitions
                # with coalition \cap interaction = intersection divided by the number of
                # coalitions of size coalition_size
                stratum_probabilities[in_stratum_and_sampled] = binom(
                    self.n - interaction_size,
                    size_stratum - intersection_size,
                )
                # Get sampled coalitions per stratum
                stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
                n_samples_helper = np.array([1, stratum_n_samples])
                coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
                # Set sampling adjustment weights for stratum
                sampling_adjustment_weights[in_stratum] = (
                    self._sampler.coalitions_counter[in_stratum]
                    * stratum_probabilities[in_stratum]
                    / (coalitions_n_samples[in_stratum])
                )
        return sampling_adjustment_weights

    def _shapiq_routine(self) -> np.ndarray:
        """Apply the SHAP-IQ routine to compute the sampling adjustment weights.

        Computes the adjusted sampling weights for the SHAP-IQ monte carlo routine.
        The method deploys no stratification and returns the relative counts divided by the
        probabilities. For details, refer to
        `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_.

        Returns:
            The sampling adjustment weights for the SHAP-IQ routine.

        """
        # Compute the number of sampled coalitions, which are not explicitly computed in the border trick
        n_samples = np.sum(self._sampler.coalitions_counter[self._sampler.is_coalition_sampled])
        n_samples_helper = np.array([1, n_samples])  # n_samples for sampled coalitions, else 1
        coalitions_n_samples = n_samples_helper[self._sampler.is_coalition_sampled.astype(int)]
        # Set weights by dividing through the probabilities
        return self._sampler.coalitions_counter / (
            self._sampler.coalitions_size_probability
            * self._sampler.coalitions_in_size_probability
            * coalitions_n_samples
        )

    def _sii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the SII discrete derivative weight given the coalition size and interaction size.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.

        """
        return 1 / (
            (self.n - interaction_size + 1) * binom(self.n - interaction_size, coalition_size)
        )

    def _bii_weight(self, interaction_size: int) -> float:
        """Returns the BII discrete derivative weight given the coalition size and interaction size.

        Args:
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        return 1 / 2 ** (self.n - interaction_size)

    def _chii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the CHII discrete derivative weight given the coalition size and interaction size.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        try:
            return interaction_size / coalition_size
        except ZeroDivisionError:
            return 0.0

    def _stii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the STII discrete derivative weight given the coalition size and interaction size.

        For details, refer to `Dhamdhere et al. (2020) <https://doi.org/10.48550/arXiv.1902.05622>`_.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        if interaction_size == self.max_order:
            return self.max_order / (self.n * binom(self.n - 1, coalition_size))
        return 1.0 * (coalition_size == 0)

    def _fsii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the FSII discrete derivative weight given the coalition size and interaction size.

        The representation is based on the FSII representation according to Theorem 19 by
        `Tsai et al. (2023) <https://doi.org/10.48550/arXiv.2203.00870>`_.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        if interaction_size == self.max_order:
            return (
                factorial(2 * self.max_order - 1)
                / factorial(self.max_order - 1) ** 2
                * factorial(self.n - coalition_size - 1)
                * factorial(coalition_size + self.max_order - 1)
                / factorial(self.n + self.max_order - 1)
            )
        msg = f"Lower order interactions are not supported for {self.index}."
        raise ValueError(msg)

    def _fbii_weight(self, interaction_size: int) -> float:
        """Returns the FSII discrete derivative weight given the coalition size and interaction size.

        The representation is based on the FBII representation according to Theorem 17 by
        `Tsai et al. (2023) <https://doi.org/10.48550/arXiv.2203.00870>`_.

        Args:
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        if interaction_size == self.max_order:
            return 1 / 2 ** (self.n - interaction_size)
        msg = f"Lower order interactions are not supported for {self.index}."
        raise ValueError(msg)

    def _weight(self, index: str, coalition_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type given coalition and interaction size.

        Args:
            index: The interaction index
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            The weight for the interaction type.

        """
        if index == "STII":
            return self._stii_weight(coalition_size, interaction_size)
        if index == "FSII":
            return self._fsii_weight(coalition_size, interaction_size)
        if index == "FBII":
            return self._fbii_weight(interaction_size)
        if index in ["SII", "SV"]:
            return self._sii_weight(coalition_size, interaction_size)
        if index in ["BII", "BV"]:
            return self._bii_weight(interaction_size)
        if index == "CHII":
            return self._chii_weight(coalition_size, interaction_size)
        msg = f"The index {index} is not supported."
        raise ValueError(msg)

    def _get_standard_form_weights(self, index: str) -> np.ndarray:
        """Computes the standard form weights for the interaction index.

        Initializes the weights for the interaction index re-written from discrete derivatives to
        standard form. Standard form according to Theorem 1 by
        `Fumagalli et al. (2023) <https://doi.org/10.48550/arXiv.2303.01179>`_.

        Args:
            index: The interaction index

        Returns:
            The standard form weights.

        """
        # init data structure
        weights = np.zeros((self.max_order + 1, self.n + 1, self.max_order + 1))
        for order in self._order_iterator:
            # fill with values specific to each index
            for coalition_size in range(self.n + 1):
                for intersection_size in range(
                    max(0, order + coalition_size - self.n),
                    min(order, coalition_size) + 1,
                ):
                    weights[order, coalition_size, intersection_size] = (-1) ** (
                        order - intersection_size
                    ) * self._weight(index, coalition_size - intersection_size, order)
        return weights
