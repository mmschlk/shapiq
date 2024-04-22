"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from typing import Callable, Optional

import numpy as np
from scipy.special import binom, factorial

from shapiq.approximator._base import Approximator
from shapiq.approximator.k_sii import KShapleyMixin
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

AVAILABLE_INDICES_REGRESSION = {"k-SII", "SII", "STII", "FSII", "SV"}


class MonteCarlo(Approximator, KShapleyMixin):
    """This class is the base class for all MonteCarlo approximators, e.g. SHAP-IQ and SVARM-IQ.

    MonteCarlo approximators are based on a representation of the interaction index as a weighted
    sum over discrete derivatives. The sum is re-written and approximated using Monte Carlo
    sampling. The sum may be stratified by coalition size or by the intersection size of the
    coalition and the interaction. The standard form for approximation is based on Theorem 1 in
    (Fumagalli et al. 2023)[https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html].

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'kSII', 'STII',
            and 'FSII'.
        stratify_coalition_size: If True, then each coalition size is estimated separately
        stratify_intersection: If True, then each coalition is stratified by the intersection with
            the interaction
        top_order: If True, then only highest order interaction values are computed, e.g. required
            for FSII
        random_state: The random state to use for the approximation. Defaults to None.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        stratify_coalition_size: bool = True,
        stratify_intersection: bool = True,
        top_order: bool = False,
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: np.ndarray = None,
    ):
        if index not in AVAILABLE_INDICES_REGRESSION:
            raise ValueError(
                f"Index {index} not available for Regression Approximator. Choose from "
                f"{AVAILABLE_INDICES_REGRESSION}."
            )
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

        estimated_indicator = True
        if np.shape(coalitions_matrix)[0] >= 2**self.n:
            # If budget exceeds number of coalitions, set estimated to False
            estimated_indicator = False

        if self.index == "k-SII":
            # If index is k-SII then SII values have been approximated, now aggregate to k-SII
            baseline_value = shapley_interactions_values[0]
            shapley_interactions_values = self.transforms_sii_to_ksii(shapley_interactions_values)
            if self.min_order == 0:
                # Reset baseline value after transformation
                shapley_interactions_values[0] = baseline_value

        return self._finalize_result(
            result=shapley_interactions_values, estimated=estimated_indicator, budget=budget
        )

    def monte_carlo_routine(
        self,
        game_values: np.ndarray,
        coalitions_matrix: np.ndarray,
        index_approximation: str,
    ) -> np.ndarray[float]:
        """Approximates the Shapley interaction values using Monte Carlo sampling.

        Args:
            game_values: The game values for the coalitions. The values are of shape
                `(n_coalitions,)`.
            coalitions_matrix: The coalitions matrix used to evaluate the game. The matrix is of
                shape `(n_coalitions, n)`.
            index_approximation: The index to approximate.
        """

        # get sampling parameters
        coalitions_size = self._sampler.coalitions_size

        # get standard form weights, i.e. (-1) ** (s-|T\cap S|) * w(t,|T \cap S|), where w is the
        # discrete derivative weight and S the interactions, T the coalition  # TODO add what "s" is (order)
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
                interaction_size, coalitions_size, intersections_size
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
                game_values_centered * interaction_weights * sampling_adjustment_weights
            )

        # manually set emptyset interaction to baseline
        if self.min_order == 0:
            shapley_interaction_values[0] = empty_coalition_value

        return shapley_interaction_values

    def _intersection_stratification(self, interaction: tuple[int, ...]) -> np.ndarray:
        """TODO: Add docstring here."""
        sampling_adjustment_weights = np.ones(self._sampler.n_coalitions)
        interaction_size = len(interaction)
        # Stratify by intersection, but not by coalition size
        for intersection in powerset(interaction):
            # Stratify by intersection of coalition and interaction
            intersection_size = len(intersection)
            intersection_binary = np.zeros(self.n, dtype=int)
            intersection_binary[list(intersection)] = 1
            # Compute current stratum
            in_stratum = np.prod(self._sampler.coalitions_matrix == intersection_binary, axis=1)
            # Flag all coalitions that belong to the stratum and are sampled
            in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
            # Compute probabilities for a sample to be placed in this stratum
            stratum_probabilities = np.ones(self._sampler.n_coalitions)
            stratum_probability = 0
            # The probability is the sum over all coalition_sizes, due to law of total expectation
            for sampling_size, sampling_size_prob in enumerate(
                self._sampler.sampling_size_probabilities
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
        """TODO: Add docstring here."""
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
        """The SVARM-IQ monte carlo routine.

        # TODO: Add description here.

        Args:
            interaction: The interaction for which the sampling adjustment weights are computed.

        Returns:
            np.ndarray: The sampling adjustment weights for the SVARM-IQ routine.
        """
        sampling_adjustment_weights = np.ones(self._sampler.n_coalitions)
        interaction_size = len(interaction)
        size_strata = np.unique(self._sampler.coalitions_size)
        for intersection in powerset(interaction):
            # stratify by intersection for interaction and coalition
            intersection_size = len(intersection)
            intersection_binary = np.zeros(self.n, dtype=int)
            intersection_binary[list(intersection)] = 1
            # Compute current intersection stratum
            in_intersection_stratum = np.prod(
                self._sampler.coalitions_matrix == intersection_binary, axis=1
            )
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
                    self._sampler.coalitions_size[in_stratum_and_sampled] - intersection_size,
                ) / binom(self.n, self._sampler.coalitions_size[in_stratum_and_sampled])
                # Get sampled coalitions per stratum
                stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
                n_samples_helper = np.array([1, stratum_n_samples])
                coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
                # Set sampling adjustment weights for stratum
                sampling_adjustment_weights[in_stratum] = (
                    self._sampler.coalitions_counter[in_stratum]
                    * stratum_probabilities[in_stratum]
                    / (
                        coalitions_n_samples[in_stratum]
                        * self._sampler.coalitions_in_size_probability[in_stratum]
                    )
                )
        return sampling_adjustment_weights

    def _shapiq_routine(self) -> np.ndarray:
        """The SHAP-IQ monte carlo routine.

        # TODO: Add description here.

        Returns:
            np.ndarray: The sampling adjustment weights for the SHAP-IQ routine.
        """
        # TODO is n_samples the right variable name here? for me this sounds like the number of coalitions sampled
        n_samples = np.sum(self._sampler.coalitions_counter[self._sampler.is_coalition_sampled])
        n_samples_helper = np.array([1, n_samples])  # n_samples for sampled coalitions, else 1
        coalitions_n_samples = n_samples_helper[self._sampler.is_coalition_sampled.astype(int)]
        # Set weights by dividing through the probabilities
        sampling_adjustment_weights = self._sampler.coalitions_counter / (
            self._sampler.coalitions_size_probability
            * self._sampler.coalitions_in_size_probability
            * coalitions_n_samples
        )
        return sampling_adjustment_weights

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

    def _stii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the STII discrete derivative weight given the coalition size and interaction size.

        Representation according to https://proceedings.mlr.press/v119/sundararajan20a.html

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return self.max_order / (self.n * binom(self.n - 1, coalition_size))
        else:
            return 1.0 * (coalition_size == 0)

    def _fsii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the FSII discrete derivative weight given the coalition size and interaction
        size.

        The representation is based on the FSII representation according to Theorem 19 in this
            [paper](https://www.jmlr.org/papers/volume24/22-0202/22-0202.pdf).

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return (
                factorial(2 * self.max_order - 1)
                / factorial(self.max_order - 1) ** 2
                * factorial(self.n - coalition_size - 1)
                * factorial(coalition_size + self.max_order - 1)
                / factorial(self.n + self.max_order - 1)
            )
        else:
            raise ValueError("Lower order interactions are not supported.")

    def _weight(self, index: str, coalition_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type given coalition and interaction size.

        Args:
            index: The interaction index
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if index == "STII":
            return self._stii_weight(coalition_size, interaction_size)
        elif index == "FSII":
            return self._fsii_weight(coalition_size, interaction_size)
        else:  # SII is default for all other indices (including k-SII or SV)
            return self._sii_weight(coalition_size, interaction_size)
        # TODO: extend to BII and BV

    def _get_standard_form_weights(self, index: str) -> np.ndarray:
        """Initializes the weights for the interaction index re-written from discrete derivatives to
        standard form. Standard form according to Theorem 1 in [Fumagalli et al 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html).

        Args:
            index: The interaction index

        Returns:
            The standard form weights.
        """
        # init data structure
        weights = np.zeros((self.max_order + 1, self.n + 1, self.max_order + 1))
        for order in self._order_iterator:
            # fill with values specific to each index
            for coalition_size in range(0, self.n + 1):
                for intersection_size in range(
                    max(0, order + coalition_size - self.n), min(order, coalition_size) + 1
                ):
                    weights[order, coalition_size, intersection_size] = (-1) ** (
                        order - intersection_size
                    ) * self._weight(index, coalition_size - intersection_size, order)
        return weights
