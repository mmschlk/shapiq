"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

import numpy as np
from scipy.special import binom, factorial, gammaln

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import log_binom, powerset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import FloatVector

ValidMonteCarloIndices = Literal["k-SII", "SII", "STII", "FSII", "FBII", "SV", "CHII", "BII", "BV"]

TIndices = TypeVar("TIndices", bound=ValidMonteCarloIndices)


class MonteCarlo(Approximator[TIndices]):
    """This class is the base class for all MonteCarlo approximators, e.g. SHAP-IQ and SVARM-IQ.

    MonteCarlo approximators are based on a representation of the interaction index as a weighted
    sum over discrete derivatives. The sum is re-written and approximated using Monte Carlo
    sampling. The sum may be stratified by coalition size or by the intersection size of the
    coalition and the interaction. The standard form for approximation is based on Theorem 1 by
    :cite:t:`Fumagalli.2023`.
    """

    valid_indices: tuple[TIndices, ...] = tuple(get_args(ValidMonteCarloIndices))
    """The valid indices for this approximator."""

    def __init__(
        self,
        n: int,
        max_order: int,
        index: ValidMonteCarloIndices = "k-SII",
        *,
        stratify_coalition_size: bool = True,
        stratify_intersection: bool = True,
        top_order: bool = False,
        random_state: int | None = None,
        pairing_trick: bool = False,
        sampling_weights: FloatVector | None = None,
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
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the Shapley interaction values using Monte Carlo sampling.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.
            **kwargs: Additional keyword arguments (unused).

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

        return InteractionValues(
            shapley_interactions_values,
            index=self.approximation_index,
            n_players=self.n,
            interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=baseline_value,
            estimated=not budget >= 2**self.n,
            estimation_budget=budget,
            target_index=self.index,
        )

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
        # interactions. The (non-negative) magnitude is kept in log-space so it stays finite for
        # many players, with the sign ``(-1) ** (s - |T cap S|)`` tracked separately.
        sign_weights, log_abs_weights = self._get_standard_form_log_weights(index_approximation)
        shapley_interaction_values = np.zeros(len(self.interaction_lookup))

        # mean center games for better performance
        empty_coalition_value = float(game_values[self._sampler.empty_coalition_index])
        game_values_centered = game_values - empty_coalition_value

        # The sampling-adjustment weights only depend on the interaction when the intersection is
        # stratified (SHAP-IQ and coalition-size-only stratification are interaction-independent).
        # Hoist those out of the per-interaction loop so they are computed once.
        precomputed_log_adjustment: np.ndarray | None = None
        if not self.stratify_intersection:
            if self.stratify_coalition_size:
                precomputed_log_adjustment = self._log_coalition_size_stratification()
            else:  # this is SHAP-IQ
                precomputed_log_adjustment = self._sampler.log_sampling_adjustment_weights

        # compute approximations per interaction with monte carlo
        for interaction, interaction_pos in self.interaction_lookup.items():
            interaction_binary = np.zeros(self.n, dtype=int)
            interaction_binary[list(interaction)] = 1
            interaction_size = len(interaction)
            # find intersection sizes with current interaction
            intersections_size = np.sum(coalitions_matrix * interaction_binary, axis=1)
            # pre-compute all coalition weights with interaction, coalition, and intersection size
            interaction_sign = sign_weights[interaction_size, coalitions_size, intersections_size]
            log_interaction_weight = log_abs_weights[
                interaction_size,
                coalitions_size,
                intersections_size,
            ]

            # get the (log) sampling adjustment weights depending on the stratification strategy
            if precomputed_log_adjustment is not None:  # SHAP-IQ / coalition-size stratification
                log_sampling_adjustment_weights = precomputed_log_adjustment
            elif self.stratify_coalition_size:  # this is SVARM-IQ
                log_sampling_adjustment_weights = self._log_svarmiq_routine(interaction)
            else:  # intersection stratification only
                log_sampling_adjustment_weights = self._log_intersection_stratification(interaction)

            # Combine the discrete-derivative weight and the sampling-adjustment weight in
            # log-space and exponentiate once. For many players the weight magnitude underflows to
            # ``0`` while the adjustment overflows to ``inf``; their binomials cancel in the log
            # sum, so the per-coalition contribution stays finite (``exp(-inf) == 0`` cleanly
            # zeroes genuinely out-of-range strata) instead of producing ``0 * inf = nan``.
            coalition_terms = (
                game_values_centered
                * interaction_sign
                * np.exp(log_interaction_weight + log_sampling_adjustment_weights)
            )
            shapley_interaction_values[interaction_pos] = np.sum(coalition_terms)

        # manually set emptyset interaction to baseline
        if self.min_order == 0:
            shapley_interaction_values[self.interaction_lookup[()]] = empty_coalition_value

        return shapley_interaction_values

    def _log_intersection_stratification(self, interaction: tuple[int, ...]) -> np.ndarray:
        """Log of the intersection-stratification sampling adjustment weights, stable for large n.

        Log-space counterpart of the intersection stratification: the in-size probability is taken
        from :attr:`CoalitionSampler.coalitions_in_size_log_probability` and the stratum
        probability is accumulated from finite binomial ratios
        (``exp(log_binom - log_binom) <= 1``), so the result is finite even for many players.

        Args:
            interaction: The interaction for the intersection stratification.

        Returns:
            The log adjusted sampling weights as numpy array for all coalitions.

        """
        log_sampling_adjustment_weights = np.zeros(self._sampler.n_coalitions)
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
            in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
            # Probability for a sample to land in this stratum: sum over coalition sizes of
            # ``sampling_size_prob * binom(n - m, size - i) / binom(n, size)``, due to law of total expectation.
            # The binomial ratios are computed in log-space to avoid overflow, and the sum is accumulated in normal space, which stays finite due to the ratios being <= 1.
            stratum_probability = 0.0
            for sampling_size, sampling_size_prob in enumerate(
                self._sampler.sampling_size_probabilities,
            ):
                if sampling_size_prob > 0:
                    stratum_probability += sampling_size_prob * np.exp(
                        log_binom(self.n - interaction_size, sampling_size - intersection_size)
                        - log_binom(self.n, sampling_size)
                    )
            log_stratum_probability = (
                np.log(stratum_probability) if stratum_probability > 0 else -np.inf
            )
            log_stratum_probabilities = np.zeros(self._sampler.n_coalitions)
            log_stratum_probabilities[in_stratum_and_sampled] = log_stratum_probability
            # Get sampled coalitions per stratum
            stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
            n_samples_helper = np.array([1, stratum_n_samples])
            coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
            # Set log-weights for current stratum
            log_sampling_adjustment_weights[in_stratum] = (
                np.log(self._sampler.coalitions_counter[in_stratum])
                + log_stratum_probabilities[in_stratum]
                - np.log(coalitions_n_samples[in_stratum])
                - np.log(self._sampler.coalitions_size_probability[in_stratum])
                - self._sampler.coalitions_in_size_log_probability[in_stratum]
            )
        return log_sampling_adjustment_weights

    def _log_coalition_size_stratification(self) -> np.ndarray:
        """Log of the coalition-size-stratification adjustment weights, stable for large n.

        Returns:
            The log adjusted sampling weights as numpy array for all coalitions.

        """
        log_sampling_adjustment_weights = np.zeros(self._sampler.n_coalitions)
        size_strata = np.unique(self._sampler.coalitions_size)
        for size_stratum in size_strata:
            in_stratum = self._sampler.coalitions_size == size_stratum
            in_stratum_and_sampled = in_stratum * self._sampler.is_coalition_sampled
            # log stratum probability ``log(1 / binom(n, size)) = -log_binom(n, size)``
            log_stratum_probabilities = np.zeros(self._sampler.n_coalitions)
            log_stratum_probabilities[in_stratum_and_sampled] = -log_binom(
                self.n,
                self._sampler.coalitions_size[in_stratum_and_sampled],
            )
            stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
            n_samples_helper = np.array([1, stratum_n_samples])
            coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
            # Set log-weights for current stratum
            log_sampling_adjustment_weights[in_stratum] = (
                np.log(self._sampler.coalitions_counter[in_stratum])
                - np.log(coalitions_n_samples[in_stratum])
                - log_stratum_probabilities[in_stratum]
            )
        return log_sampling_adjustment_weights

    def _log_svarmiq_routine(self, interaction: tuple[int, ...]) -> np.ndarray:
        """Log of the SVARM-IQ sampling adjustment weights, stable for large n.

        Log-space counterpart of :func:`Kolpaczki et al. (2024)
        <https://doi.org/10.48550/arXiv.2401.13371>`'s SVARM-IQ routine, deploying both intersection
        and coalition-size stratification. The stratum probability ``binom(n - m, size - i)`` is
        kept in log-space (``log_binom``) so it does not overflow.

        Args:
            interaction: The interaction for the intersection stratification.

        Returns:
            The log adjusted sampling weights for the SVARM-IQ routine.

        """
        log_sampling_adjustment_weights = np.zeros(self._sampler.n_coalitions)
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
                # log stratum probability ``log(binom(n - m, size - i))`` (without size probabilities  as they cancel
                # with the coalition size probabilities, hence they can be omitted here)
                log_stratum_probabilities = np.zeros(self._sampler.n_coalitions)
                log_stratum_probabilities[in_stratum_and_sampled] = log_binom(
                    self.n - interaction_size,
                    size_stratum - intersection_size,
                )
                # Get sampled coalitions per stratum
                stratum_n_samples = np.sum(self._sampler.coalitions_counter[in_stratum_and_sampled])
                n_samples_helper = np.array([1, stratum_n_samples])
                coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
                # Set sampling adjustment weights for stratum
                log_sampling_adjustment_weights[in_stratum] = (
                    np.log(self._sampler.coalitions_counter[in_stratum])
                    + log_stratum_probabilities[in_stratum]
                    - np.log(coalitions_n_samples[in_stratum])
                )
        return log_sampling_adjustment_weights

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

    def _get_standard_form_log_weights(self, index: str) -> tuple[np.ndarray, np.ndarray]:
        """Sign and log-magnitude of :meth:`_get_standard_form_weights`, stable for large ``n``.

        The discrete-derivative weights are non-negative, so the standard-form weight
        ``(-1) ** (order - intersection) * w`` factors into a sign and a magnitude. The magnitude is
        returned in log-space (``log_binom``/``gammaln`` based) so it stays finite for many players
        instead of underflowing to ``0``; the caller exponentiates it together with the (log)
        sampling-adjustment weight so the binomials cancel before exponentiation.

        Args:
            index: The interaction index.

        Returns:
            A tuple ``(sign_weights, log_abs_weights)`` of arrays shaped like
            :meth:`_get_standard_form_weights`. Unfilled entries have sign ``0`` and log ``-inf``
            (i.e. weight ``0``).

        """
        shape = (self.max_order + 1, self.n + 1, self.max_order + 1)
        sign_weights = np.zeros(shape)
        log_abs_weights = np.full(shape, -np.inf)
        for order in self._order_iterator:
            for coalition_size in range(self.n + 1):
                for intersection_size in range(
                    max(0, order + coalition_size - self.n),
                    min(order, coalition_size) + 1,
                ):
                    sign_weights[order, coalition_size, intersection_size] = (-1) ** (
                        order - intersection_size
                    )
                    log_abs_weights[order, coalition_size, intersection_size] = self._log_weight(
                        index, coalition_size - intersection_size, order
                    )
        return sign_weights, log_abs_weights

    def _log_weight(self, index: str, coalition_size: int, interaction_size: int) -> float:
        """Natural logarithm of the (non-negative) discrete-derivative weight :meth:`_weight`.

        Args:
            index: The interaction index.
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            ``log(_weight(index, coalition_size, interaction_size))`` (``-inf`` when the weight is
            ``0``), computed in log-space so it stays finite for many players.

        """
        if index == "STII":
            return self._log_stii_weight(coalition_size, interaction_size)
        if index == "FSII":
            return self._log_fsii_weight(coalition_size, interaction_size)
        if index == "FBII":
            return self._log_fbii_weight(interaction_size)
        if index in ["SII", "SV"]:
            return self._log_sii_weight(coalition_size, interaction_size)
        if index in ["BII", "BV"]:
            return self._log_bii_weight(interaction_size)
        if index == "CHII":
            return self._log_chii_weight(coalition_size, interaction_size)
        msg = f"The index {index} is not supported."
        raise ValueError(msg)

    def _log_sii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Log of :meth:`_sii_weight`."""
        return float(
            -np.log(self.n - interaction_size + 1)
            - log_binom(self.n - interaction_size, coalition_size)
        )

    def _log_bii_weight(self, interaction_size: int) -> float:
        """Log of :meth:`_bii_weight`."""
        return float(-(self.n - interaction_size) * np.log(2))

    def _log_chii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Log of :meth:`_chii_weight`."""
        if coalition_size == 0:
            return -np.inf
        return float(np.log(interaction_size) - np.log(coalition_size))

    def _log_stii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Log of :meth:`_stii_weight`."""
        if interaction_size == self.max_order:
            return float(
                np.log(self.max_order) - np.log(self.n) - log_binom(self.n - 1, coalition_size)
            )
        return 0.0 if coalition_size == 0 else -np.inf

    def _log_fsii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Log of :meth:`_fsii_weight` (factorial ratio via ``gammaln``)."""
        if interaction_size == self.max_order:
            return float(
                gammaln(2 * self.max_order)
                - 2 * gammaln(self.max_order)
                + gammaln(self.n - coalition_size)
                + gammaln(coalition_size + self.max_order)
                - gammaln(self.n + self.max_order)
            )
        msg = f"Lower order interactions are not supported for {self.index}."
        raise ValueError(msg)

    def _log_fbii_weight(self, interaction_size: int) -> float:
        """Log of :meth:`_fbii_weight`."""
        if interaction_size == self.max_order:
            return float(-(self.n - interaction_size) * np.log(2))
        msg = f"Lower order interactions are not supported for {self.index}."
        raise ValueError(msg)
