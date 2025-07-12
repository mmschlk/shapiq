"""MoebiusConverter class for computing exact Shapley Interactions from Möbius coefficients."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable


class MoebiusConverter:
    """Computes a variety of game-theoretic concepts exactly from Möbius coefficients.

    The MöbiusConverter class is used to compute various game-theoretic concepts like Shapley
    values, Shapley interactions, Banzhaf interactions, and many more from a collection of Möbius
    coefficients (also called Möbius Interactions, MI) of a game. The MoebiusConverter is built
    with the idea that some games

    Computes exact Shapley Interactions using the (sparse) Möbius representation.
    This is much faster than exact computation, if Möbius representation is sparse.

    Attributes:
        n: The number of players.
        moebius_coefficients: The InteractionValues object containing all non-zero (sparse) Möbius
            coefficients.

    """

    def __init__(self, moebius_coefficients: InteractionValues) -> None:
        """Initialize the MoebiusConverter.

        Args:
            moebius_coefficients: An InteractionValues object containing the (potentially sparse)
                Möbius coefficients.
        """
        self.moebius_coefficients: InteractionValues = moebius_coefficients
        self.n = self.moebius_coefficients.n_players
        self._computed: dict[tuple[str, int], InteractionValues] = {}  # will store all computations
        # setup callable mapping from index to computation
        self._index_mapping: dict[str, Callable[[str, int], InteractionValues]] = {
            # shapley_interaction
            "k-SII": self.moebius_to_shapley_interaction,
            "STII": self.moebius_to_shapley_interaction,
            "FSII": self.moebius_to_shapley_interaction,
            # shapley_base_interaction
            "SII": self.moebius_to_base_interaction,
            # Shapley value
            "SV": self.moebius_to_base_interaction,
            # Banzhaf Interactions
            "FBII": self.fii_routine,
        }
        self.available_indices: set[str] = set(self._index_mapping.keys())

    def __call__(self, index: str, order: int | None = None) -> InteractionValues:
        """Calls the MoebiusConverter of the specified index or value.

        Args:
            index: The index or value to compute
            order: The order of the interaction index. If not specified the maximum order
                (i.e. ``n_players``) is used. Defaults to ``None``.

        Returns:
            The desired interaction values or generalized values.

        Raises:
            ValueError: If the index is not supported.

        """
        # sanitize input
        if order is None:
            order = self.n

        if (index, order) in self._computed:  # if index is already computed, return it
            return copy.deepcopy(self._computed[(index, order)])
        if index in self.available_indices:  # if index is supported, compute it
            computation_function = self._index_mapping[index]
            computed_index: InteractionValues = computation_function(index=index, order=order)
            self._computed[(index, order)] = computed_index
            return copy.deepcopy(computed_index)
        msg = f"Index {index} not supported."
        raise ValueError(msg)

    def base_aggregation(
        self, base_interactions: InteractionValues, order: int
    ) -> InteractionValues:
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII.

        Args:
            base_interactions: InteractionValues object containing interactions up to order ``order``.
            order: The highest order of interactions considered

        Returns:
            InteractionValues object containing transformed base_interactions

        """
        from .aggregation import aggregate_base_interaction

        aggregated_interactions = aggregate_base_interaction(base_interactions, order)
        return copy.copy(aggregated_interactions)

    def moebius_to_base_interaction(self, index: str, order: int) -> InteractionValues:
        """Computes a base interaction index, e.g. SII or BII.

        Args:
            order: The order of the explanation
            index: The base interaction index, e.g. SII, BII

        Returns:
            An InteractionValues object containing the base interactions

        """
        index_to_change_back = index
        if index == "SV":
            index = "SII"
        base_interaction_dict = {}
        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))
        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    _get_moebius_distribution_weight(moebius_size, interaction_size, order, index)
                )

        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
            strict=False,
        ):
            moebius_size = len(moebius_set)
            # for higher-order Möbius sets (size > order) distribute the value on all interactions
            for interaction in powerset(moebius_set, min_size=0, max_size=order):
                val_distributed = distribution_weights[moebius_size, len(interaction)]
                # Check if Möbius value is distributed onto this interaction
                if interaction in base_interaction_dict:
                    base_interaction_dict[interaction] += moebius_val * val_distributed
                else:
                    base_interaction_dict[interaction] = moebius_val * val_distributed

        base_interaction_values = np.zeros(len(base_interaction_dict))
        base_interaction_lookup = {}

        for i, interaction in enumerate(base_interaction_dict):
            base_interaction_values[i] = base_interaction_dict[interaction]
            base_interaction_lookup[interaction] = i

        index = index_to_change_back

        return InteractionValues(
            values=base_interaction_values,
            interaction_lookup=base_interaction_lookup,
            index=index,
            min_order=1,
            max_order=order,
            n_players=self.n,
            baseline_value=self.moebius_coefficients[()],
            estimation_budget=self.moebius_coefficients.estimation_budget,
            estimated=self.moebius_coefficients.estimated,
        )

    def stii_routine(
        self,
        order: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Computes STII. Routine to distribute the Moebius coefficients onto all STII interactions.

        The lower-order interactions are equal to their Moebius coefficients, whereas the top-order
        interactions contain the distributed higher-order interactions.

        Args:
            order: The order of the explanation
            **kwargs: Additional keyword arguments are not used.

        Returns:
            An InteractionValues object containing the STII interactions.

        """
        stii_dict = {}
        index = "STII"

        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))

        stii_dict[()] = self.moebius_coefficients[()]

        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    _get_moebius_distribution_weight(moebius_size, interaction_size, order, index)
                )

        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
            strict=False,
        ):
            moebius_size = len(moebius_set)
            if moebius_size < order:
                # For STII, interaction below size order are the Möbius coefficients
                val_distributed = distribution_weights[moebius_size, moebius_size]
                if moebius_set in stii_dict:
                    stii_dict[moebius_set] += moebius_val * val_distributed
                else:
                    stii_dict[moebius_set] = moebius_val * val_distributed
            else:
                # higher-order Möbius sets (size > order) distribute to all top-order interactions
                for interaction in powerset(moebius_set, min_size=order, max_size=order):
                    val_distributed = distribution_weights[moebius_size, len(interaction)]
                    # Check if Möbius value is distributed onto this interaction
                    if interaction in stii_dict:
                        stii_dict[interaction] += moebius_val * val_distributed
                    else:
                        stii_dict[interaction] = moebius_val * val_distributed

        stii_values = np.zeros(len(stii_dict))
        stii_lookup = {}

        for i, interaction in enumerate(stii_dict):
            stii_values[i] = stii_dict[interaction]
            stii_lookup[interaction] = i

        return InteractionValues(
            values=stii_values,
            interaction_lookup=stii_lookup,
            index=index,
            min_order=0,
            max_order=order,
            n_players=self.n,
            baseline_value=self.moebius_coefficients[()],
        )

    def fii_routine(self, index: str, order: int) -> InteractionValues:
        """Computes FII. Routine to distribute the Moebius coefficients onto all FSII interactions.

        The higher-order interactions (``size > order``) are distributed onto all FSII interactions
        (``size <= order``).

        Args:
            index: The interaction index, e.g. FSII or FBII
            order: The order of the explanation

        Returns:
            An InteractionValues object containing the FSII interactions

        """
        fii_dict = {}
        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))
        for moebius_size in range(self.n + 1):
            for interaction_size in range(min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    _get_moebius_distribution_weight(moebius_size, interaction_size, order, index)
                )

        # Handle empty set / baseline values differently for FSII and FBII
        if index == "FSII":
            # Set empty set
            fii_dict[()] = self.moebius_coefficients[()]
        if index == "FBII":
            # Add empty set for FBII via Möbius coefficients
            fii_dict[()] = self.moebius_coefficients[()]
            for moebius_set, moebius_val in zip(
                self.moebius_coefficients.interaction_lookup,
                self.moebius_coefficients.values,
                strict=False,
            ):
                moebius_size = len(moebius_set)
                if moebius_size > order:
                    fii_dict[()] += (-1) ** (order) * (
                        (1 / 2) ** moebius_size * binom(moebius_size - 1, order) * moebius_val
                    )

        # Distribute Moebius coefficients
        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
            strict=False,
        ):
            moebius_size = len(moebius_set)
            # For higher-order Moebius sets (size > order) distribute the value among all
            # contained interactions
            for interaction in powerset(moebius_set, min_size=1, max_size=order):
                val_distributed = distribution_weights[moebius_size, len(interaction)]
                # Check if Möbius value is distributed onto this interaction
                if interaction in fii_dict:
                    fii_dict[interaction] += moebius_val * val_distributed
                else:
                    fii_dict[interaction] = moebius_val * val_distributed

        fii_values = np.zeros(len(fii_dict))
        fii_lookup = {}
        for i, interaction in enumerate(fii_dict):
            fii_values[i] = fii_dict[interaction]
            fii_lookup[interaction] = i

        return InteractionValues(
            values=fii_values,
            interaction_lookup=fii_lookup,
            index=index,
            min_order=0,
            max_order=order,
            n_players=self.n,
            baseline_value=fii_dict[()],
        )

    def moebius_to_shapley_interaction(self, index: str, order: int) -> InteractionValues:
        """Converts the Möbius coefficients to Shapley Interactions up to order k.

        Args:
            index: The Shapley Interaction index, e.g. k-SII, STII, FSII
            order: The order of the explanation

        Returns:
            An InteractionValues object containing the Shapley interactions

        """
        if index == "STII":
            shapley_interactions = self.stii_routine(order)
        elif index in ["FSII", "FBII"]:
            shapley_interactions = self.fii_routine(index, order)
        elif index == "k-SII":
            # The distribution formula for k-SII is not correct. We therefore compute SII and
            # aggregate the values.
            base_interactions = self.moebius_to_base_interaction(order=order, index="SII")
            self._computed["SII"] = base_interactions
            shapley_interactions = self.base_aggregation(
                base_interactions=base_interactions,
                order=order,
            )
        else:
            msg = f"Index {index} not supported. Please choose from STII, FSII, k-SII."
            raise ValueError(msg)

        self._computed[index] = shapley_interactions
        return copy.copy(shapley_interactions)


def _get_moebius_distribution_weight(
    moebius_size: int,
    interaction_size: int,
    order: int,
    index: Literal["SII", "STII", "FSII", "FBII"],
) -> float:
    """Return the distribution weights for the Möbius coefficients onto the lower-order interaction indices.

    Args:
        moebius_size: The size of the Möbius coefficient.
        interaction_size: The size of the interaction.
        order: The order of the explanation.
        index: The interaction index.

    Returns:
        A distribution weight for the given combination.

    Raises:
        ValueError: If the index is not supported.

    """
    if index == "SII":
        return _sii_distribution_weight(moebius_size, interaction_size)
    if index == "STII":
        return _stii_distribution_weight(moebius_size, interaction_size, order)
    if index == "FSII":
        return _fsii_distribution_weight(moebius_size, interaction_size, order)
    if index == "FBII":
        return _fbii_distribution_weight(moebius_size, interaction_size, order)
    msg = f"Index {index} not supported."
    raise ValueError(msg)


def _sii_distribution_weight(moebius_size: int, interaction_size: int) -> float:
    """Return the distribution weight for SII."""
    return 1 / (moebius_size - interaction_size + 1)


def _stii_distribution_weight(moebius_size: int, interaction_size: int, order: int) -> float:
    """Return the distribution weight for STII."""
    if moebius_size <= order:
        if moebius_size == interaction_size:
            return 1
        return 0
    if interaction_size == order:
        return 1 / binom(moebius_size, moebius_size - interaction_size)
    return 0


def _fsii_distribution_weight(moebius_size: int, interaction_size: int, order: int) -> float:
    """Return the distribution weight for FSII."""
    if moebius_size <= order:
        if moebius_size == interaction_size:
            return 1
        return 0
    return (
        (-1) ** (order - interaction_size)
        * (interaction_size / (order + interaction_size))
        * (
            binom(order, interaction_size)
            * binom(moebius_size - 1, order)
            / binom(moebius_size + order - 1, order + interaction_size)
        )
    )


def _fbii_distribution_weight(moebius_size: int, interaction_size: int, order: int) -> float:
    """Return the distribution weight for FBII."""
    if moebius_size <= order:
        if moebius_size == interaction_size:
            return 1
        return 0
    return (
        (-1) ** (order - interaction_size)
        * (1 / 2) ** (moebius_size - interaction_size)
        * binom(moebius_size - interaction_size - 1, order - interaction_size)
    )
