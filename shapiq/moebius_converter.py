import copy
from typing import Callable

import numpy as np
from scipy.special import bernoulli, binom

from .interaction_values import InteractionValues
from .utils.sets import powerset

ALL_AVAILABLE_CONCEPTS: dict[str, str] = {
    # Base Interactions
    "SII": "Shapley Interaction Index",
    # Shapley Interactions
    "k-SII": "k-Shapley Interaction Index",
    "STII": "Shapley-Taylor Interaction Index",
    "FSII": "Faithful Shapley Interaction Index",
}

ALL_AVAILABLE_INDICES: set[str] = set(ALL_AVAILABLE_CONCEPTS.keys())


class MoebiusConverter:
    """Computes exact Shapley Interactions using the (sparse) Möbius representation.
    Much faster than exact computation, if Möbius representation is sparse.

    Args:
        N: The set of players.
        moebius_coefficients: An InteractionValues objects containing the (sparse) Möbius coefficients

    Attributes:
        N: The set of players
        n: The number of players.
        moebius_coefficients: The InteractionValues object containing all non-zero (sparse) Möbius coefficients
        n_interactions: A pre-computed array containing the number of interactions up to the size of the index, e.g. n_interactions[4] is the number of interactions up to size 4.
    """

    def __init__(self, moebius_coefficients: InteractionValues):
        self.moebius_coefficients: InteractionValues = moebius_coefficients
        self.n = self.moebius_coefficients.n_players
        self.n_interactions = self._get_n_interactions()
        self._computed: dict = {}
        # setup callable mapping from index to computation
        self._index_mapping: dict[str, Callable[[], InteractionValues]] = {
            # shapley_interaction
            "k-SII": self.moebius_to_shapley_interaction,
            "STII": self.moebius_to_shapley_interaction,
            "FSII": self.moebius_to_shapley_interaction,
            # shapley_base_interaction
            "SII": self.moebius_to_base_interaction,
        }
        self.available_indices: set[str] = set(self._index_mapping.keys())
        self.available_concepts: dict[str, str] = ALL_AVAILABLE_CONCEPTS

    def __call__(self, index: str, order: int = None) -> InteractionValues:
        """Calls the MoebiusConverter of the specified index or value.

        Args:
            index: The index or value to compute
            order: The order of the interaction index. If not specified the maximum order
                (i.e. n_players) is used. Defaults to None.

        Returns:
            The desired interaction values or generalized values.

        Raises:
            ValueError: If the index is not supported.
        """
        if order is None:
            order = self.n

        if index in self._computed:
            return copy.deepcopy(self._computed[index])
        elif index in self.available_indices:
            computation_function = self._index_mapping[index]
            computed_index: InteractionValues = computation_function(index=index, order=order)
            self._computed[index] = computed_index
            return copy.deepcopy(computed_index)
        else:
            raise ValueError(f"Index {index} not supported.")

    def _get_n_interactions(self):
        """Pre-computes an array that contains the number of interactions up to the size of the index.

        Args:

        Returns:
            A numpy array containing the number of interactions up to the size of the index, e.g. n_interactions[4] is the number of interactions up to size 4.
        """
        n_interactions = np.zeros(self.n + 1, dtype=int)
        n_interaction = 0
        for i in range(self.n + 1):
            n_interaction += int(binom(self.n, i))
            n_interactions[i] = n_interaction
        return n_interactions

    def base_aggregation(self, base_interactions: InteractionValues, order: int):
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII

        Args:
            base_interactions: InteractionValues object containing interactions up to order "order"
            order: The highest order of interactions considered

        Returns:
            InteractionValues object containing transformed base_interactions
        """
        from .aggregation import aggregate_interaction_values

        aggregated_interactions = aggregate_interaction_values(base_interactions, order)
        return copy.copy(aggregated_interactions)

    def get_moebius_distribution_weight(
        self, moebius_size: int, interaction_size: int, order: int, index: str
    ):
        """Return the distribution weights for the Möbius coefficients onto the lower-order interaction indices.

        Args:
            moebius_size: The size of the Möbius coefficient
            interaction_size: The size of the interaction
            order: The order of the explanation
            index: The interaction index, e.g. SII, k-SII, FSII...

        Returns:
            A distribution weight for the given combination
        """
        if index == "SII":
            return 1 / (moebius_size - interaction_size + 1)
        if index == "STII":
            if moebius_size <= order:
                if moebius_size == interaction_size:
                    return 1
                else:
                    return 0
            else:
                if interaction_size == order:
                    return 1 / binom(moebius_size, moebius_size - interaction_size)
                else:
                    return 0
        if index == "FSII":
            if moebius_size <= order:
                if moebius_size == interaction_size:
                    return 1
                else:
                    return 0
            else:
                return (
                    (-1) ** (order - interaction_size)
                    * (interaction_size / (order + interaction_size))
                    * (
                        binom(order, interaction_size)
                        * binom(moebius_size - 1, order)
                        / binom(moebius_size + order - 1, order + interaction_size)
                    )
                )
        if index == "k-SII":
            raise NotImplementedError(
                "This does not work currently, workaround is implemented via SII + base_aggregation"
            )
            # This does not work currently, workaround is implemented via SII and base_aggregation
            if moebius_size <= order:
                return 1
            else:
                rslt = 0
                bernoulli_numbers = bernoulli(order)
                for k in range(order - interaction_size + 1):
                    rslt += (
                        binom(order - interaction_size, k)
                        * bernoulli_numbers[k]
                        / (1 + moebius_size - interaction_size - k)
                    )
                return rslt

    def moebius_to_base_interaction(self, order: int, index: str):
        """Computes a base interaction index, e.g. SII or BII

        Args:
            order: The order of the explanation
            index: The base interaction index, e.g. SII, BII

        Returns:
            An InteractionValues object containing the base interactions
        """
        base_interaction_dict = {}
        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))
        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    self.get_moebius_distribution_weight(
                        moebius_size, interaction_size, order, index
                    )
                )

        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
        ):
            moebius_size = len(moebius_set)
            # For higher-order Möbius sets (size > order) distribute the value among all contained interactions
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

        base_interactions = InteractionValues(
            values=base_interaction_values,
            interaction_lookup=base_interaction_lookup,
            index=index,
            min_order=1,
            max_order=order,
            n_players=self.n,
            baseline_value=self.moebius_coefficients[tuple()],
        )

        return base_interactions

    def stii_routine(self, order: int):
        """Computes STII. Routine to distribute the Moebius coefficients onto all interactions for STII.
        The lower-order interactions are equal to their Moebius coefficients, whereas the top-order interactions contain the distributed higher-order interactions.

        Args:
            order: The order of the explanation

        Returns:
            An InteractionValues object containing the STII interactions
        """
        stii_dict = {}
        index = "STII"

        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))

        stii_dict[tuple()] = self.moebius_coefficients[tuple()]

        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    self.get_moebius_distribution_weight(
                        moebius_size, interaction_size, order, index
                    )
                )

        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
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
                # For higher-order Möbius sets (size > order) distribute the value among all contained top-order interactions
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

        stii = InteractionValues(
            values=stii_values,
            interaction_lookup=stii_lookup,
            index=index,
            min_order=0,
            max_order=order,
            n_players=self.n,
            baseline_value=self.moebius_coefficients[tuple()],
        )
        return stii

    def fsii_routine(self, order: int):
        """Computes STII. Routine to distribute the Moebius coefficients onto all interactions for FSII.
        The higher-order interactions (size > order) are distributed onto all FSII interactions (size <= order).

        Args:
            order: The order of the explanation

        Returns:
            An InteractionValues object containing the FSII interactions
        """
        fsii_dict = {}
        index = "FSII"

        fsii_dict[tuple()] = self.moebius_coefficients[tuple()]

        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))
        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[moebius_size, interaction_size] = (
                    self.get_moebius_distribution_weight(
                        moebius_size, interaction_size, order, index
                    )
                )

        for moebius_set, moebius_val in zip(
            self.moebius_coefficients.interaction_lookup,
            self.moebius_coefficients.values,
        ):
            moebius_size = len(moebius_set)
            # For higher-order Möbius sets (size > order) distribute the value among all
            # contained interactions
            for interaction in powerset(moebius_set, min_size=1, max_size=order):
                val_distributed = distribution_weights[moebius_size, len(interaction)]
                # Check if Möbius value is distributed onto this interaction
                if interaction in fsii_dict:
                    fsii_dict[interaction] += moebius_val * val_distributed
                else:
                    fsii_dict[interaction] = moebius_val * val_distributed

        fsii_values = np.zeros(len(fsii_dict))
        fsii_lookup = {}

        for i, interaction in enumerate(fsii_dict):
            fsii_values[i] = fsii_dict[interaction]
            fsii_lookup[interaction] = i

        fsii = InteractionValues(
            values=fsii_values,
            interaction_lookup=fsii_lookup,
            index=index,
            min_order=0,
            max_order=order,
            n_players=self.n,
            baseline_value=self.moebius_coefficients[tuple()],
        )
        return fsii

    def moebius_to_shapley_interaction(self, index: str, order: int):
        """Converts the Möbius coefficients to Shapley Interactions up to order k

        Args:
            index: The Shapley Interaction index, e.g. k-SII, STII, FSII
            order: The order of the explanation

        Returns:
            An InteractionValues object containing the Shapley interactions
        """

        if index == "STII":
            shapley_interactions = self.stii_routine(order)
        elif index == "FSII":
            shapley_interactions = self.fsii_routine(order)
        elif index == "k-SII":
            # The distribution formula for k-SII is not correct. We therefore compute SII and
            # aggregate the values.
            base_interactions = self.moebius_to_base_interaction(order=order, index="SII")
            self._computed["SII"] = base_interactions
            shapley_interactions = self.base_aggregation(
                base_interactions=base_interactions, order=order
            )
        else:
            raise ValueError(f"Index {index} not supported. Please choose from STII, FSII, k-SII.")

        self._computed[index] = shapley_interactions
        return copy.copy(shapley_interactions)
