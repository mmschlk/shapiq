"""This module contains the exact computation mechanisms for a plethora of game theoretic concepts
like interaction indices or generalized values."""

import copy
from typing import Callable, Union

import numpy as np
from scipy.special import bernoulli, binom

from shapiq import powerset
from shapiq.interaction_values import InteractionValues

__all__ = ["ExactComputer", "get_bernoulli_weights"]


ALL_AVAILABLE_CONCEPTS: dict[str, str] = {
    "Moebius": "Moebius Transformation",
    # Base Interactions
    "SII": "Shapley Interaction Index",
    "BII": "Banzhaf Interaction Index",
    "CHII": "Chaining Interaction Index",
    # Base Generalized Values
    "SGV": "Shapley Generalized Value",
    "BGV": "Banzhaf Generalized Value",
    "CHGV": "Chaining Generalized Value",
    # Shapley Interactions
    "k-SII": "k-Shapley Interaction Index",
    "STII": "Shapley-Taylor Interaction Index",
    "FSII": "Faithful Shapley Interaction Index",
    "kADD-SHAP": "k-additive Shapley Values",
    # Probabilistic Values
    "SV": "Shapley Value",
    "BV": "Banzhaf Value",
    # Shapley Generalized Values
    "JointSV": "JointSV",
}

ALL_AVAILABLE_INDICES: set[str] = set(ALL_AVAILABLE_CONCEPTS.keys())


class ExactComputer:
    """Computes exact Shapley Interactions for specified game by evaluating the powerset of all
        $2^n$ coalitions.

    The ExactComputer class computes a variety of game theoretic concepts like interaction indices
    or generalized values. Currently, the following indices and values are supported:
        - The Moebius Transform (Moebius)
        - Shapley Interaction Index (SII)

    Args:
        n_players: The number of players in the game.
        game_fun: A callable game that takes a binary matrix of shape (n_coalitions, n_players)
            and returns a numpy array of shape (n_coalitions,) containing the game values.

    Attributes:
        n: The number of players.
        game_fun: The callable game
        baseline_value: The baseline value, i.e. the emptyset prediction
        game_values: A numpy array containing the game evaluations of all subsets
        coalition_lookup: A dictionary containing the index of every coalition in game_values
    """

    def __init__(
        self,
        n_players: int,
        game_fun: Callable[[np.ndarray], np.ndarray[float]],
    ) -> None:
        # set parameter attributes
        self.n: int = n_players
        self.game_fun = game_fun

        # set object attributes
        self._grand_coalition_tuple: tuple[int] = tuple(range(self.n))
        self._grand_coalition_set: set[int] = set(self._grand_coalition_tuple)
        self._big_M: float = 10e7
        self._n_interactions: np.ndarray = self.get_n_interactions(self.n)
        self._computed = {}  # will store all computed indices

        # evaluate the game on the powerset
        computed_game = self.compute_game_values(game_fun)
        self.baseline_value: float = computed_game[0]
        self.game_values: np.ndarray[float] = computed_game[1]
        self.coalition_lookup: dict[tuple[int], int] = computed_game[2]

        # setup callable mapping from index to computation
        self._index_mapping: dict[str, Callable[[], InteractionValues]] = {
            "Moebius": self.moebius_transform,
            # shapley_interaction
            "k-SII": self.shapley_interaction,
            "STII": self.shapley_interaction,
            "FSII": self.shapley_interaction,
            "kADD-SHAP": self.shapley_interaction,
            # base_generalized_value
            "SGV": self.base_generalized_value,
            "BGV": self.base_generalized_value,
            "CHGV": self.base_generalized_value,
            # shapley_base_interaction
            "SII": self.shapley_base_interaction,
            "BII": self.shapley_base_interaction,
            "CHII": self.shapley_base_interaction,
            # probabilistic_value
            "SV": self.probabilistic_value,
            "BV": self.probabilistic_value,
            # shapley_generalized_value
            "JointSV": self.shapley_generalized_value,
        }
        self.available_indices: set[str] = set(self._index_mapping.keys())
        self.available_concepts: dict[str, str] = ALL_AVAILABLE_CONCEPTS

    def __repr__(self) -> str:
        return f"ExactComputer(n_players={self.n}, game_fun={self.game_fun})"

    def __str__(self) -> str:
        return f"ExactComputer(n_players={self.n})"

    def __call__(self, index: str, order: int = None) -> InteractionValues:
        """Calls the computation of the specified index or value.

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

    def compute_game_values(
        self, game_fun: Callable[[np.ndarray], np.ndarray[float]]
    ) -> tuple[float, np.ndarray[float], dict[tuple[int], int]]:
        """Evaluates the game on the powerset of all coalitions.

        Args:
            game_fun: A callable game

        Returns:
            baseline value (empty prediction), all game values, and the lookup dictionary
        """

        coalition_lookup = {}
        coalition_matrix = np.zeros((2**self.n, self.n), dtype=bool)
        for i, T in enumerate(powerset(self._grand_coalition_set, min_size=0, max_size=self.n)):
            coalition_lookup[T] = i  # set lookup for the coalition
            coalition_matrix[i, T] = True  # one-hot-encode the coalition
        game_values = game_fun(coalition_matrix)  # compute the game values
        baseline_value = float(game_values[0])  # set the baseline value
        return baseline_value, game_values, coalition_lookup

    def moebius_transform(self, *args, **kwargs) -> InteractionValues:
        """Computes the Moebius transform for all 2^n coalitions of the game.

        Args:
            args and kwargs: Additional arguments and keyword arguments (not used)

        Returns:
            The Moebius transform for all coalitions stored in an InteractionValues object
        """
        # compute the Moebius transform
        moebius_transform = np.zeros(2**self.n)
        coalition_lookup = {}
        for interaction_pos, interaction in enumerate(powerset(self._grand_coalition_set)):
            coalition_lookup[interaction] = interaction_pos
            interaction_size = len(interaction)
            for coalition in powerset(interaction):
                coalition_pos = self.coalition_lookup[coalition]
                moebius_transform[interaction_pos] += (-1) ** (
                    interaction_size - len(coalition)
                ) * self.game_values[coalition_pos]

        # fill result into InteractionValues object and storage dictionary
        interaction_values = InteractionValues(
            values=moebius_transform,
            index="Moebius",
            max_order=self.n,
            min_order=0,
            n_players=self.n,
            interaction_lookup=coalition_lookup,
            estimated=False,
        )
        self._computed["Moebius"] = copy.deepcopy(interaction_values)
        return copy.deepcopy(interaction_values)

    def _base_weights(self, coalition_size: int, interaction_size: int, index: str) -> float:
        """Computes the weight of different indices in their common representation. For example, the
            weight of the discrete derivative of S given T in SII or the weight of the marginal
            contribution of S given T in SGV.

        Args:
            coalition_size: The size of the coalition from 0,...,n-interaction_size
            interaction_size: The size of the interaction from 0,...,order
            index: The computed index

        Returns:
            The base weight of the interaction index

        Raises:
            ValueError: If the index is not supported
        """

        if index in ["SII", "SGV"]:
            return 1 / (
                (self.n - interaction_size + 1) * binom(self.n - interaction_size, coalition_size)
            )
        elif index in ["BII", "BGV"]:
            return 1 / (2 ** (self.n - interaction_size))
        elif index in ["CHII", "CHGV"]:
            return interaction_size / (
                (interaction_size + coalition_size)
                * binom(self.n, interaction_size + coalition_size)
            )
        else:
            raise ValueError(f"Index {index} not supported")

    def _stii_weight(self, coalition_size: int, interaction_size: int, order: int) -> float:
        """Sets the weight for the representation of STII as a CII (using discrete derivatives).

        Args:
            coalition_size: Size of the Discrete Derivative
            interaction_size: Interaction size with s <= k
            order: Interaction order

        Returns:
            The weight of STII
        """

        if interaction_size == order:
            return float(order / (self.n * binom(self.n - 1, coalition_size)))
        else:
            if coalition_size == 0:
                return 1.0
            else:
                return 0.0

    def _get_fsii_weights(self) -> np.ndarray:
        """Pre-computes the kernel weight for the least square representation of FSII

        Returns:
            An array of the kernel weights for 0,...,n with "infinite weight" on 0 and n.
        """
        fsii_weights = np.zeros(self.n + 1, dtype=float)
        fsii_weights[0] = self._big_M
        fsii_weights[-1] = self._big_M
        for coalition_size in range(1, self.n):
            fsii_weights[coalition_size] = 1 / (
                (self.n - 1) * binom(self.n - 2, coalition_size - 1)
            )
        return fsii_weights

    def _get_stii_weights(self, order: int) -> np.ndarray:
        """Pre-computes the STII weights for the CII representation (using discrete derivatives)

        Args:
            order: The interaction order

        Returns:
            An array with pre-computed weights for t=0,...,n-k
        """

        stii_weights = np.zeros(self.n - order + 1, dtype=float)
        for t in range(self.n - order + 1):
            stii_weights[t] = self._stii_weight(t, order, order)
        return stii_weights

    def _get_discrete_derivative(
        self, interaction: Union[set[int], tuple[int]], coalition: Union[set[int], tuple[int]]
    ) -> float:
        """Computes the discrete derivative of a coalition with respect to an interaction.

        Args:
            interaction: A subset of the grand coalition as a set or tuple
            coalition: Subset of N as set of tuple

        Returns:
            The discrete derivative of the coalition with respect to the interaction
        """

        discrete_derivative = 0.0
        interaction_size = len(interaction)
        for interaction_subset in powerset(interaction):
            interaction_subset_size = len(interaction_subset)
            pos = self.coalition_lookup[
                tuple(sorted(set(coalition).union(set(interaction_subset))))
            ]
            discrete_derivative += (-1) ** (
                interaction_size - interaction_subset_size
            ) * self.game_values[pos]
        return discrete_derivative

    @staticmethod
    def get_n_interactions(n_players: int) -> np.ndarray:
        """Pre-computes an array that contains the number of interactions up to the size of the
            index (e.g. n_interactions[4] is the number of interactions up to size 4).

        Args:
            n_players: The number of players

        Returns:
            A numpy array containing the number of interactions up to the size of the index

        Examples:
            >>> ExactComputer.get_n_interactions(3)
            array([1, 4, 7, 8])  # binom(3, 0), binom(3, 0) + binom(3, 1), etc. ...
        """
        n_interactions = np.zeros(n_players + 1, dtype=int)
        n_interaction = 0
        for interaction_size in range(n_players + 1):
            n_interaction += int(binom(n_players, interaction_size))
            n_interactions[interaction_size] = n_interaction
        return n_interactions

    def _get_base_weights(self, index: str, order: int) -> np.ndarray:
        """Pre-computes all base weights for all coalition sizes (i.e. 0, ..., n-s) and all
            interaction sizes (i.e. 1, ..., order).

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            A numpy array with all base interaction weights
        """

        base_weights = np.zeros((self.n + 1, order + 1), dtype=float)
        for interaction_size in range(order + 1):
            for coalition_size in range(self.n - interaction_size + 1):
                base_weights[coalition_size, interaction_size] = self._base_weights(
                    coalition_size, interaction_size, index
                )
        return base_weights

    def base_interaction(self, index: str, order: int) -> InteractionValues:
        """Computes interactions based on representation with discrete derivatives, e.g. SII, BII.

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            An InteractionValues object containing the base interactions.
        """

        base_interaction_values = np.zeros(self._n_interactions[order])
        base_weights = self._get_base_weights(index, order)
        for coalition in powerset(self._grand_coalition_set):
            coalition_size = len(coalition)
            coalition_pos = self.coalition_lookup[coalition]
            for j, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
                interaction_size = len(interaction)
                coalition_cap_interaction = len(set(coalition).intersection(set(interaction)))
                base_interaction_values[j] += (
                    (-1) ** (interaction_size - coalition_cap_interaction)
                    * base_weights[coalition_size - coalition_cap_interaction, interaction_size]
                    * self.game_values[coalition_pos]
                )

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            interaction_lookup[interaction] = i

        # Transform into InteractionValues object and store in computed dictionary
        base_interaction = InteractionValues(
            values=base_interaction_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )
        self._computed[index] = copy.deepcopy(base_interaction)
        return copy.deepcopy(base_interaction)

    def base_generalized_value(self, index: str, order: int) -> InteractionValues:
        """Computes Base Generalized Values, i.e. probabilistic generalized values that do not
            depend on the order.

        According to the underlying representation using marginal contributions from
        [this work](https://doi.org/10.1016/j.dam.2006.05.002), the following indices are supported:
            - SGV: Shapley Generalized Value https://doi.org/10.1016/S0166-218X(00)00264-X
            - BGV: Banzhaf Generalized Value https://doi.org/10.1016/S0166-218X(00)00264-X
            - CHGV: Chaining Generalized Value https://doi.org/10.1016/j.dam.2006.05.002

        Args:
            order: The highest order of interactions
            index: The generalized value index

        Returns:
            An InteractionValues object containing generalized values.
        """

        base_generalized_values = np.zeros(self._n_interactions[order])
        base_weights = self._get_base_weights(index, order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            interaction_lookup[interaction] = i

        for i, coalition in enumerate(
            powerset(self._grand_coalition_set, min_size=0, max_size=self.n - 1)
        ):
            coalition_val = self.game_values[i]
            for j, interaction in enumerate(
                powerset((self._grand_coalition_set - set(coalition)), min_size=1, max_size=order)
            ):
                coalition_weight = base_weights[len(coalition), len(interaction)]
                base_generalized_values[
                    interaction_lookup[tuple(sorted(interaction))]
                ] += coalition_weight * (
                    self.game_values[self.coalition_lookup[tuple(sorted(coalition + interaction))]]
                    - coalition_val
                )

        # Transform into InteractionValues object
        base_generalized_values = InteractionValues(
            values=base_generalized_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )
        self._computed[index] = copy.deepcopy(base_generalized_values)
        return copy.deepcopy(base_generalized_values)

    def base_aggregation(
        self, base_interactions: InteractionValues, order: int
    ) -> InteractionValues:
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII

        Args:
            base_interactions: InteractionValues object containing interactions up to order "order"
            order: The highest order of interactions considered

        Returns:
            InteractionValues object containing transformed base_interactions
        """
        transformed_values = np.zeros(self.get_n_interactions(self.n)[order])
        transformed_lookup = {}
        bernoulli_numbers = bernoulli(order)  # lookup Bernoulli numbers
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            transformed_lookup[interaction] = i
            if len(interaction) == 0:
                # Initialize emptyset baseline value
                transformed_values[i] = base_interactions.baseline_value
            else:
                interaction_effect = base_interactions[interaction]
                subset_size = len(interaction)
                # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
                for interaction_higher_order in powerset(
                    self._grand_coalition_set, min_size=subset_size + 1, max_size=order
                ):
                    if not set(interaction).issubset(interaction_higher_order):
                        continue
                    # get the effect of T
                    interaction_tilde_effect = base_interactions[interaction_higher_order]
                    # normalization with bernoulli numbers
                    interaction_effect += (
                        bernoulli_numbers[len(interaction_higher_order) - subset_size]
                        * interaction_tilde_effect
                    )
                transformed_values[i] = interaction_effect

        # setup interaction values
        transformed_index = base_interactions.index  # raname the index (e.g. SII -> k-SII)
        if transformed_index not in ["SV", "BV"]:
            transformed_index = "k-" + transformed_index
        transformed_interactions = InteractionValues(
            values=transformed_values,
            index=transformed_index,
            min_order=0,
            max_order=order,
            interaction_lookup=transformed_lookup,
            n_players=self.n,
            estimated=False,
        )
        return copy.deepcopy(transformed_interactions)

    def compute_stii(self, order: int) -> InteractionValues:
        """Computes the STII index up to order "order".

        Args:
            order: The highest order of interactions

        Returns:
            InteractionValues object containing STII
        """

        stii_values = np.zeros(self._n_interactions[order])
        stii_values[0] = self.baseline_value  # set baseline value

        # create interaction lookup
        interaction_lookup = {}
        for interaction_pos, interaction in enumerate(
            powerset(self._grand_coalition_set, max_size=order)
        ):
            interaction_lookup[interaction] = interaction_pos

        # lower-order interactions (size < order) are the Möbius transform, i.e. discrete derivative with empty set
        for interaction in powerset(self._grand_coalition_set, max_size=order - 1):
            stii_values[interaction_lookup[interaction]] = self._get_discrete_derivative(
                interaction, tuple()
            )

        # pre-compute STII weights
        stii_weights = self._get_stii_weights(order)

        # top-order STII interactions
        for interaction in powerset(self._grand_coalition_set, min_size=order, max_size=order):
            interaction_pos = interaction_lookup[interaction]
            for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
                coalition_size = len(coalition)
                intersection_size = len(set(coalition).intersection(set(interaction)))
                stii_values[interaction_pos] += (
                    (-1) ** (order - intersection_size)
                    * stii_weights[coalition_size - intersection_size]
                    * self.game_values[coalition_pos]
                )

        # transform into InteractionValues object
        stii = InteractionValues(
            values=stii_values,
            index="STII",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )
        return copy.deepcopy(stii)

    def compute_fsii(self, order: int) -> InteractionValues:
        """Computes the FSII index up to order "order" after
            [Tsai et al. 2023](https://jmlr.org/papers/v24/22-0202.html).

        Args:
            order: The highest order of interactions

        Returns:
            InteractionValues object containing STII
        """
        fsii_weights = self._get_fsii_weights()
        least_squares_weights = np.zeros(2**self.n, dtype=float)
        coalition_matrix = np.zeros((2**self.n, self._n_interactions[order]), dtype=bool)

        # create interaction lookup
        interaction_lookup = {}
        for interaction_pos, interaction in enumerate(
            powerset(self._grand_coalition_set, max_size=order)
        ):
            interaction_lookup[interaction] = interaction_pos

        coalition_store = {}
        # Set least squares matrices
        for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
            least_squares_weights[coalition_pos] = fsii_weights[len(coalition)]
            for interaction in powerset(coalition, max_size=order):
                pos = interaction_lookup[interaction]
                coalition_matrix[coalition_pos, pos] = 1
            coalition_store[coalition] = coalition_pos
        weight_matrix_sqrt = np.sqrt(np.diag(least_squares_weights))
        coalition_matrix_weighted_sqrt = np.dot(weight_matrix_sqrt, coalition_matrix)
        game_value_weighted_sqrt = np.dot(self.game_values, weight_matrix_sqrt)
        # solve the weighted least squares (WLSQ) problem
        fsii_values, residuals, rank, singular_values = np.linalg.lstsq(
            coalition_matrix_weighted_sqrt, game_value_weighted_sqrt, rcond=None
        )

        # transform into InteractionValues object
        fsii_values[0] = self.baseline_value  # set baseline value
        fsii = InteractionValues(
            values=fsii_values,
            index="FSII",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )
        return copy.deepcopy(fsii)

    def get_jointsv_weights(self, order: int) -> np.ndarray:
        """Pre-compute JointSV weights for all coalition sizes (0, ... , n - order).

        Args:
            order: The highest order of interactions

        Returns:
            An array of pre-computed weights
        """
        weights = np.zeros(self.n, dtype=np.longdouble)
        q0den = sum([binom(self.n, s) for s in range(1, order + 1)])
        weights[0] = 1 / q0den
        # Carry out recursion
        for r in range(1, self.n):
            limd = min(order, (self.n - r))
            limn = max((r - order), 0)
            qden = sum([binom(self.n - r, s) for s in range(1, limd + 1)])
            qnum = sum([binom(r, s) * weights[s] for s in range(limn, r)])
            weights[r] = qnum / qden
        # check that the checksum is satisfied
        checksum = sum([binom(self.n, i) * weights[i] for i in range((self.n - order), self.n)])
        assert np.isclose(checksum, 1.0)
        return weights

    def compute_kadd_shap(self, order: int) -> InteractionValues:
        """Computes the kADD-SHAP index up to order "order". This is similar to FSII except that the
            coalition matrix contains the Bernoulli weights.

        The implementation is according to
        [Pelegrina et al. 2023](https://doi.org/10.1016/j.artint.2023.104014).

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values
        """

        weights = self._get_fsii_weights()
        least_squares_weights = np.zeros(2**self.n)
        coalition_matrix = np.zeros((2**self.n, self._n_interactions[order]))
        bernoulli_weights = get_bernoulli_weights(order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            interaction_lookup[interaction] = i

        for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
            least_squares_weights[coalition_pos] = weights[len(coalition)]
            for interaction in powerset(self._grand_coalition_set, min_size=1, max_size=order):
                intersection_size = len(set(coalition).intersection(interaction))
                interaction_size = len(interaction)
                # This is different from FSII
                coalition_matrix[coalition_pos, interaction_lookup[interaction]] = (
                    bernoulli_weights[interaction_size, intersection_size]
                )

        weight_matrix_sqrt = np.sqrt(np.diag(least_squares_weights))
        coalition_matrix_weighted_sqrt = np.dot(weight_matrix_sqrt, coalition_matrix)
        game_value_weighted_sqrt = np.dot(self.game_values, weight_matrix_sqrt)
        kADD_shap_values, residuals, rank, singular_values = np.linalg.lstsq(
            coalition_matrix_weighted_sqrt, game_value_weighted_sqrt, rcond=None
        )

        # Set baseline value
        kADD_shap_values[0] = self.baseline_value

        # Transform into InteractionValues object
        kADD_shap = InteractionValues(
            values=kADD_shap_values,
            index="kADD-SHAP",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )

        return kADD_shap

    def compute_joint_sv(self, order: int):
        """
        Computes the JointSV index up to order "order" according to
        https://openreview.net/forum?id=vcUmUvQCloe.

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values

        """
        jointSV_values = np.zeros(self._n_interactions[order])
        # Set baseline value
        jointSV_values[0] = self.baseline_value
        coalition_weights = self.get_jointsv_weights(order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
            interaction_lookup[interaction] = i

        for coalition_pos, coalition in enumerate(
            powerset(self._grand_coalition_set, min_size=0, max_size=self.n - 1)
        ):
            coalition_val = self.game_values[coalition_pos]
            coalition_weight = coalition_weights[len(coalition)]
            for interaction in powerset(
                self._grand_coalition_set - set(coalition), min_size=1, max_size=order
            ):
                jointSV_values[interaction_lookup[interaction]] += coalition_weight * (
                    self.game_values[self.coalition_lookup[tuple(sorted(coalition + interaction))]]
                    - coalition_val
                )

        # Transform into InteractionValues object
        jointSV = InteractionValues(
            values=jointSV_values,
            index="JointSV",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )
        return jointSV

    def shapley_generalized_value(self, order: int, index: str) -> InteractionValues:
        """
        Computes Shapley Generalized Values (i.e. Generalized Values that satisfy efficiency) after
            the underlying representation in https://doi.org/10.1016/j.dam.2006.05.002.

        Currently, the following indices are supported:
            - JointSV: https://openreview.net/forum?id=vcUmUvQCloe

        Args:
            order: The highest order of interactions
            index: The generalized value index

        Returns:
            An InteractionValues object containing generalized values

        Raises:
            ValueError: If the index is not supported.
        """
        if index == "JointSV":
            shapley_generalized_value = self.compute_joint_sv(order)
            self._computed[index] = shapley_generalized_value
            return copy.copy(shapley_generalized_value)
        else:
            raise ValueError(f"Index {index} not supported")

    def shapley_interaction(self, index: str, order: int) -> InteractionValues:
        """Computes k-additive Shapley Interactions, i.e. probabilistic interaction indices that
            depend on the order k.

        According to the underlying representation using discrete derivatives from
        https://doi.org/10.1016/j.geb.2005.03.002, the following indices are supported:
            - k-SII: k-Shapley Values https://proceedings.mlr.press/v206/bordt23a.html
            - STII:  Shapley-Taylor Interaction Index https://proceedings.mlr.press/v119/sundararajan20a.html
            - FSII: Faithful Shapley Interaction Index https://jmlr.org/papers/v24/22-0202.html
            - kADD-SHAP: k-additive Shapley Values https://doi.org/10.1016/j.artint.2023.104014

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        Raises:
            ValueError: If the index is not supported
        """
        if index == "k-SII":
            sii = self.base_interaction("SII", order)
            self._computed["SII"] = sii  # nice
            shapley_interaction = self.base_aggregation(sii, order)
        elif index == "STII":
            shapley_interaction = self.compute_stii(order)
        elif index == "FSII":
            shapley_interaction = self.compute_fsii(order)
        elif index == "kADD-SHAP":
            shapley_interaction = self.compute_kadd_shap(order)
        else:
            raise ValueError(f"Index {index} not supported")

        self._computed[index] = shapley_interaction
        return copy.copy(shapley_interaction)

    def shapley_base_interaction(self, index: str, order: int) -> InteractionValues:
        """Computes Shapley Base Interactions, i.e. probabilistic interaction indices that do not
            depend on the order.

        According to the underlying representation using discrete derivatives from
        https://doi.org/10.1016/j.geb.2005.03.002, the following indices are supported:
            - SII: Shapley Interaction Index https://link.springer.com/article/10.1007/s001820050125
            - BII: Banzhaf Interaction Index https://link.springer.com/article/10.1007/s001820050125
            - CHII: Chaining Interaction Index https://link.springer.com/chapter/10.1007/978-94-017-0647-6_5

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        """
        base_interaction = self.base_interaction(index, order)
        self._computed[index] = base_interaction
        return copy.copy(base_interaction)

    def probabilistic_value(self, index: str, *args, **kwargs) -> InteractionValues:
        """Computes common semi-values or probabilistic values, i.e. shapley values without
        efficiency axiom.

        These semi-values are special forms of interaction indices and generalized values for
        order = 1. According to the underlying representation using marginal contributions (cf.
        [semi-values](https://doi.org/10.1287/moor.6.1.122) or
        [probabilistic values](https://doi.org/10.1017/CBO9780511528446.008) the following indices
        are currently supported:
            - SV: Shapley value: https://doi.org/10.1515/9781400881970-018
            - BV: Banzhaf value: https://doi.org/10.1515/9781400881970-018

        Args:
            index: The interaction index
            args and kwargs: Additional arguments and keyword arguments (not used)

        Returns:
            An InteractionValues object containing probabilistic values

        Raises:
            ValueError: If the index is not supported
        """
        if index == "BV":
            probabilistic_value = self.base_interaction(index="BII", order=1)
        elif index == "SV":
            probabilistic_value = self.base_interaction(index="SII", order=1)
            # Change emptyset value of SII to baseline value
            probabilistic_value.baseline_value = self.baseline_value
            probabilistic_value.values[probabilistic_value.interaction_lookup[tuple()]] = (
                self.baseline_value
            )
        else:
            raise ValueError(f"Index {index} not supported")
        self._computed[index] = probabilistic_value
        return copy.copy(probabilistic_value)


def get_bernoulli_weights(order: int) -> np.ndarray:
    """Returns the bernoulli weights in the k-additive approximation via SII, e.g. used in
        kADD-SHAP.

    Args:
        order: The highest order of interactions

    Returns:
        An array containing the bernoulli weights
    """
    # TODO: We should import this in the kADD-SHAP approximator from here
    bernoulli_numbers = bernoulli(order)
    weights = np.zeros((order + 1, order + 1))
    for interaction_size in range(1, order + 1):
        for intersection_size in range(interaction_size + 1):
            for sum_index in range(1, intersection_size + 1):
                weights[interaction_size, intersection_size] += (
                    binom(intersection_size, sum_index)
                    * bernoulli_numbers[interaction_size - sum_index]
                )
    return weights
