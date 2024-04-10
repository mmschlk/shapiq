import copy
from typing import Callable

import numpy as np
from scipy.special import bernoulli, binom

from shapiq import powerset
from shapiq.interaction_values import InteractionValues


class ExactComputer:
    """Computes exact Shapley Interactions for specified game by evaluating all 2^n coalitions

    Args:
        N: The set of players.
        game_fun: A callable game


    Attributes:
        N: The set of players
        n: The number of players.
        big_M: The infinite weight for KernelSHAP
        n_interactions: A pre-computed numpy array containing the number of interactions up to the size of the index, e.g. n_interactions[4] is the nuber of all interactions up to size 4
        computed_interactions: A dictionary that stores computations of different indices
        game_fun: The callable game
        baseline_value: The baseline value, i.e. the emptyset prediction
        game_values: A numpy array containing the game evaluations of all subsets
        coalition_lookup: A dictionary containing the index of every coalition in game_values
    """

    def __init__(
        self,
        N: set,
        game_fun: Callable,
    ):
        self.N = copy.copy(N)
        self.n = len(N)
        self.big_M = 10e7
        self.n_interactions = self.get_n_interactions()
        self.computed_interactions = {}
        self.game_fun = game_fun
        self.baseline_value, self.game_values, self.coalition_lookup = self.compute_game_values(
            game_fun
        )

    def compute_game_values(self, game_fun: Callable):
        """Evaluates the game on the powerset using game_fun

        Args:
            game_fun: A callable game

        Returns:
            The baseline value (empty prediction), all game values and the corresponding lookup dictionary
        """
        coalition_lookup = {}
        # compute the game values
        coalition_matrix = np.zeros((2**self.n, self.n))
        for i, T in enumerate(powerset(self.N, min_size=0, max_size=self.n)):
            coalition_lookup[T] = i
            coalition_matrix[i, T] = 1
        game_values = game_fun(coalition_matrix)
        # Ensures that game_values
        baseline_value = game_values[0]
        # Zero-centering of game values
        # game_values -= baseline_value

        return baseline_value, game_values, coalition_lookup

    def moebius_transform(self):
        """Computes the Moebius transform for all 2^n coalitions

        Args:

        Returns:
            The Moebius transform for all coalitions stored in an InteractionValues object
        """

        moebius_transform = np.zeros(2**self.n)
        # compute the Moebius transform
        coalition_lookup = {}
        for i, S in enumerate(powerset(self.N)):
            coalition_lookup[S] = i
        for i, S in enumerate(powerset(self.N)):
            s = len(S)
            S_pos = coalition_lookup[S]
            for T in powerset(S):
                pos = self.coalition_lookup[T]
                moebius_transform[S_pos] += (-1) ** (s - len(T)) * self.game_values[pos]

        self.computed_interactions["Moebius"] = InteractionValues(
            values=moebius_transform,
            index="Moebius",
            max_order=self.n,
            min_order=0,
            n_players=self.n,
            interaction_lookup=coalition_lookup,
            estimated=False,
        )
        return copy.copy(self.computed_interactions["Moebius"])

    def base_weights(self, coalition_size: int, interaction_size: int, index: str):
        """Computes the weight of different indices in their common representation,
        e.g. the weight of the discrete derivative of S given T in SII
        e.g. the weight of the marginal contribution of S given T in SGV.

        Args:
            coalition_size: The size of the coalition from 0,...,n-interaction_size
            interaction_size: The size of the interaction from 0,...,order
            index: The computed index

        Returns:
            The base weight of the interaction index
        """

        if index in ["SII", "SGV"]:
            return 1 / (
                (self.n - interaction_size + 1) * binom(self.n - interaction_size, coalition_size)
            )
        if index in ["BII", "BGV"]:
            return 1 / (2 ** (self.n - interaction_size))
        if index in ["CHII", "CHGV"]:
            return interaction_size / (
                (interaction_size + coalition_size)
                * binom(self.n, interaction_size + coalition_size)
            )

    def stii_weight(self, coalition_size: int, interaction_size: int, order: int):
        """Sets the weight for the representation of STII as a CII (using discrete derivatives)

        Args:
            coalition_size: Size of the Discrete Derivative
            interaction_size: Interaction size with s <= k
            order: Interaction order

        Returns:
            The weight of STII
        """

        if interaction_size == order:
            return order / ((self.n) * binom(self.n - 1, coalition_size))
        else:
            if coalition_size == 0:
                return 1
            else:
                return 0

    def get_fsii_weights(self):
        """Pre-computes the kernel weight for the least square representation of FSII

        Args:

        Returns:
            An array of the kernel weights for 0,...,n with "infinite weight on 0 and n using big_M
        """
        fsii_weights = np.zeros(self.n + 1)
        fsii_weights[0] = self.big_M
        fsii_weights[-1] = self.big_M
        for coalition_size in range(1, self.n):
            fsii_weights[coalition_size] = 1 / (
                (self.n - 1) * binom(self.n - 2, coalition_size - 1)
            )
        return fsii_weights

    def get_stii_weights(self, order: int):
        """Pre-computes the STII weights for the CII representation (using discrete derivatives)

        Args:
            order: The interaction order

        Returns:
            An array with pre-computed weights for t=0,...,n-k
        """

        stii_weights = np.zeros(self.n - order + 1)
        for t in range(self.n - order + 1):
            stii_weights[t] = self.stii_weight(t, order, order)
        return stii_weights

    def get_discrete_derivative(self, interaction, coalition):
        """Computes the discrete derivative of a coalition with respect to an interaction

        Args:
            interaction: Subset of N as set or tuple
            coalition: Subset of N as set of tuple

        Returns:
            The discrete derivative of the coalition with respect to the interaction
        """

        rslt = 0
        interaction_size = len(interaction)
        for interaction_subset in powerset(interaction):
            interaction_subset_size = len(interaction_subset)
            pos = self.coalition_lookup[
                tuple(sorted(set(coalition).union(set(interaction_subset))))
            ]
            rslt += (-1) ** (interaction_size - interaction_subset_size) * self.game_values[pos]
        return rslt

    def get_n_interactions(self):
        """Pre-computes an array that contains the number of interactions up to the size of the index.

        Args:

        Returns:
            A numpy array containing the number of interactions up to the size of the index, e.g. n_interactions[4] is the number of interactions up to size 4.
        """
        n_interactions = np.zeros(self.n + 1, dtype=int)
        n_interaction = 0
        for interaction_size in range(self.n + 1):
            n_interaction += int(binom(self.n, interaction_size))
            n_interactions[interaction_size] = n_interaction
        return n_interactions

    def get_base_weights(self, index: str, order: int):
        """Pre-computes all base weights for all coalition_size=0,...,n-s and all interaction_size=1,...order

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            A numpy array with all base interaction weights
        """

        base_weights = np.zeros((self.n + 1, order + 1))
        for interaction_size in range(order + 1):
            for coalition_size in range(self.n - interaction_size + 1):
                base_weights[coalition_size, interaction_size] = self.base_weights(
                    coalition_size, interaction_size, index
                )
        return base_weights

    def base_interactions(self, index: str, order: int):
        """Computes interactions based on representation with discrete derivatives, e.g. SII, BII

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            An InteractionValues object containing the base interactions
        """

        base_interaction_values = np.zeros(self.n_interactions[order])
        base_weights = self.get_base_weights(index, order)
        for coalition in powerset(self.N):
            coalition_size = len(coalition)
            coalition_pos = self.coalition_lookup[coalition]
            for j, interaction in enumerate(powerset(self.N, max_size=order)):
                interaction_size = len(interaction)
                coalition_cap_interaction = len(set(coalition).intersection(set(interaction)))
                base_interaction_values[j] += (
                    (-1) ** (interaction_size - coalition_cap_interaction)
                    * base_weights[coalition_size - coalition_cap_interaction, interaction_size]
                    * self.game_values[coalition_pos]
                )

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = i

        # Transform into InteractionValues object
        self.computed_interactions[index] = InteractionValues(
            values=base_interaction_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )

        return copy.copy(self.computed_interactions[index])

    def base_generalized_values(self, index: str, order: int):
        """Computes the Base Generalized Values according to the representation with marginal contributions, e.g. SGV, BGV, CGV

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            An InteractionValues object containing the base generalized values
        """

        base_generalized_values = np.zeros(self.n_interactions[order])
        base_weights = self.get_base_weights(index, order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = i

        for i, coalition in enumerate(powerset(self.N, min_size=0, max_size=self.n - 1)):
            coalition_val = self.game_values[i]
            for j, interaction in enumerate(
                powerset((self.N - set(coalition)), min_size=1, max_size=order)
            ):
                coalition_weight = base_weights[len(coalition), len(interaction)]
                base_generalized_values[
                    interaction_lookup[tuple(sorted(interaction))]
                ] += coalition_weight * (
                    self.game_values[self.coalition_lookup[tuple(sorted(coalition + interaction))]]
                    - coalition_val
                )

        # Transform into InteractionValues object
        self.computed_interactions[index] = InteractionValues(
            values=base_generalized_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )

        return self.computed_interactions[index].__copy__()

    def base_aggregation(self, base_interactions: InteractionValues, order: int):
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII

        Args:
            base_interactions: InteractionValues object containing interactions up to order "order"
            order: The highest order of interactions considered

        Returns:
            InteractionValues object containing transformed base_interactions
        """
        transformed_values = np.zeros(self.get_n_interactions()[order])
        transformed_lookup = {}
        # Lookup Bernoulli numbers
        bernoulli_numbers = bernoulli(order)

        for i, interaction in enumerate(powerset(self.N, max_size=order)):
            transformed_lookup[interaction] = i
            if len(interaction) == 0:
                # Initialize emptyset baseline value
                transformed_values[i] = base_interactions.baseline_value
            else:
                S_effect = base_interactions[interaction]
                subset_size = len(interaction)
                # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
                for interaction_higher_order in powerset(
                    self.N, min_size=subset_size + 1, max_size=order
                ):
                    if not set(interaction).issubset(interaction_higher_order):
                        continue
                    # get the effect of T
                    S_tilde_effect = base_interactions[interaction_higher_order]
                    # normalization with bernoulli numbers
                    S_effect += (
                        bernoulli_numbers[len(interaction_higher_order) - subset_size]
                        * S_tilde_effect
                    )
                transformed_values[i] = S_effect

        transformed_index = "k-" + base_interactions.index

        transformed_interactions = InteractionValues(
            values=transformed_values,
            index=transformed_index,
            min_order=0,
            max_order=order,
            interaction_lookup=transformed_lookup,
            n_players=self.n,
        )

        return transformed_interactions

    def compute_stii(self, order: int):
        """Computes the STII index up to order "order"

        Args:
            order: The highest order of interactions

        Returns:
            InteractionValues object containing STII
        """

        stii_values = np.zeros(self.n_interactions[order])

        # Set baseline value
        stii_values[0] = self.baseline_value

        # Create interaction lookup
        interaction_lookup = {}
        for interaction_pos, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = interaction_pos

        # Lower-order interactions (size < order) are the Möbius transform, i.e. discrete derivative with empty set
        for interaction in powerset(self.N, max_size=order - 1):
            stii_values[interaction_lookup[interaction]] = self.get_discrete_derivative(
                interaction, ()
            )

        # Pre-compute STII weights
        stii_weights = self.get_stii_weights(order)

        # Top-order STII interactions
        for interaction in powerset(self.N, min_size=order, max_size=order):
            interaction_pos = interaction_lookup[interaction]
            for coalition_pos, coalition in enumerate(powerset(self.N)):
                coalition_size = len(coalition)
                intersection_size = len(set(coalition).intersection(set(interaction)))
                stii_values[interaction_pos] += (
                    (-1) ** (order - intersection_size)
                    * stii_weights[coalition_size - intersection_size]
                    * self.game_values[coalition_pos]
                )

        # Transform into InteractionValues object
        stii = InteractionValues(
            values=stii_values,
            index="STII",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )

        return stii

    def compute_fsii(self, order: int):
        """Computes the FSII index up to order "order"
        According to https://jmlr.org/papers/v24/22-0202.html
        Args:
            order: The highest order of interactions

        Returns:
            InteractionValues object containing STII
        """
        fsii_weights = self.get_fsii_weights()
        lstsq_weights = np.zeros(2**self.n)
        coalition_matrix = np.zeros((2**self.n, self.n_interactions[order]))

        # Create interaction lookup
        interaction_lookup = {}
        for interaction_pos, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = interaction_pos

        coalition_store = {}
        # Set LSTSQ matrices
        for coalition_pos, coalition in enumerate(powerset(self.N)):
            lstsq_weights[coalition_pos] = fsii_weights[len(coalition)]
            for interaction in powerset(coalition, max_size=order):
                pos = interaction_lookup[interaction]
                coalition_matrix[coalition_pos, pos] = 1
            coalition_store[coalition] = coalition_pos
        weight_matrix_sqrt = np.sqrt(np.diag(lstsq_weights))
        coalition_matrix_weighted_sqrt = np.dot(weight_matrix_sqrt, coalition_matrix)
        game_value_weighted_sqrt = np.dot(self.game_values, weight_matrix_sqrt)
        # Solve WLSQ problem
        fsii_values, residuals, rank, singular_values = np.linalg.lstsq(
            coalition_matrix_weighted_sqrt, game_value_weighted_sqrt, rcond=None
        )

        # Set baseline value
        fsii_values[0] = self.baseline_value

        # Transform into InteractionValues object
        fsii = InteractionValues(
            values=fsii_values,
            index="FSII",
            max_order=order,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            estimated=False,
        )

        return fsii

    def get_jointsv_weights(self, order):
        """Pre-compute JointSV weights for coalition_size=0,...,n-order

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
        # Check that the checksum is satisfied
        checksum = sum([binom(self.n, i) * weights[i] for i in range((self.n - order), self.n)])
        assert np.isclose(checksum, 1.0)
        return weights

    def get_bernoulli_weights(self, order):
        """Returns the bernoulli weights in the k-additive approximation via SII, e.g. used in kADD-SHAP

        Args:
            order: The highest order of interactions

        Returns:
            An array containing the bernoulli weights
        """

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

    def compute_kadd_shap(self, order: int):
        """Computes the kADD-SHAP index up to order "order". This is similar to FSII except that the coalition matrix contains the Bernoulli weights
        According to https://doi.org/10.1016/j.artint.2023.104014

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values
        """

        weights = self.get_fsii_weights()
        lstsq_weights = np.zeros(2**self.n)
        coalition_matrix = np.zeros((2**self.n, self.n_interactions[order]))
        bernoulli_weights = self.get_bernoulli_weights(order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = i

        for coalition_pos, coalition in enumerate(powerset(self.N)):
            lstsq_weights[coalition_pos] = weights[len(coalition)]
            for interaction in powerset(self.N, min_size=1, max_size=order):
                intersection_size = len(set(coalition).intersection(interaction))
                interaction_size = len(interaction)
                # This is different from FSII
                coalition_matrix[
                    coalition_pos, interaction_lookup[interaction]
                ] = bernoulli_weights[interaction_size, intersection_size]

        weight_matrix_sqrt = np.sqrt(np.diag(lstsq_weights))
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

    def compute_jointSV(self, order):
        """
        Computes the JointSV index up to order "order"
        According to https://openreview.net/forum?id=vcUmUvQCloe

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values

        """
        jointSV_values = np.zeros(self.n_interactions[order])
        # Set baseline value
        jointSV_values[0] = self.baseline_value
        coalition_weights = self.get_jointsv_weights(order)

        interaction_lookup = {}
        for i, interaction in enumerate(powerset(self.N, max_size=order)):
            interaction_lookup[interaction] = i

        for coalition_pos, coalition in enumerate(
            powerset(self.N, min_size=0, max_size=self.n - 1)
        ):
            coalition_val = self.game_values[coalition_pos]
            coalition_weight = coalition_weights[len(coalition)]
            for interaction in powerset(self.N - set(coalition), min_size=1, max_size=order):
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

    def shapley_generalized_values(self, order: int, index: str) -> InteractionValues:
        """
        Computes Shapley Generalized Values, i.e. Generalized Values that satisfy efficiency
        According to the underlying representation in https://doi.org/10.1016/j.dam.2006.05.002
        Currently covers:
            - JointSV: https://openreview.net/forum?id=vcUmUvQCloe
        Args:
            order: The highest order of interactions
            index: The generalized value index

        Returns:
            An InteractionValues object containing generalized values

        """
        if index == "JointSV":
            shapley_generalized_value = self.compute_jointSV(order)

        self.computed_interactions[index] = shapley_generalized_value
        return copy.copy(shapley_generalized_value)

    def shapleygeneralized_values(self, order: int, index: str) -> InteractionValues:
        """
        Computes Shapley Generalized Values, i.e. probabilistic generalized values that do not depend on the order
        According to the underlying representation using marginal contributions from https://doi.org/10.1016/j.dam.2006.05.002
        Currently covers:
            - SGV: Shapley Generalized Value https://doi.org/10.1016/S0166-218X(00)00264-X
            - BGV: Banzhaf Generalized Value https://doi.org/10.1016/S0166-218X(00)00264-X
            - CHGV: Chaining Generalized Value https://doi.org/10.1016/j.dam.2006.05.002
        Args:
            order: The highest order of interactions
            index: The generalized value index

        Returns:
            An InteractionValues object containing generalized values

        """
        if index == "JointSV":
            shapley_generalized_value = self.compute_jointSV(order)

        self.computed_interactions[index] = shapley_generalized_value
        return copy.copy(shapley_generalized_value)

    def shapley_interaction(self, order: int, index: str = "k-SII") -> InteractionValues:
        """
        Computes k-additive Shapley Interactions, i.e. probabilistic interaction indices that depend on the order k
        According to the underlying representation using discrete derivatives from https://doi.org/10.1016/j.geb.2005.03.002
        Currently covers:
            - k-SII: k-Shapley Values https://proceedings.mlr.press/v206/bordt23a.html
            - STII:  Shapley-Taylor Interaction Index https://proceedings.mlr.press/v119/sundararajan20a.html
            - FSII: Faithful Shapley Interaction Index https://jmlr.org/papers/v24/22-0202.html
            - kADD-SHAP: k-additive Shapley Values https://doi.org/10.1016/j.artint.2023.104014

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        """
        if index == "k-SII":
            sii = self.base_interactions("SII", order)
            self.computed_interactions["SII"] = sii
            shapley_interactions = self.base_aggregation(sii, order)
        if index == "STII":
            shapley_interactions = self.compute_stii(order)
        if index == "FSII":
            shapley_interactions = self.compute_fsii(order)
        if index == "kADD-SHAP":
            shapley_interactions = self.compute_kadd_shap(order)

        self.computed_interactions[index] = shapley_interactions

        return copy.copy(shapley_interactions)

    def shapley_base_interaction(self, order: int, index: str) -> InteractionValues:
        """
        Computes Shapley Base Interactions, i.e. probabilistic interaction indices that do not depend on the order
        According to the underlying representation using discrete derivatives from https://doi.org/10.1016/j.geb.2005.03.002
        Currently covers:
            - SII: Shapley Interaction Index https://link.springer.com/article/10.1007/s001820050125
            - CHII: Chaining Interaction Index https://link.springer.com/chapter/10.1007/978-94-017-0647-6_5
            - BII: Banzhaf Interaction Index https://link.springer.com/article/10.1007/s001820050125

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        """
        base_interactions = self.base_interactions(index, order)
        self.computed_interactions[index] = base_interactions
        return copy.copy(base_interactions)

    def probabilistic_values(self, index: str) -> InteractionValues:
        """Computes common semi-values or probabilistic values, i.e. shapley values without efficiency axiom. These are special of interaction indices and generalized values for order = 1.
        According to the underlying representation using marginal contributions, cf.
            - semi-values https://doi.org/10.1287/moor.6.1.122
            - probabilistic values https://doi.org/10.1017/CBO9780511528446.008

        Currently covers:
            - SV: Shapley Value https://doi.org/10.1515/9781400881970-018
            - BV: Banzhaf https://doi.org/10.1515/9781400881970-018

        Args:
            index: The interaction index

        Returns:
            An InteractionValues object containing probabilistic values
        """

        probabilistic_value = self.base_interactions(index, 1)
        self.computed_interactions[index] = probabilistic_value
        return copy.copy(probabilistic_value)
