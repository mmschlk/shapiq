# from shapiq import powerset
# TODO: Remove this part when import works
from collections.abc import Iterable
from itertools import chain, combinations
from typing import Any, Optional

import numpy as np
from scipy.special import bernoulli, binom


def powerset(
    iterable: Iterable[Any], min_size: Optional[int] = 0, max_size: Optional[int] = None
) -> Iterable[tuple]:
    s = sorted(list(iterable))
    max_size = len(s) if max_size is None else min(max_size, len(s))
    return chain.from_iterable(combinations(s, r) for r in range(max(min_size, 0), max_size + 1))


class MoebiusConverter:
    """
    Computes exact Shapley Interactions using the Möbius representation.
    Much faster than exact computation, if Möbius representation is sparse.

    :param n: Number of entities in the game, will define N={0,...,n-1}
    :param moebius_coefficients: Dictionary containing one key per non-zero moebius coefficient and its value
    """

    def __init__(self, n: int, moebius_coefficients: dict[str, float]):
        self.n = n
        self.moebius_coefficients: dict[str, float] = moebius_coefficients
        self.n_interactions = self._get_n_interactions()

    def _get_n_interactions(self):
        """
        Computes the number of interactions up to order k
        """
        n_interactions = np.zeros(self.n + 1, dtype=int)
        n_interaction = 0
        for i in range(self.n + 1):
            n_interaction += int(binom(self.n, i))
            n_interactions[i] = n_interaction
        return n_interactions

    # TODO: remove base-aggregation when formula for k-SII is fixed
    def base_aggregation(self, base_values, order):
        transformed = {}
        # Initialize emptyset baseline value
        # Lookup Bernoulli numbers
        bernoulli_numbers = bernoulli(order)

        for S in powerset(set(range(self.n)), min_size=1, max_size=order):
            subset_size = len(S)
            if S in base_values:
                S_effect = base_values[S]
            else:
                S_effect = 0
            # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
            for S_tilde in powerset(set(range(self.n)), min_size=subset_size + 1, max_size=order):
                if not set(S).issubset(S_tilde):
                    continue
                # get the effect of T
                if S_tilde in base_values:
                    S_tilde_effect = base_values[S_tilde]
                else:
                    S_tilde_effect = 0
                # normalization with bernoulli numbers
                S_effect += bernoulli_numbers[len(S_tilde) - subset_size] * S_tilde_effect
            transformed[S] = S_effect
        return transformed

    def _get_moebius_distribution_weight(
        self, moebius_size: int, interaction_size: int, order: int, index: str
    ):
        if index == "SII":
            return 1 / (moebius_size - interaction_size + 1)
        if index == "STII":
            if moebius_size <= order:
                return 1
            else:
                if interaction_size == order:
                    return 1 / binom(moebius_size, moebius_size - interaction_size)
                else:
                    return 0
        if index == "FSII":
            if moebius_size <= order:
                return 1
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

    def moebius_to_interaction(self, order, index):
        """
        Converts the Möbius coefficients to Shapley Interactions up to order k
        """
        if index == "k-SII":
            tmp_index = "k-SII"
            index = "SII"
        else:
            tmp_index = None

        rslt = {}
        # Set lowest interaction size that receives higher-order Möbius values, default is 1, for STII is order
        if index == "STII":
            # STII distributes only to interactions of maximum order
            min_receiving_interaction = order
        else:
            # Distributes value to all interactions
            min_receiving_interaction = 1

        # Set indices with no distribution, if Möbius set <= order
        if index in ["FSII", "STII", "k-SII"]:
            trivial_moebius_interactions = True
        else:
            # SII still distributes this value
            trivial_moebius_interactions = False

        # Pre-compute weights
        distribution_weights = np.zeros((self.n + 1, order + 1))
        for moebius_size in range(1, self.n + 1):
            for interaction_size in range(1, min(order, moebius_size) + 1):
                distribution_weights[
                    moebius_size, interaction_size
                ] = self._get_moebius_distribution_weight(
                    moebius_size, interaction_size, order, index
                )

        for moebius_set, moebius_val in self.moebius_coefficients.items():
            moebius_size = len(moebius_set)
            # For lower-order Möbius sets (size<= order) and indices with trivial_low_interactions directly set interaction value
            if moebius_size <= order and trivial_moebius_interactions:
                val_distributed = distribution_weights[moebius_size, moebius_size]
                if moebius_set in rslt:
                    rslt[moebius_set] += moebius_val * val_distributed
                else:
                    rslt[moebius_set] = moebius_val * val_distributed
            else:
                # For higher-order Möbius sets (size > order) distribute the value among all contained interactions
                for interaction in powerset(
                    moebius_set, min_size=min_receiving_interaction, max_size=order
                ):
                    val_distributed = distribution_weights[moebius_size, len(interaction)]
                    # Check if Möbius value is distributed onto this interaction
                    if interaction in rslt:
                        rslt[interaction] += moebius_val * val_distributed
                    else:
                        rslt[interaction] = moebius_val * val_distributed

        if tmp_index is None:
            return rslt
        else:
            return self.base_aggregation(rslt, order)


if __name__ == "__main__":
    n = 10
    """
    moebius_converter = MoebiusConverter(n, soum.moebius_coefficients_dict)
    order = 2

    values = {}
    for index in ["SII", "STII", "k-SII", "FSII"]:
        values[index] = moebius_converter.moebius_to_interaction(order, index)
        rslt = 0
        for interaction, value in values[index].items():
            if index == "SII":
                if len(interaction) == 1:
                    rslt += value
            else:
                rslt += value
        print(index, ": ", rslt)
    """
