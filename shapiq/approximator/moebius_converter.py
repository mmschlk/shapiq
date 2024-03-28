from scipy.special import bernoulli, binom
import numpy as np

# from shapiq import powerset

# TODO: Remove this part when import works
from collections.abc import Iterable
from itertools import chain, combinations
from typing import Any, Optional


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

    def convert_to_interactions(self, index, order):
        """
        Converts the Möbius coefficients to Shapley Interactions up to order k
        """
        if index == "SII":
            shapley_interactions = self.moebius_to_sii(order)
        if index in ["k-SII"]:
            # shapley_interactions = self.base_aggregation(sii, order)
        if index == "STII":
            shapley_interactions = self.moebius_to_stii(order)
        if index == "FSII":
            shapley_interactions = self.moebius_to_fsii(order)
        return shapley_interactions

    def moebius_to_sii(self, order):
        rslt = {}
        for moebius_set, moebius_val in self.moebius_coefficients.items():
            #Distribute the value among all contained interactions
            moebius_size = len(moebius_set)
            for interaction in powerset(
                moebius_set, min_size=1, max_size=int(min(moebius_size, order))
            ):
                interaction_size = len(interaction)
                if interaction in rslt:
                    rslt[interaction] += moebius_val / (moebius_size - interaction_size + 1)
                else:
                    rslt[interaction] = moebius_val / (moebius_size - interaction_size + 1)
        return rslt

    def moebius_to_ksii(self, order):
        rslt = {}
        for moebius_set, moebius_val in self.moebius_coefficients.items():
            moebius_size = len(moebius_set)
            #For lower-order Möbius sets (size<= order) directly set interaction value
            if moebius_size <= order:
                if moebius_set in rslt:
                    rslt[moebius_set] += moebius_val
                else:
                    rslt[moebius_set] = moebius_val
            else:
                #For higher-order Möbius sets (size > order) distribute the value among all contained interactions
                for interaction in powerset(moebius_set, min_size=order, max_size=order):
                    val_for_interaction = moebius_val / binom(moebius_size, moebius_size - order)
                    if interaction in rslt:
                        rslt[interaction] += val_for_interaction
                    else:
                        rslt[interaction] = val_for_interaction
        return rslt
    def moebius_to_stii(self, order):
        rslt = {}
        for moebius_set, moebius_val in self.moebius_coefficients.items():
            moebius_size = len(moebius_set)
            #For lower-order Möbius sets (size<= order) directly set interaction value
            if moebius_size <= order:
                if moebius_set in rslt:
                    rslt[moebius_set] += moebius_val
                else:
                    rslt[moebius_set] = moebius_val
            else:
                #For higher-order Möbius sets (size > order) distribute the value among all contained interactions
                for interaction in powerset(moebius_set, min_size=order, max_size=order):
                    val_for_interaction = moebius_val / binom(moebius_size, moebius_size - order)
                    if interaction in rslt:
                        rslt[interaction] += val_for_interaction
                    else:
                        rslt[interaction] = val_for_interaction
        return rslt

    def moebius_to_fsii(self, order):
        rslt = {}
        for moebius_set, moebius_val in self.moebius_coefficients.items():
            moebius_size = len(moebius_set)
            if moebius_size <= order:
                #For lower-order Möbius sets (size<= order) directly set interaction value
                if moebius_set in rslt:
                    rslt[moebius_set] += moebius_val
                else:
                    rslt[moebius_set] = moebius_val
            else:
                #For higher-order Möbius sets (size > order) distribute the value among all contained interactions
                for interaction in powerset(moebius_set, min_size=1, max_size=order):
                    interaction_size = len(interaction)
                    val_for_interaction = (
                        moebius_val
                        * (-1) ** (order - interaction_size)
                        * (interaction_size / (order + interaction_size))
                        * (
                            binom(order, interaction_size)
                            * binom(moebius_size - 1, order)
                            / binom(moebius_size + order - 1, order + interaction_size)
                        )
                    )
                    if interaction in rslt:
                        rslt[interaction] += val_for_interaction
                    else:
                        rslt[interaction] = val_for_interaction
        return rslt


if __name__ == "__main__":
    n = 5
    moebius_converter = MoebiusConverter(n, soum.moebius_coefficients_dict)
    order = 2
    sii = moebius_converter.moebius_to_sii(order)
    rslt = 0
    for set, val in sii.items():
        if len(set) == 1:
            rslt += val
    print("SII: ", rslt)
    stii = moebius_converter.moebius_to_stii(order)
    rslt = 0
    for set, val in stii.items():
        rslt += val
    print("STII: ", rslt)
    fsii = moebius_converter.moebius_to_fsii(order)
    rslt = 0
    for set, val in fsii.items():
        rslt += val
    print("FSII: ", rslt)
