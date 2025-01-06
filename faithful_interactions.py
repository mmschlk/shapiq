import copy
from typing import Optional

import numpy as np
import scipy as sp
from scipy.special import binom

from shapiq import ExactComputer, Game, InteractionValues, powerset

# plot the results
from shapiq.games.benchmark import SOUM


def _change_index(index: str) -> str:
    """Changes the index of the interaction values to the new index.

    Args:
        index: The current index of the interaction values.

    Returns:
        The new index of the interaction values.
    """
    if index in ["SV", "BV"]:  # no change for probabilistic values like SV or BV
        return index
    new_index = "-".join(("k", index))
    return new_index


def reverse_interaction_values(
    interactions: InteractionValues, order: Optional[int] = None
) -> InteractionValues:
    """Reverses k-additive interaction values into a base interaction index."""

    bernoulli_numbers = sp.special.bernoulli(order)  # used for aggregation
    baseline_value = interactions.baseline_value
    # iterate over all interactions in base_interactions and project them onto all interactions T
    # where 1 <= |T| <= order

    interactions_dict = interactions.dict_values

    for interaction in reversed(list(powerset(N, min_size=1, max_size=order))):
        value = interactions_dict[interaction]
        for base_interaction in powerset(interaction, min_size=1, max_size=len(interaction) - 1):
            scaling = float(bernoulli_numbers[len(interaction) - len(base_interaction)])
            update_interaction = scaling * value
            try:
                interactions_dict[base_interaction] -= update_interaction
            except KeyError:
                interactions_dict[base_interaction] = update_interaction

    lookup: dict[tuple[int, ...], int] = {}  # maps interactions to their index in the values vector
    aggregated_values = np.zeros(len(interactions_dict), dtype=float)
    for pos, (interaction, interaction_value) in enumerate(interactions_dict.items()):
        lookup[interaction] = pos
        aggregated_values[pos] = interaction_value

    # update the index name after the aggregation (e.g., SII -> k-SII)
    new_index = interactions.index[2:]

    return InteractionValues(
        n_players=interactions.n_players,
        values=aggregated_values,
        index=new_index,
        interaction_lookup=lookup,
        baseline_value=baseline_value,
        min_order=0,  # always order 0 for this aggregation
        max_order=order,
        estimated=interactions.estimated,
        estimation_budget=interactions.estimation_budget,
    )


def aggregate_interaction_values(
    base_interactions: InteractionValues, order: Optional[int] = None
) -> InteractionValues:
    """Aggregates the basis interaction values into an efficient interaction index."""

    bernoulli_numbers = sp.special.bernoulli(order)  # used for aggregation
    baseline_value = base_interactions.baseline_value
    transformed_dict: dict[tuple, float] = {tuple(): baseline_value}  # storage
    # iterate over all interactions in base_interactions and project them onto all interactions T
    # where 1 <= |T| <= order
    for base_interaction, pos in base_interactions.interaction_lookup.items():
        base_interaction_value = float(base_interactions.values[pos])
        for interaction in powerset(base_interaction, min_size=1, max_size=order):
            scaling = float(bernoulli_numbers[len(base_interaction) - len(interaction)])
            update_interaction = scaling * base_interaction_value
            try:
                transformed_dict[interaction] += update_interaction
            except KeyError:
                transformed_dict[interaction] = update_interaction

    lookup: dict[tuple[int, ...], int] = {}  # maps interactions to their index in the values vector
    aggregated_values = np.zeros(len(transformed_dict), dtype=float)
    for pos, (interaction, interaction_value) in enumerate(transformed_dict.items()):
        lookup[interaction] = pos
        aggregated_values[pos] = interaction_value

    # update the index name after the aggregation (e.g., SII -> k-SII)
    new_index = _change_index(base_interactions.index)

    return InteractionValues(
        n_players=base_interactions.n_players,
        values=aggregated_values,
        index=new_index,
        interaction_lookup=lookup,
        baseline_value=baseline_value,
        min_order=0,  # always order 0 for this aggregation
        max_order=order,
        estimated=base_interactions.estimated,
        estimation_budget=base_interactions.estimation_budget,
    )


def bernoulli_test(s, t_cap_s):
    val = 0
    for r in range(1, t_cap_s + 2):
        val += binom(t_cap_s, r - 1) * sp.special.bernoulli(n_players)[s - r]
    return val


class approx_game_fsii(Game):
    def __init__(self, n_players, fsii, order):
        # init the base game
        super().__init__(
            n_players,
            normalize=False,
            normalization_value=0,
            verbose=False,
        )
        self.order = order
        self._grand_coalition_set = set(range(self.n_players))
        self.game_values, self.coalition_lookup_test = self.approximate_game_values(fsii)

    def approximate_game_values(self, values):
        game_values = np.zeros(2**self.n_players)
        game_lookup = {}
        for i, T in enumerate(powerset(self._grand_coalition_set)):
            game_values[i] = 0
            game_lookup[tuple(T)] = i
            for S in powerset(T, min_size=1, max_size=self.order):
                game_values[i] += values[S]  # *(-0.5)
        return game_values, game_lookup

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(np.shape(coalitions)[0])
        for row_index, row in enumerate(coalitions):
            values[row_index] = self.game_values[
                self.coalition_lookup_test[tuple(np.where(row)[0])]
            ]
        return values


class approx_game_sii(Game):
    def __init__(self, n_players, sii, order, min_order=1):
        # init the base game
        super().__init__(
            n_players,
            normalize=False,
            normalization_value=0,
            verbose=False,
        )
        self.order = order
        self.min_order = min_order
        self._grand_coalition_set = set(range(self.n_players))
        self.bernoulli = sp.special.bernoulli(n_players)
        self.game_values, self.coalition_lookup_test = self.approximate_game_values(sii)

    def approximate_game_values(self, values):
        game_values = np.zeros(2**self.n_players)
        game_lookup = {}
        for i, T in enumerate(powerset(self._grand_coalition_set)):
            game_values[i] = 0
            game_lookup[tuple(T)] = i
            for S in powerset(
                self._grand_coalition_set, min_size=self.min_order, max_size=self.order
            ):
                t_cap_s = len(set(S).intersection(set(T)))
                s = len(S)
                game_values[i] += values[S] * self.get_coef(s, t_cap_s)  # *(-0.5)
        return game_values, game_lookup

    def get_coef(self, s, t_cap_s):
        val = 0
        if t_cap_s == 0:
            return val
        else:
            for r in range(1, t_cap_s + 1):
                val += binom(t_cap_s, r) * self.bernoulli[s - r]
            return val

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(np.shape(coalitions)[0])
        for row_index, row in enumerate(coalitions):
            values[row_index] = self.game_values[
                self.coalition_lookup_test[tuple(np.where(row)[0])]
            ]
        return values


def check_sum(values, i):
    output = 0
    for j in range(n_players):
        if i != j:
            output += values[
                tuple(
                    sorted(
                        (
                            i,
                            j,
                        )
                    )
                )
            ]
    return output


def shapley(values, i, k):
    output = 0
    for coal in powerset(N, min_size=k, max_size=k):
        if i not in coal:
            coal_array = np.zeros(n_players, dtype=bool)
            coal_array[list(coal)] = True
            coal_i = copy.copy(coal_array)
            coal_i[i] = True
            output += (
                1
                / (n_players * binom(n_players - 1, len(coal)))
                * (values(coal_i) - values(coal_array))
            )
    return output


if __name__ == "__main__":
    # read these values from the configuration file / or the printed benchmark configurations
    # game_identifier = "SentimentAnalysisLocalXAI"  # explains the sentiment of a sentence
    # game_identifier = "SOUM"
    # config_id = 1
    # n_player_id = 0
    # n_games = 3

    soum_game = SOUM(n=8, n_basis_games=100, min_interaction_size=1)

    n_players = soum_game.n_players
    N = set(range(n_players))

    order = 3
    exact_computer = ExactComputer(n_players=n_players, game_fun=soum_game)
    sii = exact_computer.shapley_base_interaction("SII", order=order)
    k_sii_transformed = aggregate_interaction_values(sii, order=order)
    k_sii = exact_computer.shapley_interaction("k-SII", order=order)
    sii_reversed = reverse_interaction_values(k_sii_transformed, order=order)
    fsii = exact_computer.shapley_interaction("FSII", order=order)
    fsii_reversed = reverse_interaction_values(fsii, order=order)
    fsii_back = aggregate_interaction_values(fsii_reversed, order=order)

    fsii_rand = copy.copy(fsii)
    fsii_rand.values = np.random.random(np.shape(fsii.values))
    fsii_rand_dict = {}
    for S, pos in fsii_rand.interaction_lookup.items():
        fsii_rand_dict[S] = fsii_rand.values[pos]
    fsii_rand.dict_values.update(fsii_rand_dict)

    approx_fsii = approx_game_fsii(n_players, fsii, order)
    approx_sii = approx_game_sii(n_players, fsii_reversed, order)
    approx_sii_2 = approx_game_sii(n_players, fsii_reversed, order, min_order=2)
    approx_sii_rand = approx_game_sii(n_players, fsii_rand, order, min_order=2)
    approx_exact_fsii = ExactComputer(n_players, approx_fsii)
    approx_exact_sii = ExactComputer(n_players, approx_sii)
    approx_exact_sii_2 = ExactComputer(n_players, approx_sii_2)
    approx_exact_sii_rand = ExactComputer(n_players, approx_sii_rand)

    for i in range(100):
        test = np.random.randint(2, size=n_players)
        print(approx_fsii(test), approx_sii(test), approx_fsii(test) - approx_sii(test))
    remainder_sv_sii = approx_exact_sii("SV", order=1)
    remainder_sv_sii_rand = approx_exact_sii_rand("SV", order=1)
    remainder_sv_sii_2 = approx_exact_sii_2("SV", order=1)
    remainder_sv_fsii = approx_exact_fsii("SV", order=1)
    print(remainder_sv_sii.values, remainder_sv_fsii.values)
    print(remainder_sv_sii_2.values, remainder_sv_sii_rand.values)
    sum = 0
    for k in range(n_players):
        val = shapley(approx_sii_2, 0, k)
        sum += val
        print(val, sum)

    order = 6
    val = 0
    for t_cap_s in range(0, order + 1):
        tmp = bernoulli_test(order, t_cap_s)
        val += tmp
        print(tmp, val)

    order = 6
    val = 0
    for t_cap_s in range(0, order):
        tmp = bernoulli_test(order, t_cap_s)
        val += tmp
        print(tmp, val)


def sum1(n, k):
    val = 0
    for t in range(k - 1, n):
        val += 1 / (n * binom(n - 1, t)) * binom(n - k, t - (k - 1))
    return val


def sum2(n, k):
    val = 0
    for t in range(n):
        for intersection_size in range(min(t, k - 1) + 1):
            for s in range(1, intersection_size + 1):
                coef1 = sp.special.bernoulli(n)[k - s]
                val += (
                    coef1
                    / (n * binom(n - 1, t))
                    * binom(intersection_size, s - 1)
                    * binom(n - k, t - intersection_size)
                )
    return val


def sum3(n, k):
    val = 0
    i = 0
    Q = tuple([q for q in range(k)])
    for T in powerset(N):
        if i not in T:
            for s in range(1, k):
                coef1 = sp.special.bernoulli(n)[k - s]
                t_cap_q = len(set(T).intersection(set(Q)))
                val += coef1 * binom(t_cap_q, s - 1) / (n * binom(n - 1, len(T)))
    return val


def sum4(n, k):
    val = 0
    i = 0
    Q = tuple([q for q in range(k)])
    for T in powerset(N):
        if i in T and set(Q).issubset(set(T)):
            val += 1 / (n * binom(n - 1, len(T) - 1))
    return val


def sum5(k, t_cap_q):
    val = 0
    bernoulli = sp.special.bernoulli(k)
    for s in range(t_cap_q + 1):
        val += bernoulli[k - 1 - s] * binom(t_cap_q, s)
    return val


def sum6(n, k):
    val = 0
    i = 0
    Q = tuple([q for q in range(k)])
    for intersection_size in range(k + 1):
        tmp = 0
        for T in powerset(N):
            t = len(T)
            pt = 1 / (n * binom(n - 1, t))
            if i not in T:
                t_cap_q = len(set(T).intersection(set(Q)))
                if t_cap_q == intersection_size:
                    for s in range(t_cap_q + 1):
                        coef1 = sp.special.bernoulli(n)[k - 1 - s]
                        tmp += coef1 * binom(t_cap_q, s) * pt
        val += tmp
        print(tmp, val)
    return val


print(sum1(n_players, 3), sum2(n_players, 3), sum3(n_players, 3), sum4(n_players, 3))

print(sum6(n_players, 7))
