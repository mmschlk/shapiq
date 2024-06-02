import numpy as np
import pandas as pd
import tqdm
from scipy.special import bernoulli, binom

from shapiq.exact import ExactComputer
from shapiq.games.benchmark.benchmark_config import load_games_from_configuration
from shapiq.games.benchmark.plot import get_game_title_name
from shapiq.games.benchmark.synthetic import SOUM
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset


def bernoulli_lambda(interaction_size, intersection_size):
    if intersection_size == 0:
        return 0
    else:
        rslt = 0
        for size in range(1, intersection_size + 1):
            rslt += bernoulli(interaction_size - size)[-1] * binom(intersection_size, size)
        return rslt


def bernoulli_lambda_with_zero(interaction_size, intersection_size):
    if intersection_size == 0:
        return 0
    else:
        rslt = 0
        for size in range(intersection_size + 1):
            rslt += bernoulli(interaction_size - size)[-1] * binom(intersection_size, size)
        return rslt


def _get_weight(n, coalition_size, weighting_scheme):
    if weighting_scheme == "uniform":
        return (1 / 2) ** n
    elif weighting_scheme == "Shapley kernel":
        if coalition_size == n or coalition_size == 0:
            return 1
        else:
            return (n - 1) / (binom(n, coalition_size) * coalition_size * (n - coalition_size))
    else:
        raise ValueError(f"Weighting Scheme {weighting_scheme} not supported.")


def get_approximation_weights(n, weighting_scheme):
    approximation_weights = np.zeros(2**n, dtype=float)
    grand_coalition_set = set(range(n))
    for coalition_pos, coalition in enumerate(powerset(grand_coalition_set)):
        approximation_weights[coalition_pos] = _get_weight(n, len(coalition), weighting_scheme)
    return approximation_weights


def approximation_via_sii(interaction_index, order):
    n_players = interaction_index.n_players
    grand_coalition_set = set(range(n_players))
    approximations = {}
    for current_order in range(1, order + 1):
        approximation_lookup = {}
        approximation_values = np.zeros(2**n_players)
        for coalition_pos, coalition in enumerate(powerset(grand_coalition_set)):
            approximation_lookup[coalition] = coalition_pos
            for interaction in powerset(grand_coalition_set, min_size=1, max_size=current_order):
                interaction_size = len(interaction)
                intersection_size = len(set(interaction).intersection(set(coalition)))
                approximation_values[coalition_pos] += (
                    bernoulli_lambda(interaction_size, intersection_size)
                    * interaction_index[interaction]
                )
        baseline_value = approximation_values[approximation_lookup[tuple()]]
        # if current_order == 1:
        approximations[current_order] = InteractionValues(
            index=interaction_index.index,
            max_order=n_players,
            n_players=n_players,
            min_order=0,
            baseline_value=baseline_value,
            interaction_lookup=approximation_lookup,
            values=approximation_values,
        )
    return approximations


def approximated_game(interaction_index):
    n_players = interaction_index.n_players
    grand_coalition_set = set(range(n_players))
    approximation_lookup = {}
    approximation_values = np.zeros(2**n_players)
    for coalition_pos, coalition in enumerate(powerset(grand_coalition_set)):
        approximation_lookup[coalition] = coalition_pos
        for interaction in powerset(coalition):
            approximation_values[coalition_pos] += interaction_index[interaction]

    baseline_value = approximation_values[approximation_lookup[tuple()]]
    approximation = InteractionValues(
        index=interaction_index.index,
        max_order=n_players,
        n_players=n_players,
        min_order=0,
        baseline_value=baseline_value,
        interaction_lookup=approximation_lookup,
        values=approximation_values,
    )
    return approximation


def convert_game_to_interaction(exact_computer):
    baseline_value = exact_computer.game_values[exact_computer.coalition_lookup[tuple()]]
    gt_game_values = InteractionValues(
        values=exact_computer.game_values,
        index="Moebius",
        interaction_lookup=exact_computer.coalition_lookup,
        n_players=exact_computer.n,
        min_order=0,
        max_order=exact_computer.n,
        baseline_value=baseline_value,
    )
    return gt_game_values


def get_approximations_for_game(game):
    n_players = game.n_players
    exact_computer = ExactComputer(n_players=n_players, game_fun=game)
    approximations = {}
    game_values = convert_game_to_interaction(exact_computer)

    for index in INDICES:
        approximations[index] = {}
        for order in tqdm.tqdm(range(1, n_players + 1), total=n_players, desc=index):
            interactions = exact_computer.shapley_interaction(index=index, order=order)
            approximations[index][order] = approximated_game(interactions)
    return approximations, game_values


def shapley_residual(game, player):
    exact_computer = ExactComputer(game.n_players, game)
    game_values = exact_computer.game_values
    coalition_lookup = exact_computer.coalition_lookup
    grand_coalition_without_player = set(game.grand_coalition) - set(player)
    marginal_contributions = np.zeros(2 ** (game.n_players - 1))
    for coalition in powerset(grand_coalition_without_player):
        coalition_with_player = tuple(sorted(set(coalition) + set((player,))))
        coalition_with_player_pos = coalition_lookup[coalition_with_player]
        coalition_pos = coalition_lookup[coalition]
        marginal_contributions[coalition_pos] = (
            game_values[coalition_with_player_pos] - game_values[coalition_pos]
        )


def get_errors_for_game(approximations, game_values, n_players):
    errors_for_game = {}
    weighted_r2_for_game = {}
    # Compute errors for complexity versus accuracy
    for weighting_scheme in WEIGHTING_SCHEMES:
        errors_for_game[weighting_scheme] = {}
        weighted_r2_for_game[weighting_scheme] = {}
        least_square_weights = get_approximation_weights(n_players, weighting_scheme)
        for index in INDICES:
            errors_for_game[weighting_scheme][index] = {}
            weighted_r2_for_game[weighting_scheme][index] = {}
            for order in range(1, n_players + 1):
                game_values.index = approximations[index][
                    order
                ].index  # match indices as a workaround
                errors_for_game[weighting_scheme][index][order] = np.sum(
                    least_square_weights * (approximations[index][order] - game_values).values ** 2
                )
                weighted_average = np.sum(
                    least_square_weights * approximations[index][order].values
                ) / np.sum(least_square_weights)
                total_sum_of_squares = np.sum(
                    least_square_weights
                    * (approximations[index][order] - weighted_average).values ** 2
                )
                weighted_r2_for_game[weighting_scheme][index][order] = (
                    1 - errors_for_game[weighting_scheme][index][order] / total_sum_of_squares
                )

    return errors_for_game, weighted_r2_for_game


def save_results(errors, weighted_r2, game_id, game_title):
    tmp = []
    for weighting_scheme in WEIGHTING_SCHEMES:
        for index in INDICES:
            df = pd.DataFrame()
            df["l2"] = errors[weighting_scheme][index]
            df["r2"] = weighted_r2[weighting_scheme][index]
            df["order"] = df.index
            df["index"] = index
            df["weighting_scheme"] = weighting_scheme
            df["game_title"] = game_title
            tmp.append(df)
        df_results = pd.concat(tmp)
        df_results.to_csv("results/" + game_id + ".csv")


if __name__ == "__main__":
    n_players = 10

    INDICES = ["k-SII", "STII", "FSII", "FBII"]
    INDEX_COLORS = {
        "FBII": "#ef27a6",
        "FSII": "#7d53de",
        "STII": "#00b4d8",
        "k-SII": "#ff6f00",
        "SII": "#ffba08",
    }
    XLABEL = "Explanation Order"

    WEIGHTING_SCHEMES = ["uniform", "Shapley kernel"]
    errors = {}
    weighted_r2 = {}
    approximations = {}

    RUN_SYNTHETIC_INTERACTION_EXPERIMENT = False
    RUN_BENCHMARK_GAMES_EXPERIMENT = True

    if RUN_SYNTHETIC_INTERACTION_EXPERIMENT:
        INTERACTION_RANGE = range(2, n_players + 1)
        # Single interaction per size
        n_players = 10
        for interaction_size in INTERACTION_RANGE:
            game = SOUM(
                n=n_players,
                n_basis_games=1,
                min_interaction_size=interaction_size,
                max_interaction_size=interaction_size,
            )
            approximations, game_values = get_approximations_for_game(game)
            errors[interaction_size], weighted_r2[interaction_size] = get_errors_for_game(
                approximations, game_values, n_players
            )

            save_results(
                errors[interaction_size],
                weighted_r2[interaction_size],
                "SOUM_" + str(interaction_size),
                "SOUM",
            )

    if RUN_BENCHMARK_GAMES_EXPERIMENT:
        N_GAMES = 30
        game_names = [
            # "AdultCensusClusterExplanation",
            "AdultCensusDatasetValuation",
            "AdultCensusEnsembleSelection",
            "AdultCensusFeatureSelection",
            "AdultCensusGlobalXAI",
            "AdultCensusLocalXAI",
            "AdultCensusRandomForestEnsembleSelection",
            "AdultCensusUnsupervisedData",
            "BikeSharingClusterExplanation",
            "BikeSharingDatasetValuation",
            "BikeSharingEnsembleSelection",
            "BikeSharingFeatureSelection",
            "BikeSharingGlobalXAI",
            "BikeSharingLocalXAI",
            "BikeSharingRandomForestEnsembleSelection",
            "BikeSharingUnsupervisedData",
            "CaliforniaHousingClusterExplanation",
            "CaliforniaHousingDatasetValuation",
            "CaliforniaHousingEnsembleSelection",
            "CaliforniaHousingFeatureSelection",
            "CaliforniaHousingGlobalXAI",
            "CaliforniaHousingLocalXAI",
            "CaliforniaHousingRandomForestEnsembleSelection",
            "CaliforniaHousingUnsupervisedData",
            "ImageClassifierLocalXAI",
        ]

        game_names_dict = {}
        game_list = []
        for game_name in game_names:
            if game_name == "ImageClassifierLocalXAI":
                n_player_id = 1
            else:
                n_player_id = 0

            loaded_game = load_games_from_configuration(
                game_class=game_name, configuration=1, n_player_id=n_player_id, n_games=N_GAMES
            )
            game_list.append(loaded_game)

        for i, game_iterator in enumerate(game_list):
            game_title = get_game_title_name(game_names[i])
            print(game_title)
            for game in game_iterator:
                game_id = game_names[i] + "_" + game.game_id
                n_players = game.n_players
                if n_players <= 10:
                    approximations, game_values = get_approximations_for_game(game)
                    errors[game_id], weighted_r2[game_id] = get_errors_for_game(
                        approximations, game_values, game.n_players
                    )
                    save_results(errors[game_id], weighted_r2[game_id], game_id, game_title)
