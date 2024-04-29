"""This module contains the benchmark conducted on the language model game."""

import os
import sys
from typing import Optional
import glob
import multiprocessing as mp

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from shapiq.games import Game
from shapiq.games.benchmark import SentimentAnalysisLocalXAI
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import count_interactions


def _remove_empty_value(interaction: InteractionValues) -> InteractionValues:
    """Manually sets the empty value to zero.

    Args:
        interaction: The interaction values to remove the empty value from.

    Returns:
        The interaction values without the empty value.
    """
    try:
        empty_index = interaction.interaction_lookup[()]
        interaction.values[empty_index] = 0
    except KeyError:
        return interaction

    return interaction


def _average_metric(metric_value, interaction: InteractionValues) -> float:
    """Average the interaction values over all interactions.

    Args:
        interaction: The interaction values to average.

    Returns:
        The average interaction value.
    """
    n_interactions = count_interactions(
        n=interaction.n_players, min_order=interaction.min_order, max_order=interaction.max_order
    )
    return metric_value / n_interactions


def compute_mse(
    ground_truth: InteractionValues, estimated: InteractionValues, average: bool = True
) -> float:
    """Compute the mean squared error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    mse = float(np.mean(diff_values**2))

    if average:
        mse = _average_metric(mse, ground_truth)

    return mse


def compute_mae(
    ground_truth: InteractionValues, estimated: InteractionValues, average: bool = True
) -> float:
    """Compute the mean absolute error between two interaction values."""
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    mae = float(np.mean(np.abs(diff_values)))

    if average:
        mae = _average_metric(mae, ground_truth)

    return mae


def compute_precision_at_k(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> float:
    """Compute the precision at k between two interaction values."""
    ground_truth_values = _remove_empty_value(ground_truth)
    estimated_values = _remove_empty_value(estimated)

    top_k, _ = ground_truth_values.get_top_k_interactions(k=k)
    top_k_estimated, _ = estimated_values.get_top_k_interactions(k=k)

    precision_at_k = len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k

    return precision_at_k


def pre_compute_and_store(
    game: Game, save_dir: Optional[str] = None, game_id: Optional[str] = None
) -> str:
    """Pre-compute the values for the given game and store them in a file.

    Args:
        game: The game to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_id: The ID of the game. If not provided, the ID is determined at random.

    Returns:
        The path to the file where the values are stored.
    """

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), game.game_name, str(game.n_players))
    else:  # check if n_players is in the path
        if str(game.n_players) not in save_dir:
            save_dir = os.path.join(save_dir, str(game.n_players))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if game_id is None:
        game_id = str(hash(game))[:8]

    save_path = os.path.join(save_dir, game_id)

    game.precompute()
    game.save_values(path=save_path)
    return save_path


def get_all_metrics(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> dict:
    """Get all metrics for the interaction values."""
    metrics = {}
    metrics["MSE"] = compute_mse(ground_truth, estimated)
    metrics["MAE"] = compute_mae(ground_truth, estimated)
    metrics["Precision@k"] = compute_precision_at_k(ground_truth, estimated, k=k)
    metrics["Precision@k_k"] = k
    return metrics


def pre_compute_and_store_from_list(
    games: list[Game],
    save_dir: Optional[str] = None,
    game_ids: Optional[list[str]] = None,
    n_jobs: int = 1,
) -> list[str]:
    """Pre-compute the values for the games stored in the given file.

    Args:
        games: The games to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_ids: The IDs of the games. If not provided, the IDs are determined at random.
        n_jobs: The number of parallel jobs to run.

    Returns:
        The paths to the files where the values are stored.
    """

    if game_ids is None:
        game_ids = [None] * len(games)

    if n_jobs == 1:
        return [
            pre_compute_and_store(game, save_dir, game_id) for game, game_id in zip(games, game_ids)
        ]

    with mp.Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    pre_compute_and_store,
                    [(game, save_dir, game_id) for game, game_id in zip(games, game_ids)],
                ),
                total=len(games),
            )
        )

    return results


def pre_compute_imdb(n_games: int, n_players: int) -> None:
    """Loads the IMDB dataset and pre-computes the values for the sentiment analysis game.

    Args:
        n_games: The number of games to pre-compute the values for.
        n_players: The number of players in the game.
    """

    # load the IMDB dataset
    imdb_data = pd.read_csv(os.path.join("data", "simplified_imdb.csv"))
    imdb_data = imdb_data[imdb_data["length"] == n_players]

    # make text column into unique identifier by taking the first letter of each word
    imdb_data["game_id"] = imdb_data["text"].apply(
        lambda x: "".join([word[0] for word in x.split()])
    )

    # read the games that have already been pre-computed
    try:
        all_files = os.listdir(
            os.path.join("precomputed", "SentimentAnalysis(Game)", str(n_players))
        )
    except FileNotFoundError:
        all_files = []
    # get game_ids from the files
    all_game_ids = set([file.split(".")[0] for file in all_files])
    print(f"Found {len(all_game_ids)} games precomputed.")

    # get the games that have not been pre-computed
    imdb_data = imdb_data[~imdb_data["game_id"].isin(all_game_ids)]

    # sample random games
    imdb_data = imdb_data.sample(n=n_games)

    # get the games
    games, game_ids = [], []
    for _, row in imdb_data.iterrows():
        game = SentimentAnalysisLocalXAI(input_text=row["text"], verbose=True)
        games.append(game)
        game_ids.append(row["game_id"])

    # pre-compute the values for the games
    save_dir = os.path.join("precomputed", "SentimentAnalysis(Game)")
    print(f"Precomputing {n_games} games with {n_players} players.")
    pre_compute_and_store_from_list(games, save_dir=save_dir, game_ids=game_ids, n_jobs=1)


if __name__ == "__main__":

    PRE_COMPUTE_IMDB = False
    N_PLAYERS = 10
    N_GAMES = 5
    MAX_ORDER = 2
    INDEX = "SII"

    if N_PLAYERS == 10:
        BUDGETS = [400, 500, 600, 700, 800, 900, 2**10]
    else:
        BUDGETS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 2**14]

    # sanitize input parameters
    assert INDEX in ["SII", "k-SII"], f"Invalid index {INDEX}. Must be either 'SII' or 'k-SII'."

    if PRE_COMPUTE_IMDB:
        # pre-compute values for multiple games
        pre_compute_imdb(n_games=10, n_players=N_PLAYERS)
        print("Pre-computed games.")
        sys.exit(0)

    # load the pre-computed games
    files = glob.glob(os.path.join("precomputed", "SentimentAnalysis(Game)", str(N_PLAYERS), "*"))
    games = [Game(path_to_values=file) for file in files]

    print("Loaded", len(games), "games.")

    # load benchmark approximators
    from shapiq.approximator import KernelSHAPIQ
    from shapiq.approximator import InconsistentKernelSHAPIQ
    from shapiq.approximator import SHAPIQ
    from shapiq.approximator import SVARMIQ
    from shapiq.approximator import PermutationSamplingSII

    approximators = []
    kernel_shap_iq = KernelSHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    approximators.append(kernel_shap_iq)
    inconsistent_kernel_shap_iq = InconsistentKernelSHAPIQ(
        n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER
    )
    approximators.append(inconsistent_kernel_shap_iq)
    shap_iq = SHAPIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    approximators.append(shap_iq)
    svarm_iq = SVARMIQ(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    approximators.append(svarm_iq)
    permutation = PermutationSamplingSII(n=N_PLAYERS, index=INDEX, max_order=MAX_ORDER)
    approximators.append(permutation)

    # print experiment parameters
    print(f"Experiment parameters:")
    print(f"Number of players: {N_PLAYERS}")
    print(f"Number of games: {min(len(games), N_GAMES)}")
    print(f"Interaction index: {INDEX}")
    print(f"Interaction order: {MAX_ORDER}")
    print(f"Budgets: {BUDGETS}, resulting in {sum(BUDGETS)} evaluations per approximator per game.")

    pbar = tqdm(total=sum(BUDGETS) * min(len(games), N_GAMES) * len(approximators))

    results = []
    for budget in BUDGETS:
        for i, game in enumerate(games):
            if i >= N_GAMES:
                break
            # print("Benchmarking", game.game_name, "with", game.n_players, "players.")
            exact_values = game.exact_values(index=INDEX, order=MAX_ORDER)
            for approximator in approximators:
                approx_values = approximator.approximate(budget=budget, game=game)
                for order in range(1, exact_values.max_order + 1):
                    exact_values_order = exact_values.get_n_order(order)
                    approx_values_order = approx_values.get_n_order(order)
                    metrics = get_all_metrics(exact_values_order, approx_values_order)
                    run_result = {
                        "game": game.game_name,
                        "game_id": game.game_id,
                        "n_players": game.n_players,
                        "budget": budget,
                        "approximator": approximator.__class__.__name__,
                        "order": order,
                    }
                    run_result.update(metrics)
                    results.append(run_result)
                pbar.update(budget)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
