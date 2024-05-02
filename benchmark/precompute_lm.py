"""This module contains helper functions to pre-compute Sentiment Analysis games from the simplified
IMDB dataset."""

import os

import pandas as pd

from shapiq.games.benchmark import SentimentAnalysisLocalXAI
from shapiq.games.benchmark import pre_compute_and_store_from_list, get_game_files


def pre_compute_imdb(n_games: int, n_players: int, n_jobs: int = 1) -> None:
    """Loads the IMDB dataset and pre-computes the values for the sentiment analysis game.

    Args:
        n_games: The number of games to pre-compute the values for.
        n_players: The number of players in the game.
        n_jobs: The number of parallel jobs to run. Default is 1.
    """

    # load the IMDB dataset
    imdb_data = pd.read_csv(os.path.join("data", "simplified_imdb.csv"))
    imdb_data = imdb_data[imdb_data["length"] == n_players]

    # make text column into unique identifier by taking the first letter of each word
    imdb_data["game_id"] = imdb_data["text"].apply(
        lambda x: "".join([word[0] for word in x.split()])
    )

    # read the games that have already been pre-computed
    all_game_files = get_game_files(SentimentAnalysisLocalXAI, n_players=n_players)
    # get game_ids from the files
    all_game_ids = set([file.split(".")[0] for file in all_game_files])
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
    # save_dir = os.path.join("precomputed", "SentimentAnalysis(Game)")
    print(f"Precomputing {n_games} games with {n_players} players.")
    pre_compute_and_store_from_list(games, game_ids=game_ids, n_jobs=n_jobs)


if __name__ == "__main__":

    pre_compute_imdb(n_games=1, n_players=10, n_jobs=1)
