"""This script runs all benchmarks for all pre-computed configurations configuration."""

import argparse

from shapiq.games.benchmark.run import run_benchmark_from_configuration

if __name__ == "__main__":

    # default values
    game = "SentimentAnalysisLocalXAI"
    config_id = 0
    n_player_id = 0
    index = "k-SII"
    order = 2
    n_games = 2
    n_jobs = 1

    # parse arguments if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default=game)
    parser.add_argument("--config_id", type=int, default=config_id)
    parser.add_argument("--n_player_id", type=int, default=n_player_id)
    parser.add_argument("--n_games", type=int, default=n_games, help="-1 for all games")
    parser.add_argument("--index", type=str, default=index)
    parser.add_argument("--order", type=int, default=order)
    parser.add_argument("--n_jobs", type=int, default=n_jobs)
    args = parser.parse_args()

    # parse arguments
    game = args.game
    config_id = args.config_id
    n_player_id = args.n_player_id
    n_games = args.n_games
    if n_games == -1:
        n_games = None
    index = args.index
    order = args.order
    n_jobs = args.n_jobs

    run_benchmark_from_configuration(
        index=index,
        order=order,
        game_class=game,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
        n_jobs=n_jobs,
        max_budget=2_000,
    )
