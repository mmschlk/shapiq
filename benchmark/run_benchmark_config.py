"""This script runs the benchmark from a specified configuration."""

from shapiq.games.benchmark.run import run_benchmark_from_configuration

if __name__ == "__main__":

    # default values
    game = "SentimentAnalysisLocalXAI"
    config_id = 0
    n_player_id = 0
    n_games = 2
    index = "k-SII"
    order = 2

    # parse arguments if provided
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--game", type=str, default=game)

    run_benchmark_from_configuration(
        index=index,
        order=order,
        game_class=game,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
        n_jobs=1,
        max_budget=2_000,
    )
