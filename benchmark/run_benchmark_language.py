"""This script runs the benchmark for the language model as an example."""

from tqdm.auto import tqdm

from shapiq.approximator import (
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAPIQ,
    PermutationSamplingSII,
)
from shapiq.games.benchmark import run_benchmark
from shapiq.games.benchmark.benchmark_config import (
    BENCHMARK_CONFIGURATIONS,
    get_game_class_from_name,
    load_games_from_configuration,
)

if __name__ == "__main__":

    game = "SentimentAnalysisLocalXAI"
    config_id = 0
    n_player_id = 0
    n_games = 2

    index = "k-SII"
    order = 2

    # get the correct configuration
    game_class = get_game_class_from_name(game)
    run_config = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["configurations"][config_id]

    # load the games
    games_generator = load_games_from_configuration(
        game_class, run_config, n_player_id=n_player_id, n_games=n_games
    )
    games_list = list(games_generator)
    n_players = games_list[0].n_players
    game_id = games_list[0].game_id

    # check that all games are actually pre-computed
    assert all(game.precomputed for game in games_list)
    assert all(game.is_normalized for game in games_list)
    print(f"Loaded {len(games_list)} games. All of them are pre-computed and normalized.")

    # get approximators
    approximators = [
        KernelSHAPIQ(n=n_players, index=index, max_order=order),
        InconsistentKernelSHAPIQ(n=n_players, index=index, max_order=order),
        SHAPIQ(n=n_players, index=index, max_order=order),
        SVARMIQ(n=n_players, index=index, max_order=order),
        PermutationSamplingSII(n=n_players, index=index, max_order=order),
    ]

    # get the exact values
    print("Computing the exact values for the games.")
    gt_values = [game.exact_values(index=index, order=order) for game in tqdm(games_list)]

    # set the benchmark name
    benchmark_name = "_".join([game, game_id, index, str(order)])

    # run the benchmark
    run_benchmark(
        approximators=approximators,
        games=games_list,
        gt_values=gt_values,
        max_budget=2_000,
        n_jobs=6,
        benchmark_name=benchmark_name,
    )
