import matplotlib.pyplot as plt

from shapiq.approximator import SVARM, KernelSHAP, PermutationSamplingSV, kADDSHAP
from shapiq.approximator.regression.symshap import symSHAP

# plot the results
from shapiq.benchmark import (
    load_games_from_configuration,
    plot_approximation_quality,
    run_benchmark,
)

if __name__ == "__main__":
    # read these values from the configuration file / or the printed benchmark configurations
    game_identifier = "SentimentAnalysisLocalXAI"  # explains the sentiment of a sentence
    # game_identifier = "SOUM"
    config_id = 1
    n_player_id = 0
    n_games = 3

    games = load_games_from_configuration(
        game_class=game_identifier, n_player_id=n_player_id, config_id=config_id, n_games=n_games
    )

    games = list(games)  # convert to list (the generator is consumed)
    n_players = games[0].n_players

    # get the index and order
    index = "SV"
    order = 1
    save_path = "sv_benchmark_results.json"

    sv_approximators = [
        KernelSHAP(n=n_players, random_state=42),
        SVARM(n=n_players, random_state=42),
        PermutationSamplingSV(n=n_players, random_state=42),
        kADDSHAP(n=n_players, random_state=42, max_order=3),
        symSHAP(n=n_players, random_state=42, max_order=3),
    ]

    results = run_benchmark(
        index=index,
        order=order,
        games=games,
        approximators=sv_approximators,
        save_path=save_path,
        # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
        budget_steps=[500, 700, 1000, 2500, 5000, 10000, 20000, 2**n_players],
        rerun_if_exists=True,  # if True, the benchmark will rerun the approximators even if the results file exists
        n_jobs=6,  # number of parallel jobs
    )

    plot_approximation_quality(results, log_scale_y=True)
    plt.show()
