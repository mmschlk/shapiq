import matplotlib.pyplot as plt

from shapiq import kADDSHAP
from shapiq.approximator import SVARM, KernelSHAP
from shapiq.approximator.regression.shapleygax import ShapleyGAX

# plot the results
from shapiq.benchmark import (
    load_games_from_configuration,
    plot_approximation_quality,
    run_benchmark,
)
from shapiq.utils import powerset

if __name__ == "__main__":
    # read these values from the configuration file / or the printed benchmark configurations
    # game_identifier = "SentimentAnalysisLocalXAI"  # explains the sentiment of a sentence
    # game_identifier = "ImageClassifierLocalXAI"
    game_identifier = "SOUM"
    config_id = 1
    n_player_id = 0
    n_games = 10

    games = load_games_from_configuration(
        game_class=game_identifier, n_player_id=n_player_id, config_id=config_id, n_games=n_games
    )

    games = list(games)  # convert to list (the generator is consumed)
    n_players = games[0].n_players

    # get the index and order
    index = "SV"
    order = 1
    save_path = "sv_benchmark_results.json"

    gax_interactions_individuals = {}
    N = set(range(n_players))

    pos = 0
    for S in powerset(N, max_size=2):
        gax_interactions_individuals[S] = pos
        pos += 1
        S_complement = tuple(sorted(N - set(S)))
        gax_interactions_individuals[S_complement] = pos
        pos += 1
    shapleyGAX_individuals = ShapleyGAX(n=n_players, gax_interactions=gax_interactions_individuals)

    sv_approximators = [
        KernelSHAP(n=n_players, random_state=42),
        SVARM(n=n_players, random_state=42),
        # PermutationSamplingSV(n=n_players, random_state=42),
        shapleyGAX_individuals,
        kADDSHAP(n=n_players, random_state=42, max_order=2),
        # symSHAP(n=n_players, random_state=42, max_order=2),
        # FaithSHAP(n=n_players, random_state=42, max_order=2, mirrored=True),
    ]

    results = run_benchmark(
        index=index,
        order=order,
        games=games,
        approximators=sv_approximators,
        save_path=save_path,
        # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
        budget_steps=[750, 1000, 1500, 2000, 3000, 4000, 6000, 8000],
        rerun_if_exists=True,  # if True, the benchmark will rerun the approximators even if the results file exists
        n_jobs=6,  # number of parallel jobs
    )

    plt.figure()
    plot_approximation_quality(
        results,
        log_scale_y=True,
    )
    plt.show()
