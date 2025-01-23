import matplotlib.pyplot as plt
import numpy as np

from shapiq.approximator.regression.shapleygax import ExplanationBasisGenerator, ShapleyGAX

# plot the results
from shapiq.benchmark import (
    load_games_from_configuration,
    plot_approximation_quality,
    run_benchmark,
)

if __name__ == "__main__":
    # read these values from the configuration file / or the printed benchmark configurations
    # game_identifier = "SentimentAnalysisLocalXAI"  # explains the sentiment of a sentence
    # game_identifier = "ImageClassifierLocalXAI"
    # game_identifier = "SOUM"
    game_identifier = "SentimentAnalysisLocalXAI"
    config_id = 1
    n_player_id = 0
    n_games = 5

    games = load_games_from_configuration(
        game_class=game_identifier, n_player_id=n_player_id, config_id=config_id, n_games=n_games
    )

    games = list(games)  # convert to list (the generator is consumed)
    n_players = games[0].n_players
    N = set(range(n_players))

    # get the index and order
    index = "SV"
    order = 1
    save_path = "interactionanalysis_" + game_identifier + ".json"

    budget_steps = np.arange(1000, 10000, 1000)
    budget_steps = np.hstack((budget_steps, [2**n_players]))
    n_explanations = [
        n_players + 1,
        20,
        50,
        100,
        250,
        500,
        1000,
        2000,
        # 4000,
        # 8000,
        2**n_players,
    ]

    sv_approximators = []
    basis_gen = ExplanationBasisGenerator(N)
    # for budget in budget_steps:
    for n_explanation_terms in n_explanations:
        explanation_basis = basis_gen.generate_stochastic_explanation_basis(
            n_explanation_terms=n_explanation_terms
        )
        computer = ShapleyGAX(n=n_players, explanation_basis=explanation_basis)
        computer.name = str(f"{n_explanation_terms}" + "-Stoch.")
        sv_approximators.append(computer)

    results = run_benchmark(
        index=index,
        order=order,
        games=games,
        approximators=sv_approximators,
        save_path=save_path,
        # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
        budget_steps=budget_steps,
        rerun_if_exists=False,  # if True, the benchmark will rerun the approximators even if the results file exists
        n_jobs=14,  # number of parallel jobs
    )

    plt.figure()
    plot_approximation_quality(results, metric="MSE", log_scale_y=True)
    plt.show()
