import matplotlib.pyplot as plt

from shapiq.approximator import SVARM, PermutationSamplingSV
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
    game_identifier = "ImageClassifierLocalXAI"
    # game_identifier = "SOUM"
    config_id = 1
    n_player_id = 0
    n_games = 10

    games = load_games_from_configuration(
        game_class=game_identifier, n_player_id=n_player_id, config_id=config_id, n_games=n_games
    )

    games = list(games)  # convert to list (the generator is consumed)
    n_players = games[0].n_players
    N = set(range(n_players))

    # get the index and order
    index = "SV"
    order = 1
    save_path = "approximation_" + game_identifier + ".json"

    basis_gen = ExplanationBasisGenerator(N)

    explanation_add = basis_gen.generate_kadd_explanation_basis(max_order=1)
    explanation_kconj_1 = basis_gen.generate_ksym_explanation_basis(max_order=1)
    explanation_kadd_2 = basis_gen.generate_kadd_explanation_basis(max_order=2)
    explanation_kconj_2 = basis_gen.generate_ksym_explanation_basis(max_order=2)
    explanation_stoch_150 = basis_gen.generate_stochastic_explanation_basis(
        n_explanation_terms=150, conjugate=True
    )
    explanation_stoch_250 = basis_gen.generate_stochastic_explanation_basis(
        n_explanation_terms=250, conjugate=True
    )

    kadd_1 = ShapleyGAX(n=n_players, explanation_basis=explanation_add)
    kadd_1.name = "1-Add."
    kconj_1 = ShapleyGAX(n=n_players, explanation_basis=explanation_kconj_1)
    kconj_1.name = "1-Conj."
    kconj_2 = ShapleyGAX(n=n_players, explanation_basis=explanation_kconj_2)
    kconj_2.name = "2-Conj."
    kadd_2 = ShapleyGAX(n=n_players, explanation_basis=explanation_kadd_2)
    kadd_2.name = "2-Add."
    stoch_150 = ShapleyGAX(n=n_players, explanation_basis=explanation_stoch_150)
    stoch_150.name = "150-Stoch."
    stoch_250 = ShapleyGAX(n=n_players, explanation_basis=explanation_stoch_250)
    stoch_250.name = "250-Stoch."

    sv_approximators = [
        kadd_1,
        kconj_1,
        kadd_2,
        kconj_2,
        stoch_150,
        stoch_250,
        SVARM(n=n_players, random_state=42),
        PermutationSamplingSV(n=n_players, random_state=42),
    ]

    results = run_benchmark(
        index=index,
        order=order,
        games=games,
        approximators=sv_approximators,
        save_path=save_path,
        # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
        budget_steps=[1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000],
        rerun_if_exists=False,  # if True, the benchmark will rerun the approximators even if the results file exists
        n_jobs=14,  # number of parallel jobs
    )

    plt.figure()
    plot_approximation_quality(results, metric="Precision@5", log_scale_y=True)
    plt.show()
