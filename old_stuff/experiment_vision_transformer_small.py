"""This script evaluates the performance of different approximators on the Vision Transformer
model with grouped patches. It uses the pre-computed benchmark games from shapiq."""
import copy

import pandas as pd

import shapiq
from shapiq.benchmark import (
    load_games_from_configuration,
    print_benchmark_configurations,
)
from shapiq.approximator.regression.polyshap import (
    ShapleyGAX,
    ExplanationBasisGenerator,
)
from experiment_vision_transformer import _run_approximation

RANDOM_SEED = 42

if __name__ == "__main__":
    budgets = [2_000, 5_000, 10_000, 20_000]
    n_games = 20

    print_benchmark_configurations()
    games = load_games_from_configuration(
        game_class="ImageClassifierLocalXAI",
        n_player_id=2,
        config_id=1,
        n_games=n_games,
    )
    games = list(games)  # convert to list (the generator is consumed)
    n_players = games[0].n_players
    print(f"Loaded {len(games)} games with {n_players} players.")

    # evaluate the approximators
    results = []
    for game_id, game in enumerate(games):
        computer = shapiq.ExactComputer(game=game, n_players=game.n_players)
        gt_sv = computer(index="SV", order=1)

        for budget in budgets:
            # run kernel SHAP ----------------------------------------------------------------------
            name = "KernelSHAP"
            basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
            explanation_basis = basis_gen.generate_kadd_explanation_basis(1)
            approximator = ShapleyGAX(
                n=game.n_players,
                random_state=RANDOM_SEED,
                explanation_basis=explanation_basis,
            )
            result = _run_approximation(
                approximator,
                game,
                gt_sv,
                budget,
                name,
                str(game_id),
                print_estimate=False,
            )
            results.append(copy.deepcopy(result))

            # run SHAPley GAX 50 -------------------------------------------------------------------
            name = "ShapleyGAX (50)"
            basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
            explanation_basis = basis_gen.generate_stochastic_explanation_basis(50)
            approximator = ShapleyGAX(
                n=game.n_players,
                random_state=RANDOM_SEED,
                explanation_basis=explanation_basis,
            )
            result = _run_approximation(
                approximator,
                game,
                gt_sv,
                budget,
                name,
                str(game_id),
                print_estimate=False,
            )
            results.append(copy.deepcopy(result))

            # run SHAPley GAX 2add -----------------------------------------------------------------
            name = "ShapleyGAX (2-add)"
            basis_gen = ExplanationBasisGenerator(N=set(range(game.n_players)))
            explanation_basis = basis_gen.generate_kadd_explanation_basis(2)
            approximator = ShapleyGAX(
                n=game.n_players,
                random_state=RANDOM_SEED,
                explanation_basis=explanation_basis,
            )
            result = _run_approximation(
                approximator,
                game,
                gt_sv,
                budget,
                name,
                str(game_id),
                print_estimate=False,
            )
            results.append(copy.deepcopy(result))

            # run PermutationSamplingSV -------------------------------------------------------------
            name = "PermutationSamplingSV"
            approximator = shapiq.PermutationSamplingSV(
                n=game.n_players, random_state=RANDOM_SEED
            )
            result = _run_approximation(
                approximator,
                game,
                gt_sv,
                budget,
                name,
                str(game_id),
                print_estimate=False,
            )
            results.append(copy.deepcopy(result))

            # run SVARM ----------------------------------------------------------------------------
            name = "SVARM"
            approximator = shapiq.SVARM(n=game.n_players, random_state=RANDOM_SEED)
            result = _run_approximation(
                approximator,
                game,
                gt_sv,
                budget,
                name,
                str(game_id),
                print_estimate=False,
            )
            results.append(copy.deepcopy(result))

            results_df = pd.DataFrame(results)
            results_df.to_csv("results_vit_grouped_patches.csv", index=False)

    print("Done.")
