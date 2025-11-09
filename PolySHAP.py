from __future__ import annotations

from shapiq import ExactComputer
from shapiq.approximator.regression.polyshap import (
    ExplanationFrontierGenerator,
    PolySHAP,
)

from shapiq import KernelSHAP

if __name__ == "__main__":
    import numpy as np

    from shapiq.games.benchmark.synthetic import SOUM

    n_players = 14
    random_state = 42
    budget = 2**n_players

    # Initialize the SOUM game
    game = SOUM(n=n_players, n_basis_games=150, random_state=random_state)

    explanation_frontier = ExplanationFrontierGenerator(set(range(n_players)))
    shapley_frontier = explanation_frontier.generate_kadd(max_order=1)

    kernelshap=  KernelSHAP(n=n_players,pairing_trick=True,replacement=False,random_state=random_state)
    kernelshap.approximate(budget=budget, game=game)

    approx = PolySHAP(
        explanation_frontier=shapley_frontier,
        n=n_players,
        random_state=random_state,
        replacement=False,
        pairing_trick=True,
    )

    polyshap_estimates = approx.approximate(budget=budget, game=game)

    print("runtime Old",kernelshap.runtime_last_approximate_run)
    print("runtime",approx.runtime_last_approximate_run)

    exact_computer = ExactComputer(n_players, game)
    ground_truth = exact_computer(index="SV", order=1)
    print("Approximation:")
    print(polyshap_estimates.values)

    # compare mse between approx and ground-truth
    mse = np.sum((polyshap_estimates.values - ground_truth.values) ** 2) / n_players

    print(f"\nMSE: {mse}")
