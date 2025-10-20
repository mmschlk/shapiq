from __future__ import annotations

from shapiq import ExactComputer, UnbiasedKernelSHAP
from shapiq.approximator.regression.polyshap import (
    ExplanationFrontierGenerator,
    PolySHAP,
)
from shapiq.approximator.regressionMSR import RegressionMSR

if __name__ == "__main__":
    import numpy as np

    from shapiq.games.benchmark.synthetic import SOUM

    n_players = 14
    random_state = 42
    budget = 2**n_players

    # Initialize the SOUM game
    game = SOUM(n=n_players, n_basis_games=1500, random_state=random_state)

    explanation_frontier = ExplanationFrontierGenerator(set(range(n_players)))
    shapley_frontier = explanation_frontier.generate_kadd(max_order=1)

    approx = PolySHAP(
        explanation_frontier=shapley_frontier,
        n=n_players,
        random_state=random_state,
        replacement=False,
        pairing_trick=True,
    )

    approx2 = UnbiasedKernelSHAP(
        n=n_players, random_state=random_state, pairing_trick=True, replacement=False
    )

    approx = RegressionMSR(
        n=n_players,
        random_state=random_state,
        replacement=False,
        pairing_trick=True,
        shapley_weighted_inputs=False,
        regression_adjustment=True,
        residual_estimator=approx,
    )

    regressionmsr_estimates = approx.approximate(budget=budget, game=game)

    exact_computer = ExactComputer(n_players, game)
    ground_truth = exact_computer(index="SV", order=1)
    print("Approximation:")
    print(regressionmsr_estimates.values)
    print("Ground-truth:")
    print(ground_truth.values)

    # compare mse between approx and ground-truth
    mse = np.sum((regressionmsr_estimates.values - ground_truth.values) ** 2) / n_players

    print(f"\nMSE: {mse}")
