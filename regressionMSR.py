from shapiq.approximator.regressionMSR import RegressionMSR
from shapiq import ExactComputer

if __name__ == "__main__":
    import numpy as np
    from shapiq import PermutationSamplingSV, InteractionValues
    from shapiq.games.benchmark.synthetic import SOUM

    n_players = 9
    random_state = 42
    budget = 15

    # Initialize the SOUM game
    game = SOUM(n=n_players, n_basis_games=20, random_state=random_state)

    approx = RegressionMSR(n=n_players, random_state=random_state)

    regressionmsr_estimates = approx.approximate(budget=budget, game=game)

    exact_computer = ExactComputer(n_players, game)
    ground_truth = exact_computer(index="SV", order=1)
    print("Approximation:")
    print(regressionmsr_estimates.values)

    # compare mse between approx and ground-truth
    mse = (
        np.sum((regressionmsr_estimates.values - ground_truth.values) ** 2) / n_players
    )

    print(f"\nMSE: {mse}")
