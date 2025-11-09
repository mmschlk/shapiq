from shapiq.approximator.regressionMSR import RegressionMSR
from shapiq import ExactComputer

if __name__ == "__main__":
    import numpy as np
    from shapiq import PermutationSamplingSV, InteractionValues
    from shapiq.games.benchmark.synthetic import SOUM

    n_players = 14
    random_state = 42
    budget = 15

    # Initialize the SOUM game
    game = SOUM(n=n_players, n_basis_games=20, random_state=random_state)

    approx = PermutationSamplingSV(n=n_players, random_state=random_state, pairing_trick=True)

    permutation_estimates = approx.approximate(budget=budget, game=game)

    exact_computer = ExactComputer(n_players, game)
    ground_truth = exact_computer(index="SV", order=1)
    print("Approximation:")
    print(permutation_estimates.values)

    # compare mse between approx and ground-truth
    mse = (
        np.sum((permutation_estimates.values - ground_truth.values) ** 2) / n_players
    )

    print(f"\nMSE: {mse}")
