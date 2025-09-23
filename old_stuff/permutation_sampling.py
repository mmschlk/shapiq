### Write a small test using SOUM game with 6 players to verify that pairing trick works correctly for permutation sampling

if __name__ == "__main__":
    import numpy as np
    from shapiq import PermutationSamplingSV, InteractionValues
    from shapiq.games.benchmark.synthetic import SOUM

    n_players = 8
    random_state = 42
    budget = 100

    # Initialize the SOUM game
    game = SOUM(n=n_players, n_basis_games=20, random_state=random_state)

    # Without pairing trick
    perm_sampling_no_pairing = PermutationSamplingSV(
        n=n_players, pairing_trick=False, random_state=random_state
    )
    approx_no_pairing = perm_sampling_no_pairing.approximate(
        budget=budget,
        game=game,
    )

    # With pairing trick
    perm_sampling_with_pairing = PermutationSamplingSV(
        n=n_players, pairing_trick=True, random_state=random_state
    )
    approx_with_pairing = perm_sampling_with_pairing.approximate(
        budget=budget,
        game=game,
    )

    print("Approximation without pairing trick:")
    print(approx_no_pairing.values)

    print("\nApproximation with pairing trick:")
    print(approx_with_pairing.values)

    ground_truth = game.exact_values(index="SV", order=1)

    # compare mse between approx and ground-truth
    mse_no_pairing = (
        np.sum((approx_no_pairing.values - ground_truth.values) ** 2) / n_players
    )
    mse_with_pairing = (
        np.sum((approx_with_pairing.values - ground_truth.values) ** 2) / n_players
    )
    print(f"\nMSE without pairing trick: {mse_no_pairing}")
    print(f"MSE with pairing trick: {mse_with_pairing}")
