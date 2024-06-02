"""This test module tests the ExactComputer class."""

import numpy as np

from shapiq.exact import ExactComputer
from shapiq.games.benchmark.synthetic.soum import SOUM

if __name__ == "__main__":
    n = 8
    N = set(range(n))
    order = 2

    soum = SOUM(n, n_basis_games=1, min_interaction_size=5)
    soum.linear_coefficients[0] = 1
    predicted_value = soum(np.ones(n))[0]
    # Compute via exactComputer
    exact_computer = ExactComputer(n_players=n, game_fun=soum)

    sii = exact_computer.base_interaction("SII", order)
    fsii = exact_computer.shapley_interaction("FSII", order)
    stii = exact_computer.shapley_interaction("STII", order)
    ksii = exact_computer.shapley_interaction("k-SII", order)

    print(
        "order: ",
        order,
        ", interaction: ",
        soum.unanimity_games[0].interaction,
        ", value:",
        predicted_value,
    )

    print("UNIQUE VALUES IN INTERACTION INDEX")

    print("SII", np.unique(np.round(sii.values, 3)))
    print("FSII", np.unique(np.round(fsii.values, 3)))
    print("STII", np.unique(np.round(stii.values, 3)))
    print("k-SII", np.unique(np.round(ksii.values, 3)))

    sii_scaled = sii
    sii_scaled.values[0] = 0
    sii_scaled.values *= predicted_value / np.sum(sii_scaled.values)

    print("Scaled SII", np.unique(np.round(sii_scaled.values, 3)))
