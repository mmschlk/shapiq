import numpy as np

from shapiq.approximator.kernelshapiq import KernelSHAPIQ
from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq.games.soum import SOUM


def test_approximator_kernelshapiq():
    for RANDOM_STATE in range(10):
        n = np.random.randint(low=8, high=12)
        total_budget = 2**n
        N = set(range(n))
        order = 2
        n_basis_games = np.random.randint(low=10, high=200)
        soum = SOUM(n, n_basis_games=n_basis_games)
        index = "SII"

        predicted_value = soum(np.ones(n))[0]

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(N, soum.moebius_coefficients)

        sii = moebius_converter(index=index, order=order)
        kernelshapiq = KernelSHAPIQ(n=n, order=order, index=index)

        squared_errors = {}

        LOWEST_BUDGET_PERC = 10
        PREV_BUDGET_PERC = 0
        N_BUDGET_STEPS = 5
        N_ITERATIONS = 5

        approximation_improvement_counter = 0

        for budget_perc in np.linspace(LOWEST_BUDGET_PERC, 100, N_BUDGET_STEPS):
            budget = int(budget_perc / 100 * total_budget)

            sii_approximated = kernelshapiq.approximate(budget=budget, game=soum)
            for iteration in range(N_ITERATIONS - 1):
                sii_approximated += kernelshapiq.approximate(budget=budget, game=soum)
            sii_approximated *= 1 / N_ITERATIONS

            # Assert efficiency
            assert (
                np.sum(sii_approximated.values[:n]) + sii_approximated.baseline_value
            ) - predicted_value < 10e-5

            # Compute squared errors
            squared_errors[budget_perc] = np.mean(((sii_approximated - sii).values) ** 2)
            if PREV_BUDGET_PERC in squared_errors:
                approximation_improvement_counter += (
                    squared_errors[budget_perc] < squared_errors[PREV_BUDGET_PERC]
                )
            PREV_BUDGET_PERC = budget_perc

        # Assert 80%-ratio of improvements over previous calculation
        assert approximation_improvement_counter / N_BUDGET_STEPS >= 0.8
