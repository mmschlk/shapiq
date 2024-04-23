import numpy as np

from shapiq.exact import ExactComputer
from shapiq.approximator.regression.kadd_shap import kADDSHAP
from shapiq.games.benchmark import SOUM


def test_approximator_kaddshap():
    N_RUNS = 10
    LOWEST_BUDGET_PERC = 10
    PREV_BUDGET_PERC = 0
    N_BUDGET_STEPS = 5
    N_ITERATIONS = 5
    approximation_improvement_counter = 0

    for RANDOM_STATE in range(N_RUNS):
        n = np.random.randint(low=6, high=8)
        total_budget = 2**n
        N = set(range(n))
        max_order = np.random.randint(low=1, high=n)
        n_basis_games = np.random.randint(low=10, high=200)
        soum = SOUM(n, n_basis_games=n_basis_games)
        index = "kADD-SHAP"

        predicted_value = soum(np.ones(n))[0]

        # For ground truth comparison - kADD-SHAP
        exact_computer = ExactComputer(n_players=n, game_fun=soum)
        kadd_shap = exact_computer.shapley_interaction(index="kADD-SHAP", order=max_order)

        kadd_shap_approximator = kADDSHAP(n=n, max_order=max_order)

        squared_errors = {}

        for budget_perc in np.linspace(LOWEST_BUDGET_PERC, 100, N_BUDGET_STEPS):
            budget = int(budget_perc / 100 * total_budget)

            sii_approximated = kadd_shap_approximator.approximate(budget=budget, game=soum)
            for iteration in range(N_ITERATIONS - 1):
                sii_approximated += kadd_shap_approximator.approximate(budget=budget, game=soum)
            sii_approximated *= 1 / N_ITERATIONS

            # Assert efficiency
            assert (
                np.sum(
                    sii_approximated.values[
                        np.array(
                            [
                                pos
                                for key, pos in sii_approximated.interaction_lookup.items()
                                if len(key) == 1
                            ]
                        )
                    ]
                )
                + sii_approximated.baseline_value
            ) - predicted_value < 10e-5

            # Compute squared errors
            squared_errors[budget_perc] = np.mean(((sii_approximated - kadd_shap).values) ** 2)
            if PREV_BUDGET_PERC in squared_errors:
                approximation_improvement_counter += (
                    squared_errors[budget_perc] < squared_errors[PREV_BUDGET_PERC]
                )
            PREV_BUDGET_PERC = budget_perc

        # Assert exact values for 100% budget
        assert squared_errors[100] < 10e-7

    # Assert 80%-ratio of improvements over previous calculation
    assert approximation_improvement_counter / ((N_BUDGET_STEPS - 1) * N_RUNS) >= 0.8
