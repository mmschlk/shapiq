import numpy as np

from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.games.soum import SOUM

import pytest


@pytest.mark.parametrize(
    "index,N_BATCH_SIZE",
    [
        ("SII", None),
        ("STII", None),
        ("k-SII", None),
        ("FSII", None),
        ("SII", 50),
        ("STII", 50),
        ("k-SII", 50),
        ("FSII", 50),
    ],
)
def test_montecarlo_base_example(index, N_BATCH_SIZE):
    N_RUNS = 10
    for i in range(N_RUNS):
        n = np.random.randint(low=6, high=10)
        budget = int(2**n)
        N = set(range(n))
        max_order = np.random.randint(low=1, high=min(5, n))
        n_basis_games = np.random.randint(low=10, high=200)
        soum = SOUM(n, n_basis_games=n_basis_games)

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(N, soum.moebius_coefficients)

        shapley_interactions = moebius_converter(index=index, order=max_order)
        # set emptyset value to baseline, required for SII
        shapley_interactions.values[
            shapley_interactions.interaction_lookup[tuple()]
        ] = shapley_interactions.baseline_value

        if index == "FSII":
            # Only top-order for FSII
            min_order = max_order
            top_order = True
        else:
            top_order = False
            min_order = 0

        montecarlo_approximator = SHAPIQ(n=n, max_order=max_order, index=index, top_order=top_order)
        shapley_interactions_approximated = montecarlo_approximator.approximate(
            budget=budget, game=soum
        )

        diff = 0
        for interaction, interaction_pos in shapley_interactions.interaction_lookup.items():
            if len(interaction) >= min_order:
                diff += (
                    shapley_interactions[interaction]
                    - shapley_interactions_approximated[interaction]
                ) ** 2
        assert diff < 10e-7


@pytest.mark.parametrize(
    "index",
    [("STII"), ("k-SII"), ("FSII")],
)
def test_montecarlo_approximation(index):
    N_RUNS = 10
    LOWEST_BUDGET_PERC = 10
    PREV_BUDGET_PERC = 0
    N_BUDGET_STEPS = 5
    N_ITERATIONS = 5
    approximation_improvement_counter = 0

    def _check_difference(shapley_interactions, shapley_interactions_approximated, min_order):
        diff = 0
        for interaction, interaction_pos in shapley_interactions.interaction_lookup.items():
            if len(interaction) >= min_order:
                diff += (
                    shapley_interactions[interaction]
                    - shapley_interactions_approximated[interaction]
                ) ** 2
        return diff

    for i in range(N_RUNS):
        n = np.random.randint(low=6, high=10)
        total_budget = 2**n
        N = set(range(n))
        max_order = np.random.randint(low=1, high=min(5, n))
        n_basis_games = np.random.randint(low=10, high=200)
        soum = SOUM(n, n_basis_games=n_basis_games)

        predicted_value = soum(np.ones(n))

        # Compute via sparse Möbius representation
        moebius_converter = MoebiusConverter(N, soum.moebius_coefficients)

        shapley_interactions = moebius_converter(index=index, order=max_order)
        # set emptyset value to baseline, required for SII
        shapley_interactions.values[
            shapley_interactions.interaction_lookup[tuple()]
        ] = shapley_interactions.baseline_value

        if index == "FSII":
            # Only top-order for FSII
            min_order = max_order
            top_order = True
        else:
            top_order = False
            min_order = 0

        montecarlo_approximator = SHAPIQ(n=n, max_order=max_order, index=index, top_order=top_order)

        squared_errors = {}

        for budget_perc in np.linspace(LOWEST_BUDGET_PERC, 100, N_BUDGET_STEPS):
            budget = int(budget_perc / 100 * total_budget)

            shapley_interactions_approximated = montecarlo_approximator.approximate(
                budget=budget, game=soum
            )
            for iteration in range(N_ITERATIONS - 1):
                shapley_interactions_approximated += montecarlo_approximator.approximate(
                    budget=budget, game=soum
                )
            shapley_interactions_approximated *= 1 / N_ITERATIONS

            # Compute squared errors
            squared_errors[budget_perc] = _check_difference(
                shapley_interactions, shapley_interactions_approximated, min_order
            )
            if PREV_BUDGET_PERC in squared_errors:
                approximation_improvement_counter += (
                    squared_errors[budget_perc] < squared_errors[PREV_BUDGET_PERC]
                )
            PREV_BUDGET_PERC = budget_perc

            if min_order == 0:
                # Assert efficiency (not for FSII)
                assert (
                    np.sum(
                        shapley_interactions_approximated.values[
                            np.array(
                                [
                                    pos
                                    for key, pos in shapley_interactions_approximated.interaction_lookup.items()
                                    if len(key) > 0
                                ]
                            )
                        ]
                    )
                    + shapley_interactions_approximated.baseline_value
                ) - predicted_value < 10e-5

        # Assert exact values for 100% budget
        assert squared_errors[100] < 10e-7

    # Assert 80%-ratio of improvements over previous calculation
    assert approximation_improvement_counter / ((N_BUDGET_STEPS - 1) * N_RUNS) >= 0.8
