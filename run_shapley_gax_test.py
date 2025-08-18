"""This test script runs approximators like KernelSHAP, SVARM, or Permutation Sampling on
increasing budget sizes to see how long it takes to run the approximation."""

import itertools
from time import time

import numpy as np
import pandas as pd

from shapiq.approximator.regression.shapleygax import ExplanationBasisGenerator, ShapleyGAX


def dummy_game(coalitions):
    return np.random.uniform(0, 1, size=(coalitions.shape[0],))


if __name__ == "__main__":

    budgets = [5_000, 10_000, 15_000, 20_000]
    n_players = [16] #[20, 50, 100, 150]
    n_orders = [2] #[1, 2]

    params_cross_product = itertools.product(n_players, n_orders, budgets)

    results_dicts = []
    for n, order, budget in params_cross_product:
        # generate explanation basis
        start = time()
        explanation_basis = ExplanationBasisGenerator(N=list(range(n)))
        explanation_basis = explanation_basis.generate_kadd_explanation_basis(max_order=order)
        end = time()
        basis_time = end - start

        # init approximator
        start = time()
        approximator = ShapleyGAX(n=n, random_state=42, explanation_basis=explanation_basis)
        end = time()
        init_time = end - start

        # run approximator
        start = time()
        approximator.approximate(budget=budget, game=dummy_game)
        end = time()
        approx_time = end - start

        # total time
        total_time = basis_time + init_time + approx_time

        results = {
            "approximator": approximator.__class__.__name__,
            "budget": budget,
            "n_players": n,
            "order": order,
            "basis_time": basis_time,
            "init_time": init_time,
            "approx_time": approx_time,
            "total_time": total_time,
        }
        print(results)
        results_dicts.append(results)

    # save results
    results_df = pd.DataFrame(results_dicts)
    results_df.to_csv("ShapleyGAX_runtimes.csv", index=False)
    print(results_df)
