"""This test script runs approximators like KernelSHAP, SVARM, or Permutation Sampling on
increasing budget sizes to see how long it takes to run the approximation."""

from time import time

import numpy as np
import pandas as pd

from shapiq import KernelSHAP, PermutationSamplingSV


def dummy_game(coalitions):
    return np.random.uniform(0, 1, size=(coalitions.shape[0],))


if __name__ == "__main__":

    budgets = [
        10_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
        800_000,
        900_000,
        1_000_000,
    ]

    approximators = [
        KernelSHAP(n=144, random_state=42),
        PermutationSamplingSV(n=144, random_state=42),
    ]

    for approximator in approximators:
        results_dicts = []
        for budget in budgets:
            print(f"Running approximator {approximator.__class__.__name__} on budget {budget}")
            start = time()
            approximator.approximate(budget=budget, game=dummy_game)
            end = time()
            elapsed_time = end - start
            results = {
                "approximator": approximator.__class__.__name__,
                "budget": budget,
                "time": elapsed_time,
            }
            print(results)
            results_dicts.append(results)

        results_df = pd.DataFrame(results_dicts)
        results_df.to_csv(f"{approximator.__class__.__name__}_runtimes.csv", index=False)
        print(results_df)
