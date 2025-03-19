"""This script is used to time and profile the regression estimations in large player domains."""

import time

import joblib
import pandas as pd

from shapiq import KernelSHAP, KernelSHAPIQ, RegressionFSII
from shapiq.games.benchmark import RandomGame


def time_approximation(approx_class, n_players, budget, order) -> dict:
    game = RandomGame(n=n_players, random_state=42)
    start_time = time.time()
    computer = approx_class(n=n_players, max_order=order)
    init_time = time.time() - start_time
    start_time = time.time()
    _ = computer.approximate(budget=budget, game=game)
    compute_time = time.time() - start_time
    return {
        "approximator": approx_class.__name__,
        "order": order,
        "n_players": n_players,
        "budget": budget,
        "init_time": init_time,
        "compute_time": compute_time,
        "total_time": init_time + compute_time,
    }


def do_runtime_experiment() -> None:
    """Conducts the runtime experiment."""
    # construct the parameter grid
    paramters = []
    for order in ORDERS:
        for n_player in N_PLAYERS:
            for budget in BUDGETS:
                for n_run in range(N_RUNS):
                    for approx_class in APPROXIMATORS:
                        if order > 1 and approx_class == KernelSHAP:
                            continue
                        paramters.append(
                            {
                                "approx_class": approx_class,
                                "n_players": n_player,
                                "budget": budget,
                                "order": order,
                            }
                        )

    # run the experiment
    parallel = joblib.Parallel(n_jobs=N_JOBS)
    results = parallel(joblib.delayed(time_approximation)(**params) for params in paramters)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH)


if __name__ == "__main__":

    # single benchmark
    do_single_bench = True
    if do_single_bench:
        time_approximation(approx_class=RegressionFSII, n_players=50, budget=50_000, order=2)

    # runtime experiment
    do_runtime = False
    N_JOBS = 8
    N_RUNS = 3
    N_PLAYERS = [10, 12]
    BUDGETS = [10_000]
    ORDERS = [1, 2]
    APPROXIMATORS = (KernelSHAP, RegressionFSII, KernelSHAPIQ)

    results = []
    RESULTS_PATH = "results.csv"

    if do_runtime:
        do_runtime_experiment()
