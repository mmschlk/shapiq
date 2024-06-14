"""This test module contains all tests for the configuration of benchmark games."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from shapiq.approximator import KernelSHAP, PermutationSamplingSII, kADDSHAP
from shapiq.games.benchmark.benchmark_config import load_games_from_configuration
from shapiq.games.benchmark.plot import plot_approximation_quality
from shapiq.games.benchmark.run import run_benchmark, run_benchmark_from_configuration


@pytest.mark.parametrize("index, n_jobs", [("SV", 2), ("k-SII", 2), ("SV", 1)])
def test_benchmark(index, n_jobs):
    """Tests the general benchmark setup with pre-computed games."""

    game_identifier = "ImageClassifierLocalXAI"
    n_players = 9
    config_id = 1
    n_player_id = 1

    if index == "SV":
        order = 1
        approximators = [
            kADDSHAP(n=n_players, random_state=42, max_order=2),
        ]
    elif index == "k-SII":
        order = 2
        approximators = [
            PermutationSamplingSII(n=n_players, random_state=42, index=index),
        ]
    else:
        raise ValueError("Wrong index for test.")

    games = load_games_from_configuration(
        game_class=game_identifier, n_player_id=n_player_id, config_id=config_id, n_games=2
    )
    games = list(games)  # convert to list (the generator is consumed)
    assert games[0].n_players == n_players

    save_path = f"{index}_benchmark_results.json"
    results = run_benchmark(
        index=index,
        order=order,
        games=games,
        approximators=approximators,
        save_path=save_path,
        # alternatively, you can set also max_budget (e.g. 10_000) and budget_step to 0.05 (in percentage of max_budget)
        budget_steps=[50, 100],
        rerun_if_exists=True,
        # if True, the benchmark will rerun the approximators even if the results file exists
        n_jobs=n_jobs,  # number of parallel jobs
    )
    assert os.path.exists(save_path)

    fig, axis = plot_approximation_quality(data=results)
    assert isinstance(fig, plt.Figure) and isinstance(axis, plt.Axes)

    # clean up
    os.remove(save_path)


def test_benchmark_config():
    """Tests the general benchmark setup with pre-computed games."""

    game_identifier = "ImageClassifierLocalXAI"
    n_players = 9
    config_id = 1
    n_player_id = 1
    index = "SV"
    order = 1
    approximators = [
        KernelSHAP(n=n_players, random_state=42),
    ]

    run_benchmark_from_configuration(
        index=index,
        order=order,
        game_class=game_identifier,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=2,
        approximators=approximators,
        rerun_if_exists=True,
        n_jobs=1,
        max_budget=200,
    )
    result_dir = "results"
    assert os.path.exists(result_dir)
    file_name = os.listdir(result_dir)[0]
    save_path = os.path.join(result_dir, file_name)
    results = pd.read_json(save_path)

    fig, axis = plot_approximation_quality(data=results)
    assert isinstance(fig, plt.Figure) and isinstance(axis, plt.Axes)

    # clean up
    os.remove(save_path)
    os.rmdir(result_dir)
    assert not os.path.exists("results")
