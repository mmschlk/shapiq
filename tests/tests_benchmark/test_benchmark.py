"""This test module contains all tests for the configuration of benchmark games."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from shapiq.approximator import KernelSHAP, PermutationSamplingSII, kADDSHAP
from shapiq.benchmark import (
    download_game_data,
    load_benchmark_results,
    load_games_from_configuration,
    plot_approximation_quality,
    run_benchmark,
    run_benchmark_from_configuration,
)


@pytest.mark.parametrize("index, n_jobs", [("SV", 2), ("k-SII", 2), ("SV", 1)])
def test_benchmark(index, n_jobs):
    """Tests the general benchmark setup with pre-computed games."""
    game_identifier = "ImageClassifierLocalXAI"
    n_players = 9
    config_id = 1
    n_player_id = 1
    n_games = 2

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
        game_class=game_identifier,
        n_player_id=n_player_id,
        config_id=config_id,
        n_games=n_games,
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
    n_games = 2
    approximators = [
        KernelSHAP(n=n_players, random_state=42),
    ]

    run_benchmark_from_configuration(
        index=index,
        order=order,
        game_class=game_identifier,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
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

    # check if the results can be loaded correctly
    data_loaded, path_loaded = load_benchmark_results(save_path)
    assert data_loaded is not None
    assert path_loaded == save_path
    data_loaded_no_path, path_loaded_no_path = load_benchmark_results(
        index=index,
        order=order,
        game_class=game_identifier,
        game_configuration=config_id,
        game_n_player_id=n_player_id,
        game_n_games=n_games,
    )
    assert data_loaded_no_path is not None
    assert str(path_loaded_no_path) == save_path
    assert len(data_loaded) == len(data_loaded_no_path)

    fig, axis = plot_approximation_quality(data=results)
    assert isinstance(fig, plt.Figure) and isinstance(axis, plt.Axes)

    # clean up
    os.remove(save_path)
    os.rmdir(result_dir)
    assert not os.path.exists("results")


def test_download_games():
    """Tests the download of games from the configuration in the correct location."""
    # files will be stored in shapiq/benchmark/precomputed
    from shapiq.benchmark.precompute import SHAPIQ_DATA_DIR

    game_name = "CaliforniaHousing_GlobalExplanation_Game"
    n_players = 8
    file_name = "model_name=decision_tree_loss_function=r2_score_1"

    # check that dir and file do not exist
    save_dir = os.path.join(SHAPIQ_DATA_DIR, game_name, str(n_players))
    save_path = os.path.join(save_dir, file_name + ".npz")
    try:
        os.remove(save_path)
    except FileNotFoundError:
        pass

    # check that file does not exist
    assert not os.path.exists(save_path)

    # download the game data
    download_game_data(game_name=game_name, n_players=n_players, file_name=file_name)

    # check that dir and file exist
    assert os.path.exists(save_dir)
    assert os.path.exists(save_path)

    with pytest.raises(FileNotFoundError):
        download_game_data(game_name=game_name, n_players=n_players, file_name="wrong_file_name")
