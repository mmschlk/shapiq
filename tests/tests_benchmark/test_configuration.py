"""This test module contains all tests for the configuration of benchmark games."""

import pytest

from shapiq.benchmark.configuration import (
    GAME_NAME_TO_CLASS_MAPPING,
    get_name_from_game_class,
    print_benchmark_configurations,
)
from shapiq.benchmark.load import download_game_data, load_games_from_configuration


def test_print_config():
    print_benchmark_configurations()
    assert True


def test_getting_games():
    game_class = GAME_NAME_TO_CLASS_MAPPING["AdultCensusLocalXAI"]
    get_name_from_game_class(game_class)

    with pytest.raises(ValueError):
        get_name_from_game_class(None)


def test_loading():
    game_class = GAME_NAME_TO_CLASS_MAPPING["CaliforniaHousingLocalXAI"]
    _ = next(load_games_from_configuration(game_class, 1, 1))
    _ = next(load_games_from_configuration(game_class, 1, 1, only_pre_computed=True))
    _ = next(
        load_games_from_configuration(
            game_class, 1, 1, only_pre_computed=False, check_pre_computed=False
        )
    )
    _ = next(
        load_games_from_configuration(
            game_class, {"model_name": "decision_tree", "imputer": "marginal"}, 1
        )
    )

    game_class = GAME_NAME_TO_CLASS_MAPPING["SynthDataTreeSHAPIQXAI"]
    _ = next(load_games_from_configuration(game_class, config_id=0, n_games=1))


def test_download():
    game_name = "CaliforniaHousing_GlobalExplanation_Game"
    name = "model_name=decision_tree_loss_function=r2_score_1"
    download_game_data(game_name=game_name, n_players=8, file_name=name)

    with pytest.raises(FileNotFoundError):
        download_game_data(game_name=game_name, n_players=8, file_name="invalid")
