"""This test module contains all tests for the configuration of benchmark games."""

from shapiq.games.benchmark import AdultCensusLocalXAI, CaliforniaHousingGlobalXAI
from shapiq.games.benchmark.benchmark_config import (
    BENCHMARK_CONFIGURATIONS,
    BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS,
    get_game_file_name_from_config,
    load_games_from_configuration,
    print_benchmark_configurations,
)


def test_print_config():
    print_benchmark_configurations()
    assert True


def test_load_games_with_configuration():
    for game_class, configurations in BENCHMARK_CONFIGURATIONS.items():
        for configuration in configurations:
            for config in configuration["configurations"]:
                game = next(
                    load_games_from_configuration(game_class, configuration=config, n_games=1)
                )
                # assert game.game_name == game_class.get_game_name() # TODO
                assert game.verbose == BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"]


def test_get_game_file_name_from_config():
    adult_census_config = BENCHMARK_CONFIGURATIONS[AdultCensusLocalXAI][0]["configurations"][0]
    adult_file_name = get_game_file_name_from_config(adult_census_config, iteration=1)
    expected = "model_name=decision_tree_class_to_explain=1_1"
    assert adult_file_name == expected

    california_housing_config = BENCHMARK_CONFIGURATIONS[CaliforniaHousingGlobalXAI][0][
        "configurations"
    ][1]
    california_file_name = get_game_file_name_from_config(california_housing_config, iteration=73)
    expected = "model_name=random_forest_loss_function=r2_score_73"
    assert california_file_name == expected
