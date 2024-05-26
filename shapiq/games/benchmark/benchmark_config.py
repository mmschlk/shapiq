"""This module contains the configuration of all benchmark games and how they are set up.

The configuration of the benchmark games is stored in a dictionary with the game class as the key
and a dictionary of configurations as the value. The configurations contain the following
information:

- `BENCHMARK_CONFIGURATIONS_ALL_PARAMS`: A dictionary of default parameters that will be passed to
    all games. This dictionary contains the following keys:
    - `normalize`: A boolean flag to normalize the data before training the model. Defaults to True.
    - `verbose`: A boolean flag to print the validation score of the model if trained. Defaults to
        False.
    - `random_state`: The random state to use for the games that do not iterate over the random
        state. Defaults to 42.

- `BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS`: A list of default iterations that will be used to
    create the benchmark games if not overwritten in the configuration.

- `BENCHMARK_CONFIGURATIONS`: A dictionary of the benchmark games and their configurations. The key
    is the game class, and the value is a dictionary with the following keys:
    - `configurations`: A list of dictionaries containing the configurations for the game.
    - `iteration_parameter`: The parameter that will be used to iterate over the configurations and
        populated by the `BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS`.
    - `iteration_parameter_values`: An optional list of values for the iteration parameter. If
        provided, the `BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS` will be ignored.
"""

import os
from collections.abc import Generator
from typing import Any, Optional, Union

from .. import Game
from . import (
    AdultCensusClusterExplanation,
    AdultCensusDatasetValuation,
    AdultCensusEnsembleSelection,
    AdultCensusFeatureSelection,
    AdultCensusGlobalXAI,
    AdultCensusLocalXAI,
    AdultCensusRandomForestEnsembleSelection,
    AdultCensusUnsupervisedData,
    BikeSharingClusterExplanation,
    BikeSharingDatasetValuation,
    BikeSharingEnsembleSelection,
    BikeSharingFeatureSelection,
    BikeSharingGlobalXAI,
    BikeSharingLocalXAI,
    BikeSharingRandomForestEnsembleSelection,
    BikeSharingUnsupervisedData,
    CaliforniaHousingClusterExplanation,
    CaliforniaHousingDatasetValuation,
    CaliforniaHousingEnsembleSelection,
    CaliforniaHousingFeatureSelection,
    CaliforniaHousingGlobalXAI,
    CaliforniaHousingLocalXAI,
    CaliforniaHousingRandomForestEnsembleSelection,
    CaliforniaHousingUnsupervisedData,
    SentimentAnalysisLocalXAI,
)
from .precompute import SHAPIQ_DATA_DIR

# default params that will be passed to any game
BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS: dict[str, Any] = {
    "normalize": True,
    "verbose": False,
    "random_state": 42,
}

# default iterations for the benchmark games will be passed if not overwritten in the configuration
BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS: list[int] = list(range(1, 31))

# Sentences used for the Sentiment Analysis Game:

SENTIMENT_ANALYSIS_TEXTS = [
    "I remember when this came out a lot of kids were nuts about it",
    "This is a great movie for the true romantics and sports lovers alike",
    "This is strictly a review of the pilot episode as it appears on DVD",
    "I was quite impressed with this movie as a child of eight or nine",
    "When Family Guy first premiered I was not in a discriminating mood",
    "Tom Clancy uses Alesandr Nevsky in his book Red Storm Rising",
    "I won't spend a lot of time nor energy on this comment",
    "This was such a terrible film almost a comedy sketch of a noir film",
    "Well maybe I'm just having a bad run with Hindi movies lately",
    "The Duke is a very silly film--a dog becoming a duke",
    "More exciting than the Wesley Snipes film and with better characters too",
    "This movie was Jerry Bruckheimer's idea to sell some records",
    "Karl Jr and his dad are now running an army on a remote island",
    "The film has no connection with the real life in Bosnia in those days",
    "No doubt Frank Sinatra was a talented actor as well as a talented singer",
    "Yesterday my Spanish / Catalan wife and myself saw this emotional lesson in history",
    "The 3rd and in my view the best of the Blackadder series",
    "With some films it is really hard to tell for whom they were made",
    "Not only is this film entertaining with excellent comedic acting but also interesting politically",
    "This was Eisenstein's first completed project in over ten years",
    "Many of these other viewers complain that the story line has already been attempted",
    "Oh boy it's another comet-hitting-the-earth film",
    "First of all DO NOT call this a remake of the '63 film",
    "Great little short film that aired a while ago on SBS here in Aus",
    "If you really loved GWTW you will find quite disappointing the story",
    "Probably New Zealands worst Movie ever madeThe Jokes They are not funny",
    "This film is about the worst I have seen in a very long time",
    "This is by far the worst movie I have ever seen in the cinema",
    "If anybody really wants to understand Hitler read WWI history not WWII history",
    "This movie is more Lupin then most especially coming from Funimation",
]


# stores the configurations of all the benchmark games and how they are set up
BENCHMARK_CONFIGURATIONS: dict[Game.__class__, dict[str, Union[str, list[dict]]]] = {
    # local xai configurations ---------------------------------------------------------------------
    AdultCensusLocalXAI: {
        "configurations": [
            {"model_name": "decision_tree", "class_to_explain": 1},
            {"model_name": "random_forest", "class_to_explain": 1},
            {"model_name": "gradient_boosting", "class_to_explain": 1},
        ],
        "iteration_parameter": "x",
        "n_players": 14,
    },
    BikeSharingLocalXAI: {
        "configurations": [
            {"model_name": "decision_tree"},
            {"model_name": "random_forest"},
            {"model_name": "gradient_boosting"},
        ],
        "iteration_parameter": "x",
        "n_players": 12,
    },
    CaliforniaHousingLocalXAI: {
        "configurations": [
            {"model_name": "decision_tree"},
            {"model_name": "random_forest"},
            {"model_name": "gradient_boosting"},
            {"model_name": "neural_network"},
        ],
        "iteration_parameter": "x",
        "n_players": 8,
    },
    SentimentAnalysisLocalXAI: {
        "configurations": [{"mask_strategy": "mask"}],
        "iteration_parameter": "input_text",
        "iteration_parameter_values": list(range(1, len(SENTIMENT_ANALYSIS_TEXTS) + 1)),
        "iteration_parameter_values_names": SENTIMENT_ANALYSIS_TEXTS,
        "n_players": 14,
    },
    # TODO add image local xai config
    # global xai configurations --------------------------------------------------------------------
    AdultCensusGlobalXAI: {
        "configurations": [
            {"model_name": "decision_tree", "loss_function": "accuracy_score"},
            {"model_name": "random_forest", "loss_function": "accuracy_score"},
            {"model_name": "gradient_boosting", "loss_function": "accuracy_score"},
        ],
        "iteration_parameter": "random_state",
        "n_players": 14,
    },
    BikeSharingGlobalXAI: {
        "configurations": [
            {"model_name": "decision_tree", "loss_function": "r2_score"},
            {"model_name": "random_forest", "loss_function": "r2_score"},
            {"model_name": "gradient_boosting", "loss_function": "r2_score"},
        ],
        "iteration_parameter": "random_state",
        "n_players": 12,
    },
    CaliforniaHousingGlobalXAI: {
        "configurations": [
            {"model_name": "decision_tree", "loss_function": "r2_score"},
            {"model_name": "random_forest", "loss_function": "r2_score"},
            {"model_name": "gradient_boosting", "loss_function": "r2_score"},
            {"model_name": "neural_network", "loss_function": "r2_score"},
        ],
        "iteration_parameter": "random_state",
        "n_players": 8,
    },
    # feature selection configurations -------------------------------------------------------------
    AdultCensusFeatureSelection: {
        "configurations": [
            {"model_name": "decision_tree"},
            {"model_name": "random_forest"},
            {"model_name": "gradient_boosting"},
        ],
        "iteration_parameter": "random_state",
        "n_players": 14,
    },
    BikeSharingFeatureSelection: {
        "configurations": [
            {"model_name": "decision_tree"},
            {"model_name": "random_forest"},
            {"model_name": "gradient_boosting"},
        ],
        "iteration_parameter": "random_state",
        "n_players": 12,
    },
    CaliforniaHousingFeatureSelection: {
        "configurations": [
            {"model_name": "decision_tree"},
            {"model_name": "random_forest"},
            {"model_name": "gradient_boosting"},
            # TODO: think of adding neural network to the feature selection
        ],
        "iteration_parameter": "random_state",
        "n_players": 8,
    },
    # ensemble selection configurations ------------------------------------------------------------
    AdultCensusEnsembleSelection: {
        "configurations": [{"loss_function": "accuracy_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    BikeSharingEnsembleSelection: {
        "configurations": [{"loss_function": "r2_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    CaliforniaHousingEnsembleSelection: {
        "configurations": [{"loss_function": "r2_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    # ensemble selection with random forest configurations -----------------------------------------
    AdultCensusRandomForestEnsembleSelection: {
        "configurations": [{"loss_function": "accuracy_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    BikeSharingRandomForestEnsembleSelection: {
        "configurations": [{"loss_function": "r2_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    CaliforniaHousingRandomForestEnsembleSelection: {
        "configurations": [{"loss_function": "r2_score", "n_members": 10}],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    # dataset valuation configurations -------------------------------------------------------------
    AdultCensusDatasetValuation: {
        "configurations": [
            {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
        ],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    BikeSharingDatasetValuation: {
        "configurations": [
            {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
        ],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    CaliforniaHousingDatasetValuation: {
        "configurations": [
            {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
            {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
        ],
        "iteration_parameter": "random_state",
        "n_players": 10,
    },
    # cluster explanation configurations -----------------------------------------------------------
    AdultCensusClusterExplanation: {
        "configurations": [
            {"cluster_method": "kmeans", "score_method": "silhouette_score"},
            {"cluster_method": "agglomerative", "score_method": "calinski_harabasz_score"},
        ],
        "iteration_parameter": "random_state",  # for agglomerative this does not change the game
        "iteration_parameter_values": [1],  # for agglomerative this does not change the game
        "n_players": 14,
    },
    BikeSharingClusterExplanation: {
        "configurations": [
            {"cluster_method": "kmeans", "score_method": "silhouette_score"},
            {"cluster_method": "agglomerative", "score_method": "calinski_harabasz_score"},
        ],
        "iteration_parameter": "random_state",  # for agglomerative this does not change the game
        "iteration_parameter_values": [1],  # for agglomerative this does not change the game
        "n_players": 12,
    },
    CaliforniaHousingClusterExplanation: {
        "configurations": [
            {"cluster_method": "kmeans", "score_method": "silhouette_score"},
            {"cluster_method": "agglomerative", "score_method": "calinski_harabasz_score"},
        ],
        "iteration_parameter": "random_state",  # for agglomerative this does not change the game
        "iteration_parameter_values": [1],  # for agglomerative this does not change the game
        "n_players": 8,
    },
    # unsupervised data configurations -------------------------------------------------------------
    AdultCensusUnsupervisedData: {
        "configurations": [{}],
        "iteration_parameter": "random_state",  # this does not change the game
        "iteration_parameter_values": [1],  # this does not change the game
        "n_players": 14,
    },
    BikeSharingUnsupervisedData: {
        "configurations": [{}],
        "iteration_parameter": "random_state",  # this does not change the game
        "iteration_parameter_values": [1],  # this does not change the game
        "n_players": 12,
    },
    CaliforniaHousingUnsupervisedData: {
        "configurations": [{}],
        "iteration_parameter": "random_state",  # this does not change the game
        "iteration_parameter_values": [1],  # this does not change the game
        "n_players": 8,
    },
}


def get_game_file_name_from_config(configuration: dict[str, Any], iteration: int) -> str:
    """Get the file name for the game data with the given configuration and iteration.

    Args:
        configuration: A configuration of the game class.
        iteration: The iteration of the game.

    Returns:
        The file name of the game data
    """
    file_name = "_".join(f"{key}={value}" for key, value in configuration.items())
    file_name = "_".join([file_name, str(iteration)])
    return file_name


def load_game_data(
    game_class: Game.__class__, configuration: dict[str, Any], iteration: int = 1
) -> Game:
    """Loads the precomputed game data for the given game class and configuration.

    Args:
        game_class: The class of the game
        configuration: The configuration to use to load the game
        iteration: The iteration of the game to load

    Returns:
        An initialized game object with the given configuration

    Raises:
        FileNotFoundError: If the file with the precomputed values does not exist
    """
    n_players = BENCHMARK_CONFIGURATIONS[game_class]["n_players"]
    file_name = get_game_file_name_from_config(configuration, iteration)

    path_to_values = str(
        os.path.join(
            SHAPIQ_DATA_DIR,
            game_class.get_game_name(),
            str(n_players),
            f"{file_name}.npz",
        )
    )
    try:
        return Game(path_to_values=path_to_values)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"File {path_to_values} does not exist. Are you sure it was created/pre-computed? "
            f"Consider pre-computing the game or fetching the data from the repository."
        ) from error


def print_benchmark_configurations() -> None:
    """Print the configurations of the benchmark games."""
    # print default parameters
    print("Default Parameters:")
    print(f"Normalize: {BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS['normalize']}")
    print(f"Verbose: {BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS['verbose']}")
    print(f"Random State: {BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS['random_state']}")
    print()

    # print default iterations
    print("Default Iterations:")
    print(BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS)
    print()

    # print configurations of the benchmark games
    for game_class, config in BENCHMARK_CONFIGURATIONS.items():
        param_values = config.get(
            "iteration_parameter_values", BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS
        )
        print(f"Game: {game_class.get_game_name()}")
        print(f"Configurations: {config['configurations']}")
        print(f"Iteration Parameter: {config['iteration_parameter']}")
        print(f"Iteration Parameter Values: {param_values}")
        print()


def load_games(
    game_class: Game.__class__, configuration: dict[str, Any], n_games: Optional[int] = None
) -> Generator[Game, None, None]:
    """Load the game with the given configuration.

    Args:
        game_class: The class of the game to load with the configuration.
        configuration: The configuration to use to load the game.
        n_games: The number of games to load. Defaults to None.

    Returns:
        An initialized game object with the given configuration.
    """
    params = {}

    # get the default parameters
    default_params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
    params.update(default_params)

    # get the class-specific configurations
    config_of_class = BENCHMARK_CONFIGURATIONS[game_class]
    iteration_param = config_of_class["iteration_parameter"]
    iteration_param_values = config_of_class.get(
        "iteration_parameter_values", BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS
    )
    # if names exist take these, otherwise use the values
    iteration_param_values = config_of_class.get(
        "iteration_parameter_values_names", iteration_param_values
    )
    params.update(configuration)

    # create the generator of games
    n_games = len(iteration_param_values) if n_games is None else n_games
    for i in range(n_games):
        params[iteration_param] = iteration_param_values[i]
        yield game_class(**params)
