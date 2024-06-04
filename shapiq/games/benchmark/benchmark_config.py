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
    - `configurations`: A list of dictionaries containing the configurations for a game. Each
        dictionary contains application/game specific parameters.
    - `iteration_parameter`: The parameter that will be used to iterate over the configurations and
        create multiple games with different random states / data points. This parameter must be
        present in the game class. For example, the `random_state` parameter is used to iterate over
        different random states or the `x` parameter is used to iterate over different data points
        in local XAI games.
    - `n_players`: The number of players in the configurations. A game class can have different
        configurations with different numbers of players, but all game classes have at least one
        set of number of players.
    - `precompute`: An boolean flag to denoting weather the game should be precomputed or not. If
        the game is precomputed, then all game evaluations are stored in a file (in the
        `SHAPIQ_DATA_DIR` directory) and can be loaded later. If the game is not precomputed, then
        the game evaluations are computed on the fly during the benchmark (significantly slower).
"""

import os
import time
from collections.abc import Generator
from typing import Any, Optional, Union

import requests

from ...approximator import (
    FSII_APPROXIMATORS,
    SI_APPROXIMATORS,
    SII_APPROXIMATORS,
    STII_APPROXIMATORS,
    SV_APPROXIMATORS,
    Approximator,
)
from .. import Game
from . import (
    SOUM,
    AdultCensusDatasetValuation,
    AdultCensusDataValuation,
    AdultCensusEnsembleSelection,
    AdultCensusFeatureSelection,
    AdultCensusGlobalXAI,
    AdultCensusLocalXAI,
    AdultCensusRandomForestEnsembleSelection,
    AdultCensusUncertaintyExplanation,
    AdultCensusUnsupervisedData,
    BikeSharingClusterExplanation,
    BikeSharingDatasetValuation,
    BikeSharingDataValuation,
    BikeSharingEnsembleSelection,
    BikeSharingFeatureSelection,
    BikeSharingGlobalXAI,
    BikeSharingLocalXAI,
    BikeSharingRandomForestEnsembleSelection,
    BikeSharingUnsupervisedData,
    CaliforniaHousingClusterExplanation,
    CaliforniaHousingDatasetValuation,
    CaliforniaHousingDataValuation,
    CaliforniaHousingEnsembleSelection,
    CaliforniaHousingFeatureSelection,
    CaliforniaHousingGlobalXAI,
    CaliforniaHousingLocalXAI,
    CaliforniaHousingRandomForestEnsembleSelection,
    CaliforniaHousingUnsupervisedData,
    ImageClassifierLocalXAI,
    SentimentAnalysisLocalXAI,
    # not to be precomputed
    SynthDataTreeSHAPIQXAI,
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

IMAGENET_EXAMPLE_FILES = [
    "ILSVRC2012_val_00000014.JPEG",
    "ILSVRC2012_val_00000048.JPEG",
    "ILSVRC2012_val_00000115.JPEG",
    "ILSVRC2012_val_00000138.JPEG",
    "ILSVRC2012_val_00000150.JPEG",
    "ILSVRC2012_val_00000154.JPEG",
    "ILSVRC2012_val_00000178.JPEG",
    "ILSVRC2012_val_00000204.JPEG",
    "ILSVRC2012_val_00000206.JPEG",
    "ILSVRC2012_val_00000212.JPEG",
    "ILSVRC2012_val_00000220.JPEG",
    "ILSVRC2012_val_00000232.JPEG",
    "ILSVRC2012_val_00000242.JPEG",
    "ILSVRC2012_val_00000253.JPEG",
    "ILSVRC2012_val_00000270.JPEG",
    "ILSVRC2012_val_00000286.JPEG",
    "ILSVRC2012_val_00000294.JPEG",
    "ILSVRC2012_val_00000299.JPEG",
    "ILSVRC2012_val_00000325.JPEG",
    "ILSVRC2012_val_00000330.JPEG",
    "ILSVRC2012_val_00000343.JPEG",
    "ILSVRC2012_val_00000356.JPEG",
    "ILSVRC2012_val_00000367.JPEG",
    "ILSVRC2012_val_00001143.JPEG",
    "ILSVRC2012_val_00001915.JPEG",
    "ILSVRC2012_val_00002541.JPEG",
    "ILSVRC2012_val_00005815.JPEG",
    "ILSVRC2012_val_00010860.JPEG",
    "ILSVRC2012_val_00010863.JPEG",
    "ILSVRC2012_val_00028489.JPEG",
]
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "imagenet_examples")
IMAGENET_EXAMPLE_FILES = [os.path.join(IMAGE_FOLDER, file) for file in IMAGENET_EXAMPLE_FILES]

# stores the configurations of all the benchmark games and how they are set up
BENCHMARK_CONFIGURATIONS: dict[Game.__class__, list[dict[str, Any]]] = {
    # local xai configurations ---------------------------------------------------------------------
    AdultCensusLocalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "imputer": "marginal"},
                {"model_name": "random_forest", "imputer": "marginal"},
                {"model_name": "gradient_boosting", "imputer": "marginal"},
                {"model_name": "decision_tree", "imputer": "conditional"},
                {"model_name": "random_forest", "imputer": "conditional"},
                {"model_name": "gradient_boosting", "imputer": "conditional"},
            ],
            "iteration_parameter": "x",
            "n_players": 14,
            "precompute": True,
        },
    ],
    BikeSharingLocalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "imputer": "marginal"},
                {"model_name": "random_forest", "imputer": "marginal"},
                {"model_name": "gradient_boosting", "imputer": "marginal"},
                {"model_name": "decision_tree", "imputer": "conditional"},
                {"model_name": "random_forest", "imputer": "conditional"},
                {"model_name": "gradient_boosting", "imputer": "conditional"},
            ],
            "iteration_parameter": "x",
            "n_players": 12,
            "precompute": True,
        },
    ],
    CaliforniaHousingLocalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "imputer": "marginal"},
                {"model_name": "random_forest", "imputer": "marginal"},
                {"model_name": "gradient_boosting", "imputer": "marginal"},
                {"model_name": "neural_network", "imputer": "marginal"},
                {"model_name": "decision_tree", "imputer": "conditional"},
                {"model_name": "random_forest", "imputer": "conditional"},
                {"model_name": "gradient_boosting", "imputer": "conditional"},
                {"model_name": "neural_network", "imputer": "conditional"},
            ],
            "iteration_parameter": "x",
            "n_players": 8,
            "precompute": True,
        },
    ],
    # Local XAI with Sentiment Analysis configurations ---------------------------------------------
    SentimentAnalysisLocalXAI: [
        {
            "configurations": [{"mask_strategy": "mask"}],
            "iteration_parameter": "input_text",
            "iteration_parameter_values": list(range(1, len(SENTIMENT_ANALYSIS_TEXTS) + 1)),
            "iteration_parameter_values_names": SENTIMENT_ANALYSIS_TEXTS,
            "n_players": 14,
            "precompute": True,
        },
    ],
    # Local XAI with Image Classifier configurations -----------------------------------------------
    ImageClassifierLocalXAI: [
        {
            "configurations": [{"model_name": "resnet_18", "n_superpixel_resnet": 14}],
            "iteration_parameter": "x_explain_path",
            "iteration_parameter_values": list(range(1, len(IMAGENET_EXAMPLE_FILES) + 1)),
            "iteration_parameter_values_names": IMAGENET_EXAMPLE_FILES,
            "n_players": 14,
            "precompute": True,
        },
        {
            "configurations": [{"model_name": "vit_9_patches"}],
            "iteration_parameter": "x_explain_path",
            "iteration_parameter_values": list(range(1, len(IMAGENET_EXAMPLE_FILES) + 1)),
            "iteration_parameter_values_names": IMAGENET_EXAMPLE_FILES,
            "n_players": 9,
            "precompute": True,
        },
        {
            "configurations": [{"model_name": "vit_16_patches"}],
            "iteration_parameter": "x_explain_path",
            "iteration_parameter_values": list(range(1, len(IMAGENET_EXAMPLE_FILES) + 1)),
            "iteration_parameter_values_names": IMAGENET_EXAMPLE_FILES,
            "n_players": 16,
            "precompute": True,
        },
    ],
    # global xai configurations --------------------------------------------------------------------
    AdultCensusGlobalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "loss_function": "accuracy_score"},
                {"model_name": "random_forest", "loss_function": "accuracy_score"},
                {"model_name": "gradient_boosting", "loss_function": "accuracy_score"},
            ],
            "iteration_parameter": "random_state",
            "n_players": 14,
            "precompute": True,
        },
    ],
    BikeSharingGlobalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "loss_function": "r2_score"},
                {"model_name": "random_forest", "loss_function": "r2_score"},
                {"model_name": "gradient_boosting", "loss_function": "r2_score"},
            ],
            "iteration_parameter": "random_state",
            "n_players": 12,
            "precompute": True,
        },
    ],
    CaliforniaHousingGlobalXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "loss_function": "r2_score"},
                {"model_name": "random_forest", "loss_function": "r2_score"},
                {"model_name": "gradient_boosting", "loss_function": "r2_score"},
                {"model_name": "neural_network", "loss_function": "r2_score"},
            ],
            "iteration_parameter": "random_state",
            "n_players": 8,
            "precompute": True,
        },
    ],
    # feature selection configurations -------------------------------------------------------------
    AdultCensusFeatureSelection: [
        {
            "configurations": [
                {"model_name": "decision_tree"},
                {"model_name": "random_forest"},
                {"model_name": "gradient_boosting"},
            ],
            "iteration_parameter": "random_state",
            "n_players": 14,
            "precompute": True,
        },
    ],
    BikeSharingFeatureSelection: [
        {
            "configurations": [
                {"model_name": "decision_tree"},
                {"model_name": "random_forest"},
                {"model_name": "gradient_boosting"},
            ],
            "iteration_parameter": "random_state",
            "n_players": 12,
            "precompute": True,
        },
    ],
    CaliforniaHousingFeatureSelection: [
        {
            "configurations": [
                {"model_name": "decision_tree"},
                {"model_name": "random_forest"},
                {"model_name": "gradient_boosting"},
                # {"model_name": "neural_network"}  # not possible atm. needs dynamic input size
            ],
            "iteration_parameter": "random_state",
            "n_players": 8,
            "precompute": True,
        },
    ],
    # ensemble selection configurations ------------------------------------------------------------
    AdultCensusEnsembleSelection: [
        {
            "configurations": [{"loss_function": "accuracy_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    BikeSharingEnsembleSelection: [
        {
            "configurations": [{"loss_function": "r2_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    CaliforniaHousingEnsembleSelection: [
        {
            "configurations": [{"loss_function": "r2_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    # ensemble selection with random forest configurations -----------------------------------------
    AdultCensusRandomForestEnsembleSelection: [
        {
            "configurations": [{"loss_function": "accuracy_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    BikeSharingRandomForestEnsembleSelection: [
        {
            "configurations": [{"loss_function": "r2_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    CaliforniaHousingRandomForestEnsembleSelection: [
        {
            "configurations": [{"loss_function": "r2_score", "n_members": 10}],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
    ],
    # dataset valuation configurations -------------------------------------------------------------
    AdultCensusDatasetValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
            ],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 14},
            ],
            "iteration_parameter": "random_state",
            "n_players": 14,
            "iteration_parameter_values": list(range(1, 5 + 1)),
            "precompute": True,
        },
    ],
    BikeSharingDatasetValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
            ],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 14},
            ],
            "iteration_parameter": "random_state",
            "n_players": 14,
            "iteration_parameter_values": list(range(1, 5 + 1)),
            "precompute": True,
        },
    ],
    CaliforniaHousingDatasetValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 10},
                {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 10},
            ],
            "iteration_parameter": "random_state",
            "n_players": 10,
            "precompute": True,
        },
        {
            "configurations": [
                {"model_name": "decision_tree", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "random_forest", "player_sizes": "increasing", "n_players": 14},
                # {"model_name": "gradient_boosting", "player_sizes": "increasing", "n_players": 14},
            ],
            "iteration_parameter": "random_state",
            "n_players": 14,
            "iteration_parameter_values": list(range(1, 5 + 1)),
            "precompute": True,
        },
    ],
    # data valuation configurations ----------------------------------------------------------------
    AdultCensusDataValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "n_data_points": 15},
                {"model_name": "random_forest", "n_data_points": 15},
            ],
            "iteration_parameter": "random_state",
            "n_players": 15,
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "precompute": True,
        }
    ],
    BikeSharingDataValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "n_data_points": 15},
                {"model_name": "random_forest", "n_data_points": 15},
            ],
            "iteration_parameter": "random_state",
            "n_players": 15,
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "precompute": True,
        }
    ],
    CaliforniaHousingDataValuation: [
        {
            "configurations": [
                {"model_name": "decision_tree", "n_data_points": 15},
                {"model_name": "random_forest", "n_data_points": 15},
            ],
            "iteration_parameter": "random_state",
            "n_players": 15,
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "precompute": True,
        }
    ],
    # cluster explanation configurations -----------------------------------------------------------
    BikeSharingClusterExplanation: [
        {
            "configurations": [
                {"cluster_method": "kmeans", "score_method": "silhouette_score"},
                {"cluster_method": "agglomerative", "score_method": "calinski_harabasz_score"},
            ],
            "iteration_parameter": "random_state",  # for agglomerative this does not change the game
            "iteration_parameter_values": [1],  # for agglomerative this does not change the game
            "n_players": 12,
            "precompute": True,
        },
    ],
    CaliforniaHousingClusterExplanation: [
        {
            "configurations": [
                {"cluster_method": "kmeans", "score_method": "silhouette_score"},
                {"cluster_method": "agglomerative", "score_method": "calinski_harabasz_score"},
            ],
            "iteration_parameter": "random_state",  # for agglomerative this does not change the game
            "iteration_parameter_values": [1],  # for agglomerative this does not change the game
            "n_players": 8,
            "precompute": True,
        },
    ],
    # unsupervised data configurations -------------------------------------------------------------
    AdultCensusUnsupervisedData: [
        {
            "configurations": [{}],
            "iteration_parameter": "random_state",  # this does not change the game
            "iteration_parameter_values": [1],  # this does not change the game
            "n_players": 14,
            "precompute": True,
        },
    ],
    BikeSharingUnsupervisedData: [
        {
            "configurations": [{}],
            "iteration_parameter": "random_state",  # this does not change the game
            "iteration_parameter_values": [1],  # this does not change the game
            "n_players": 12,
            "precompute": True,
        },
    ],
    CaliforniaHousingUnsupervisedData: [
        {
            "configurations": [{}],
            "iteration_parameter": "random_state",  # this does not change the game
            "iteration_parameter_values": [1],  # this does not change the game
            "n_players": 8,
            "precompute": True,
        },
    ],
    # uncertainty explanation configurations -------------------------------------------------------
    AdultCensusUncertaintyExplanation: [
        {
            "configurations": [
                {"uncertainty_to_explain": "total", "imputer": "marginal"},
                {"uncertainty_to_explain": "total", "imputer": "conditional"},
                {"uncertainty_to_explain": "aleatoric", "imputer": "marginal"},
                {"uncertainty_to_explain": "aleatoric", "imputer": "conditional"},
                {"uncertainty_to_explain": "epistemic", "imputer": "marginal"},
                {"uncertainty_to_explain": "epistemic", "imputer": "conditional"},
            ],
            "iteration_parameter": "x",
            "n_players": 14,
            "precompute": True,
        },
    ],
    # TreeSHAPIQXAI configurations -----------------------------------------------------------------
    SynthDataTreeSHAPIQXAI: [
        {
            "configurations": [
                {"model_name": "decision_tree", "classification": True, "n_features": 30},
                {"model_name": "random_forest", "classification": True, "n_features": 30},
                {"model_name": "decision_tree", "classification": False, "n_features": 30},
                {"model_name": "random_forest", "classification": False, "n_features": 30},
            ],
            "iteration_parameter": "x",
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "n_players": 30,
            "precompute": False,
        },
    ],
    # SOUM configurations --------------------------------------------------------------------------
    SOUM: [
        {
            "configurations": [
                {
                    "n": 15,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 15,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
                {
                    "n": 15,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 15,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
            ],
            "iteration_parameter": "random_state",
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "n_players": 15,
            "precompute": True,
        },
        {
            "configurations": [
                {
                    "n": 30,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 30,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
                {
                    "n": 30,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 25,
                },
                {
                    "n": 30,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 30,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
                {
                    "n": 30,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 25,
                },
            ],
            "iteration_parameter": "random_state",
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "n_players": 30,
            "precompute": False,
        },
        {
            "configurations": [
                {
                    "n": 50,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 50,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
                {
                    "n": 50,
                    "n_basis_games": 30,
                    "min_interaction_size": 1,
                    "max_interaction_size": 25,
                },
                {
                    "n": 50,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 5,
                },
                {
                    "n": 50,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 15,
                },
                {
                    "n": 50,
                    "n_basis_games": 150,
                    "min_interaction_size": 1,
                    "max_interaction_size": 25,
                },
            ],
            "iteration_parameter": "random_state",
            "iteration_parameter_values": list(range(1, 10 + 1)),
            "n_players": 50,
            "precompute": False,
        },
    ],
}


GAME_NAME_TO_CLASS_MAPPING = {
    "AdultCensusDatasetValuation": AdultCensusDatasetValuation,
    "AdultCensusDataValuation": AdultCensusDataValuation,
    "AdultCensusEnsembleSelection": AdultCensusEnsembleSelection,
    "AdultCensusFeatureSelection": AdultCensusFeatureSelection,
    "AdultCensusGlobalXAI": AdultCensusGlobalXAI,
    "AdultCensusLocalXAI": AdultCensusLocalXAI,
    "AdultCensusRandomForestEnsembleSelection": AdultCensusRandomForestEnsembleSelection,
    "AdultCensusUnsupervisedData": AdultCensusUnsupervisedData,
    "AdultCensusUncertaintyExplanation": AdultCensusUncertaintyExplanation,
    "BikeSharingClusterExplanation": BikeSharingClusterExplanation,
    "BikeSharingDataValuation": BikeSharingDataValuation,
    "BikeSharingDatasetValuation": BikeSharingDatasetValuation,
    "BikeSharingEnsembleSelection": BikeSharingEnsembleSelection,
    "BikeSharingFeatureSelection": BikeSharingFeatureSelection,
    "BikeSharingGlobalXAI": BikeSharingGlobalXAI,
    "BikeSharingLocalXAI": BikeSharingLocalXAI,
    "BikeSharingRandomForestEnsembleSelection": BikeSharingRandomForestEnsembleSelection,
    "BikeSharingUnsupervisedData": BikeSharingUnsupervisedData,
    "CaliforniaHousingClusterExplanation": CaliforniaHousingClusterExplanation,
    "CaliforniaHousingDatasetValuation": CaliforniaHousingDatasetValuation,
    "CaliforniaHousingDataValuation": CaliforniaHousingDataValuation,
    "CaliforniaHousingEnsembleSelection": CaliforniaHousingEnsembleSelection,
    "CaliforniaHousingFeatureSelection": CaliforniaHousingFeatureSelection,
    "CaliforniaHousingGlobalXAI": CaliforniaHousingGlobalXAI,
    "CaliforniaHousingLocalXAI": CaliforniaHousingLocalXAI,
    "CaliforniaHousingRandomForestEnsembleSelection": CaliforniaHousingRandomForestEnsembleSelection,
    "CaliforniaHousingUnsupervisedData": CaliforniaHousingUnsupervisedData,
    "SentimentAnalysisLocalXAI": SentimentAnalysisLocalXAI,
    "ImageClassifierLocalXAI": ImageClassifierLocalXAI,
    "SynthDataTreeSHAPIQXAI": SynthDataTreeSHAPIQXAI,
    "SOUM": SOUM,
}
GAME_CLASS_TO_NAME_MAPPING = {
    game_cls: name for name, game_cls in GAME_NAME_TO_CLASS_MAPPING.items()
}

APPROXIMATION_CONFIGURATIONS: dict[str, Approximator.__class__] = {
    "SV": SV_APPROXIMATORS,
    "SI": SI_APPROXIMATORS,
    "SII": SII_APPROXIMATORS,
    "k-SII": SII_APPROXIMATORS,  # "k-SII" is the same as "SII"
    "STII": STII_APPROXIMATORS,
    "FSII": FSII_APPROXIMATORS,
}

APPROXIMATION_NAME_TO_CLASS_MAPPING = {
    approx.__name__: approx
    for approx_list in APPROXIMATION_CONFIGURATIONS.values()
    for approx in approx_list
}

# contains all parameters that will be passed to the approximators at initialization
APPROXIMATION_BENCHMARK_PARAMS: dict[Approximator.__class__, tuple[str]] = {}
APPROXIMATION_BENCHMARK_PARAMS.update(
    {approx: ("n", "random_state") for approx in SV_APPROXIMATORS}
)
APPROXIMATION_BENCHMARK_PARAMS.update(
    {
        approx: ("n", "random_state", "index", "max_order")
        for approx in SI_APPROXIMATORS + SII_APPROXIMATORS + STII_APPROXIMATORS + FSII_APPROXIMATORS
    }
)


def get_game_file_name_from_config(
    configuration: dict[str, Any], iteration: Optional[int] = None
) -> str:
    """Get the file name for the game data with the given configuration and iteration.

    Args:
        configuration: A configuration of the game class.
        iteration: The iteration of the game. Defaults to None.

    Returns:
        The file name of the game data
    """
    file_name = "_".join(f"{key}={value}" for key, value in configuration.items())
    if iteration is not None:
        file_name = "_".join([file_name, str(iteration)])
    return file_name


def load_game_data(
    game_class: Game.__class__,
    configuration: dict[str, Any],
    iteration: int = 1,
    n_player_id: int = 0,
) -> Game:
    """Loads the precomputed game data for the given game class and configuration.

    Args:
        game_class: The class of the game
        configuration: The configuration to use to load the game
        iteration: The iteration of the game to load
        n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.

    Returns:
        An initialized game object with the given configuration

    Raises:
        FileNotFoundError: If the file with the precomputed values does not exist
    """
    n_players = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["n_players"]
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
        return Game(
            path_to_values=path_to_values,
            verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
            normalize=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["normalize"],
        )
    except FileNotFoundError:
        # download the game data if it does not exist
        download_game_data(game_class.get_game_name(), n_players, file_name)
        try:
            return Game(
                path_to_values=path_to_values,
                verbose=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["verbose"],
                normalize=BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS["normalize"],
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Game data for game {game_class.get_game_name()} with configuration "
                f"{configuration} and iteration {iteration} could not be found."
            ) from error


def download_game_data(game_name: str, n_players: int, file_name: str) -> None:
    """Downloads the game file from the repository.

    Args:
        game_name: The name of the game.
        n_players: The number of players in the game.
        file_name: The name of the file to download.

    Raises:
        FileNotFoundError: If the file could not be downloaded.
    """
    github_url = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/precomputed_games"

    # create the directory if it does not exist
    game_dir = str(os.path.join(SHAPIQ_DATA_DIR, game_name, str(n_players)))
    os.makedirs(game_dir, exist_ok=True)

    # download the file
    path = os.path.join(game_dir, f"{file_name}.npz")
    url = f"{github_url}/{game_name}/{n_players}/{file_name}.npz"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        raise FileNotFoundError(
            f"Could not download the game data from {url}. Check if configuration is correct."
        ) from error
    with open(path, "wb") as file:
        file.write(response.content)
        time.sleep(0.01)


def get_game_class_from_name(game_name: str) -> Game.__class__:
    """Get the game class from the name of the game.

    Args:
        game_name: The name of the game.

    Returns:
        The class of the game
    """
    return GAME_NAME_TO_CLASS_MAPPING[game_name]


def get_name_from_game_class(game_class: Game.__class__) -> str:
    """Get the name of the game from the class of the game

    Args:
        game_class: The class of the game.

    Returns:
        The name of the game.
    """
    for name, game_cls in GAME_NAME_TO_CLASS_MAPPING.items():
        if game_cls == game_class:
            return name
    raise ValueError(f"Game class {game_class} not found in the mapping.")


def print_benchmark_configurations() -> None:
    """Print the configurations of the benchmark games."""
    # print configurations of the benchmark games
    game_classes = list(BENCHMARK_CONFIGURATIONS.keys())
    game_identifiers = [GAME_CLASS_TO_NAME_MAPPING[game_class] for game_class in game_classes]
    game_identifiers = sorted(game_identifiers)
    for game_identifier in game_identifiers:
        game_class = GAME_NAME_TO_CLASS_MAPPING[game_identifier]
        config_per_player_id = BENCHMARK_CONFIGURATIONS[game_class]
        print(f"Game: {game_identifier}")
        for player_id, configurations in enumerate(config_per_player_id):
            print(f"Player ID: {player_id}")
            print(f"Number of Players: {configurations['n_players']}")
            print(f"Number of configurations: {len(configurations['configurations'])}")
            print(f"Is the Benchmark Pre-computed: {configurations['precompute']}")
            print(f"Iteration Parameter: {configurations['iteration_parameter']}")
            print("Configurations:")
            for i, configuration in enumerate(configurations["configurations"]):
                print(f"Configuration {i + 1}: {configuration}")
        print()


def load_games_from_configuration(
    game_class: Union[Game.__class__, str],
    config_id: int,
    n_games: Optional[int] = None,
    n_player_id: int = 0,
    check_pre_computed: bool = True,
    only_pre_computed: bool = True,
) -> Generator[Game, None, None]:
    """Load the game with the given configuration from disk or create it if it does not exist.

    Args:
        game_class: The class of the game to load with the configuration.
        config_id: The configuration to use to load the game.
        n_games: The number of games to load. Defaults to None.
        n_player_id: The player ID to use. Defaults to 0. Not all games have multiple player IDs.
        check_pre_computed: A flag to check if the game is pre-computed (load from disk). Defaults
            to True.
        only_pre_computed: A flag to only load the pre-computed games. Defaults to False.

    Returns:
        An initialized game object with the given configuration.
    """
    game_class = (
        GAME_NAME_TO_CLASS_MAPPING[game_class] if isinstance(game_class, str) else game_class
    )

    # get config if it is an int
    configuration: dict = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]["configurations"][
        config_id - 1
    ]
    params = {}

    # get the default parameters
    default_params = BENCHMARK_CONFIGURATIONS_DEFAULT_PARAMS.copy()
    params.update(default_params)
    params.update(configuration)

    # get the class-specific configurations of how the iterations are set up
    config_of_class = BENCHMARK_CONFIGURATIONS[game_class][n_player_id]
    game_should_be_precomputed = config_of_class["precompute"]
    iteration_param = config_of_class["iteration_parameter"]
    iteration_param_values = config_of_class.get(
        "iteration_parameter_values", BENCHMARK_CONFIGURATIONS_DEFAULT_ITERATIONS
    )
    iteration_param_values_names = config_of_class.get(
        "iteration_parameter_values_names", iteration_param_values
    )

    # create the generator of games
    n_games = (
        len(iteration_param_values)
        if n_games is None
        else min(n_games, len(iteration_param_values))
    )
    for i in range(n_games):
        game_iteration = iteration_param_values[i]  # from 1 to 30
        game_iteration_value = iteration_param_values_names[i]  # i.e. the sentence or random state
        params[iteration_param] = game_iteration_value  # set the iteration parameter
        if not game_should_be_precomputed:  # e.g. for SynthDataTreeSHAPIQXAI
            yield game_class(**params)
        elif not check_pre_computed and not only_pre_computed:
            yield game_class(**params)
        else:
            try:  # try to load the game from disk
                yield load_game_data(
                    game_class, configuration, iteration=game_iteration, n_player_id=n_player_id
                )
            except FileNotFoundError:
                if only_pre_computed:  # if only pre-computed games are requested, skip the game
                    continue
                else:  # fallback to creating the game which is not pre-computed
                    yield game_class(**params)
