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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapiq.approximator import (
    FSII_APPROXIMATORS,
    SI_APPROXIMATORS,
    SII_APPROXIMATORS,
    STII_APPROXIMATORS,
    SV_APPROXIMATORS,
    Approximator,
)
from shapiq.games.benchmark import (
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

if TYPE_CHECKING:
    from shapiq.games.base import Game

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

# ImageNet example files for the Image Classifier Game:
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

IMAGE_FOLDER = Path(__file__).parent / "imagenet_examples"
IMAGENET_EXAMPLE_FILES = [IMAGE_FOLDER / file for file in IMAGENET_EXAMPLE_FILES]

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
        },
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
        },
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
        },
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
    dict.fromkeys(SV_APPROXIMATORS, ("n", "random_state")),
)
APPROXIMATION_BENCHMARK_PARAMS.update(
    dict.fromkeys(
        SI_APPROXIMATORS + SII_APPROXIMATORS + STII_APPROXIMATORS + FSII_APPROXIMATORS,
        ("n", "random_state", "index", "max_order"),
    ),
)


def get_game_file_name_from_config(
    configuration: dict[str, Any],
    iteration: int | None = None,
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


def get_game_class_from_name(game_name: str) -> Game.__class__:
    """Get the game class from the name of the game.

    Args:
        game_name: The name of the game.

    Returns:
        The class of the game

    """
    return GAME_NAME_TO_CLASS_MAPPING[game_name]


def get_name_from_game_class(game_class: Game.__class__) -> str:
    """Get the name of the game from the class of the game.

    Args:
        game_class: The class of the game.

    Returns:
        The name of the game.

    """
    for name, game_cls in GAME_NAME_TO_CLASS_MAPPING.items():
        if game_cls == game_class:
            return name
    msg = f"Game class {game_class} not found in the mapping."
    raise ValueError(msg)


def print_benchmark_configurations() -> None:
    """Print the configurations of the benchmark games."""
    game_classes = list(BENCHMARK_CONFIGURATIONS.keys())
    game_identifiers = [GAME_CLASS_TO_NAME_MAPPING[game_class] for game_class in game_classes]
    game_identifiers = sorted(game_identifiers)
    for game_identifier in game_identifiers:
        game_class = GAME_NAME_TO_CLASS_MAPPING[game_identifier]
        config_per_player_id = BENCHMARK_CONFIGURATIONS[game_class]
        for _player_id, configurations in enumerate(config_per_player_id):
            for _i, _configuration in enumerate(configurations["configurations"]):
                pass
