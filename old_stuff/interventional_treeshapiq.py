from shapiq.games.benchmark.local_xai import AdultCensus, CaliforniaHousing, BikeSharing
from shapiq.games.benchmark.local_xai.benchmark_tabular import (
    ForestFires,
    RealEstate,
    BreastCancer,
    NHANESI,
    WineQuality,
    CommunitiesAndCrime,
    IndependentLinear60,
    Corrgroups60,
)

from shapiq.games.benchmark.treeshapiq_xai import TreeSHAPIQXAI

from shapiq.explainer.tree import TreeSHAPIQ

from shapiq import TreeExplainer
import numpy as np

from shapiq import ExactComputer

from shapiq import KernelSHAP, PermutationSamplingSV, SPEX
from shapiq.approximator.regression.polyshap import (
    ShapleyGAX,
    ExplanationBasisGenerator,
)

from shapiq.utils.empirical_leverage_scores import get_leverage_scores

from scipy.special import binom

import multiprocessing as mp
import tqdm


if __name__ == "__main__":
    ID_EXPLANATIONS = range(
        10
    )  # range(10,30) # ids of test instances to explain, can be used to compute new ids
    RANDOM_STATE = 40  # random state for the games
    # run the benchmark for the games
    game = CaliforniaHousing(
        model_name="random_forest", imputer="baseline", random_state=RANDOM_STATE
    )

    feature_perturbation = "interventional"
    background_data = game.setup.x_test[:100, :]

    x_explain = game.setup.x_test[ID_EXPLANATIONS, :]
    tree_game = TreeSHAPIQXAI(x_explain, game.setup.model, verbose=False)
