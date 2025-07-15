"""shapiq: Shapley Interactions for Machine Learning.

shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

__version__ = "1.3.1"

# approximator classes
from .approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFBII,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from .benchmark import (
    BENCHMARK_CONFIGURATIONS,
    GAME_CLASS_TO_NAME_MAPPING,
    GAME_NAME_TO_CLASS_MAPPING,
    download_game_data,
    load_benchmark_results,
    load_game_data,
    load_games_from_configuration,
    plot_approximation_quality,
    print_benchmark_configurations,
    run_benchmark,
    run_benchmark_from_configuration,
)

# dataset functions
from .datasets import load_adult_census, load_bike_sharing, load_california_housing

# explainer classes
from .explainer import (
    AgnosticExplainer,
    Explainer,
    TabPFNExplainer,
    TabularExplainer,
    TreeExplainer,
)

# exact computer classes
from .game_theory.exact import ExactComputer

# game classes
# imputer classes
from .games import BaselineImputer, ConditionalImputer, Game, MarginalImputer, TabPFNImputer

# base classes
from .interaction_values import InteractionValues

# plotting functions
from .plot import (
    bar_plot,
    beeswarm_plot,
    force_plot,
    network_plot,
    sentence_plot,
    si_graph_plot,
    stacked_bar_plot,
    upset_plot,
    waterfall_plot,
)

# public utils functions
from .utils import (  # sets.py  # tree.py
    get_explicit_subsets,
    powerset,
    safe_isinstance,
    split_subsets_budget,
)

__all__ = [
    # version
    "__version__",
    # base
    "InteractionValues",
    "ExactComputer",
    "Game",
    # approximators
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "PermutationSamplingSV",
    "StratifiedSamplingSV",
    "OwenSamplingSV",
    "KernelSHAP",
    "RegressionFSII",
    "RegressionFBII",
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "SHAPIQ",
    "SVARM",
    "SVARMIQ",
    "kADDSHAP",
    "UnbiasedKernelSHAP",
    "SPEX",
    # explainers
    "Explainer",
    "TabularExplainer",
    "TreeExplainer",
    "TabPFNExplainer",
    "AgnosticExplainer",
    # imputers
    "MarginalImputer",
    "BaselineImputer",
    "ConditionalImputer",
    "TabPFNImputer",
    # plots
    "network_plot",
    "stacked_bar_plot",
    "force_plot",
    "bar_plot",
    "si_graph_plot",
    "waterfall_plot",
    "sentence_plot",
    "upset_plot",
    "beeswarm_plot",
    # public utils
    "powerset",
    "get_explicit_subsets",
    "split_subsets_budget",
    "safe_isinstance",
    # datasets
    "load_bike_sharing",
    "load_adult_census",
    "load_california_housing",
    # benchmark
    "plot_approximation_quality",
    "run_benchmark",
    "run_benchmark_from_configuration",
    "load_benchmark_results",
    "print_benchmark_configurations",
    "BENCHMARK_CONFIGURATIONS",
    "GAME_CLASS_TO_NAME_MAPPING",
    "GAME_NAME_TO_CLASS_MAPPING",
    "load_games_from_configuration",
    "download_game_data",
    "load_game_data",
]
