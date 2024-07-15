"""shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

__version__ = "1.0.1.9001"

# approximator classes
from .approximator import (
    SHAPIQ,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)

# dataset functions
from .datasets import load_adult_census, load_bike_sharing, load_california_housing

# exact computer classes
from .exact import ExactComputer

# explainer classes
from .explainer import Explainer, TabularExplainer, TreeExplainer

# game classes
from .games import ConditionalImputer, Game, MarginalImputer

# base classes
from .interaction_values import InteractionValues

# plotting functions
from .plot import bar_plot, force_plot, network_plot, si_graph_plot, stacked_bar_plot

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
    "KernelSHAPIQ",
    "InconsistentKernelSHAPIQ",
    "SHAPIQ",
    "SVARM",
    "SVARMIQ",
    "kADDSHAP",
    "UnbiasedKernelSHAP",
    # explainers
    "Explainer",
    "TabularExplainer",
    "TreeExplainer",
    # imputers
    "MarginalImputer",
    "ConditionalImputer",
    # plots
    "network_plot",
    "stacked_bar_plot",
    "force_plot",
    "bar_plot",
    "si_graph_plot",
    # public utils
    "powerset",
    "get_explicit_subsets",
    "split_subsets_budget",
    "safe_isinstance",
    # datasets
    "load_bike_sharing",
    "load_adult_census",
    "load_california_housing",
]
