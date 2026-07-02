"""shapiq: Shapley Interactions for Machine Learning.

shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version.py is generated at build time
    __version__ = "0.0.0"

# approximator classes
from .approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    OddSHAP,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    ProxySHAP,
    ProxySPEX,
    RegressionFBII,
    RegressionFSII,
    RegressionMSR,
    ShaplEIG,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)

# dataset functions
from .datasets import load_adult_census, load_bike_sharing, load_california_housing

# explainer classes
from .explainer import (
    AgnosticExplainer,
    Explainer,
    TabPFNExplainer,
    TabularExplainer,
)

# game classes
from .game import Game

# exact computer classes
from .game_theory.exact import ExactComputer

# imputer classes
from .imputer import (
    BaselineImputer,
    GaussianCopulaImputer,
    GaussianImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)

# base classes
from .interaction_values import InteractionValues

# plotting functions
from .plot import (
    bar_plot,
    beeswarm_plot,
    force_plot,
    network_plot,
    scatter_plot,
    sentence_plot,
    si_graph_plot,
    stacked_bar_plot,
    upset_plot,
    waterfall_plot,
)
from .tree import TreeExplainer

# public utils functions
from .utils import (  # sets.py  # tree.py
    get_explicit_subsets,
    powerset,
    safe_isinstance,
    split_subsets_budget,
)
from .vision import ImageExplainer

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
    "ProxySHAP",
    "ShaplEIG",
    "ProxySPEX",
    "RegressionMSR",
    "OddSHAP",
    # explainers
    "Explainer",
    "TabularExplainer",
    "TreeExplainer",
    "TabPFNExplainer",
    "AgnosticExplainer",
    "ImageExplainer",
    # imputers
    "MarginalImputer",
    "BaselineImputer",
    "GenerativeConditionalImputer",
    "TabPFNImputer",
    "GaussianImputer",
    "GaussianCopulaImputer",
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
    "scatter_plot",
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
