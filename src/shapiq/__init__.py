"""shapiq: Shapley Interactions for Machine Learning.

shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# approximator classes
from ._deprecated_redirects import try_import_deprecated
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
    ProxySPEX,
    RegressionFBII,
    RegressionFSII,
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
    TreeExplainer,
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
    "ProxySPEX",
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


def __getattr__(name: str) -> ModuleType | None:
    """Redirect deprecated imports to the new module."""
    return try_import_deprecated(name)
