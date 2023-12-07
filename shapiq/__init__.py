"""shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""
from __version__ import __version__

# approximator classes
from .approximator import (
    PermutationSamplingSII,
    PermutationSamplingSTI,
    RegressionSII,
    RegressionFSI,
    ShapIQ,
)

# explainer classes
from .explainer import InteractionExplainer

# game classes
from .games import DummyGame

# plotting functions
from .plot import network_plot

# public utils functions
from .utils import (  # sets.py  # tree.py
    get_conditional_sample_weights,
    get_explicit_subsets,
    get_parent_array,
    powerset,
    split_subsets_budget,
)

__all__ = [
    # version
    "__version__",
    # approximators
    "ShapIQ",
    "PermutationSamplingSII",
    "PermutationSamplingSTI",
    "RegressionSII",
    "RegressionFSI",
    # explainers
    "InteractionExplainer",
    # games
    "DummyGame",
    # plots
    "network_plot",
    # public utils
    "powerset",
    "get_explicit_subsets",
    "split_subsets_budget",
    "get_conditional_sample_weights",
    "get_parent_array",
]
