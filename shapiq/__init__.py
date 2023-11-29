"""shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

import warnings

from .__version__ import __version__

# approximator classes
from .approximator import ShapIQ
from .approximator import PermutationSamplingSII
from .approximator import PermutationSamplingSTI
from .approximator import RegressionFSI

# explainer classes
from .explainer import Explainer

# plotting functions
from .plot import network_plot

# game classes
from .games import DummyGame

# public utils functions
from .utils import powerset, get_explicit_subsets, split_subsets_budget  # sets.py
from .utils import get_conditional_sample_weights, get_parent_array  # tree.py


__all__ = [
    # approximators
    "ShapIQ",
    "PermutationSamplingSII",
    "PermutationSamplingSTI",
    "RegressionFSI",
    # explainers
    "Explainer",
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
