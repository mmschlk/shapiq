"""shapiq is a library creating explanations for machine learning models based on
the well established Shapley value and its generalization to interaction.
"""

import warnings

from .__version__ import __version__

# plotting functions
from .plot import network_plot


__all__ = [
    # plots
    "network_plot",
]
