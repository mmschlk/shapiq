"""Graph-based explanation algorithms, including GraphSHAP-IQ.

Provides :class:`GraphSHAPIQ` for computing exact and approximate Shapley
interaction values for graph neural networks, along with supporting
algorithms and utilities.
"""

from .base import GraphGame
from .explainer import GraphExplainer
from .graphshapiq import GraphSHAPIQ

__all__ = ["GraphGame", "GraphExplainer", "GraphSHAPIQ"]
