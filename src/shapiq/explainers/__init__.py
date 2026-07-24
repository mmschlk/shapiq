"""Explainer entry points: everything here maps a game to a game."""

from __future__ import annotations

from shapiq.explainers._approximator import Approximator
from shapiq.explainers._binding import Explainer
from shapiq.explainers._exact import ExactExplainer
from shapiq.explainers._montecarlo import SHAPIQ, SVARMIQ
from shapiq.explainers._permutation import PermutationSampling
from shapiq.explainers._regression import Regression
from shapiq.explainers._tree import TreeExplainer

__all__ = [
    "SHAPIQ",
    "SVARMIQ",
    "Approximator",
    "ExactExplainer",
    "Explainer",
    "PermutationSampling",
    "Regression",
    "TreeExplainer",
]
