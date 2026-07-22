"""Explainer abstractions."""

from __future__ import annotations

from shapiq.explainers._approximator import Approximator
from shapiq.explainers._base import Explainer
from shapiq.explainers._exact import ExactExplainer
from shapiq.explainers._permutation import PermutationSampling
from shapiq.explainers._regression import Regression
from shapiq.explainers._tree import TreeExplainer

__all__ = [
    "Approximator",
    "ExactExplainer",
    "Explainer",
    "PermutationSampling",
    "Regression",
    "TreeExplainer",
]
