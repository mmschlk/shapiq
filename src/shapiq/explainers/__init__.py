"""Explainer abstractions."""

from __future__ import annotations

from shapiq.explainers._approximator import Approximator
from shapiq.explainers._base import Explainer
from shapiq.explainers._evidence import EvidenceApproximator
from shapiq.explainers._exact import ExactExplainer, ExactIndex
from shapiq.explainers._permutation import PermutationIndex, PermutationSampling
from shapiq.explainers._regression import KernelRegressionIndex, Regression

__all__ = [
    "Approximator",
    "EvidenceApproximator",
    "ExactExplainer",
    "ExactIndex",
    "Explainer",
    "KernelRegressionIndex",
    "PermutationIndex",
    "PermutationSampling",
    "Regression",
]
