"""Explanation array abstractions."""

from __future__ import annotations

from shapiq.explanations._base import ExplanationArray
from shapiq.explanations._dense import DenseExplanationArray
from shapiq.explanations._sparse import SparseExplanationArray

__all__ = ["DenseExplanationArray", "ExplanationArray", "SparseExplanationArray"]
