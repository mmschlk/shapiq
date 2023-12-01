"""This module contains the approximators to estimate the Shapley interaction values."""
from ._base import convert_nsii_into_one_dimension, transforms_sii_to_nsii  # TODO add to tests
from .permutation.sii import PermutationSamplingSII
from .permutation.sti import PermutationSamplingSTI
from .regression import Regression
from .shapiq import ShapIQ

__all__ = [
    "PermutationSamplingSII",
    "PermutationSamplingSTI",
    "Regression",
    "ShapIQ",
    "transforms_sii_to_nsii",
    "convert_nsii_into_one_dimension",
]
