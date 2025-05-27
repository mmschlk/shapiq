"""Sparse fourier-based approximators for higher-order interactions."""

from ._base import Sparse
from .spex import SPEX

__all__ = [
    "SPEX",
    "Sparse",
]
