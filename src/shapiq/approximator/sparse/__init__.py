"""Sparse fourier-based approximators for higher-order interactions."""

from .base import Sparse
from .proxyspex import ProxySPEX
from .spex import SPEX

__all__ = [
    "ProxySPEX",
    "SPEX",
    "Sparse",
]
