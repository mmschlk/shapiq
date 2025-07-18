"""This module contains all global explanation benchmark games."""

from .base import GlobalExplanation
from .benchmark_tabular import AdultCensus, BikeSharing, CaliforniaHousing

__all__ = [
    "GlobalExplanation",
    "AdultCensus",
    "BikeSharing",
    "CaliforniaHousing",
]
