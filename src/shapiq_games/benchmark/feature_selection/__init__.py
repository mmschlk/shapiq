"""This module contains the feature selection benchmarking games."""

from .base import FeatureSelection
from .benchmark import AdultCensus, BikeSharing, CaliforniaHousing

__all__ = ["FeatureSelection", "AdultCensus", "BikeSharing", "CaliforniaHousing"]
