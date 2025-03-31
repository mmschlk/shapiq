"""This module contains the TreeSHAP-IQ explanation benchmark games."""

from .base import TreeSHAPIQXAI
from .benchmark import AdultCensus, BikeSharing, CaliforniaHousing, SynthData

__all__ = ["TreeSHAPIQXAI", "CaliforniaHousing", "AdultCensus", "BikeSharing", "SynthData"]
