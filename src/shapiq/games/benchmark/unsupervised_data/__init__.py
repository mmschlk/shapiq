"""This module contains the benchmark games for the unsupervised data analysis setting."""

from .base import UnsupervisedData
from .benchmark import AdultCensus, BikeSharing, CaliforniaHousing

__all__ = ["UnsupervisedData", "AdultCensus", "BikeSharing", "CaliforniaHousing"]
