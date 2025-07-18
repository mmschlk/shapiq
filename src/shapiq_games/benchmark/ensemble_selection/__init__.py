"""This module contains the ensemble selection benchmarks."""

from .base import EnsembleSelection, RandomForestEnsembleSelection
from .benchmark import (
    AdultCensus as AdultCensusEnsembleSelection,
    BikeSharing as BikeSharingEnsembleSelection,
    CaliforniaHousing as CaliforniaHousingEnsembleSelection,
)
from .benchmark_random_forest import (
    AdultCensus as AdultCensusRandomForestEnsembleSelection,
    BikeSharing as BikeSharingRandomForestEnsembleSelection,
    CaliforniaHousing as CaliforniaHousingRandomForestEnsembleSelection,
)

__all__ = [
    "EnsembleSelection",
    "RandomForestEnsembleSelection",
    "AdultCensusEnsembleSelection",
    "BikeSharingEnsembleSelection",
    "CaliforniaHousingEnsembleSelection",
    "AdultCensusRandomForestEnsembleSelection",
    "BikeSharingRandomForestEnsembleSelection",
    "CaliforniaHousingRandomForestEnsembleSelection",
]
