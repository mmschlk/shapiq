"""This module contains the ensemble selection benchmarks."""

from .base import EnsembleSelection, RandomForestEnsembleSelection
from .benchmark import AdultCensus as AdultCensusEnsembleSelection
from .benchmark import BikeSharing as BikeSharingEnsembleSelection
from .benchmark import CaliforniaHousing as CaliforniaHousingEnsembleSelection
from .benchmark_random_forest import AdultCensus as AdultCensusRandomForestEnsembleSelection
from .benchmark_random_forest import BikeSharing as BikeSharingRandomForestEnsembleSelection
from .benchmark_random_forest import (
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

# Path: shapiq/games/benchmark/ensemble_selection/__init__.py
