"""This module contains the Interventional TreeSHAP-IQ explanation benchmark games."""

from .base import InterventionalGame
from .benchmark import (
    AdultCensus,
    Annealing,
    Arrhythmia,
    BikeSharing,
    BreastCancer,
    CaliforniaHousing,
    Hepatitis,
    Ionosphere,
    Mushroom,
    Nursery,
    Soybean,
    SynthData,
    Thyroid,
    Zoo,
)

__all__ = [
    "InterventionalGame",
    "AdultCensus",
    "Annealing",
    "Arrhythmia",
    "BikeSharing",
    "BreastCancer",
    "CaliforniaHousing",
    "Hepatitis",
    "Ionosphere",
    "Mushroom",
    "Nursery",
    "Soybean",
    "SynthData",
    "Thyroid",
    "Zoo",
]
