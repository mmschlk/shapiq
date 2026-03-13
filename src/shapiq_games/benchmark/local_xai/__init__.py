"""This module contains all local explanation benchmark games."""

from .base import LocalExplanation
from .benchmark_image import ImageClassifier
from .benchmark_language import SentimentAnalysis
from .benchmark_tabular import (
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
    Thyroid,
    Zoo,
)

__all__ = [
    "LocalExplanation",
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
    "Thyroid",
    "Zoo",
    "SentimentAnalysis",
    "ImageClassifier",
]
