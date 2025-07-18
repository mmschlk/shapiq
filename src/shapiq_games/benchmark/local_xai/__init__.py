"""This module contains all local explanation benchmark games."""

from .base import LocalExplanation
from .benchmark_image import ImageClassifier
from .benchmark_language import SentimentAnalysis
from .benchmark_tabular import AdultCensus, BikeSharing, CaliforniaHousing

__all__ = [
    "LocalExplanation",
    "AdultCensus",
    "BikeSharing",
    "CaliforniaHousing",
    "SentimentAnalysis",
    "ImageClassifier",
]
