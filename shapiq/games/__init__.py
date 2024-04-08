"""This module contains sample game functions for the shapiq package."""

from .base import Game
from .benchmark_tabular import AdultCensus, BikeRegression, CaliforniaHousing
from .dummy import DummyGame
from .image_classifier import ImageClassifierGame
from .imputer import MarginalImputer
from .sentiment_language import SentimentClassificationGame
from .tabular import FeatureSelectionGame, LocalExplanation

__all__ = [
    "DummyGame",
    "Game",
    "MarginalImputer",
    "SentimentClassificationGame",
    "LocalExplanation",
    "FeatureSelectionGame",
    "CaliforniaHousing",
    "BikeRegression",
    "AdultCensus",
    "ImageClassifierGame",
]
