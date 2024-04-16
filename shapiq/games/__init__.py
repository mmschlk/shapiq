"""This module contains sample game functions for the shapiq package."""

from .base import Game
from .benchmark import (
    AdultCensus,
    BikeRegression,
    CaliforniaHousing,
    ImageClassifierGame,
    SentimentClassificationGame,
)
from .dummy import DummyGame
from .feature_selection import FeatureSelectionGame
from .imputer import MarginalImputer
from .local_xai import LocalExplanation

__all__ = [
    "DummyGame",
    "Game",
    "MarginalImputer",
    "SentimentClassificationGame",
    "LocalExplanation",
    "CaliforniaHousing",
    "BikeRegression",
    "AdultCensus",
    "ImageClassifierGame",
    "FeatureSelectionGame",
]
