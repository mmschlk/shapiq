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
from .imputer import MarginalImputer
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
