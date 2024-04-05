"""This module contains sample game functions for the shapiq package."""

from .base import Game
from .dummy import DummyGame
from .imputer import MarginalImputer
from .sentiment_language import SentimentClassificationGame
from .tabular import CaliforniaHousing, FeatureSelectionGame, LocalExplanation

__all__ = [
    "DummyGame",
    "Game",
    "MarginalImputer",
    "SentimentClassificationGame",
    "LocalExplanation",
    "FeatureSelectionGame",
    "CaliforniaHousing",
]
