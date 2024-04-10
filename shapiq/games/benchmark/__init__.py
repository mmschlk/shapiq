"""This module contains all benchmark games."""

from .image_classifier import ImageClassifierGame
from .sentiment_language import SentimentClassificationGame
from .tabular import AdultCensus, BikeRegression, CaliforniaHousing

__all__ = [
    "ImageClassifierGame",
    "SentimentClassificationGame",
    "AdultCensus",
    "BikeRegression",
    "CaliforniaHousing",
]
