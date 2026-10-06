"""Image classification games (vision transformers, ResNet, custom classifiers)."""

from ._superpixels import get_superpixels
from .image_classifier import ImageClassifier

__all__ = ["ImageClassifier", "get_superpixels"]
