"""Shapley interaction explanations for machine learning models."""

from shapiq.coalitions import CoalitionArray, DenseCoalitionArray
from shapiq.errors import (
    HistoryError,
    InsufficientSamplesError,
    SamplingError,
    UnsupportedGameError,
)
from shapiq.explainers import (
    Approximator,
    ExactExplainer,
    Explainer,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
)
from shapiq.explanations import DenseExplanationArray, ExplanationArray, SparseExplanationArray
from shapiq.games import (
    CallableGame,
    Game,
    LinkFunction,
    MaskedGame,
    MaskedPredictor,
    Masker,
    Model,
    ModelMaskedPredictor,
)
from shapiq.interactions import (
    Interaction,
    InteractionIndexName,
    InteractionOrientation,
    iter_interactions,
    normalize_interaction,
    validate_interaction_metadata,
)
from shapiq.sampling import (
    ApproximationState,
    PermutationSIISampler,
    PermutationSTIISampler,
    PermutationWalkSampler,
    Sampler,
    SamplingState,
)

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version.py is generated at build time
    __version__ = "0.0.0"

__all__ = [
    "ApproximationState",
    "Approximator",
    "CallableGame",
    "CoalitionArray",
    "DenseCoalitionArray",
    "DenseExplanationArray",
    "ExactExplainer",
    "Explainer",
    "ExplanationArray",
    "Game",
    "HistoryError",
    "InsufficientSamplesError",
    "Interaction",
    "InteractionIndexName",
    "InteractionOrientation",
    "LinkFunction",
    "MaskedGame",
    "MaskedPredictor",
    "Masker",
    "Model",
    "ModelMaskedPredictor",
    "PermutationSIISampler",
    "PermutationSTIISampler",
    "PermutationSamplingSII",
    "PermutationSamplingSTII",
    "PermutationSamplingSV",
    "PermutationWalkSampler",
    "Sampler",
    "SamplingError",
    "SamplingState",
    "SparseExplanationArray",
    "UnsupportedGameError",
    "__version__",
    "iter_interactions",
    "normalize_interaction",
    "validate_interaction_metadata",
]
