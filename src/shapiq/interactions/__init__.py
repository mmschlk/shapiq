"""Interaction metadata and helpers."""

from __future__ import annotations

from shapiq.interactions._indices import (
    BII,
    BV,
    FSII,
    SII,
    STII,
    SV,
    InteractionIndex,
    RegressionIndex,
    WeightedDerivativeIndex,
)
from shapiq.interactions._iteration import iter_interactions
from shapiq.interactions._types import (
    Interaction,
    InteractionIndexName,
    InteractionOrientation,
    OrderSemantics,
)
from shapiq.interactions._validation import normalize_interaction, validate_interaction_metadata

__all__ = [
    "BII",
    "BV",
    "FSII",
    "SII",
    "STII",
    "SV",
    "Interaction",
    "InteractionIndex",
    "InteractionIndexName",
    "InteractionOrientation",
    "OrderSemantics",
    "RegressionIndex",
    "WeightedDerivativeIndex",
    "iter_interactions",
    "normalize_interaction",
    "validate_interaction_metadata",
]
