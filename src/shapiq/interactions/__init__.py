"""Interaction metadata and helpers."""

from __future__ import annotations

from shapiq.interactions._iteration import iter_interactions
from shapiq.interactions._types import Interaction, InteractionIndexName, InteractionOrientation
from shapiq.interactions._validation import normalize_interaction, validate_interaction_metadata

__all__ = [
    "Interaction",
    "InteractionIndexName",
    "InteractionOrientation",
    "iter_interactions",
    "normalize_interaction",
    "validate_interaction_metadata",
]
