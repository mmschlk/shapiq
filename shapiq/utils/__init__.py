"""This module contains utility functions for the shapiq package."""

from .modules import safe_isinstance
from .sets import (
    generate_interaction_lookup,
    get_explicit_subsets,
    pair_subset_sizes,
    powerset,
    split_subsets_budget,
    transform_array_to_coalitions,
    transform_coalitions_to_array,
)

__all__ = [
    # sets
    "powerset",
    "pair_subset_sizes",
    "split_subsets_budget",
    "get_explicit_subsets",
    "generate_interaction_lookup",
    "transform_coalitions_to_array",
    "transform_array_to_coalitions",
    # modules
    "safe_isinstance",
]
