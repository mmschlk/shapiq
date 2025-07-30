"""Utility functions for the shapiq package."""

from .datasets import shuffle_data
from .errors import raise_deprecation_warning
from .modules import check_import_module, safe_isinstance
from .sets import (
    count_interactions,
    generate_interaction_lookup,
    generate_interaction_lookup_from_coalitions,
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
    "generate_interaction_lookup_from_coalitions",
    "transform_coalitions_to_array",
    "transform_array_to_coalitions",
    "count_interactions",
    # modules
    "safe_isinstance",
    "check_import_module",
    # datasets
    "shuffle_data",
    # errors
    "raise_deprecation_warning",
]
