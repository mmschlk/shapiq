"""Utility functions for the shapiq package."""

from .datasets import shuffle_data
from .modules import check_import_module, safe_isinstance
from .sets import (
    count_interactions,
    generate_interaction_lookup,
    get_explicit_subsets,
    pair_subset_sizes,
    powerset,
    split_subsets_budget,
    transform_array_to_coalitions,
    transform_coalitions_to_array,
)
from .types import Model

__all__ = [
    # types
    "Model",
    # sets
    "powerset",
    "pair_subset_sizes",
    "split_subsets_budget",
    "get_explicit_subsets",
    "generate_interaction_lookup",
    "transform_coalitions_to_array",
    "transform_array_to_coalitions",
    "count_interactions",
    # modules
    "safe_isinstance",
    "check_import_module",
    # datasets
    "shuffle_data",
]
