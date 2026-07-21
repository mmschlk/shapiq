"""Imputation strategies for handling missing feature coalitions in tabular explanations.

All imputers inherit from :class:`~shapiq.imputer.Imputer` and convert a model prediction
function into a cooperative game by imputing unobserved feature values.
"""

from .base import Imputer
from .baseline_imputer import BaselineImputer
from .gaussian_copula_imputer import GaussianCopulaImputer
from .gaussian_imputer import GaussianImputer
from .generative_conditional_imputer import GenerativeConditionalImputer
from .marginal_imputer import MarginalImputer
from .tabpfn_imputer import TabPFNImputer

__all__ = [
    "Imputer",
    "MarginalImputer",
    "GenerativeConditionalImputer",
    "BaselineImputer",
    "TabPFNImputer",
    "GaussianImputer",
    "GaussianCopulaImputer",
    "TextImputer",
]


def __getattr__(name: str) -> type:
    if name == "TextImputer":
        from .text import TextImputer

        return TextImputer

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
