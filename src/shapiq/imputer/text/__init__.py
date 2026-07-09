"""Imputation strategies for handling missing feature coalitions.

All imputers inherit from :class:`~shapiq.imputer.Imputer` and convert a model

prediction function into a cooperative game by imputing unobserved feature values.

"""

from .imputer import TextImputer

__all__ = ["TextImputer"]